# SPDX-License-Identifier: Apache-2.0
"""Person AP on COCO val2017, computed by pycocotools for one precision."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from peddet_orin import debug, runlog
from peddet_orin.cli import (
    ANNOTATIONS_RELATIVE,
    IMAGES,
    IMAGES_RELATIVE,
    HelpFormatter,
    add_common_options,
    check_device,
    check_engine,
    engine_sha256,
    load_backend,
    release_backend,
    usage_error,
)
from peddet_orin.model import load_detector
from peddet_orin.postprocess import PERSON_LABEL, Postprocessor
from peddet_orin.preprocess import load_rgb, preprocess

PERSON_CATEGORY_ID = 1
AP_INDEX = 0  # COCOeval.stats[0] is AP@[.5:.95].
AP50_INDEX = 1  # COCOeval.stats[1] is AP at IoU 0.50.
PROGRESS_EVERY = 500
# The eval of one engine is deterministic; INT8 engines from different builds score differently, as the notes show.
# The field is logged for uniformity.
EVAL_SEED = 0


def person_ap(ground_truth: COCO, detections: list[dict], image_ids: list[int]) -> dict[str, float]:
    """Score person detections, returning AP@[.5:.95] and AP50 as fractions."""
    if not detections:
        # pycocotools cannot load an empty result list; finding nobody scores zero.
        return {"ap": 0.0, "ap50": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        results = ground_truth.loadRes(detections)
        evaluator = COCOeval(ground_truth, results, "bbox")
        evaluator.params.catIds = [PERSON_CATEGORY_ID]
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {"ap": float(evaluator.stats[AP_INDEX]), "ap50": float(evaluator.stats[AP50_INDEX])}


def detect_people(
    backend, postprocess: Postprocessor, ground_truth: COCO, image_ids: list[int], images_dir: Path = IMAGES
) -> list[dict]:
    """Run the full pipeline over ``image_ids`` and return COCO-format person results."""
    results: list[dict] = []
    for index, image_id in enumerate(image_ids, start=1):
        image = load_rgb(images_dir / ground_truth.loadImgs([image_id])[0]["file_name"])
        box_regression, class_logits = backend(preprocess(image))
        found = postprocess(box_regression, class_logits, image.shape[-2:])
        people = found.labels == PERSON_LABEL
        for (x1, y1, x2, y2), score in zip(found.boxes[people].tolist(), found.scores[people].tolist(), strict=True):
            results.append(
                {
                    "image_id": image_id,
                    "category_id": PERSON_CATEGORY_ID,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )
        if index % PROGRESS_EVERY == 0:
            debug.log(f"evaluated {index}/{len(image_ids)}")
            print(f"{index}/{len(image_ids)}", flush=True)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m peddet_orin.evaluate", description=__doc__, formatter_class=HelpFormatter
    )
    add_common_options(parser)
    return parser


def check_data(prog: str, data_dir: Path) -> tuple[Path, Path]:
    """Return the annotations file and image directory, exiting with the fetch command if either is missing."""
    annotations, images = data_dir / ANNOTATIONS_RELATIVE, data_dir / IMAGES_RELATIVE
    if not annotations.is_file():
        usage_error(prog, f"annotations file {annotations} is missing; run python scripts/fetch_coco.py annotations")
    if not images.is_dir():
        usage_error(prog, f"image directory {images} is missing; run python scripts/fetch_coco.py val")
    return annotations, images


# The raw detections of each eval, under the git-ignored data directory: their sha256 is logged, the file is not
# committed.
DETECTIONS_RELATIVE = Path("detections")
# Each eval's detections file is named for its precision and start time, so the runs of a window do not overwrite
# each other; the eval log names it, and the file stays under the git-ignored data directory.
DETECTIONS_SUFFIX = ".detections.json"


def input_hashes(annotations: Path, image_names: list[str], detections: list[dict], detections_path: Path) -> dict:
    """The sha256 of the annotations file, of the sorted image names and of the raw detections, which are written
    to ``detections_path`` in the canonical form whose bytes the hash covers."""
    raw = json.dumps(detections, sort_keys=True, separators=(",", ":")).encode()
    detections_path.parent.mkdir(parents=True, exist_ok=True)
    detections_path.write_bytes(raw)
    return {
        "annotations_sha256": runlog.file_sha256(annotations),
        "images_sha256": runlog.names_sha256(sorted(image_names)),
        "detections_sha256": hashlib.sha256(raw).hexdigest(),
    }


def score_line(precision: str, scores: dict) -> str:
    """The line evaluate prints: person AP and AP50 in percentage points, as the README's table gives them."""
    return f"{precision}: person AP {scores['ap'] * 100:.2f}, AP50 {scores['ap50'] * 100:.2f} (percentage points)"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    annotations, images = check_data(parser.prog, args.data_dir)
    engine = check_engine(parser.prog, args.precision, args.engine)
    check_device(parser.prog, args.precision, args.device)

    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotations))
    image_ids = sorted(ground_truth.getImgIds())
    if not image_ids:
        usage_error(parser.prog, f"annotations file {annotations} lists no images")
    with runlog.RunLog("eval", args.precision, EVAL_SEED, {"images": len(image_ids)}) as log:
        engine_digest = engine_sha256(engine)
        threads = torch.get_num_threads()
        detector = load_detector(pretrained=True)
        postprocess = Postprocessor.from_torchvision(detector)
        backend = load_backend(parser.prog, args.precision, args.device, detector, engine)
        try:
            detections = detect_people(backend, postprocess, ground_truth, image_ids, images)
        finally:
            release_backend(backend)
        scores = person_ap(ground_truth, detections, image_ids)
        names = [image["file_name"] for image in ground_truth.loadImgs(image_ids)]
        stamp = log.start_utc.replace("-", "").replace(":", "")
        detections_path = args.data_dir / DETECTIONS_RELATIVE / f"eval_{args.precision}_{stamp}{DETECTIONS_SUFFIX}"
        log.results.update(input_hashes(annotations, names, detections, detections_path))
        log.results["detections_file"] = detections_path.name
        log.results.update(
            {
                **scores,
                "runtime": backend.label,
                "detections": len(detections),
                "engine_sha256": engine_digest,
                "torch_num_threads": threads,
            }
        )
    print(score_line(args.precision, scores))
    return 0


if __name__ == "__main__":
    sys.exit(main())
