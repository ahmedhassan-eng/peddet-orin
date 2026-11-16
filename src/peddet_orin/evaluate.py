# SPDX-License-Identifier: Apache-2.0
"""Person AP on COCO val2017, computed by pycocotools for one precision."""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

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
EVAL_SEED = 0  # Evaluation is deterministic; the field is logged for uniformity.


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
        detector = load_detector(pretrained=True)
        postprocess = Postprocessor.from_torchvision(detector)
        backend = load_backend(parser.prog, args.precision, args.device, detector, engine)
        try:
            detections = detect_people(backend, postprocess, ground_truth, image_ids, images)
        finally:
            release_backend(backend)
        scores = person_ap(ground_truth, detections, image_ids)
        log.results.update({**scores, "runtime": backend.label, "detections": len(detections)})
    print(f"{args.precision}: person AP {scores['ap']:.4f}, AP50 {scores['ap50']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
