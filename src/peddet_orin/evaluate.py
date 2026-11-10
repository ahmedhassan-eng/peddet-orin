"""Person AP on COCO val2017, computed by pycocotools for one precision."""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from peddet_orin import debug, runlog
from peddet_orin.backends import PytorchBackend, TensorRtBackend
from peddet_orin.engine_build import ARTIFACTS
from peddet_orin.model import load_detector
from peddet_orin.postprocess import PERSON_LABEL, Postprocessor
from peddet_orin.preprocess import load_rgb, preprocess

DATA_DIR = runlog.REPO_ROOT / "data"
ANNOTATIONS = DATA_DIR / "annotations" / "instances_val2017.json"
IMAGES = DATA_DIR / "val2017"
PERSON_CATEGORY_ID = 1
AP_INDEX = 0  # COCOeval.stats[0] is AP@[.5:.95].
AP50_INDEX = 1  # COCOeval.stats[1] is AP at IoU 0.50.
PROGRESS_EVERY = 500
EVAL_SEED = 0  # Evaluation is deterministic; the field is logged for uniformity.
PRECISIONS = ("fp32", "fp16", "int8")
CUDA = "cuda"


def make_backend(precision: str, device: str, detector):
    """Build the backend for a precision. FP32 is PyTorch, the rest TensorRT."""
    if precision == "fp32":
        return PytorchBackend(detector, device)
    return TensorRtBackend(ARTIFACTS / f"{precision}.engine")


def person_ap(ground_truth: COCO, detections: list[dict], image_ids: list[int]) -> dict[str, float]:
    """Score person detections, returning AP@[.5:.95] and AP50 as fractions."""
    with contextlib.redirect_stdout(io.StringIO()):
        results = ground_truth.loadRes(detections)
        evaluator = COCOeval(ground_truth, results, "bbox")
        evaluator.params.catIds = [PERSON_CATEGORY_ID]
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {"ap": float(evaluator.stats[AP_INDEX]), "ap50": float(evaluator.stats[AP50_INDEX])}


def detect_people(backend, postprocess: Postprocessor, ground_truth: COCO, image_ids: list[int]) -> list[dict]:
    """Run the full pipeline over ``image_ids`` and return COCO-format person results."""
    results: list[dict] = []
    for index, image_id in enumerate(image_ids, start=1):
        image = load_rgb(IMAGES / ground_truth.loadImgs([image_id])[0]["file_name"])
        box_regression, class_logits = backend(preprocess(image))
        found = postprocess(box_regression, class_logits, image.shape[-2:])
        people = found.labels == PERSON_LABEL
        for (x1, y1, x2, y2), score in zip(found.boxes[people].tolist(), found.scores[people].tolist()):
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=PRECISIONS, required=True)
    parser.add_argument("--device", default=CUDA, help="torch device for fp32; TensorRT always uses the GPU")
    args = parser.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(ANNOTATIONS))
    image_ids = sorted(ground_truth.getImgIds())
    with runlog.RunLog("eval", args.precision, EVAL_SEED, {"images": len(image_ids)}) as log:
        detector = load_detector(pretrained=True)
        postprocess = Postprocessor.from_torchvision(detector)
        backend = make_backend(args.precision, args.device, detector)
        detections = detect_people(backend, postprocess, ground_truth, image_ids)
        scores = person_ap(ground_truth, detections, image_ids)
        log.results.update({**scores, "runtime": backend.label, "detections": len(detections)})
    print(f"{args.precision}: person AP {scores['ap']:.4f}, AP50 {scores['ap50']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
