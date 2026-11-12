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
from peddet_orin.backends import PytorchBackend, TensorRtBackend
from peddet_orin.engine_build import ARTIFACTS
from peddet_orin.model import load_detector
from peddet_orin.postprocess import PERSON_LABEL, Postprocessor
from peddet_orin.preprocess import load_rgb, preprocess

DATA_DIR = runlog.REPO_ROOT / "data"
ANNOTATIONS_RELATIVE = Path("annotations") / "instances_val2017.json"
IMAGES_RELATIVE = Path("val2017")
ANNOTATIONS = DATA_DIR / ANNOTATIONS_RELATIVE
IMAGES = DATA_DIR / IMAGES_RELATIVE
PERSON_CATEGORY_ID = 1
AP_INDEX = 0  # COCOeval.stats[0] is AP@[.5:.95].
AP50_INDEX = 1  # COCOeval.stats[1] is AP at IoU 0.50.
PROGRESS_EVERY = 500
EVAL_SEED = 0  # Evaluation is deterministic; the field is logged for uniformity.
PRECISIONS = ("fp32", "fp16", "int8")
CUDA = "cuda"
USAGE_ERROR_STATUS = 2
# What a TensorRT backend may raise when the runtime cannot deserialise an engine file.
ENGINE_LOAD_ERRORS = (AttributeError, RuntimeError, TypeError, ValueError)


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show each option's default, except where there is none or it depends on another option."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default is None:
            return action.help or ""
        return super()._get_help_string(action)


def usage_error(prog: str, message: str) -> None:
    """Print a one-line error to stderr and exit with argparse's usage-error status."""
    print(f"{prog}: error: {message}", file=sys.stderr)
    raise SystemExit(USAGE_ERROR_STATUS)


def default_engine(precision: str) -> Path:
    """The engine path engine_build writes for a precision."""
    return ARTIFACTS / f"{precision}.engine"


def build_command(precision: str) -> str:
    return f"python -m peddet_orin.engine_build --precision {precision}"


def check_engine(prog: str, precision: str, engine: Path | None) -> Path | None:
    """Return the engine path a TensorRT precision needs, exiting if it is missing; None for fp32."""
    if precision == "fp32":
        return None
    path = engine if engine is not None else default_engine(precision)
    if not path.is_file():
        usage_error(prog, f"engine file {path} does not exist; build it with {build_command(precision)}")
    return path


def make_backend(precision: str, device: str, detector, engine_path: Path | None = None):
    """Build the backend for a precision. FP32 is PyTorch, the rest TensorRT."""
    if precision == "fp32":
        return PytorchBackend(detector, device)
    return TensorRtBackend(engine_path if engine_path is not None else default_engine(precision))


def release_backend(backend) -> None:
    """Free a backend's device buffers and stream now, rather than when it is collected."""
    close = getattr(backend, "close", None)
    if close is not None:
        close()


def load_backend(prog: str, precision: str, device: str, detector, engine_path: Path | None):
    """Build the backend, exiting with one line if TensorRT cannot load the engine."""
    if precision == "fp32":
        return make_backend(precision, device, detector)
    rebuild = f"rebuild it with {build_command(precision)}"
    try:
        backend = make_backend(precision, device, detector, engine_path)
    except ENGINE_LOAD_ERRORS as error:
        usage_error(prog, f"TensorRT could not load engine {engine_path} ({error}); {rebuild}")
    if backend is None or getattr(backend, "engine", True) is None:
        usage_error(prog, f"TensorRT could not load engine {engine_path}; {rebuild}")
    return backend


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


def add_common_options(parser: argparse.ArgumentParser) -> None:
    """The options evaluate and benchmark share."""
    parser.add_argument("--precision", choices=PRECISIONS, required=True, help="precision to measure (required)")
    parser.add_argument("--device", default=CUDA, help="torch device for fp32; TensorRT always uses the GPU")
    parser.add_argument(
        "--data-dir", type=Path, default=DATA_DIR, help="directory holding annotations/ and val2017/ from fetch_coco"
    )
    parser.add_argument(
        "--engine",
        type=Path,
        default=None,
        help="TensorRT engine for fp16 or int8 (default: artifacts/<precision>.engine in the repository)",
    )


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
