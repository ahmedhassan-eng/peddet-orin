"""Decide by running whether the installed PyTorch can really use this GPU.

The PyTorch wheels for this board are built for datacenter compute capabilities and
carry no slice for the board's own architecture. The driver instead JIT-compiles
the included PTX on the first kernel launch. Whether that works is a question for
an experiment, not for ``torch.cuda.get_arch_list()``, so this module runs two small
operations on the GPU and compares them with the same operations on the CPU:

* a floating-point matrix multiply, compared element by element within a tolerance;
* torchvision's ``nms`` on synthetic boxes, whose kept indices must be identical.

Both results go into a run log, and a GPU kernel that cannot run counts as a
failed comparison. The exit status is non-zero if the check fails.
"""
from __future__ import annotations

import re
import sys
import warnings

import torch
from torchvision.ops import nms

from peddet_orin import debug, runlog
from peddet_orin.model import SM87_WARNING_PATTERN

CUDA = "cuda"
CPU = "cpu"
CHECK_SEED = 87
MATRIX_SIZE = 256
# Absolute tolerance on the product of matrices with entries in [0, 1). The largest
# element is about MATRIX_SIZE / 4, so this is a few parts per million of it.
MATMUL_TOLERANCE = 1e-3
BOX_COUNT = 2000
CANVAS_PIXELS = 320.0
MIN_BOX_PIXELS = 8.0
MAX_BOX_PIXELS = 64.0
NMS_IOU = 0.5
NOT_RUN = "not run"  # Stands in for a measurement whose operation raised an error.
SLICE_PATTERN = re.compile(r"sm_(\d)(\d+)$")
NO_COMPATIBLE_SLICE = "none"


def matmul_difference(generator: torch.Generator) -> float:
    """Return the largest absolute element difference between GPU and CPU matmul."""
    left = torch.rand(MATRIX_SIZE, MATRIX_SIZE, generator=generator)
    right = torch.rand(MATRIX_SIZE, MATRIX_SIZE, generator=generator)
    on_cpu = left @ right
    on_gpu = (left.to(CUDA) @ right.to(CUDA)).cpu()
    return float((on_cpu - on_gpu).abs().max())


def synthetic_boxes(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Make random xyxy boxes and scores, dense enough that NMS removes many."""
    corners = torch.rand(BOX_COUNT, 2, generator=generator) * CANVAS_PIXELS
    span = MAX_BOX_PIXELS - MIN_BOX_PIXELS
    sizes = MIN_BOX_PIXELS + torch.rand(BOX_COUNT, 2, generator=generator) * span
    scores = torch.rand(BOX_COUNT, generator=generator)
    return torch.cat([corners, corners + sizes], dim=1), scores


def nms_comparison(generator: torch.Generator) -> dict[str, int | bool]:
    """Compare the indices kept by NMS on the GPU with those kept on the CPU."""
    boxes, scores = synthetic_boxes(generator)
    on_cpu = nms(boxes, scores, NMS_IOU)
    on_gpu = nms(boxes.to(CUDA), scores.to(CUDA), NMS_IOU).cpu()
    return {
        "nms_identical": torch.equal(on_cpu, on_gpu),
        "nms_kept_cpu": int(on_cpu.numel()),
        "nms_kept_gpu": int(on_gpu.numel()),
        "nms_input_boxes": BOX_COUNT,
    }


def compatible_slice(arch_list: list[str], major: int, minor: int) -> str:
    """Return the newest slice a device of this capability can load, by CUDA's rule.

    A CUDA binary compiled for capability X.y runs on X.z when z >= y, and never
    across major versions. The driver does not report which slice it loaded, so this
    is an inference from that rule and the wheel's own list, and is logged as one.
    """
    candidates = []
    for arch in arch_list:
        match = SLICE_PATTERN.match(arch)
        if match and int(match.group(1)) == major and int(match.group(2)) <= minor:
            candidates.append((int(match.group(2)), arch))
    return max(candidates)[1] if candidates else NO_COMPATIBLE_SLICE


def attempt(operation, generator: torch.Generator) -> tuple[object, str]:
    """Run one comparison, returning (result, error text); only one of them is set.

    A GPU kernel that cannot run is an answer to the question being asked, so it is
    recorded in the log instead of ending the program. The error text is reduced to
    its first line, since the rest is generic debugging advice.
    """
    try:
        return operation(generator), ""
    except RuntimeError as error:  # torch.AcceleratorError derives from RuntimeError.
        return None, str(error).splitlines()[0]


def main() -> int:
    counts = {"matrix_size": MATRIX_SIZE, "nms_boxes": BOX_COUNT}
    with runlog.RunLog("gpu_check", "fp32", CHECK_SEED, counts) as log:
        generator = torch.Generator().manual_seed(CHECK_SEED)
        difference, comparison, matmul_error, nms_error = None, None, "", ""
        with warnings.catch_warnings():
            # The first CUDA use warns about this board's compute capability. That
            # single message is expected here and every other warning stays visible.
            warnings.filterwarnings("ignore", message=SM87_WARNING_PATTERN)
            available = torch.cuda.is_available()
            if available:
                with debug.profile("gpu check"):
                    difference, matmul_error = attempt(matmul_difference, generator)
                    comparison, nms_error = attempt(nms_comparison, generator)
        matmul_ok = difference is not None and difference <= MATMUL_TOLERANCE
        nms_ok = comparison is not None and bool(comparison["nms_identical"])
        usable = matmul_ok and nms_ok
        arch_list = list(torch.cuda.get_arch_list()) if available else []
        major, minor = torch.cuda.get_device_capability(0) if available else (0, 0)
        log.results.update(
            {
                "cuda_available": available,
                "arch_list": arch_list,
                "device_compute_capability": f"{major}.{minor}",
                "inferred_slice": compatible_slice(arch_list, major, minor),
                "matmul_max_abs_difference": difference if difference is not None else NOT_RUN,
                "matmul_tolerance": MATMUL_TOLERANCE,
                "matmul_matches": matmul_ok,
                "matmul_error": matmul_error,
                "nms_comparison": comparison if comparison is not None else NOT_RUN,
                "nms_matches": nms_ok,
                "nms_error": nms_error,
                "gpu_torch_usable": usable,
            }
        )
    print(f"gpu torch usable: {usable}")
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())
