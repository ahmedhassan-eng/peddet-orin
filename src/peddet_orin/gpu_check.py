# SPDX-License-Identifier: Apache-2.0
"""Decide by running whether the installed PyTorch can really use this GPU.

PyTorch itself declares this GPU unsupported. On the first CUDA use it warns that
the device's compute capability is outside what its build supports, that the
build's sm_80 slice excludes capability 8.7, and that no published build of this
PyTorch release for its CUDA version supports the device. Neither PyTorch nor the
driver reports which binary slice, or which PTX, actually runs, so this module
does not reason from ``torch.cuda.get_arch_list()``. It runs two small operations
on the GPU and compares them with the same operations on the CPU:

* a floating-point matrix multiply, compared element by element within a tolerance;
* torchvision's ``nms`` on synthetic boxes, whose kept indices must be identical.

PyTorch's warning is shown unchanged and recorded word for word. Both results go
into a run log, and a GPU kernel that cannot run counts as a failed comparison.
The exit status is non-zero if the check fails.
"""

from __future__ import annotations

import argparse
import sys
import warnings

import torch
from torchvision.ops import nms

from peddet_orin import debug, runlog
from peddet_orin.model import use_strict_fp32

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
# A slice names the capability as one integer, major * 10 + minor: sm_87 is 8.7, sm_100 is 10.0.


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


def first_cuda_use() -> str:
    """Initialise CUDA and return the text of the warnings PyTorch emits doing so.

    The warnings are recorded, not filtered, and shown again unchanged, so whoever
    runs the check still sees exactly what PyTorch said. "" means it said nothing.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.init()
    for warning in caught:
        warnings.showwarning(warning.message, warning.category, warning.filename, warning.lineno)
    return "\n".join(str(warning.message) for warning in caught)


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


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m peddet_orin.gpu_check",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    parse_arguments(argv)
    counts = {"matrix_size": MATRIX_SIZE, "nms_boxes": BOX_COUNT}
    with runlog.RunLog("gpu_check", "fp32", CHECK_SEED, counts) as log:
        generator = torch.Generator().manual_seed(CHECK_SEED)
        difference, comparison, matmul_error, nms_error, torch_warning = None, None, "", "", ""
        cudnn_allow_tf32, matmul_allow_tf32 = use_strict_fp32()
        available = torch.cuda.is_available()
        if available:
            torch_warning = first_cuda_use()
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
                "matmul_max_abs_difference": difference if difference is not None else NOT_RUN,
                "matmul_tolerance": MATMUL_TOLERANCE,
                "matmul_matches": matmul_ok,
                "matmul_error": matmul_error,
                "nms_comparison": comparison if comparison is not None else NOT_RUN,
                "nms_matches": nms_ok,
                "nms_error": nms_error,
                "gpu_torch_usable": usable,
                "torch_warning": torch_warning,
                "cudnn_allow_tf32": cudnn_allow_tf32,
                "matmul_allow_tf32": matmul_allow_tf32,
            }
        )
    print(f"gpu torch usable: {usable}")
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())
