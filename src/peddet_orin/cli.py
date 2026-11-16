# SPDX-License-Identifier: Apache-2.0
"""The command-line pieces evaluate and benchmark share: data and engine paths, common options, the one-line
usage error, and the backend for each precision."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from peddet_orin import runlog
from peddet_orin.backends import PytorchBackend, TensorRtBackend
from peddet_orin.engine_build import ARTIFACTS

DATA_DIR = runlog.REPO_ROOT / "data"
ANNOTATIONS_RELATIVE = Path("annotations") / "instances_val2017.json"
IMAGES_RELATIVE = Path("val2017")
ANNOTATIONS = DATA_DIR / ANNOTATIONS_RELATIVE
IMAGES = DATA_DIR / IMAGES_RELATIVE
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


def check_device(prog: str, precision: str, device: str) -> None:
    """Exit with one line if FP32's torch device is not a device name, or is CUDA on a host where PyTorch finds none.
    TensorRT precisions always use the GPU and ignore ``--device``."""
    if precision != "fp32":
        return
    try:
        kind = torch.device(device).type
    except (RuntimeError, ValueError):
        usage_error(prog, f"--device {device} is not a torch device; use cuda or cpu")
    if kind == CUDA and not torch.cuda.is_available():
        usage_error(prog, f"--device {device}: PyTorch finds no usable CUDA device here; use --device cpu")


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
