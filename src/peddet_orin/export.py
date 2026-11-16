# SPDX-License-Identifier: Apache-2.0
"""Export the network to ONNX and check it against PyTorch under ONNX Runtime on CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from peddet_orin import runlog
from peddet_orin.model import SsdNetwork, load_detector
from peddet_orin.preprocess import INPUT_SHAPE, normalise

OPSET = 17
INPUT_NAME = "input"
OUTPUT_NAMES = ("bbox_regression", "cls_logits")
# Largest allowed absolute element difference between ONNX Runtime and PyTorch.
MATCH_TOLERANCE = 1e-3
CHECK_SEED = 1234
CHECK_BATCHES = 4
DEFAULT_ONNX = runlog.REPO_ROOT / "artifacts" / "ssdlite320.onnx"


class OnnxMismatch(AssertionError):
    """ONNX Runtime and PyTorch disagree by more than the tolerance."""


def export(network: SsdNetwork, path: Path) -> None:
    """Write the network to ``path`` with a fixed 1x3x320x320 input."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        network.eval(),
        (torch.zeros(INPUT_SHAPE),),
        str(path),
        input_names=[INPUT_NAME],
        output_names=list(OUTPUT_NAMES),
        opset_version=OPSET,
        do_constant_folding=True,
        dynamo=False,
    )


def max_difference(
    network: SsdNetwork,
    onnx_path: Path,
    raw_batches: list[torch.Tensor],
    normalise_onnx_input: bool = True,
) -> float:
    """Return the largest element difference over batches of [0, 1] pixels.

    PyTorch always sees normalised input. ``normalise_onnx_input=False`` skips the
    step on the ONNX side only, which is how the check is shown to detect that bug.
    """
    # Imported here, not at the top: loading ONNX Runtime can print its own warnings, and a command's input
    # checks must stay one line on stderr (PRD R-14).
    import onnxruntime

    onnx_runtime = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for raw in raw_batches:
        with torch.no_grad():
            expected = network(normalise(raw))
        onnx_input = normalise(raw) if normalise_onnx_input else raw
        actual = onnx_runtime.run(list(OUTPUT_NAMES), {INPUT_NAME: onnx_input.numpy()})
        for want, got in zip(expected, actual, strict=True):
            worst = max(worst, float(np.max(np.abs(want.numpy() - got))))
    return worst


def assert_matches(
    network: SsdNetwork,
    onnx_path: Path,
    raw_batches: list[torch.Tensor],
    tolerance: float = MATCH_TOLERANCE,
    normalise_onnx_input: bool = True,
) -> float:
    """Raise ``OnnxMismatch`` if outputs differ by more than ``tolerance``."""
    worst = max_difference(network, onnx_path, raw_batches, normalise_onnx_input)
    if worst > tolerance:
        raise OnnxMismatch(f"max difference {worst:.6g} exceeds tolerance {tolerance:g}")
    return worst


def seeded_batches(seed: int, count: int) -> list[torch.Tensor]:
    """Make reproducible random batches of [0, 1] pixels."""
    generator = torch.Generator().manual_seed(seed)
    return [torch.rand(INPUT_SHAPE, generator=generator) for _ in range(count)]


def display_path(path: Path) -> str:
    """Show a path relative to the repository root when it lies under it, else as given."""
    try:
        return str(Path(path).resolve().relative_to(runlog.REPO_ROOT))
    except ValueError:
        return str(path)


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m peddet_orin.export",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ONNX,
        help="where to write the ONNX file; parent directories are created",
    )
    return parser.parse_args(argv)


def export_results(worst: float) -> dict[str, object]:
    """The results an export log records, as schema version 2 names them."""
    return {
        "max_abs_difference": worst,
        "tolerance": MATCH_TOLERANCE,
        "opset": OPSET,
        "input_shape": list(INPUT_SHAPE),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    with runlog.RunLog("export", "onnx", seed=CHECK_SEED, iterations={"check_batches": CHECK_BATCHES}) as log:
        network = SsdNetwork(load_detector(pretrained=True))
        export(network, args.output)
        worst = assert_matches(network, args.output, seeded_batches(CHECK_SEED, CHECK_BATCHES))
        log.results.update(export_results(worst))
    print(f"wrote {display_path(args.output)}; onnx matches pytorch: max abs difference {worst:.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
