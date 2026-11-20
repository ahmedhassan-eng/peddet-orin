# SPDX-License-Identifier: Apache-2.0
"""Measure, on the board, the quantities the bounds in tests/test_gpu.py are set from.

Prints one line per quantity, "BOUND <name> <value>":

* cuda_fp32_relative_error: PyTorch on CUDA in FP32, TF32 off, against the CPU, on the tests' seeded input;
* fp16_vs_fp32_trt_cosine and int8_vs_fp32_trt_cosine: the TensorRT FP16 and INT8 engines' lowest output cosine
  similarity to the TensorRT FP32 engine's, built from the tests' seeded network and calibration set;
* int8_unnormalised_vs_fp32_trt_cosine: the same for an INT8 engine calibrated on unnormalised input, the
  int8-calibration-unnormalised plant, so the INT8 floor can be set between the two.

It uses the tests' own helpers and inputs, and needs CUDA and TensorRT: without CUDA it prints one line and exits 2.
It writes its engines to a temporary directory and no run log.
Usage: python scripts/gpu_bounds.py
"""

import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))
import test_gpu  # noqa: E402

EXIT_NO_CUDA = 2


def bound(name: str, value: float) -> None:
    print(f"BOUND {name} {value:.6g}", flush=True)


def main() -> int:
    if not torch.cuda.is_available():
        print("gpu_bounds: error: no CUDA device; run this on the board", file=sys.stderr)
        return EXIT_NO_CUDA
    from peddet_orin import engine_build
    from peddet_orin.backends import TensorRtBackend
    from peddet_orin.export import export
    from peddet_orin.model import SsdNetwork

    bound("cuda_fp32_relative_error", test_gpu.cuda_forward_error())

    with tempfile.TemporaryDirectory(prefix="gpu-bounds-") as name:
        work = Path(name)
        onnx_path = work / "network.onnx"
        export(SsdNetwork(test_gpu.lively_detector(test_gpu.SEED)).eval(), onnx_path)
        calibration, manifest = test_gpu.write_calibration_set(work)
        images = engine_build.verify_calibration(calibration, manifest, test_gpu.INT8_CALIBRATION_IMAGES)
        engines = {precision: work / f"{precision}.engine" for precision in engine_build.PRECISIONS}
        engine_build.build_fp32_trt(onnx_path, engines[engine_build.FP32_TRT])
        engine_build.build_fp16(onnx_path, engines["fp16"])
        engine_build.build_int8(onnx_path, engines["int8"], images)
        for precision in ("fp16", "int8"):
            with TensorRtBackend(engines[precision]) as engine:
                cosine = test_gpu.cosine_to_fp32_trt(engines[engine_build.FP32_TRT], engine)
            bound(f"{precision}_vs_fp32_trt_cosine", cosine)
        # The INT8 floor must sit between the clean engine and one calibrated on unnormalised input, the
        # int8-calibration-unnormalised plant, so that engine's cosine is measured too, with the same plant.
        clean_batch = engine_build._calibration_batch
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(
                engine_build, "_calibration_batch", lambda path: (clean_batch(path) * 0.5 + 0.5) * 255.0
            )
            # Its own directory, since the calibration cache is written next to the engine: the clean build's
            # cache stays as it was.
            (work / "unnormalised").mkdir()
            planted = work / "unnormalised" / "int8.engine"
            engine_build.build_int8(onnx_path, planted, images)
        with TensorRtBackend(planted) as engine:
            bound(
                "int8_unnormalised_vs_fp32_trt_cosine",
                test_gpu.cosine_to_fp32_trt(engines[engine_build.FP32_TRT], engine),
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
