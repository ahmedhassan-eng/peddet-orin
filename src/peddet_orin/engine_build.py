"""Build TensorRT FP16 and INT8 engines from the ONNX network, on the device.

INT8 uses IInt8EntropyCalibrator2 over the seeded calibration set. Should that
fail, it falls back to ONNX post-training quantisation, built from the resulting
Q/DQ graph. There is no path that builds INT8 without calibration data. The route
taken is recorded in the run log and appears in the published table.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import onnxruntime.quantization as ort_quantization

from peddet_orin import cuda_native, debug, runlog
from peddet_orin.export import DEFAULT_ONNX, INPUT_NAME
from peddet_orin.preprocess import load_rgb, preprocess

ARTIFACTS = runlog.REPO_ROOT / "artifacts"
CALIBRATION_DIR = runlog.REPO_ROOT / "data" / "calibration_train2017"
CALIBRATION_CACHE = ARTIFACTS / "int8_calibration.cache"
QDQ_ONNX = ARTIFACTS / "ssdlite320_qdq.onnx"
CALIBRATION_BATCH = 1
CALIBRATION_SEED = 2017  # The seed used to select the calibration images.
EXPECTED_CALIBRATION_IMAGES = 500
# TensorRT 10 deprecates this calibrator; only warnings naming it are filtered.
CALIBRATOR_DEPRECATION_PATTERN = r".*IInt8EntropyCalibrator2.*deprecated.*"
PATH_ENTROPY = "IInt8EntropyCalibrator2"
PATH_QDQ = "onnx-ptq-qdq"
PRECISIONS = ("fp16", "int8")


def calibration_images() -> list[Path]:
    """List the fetched calibration images, refusing an incomplete set."""
    images = sorted(CALIBRATION_DIR.glob("*.jpg"))
    if len(images) != EXPECTED_CALIBRATION_IMAGES:
        raise RuntimeError(f"expected {EXPECTED_CALIBRATION_IMAGES} calibration images, found {len(images)}")
    return images


def _calibration_batch(path: Path) -> np.ndarray:
    return preprocess(load_rgb(path)).numpy()


def _make_calibrator(images: list[Path]):
    """Create the entropy calibrator, which feeds TensorRT through a device buffer."""
    import tensorrt

    class EntropyCalibrator(tensorrt.IInt8EntropyCalibrator2):
        def __init__(self) -> None:
            tensorrt.IInt8EntropyCalibrator2.__init__(self)
            self.remaining = iter(images)
            self.buffer: cuda_native.DeviceBuffer | None = None

        def get_batch(self, names):
            path = next(self.remaining, None)
            if path is None:
                return None
            batch = _calibration_batch(path)
            if self.buffer is None:
                self.buffer = cuda_native.DeviceBuffer(batch.shape, np.float32)
            self.buffer.write(batch)
            return [self.buffer.address]

        def get_batch_size(self) -> int:
            return CALIBRATION_BATCH

        def get_algorithm(self):
            return tensorrt.CalibrationAlgoType.ENTROPY_CALIBRATION_2

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache) -> None:
            CALIBRATION_CACHE.write_bytes(bytes(cache))

    return EntropyCalibrator()


def _build(onnx_path: Path, output: Path, configure) -> None:
    """Parse the ONNX file, apply ``configure`` to the builder config, and serialise."""
    import tensorrt

    logger = tensorrt.Logger()
    builder = tensorrt.Builder(logger)
    network = builder.create_network()
    parser = tensorrt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "; ".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed: {errors}")
    config = builder.create_builder_config()
    configure(config, tensorrt)
    with debug.profile("engine build"):
        serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT returned no engine")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(serialized))


def build_fp16(onnx_path: Path, output: Path) -> None:
    def configure(config, tensorrt) -> None:
        config.set_flag(tensorrt.BuilderFlag.FP16)

    _build(onnx_path, output, configure)


def _build_int8_entropy(onnx_path: Path, output: Path, images: list[Path]) -> None:
    def configure(config, tensorrt) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=CALIBRATOR_DEPRECATION_PATTERN)
            config.int8_calibrator = _make_calibrator(images)
        config.set_flag(tensorrt.BuilderFlag.INT8)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=CALIBRATOR_DEPRECATION_PATTERN)
        _build(onnx_path, output, configure)


class _QdqReader(ort_quantization.CalibrationDataReader):
    """Feed the calibration set to ONNX Runtime's static quantiser."""

    def __init__(self, images: list[Path]) -> None:
        self.remaining = iter(images)

    def get_next(self):
        path = next(self.remaining, None)
        return None if path is None else {INPUT_NAME: _calibration_batch(path)}


def _build_int8_qdq(onnx_path: Path, output: Path, images: list[Path]) -> None:
    """Fallback: calibrate in ONNX Runtime, then let TensorRT honour the Q/DQ nodes."""
    ort_quantization.quantize_static(
        str(onnx_path),
        str(QDQ_ONNX),
        _QdqReader(images),
        quant_format=ort_quantization.QuantFormat.QDQ,
        activation_type=ort_quantization.QuantType.QInt8,
        weight_type=ort_quantization.QuantType.QInt8,
        per_channel=True,
        calibrate_method=ort_quantization.CalibrationMethod.Entropy,
    )

    def configure(config, tensorrt) -> None:
        config.set_flag(tensorrt.BuilderFlag.INT8)

    _build(QDQ_ONNX, output, configure)


def build_int8(onnx_path: Path, output: Path) -> tuple[str, str]:
    """Build INT8, returning the path taken and the entropy failure text, if any."""
    images = calibration_images()
    try:
        _build_int8_entropy(onnx_path, output, images)
        return PATH_ENTROPY, ""
    except Exception as error:  # TensorRT surfaces build failures as many exception types.
        failure = f"{type(error).__name__}: {runlog.relative_command([str(error)])}"
    _build_int8_qdq(onnx_path, output, images)
    return PATH_QDQ, failure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--precision", choices=PRECISIONS, required=True)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    args = parser.parse_args()
    output = ARTIFACTS / f"{args.precision}.engine"
    calibration_count = EXPECTED_CALIBRATION_IMAGES if args.precision == "int8" else 0
    with runlog.RunLog("build", args.precision, CALIBRATION_SEED, {"calibration_images": calibration_count}) as log:
        started = time.perf_counter()
        if args.precision == "fp16":
            build_fp16(args.onnx, output)
        else:
            route, failure = build_int8(args.onnx, output)
            log.results.update({"int8_path": route, "entropy_failure": failure})
        log.results.update(
            {"build_seconds": round(time.perf_counter() - started, 1), "engine_bytes": output.stat().st_size}
        )
    print(f"built {output.relative_to(runlog.REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
