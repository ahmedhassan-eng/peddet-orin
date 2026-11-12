# SPDX-License-Identifier: Apache-2.0
"""Build TensorRT FP16 and INT8 engines from the ONNX network, on the device.

INT8 uses IInt8EntropyCalibrator2 over the seeded calibration set. Should the
TensorRT build fail, it falls back to ONNX post-training quantisation, built from
the resulting Q/DQ graph. There is no path that builds INT8 without calibration
data, and the calibration images are checked against the committed sha256
manifest before any build starts. The route taken is recorded in the run log and
appears in the published table.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import onnxruntime.quantization as ort_quantization

from peddet_orin import cuda_native, debug, runlog
from peddet_orin.backends import tensorrt_logger
from peddet_orin.export import DEFAULT_ONNX, INPUT_NAME, display_path
from peddet_orin.preprocess import load_rgb, preprocess

ARTIFACTS = runlog.REPO_ROOT / "artifacts"
CALIBRATION_DIR = runlog.REPO_ROOT / "data" / "calibration_train2017"
MANIFEST_PATH = runlog.REPO_ROOT / "manifests" / "calibration_train2017.sha256"
# Written next to the INT8 engine, so --output keeps every build product together.
CALIBRATION_CACHE_NAME = "int8_calibration.cache"
QDQ_ONNX_NAME = "ssdlite320_qdq.onnx"
CALIBRATION_BATCH = 1
CALIBRATION_SEED = 2017  # The seed used to select the calibration images.
EXPECTED_CALIBRATION_IMAGES = 500
# TensorRT 10 deprecates this calibrator. The warning it emits reads
# "Use Deprecated in TensorRT 10.1. Superseded by explicit quantization. instead."
# and only that warning is filtered, only while the calibrator is set up and used.
CALIBRATOR_DEPRECATION_PATTERN = r"Use Deprecated in TensorRT [\d.]+\. Superseded by explicit quantization\. instead\."
PATH_ENTROPY = "IInt8EntropyCalibrator2"
PATH_QDQ = "onnx-ptq-qdq"
NOT_APPLICABLE = "-"
PRECISIONS = ("fp16", "int8")
EXIT_BAD_INPUT = 2
HASH_CHUNK_BYTES = 1 << 20


class InputError(ValueError):
    """An input the build needs is missing or wrong; the message is one line."""


class TensorRtBuildError(RuntimeError):
    """TensorRT parsed the network but returned no engine."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(manifest: Path) -> dict[str, str]:
    """Parse a ``sha256  name`` manifest into ``{name: sha256}``."""
    if not manifest.is_file():
        raise InputError(f"calibration manifest not found: {display_path(manifest)}")
    entries = {}
    for number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 2:
            raise InputError(f"calibration manifest line {number} is not 'sha256  name': {line!r}")
        digest, name = fields
        entries[name] = digest
    return entries


def verify_calibration(
    directory: Path,
    manifest: Path | None = None,
    expected_count: int | None = None,
) -> list[Path]:
    """Return the calibration images, sorted, after checking them against the manifest.

    The directory must hold exactly the files the manifest lists, each with the
    listed sha256, and the manifest must list ``expected_count`` files. Anything
    else raises ``InputError`` with a one-line reason.
    """
    manifest = MANIFEST_PATH if manifest is None else manifest
    expected_count = EXPECTED_CALIBRATION_IMAGES if expected_count is None else expected_count
    entries = read_manifest(manifest)
    if len(entries) != expected_count:
        raise InputError(f"calibration manifest lists {len(entries)} images, expected {expected_count}")
    if not directory.is_dir():
        raise InputError(f"calibration directory not found: {display_path(directory)}")
    present = {path.name for path in directory.iterdir() if path.is_file()}
    missing = sorted(set(entries) - present)
    if missing:
        raise InputError(
            f"{len(missing)} of {expected_count} calibration images missing from {display_path(directory)}, "
            f"first {missing[0]}"
        )
    extra = sorted(present - set(entries))
    if extra:
        raise InputError(
            f"{len(extra)} files in {display_path(directory)} are not in the calibration manifest, first {extra[0]}"
        )
    for name in sorted(entries):
        if file_sha256(directory / name) != entries[name]:
            raise InputError(f"calibration image {name} does not match its sha256 in {display_path(manifest)}")
    return [directory / name for name in sorted(entries)]


def require_onnx(onnx_path: Path) -> None:
    if not Path(onnx_path).is_file():
        raise InputError(f"ONNX file not found: {display_path(onnx_path)}; run python -m peddet_orin.export first")


def _calibration_batch(path: Path) -> np.ndarray:
    return preprocess(load_rgb(path)).numpy()


def _make_calibrator(images: list[Path], cache_path: Path):
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

        def release(self) -> None:
            """Free the device buffer; the build calls this once TensorRT is done."""
            if self.buffer is not None:
                self.buffer.free()
                self.buffer = None

        def get_batch_size(self) -> int:
            return CALIBRATION_BATCH

        def get_algorithm(self):
            return tensorrt.CalibrationAlgoType.ENTROPY_CALIBRATION_2

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache) -> None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(bytes(cache))

    return EntropyCalibrator()


def _build(onnx_path: Path, output: Path, configure) -> None:
    """Parse the ONNX file, apply ``configure`` to the builder config, and serialise."""
    import tensorrt

    logger = tensorrt_logger()
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
        raise TensorRtBuildError("TensorRT returned no engine")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(serialized))


def build_fp16(onnx_path: Path, output: Path) -> None:
    def configure(config, tensorrt) -> None:
        config.set_flag(tensorrt.BuilderFlag.FP16)

    _build(onnx_path, output, configure)


def _build_int8_entropy(onnx_path: Path, output: Path, images: list[Path]) -> None:
    calibrator = None

    def configure(config, tensorrt) -> None:
        nonlocal calibrator
        calibrator = _make_calibrator(images, output.parent / CALIBRATION_CACHE_NAME)
        config.int8_calibrator = calibrator
        config.set_flag(tensorrt.BuilderFlag.INT8)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=CALIBRATOR_DEPRECATION_PATTERN)
        try:
            _build(onnx_path, output, configure)
        finally:
            if calibrator is not None:
                calibrator.release()


class _QdqReader(ort_quantization.CalibrationDataReader):
    """Feed the calibration set to ONNX Runtime's static quantiser."""

    def __init__(self, images: list[Path]) -> None:
        self.remaining = iter(images)

    def get_next(self):
        path = next(self.remaining, None)
        return None if path is None else {INPUT_NAME: _calibration_batch(path)}


def _build_int8_qdq(onnx_path: Path, output: Path, images: list[Path]) -> None:
    """Fallback: calibrate in ONNX Runtime, then let TensorRT honour the Q/DQ nodes."""
    qdq_onnx = output.parent / QDQ_ONNX_NAME
    qdq_onnx.parent.mkdir(parents=True, exist_ok=True)
    ort_quantization.quantize_static(
        str(onnx_path),
        str(qdq_onnx),
        _QdqReader(images),
        quant_format=ort_quantization.QuantFormat.QDQ,
        activation_type=ort_quantization.QuantType.QInt8,
        weight_type=ort_quantization.QuantType.QInt8,
        per_channel=True,
        calibrate_method=ort_quantization.CalibrationMethod.Entropy,
        # TensorRT dequantizes activations and weights only, so an int32 bias DequantizeLinear fails to parse, and
        # it accepts only symmetric INT8, so every zero point must be 0.
        extra_options={"QuantizeBias": False, "ActivationSymmetric": True, "WeightSymmetric": True},
    )

    def configure(config, tensorrt) -> None:
        config.set_flag(tensorrt.BuilderFlag.INT8)

    _build(qdq_onnx, output, configure)


def build_int8(onnx_path: Path, output: Path, images: list[Path]) -> tuple[str, str]:
    """Build INT8, returning the path taken and the entropy failure text, if any.

    Only a TensorRT build that returns no engine sends the build to the Q/DQ
    fallback. A missing input or any other error propagates unchanged.
    """
    require_onnx(onnx_path)
    if not images:
        raise InputError("INT8 needs calibration images and none were given")
    try:
        _build_int8_entropy(onnx_path, output, images)
        return PATH_ENTROPY, ""
    except TensorRtBuildError as error:
        failure = f"{type(error).__name__}: {runlog.relative_command([str(error)])}"
    _build_int8_qdq(onnx_path, output, images)
    return PATH_QDQ, failure


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m peddet_orin.engine_build",
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--precision", choices=PRECISIONS, required=True, help="engine precision to build")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX, help="ONNX network written by peddet_orin.export")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="where to write the engine (default: artifacts/<precision>.engine); "
        "INT8 also writes its calibration cache or Q/DQ graph next to it",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=CALIBRATION_DIR,
        help="directory holding exactly the INT8 calibration images listed in "
        "manifests/calibration_train2017.sha256",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    output = args.output if args.output is not None else ARTIFACTS / f"{args.precision}.engine"
    int8 = args.precision == "int8"
    try:
        require_onnx(args.onnx)
        images = verify_calibration(args.calibration_dir) if int8 else []
    except InputError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_BAD_INPUT
    manifest_digest = file_sha256(MANIFEST_PATH) if int8 else NOT_APPLICABLE
    with runlog.RunLog("build", args.precision, CALIBRATION_SEED, {"calibration_images": len(images)}) as log:
        started = time.perf_counter()
        if int8:
            route, failure = build_int8(args.onnx, output, images)
        else:
            build_fp16(args.onnx, output)
            route, failure = NOT_APPLICABLE, NOT_APPLICABLE
        log.results.update(
            {
                "int8_path": route,
                "entropy_failure": failure,
                "calibration_manifest_sha256": manifest_digest,
                "build_seconds": round(time.perf_counter() - started, 1),
                "engine_bytes": output.stat().st_size,
            }
        )
    print(f"built {display_path(output)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
