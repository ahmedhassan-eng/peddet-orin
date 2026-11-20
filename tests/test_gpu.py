# SPDX-License-Identifier: Apache-2.0
"""PyTorch on CUDA and the TensorRT FP32, FP16 and INT8 engines, against PyTorch on the CPU.

Most need the Jetson: they skip where there is no CUDA device or no TensorRT, as in CI.
They use an unweighted, seeded model and synthetic calibration images, so they need no download.
"""

import hashlib
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from helpers import lively_detector
from peddet_orin.export import OUTPUT_NAMES
from peddet_orin.model import SsdNetwork
from peddet_orin.preprocess import INPUT_SHAPE, normalise

SEED = 654
# Worst absolute difference relative to the largest reference output, for PyTorch on CUDA in FP32 (TF32 off) against
# the CPU. Set from one board run of scripts/gpu_bounds.py (its log's sha256 begins 0fb14a18dc74f700): "BOUND
# cuda_fp32_relative_error 1.3499e-05", so the bound is about 7.4 times the measured error. With TF32 on, that run
# measured 1.12932e-05, no larger, so this bound cannot tell TF32 from FP32 (a QUEUE row).
CUDA_FP32_RELATIVE_TOLERANCE = 1e-4
# FP16 is checked by the cosine similarity of each output, since its worst single element varies from build to
# build. By the author's record of runs on this board: correct engines gave at least 0.9991, un-normalised input
# at most 0.723, and a network with other weights about 0.
FP16_MIN_COSINE = 0.99

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
needs_tensorrt = pytest.mark.skipif(importlib.util.find_spec("tensorrt") is None, reason="TensorRT is not installed")


def seeded_batch() -> torch.Tensor:
    raw = torch.rand(INPUT_SHAPE, generator=torch.Generator().manual_seed(SEED))
    return normalise(raw)


def relative_error(reference: tuple[torch.Tensor, ...], candidate: tuple[torch.Tensor, ...]) -> float:
    worst = 0.0
    for want, got in zip(reference, candidate, strict=True):
        assert want.shape == got.shape
        got = got.cpu()
        # A NaN or infinity on either side fails the check: max() would drop a NaN ratio.
        if not (torch.isfinite(want).all() and torch.isfinite(got).all()):
            return math.inf
        worst = max(worst, float((want - got).abs().max() / want.abs().max()))
    return worst


def cpu_reference(network: SsdNetwork, batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
    with torch.no_grad():
        return tuple(output.clone() for output in network(batch))


def cuda_forward_error() -> float:
    """The relative error of the seeded network's forward pass in PyTorch on CUDA against the CPU."""
    from peddet_orin.backends import PytorchBackend

    detector = lively_detector(SEED)
    batch = seeded_batch()
    reference = cpu_reference(SsdNetwork(detector).eval(), batch)
    # The backend moves the detector to the GPU, so the CPU reference is taken first.
    return relative_error(reference, PytorchBackend(detector, "cuda")(batch))


@needs_cuda
def test_cuda_forward_matches_cpu():
    assert cuda_forward_error() < CUDA_FP32_RELATIVE_TOLERANCE


def min_cosine(reference: tuple[torch.Tensor, ...], candidate: tuple[torch.Tensor, ...]) -> float:
    return min(
        float(torch.nn.functional.cosine_similarity(want.flatten(), got.cpu().flatten(), dim=0))
        for want, got in zip(reference, candidate, strict=True)
    )


@pytest.fixture(scope="module")
def fp16_engine(tmp_path_factory):
    """Build one FP16 engine from the seeded network; the build takes minutes."""
    from peddet_orin.backends import TensorRtBackend
    from peddet_orin.engine_build import build_fp16
    from peddet_orin.export import export

    network = SsdNetwork(lively_detector(SEED)).eval()
    work = tmp_path_factory.mktemp("fp16")
    export(network, work / "network.onnx")
    build_fp16(work / "network.onnx", work / "fp16.engine")
    with TensorRtBackend(work / "fp16.engine") as engine:
        yield network, engine


@needs_cuda
@needs_tensorrt
def test_tensorrt_fp16_engine_matches_cpu(fp16_engine):
    network, engine = fp16_engine
    batch = seeded_batch()
    assert min_cosine(cpu_reference(network, batch), engine(batch)) >= FP16_MIN_COSINE


@needs_cuda
@needs_tensorrt
def test_tensorrt_check_fails_on_unnormalised_input(fp16_engine):
    network, engine = fp16_engine
    raw = torch.rand(INPUT_SHAPE, generator=torch.Generator().manual_seed(SEED))
    assert min_cosine(cpu_reference(network, normalise(raw)), engine(raw)) < FP16_MIN_COSINE


@needs_cuda
@needs_tensorrt
def test_tensorrt_backend_waits_for_its_stream_before_reading_the_outputs(fp16_engine, monkeypatch):
    # cudaMemcpy on the legacy default stream waits for the engine's blocking stream anyway, so a dropped synchronize
    # gives the same numbers here; the order of the calls is what shows it.
    _, engine = fp16_engine
    calls = []
    synchronize = engine.stream.synchronize
    monkeypatch.setattr(engine.stream, "synchronize", lambda: (calls.append("synchronize"), synchronize())[1])
    for name in OUTPUT_NAMES:
        read = engine.buffers[name].read
        monkeypatch.setattr(engine.buffers[name], "read", lambda read=read: (calls.append("read"), read())[1])
    engine(seeded_batch())
    assert calls == ["synchronize", *["read"] * len(OUTPUT_NAMES)]


# A first bound: the board's GPU runs have passed it, and its margin is not recorded. It is ten times the bound for
# PyTorch on CUDA. It is not shown to separate FP32 from TF32 or FP16; the logged builder flags are what show that no
# reduced-precision flag was set. No check of the engine's per-layer precision is made: TensorRT's IEngineInspector
# reports it only for an engine built with ProfilingVerbosity.DETAILED, which the measured engines are not.
TENSORRT_FP32_RELATIVE_TOLERANCE = 1e-3


@needs_cuda
@needs_tensorrt
def test_tensorrt_fp32_engine_matches_cpu_and_logs_no_reduced_precision(tmp_path, monkeypatch):
    from peddet_orin import engine_build
    from peddet_orin.backends import TensorRtBackend
    from peddet_orin.export import export

    network = SsdNetwork(lively_detector(SEED)).eval()
    export(network, tmp_path / "network.onnx")
    RecordingBuildLog.opened = []
    monkeypatch.setattr(engine_build.runlog, "RunLog", RecordingBuildLog)
    output = tmp_path / "fp32-trt.engine"
    arguments = ["--precision", "fp32-trt", "--onnx", str(tmp_path / "network.onnx"), "--output", str(output)]
    assert engine_build.main(arguments) == 0
    results = RecordingBuildLog.opened[0].results
    assert results["builder_flags"] == engine_build.runlog.BUILDER_FLAG_NAMES["fp32-trt"] == []
    assert results["engine_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    batch = seeded_batch()
    with TensorRtBackend(output) as engine:
        assert relative_error(cpu_reference(network, batch), engine(batch)) < TENSORRT_FP32_RELATIVE_TOLERANCE


# The TensorRT FP16 and INT8 engines against the TensorRT FP32 engine, by the cosine similarity of each output: the
# same runtime on the same input, so what differs is the reduced precision. Set from one board run of
# scripts/gpu_bounds.py (its log's sha256 begins 0fb14a18dc74f700): "BOUND fp16_vs_fp32_trt_cosine 0.999734",
# "BOUND int8_vs_fp32_trt_cosine 0.917264", and an INT8 engine calibrated on unnormalised input (the
# int8-calibration-unnormalised plant) "BOUND int8_unnormalised_vs_fp32_trt_cosine 0". Two later board runs gave
# INT8 cosines of 0.917473 and 0.917557, and FP16 0.999799 and 0.999851; the unnormalised engine gave 0 each time.
FP16_VS_FP32_TRT_MIN_COSINE = 0.999
INT8_VS_FP32_TRT_MIN_COSINE = 0.85


def onnx_argument(arguments: list[str]) -> Path:
    return Path(arguments[arguments.index("--onnx") + 1])


def cosine_to_fp32_trt(fp32_trt_engine: Path, candidate) -> float:
    """The lowest cosine similarity of ``candidate``'s outputs on the seeded batch to the TensorRT FP32 engine's."""
    from peddet_orin.backends import TensorRtBackend

    batch = seeded_batch()
    with TensorRtBackend(fp32_trt_engine) as engine:
        reference = engine(batch)
    return min_cosine(reference, candidate(batch))


# INT8 builds use a small, test-only calibration set: the real 500 images are not needed to show each route works.
INT8_CALIBRATION_IMAGES = 8
CALIBRATION_IMAGE_SIZE = (400, 300)  # Width, height.


class RecordingBuildLog:
    """Stands in for RunLog during a test build, keeping the results instead of writing under runs/."""

    opened: list["RecordingBuildLog"] = []

    def __init__(self, kind, precision, seed, iterations) -> None:
        self.kind, self.precision, self.iterations = kind, precision, iterations
        self.results = {}
        RecordingBuildLog.opened.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exception) -> bool:
        return False


def write_calibration_set(directory: Path) -> tuple[Path, Path]:
    """Write 8 seeded calibration JPEGs under ``directory`` and their manifest; return both paths."""
    calibration = directory / "calibration"
    calibration.mkdir()
    generator = np.random.default_rng(SEED)
    lines = []
    for index in range(INT8_CALIBRATION_IMAGES):
        name = f"{index:012d}.jpg"
        pixels = generator.integers(0, 256, (CALIBRATION_IMAGE_SIZE[1], CALIBRATION_IMAGE_SIZE[0], 3), dtype=np.uint8)
        Image.fromarray(pixels).save(calibration / name)
        lines.append(f"{hashlib.sha256((calibration / name).read_bytes()).hexdigest()}  {name}\n")
    manifest = directory / "calibration.sha256"
    manifest.write_text("".join(lines))
    return calibration, manifest


@pytest.fixture
def int8_inputs(tmp_path, monkeypatch):
    """Export the seeded network and write 8 calibration JPEGs with a test-only manifest."""
    from peddet_orin import engine_build
    from peddet_orin.export import export

    network = SsdNetwork(lively_detector(SEED)).eval()
    onnx_path = tmp_path / "network.onnx"
    export(network, onnx_path)
    calibration, manifest = write_calibration_set(tmp_path)
    monkeypatch.setattr(engine_build, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(engine_build, "EXPECTED_CALIBRATION_IMAGES", INT8_CALIBRATION_IMAGES)
    RecordingBuildLog.opened = []
    monkeypatch.setattr(engine_build.runlog, "RunLog", RecordingBuildLog)
    output = tmp_path / "engines" / "int8.engine"
    arguments = [
        "--precision",
        "int8",
        "--onnx",
        str(onnx_path),
        "--output",
        str(output),
        "--calibration-dir",
        str(calibration),
    ]
    return network, arguments, output


def check_int8_engine_runs(network: SsdNetwork, engine_path: Path) -> None:
    """The engine loads and gives finite outputs of the network's shapes."""
    from peddet_orin.backends import TensorRtBackend

    batch = seeded_batch()
    reference = cpu_reference(network, batch)
    with TensorRtBackend(engine_path) as engine:
        outputs = engine(batch)
    for want, got in zip(reference, outputs, strict=True):
        assert got.shape == want.shape
        assert bool(torch.isfinite(got).all())


@needs_cuda
@needs_tensorrt
def test_tensorrt_int8_entropy_engine_builds_runs_and_logs_its_path(int8_inputs):
    from peddet_orin import engine_build

    network, arguments, output = int8_inputs
    assert engine_build.main(arguments) == 0
    log = RecordingBuildLog.opened[0]
    assert log.results["int8_path"] == engine_build.PATH_ENTROPY
    assert log.results["entropy_failure"] == ""
    assert log.results["builder_flags"] == engine_build.runlog.BUILDER_FLAG_NAMES["int8"] == ["FP16", "INT8", "TF32"]
    assert log.results["engine_bytes"] == output.stat().st_size > 0
    assert log.iterations == {"calibration_images": INT8_CALIBRATION_IMAGES}
    check_int8_engine_runs(network, output)


@needs_cuda
@needs_tensorrt
def test_tensorrt_int8_qdq_fallback_engine_builds_runs_and_logs_its_path(int8_inputs, monkeypatch):
    from peddet_orin import engine_build

    def entropy_fails(*arguments):
        raise engine_build.TensorRtBuildError("entropy build failure forced by the test")

    monkeypatch.setattr(engine_build, "_build_int8_entropy", entropy_fails)
    network, arguments, output = int8_inputs
    assert engine_build.main(arguments) == 0
    log = RecordingBuildLog.opened[0]
    assert log.results["int8_path"] == engine_build.PATH_QDQ
    assert log.results["builder_flags"] == engine_build.runlog.BUILDER_FLAG_NAMES["int8"] == ["FP16", "INT8", "TF32"]
    assert log.results["entropy_failure"] == "TensorRtBuildError: entropy build failure forced by the test"
    assert (output.parent / engine_build.QDQ_ONNX_NAME).is_file()
    check_int8_engine_runs(network, output)


@needs_cuda
@needs_tensorrt
def test_tensorrt_fp16_and_int8_engines_match_the_fp32_engine(fp16_engine, int8_inputs):
    from peddet_orin import engine_build
    from peddet_orin.backends import TensorRtBackend

    _, fp16 = fp16_engine
    _, arguments, int8_output = int8_inputs
    assert engine_build.main(arguments) == 0
    fp32_trt = int8_output.parent / "fp32-trt.engine"
    engine_build.build_fp32_trt(onnx_argument(arguments), fp32_trt)
    assert cosine_to_fp32_trt(fp32_trt, fp16) >= FP16_VS_FP32_TRT_MIN_COSINE
    with TensorRtBackend(int8_output) as int8:
        assert cosine_to_fp32_trt(fp32_trt, int8) >= INT8_VS_FP32_TRT_MIN_COSINE


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("side", ["reference", "candidate"])
def test_a_non_finite_output_fails_the_relative_error_check(bad, side):
    reference = (torch.ones(2, 3), torch.ones(4))
    candidate = (torch.ones(2, 3), torch.ones(4))
    target = reference if side == "reference" else candidate
    target[1][2] = bad
    assert relative_error(reference, candidate) == math.inf
    assert relative_error((torch.ones(2, 3),), (torch.ones(2, 3),)) == 0.0
