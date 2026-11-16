# SPDX-License-Identifier: Apache-2.0
"""PyTorch on CUDA and TensorRT FP16 and INT8 engines, against PyTorch on the CPU.

These need the Jetson: they skip where there is no CUDA device or no TensorRT, as in CI.
They use an unweighted, seeded model and synthetic calibration images, so they need no download.
"""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from helpers import lively_detector
from peddet_orin.model import SsdNetwork
from peddet_orin.preprocess import INPUT_SHAPE, normalise

SEED = 654
# Worst absolute difference relative to the largest reference output.
CUDA_RELATIVE_TOLERANCE = 1e-2
# FP16 is checked by the cosine similarity of each output, since its worst single element varies from build to
# build. Measured on this board: correct engines gave at least 0.9991, un-normalised input at most 0.723, and a
# network with other weights about 0.
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
        worst = max(worst, float((want - got.cpu()).abs().max() / want.abs().max()))
    return worst


def cpu_reference(network: SsdNetwork, batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
    with torch.no_grad():
        return tuple(output.clone() for output in network(batch))


@needs_cuda
def test_cuda_forward_matches_cpu():
    from peddet_orin.backends import PytorchBackend

    detector = lively_detector(SEED)
    batch = seeded_batch()
    reference = cpu_reference(SsdNetwork(detector).eval(), batch)
    # The backend moves the detector to the GPU, so the CPU reference is taken first.
    assert relative_error(reference, PytorchBackend(detector, "cuda")(batch)) < CUDA_RELATIVE_TOLERANCE


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


@pytest.fixture
def int8_inputs(tmp_path, monkeypatch):
    """Export the seeded network and write 8 calibration JPEGs with a test-only manifest."""
    from peddet_orin import engine_build
    from peddet_orin.export import export

    network = SsdNetwork(lively_detector(SEED)).eval()
    onnx_path = tmp_path / "network.onnx"
    export(network, onnx_path)
    calibration = tmp_path / "calibration"
    calibration.mkdir()
    generator = np.random.default_rng(SEED)
    lines = []
    for index in range(INT8_CALIBRATION_IMAGES):
        name = f"{index:012d}.jpg"
        pixels = generator.integers(0, 256, (CALIBRATION_IMAGE_SIZE[1], CALIBRATION_IMAGE_SIZE[0], 3), dtype=np.uint8)
        Image.fromarray(pixels).save(calibration / name)
        lines.append(f"{hashlib.sha256((calibration / name).read_bytes()).hexdigest()}  {name}\n")
    manifest = tmp_path / "calibration.sha256"
    manifest.write_text("".join(lines))
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
    assert log.results["entropy_failure"] == "TensorRtBuildError: entropy build failure forced by the test"
    assert (output.parent / engine_build.QDQ_ONNX_NAME).is_file()
    check_int8_engine_runs(network, output)
