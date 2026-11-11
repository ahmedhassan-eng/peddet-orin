"""PyTorch on CUDA and a TensorRT FP16 engine, against PyTorch on the CPU.

These need the Jetson: they skip where there is no CUDA device or no TensorRT, as in CI.
They use an unweighted, seeded model, so they need no download.
"""
import importlib.util

import pytest
import torch

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
    return network, TensorRtBackend(work / "fp16.engine")


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
