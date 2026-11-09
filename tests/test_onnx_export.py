"""ONNX Runtime on CPU against PyTorch, using an unweighted model and a fixed seed."""
import pytest
import torch

from peddet_orin.export import OnnxMismatch, assert_matches, export, seeded_batches
from peddet_orin.model import SsdNetwork

from helpers import lively_detector

SEED = 321
BATCHES = 2
# Far above the match tolerance, so a wrong graph could not pass by being near zero.
MEANINGFUL_OUTPUT = 0.1


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    network = SsdNetwork(lively_detector(SEED))
    path = tmp_path_factory.mktemp("onnx") / "network.onnx"
    export(network, path)
    return network, path


def test_model_outputs_are_large_enough_to_make_the_check_meaningful(exported):
    network, _ = exported
    with torch.no_grad():
        outputs = network(torch.rand(1, 3, 320, 320, generator=torch.Generator().manual_seed(SEED)))
    assert all(output.abs().max() > MEANINGFUL_OUTPUT for output in outputs)


def test_onnx_matches_pytorch(exported):
    network, path = exported
    assert assert_matches(network, path, seeded_batches(SEED, BATCHES)) >= 0.0


def test_check_fails_when_normalisation_is_skipped(exported):
    network, path = exported
    with pytest.raises(OnnxMismatch):
        assert_matches(network, path, seeded_batches(SEED, BATCHES), normalise_onnx_input=False)


def test_graph_outputs_are_raw_regression_and_logits(exported):
    network, _ = exported
    box_regression, class_logits = network(torch.zeros(1, 3, 320, 320))
    assert box_regression.shape[-1] == 4
    assert class_logits.shape[:2] == box_regression.shape[:2]
