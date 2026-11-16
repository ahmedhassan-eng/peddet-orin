# SPDX-License-Identifier: Apache-2.0
"""ONNX Runtime on CPU against PyTorch, using an unweighted model and a fixed seed."""

import onnx
import pytest
import torch

from helpers import lively_detector
from peddet_orin.export import (
    INPUT_NAME,
    MATCH_TOLERANCE,
    OPSET,
    OUTPUT_NAMES,
    OnnxMismatch,
    assert_matches,
    export,
    export_results,
    max_difference,
    parse_arguments,
    seeded_batches,
)
from peddet_orin.model import SsdNetwork

SEED = 321
BATCHES = 2
# Far above the match tolerance, so a wrong graph could not pass by being near zero.
MEANINGFUL_OUTPUT = 0.1
EXPECTED_OPSET = 17
EXPECTED_INPUT_SHAPE = [1, 3, 320, 320]
ANCHORS = 3234  # SSDLite320's anchor count over its six feature maps.
COCO_CLASSES = 91  # Background plus torchvision's 90 COCO category slots.
EXPECTED_OUTPUT_SHAPES = {"bbox_regression": [1, ANCHORS, 4], "cls_logits": [1, ANCHORS, COCO_CLASSES]}
# Every operator the exported network uses. Decoding, NMS or TopK in the graph would add to this set.
EXPECTED_OPERATORS = {
    "Add",
    "Clip",
    "Concat",
    "Constant",
    "Conv",
    "GlobalAveragePool",
    "HardSigmoid",
    "HardSwish",
    "Identity",
    "Mul",
    "Relu",
    "Reshape",
    "Transpose",
}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    network = SsdNetwork(lively_detector(SEED))
    path = tmp_path_factory.mktemp("onnx") / "network.onnx"
    export(network, path)
    return network, path


def dims(value_info) -> list[int]:
    return [dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim]


def test_model_outputs_are_large_enough_to_make_the_check_meaningful(exported):
    network, _ = exported
    with torch.no_grad():
        outputs = network(torch.rand(1, 3, 320, 320, generator=torch.Generator().manual_seed(SEED)))
    assert all(output.abs().max() > MEANINGFUL_OUTPUT for output in outputs)


def test_onnx_matches_pytorch(exported):
    network, path = exported
    worst = max_difference(network, path, seeded_batches(SEED, BATCHES))
    assert 0.0 <= worst <= MATCH_TOLERANCE
    assert assert_matches(network, path, seeded_batches(SEED, BATCHES)) == worst


def test_check_fails_when_normalisation_is_skipped(exported):
    network, path = exported
    with pytest.raises(OnnxMismatch):
        assert_matches(network, path, seeded_batches(SEED, BATCHES), normalise_onnx_input=False)


def test_graph_outputs_are_raw_regression_and_logits(exported):
    _, path = exported
    graph_model = onnx.load(str(path))
    onnx.checker.check_model(graph_model)
    graph = graph_model.graph
    assert [(item.name, dims(item)) for item in graph.input] == [(INPUT_NAME, EXPECTED_INPUT_SHAPE)]
    assert [(item.name, dims(item)) for item in graph.output] == list(EXPECTED_OUTPUT_SHAPES.items())
    assert tuple(EXPECTED_OUTPUT_SHAPES) == OUTPUT_NAMES
    assert [(entry.domain, entry.version) for entry in graph_model.opset_import] == [("", EXPECTED_OPSET)]
    assert {node.op_type for node in graph.node} == EXPECTED_OPERATORS


def test_export_log_results_carry_the_input_shape():
    results = export_results(0.25)
    assert results == {
        "max_abs_difference": 0.25,
        "tolerance": MATCH_TOLERANCE,
        "opset": OPSET,
        "input_shape": EXPECTED_INPUT_SHAPE,
    }
    assert all(isinstance(side, int) for side in results["input_shape"])


def test_export_cli_help_names_its_module_and_documents_output(capsys):
    with pytest.raises(SystemExit) as stopped:
        parse_arguments(["--help"])
    assert stopped.value.code == 0
    text = capsys.readouterr().out
    assert text.startswith("usage: python -m peddet_orin.export")
    assert "where to write the ONNX file" in text
    assert "(default:" in text
