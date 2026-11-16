# SPDX-License-Identifier: Apache-2.0
"""The shared postprocess, on synthetic tensors whose answers are known."""

import pytest
import torch
from torchvision.models.detection._utils import BoxCoder

from helpers import lively_detector
from peddet_orin.model import SsdNetwork
from peddet_orin.postprocess import Postprocessor
from peddet_orin.preprocess import INPUT_SIZE, preprocess

CODER_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
MAX_DETECTIONS = 10
TOP_K = 10
HIGH_LOGIT = 10.0
SEED = 99
PERSON = 1
OTHER = 2
# The first two anchors overlap with IoU far above the threshold; the third is apart.
ANCHORS = torch.tensor([[10.0, 10.0, 110.0, 110.0], [12.0, 12.0, 112.0, 112.0], [200.0, 200.0, 300.0, 300.0]])


def make_postprocessor(nms: float = NMS_THRESHOLD) -> Postprocessor:
    return Postprocessor(ANCHORS, BoxCoder(CODER_WEIGHTS), SCORE_THRESHOLD, nms, MAX_DETECTIONS, TOP_K)


def logits_for(labels: list[int], num_classes: int = 3) -> torch.Tensor:
    logits = torch.zeros(1, len(labels), num_classes)
    for index, label in enumerate(labels):
        logits[0, index, label] = HIGH_LOGIT - index  # Earlier anchors score higher.
    return logits


def zero_regression() -> torch.Tensor:
    return torch.zeros(1, len(ANCHORS), 4)


def test_zero_regression_decodes_to_the_anchors_and_nms_removes_the_overlap():
    found = make_postprocessor()(zero_regression(), logits_for([PERSON, PERSON, PERSON]), INPUT_SIZE)
    assert found.labels.tolist() == [PERSON, PERSON]
    assert torch.allclose(found.boxes, ANCHORS[[0, 2]])


def test_overlapping_boxes_of_different_classes_both_survive():
    found = make_postprocessor()(zero_regression(), logits_for([PERSON, OTHER, 0]), INPUT_SIZE)
    assert sorted(found.labels.tolist()) == [PERSON, OTHER]


def test_box_regression_is_decoded_with_the_coder_weights():
    regression = zero_regression()
    regression[0, 2, 0] = CODER_WEIGHTS[0] * 0.1  # Shift by a tenth of the 100-pixel width.
    found = make_postprocessor()(regression, logits_for([0, 0, PERSON]), INPUT_SIZE)
    assert torch.allclose(found.boxes, torch.tensor([[210.0, 200.0, 310.0, 300.0]]))


def test_background_and_low_scores_are_dropped():
    logits = torch.zeros(1, 3, 3)
    logits[0, 0, 0] = HIGH_LOGIT  # Confident background yields no detection.
    found = make_postprocessor()(zero_regression(), logits, INPUT_SIZE)
    assert found.boxes.shape[0] == 0


def test_boxes_are_clipped_and_scaled_to_the_original_image():
    regression = zero_regression()
    regression[0, 2, 0] = CODER_WEIGHTS[0] * 1.0  # Pushes the box past the right edge.
    height, width = INPUT_SIZE[0] * 2, INPUT_SIZE[1] * 3
    found = make_postprocessor()(regression, logits_for([0, 0, PERSON]), (height, width))
    expected = torch.tensor([[300.0 * 3, 200.0 * 2, 320.0 * 3, 300.0 * 2]])
    assert torch.allclose(found.boxes, expected)


def test_scores_come_back_in_falling_order():
    found = make_postprocessor(nms=0.99)(zero_regression(), logits_for([PERSON, PERSON, PERSON]), INPUT_SIZE)
    assert found.scores.tolist() == sorted(found.scores.tolist(), reverse=True)
    assert found.boxes.shape[0] == 3


@pytest.mark.parametrize(
    "original_size",
    [(320,), (320, 320, 3), (0, 320), (320, -1), (320.0, 320), (True, 320), "hw"],
)
def test_original_size_must_be_two_positive_ints(original_size):
    with pytest.raises(ValueError, match="original_size"):
        make_postprocessor()(zero_regression(), logits_for([PERSON, PERSON, PERSON]), original_size)


def test_original_size_accepts_a_torch_size():
    found = make_postprocessor()(zero_regression(), logits_for([PERSON, 0, 0]), torch.Size(INPUT_SIZE))
    assert found.boxes.shape[0] == 1


def test_regression_with_the_wrong_anchor_count_is_rejected():
    regression = torch.zeros(1, len(ANCHORS) + 1, 4)
    logits = torch.zeros(1, len(ANCHORS) + 1, 3)
    with pytest.raises(ValueError, match="4 anchors, the postprocess has 3"):
        make_postprocessor()(regression, logits, INPUT_SIZE)


def test_logits_with_the_wrong_anchor_count_are_rejected():
    with pytest.raises(ValueError, match="class logits has 2 anchors"):
        make_postprocessor()(zero_regression(), torch.zeros(1, 2, 3), INPUT_SIZE)


def test_regression_without_four_coordinates_is_rejected():
    with pytest.raises(ValueError, match=r"shape \(1, anchors, 4\)"):
        make_postprocessor()(torch.zeros(1, len(ANCHORS), 5), logits_for([PERSON, 0, 0]), INPUT_SIZE)


@pytest.fixture(scope="module")
def detector():
    return lively_detector(SEED)


def test_pipeline_equals_torchvision_detector(detector):
    """The split network plus postprocess must reproduce torchvision's own output."""
    image = torch.rand(3, 240, 400, generator=torch.Generator().manual_seed(SEED))
    with torch.no_grad():
        expected = detector([image])[0]
        box_regression, class_logits = SsdNetwork(detector)(preprocess(image))
    found = Postprocessor.from_torchvision(detector)(box_regression, class_logits, image.shape[-2:])
    assert torch.allclose(found.boxes, expected["boxes"], atol=1e-3)
    assert torch.allclose(found.scores, expected["scores"], atol=1e-5)
    assert torch.equal(found.labels, expected["labels"])
