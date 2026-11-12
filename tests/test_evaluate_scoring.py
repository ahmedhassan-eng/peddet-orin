# SPDX-License-Identifier: Apache-2.0
"""Person AP scoring against synthetic COCO annotations whose answers are known by hand."""

import contextlib
import io
import json

import pytest
import torch
from PIL import Image
from pycocotools.coco import COCO

from peddet_orin import evaluate
from peddet_orin.postprocess import Detections

PERSON = 1
CAR = 3
# xywh person boxes: two on image 1, one on image 2.
PERSON_BOXES = {1: [[10, 10, 100, 100], [200, 50, 80, 160]], 2: [[30, 40, 120, 90]]}
CAR_BOX = [300, 300, 60, 40]


def synthetic_coco(directory) -> COCO:
    """Write a small COCO annotation file and load it."""
    annotations = []
    for image_id, boxes in PERSON_BOXES.items():
        for box in boxes:
            annotations.append({"image_id": image_id, "category_id": PERSON, "bbox": box})
    annotations.append({"image_id": 1, "category_id": CAR, "bbox": CAR_BOX})
    for number, annotation in enumerate(annotations, start=1):
        annotation.update({"id": number, "area": annotation["bbox"][2] * annotation["bbox"][3], "iscrowd": 0})
    record = {
        "images": [
            {"id": image_id, "file_name": f"{image_id:012d}.jpg", "width": 640, "height": 480}
            for image_id in PERSON_BOXES
        ],
        "annotations": annotations,
        "categories": [{"id": PERSON, "name": "person"}, {"id": CAR, "name": "car"}],
    }
    path = directory / "instances.json"
    path.write_text(json.dumps(record))
    with contextlib.redirect_stdout(io.StringIO()):
        return COCO(str(path))


def as_detections(boxes_by_image, category=PERSON) -> list[dict]:
    return [
        {"image_id": image_id, "category_id": category, "bbox": list(box), "score": 0.9}
        for image_id, boxes in boxes_by_image.items()
        for box in boxes
    ]


@pytest.fixture
def ground_truth(tmp_path):
    return synthetic_coco(tmp_path)


def test_evaluate_ground_truth_as_detections_scores_one(ground_truth):
    scores = evaluate.person_ap(ground_truth, as_detections(PERSON_BOXES), sorted(PERSON_BOXES))
    assert scores == {"ap": pytest.approx(1.0), "ap50": pytest.approx(1.0)}


def test_evaluate_shifted_boxes_lower_the_ap(ground_truth):
    shifted = {image_id: [[x + 0.3 * w, y, w, h] for x, y, w, h in boxes] for image_id, boxes in PERSON_BOXES.items()}
    scores = evaluate.person_ap(ground_truth, as_detections(shifted), sorted(PERSON_BOXES))
    # A 30% horizontal shift gives IoU 0.7/1.3 = 0.54: matched at IoU 0.50 only.
    assert scores["ap50"] == pytest.approx(1.0)
    assert scores["ap"] < 0.2


def test_evaluate_scores_only_the_person_category(ground_truth):
    # The car in the ground truth has no detection; it would halve a mean over all categories.
    assert evaluate.person_ap(ground_truth, as_detections(PERSON_BOXES), sorted(PERSON_BOXES))["ap"] == pytest.approx(
        1.0
    )
    # Perfect boxes labelled as another class are not person detections.
    as_cars = evaluate.person_ap(ground_truth, as_detections(PERSON_BOXES, CAR), sorted(PERSON_BOXES))
    assert as_cars == {"ap": 0.0, "ap50": 0.0}


def test_evaluate_ap_is_averaged_over_iou_thresholds(ground_truth):
    # Each box keeps its width and 72% of its height: IoU 0.72 matches at thresholds 0.50 to 0.70,
    # five of the ten in AP@[.5:.95], so AP is 0.5 while AP50 is 1.0.
    trimmed = {image_id: [[x, y, w, 0.72 * h] for x, y, w, h in boxes] for image_id, boxes in PERSON_BOXES.items()}
    scores = evaluate.person_ap(ground_truth, as_detections(trimmed), sorted(PERSON_BOXES))
    assert scores["ap50"] == pytest.approx(1.0)
    assert scores["ap"] == pytest.approx(0.5)


def test_evaluate_no_person_found_scores_zero(ground_truth):
    assert evaluate.person_ap(ground_truth, [], sorted(PERSON_BOXES)) == {"ap": 0.0, "ap50": 0.0}


class _FixedPostprocess:
    """Return the ground-truth person boxes of each image as xyxy, plus a non-person box."""

    def __init__(self) -> None:
        self.image_ids = iter(sorted(PERSON_BOXES))

    def __call__(self, box_regression, class_logits, original_size) -> Detections:
        boxes = [[x, y, x + w, y + h] for x, y, w, h in PERSON_BOXES[next(self.image_ids)]]
        boxes.append([0.0, 0.0, 5.0, 5.0])
        labels = [PERSON] * (len(boxes) - 1) + [CAR]
        return Detections(
            torch.tensor(boxes, dtype=torch.float32), torch.full((len(boxes),), 0.9), torch.tensor(labels)
        )


def _fake_backend(batch):
    return torch.zeros(1), torch.zeros(1)


def test_evaluate_writes_detection_boxes_as_xywh(tmp_path, ground_truth):
    for image_id in PERSON_BOXES:
        Image.new("RGB", (64, 48)).save(tmp_path / f"{image_id:012d}.jpg")
    found = evaluate.detect_people(_fake_backend, _FixedPostprocess(), ground_truth, sorted(PERSON_BOXES), tmp_path)
    assert [result["bbox"] for result in found] == [
        box for image_id in sorted(PERSON_BOXES) for box in PERSON_BOXES[image_id]
    ]
    assert {result["category_id"] for result in found} == {evaluate.PERSON_CATEGORY_ID}
    assert evaluate.person_ap(ground_truth, found, sorted(PERSON_BOXES))["ap"] == pytest.approx(1.0)
