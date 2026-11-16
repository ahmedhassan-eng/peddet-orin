# SPDX-License-Identifier: Apache-2.0
"""The one postprocess: anchors, box decoding, score filtering and NMS.

The exported network stops at raw box regression and class logits. Everything
after that lives here, outside any graph, and every precision goes through this
same code, so accuracy differences between precisions can only come from the
network outputs.

The anchors and the box coder are torchvision's own objects, taken from an SSDLite
model built without weights. The thresholds are read from that same model, which
keeps this module from carrying a second copy of torchvision's configuration.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import boxes as box_ops

from peddet_orin.preprocess import INPUT_SHAPE, INPUT_SIZE

BACKGROUND_LABEL = 0
PERSON_LABEL = 1
LOGITS_DIM = -1
# The deployment-style postprocess keeps the person class only, with one fixed score threshold. 0.5 is a fixed
# midpoint chosen before any deployment-style run; it is not tuned on the results. Everything else is torchvision's.
DEPLOY_SCORE_THRESHOLD = 0.5
DEPLOY_LABELS = (PERSON_LABEL,)


@dataclass
class Detections:
    """Detections for one image, with boxes as xyxy in original-image pixels."""

    boxes: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor


class Postprocessor:
    """Turn raw network outputs into detections, for a batch holding a single image."""

    def __init__(
        self,
        anchors: torch.Tensor,
        box_coder,
        score_threshold: float,
        nms_iou_threshold: float,
        detections_per_image: int,
        top_k_per_class: int,
        num_classes: int | None = None,
        kept_labels: tuple[int, ...] | None = None,
    ) -> None:
        self.anchors = anchors
        self.box_coder = box_coder
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.detections_per_image = detections_per_image
        self.top_k_per_class = top_k_per_class
        # The classes the head scores, background included; recorded in benchmark logs, not used to decode.
        self.num_classes = num_classes
        # The labels kept, or None for every class but background.
        self.kept_labels = kept_labels

    @classmethod
    def from_torchvision(cls, model) -> Postprocessor:
        """Build the postprocess from an SSD model's own generator, coder and settings."""
        probe = torch.zeros(INPUT_SHAPE)
        with torch.no_grad():
            features = list(model.backbone(probe).values())
        image_list = ImageList(probe, [INPUT_SIZE])
        anchors = model.anchor_generator(image_list, features)[0]
        return cls(
            anchors=anchors,
            box_coder=model.box_coder,
            score_threshold=model.score_thresh,
            nms_iou_threshold=model.nms_thresh,
            detections_per_image=model.detections_per_img,
            top_k_per_class=model.topk_candidates,
            num_classes=model.head.classification_head.num_columns,
        )

    def for_deployment(self) -> Postprocessor:
        """The deployment-style postprocess: the person class only, at ``DEPLOY_SCORE_THRESHOLD``, the rest as is."""
        return Postprocessor(
            anchors=self.anchors,
            box_coder=self.box_coder,
            score_threshold=DEPLOY_SCORE_THRESHOLD,
            nms_iou_threshold=self.nms_iou_threshold,
            detections_per_image=self.detections_per_image,
            top_k_per_class=self.top_k_per_class,
            num_classes=self.num_classes,
            kept_labels=DEPLOY_LABELS,
        )

    def check_inputs(
        self,
        box_regression: torch.Tensor,
        class_logits: torch.Tensor,
        original_size: Sequence[int],
    ) -> None:
        """Raise ``ValueError`` unless the outputs fit the anchors and the size is (H, W)."""
        anchor_count = self.anchors.shape[0]
        if box_regression.ndim != 3 or box_regression.shape[0] != 1 or box_regression.shape[2] != 4:
            raise ValueError(f"expected box regression of shape (1, anchors, 4), got {tuple(box_regression.shape)}")
        if class_logits.ndim != 3 or class_logits.shape[0] != 1:
            raise ValueError(f"expected class logits of shape (1, anchors, classes), got {tuple(class_logits.shape)}")
        for name, count in (("box regression", box_regression.shape[1]), ("class logits", class_logits.shape[1])):
            if count != anchor_count:
                raise ValueError(f"{name} has {count} anchors, the postprocess has {anchor_count}")
        size = tuple(original_size)
        if len(size) != 2 or not all(_is_positive_int(side) for side in size):
            raise ValueError(f"expected original_size as (height, width) positive ints, got {original_size!r}")

    def __call__(
        self,
        box_regression: torch.Tensor,
        class_logits: torch.Tensor,
        original_size: Sequence[int],
    ) -> Detections:
        """Decode one image's outputs; ``original_size`` is (height, width)."""
        self.check_inputs(box_regression, class_logits, original_size)
        boxes = self.box_coder.decode_single(box_regression[0], self.anchors)
        boxes = box_ops.clip_boxes_to_image(boxes, INPUT_SIZE)
        scores = torch.softmax(class_logits[0], dim=LOGITS_DIM)

        kept_boxes, kept_scores, kept_labels = [], [], []
        for label in range(scores.shape[LOGITS_DIM]):
            if label == BACKGROUND_LABEL or (self.kept_labels is not None and label not in self.kept_labels):
                continue
            class_scores = scores[:, label]
            mask = class_scores > self.score_threshold
            class_scores, class_boxes = class_scores[mask], boxes[mask]
            count = min(self.top_k_per_class, class_scores.numel())
            class_scores, order = class_scores.topk(count)
            kept_boxes.append(class_boxes[order])
            kept_scores.append(class_scores)
            kept_labels.append(torch.full((count,), label, dtype=torch.int64))

        all_boxes = torch.cat(kept_boxes)
        all_scores = torch.cat(kept_scores)
        all_labels = torch.cat(kept_labels)
        # batched_nms suppresses only within a class and returns indices by falling score.
        keep = box_ops.batched_nms(all_boxes, all_scores, all_labels, self.nms_iou_threshold)
        keep = keep[: self.detections_per_image]
        return Detections(_to_original(all_boxes[keep], original_size), all_scores[keep], all_labels[keep])


def _is_positive_int(value: object) -> bool:
    return isinstance(value, numbers.Integral) and not isinstance(value, bool) and value > 0


def _to_original(boxes: torch.Tensor, original_size: Sequence[int]) -> torch.Tensor:
    """Scale xyxy boxes from network-input pixels to original-image pixels."""
    input_height, input_width = INPUT_SIZE
    height, width = original_size
    factors = torch.tensor([width / input_width, height / input_height] * 2, dtype=boxes.dtype)
    return boxes * factors
