"""Model loading, and the network-only module that gets exported.

``SsdNetwork`` holds the backbone and head and nothing else: it takes a normalised
1x3x320x320 batch and returns the raw box regression and class logits. Anchors,
decoding, filtering and NMS belong to ``peddet_orin.postprocess``.
"""
from __future__ import annotations

import warnings

import torch
from torch import nn
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

import peddet_orin.debug as debug

BBOX_KEY = "bbox_regression"
LOGITS_KEY = "cls_logits"
# Torch warns about compute capability 8.7 on its first CUDA use; only that warning is filtered.
SM87_WARNING_PATTERN = r".*compute capability.*8\.7.*"


def load_detector(pretrained: bool) -> nn.Module:
    """Build SSDLite in eval mode. Without ``pretrained`` it fetches nothing at all."""
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1 if pretrained else None
    # weights_backbone defaults to an ImageNet download, so it must be disabled explicitly.
    model = ssdlite320_mobilenet_v3_large(weights=weights, weights_backbone=None)
    debug.log(f"detector loaded, pretrained={pretrained}")
    return model.eval()


def to_device(module: nn.Module, device: str) -> nn.Module:
    """Move a module, filtering only the compute-capability warning for this board."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=SM87_WARNING_PATTERN)
        return module.to(device)


class SsdNetwork(nn.Module):
    """Backbone plus head: normalised batch in, raw regression and logits out."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.backbone = detector.backbone
        self.head = detector.head

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = list(self.backbone(batch).values())
        outputs = self.head(features)
        return outputs[BBOX_KEY], outputs[LOGITS_KEY]
