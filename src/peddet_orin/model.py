"""Model loading, and the network-only module that gets exported.

``SsdNetwork`` holds the backbone and head and nothing else: it takes a normalised
1x3x320x320 batch and returns the raw box regression and class logits. Anchors,
decoding, filtering and NMS belong to ``peddet_orin.postprocess``.

No warning is filtered here. PyTorch warns on its first CUDA use that it does not
support this board's GPU, and that warning stays visible to whoever runs the code.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

import peddet_orin.debug as debug

BBOX_KEY = "bbox_regression"
LOGITS_KEY = "cls_logits"


def load_detector(pretrained: bool) -> nn.Module:
    """Build SSDLite in eval mode. Without ``pretrained`` it fetches nothing at all."""
    weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1 if pretrained else None
    # weights_backbone defaults to an ImageNet download, so it must be disabled explicitly.
    model = ssdlite320_mobilenet_v3_large(weights=weights, weights_backbone=None)
    debug.log(f"detector loaded, pretrained={pretrained}")
    return model.eval()


def use_strict_fp32() -> tuple[bool, bool]:
    """Turn TF32 off for cuDNN and matmul, so FP32 runs are FP32; return the flags in effect.

    The returned pair is (cudnn allow_tf32, matmul allow_tf32), read back after setting.
    """
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return bool(torch.backends.cudnn.allow_tf32), bool(torch.backends.cuda.matmul.allow_tf32)


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
