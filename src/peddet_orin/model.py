"""Model loading, and the network-only module that gets exported.

``SsdNetwork`` holds the backbone and head and nothing else: it takes a normalised
1x3x320x320 batch and returns the raw box regression and class logits. Anchors,
decoding, filtering and NMS belong to ``peddet_orin.postprocess``.

No warning is filtered here. PyTorch warns on its first CUDA use that it does not
support this board's GPU, and that warning stays visible to whoever runs the code.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

import peddet_orin.debug as debug

BBOX_KEY = "bbox_regression"
LOGITS_KEY = "cls_logits"
# The full sha256 of torchvision's COCO_V1 weights file. torchvision checks only the 8-digit prefix in the
# file name, and only when it downloads, so a cached file is checked here on every load.
WEIGHTS_SHA256 = "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2"
HASH_CHUNK_BYTES = 1 << 20


class WeightsError(RuntimeError):
    """The weights file does not match its pinned sha256."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weights_file() -> Path:
    """Return the cached COCO_V1 weights file, downloading it first if absent."""
    url = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.url
    path = Path(torch.hub.get_dir()) / "checkpoints" / url.rsplit("/", 1)[-1]
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(path), progress=False)
    return path


def verified_state_dict(path: Path) -> dict:
    """Load a weights file as tensors only, after checking its full sha256."""
    digest = file_sha256(path)
    if digest != WEIGHTS_SHA256:
        raise WeightsError(
            f"weights file {path.name} has sha256 {digest}, expected {WEIGHTS_SHA256}; delete it to refetch"
        )
    return torch.load(path, map_location="cpu", weights_only=True)


def load_detector(pretrained: bool) -> nn.Module:
    """Build SSDLite in eval mode. Without ``pretrained`` it fetches nothing at all.

    The pretrained weights are loaded as tensors only (``weights_only=True``), after their full sha256 is checked,
    so a replaced file in the cache can neither run code nor load silently.
    """
    # weights_backbone defaults to an ImageNet download, so it must be disabled explicitly. COCO has 91 classes.
    model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=91)
    if pretrained:
        model.load_state_dict(verified_state_dict(weights_file()))
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
