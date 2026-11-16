# SPDX-License-Identifier: Apache-2.0
"""The single image preprocessing path shared by calibration, evaluation and benchmarks.

torchvision's SSDLite normalises each image and resizes it bilinearly to a fixed
320x320. The network graph that this project exports begins after that step, so
the same operations are done here, once, for every precision.

Normalisation is a separate function because the ONNX-against-PyTorch check has
to be able to skip it and observe the failure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

INPUT_SIZE = (320, 320)
INPUT_SHAPE = (1, 3, *INPUT_SIZE)
# SSDLite320's documented transform maps [0, 1] pixels onto [-1, 1].
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
PIXEL_MAX = 255.0


def load_rgb(path: Path | str) -> torch.Tensor:
    """Decode an image file to a CHW float tensor in [0, 1]; greyscale becomes RGB."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).float() / PIXEL_MAX


def resize(image: torch.Tensor) -> torch.Tensor:
    """Resize a CHW image to an NCHW batch of one at the fixed network input size."""
    return functional.interpolate(image[None], size=INPUT_SIZE, mode="bilinear", align_corners=False)


def normalise(batch: torch.Tensor) -> torch.Tensor:
    """Map [0, 1] pixels to the range the network was trained on."""
    return (batch - IMAGE_MEAN) / IMAGE_STD


def check_image(image: torch.Tensor) -> None:
    """Raise ``ValueError`` unless ``image`` is a CHW RGB float tensor with values in [0, 1].

    An HWC tensor or 0-255 pixels would otherwise pass through and give a wrong
    batch shape or wrong values, with no error anywhere.
    """
    if not isinstance(image, torch.Tensor):
        raise ValueError(f"expected a torch.Tensor image, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"expected a CHW image with 3 channels first, got shape {tuple(image.shape)}")
    if image.shape[1] == 0 or image.shape[2] == 0:
        raise ValueError(f"expected a non-empty image, got shape {tuple(image.shape)}")
    if not image.is_floating_point():
        raise ValueError(f"expected a float image with values in [0, 1], got dtype {image.dtype}")
    low, high = float(image.min()), float(image.max())
    if not (low >= 0.0 and high <= 1.0):
        raise ValueError(f"expected values in [0, 1], got values from {low:g} to {high:g}")


def preprocess(image: torch.Tensor) -> torch.Tensor:
    """Run the complete preprocessing for one decoded image, after checking it.

    Normalising before resizing is torchvision's own order. The result is the same
    in exact arithmetic, but matching the order also matches the rounding.
    """
    check_image(image)
    return resize(normalise(image))
