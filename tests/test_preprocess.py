# SPDX-License-Identifier: Apache-2.0
"""The shared preprocessing refuses images that would silently give a wrong batch."""

import pytest
import torch

from peddet_orin.preprocess import INPUT_SHAPE, PIXEL_MAX, preprocess

HEIGHT, WIDTH = 48, 64


def chw_image() -> torch.Tensor:
    return torch.rand(3, HEIGHT, WIDTH, generator=torch.Generator().manual_seed(5))


def test_a_chw_float_image_in_unit_range_gives_one_network_batch():
    batch = preprocess(chw_image())
    assert tuple(batch.shape) == INPUT_SHAPE
    assert float(batch.min()) >= -1.0 and float(batch.max()) <= 1.0


def test_an_hwc_image_is_rejected_instead_of_reshaped():
    with pytest.raises(ValueError, match=r"3 channels first, got shape \(48, 64, 3\)"):
        preprocess(chw_image().permute(1, 2, 0))


def test_uint8_pixels_are_rejected():
    pixels = (chw_image() * PIXEL_MAX).to(torch.uint8)
    with pytest.raises(ValueError, match="dtype torch.uint8"):
        preprocess(pixels)


def test_float_pixels_on_a_0_to_255_scale_are_rejected():
    with pytest.raises(ValueError, match=r"values in \[0, 1\]"):
        preprocess(chw_image() * PIXEL_MAX)


def test_negative_pixels_are_rejected():
    with pytest.raises(ValueError, match=r"values in \[0, 1\]"):
        preprocess(chw_image() - 0.5)


def test_nan_pixels_are_rejected():
    image = chw_image()
    image[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match=r"values in \[0, 1\]"):
        preprocess(image)


@pytest.mark.parametrize("shape", [(1, 3, HEIGHT, WIDTH), (HEIGHT, WIDTH), (1, HEIGHT, WIDTH), (4, HEIGHT, WIDTH)])
def test_images_without_three_leading_channels_are_rejected(shape):
    with pytest.raises(ValueError, match="3 channels first"):
        preprocess(torch.rand(shape))


@pytest.mark.parametrize("shape", [(3, 0, WIDTH), (3, HEIGHT, 0)])
def test_an_empty_image_is_rejected(shape):
    with pytest.raises(ValueError, match="non-empty"):
        preprocess(torch.rand(shape))


def test_a_numpy_array_is_rejected():
    with pytest.raises(ValueError, match="torch.Tensor"):
        preprocess(chw_image().numpy())
