# SPDX-License-Identifier: Apache-2.0
"""Test support: an SSDLite with no downloaded weights but meaningful outputs."""
import torch
from torch import nn

from peddet_orin.model import load_detector


def lively_detector(seed: int) -> nn.Module:
    """Build an unweighted detector whose outputs are far from zero.

    With ``weights=None`` the network's outputs are around 1e-15, so any two graphs
    would agree within a tolerance. Re-initialising every convolution with a seeded
    Kaiming draw gives outputs of ordinary size, without fetching any weights.
    """
    torch.manual_seed(seed)
    detector = load_detector(pretrained=False)
    for module in detector.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return detector.eval()
