# SPDX-License-Identifier: Apache-2.0
"""The pretrained weights load only after their full sha256 matches, and only as tensors."""

import pytest
import torch

from peddet_orin import model


def test_a_weights_file_with_another_sha256_is_refused(tmp_path):
    fake = tmp_path / "weights.pth"
    torch.save({"weight": torch.zeros(1)}, fake)
    with pytest.raises(model.WeightsError, match="expected " + model.WEIGHTS_SHA256):
        model.verified_state_dict(fake)


def test_a_matching_weights_file_loads_as_tensors_only(tmp_path, monkeypatch):
    path = tmp_path / "weights.pth"
    torch.save({"weight": torch.ones(2)}, path)
    monkeypatch.setattr(model, "WEIGHTS_SHA256", model.file_sha256(path))
    calls = []
    real_load = torch.load

    def recording_load(*args, **kwargs):
        calls.append(kwargs.get("weights_only"))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)
    assert torch.equal(model.verified_state_dict(path)["weight"], torch.ones(2))
    assert calls == [True]


def test_the_cached_weights_file_is_used_without_a_download(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path))
    cached = tmp_path / "checkpoints" / model.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1.url.rsplit("/", 1)[-1]
    cached.parent.mkdir()
    cached.write_bytes(b"cached")
    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda *a, **k: pytest.fail("downloaded"))
    assert model.weights_file() == cached


def test_an_untrained_detector_loads_no_weights(monkeypatch):
    monkeypatch.setattr(model, "weights_file", lambda: pytest.fail("weights requested"))
    assert not model.load_detector(pretrained=False).training
