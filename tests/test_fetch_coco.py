# SPDX-License-Identifier: Apache-2.0
"""scripts/fetch_coco.py on CPU, with every download replaced by a local stand-in."""
import hashlib
import re

import pytest

import fetch_coco

TRAIN = [f"{number:012d}.jpg" for number in range(1, 41)]
VAL = [f"{number:012d}.jpg" for number in range(100, 110)]
COUNT = 5


def image_bytes(name: str) -> bytes:
    return f"image {name}".encode()


def manifest_text(names: list[str]) -> str:
    return "".join(f"{hashlib.sha256(image_bytes(name)).hexdigest()}  {name}\n" for name in names)


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """Point the script at tmp_path, list TRAIN and VAL, and serve each image from memory."""
    downloads: list[str] = []

    def fake_stream(url, destination, digests):
        assert url.startswith("https://images.cocodataset.org/")
        downloads.append(url)
        data = image_bytes(url.rsplit("/", 1)[1])
        destination.write_bytes(data)
        return {name: hashlib.new(name, data).hexdigest() for name in digests}

    monkeypatch.setattr(fetch_coco, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(fetch_coco, "MANIFEST_PATH", tmp_path / "manifests" / "calibration.sha256")
    monkeypatch.setattr(fetch_coco, "CALIBRATION_COUNT", COUNT)
    monkeypatch.setattr(fetch_coco, "_stream_to", fake_stream)
    monkeypatch.setattr(fetch_coco, "_image_names",
                        lambda name: list(VAL) if name == fetch_coco.VAL_IMAGE_LIST else list(TRAIN))
    return downloads


def chosen() -> list[str]:
    return fetch_coco.select_calibration_names(TRAIN, fetch_coco.CALIBRATION_SEED, COUNT)


def test_the_seeded_selection_is_deterministic_and_depends_on_the_seed():
    first = fetch_coco.select_calibration_names(TRAIN, 2017, COUNT)
    assert first == fetch_coco.select_calibration_names(list(reversed(TRAIN)), 2017, COUNT)
    assert len(first) == COUNT and first == sorted(first) and set(first) <= set(TRAIN)
    assert fetch_coco.select_calibration_names(TRAIN, 2018, COUNT) != first


def test_a_calibration_set_overlapping_val2017_is_rejected(offline, monkeypatch):
    fetch_coco.MANIFEST_PATH.parent.mkdir(parents=True)
    fetch_coco.MANIFEST_PATH.write_text(manifest_text(chosen()))
    monkeypatch.setattr(fetch_coco, "_image_names",
                        lambda name: chosen()[:1] if name == fetch_coco.VAL_IMAGE_LIST else list(TRAIN))
    with pytest.raises(RuntimeError, match="overlaps val2017 in 1 files"):
        fetch_coco.fetch_calibration()
    assert offline == []


def test_a_manifest_with_the_wrong_image_count_is_rejected(offline):
    fetch_coco.MANIFEST_PATH.parent.mkdir(parents=True)
    fetch_coco.MANIFEST_PATH.write_text(manifest_text(chosen()[:-1]))
    with pytest.raises(RuntimeError, match=f"manifest lists {COUNT - 1} images, expected {COUNT}"):
        fetch_coco.fetch_calibration()
    assert offline == []


def test_a_train_list_shorter_than_the_calibration_count_is_rejected():
    with pytest.raises(RuntimeError, match=f"train2017 lists 3 images, fewer than the {COUNT}"):
        fetch_coco.select_calibration_names(TRAIN[:3], 2017, COUNT)


def test_a_missing_manifest_is_an_error_without_the_flag(offline, capsys):
    assert fetch_coco.main(["calibration"]) == 1
    error = capsys.readouterr().err.strip()
    assert "\n" not in error and "--write-manifest" in error
    assert offline == [] and not fetch_coco.MANIFEST_PATH.exists()


def test_write_manifest_records_a_missing_manifest_from_the_fetched_images(offline):
    assert fetch_coco.main(["calibration", "--write-manifest"]) == 0
    assert fetch_coco.MANIFEST_PATH.read_text() == manifest_text(chosen())
    assert len(offline) == COUNT


def test_an_existing_manifest_is_verified_and_not_rewritten(offline):
    fetch_coco.MANIFEST_PATH.parent.mkdir(parents=True)
    fetch_coco.MANIFEST_PATH.write_text(manifest_text(chosen()))
    before = fetch_coco.MANIFEST_PATH.stat().st_mtime_ns
    fetch_coco.fetch_calibration()
    assert fetch_coco.MANIFEST_PATH.stat().st_mtime_ns == before


def test_a_hash_mismatch_keeps_the_file_and_names_it(offline, capsys):
    fetch_coco.MANIFEST_PATH.parent.mkdir(parents=True)
    fetch_coco.MANIFEST_PATH.write_text(manifest_text(chosen()))
    mine = fetch_coco.DATA_DIR / fetch_coco.CALIBRATION_DIR_NAME / chosen()[0]
    mine.parent.mkdir(parents=True)
    mine.write_bytes(b"the user's own edited copy")
    assert fetch_coco.main(["calibration"]) == 1
    error = capsys.readouterr().err.strip()
    assert "\n" not in error and "sha256 mismatch" in error and str(mine) in error
    assert mine.read_bytes() == b"the user's own edited copy"


def test_every_download_url_is_https():
    urls = [value for name, value in vars(fetch_coco).items() if name.endswith(("_URL", "_URL_PREFIX"))]
    assert len(urls) == 4
    assert all(url.startswith("https://images.cocodataset.org/") for url in urls), urls


def test_a_plain_http_download_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not HTTPS"):
        fetch_coco._stream_to("http://images.cocodataset.org/zips/val2017.zip", tmp_path / "val2017.zip", ("md5",))
    assert list(tmp_path.iterdir()) == []


def test_every_subcommand_has_help_and_the_docstring_counts_them():
    parser = fetch_coco.build_parser()
    commands = next(action for action in parser._actions if action.dest == "what")
    helps = {choice.dest: choice.help for choice in commands._choices_actions}
    assert set(helps) == {"annotations", "val", "train", "calibration"}
    assert all(helps.values()), helps
    assert re.search(r"^Four independent subcommands", fetch_coco.__doc__, re.M)
    assert "--write-manifest" in commands.choices["calibration"].format_help()
