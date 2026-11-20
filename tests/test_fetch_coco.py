# SPDX-License-Identifier: Apache-2.0
"""scripts/fetch_coco.py on CPU, with every download replaced by a local stand-in."""

import hashlib
import http.client
import io
import json
import re
import urllib.parse

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
        assert url.startswith("https://s3.amazonaws.com/images.cocodataset.org/")
        downloads.append(url)
        data = image_bytes(url.rsplit("/", 1)[1])
        destination.write_bytes(data)
        return {name: hashlib.new(name, data).hexdigest() for name in digests}

    monkeypatch.setattr(fetch_coco, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(fetch_coco, "MANIFEST_PATH", tmp_path / "manifests" / "calibration.sha256")
    monkeypatch.setattr(fetch_coco, "CALIBRATION_COUNT", COUNT)
    monkeypatch.setattr(fetch_coco, "_stream_to", fake_stream)
    monkeypatch.setattr(
        fetch_coco, "_image_names", lambda name: list(VAL) if name == fetch_coco.VAL_IMAGE_LIST else list(TRAIN)
    )
    return downloads


def chosen() -> list[str]:
    return fetch_coco.select_calibration_names(TRAIN, fetch_coco.CALIBRATION_SEED, COUNT)


def test_the_seeded_selection_is_deterministic_and_depends_on_the_seed():
    first = fetch_coco.select_calibration_names(TRAIN, 2017, COUNT)
    assert first == fetch_coco.select_calibration_names(list(reversed(TRAIN)), 2017, COUNT)
    assert len(first) == COUNT and first == sorted(first) and set(first) <= set(TRAIN)
    assert fetch_coco.select_calibration_names(TRAIN, 2018, COUNT) != first


def test_the_calibration_seed_picks_the_published_selection():
    # The manifest was drawn with seed 2017; this known answer fails if the module's seed changes.
    assert fetch_coco.select_calibration_names(TRAIN, fetch_coco.CALIBRATION_SEED, COUNT) == [
        "000000000013.jpg",
        "000000000021.jpg",
        "000000000022.jpg",
        "000000000029.jpg",
        "000000000033.jpg",
    ]


def test_a_calibration_set_overlapping_val2017_is_rejected(offline, monkeypatch):
    fetch_coco.MANIFEST_PATH.parent.mkdir(parents=True)
    fetch_coco.MANIFEST_PATH.write_text(manifest_text(chosen()))
    monkeypatch.setattr(
        fetch_coco, "_image_names", lambda name: chosen()[:1] if name == fetch_coco.VAL_IMAGE_LIST else list(TRAIN)
    )
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


def test_an_archive_md5_mismatch_keeps_the_archive_and_names_it(tmp_path, monkeypatch, capsys):
    def fake_stream(url, destination, digests):
        destination.write_bytes(b"not the published archive")
        return {"md5": "0" * 32}

    monkeypatch.setattr(fetch_coco, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(fetch_coco, "_stream_to", fake_stream)
    assert fetch_coco.main(["annotations"]) == 1
    archive = fetch_coco.DATA_DIR / fetch_coco.ANNOTATIONS_ZIP_URL.rsplit("/", 1)[1]
    error = capsys.readouterr().err.strip()
    assert "\n" not in error and "md5 mismatch" in error and str(archive) in error and "left in place" in error
    assert archive.read_bytes() == b"not the published archive"


class Response(io.BytesIO):
    """A urlopen response: its bytes, and the address they finally came from."""

    def __init__(self, payload: bytes, final_url: str):
        super().__init__(payload)
        self.final_url = final_url

    def geturl(self) -> str:
        return self.final_url


def fake_urlopen(payload: bytes | None):
    """A stand-in for urllib.request.urlopen that serves ``payload``, or fails if it is None."""

    def urlopen(url, timeout):
        if payload is None:
            raise OSError("connection refused")
        return Response(payload, url)

    return urlopen


def test_a_second_run_replaces_a_kept_archive_once_the_new_download_completes(tmp_path, monkeypatch):
    archive = tmp_path / "annotations_trainval2017.zip"
    archive.write_bytes(b"the archive a failed md5 check kept")
    monkeypatch.setattr(fetch_coco.urllib.request, "urlopen", fake_urlopen(b"the new download"))
    digests = fetch_coco._stream_to("https://s3.amazonaws.com/images.cocodataset.org/a.zip", archive, ("md5",))
    assert archive.read_bytes() == b"the new download"
    assert digests == {"md5": hashlib.md5(b"the new download").hexdigest()}
    assert sorted(path.name for path in tmp_path.iterdir()) == [archive.name]


def test_a_second_run_whose_download_fails_leaves_the_kept_archive_as_it_is(tmp_path, monkeypatch):
    archive = tmp_path / "annotations_trainval2017.zip"
    archive.write_bytes(b"the archive a failed md5 check kept")
    monkeypatch.setattr(fetch_coco.urllib.request, "urlopen", fake_urlopen(None))
    monkeypatch.setattr(fetch_coco.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="giving up"):
        fetch_coco._stream_to("https://s3.amazonaws.com/images.cocodataset.org/a.zip", archive, ("md5",))
    assert archive.read_bytes() == b"the archive a failed md5 check kept"
    assert sorted(path.name for path in tmp_path.iterdir()) == [archive.name]


def test_a_download_that_breaks_on_every_attempt_leaves_no_partial_file(tmp_path, monkeypatch):
    class Broken(io.BytesIO):
        def geturl(self):
            return "https://s3.amazonaws.com/images.cocodataset.org/a.zip"

        def read(self, size=-1):
            if self.tell():
                raise OSError("connection reset")
            return super().read(4)

    archive = tmp_path / "val2017.zip"
    monkeypatch.setattr(fetch_coco.urllib.request, "urlopen", lambda url, timeout: Broken(b"half an archive"))
    monkeypatch.setattr(fetch_coco.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="giving up"):
        fetch_coco._stream_to("https://s3.amazonaws.com/images.cocodataset.org/a.zip", archive, ("md5",))
    assert list(tmp_path.iterdir()) == []


def test_every_download_url_is_https():
    urls = [value for name, value in vars(fetch_coco).items() if name.endswith(("_URL", "_URL_PREFIX"))]
    assert len(urls) == 4
    assert all(url.startswith("https://s3.amazonaws.com/images.cocodataset.org/") for url in urls), urls


def test_a_plain_http_download_is_refused(tmp_path):
    with pytest.raises(ValueError, match="not HTTPS"):
        fetch_coco._stream_to(
            "http://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip", tmp_path / "val2017.zip", ("md5",)
        )
    assert list(tmp_path.iterdir()) == []


def test_every_subcommand_has_help_and_the_docstring_counts_them():
    parser = fetch_coco.build_parser()
    commands = next(action for action in parser._actions if action.dest == "what")
    helps = {choice.dest: choice.help for choice in commands._choices_actions}
    assert set(helps) == {"annotations", "val", "calibration"}
    assert all(helps.values()), helps
    assert re.search(r"^Three independent subcommands", fetch_coco.__doc__, re.M)
    assert "--write-manifest" in commands.choices["calibration"].format_help()


def test_an_interrupted_extraction_is_fetched_again(tmp_path, monkeypatch):
    import zipfile

    monkeypatch.setattr(fetch_coco, "DATA_DIR", tmp_path)
    url = "https://s3.amazonaws.com/images.cocodataset.org/zips/demo.zip"
    fetches = []

    def fake_stream(source, destination, algorithms):
        fetches.append(source)
        with zipfile.ZipFile(destination, "w") as bundle:
            bundle.writestr("demo/000000000139.jpg", b"jpeg")
        return {"md5": "m", "sha256": "s"}

    monkeypatch.setattr(fetch_coco, "_stream_to", fake_stream)
    real_extract = zipfile.ZipFile.extractall

    def interrupted(self, path=None, members=None, pwd=None):
        real_extract(self, path, members, pwd)
        raise OSError("disk full")

    monkeypatch.setattr(zipfile.ZipFile, "extractall", interrupted)
    with pytest.raises(OSError):
        fetch_coco._fetch_archive(url, "m", "s", "demo/000000000139.jpg")
    # The marker file exists, but the extraction did not finish, so it is not taken as present.
    assert (tmp_path / "demo" / "000000000139.jpg").exists() and not fetch_coco.extracted_stamp(url).exists()
    monkeypatch.setattr(zipfile.ZipFile, "extractall", real_extract)
    fetch_coco._fetch_archive(url, "m", "s", "demo/000000000139.jpg")
    assert len(fetches) == 2 and fetch_coco.extracted_stamp(url).read_text() == "1 members\n"
    fetch_coco._fetch_archive(url, "m", "s", "demo/000000000139.jpg")
    assert len(fetches) == 2


# The names on the certificate S3 serves for s3.amazonaws.com and for a bucket host such as images.cocodataset.org.
S3_CERTIFICATE_NAMES = ("s3.amazonaws.com", "*.s3.amazonaws.com")


def _certificate_covers(name: str, host: str) -> bool:
    """A certificate name covers a host when it equals it, or, as a wildcard, stands for exactly one leftmost label."""
    if name.startswith("*."):
        label, _, rest = host.partition(".")
        return bool(label) and rest == name[2:]
    return host == name


def test_every_download_host_is_covered_by_the_certificate_it_serves():
    urls = [value for name, value in vars(fetch_coco).items() if name.endswith(("_URL", "_URL_PREFIX"))]
    assert len(urls) >= 4
    hosts = {urllib.parse.urlsplit(url).hostname for url in urls}
    assert all(any(_certificate_covers(name, host) for name in S3_CERTIFICATE_NAMES) for host in hosts), hosts


@pytest.mark.parametrize(
    ("host", "covered"),
    [
        ("s3.amazonaws.com", True),
        ("bucket.s3.amazonaws.com", True),
        ("images.cocodataset.org", False),
        ("images.cocodataset.org.s3.amazonaws.com", False),
    ],
)
def test_the_certificate_check_needs_one_label_under_a_wildcard(host, covered):
    assert any(_certificate_covers(name, host) for name in S3_CERTIFICATE_NAMES) is covered


def test_a_redirect_away_from_https_is_refused_and_leaves_no_file(tmp_path, monkeypatch):
    plain = "http" + "://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip"
    monkeypatch.setattr(fetch_coco.urllib.request, "urlopen", lambda url, timeout: Response(b"bytes", plain))
    with pytest.raises(ValueError, match="redirected away from HTTPS"):
        fetch_coco._stream_to(fetch_coco.VAL_ZIP_URL, tmp_path / "val2017.zip", ("md5",))
    assert list(tmp_path.iterdir()) == []


def test_the_command_ends_a_refused_redirect_with_one_line_not_a_traceback(monkeypatch, capsys):
    def refused(*arguments):
        raise ValueError("refusing a download redirected away from HTTPS")

    monkeypatch.setattr(fetch_coco, "_fetch_archive", refused)
    assert fetch_coco.main(["val"]) == 1
    error = capsys.readouterr().err
    assert error == "fetch_coco: refusing a download redirected away from HTTPS\n"


def test_a_cut_connection_is_retried_and_then_succeeds(tmp_path, monkeypatch):
    calls = []

    def flaky(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise http.client.IncompleteRead(b"half")
        return Response(b"whole", url)

    monkeypatch.setattr(fetch_coco.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(fetch_coco.time, "sleep", lambda seconds: None)
    digests = fetch_coco._stream_to(fetch_coco.VAL_ZIP_URL, tmp_path / "val2017.zip", ("sha256",))
    assert len(calls) == 2 and digests == {"sha256": hashlib.sha256(b"whole").hexdigest()}


def test_an_archive_sha256_mismatch_keeps_the_archive_and_names_it(tmp_path, monkeypatch, capsys):
    def fake_stream(url, destination, digests):
        destination.write_bytes(b"right md5, wrong sha256")
        return {"md5": fetch_coco.ANNOTATIONS_ZIP_MD5, "sha256": "0" * 64}

    monkeypatch.setattr(fetch_coco, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(fetch_coco, "_stream_to", fake_stream)
    assert fetch_coco.main(["annotations"]) == 1
    assert "sha256 mismatch" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["../escape.jpg", "/" + "000000000001.jpg", "sub/000000000001.jpg", "..", ""])
def test_an_unsafe_image_file_name_is_refused(tmp_path, monkeypatch, name):
    monkeypatch.setattr(fetch_coco, "DATA_DIR", tmp_path)
    (tmp_path / "list.json").write_text(
        json.dumps({"images": [{"file_name": "000000000001.jpg"}, {"file_name": name}]})
    )
    with pytest.raises(RuntimeError, match="unsafe image file names"):
        fetch_coco._image_names("list.json")
