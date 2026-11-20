# SPDX-License-Identifier: Apache-2.0
"""Fetch the COCO 2017 files this project needs, from COCO's S3 bucket, path-style, over HTTPS only.

Three independent subcommands exist so that each artefact can be fetched when it
is first needed:

* ``annotations`` -- the trainval annotation archive (read by evaluation and the calibration fetch).
* ``val``         -- the val2017 image archive (needed by every accuracy run).
* ``calibration`` -- a fixed, seeded sample of train2017 images, fetched one file
  at a time and verified against the sha256 hashes in the committed manifest
  ``manifests/calibration_train2017.sha256``. A missing manifest is an error:
  only ``calibration --write-manifest`` records a new one from what it fetches.

Every download uses HTTPS. The archives are also verified against the MD5 digests
pinned below. A file that fails its check is never deleted from ``data/``: the
fetch stops with an error naming it. Nothing fetched here is committed: everything
lands under ``data/``, which is git-ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import random
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
MANIFEST_PATH = REPO_ROOT / "manifests" / "calibration_train2017.sha256"

# COCO's files are in the S3 bucket images.cocodataset.org. That host name serves S3's certificate, whose names
# (s3.amazonaws.com, *.s3.amazonaws.com) a dotted bucket name does not match, so HTTPS to it fails verification. The
# path-style form names the bucket in the path and the host s3.amazonaws.com, which the certificate covers.
COCO_BASE_URL = "https://s3.amazonaws.com/images.cocodataset.org/"
ANNOTATIONS_ZIP_URL = COCO_BASE_URL + "annotations/annotations_trainval2017.zip"
ANNOTATIONS_ZIP_MD5 = "f4bbac642086de4f52a3fdda2de5fa2c"
# The sha256 of each archive, read from one verified fetch on 2026-11-19 whose MD5 matched the pin above; MD5 alone
# is not collision-resistant.
ANNOTATIONS_ZIP_SHA256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
VAL_ZIP_URL = COCO_BASE_URL + "zips/val2017.zip"
VAL_ZIP_MD5 = "442b8da7639aecaf257c1dceb8ba8c80"
VAL_ZIP_SHA256 = "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
TRAIN_IMAGE_URL_PREFIX = COCO_BASE_URL + "train2017/"

# The captions file lists every train2017 image and is far smaller to parse than
# the instances file, which matters on an 8 GB board.
TRAIN_IMAGE_LIST = "annotations/captions_train2017.json"
VAL_IMAGE_LIST = "annotations/captions_val2017.json"

CALIBRATION_SEED = 2017
CALIBRATION_COUNT = 500
CALIBRATION_DIR_NAME = "calibration_train2017"

CHUNK_BYTES = 1 << 20
HTTPS_PREFIX = "https://"
HTTP_TIMEOUT_S = 60
RETRIES = 4
RETRY_SLEEP_S = 5


def _stream_to(url: str, destination: Path, digests: tuple[str, ...]) -> dict[str, str]:
    """Download ``url`` to ``destination`` atomically, returning the hex digests.

    The file is written beside its destination and renamed only once complete, so
    an interrupted run can never leave a truncated file that looks finished. If the
    destination already exists, for example an archive a failed MD5 check kept, it
    is left as it is until a new download completes, and then replaced by it.
    """
    if not url.startswith(HTTPS_PREFIX):
        raise ValueError(f"refusing a download that is not HTTPS: {url}")
    partial = destination.with_name(destination.name + ".partial")
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            hashers = {name: hashlib.new(name) for name in digests}
            with urllib.request.urlopen(url, timeout=HTTP_TIMEOUT_S) as response:
                # A redirect is followed by urllib, so the address the bytes came from is checked, not only the first.
                final = response.geturl()
                if not final.startswith(HTTPS_PREFIX):
                    raise ValueError(f"refusing a download redirected away from HTTPS: {url} -> {final}")
                with partial.open("wb") as sink:
                    while chunk := response.read(CHUNK_BYTES):
                        sink.write(chunk)
                        for hasher in hashers.values():
                            hasher.update(chunk)
            partial.replace(destination)
            return {name: hasher.hexdigest() for name, hasher in hashers.items()}
        # URLError and timeouts derive from OSError; a connection cut mid-body raises IncompleteRead, which does not.
        except (OSError, http.client.IncompleteRead) as error:
            last_error = error
            time.sleep(RETRY_SLEEP_S * attempt)
    # The last attempt's truncated file is removed, so a run that gives up leaves nothing half-written behind.
    partial.unlink(missing_ok=True)
    raise RuntimeError(f"giving up on {url}: {last_error}")


def extracted_stamp(url: str) -> Path:
    """The file written once an archive is fully extracted; an extraction that stops early leaves none."""
    return DATA_DIR / f".{url.rsplit('/', 1)[1]}.extracted"


def _fetch_archive(url: str, expected_md5: str, expected_sha256: str, marker: str) -> None:
    """Fetch and extract one archive unless a completed extraction of it is recorded and its marker file exists."""
    stamp = extracted_stamp(url)
    if stamp.exists() and (DATA_DIR / marker).exists():
        print(f"present: {marker}")
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / url.rsplit("/", 1)[1]
    actual = _stream_to(url, archive, ("md5", "sha256"))
    if actual["md5"] != expected_md5:
        raise RuntimeError(f"md5 mismatch for {archive}: got {actual['md5']}, expected {expected_md5}; left in place")
    if actual["sha256"] != expected_sha256:
        raise RuntimeError(
            f"sha256 mismatch for {archive}: got {actual['sha256']}, expected {expected_sha256}; left in place"
        )
    # ZipFile.extractall strips absolute prefixes and ".." members, so extraction
    # stays inside DATA_DIR even for a malformed archive.
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(DATA_DIR)
        stamp.write_text(f"{len(bundle.namelist())} members\n")
    archive.unlink()
    print(f"fetched: {marker}")


def _image_names(relative_json: str) -> list[str]:
    """Return the sorted image file names listed by a COCO annotation file."""
    with (DATA_DIR / relative_json).open() as handle:
        names = sorted(image["file_name"] for image in json.load(handle)["images"])
    unsafe = [name for name in names if not safe_name(name)]
    if unsafe:
        raise RuntimeError(f"{relative_json} lists {len(unsafe)} unsafe image file names, the first {unsafe[0]!r}")
    return names


def safe_name(name: str) -> bool:
    """Whether an image file name stays inside the directory it is joined to: no absolute path, no separator, no
    "..", and not empty."""
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        return False
    return not Path(name).is_absolute()


def select_calibration_names(train_names: list[str], seed: int, count: int) -> list[str]:
    """Choose the calibration sample: a seeded draw from the sorted name list."""
    if len(train_names) < count:
        raise RuntimeError(f"train2017 lists {len(train_names)} images, fewer than the {count} to calibrate on")
    return sorted(random.Random(seed).sample(sorted(train_names), count))


def _read_manifest() -> dict[str, str]:
    """Parse the committed manifest into a ``{file name: sha256}`` mapping."""
    pairs = (line.split() for line in MANIFEST_PATH.read_text().splitlines() if line.strip())
    return {name: digest for digest, name in pairs}


def fetch_calibration(write_manifest: bool = False) -> None:
    """Fetch the calibration images one by one and verify them against the manifest.

    Without the manifest this stops before fetching anything, unless ``write_manifest``
    is set: then the hashes of what is fetched become the new manifest. An existing
    manifest is never rewritten. A file whose hash differs is left where it is.
    """
    manifest_exists = MANIFEST_PATH.exists()
    if not manifest_exists and not write_manifest:
        raise RuntimeError(f"no manifest at {MANIFEST_PATH.name}: restore it, or pass --write-manifest to record one")
    known = _read_manifest() if manifest_exists else {}
    if manifest_exists and len(known) != CALIBRATION_COUNT:
        raise RuntimeError(f"manifest lists {len(known)} images, expected {CALIBRATION_COUNT}")
    train_names = _image_names(TRAIN_IMAGE_LIST)
    val_names = set(_image_names(VAL_IMAGE_LIST))
    chosen = select_calibration_names(train_names, CALIBRATION_SEED, CALIBRATION_COUNT)
    overlap = val_names.intersection(chosen)
    if overlap:
        raise RuntimeError(f"calibration set overlaps val2017 in {len(overlap)} files")
    if manifest_exists and sorted(known) != chosen:
        raise RuntimeError("manifest does not list the seeded calibration selection")

    target = DATA_DIR / CALIBRATION_DIR_NAME
    target.mkdir(parents=True, exist_ok=True)
    recorded: dict[str, str] = {}
    for index, name in enumerate(chosen, start=1):
        path = target / name
        if path.exists():
            recorded[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        else:
            recorded[name] = _stream_to(TRAIN_IMAGE_URL_PREFIX + name, path, ("sha256",))["sha256"]
        if name in known and known[name] != recorded[name]:
            raise RuntimeError(f"sha256 mismatch for {path}: it does not match {MANIFEST_PATH.name}; left in place")
        if index % 100 == 0:
            print(f"calibration: {index}/{len(chosen)}", flush=True)

    if not manifest_exists:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text("".join(f"{recorded[name]}  {name}\n" for name in chosen))
        print(f"manifest written: {MANIFEST_PATH.name} ({len(chosen)} images)")
    print(f"calibration images verified: {len(chosen)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="what", required=True, metavar="{annotations,val,calibration}")
    commands.add_parser(
        "annotations", help="the trainval2017 annotation archive, which evaluation and the calibration fetch need"
    )
    commands.add_parser("val", help="the val2017 image archive, which every accuracy run needs")
    calibration = commands.add_parser(
        "calibration", help=f"the {CALIBRATION_COUNT} seeded train2017 calibration images, verified by sha256"
    )
    calibration.add_argument(
        "--write-manifest",
        action="store_true",
        help="when the manifest is missing, record the fetched images' hashes as a new one",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.what == "annotations":
            _fetch_archive(
                ANNOTATIONS_ZIP_URL, ANNOTATIONS_ZIP_MD5, ANNOTATIONS_ZIP_SHA256, "annotations/instances_val2017.json"
            )
        elif args.what == "val":
            _fetch_archive(VAL_ZIP_URL, VAL_ZIP_MD5, VAL_ZIP_SHA256, "val2017/000000000139.jpg")
        else:
            fetch_calibration(write_manifest=args.write_manifest)
    except (RuntimeError, ValueError) as error:
        # ValueError: a download redirected away from HTTPS (_stream_to), refused before any file is written.
        print(f"fetch_coco: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
