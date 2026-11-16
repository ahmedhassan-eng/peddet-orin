# SPDX-License-Identifier: Apache-2.0
"""Write a hashed lock beside a pin file: the same pins, each with the sha256 of every file of that release.

Usage: python scripts/lock_hashes.py <pin file> <lock file> [--extra-index-url URL]

The pin file stays the readable source; the lock carries its lines unchanged, the index options included, and adds a
``--hash`` for every file the release has on PyPI and on each index the pin file names with ``--extra-index-url``,
so ``pip install --require-hashes -r <lock>`` accepts the file pip picks for its platform and refuses any other.
Nothing is resolved: a pin's version is taken as it is. It needs the network; the tests check a lock offline against
its pin file (``tests/test_locks.py``).
"""

from __future__ import annotations

import argparse
import html.parser
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

PIN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^;\s]+)\s*(?:;\s*(?P<marker>.+))?$")
INDEX = re.compile(r"^--extra-index-url\s+(?P<url>\S+)$")
PYPI_JSON = "https://pypi.org/pypi/{name}/{version}/json"
TIMEOUT_S = 60
ATTEMPTS = 3


def normalised(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _get(url: str) -> bytes | None:
    """The body at a URL, or None when it is not there (404, or 403 from an index served from a bucket, as the
    PyTorch index answers for a project it does not hold); other failures are retried, then raised."""
    for attempt in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as response:  # noqa: S310 (fixed https URLs)
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code in (403, 404):
                return None
            if attempt == ATTEMPTS - 1:
                raise
        except OSError:
            if attempt == ATTEMPTS - 1:
                raise
    return None


class _Links(html.parser.HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.hrefs += [value for key, value in attrs if key == "href" and value]


def pypi_hashes(name: str, version: str) -> set[str]:
    body = _get(PYPI_JSON.format(name=name, version=version))
    if body is None:
        return set()
    return {entry["digests"]["sha256"] for entry in json.loads(body)["urls"]}


def index_hashes(index: str, name: str, version: str) -> set[str]:
    """The sha256 of each file of one release on a PEP 503 simple index, read from the links' fragments."""
    body = _get(f"{index.rstrip('/')}/{normalised(name)}/")
    if body is None:
        return set()
    parser = _Links()
    parser.feed(body.decode())
    found = set()
    for href in parser.hrefs:
        path, _, fragment = href.partition("#sha256=")
        filename = urllib.parse.unquote(path.rsplit("/", 1)[-1])
        stem = re.sub(r"(\.tar\.gz|\.zip|\.whl)$", "", filename)
        parts = stem.split("-")
        if fragment and len(parts) >= 2 and normalised(parts[0]) == normalised(name) and parts[1] == version:
            found.add(fragment)
    return found


def lock_lines(text: str, extra_indexes: list[str] = ()) -> list[str]:
    """The lock's lines for a pin file's text; raise ValueError when a pin has no file to hash. An extra index given
    here, as the board's CUDA wheels need, is written into the lock as well."""
    indexes = [match.group("url") for line in text.splitlines() if (match := INDEX.match(line.strip()))]
    lines = [f"--extra-index-url {url}" for url in extra_indexes if url not in indexes]
    indexes += [url for url in extra_indexes if url not in indexes]
    for line in text.splitlines():
        match = PIN.match(line.strip())
        if not match:
            lines.append(line)
            continue
        name, version = match.group("name"), match.group("version")
        hashes = pypi_hashes(name, version)
        for index in indexes:
            hashes |= index_hashes(index, name, version)
        if not hashes:
            raise ValueError(f"no file found to hash for {name}=={version}")
        lines.append(line.strip() + " \\")
        ordered = sorted(hashes)
        lines += [
            f"    --hash=sha256:{digest}" + (" \\" if index < len(ordered) - 1 else "")
            for index, digest in enumerate(ordered)
        ]
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pins", type=Path, help="pin file to lock, e.g. requirements/ci-cpu.txt")
    parser.add_argument("lock", type=Path, help="lock file to write, e.g. requirements/ci-cpu.lock")
    parser.add_argument("--extra-index-url", action="append", default=[], help="another index to hash files from")
    args = parser.parse_args(argv)
    try:
        lines = lock_lines(args.pins.read_text(), args.extra_index_url)
    except (ValueError, OSError) as error:
        print(f"lock_hashes: {error}", file=sys.stderr)
        return 1
    header = f"# Generated by scripts/lock_hashes.py from {args.pins.as_posix()}; do not edit by hand.\n"
    args.lock.write_text(header + "\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
