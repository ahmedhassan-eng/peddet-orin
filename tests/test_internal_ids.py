# SPDX-License-Identifier: Apache-2.0
"""No tracked file cites a message id from the project's private working notes.

Those ids name messages nobody outside can read, so a file citing one points at nothing. The patterns are a role
letter, a hyphen and four digits, with an optional second role letter and hyphen between the two; and the ids of
review findings, a letter and one or two digits in parentheses at the end of a subject, or a reviewer
letter, a hyphen, a severity letter and a number. Commit messages are scanned too. The planted ids below are built
from parts, so this file does not match itself.
"""

import re
import subprocess
from pathlib import Path

import pytest

from strict_skips import checkout_skip_reason

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKOUT_SKIP = checkout_skip_reason(REPO_ROOT)
needs_checkout = pytest.mark.skipif(CHECKOUT_SKIP is not None, reason=CHECKOUT_SKIP or "")
ROLE = "[A-Z]"
FINDING = "[FXRm][0-9]{1,2}"
INTERNAL_ID = re.compile(
    rf"\b{ROLE}-(?:{ROLE}-)?[0-9]{{4}}\b"
    rf"|\({FINDING}(?:,\s*{FINDING})*\)"
    r"|\b[ABCD]-[BMmn][0-9]{1,2}\b"
    r"|\(items? [0-9]"
)
HISTORY_SKIP = checkout_skip_reason(REPO_ROOT, history=True)
needs_history = pytest.mark.skipif(HISTORY_SKIP is not None, reason=HISTORY_SKIP or "")


def internal_ids(files: dict[str, str]) -> list[str]:
    """Each ``file:line: id`` that matches the pattern, in file and line order."""
    return [
        f"{name}:{number}: {match.group()}"
        for name, text in sorted(files.items())
        for number, line in enumerate(text.splitlines(), start=1)
        for match in INTERNAL_ID.finditer(line)
    ]


def tracked_names() -> list[str]:
    done = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return done.stdout.splitlines()


def scanned(files: dict[str, bytes]) -> dict[str, str]:
    """The texts that are scanned: every file that reads as text. No file is exempt."""
    texts = {}
    for name, data in files.items():
        try:
            texts[name] = data.decode()
        except UnicodeDecodeError:
            continue
    return texts


def tracked_files() -> dict[str, bytes]:
    return {name: (REPO_ROOT / name).read_bytes() for name in tracked_names() if (REPO_ROOT / name).is_file()}


@needs_checkout
def test_no_tracked_file_cites_an_internal_id():
    assert internal_ids(scanned(tracked_files())) == []


@needs_history
def test_no_commit_message_cites_an_internal_id():
    done = subprocess.run(
        ["git", "log", "--format=%H%n%B%x00"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    messages = {}
    for entry in done.stdout.split("\0"):
        sha, _, body = entry.strip("\n").partition("\n")
        if sha:
            messages[sha[:12]] = body
    assert len(messages) > 30
    assert internal_ids(messages) == []


PLANTED = "# see " + "-".join(["Q", "R", "0042"]) + "\n"


def test_an_id_planted_in_any_tracked_path_is_found():
    planted = "-".join(["Q", "R", "0042"])
    paths = ["scripts/as-run/sysmon.sh", "scripts/planted.sh"]
    files = {**{path: PLANTED.encode() for path in paths}, "README.md": b"plain\n"}
    assert internal_ids(scanned(files)) == [f"{path}:1: {planted}" for path in paths]


@pytest.mark.parametrize(
    "line",
    [
        "# see " + "-".join(["Q", "R", "0042"]),
        "fixed in " + "-".join(["Q", "0042"]) + ".",
        "(" + "-".join(["Q", "R", "0042"]) + ".3)",
        "Give the build log one form per case (" + "m" + "8)",
        "Compare with the judged median (" + "X" + "1, " + "F" + "20)",
        "a finding " + "-".join(["A", "M1"]) + " of the review",
        "the review's (" + "items 2" + " and 3)",
    ],
)
def test_a_planted_id_is_found(line):
    assert internal_ids({"planted.txt": f"first\n{line}\n"}) == [f"planted.txt:2: {INTERNAL_ID.search(line).group()}"]


@pytest.mark.parametrize(
    "line",
    ["PRD R-11", "R-1 to R-14", "ISO-8601", "COCO-2017", "UTF-8", "sha256 ab-1234x", "Jetson Linux R39", "F1 (FP16)"],
)
def test_ordinary_text_is_not_an_id(line):
    assert internal_ids({"plain.txt": line}) == []
