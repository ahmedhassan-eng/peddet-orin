# SPDX-License-Identifier: Apache-2.0
"""Every commit id that a tracked file cites is a commit in this repository's history.

A cited id that is not in the history points at work nobody can check out, such as an unpushed or rewritten
commit. Action SHAs in the workflow name commits of other repositories: test_ci_contract.py checks each is a full
SHA with its release tag beside it, which this test cannot resolve offline; by the author's record, each was
matched against its tag on GitHub when it was pinned.
"""

import re
import subprocess
from pathlib import Path

import pytest

from strict_skips import checkout_skip_reason

REPO_ROOT = Path(__file__).resolve().parents[1]
# A release tarball has no .git and a shallow clone lacks the history; both skip with a fixed reason.
HISTORY_SKIP = checkout_skip_reason(REPO_ROOT, history=True)
needs_history = pytest.mark.skipif(HISTORY_SKIP is not None, reason=HISTORY_SKIP or "")
# A run log's commit field, or a 7- to 40-digit hex word after "commit" or "at" in prose.
LOG_COMMIT = re.compile(r'"commit":\s*"(?P<id>[0-9a-f]{7,40})"')
PROSE_COMMIT = re.compile(r"\b(?:commit|commits|at)\s+`?(?P<id>[0-9a-f]{7,40})\b`?")
# The workflow pins other repositories' commits, the manifest holds sha256 hashes, and these two test files plant
# ids that are not in the history on purpose.
SKIP = {
    ".github/workflows/ci.yml",
    "manifests/calibration_train2017.sha256",
    "tests/test_commit_ids.py",
    "tests/planted_defects.py",
}


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False)


def cited_ids(files: dict[str, str]) -> dict[str, set[str]]:
    """Return ``{id: files citing it}`` for every commit-like id in the given texts."""
    found: dict[str, set[str]] = {}
    for name, text in files.items():
        for pattern in (LOG_COMMIT, PROSE_COMMIT):
            for match in pattern.finditer(text):
                if re.search(r"[a-f]", match.group("id")) and re.search(r"\d", match.group("id")):
                    found.setdefault(match.group("id"), set()).add(name)
    return found


def in_history(commit_id: str) -> bool:
    full = git("rev-parse", "--verify", "--quiet", f"{commit_id}^{{commit}}")
    return full.returncode == 0 and git("merge-base", "--is-ancestor", full.stdout.strip(), "HEAD").returncode == 0


def tracked_texts() -> dict[str, str]:
    names = git("ls-files").stdout.splitlines()
    texts = {}
    for name in names:
        if name in SKIP:
            continue
        try:
            texts[name] = (REPO_ROOT / name).read_text()
        except UnicodeDecodeError:
            continue
    return texts


def not_in_history(found: dict[str, set[str]]) -> list[str]:
    return sorted(
        f"{commit_id} in {', '.join(sorted(files))}" for commit_id, files in found.items() if not in_history(commit_id)
    )


@needs_history
def test_history_is_complete():
    # A shallow clone lacks the commits the logs cite; CI checks out with fetch-depth 0.
    assert git("rev-parse", "--is-shallow-repository").stdout.strip() == "false"


@needs_history
def test_every_cited_commit_is_in_the_history():
    found = cited_ids(tracked_texts())
    assert any(name.startswith("runs/") for files in found.values() for name in files), "no run log cites a commit"
    assert not_in_history(found) == []


def test_a_commit_id_that_is_not_in_the_history_is_detected():
    planted = {
        "runs/planted.json": '{"commit": "0123456789abcdef0123456789abcdef01234567"}',
        "NOTES.md": "Measured at commit 1a2b3c4d.",
    }
    assert not_in_history(cited_ids(planted)) == [
        "0123456789abcdef0123456789abcdef01234567 in runs/planted.json",
        "1a2b3c4d in NOTES.md",
    ]
