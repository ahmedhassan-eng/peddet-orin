# SPDX-License-Identifier: Apache-2.0
"""Outside a full clone, the repository-rule tests skip with their fixed reason instead of failing.

A release tarball (``git archive HEAD``) has no .git, and ``git clone --depth 1`` has no history.
Each test builds such a tree from this checkout's tracked files in tmp_path and runs the
repository-rule tests there in a fresh pytest process under PDORIN_STRICT_SKIPS=1.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from strict_skips import NO_GIT_REASON, SHALLOW_REASON, checkout_skip_reason

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SKIP = checkout_skip_reason(REPO_ROOT)
pytestmark = pytest.mark.skipif(SOURCE_SKIP is not None, reason=SOURCE_SKIP or "")
# The noreply address the project commits with; nothing here is pushed anywhere.
IDENTITY = (
    "-c",
    "user.name=peddet-orin tests",
    "-c",
    "user.email=1659721+ahmedhassan-eng@users.noreply.github.com",
    "-c",
    "commit.gpgsign=false",
    "-c",
    "init.defaultBranch=main",
)
# test_a_commit_id_that_is_not_in_the_history_is_detected needs no checkout, so it passes in every tree.
HISTORY_TESTS = ["tests/test_commit_ids.py"]
LISTING_TESTS = [
    *HISTORY_TESTS,
    "tests/test_readme_tree.py",
    "tests/test_planted_defects.py",
    "tests/test_doc_test_refs.py::test_every_cited_test_is_collected",
]


def git(cwd: Path, *args: str) -> str:
    done = subprocess.run(["git", *IDENTITY, *args], cwd=cwd, capture_output=True, text=True, check=True)
    return done.stdout


def copy_tracked(destination: Path) -> Path:
    """Copy this checkout's tracked files, as they are in the working tree, without .git."""
    for name in git(REPO_ROOT, "ls-files").splitlines():
        source = REPO_ROOT / name
        if source.is_file():
            (destination / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination / name)
    return destination


def shallow_clone(tmp_path: Path) -> Path:
    """Commit the tracked files twice in a scratch repository and clone it with --depth 1."""
    origin = copy_tracked(tmp_path / "origin")
    git(origin, "init", "-q")
    git(origin, "add", "-A")
    git(origin, "commit", "-q", "-m", "first")
    git(origin, "commit", "-q", "--allow-empty", "-m", "second")
    clone = tmp_path / "shallow"
    git(tmp_path, "clone", "-q", "--depth", "1", origin.as_uri(), str(clone))
    assert git(clone, "rev-parse", "--is-shallow-repository").strip() == "true"
    return clone


def run_subset(tree: Path, tests: list[str], ci: bool) -> subprocess.CompletedProcess:
    # Without PYTHONPATH the tree's own src/ and scripts/ are imported, from its pyproject.toml.
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "CI")}
    env["PDORIN_STRICT_SKIPS"] = "1"
    if ci:
        env["CI"] = "true"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", *tests],
        cwd=tree,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def skip_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("SKIPPED")]


def test_a_release_tarball_tree_skips_the_repository_rules(tmp_path):
    tree = copy_tracked(tmp_path / "tarball")
    assert not (tree / ".git").exists()
    done = run_subset(tree, LISTING_TESTS, ci=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert " failed" not in done.stdout and " error" not in done.stdout, done.stdout
    skips = skip_lines(done.stdout)
    # 2 history tests, 5 tree tests, 3 registry tests and 1 cited-test check, all for the one reason.
    assert "11 skipped" in done.stdout and "1 passed" in done.stdout, done.stdout
    assert skips and all(line.endswith(NO_GIT_REASON) for line in skips), skips


def test_a_shallow_clone_skips_the_history_tests(tmp_path):
    clone = shallow_clone(tmp_path)
    done = run_subset(clone, HISTORY_TESTS, ci=False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "2 skipped" in done.stdout and "1 passed" in done.stdout, done.stdout
    skips = skip_lines(done.stdout)
    assert skips and all(line.endswith(SHALLOW_REASON) for line in skips), skips


def test_a_shallow_clone_in_ci_still_fails_a_strict_run(tmp_path):
    clone = shallow_clone(tmp_path)
    done = run_subset(clone, HISTORY_TESTS, ci=True)
    assert done.returncode != 0
    assert "unexpected skips: test_every_cited_commit_is_in_the_history, test_history_is_complete" in done.stdout
