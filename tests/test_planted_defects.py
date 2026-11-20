# SPDX-License-Identifier: Apache-2.0
"""Every entry of the planted-defect registry still applies, and names a test that pytest collects."""

import subprocess
from pathlib import Path

import pytest

from planted_defects import DEFECTS
from strict_skips import checkout_skip_reason
from test_doc_test_refs import collected

REPO_ROOT = Path(__file__).resolve().parents[1]
# The tests marked needs_checkout read git ls-files, which a release tarball (no .git) cannot answer.
CHECKOUT_SKIP = checkout_skip_reason(REPO_ROOT)
pytestmark = pytest.mark.skipif(CHECKOUT_SKIP is not None, reason=CHECKOUT_SKIP or "")
KEYS = {"id", "file", "old", "new", "catches"}


def registry_problems(defects: list[dict], tracked: set[str], nodes: frozenset[tuple[str, str]]) -> list[str]:
    problems = []
    ids = [defect.get("id") for defect in defects]
    problems += [f"id used twice: {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for defect in defects:
        name = defect.get("id")
        if not KEYS <= set(defect) or set(defect) - KEYS - {"needs"}:
            problems.append(f"{name}: keys must be {sorted(KEYS)} and optionally needs")
            continue
        if defect["file"] not in tracked:
            problems.append(f"{name}: file not tracked: {defect['file']}")
            continue
        count = (REPO_ROOT / defect["file"]).read_text().count(defect["old"])
        if count != 1:
            problems.append(f"{name}: old text occurs {count} times")
        if defect["old"] == defect["new"]:
            problems.append(f"{name}: plants nothing")
        file, _, test = defect["catches"].partition("::")
        if (file.removeprefix("tests/"), test) not in nodes:
            problems.append(f"{name}: catching test not collected: {defect['catches']}")
    return problems


def tracked() -> set[str]:
    done = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return set(done.stdout.splitlines())


def test_every_planted_defect_applies_and_names_a_collected_test():
    assert len(DEFECTS) >= 10
    assert registry_problems(DEFECTS, tracked(), collected()) == []


def test_a_stale_registry_entry_is_detected():
    stale = dict(DEFECTS[0], id="stale", old="text that is nowhere", catches="tests/test_gone.py::test_gone")
    assert registry_problems([stale], tracked(), collected()) == [
        "stale: old text occurs 0 times",
        "stale: catching test not collected: tests/test_gone.py::test_gone",
    ]


def test_a_duplicate_id_is_detected():
    assert registry_problems([DEFECTS[0], DEFECTS[0]], tracked(), collected()) == [f"id used twice: {DEFECTS[0]['id']}"]


def test_the_planted_check_passes_only_when_every_entry_ran_and_was_caught():
    import check_planted_defects

    assert check_planted_defects.verdict(0, 0) == "PLANTED DONE PASS"
    assert check_planted_defects.verdict(0, 1) == "PLANTED DONE INCOMPLETE 1 not run"
    assert check_planted_defects.verdict(1, 1) == "PLANTED DONE FAIL"
    assert check_planted_defects.TESTS_FAILED == 1
