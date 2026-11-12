# SPDX-License-Identifier: Apache-2.0
"""Every PRD requirement names at least one test that CI collects and runs without skipping."""
import re
from pathlib import Path

from strict_skips import EXPECTED_SKIPS
from test_doc_test_refs import collected

PRD = Path(__file__).resolve().parents[1] / "PRD.md"
ROW = re.compile(r"^\| (?P<id>R-\d+) \|(?P<cells>.*)\|$", re.M)
NODE = re.compile(r"`tests/(?P<file>test_\w+\.py)::(?P<name>test_\w+)`")
CI_COLUMN = 3  # Requirement, Acceptance check, Covered in CI by, Orin-run


def coverage(prd: str) -> dict[str, list[tuple[str, str]]]:
    """Return ``{R-id: [(file, test)]}`` from the PRD's "Covered in CI by" column."""
    rows = {}
    for match in ROW.finditer(prd):
        cells = [cell.strip() for cell in match.group("cells").split(" | ")]
        rows[match.group("id")] = [(m.group("file"), m.group("name")) for m in NODE.finditer(cells[CI_COLUMN - 1])]
    return rows


def coverage_problems(rows: dict[str, list[tuple[str, str]]], nodes: frozenset[tuple[str, str]]) -> list[str]:
    problems = []
    numbers = [int(rid[2:]) for rid in rows]
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append(f"R-ids are not R-1 to R-{len(numbers)} in order: {list(rows)}")
    for rid, tests in rows.items():
        if not tests:
            problems.append(f"{rid}: no CI test")
        for file, name in tests:
            if (file, name) not in nodes:
                problems.append(f"{rid}: not collected: tests/{file}::{name}")
            elif name in EXPECTED_SKIPS:
                problems.append(f"{rid}: skips in CI: tests/{file}::{name}")
    return problems


def test_every_requirement_is_covered_by_a_ci_test():
    rows = coverage(PRD.read_text())
    assert len(rows) == 14
    assert coverage_problems(rows, collected()) == []


def test_a_requirement_with_no_ci_test_is_detected():
    rows = dict(coverage(PRD.read_text()), **{"R-3": []})
    assert coverage_problems(rows, collected()) == ["R-3: no CI test"]


def test_a_gpu_only_test_does_not_count_as_ci_coverage():
    rows = dict(coverage(PRD.read_text()), **{"R-4": [("test_gpu.py", "test_tensorrt_fp16_engine_matches_cpu")]})
    assert coverage_problems(rows, collected()) == [
        "R-4: skips in CI: tests/test_gpu.py::test_tensorrt_fp16_engine_matches_cpu"
    ]
