"""Every test file and test name that a tracked document cites exists and is collected by pytest."""
import os
import re
import subprocess
import sys
from functools import cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CITED_FILE = re.compile(r"\btests/(?P<file>test_\w+\.py)(?:::(?P<name>test_\w+))?")
CITED_NAME = re.compile(r"`(?P<name>test_\w+)`")


def tracked_documents() -> dict[str, str]:
    done = subprocess.run(["git", "ls-files", "*.md"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return {name: (REPO_ROOT / name).read_text() for name in done.stdout.splitlines()}


def citations(documents: dict[str, str]) -> set[tuple[str, str | None]]:
    """Return ``(file or None, name or None)`` pairs: a cited file, a file with a test, or a bare test name."""
    found = set()
    for text in documents.values():
        for match in CITED_FILE.finditer(text):
            found.add((match.group("file"), match.group("name")))
        for match in CITED_NAME.finditer(text):
            found.add((None, match.group("name")))
    return found


@cache
def collected() -> frozenset[tuple[str, str]]:
    """Return ``(file, test name)`` for every test pytest collects, parameters stripped."""
    env = {key: value for key, value in os.environ.items() if key != "PDORIN_STRICT_SKIPS"}
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=REPO_ROOT, capture_output=True, text=True, env=env, check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    nodes = [line for line in done.stdout.splitlines() if line.startswith("tests/") and "::" in line]
    return frozenset((node.split("::")[0][len("tests/"):], node.split("::")[-1].split("[")[0]) for node in nodes)


def missing(cited: set[tuple[str | None, str | None]], nodes: frozenset[tuple[str, str]]) -> list[str]:
    files = {file for file, _ in nodes}
    names = {name for _, name in nodes}
    problems = []
    for file, name in sorted(cited, key=str):
        if file and file not in files:
            problems.append(f"no collected test file: tests/{file}")
        elif file and name and (file, name) not in nodes:
            problems.append(f"not collected: tests/{file}::{name}")
        elif not file and name not in names:
            problems.append(f"no collected test named {name}")
    return sorted(problems)


def test_every_cited_test_is_collected():
    cited = citations(tracked_documents())
    assert cited, "no document cites a test"
    assert missing(cited, collected()) == []


def test_a_cited_test_that_does_not_exist_is_detected():
    planted = {"PLANTED.md": "See `tests/test_run_logs.py::test_no_such_check` and `test_also_missing`."}
    assert missing(citations(planted), collected()) == [
        "no collected test named test_also_missing",
        "not collected: tests/test_run_logs.py::test_no_such_check",
    ]


def test_a_cited_file_that_does_not_exist_is_detected():
    assert missing(citations({"PLANTED.md": "`tests/test_gone.py`"}), collected()) == [
        "no collected test file: tests/test_gone.py"
    ]
