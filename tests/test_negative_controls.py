# SPDX-License-Identifier: Apache-2.0
"""The README's negative-control sentence states the true coverage of the planted-defect registry.

A test file counts as covered when a registry entry names one of its tests, and as carrying an in-file negative
test when one of its test names says that a planted or bad input is detected, refused, rejected, blocked, found or
fails. QUEUE.md names every test file that has neither.
"""
import re
from pathlib import Path

import planted_defects

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
QUEUE = REPO_ROOT / "QUEUE.md"
NEGATIVE_NAME = re.compile(
    r"_(detected|refused|rejected|rejects|refuses|fails|fail|blocked|found|raises|is_out|not_exempt|reported)(_|$)"
)
SENTENCE = re.compile(
    r"`tests/planted_defects\.py` registers (?P<n>\d+) planted defects.*?the release check requires all "
    r"(?P<caught>\d+) to be caught\. They cover (?P<x>\d+) of the (?P<y>\d+) test files, (?P<z>\d+) more carry "
    r"in-file negative tests"
)


def listed_test_files() -> list[Path]:
    return sorted((REPO_ROOT / "tests").glob("test_*.py"))


def coverage(defects: list[dict], files: list[Path]) -> tuple[int, list[str], list[str], list[str]]:
    """The registry size, and the test files that are covered, that carry in-file negative tests, and the rest."""
    named = {defect["catches"].split("::", 1)[0] for defect in defects}
    covered, negative, rest = [], [], []
    for path in files:
        name = f"tests/{path.name}"
        tests = re.findall(r"^def (test_\w+)", path.read_text(), re.M)
        if name in named:
            covered.append(name)
        elif any(NEGATIVE_NAME.search(test) for test in tests):
            negative.append(name)
        else:
            rest.append(name)
    return len(defects), covered, negative, rest


def sentence_problems(readme: str, defects: list[dict], files: list[Path]) -> list[str]:
    """Each number of the README's negative-control sentence that differs from the registry and the test files,
    as "<name>: stated <s>, true <t>"; a missing sentence is one problem."""
    match = SENTENCE.search(" ".join(readme.split()))
    if not match:
        return ["README has no negative-control sentence"]
    count, covered, negative, _ = coverage(defects, files)
    true = {"n": count, "caught": count, "x": len(covered), "y": len(files), "z": len(negative)}
    return [f"{name}: stated {match[name]}, true {value}" for name, value in true.items() if int(match[name]) != value]


def test_the_readme_states_the_registrys_true_coverage():
    assert sentence_problems(README.read_text(), planted_defects.DEFECTS, listed_test_files()) == []


def test_queue_names_every_test_file_without_a_negative_control():
    *_, rest = coverage(planted_defects.DEFECTS, listed_test_files())
    text = QUEUE.read_text()
    assert rest and all(f"`{name}`" in text for name in rest), rest


def test_a_readme_sentence_with_one_number_changed_is_refused():
    readme = README.read_text()
    count, covered, *_ = coverage(planted_defects.DEFECTS, listed_test_files())
    right = f"They cover {len(covered)} of the {len(listed_test_files())} test files"
    wrong = f"They cover {len(covered) + 1} of the {len(listed_test_files())} test files"
    assert right in readme
    assert sentence_problems(readme.replace(right, wrong), planted_defects.DEFECTS, listed_test_files()) == [
        f"x: stated {len(covered) + 1}, true {len(covered)}"
    ]
    assert sentence_problems(readme, planted_defects.DEFECTS, listed_test_files()) == []
    assert sentence_problems("no sentence here", planted_defects.DEFECTS, listed_test_files()) == [
        "README has no negative-control sentence"
    ]
    assert count == len(planted_defects.DEFECTS)


def test_ordinary_test_names_are_not_counted_as_negative_tests():
    for name in ("test_write_then_read_round_trips", "test_slice_names_parse", "test_failover_is_logged"):
        assert not NEGATIVE_NAME.search(name), name
    for name in ("test_a_stale_entry_is_detected", "test_uint8_pixels_are_rejected", "test_an_empty_new_run_fails"):
        assert NEGATIVE_NAME.search(name), name
