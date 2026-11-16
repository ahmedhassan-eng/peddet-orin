# SPDX-License-Identifier: Apache-2.0
"""The README's negative-control sentence states the true coverage of the planted-defect registry.

A test file counts as covered when a registry entry names one of its tests, and as carrying an in-file negative
test when NEGATIVE_TESTS below names one of its tests that plants or feeds a bad input and checks that it is
refused. QUEUE.md names every test file that has neither.
"""
import re
from pathlib import Path

import planted_defects

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
QUEUE = REPO_ROOT / "QUEUE.md"
# One in-file negative test for each test file that no registry entry names; each must exist in its file.
NEGATIVE_TESTS = {
    "tests/test_checkout_kinds.py": "test_a_shallow_clone_in_ci_still_fails_a_strict_run",
    "tests/test_device_resources.py": "test_write_rejects_a_wrong_shape_with_the_same_byte_count",
    "tests/test_internal_ids.py": "test_an_id_planted_in_any_tracked_path_is_found",
    "tests/test_measure_cli_checks.py": "test_measure_corrupt_engine_is_refused",
    "tests/test_negative_controls.py": "test_a_readme_sentence_with_one_number_changed_is_refused",
    "tests/test_network_guard.py": "test_a_test_that_opens_a_socket_fails",
    "tests/test_packaging_metadata.py": "test_an_unpinned_transitive_dependency_is_detected",
    "tests/test_planted_defects.py": "test_a_stale_registry_entry_is_detected",
    "tests/test_preprocess.py": "test_uint8_pixels_are_rejected",
    "tests/test_repo_root.py": "test_pdorin_root_without_a_pyproject_is_refused",
    "tests/test_strict_skips.py": "test_an_unexpected_skip_fails_a_strict_run",
    "tests/test_weights.py": "test_a_weights_file_with_another_sha256_is_refused",
}
SENTENCE = re.compile(
    r"`tests/planted_defects\.py` registers (?P<n>\d+) planted defects.*?the release check requires all "
    r"(?P<caught>\d+) to be caught\. They cover (?P<x>\d+) of the (?P<y>\d+) test files, (?P<z>\d+) more carry "
    r"in-file negative tests"
)


def listed_test_files() -> list[Path]:
    return sorted((REPO_ROOT / "tests").glob("test_*.py"))


def missing_negative_tests(files: list[Path], named: dict[str, str] = NEGATIVE_TESTS) -> list[str]:
    """Each ``file::test`` in ``named`` whose file is not among ``files`` or does not define that test."""
    present = {f"tests/{path.name}": set(re.findall(r"^def (test_\w+)", path.read_text(), re.M)) for path in files}
    return [f"{name}::{test}" for name, test in sorted(named.items()) if test not in present.get(name, set())]


def coverage(defects: list[dict], files: list[Path]) -> tuple[int, list[str], list[str], list[str]]:
    """The registry size, and the test files that are covered, that carry in-file negative tests, and the rest."""
    named = {defect["catches"].split("::", 1)[0] for defect in defects}
    covered, negative, rest = [], [], []
    for path in files:
        name = f"tests/{path.name}"
        if name in named:
            covered.append(name)
        elif name in NEGATIVE_TESTS:
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


def test_every_named_negative_test_exists_in_its_file():
    assert missing_negative_tests(listed_test_files()) == []


def test_a_named_negative_test_that_does_not_exist_is_refused():
    named = {**NEGATIVE_TESTS, "tests/test_weights.py": "test_no_such_test_is_refused"}
    assert missing_negative_tests(listed_test_files(), named) == ["tests/test_weights.py::test_no_such_test_is_refused"]
