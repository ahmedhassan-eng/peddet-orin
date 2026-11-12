# SPDX-License-Identifier: Apache-2.0
"""No two tests in the suite share a name, so a report line always points at one test."""
import ast
from collections import Counter
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent


def names_in(source: str) -> list[str]:
    """Return every test function name in a module, including methods of test classes."""
    names = []
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
            names.append(node.name)
        elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            names.extend(
                f"{node.name}.{item.name}"
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name.startswith("test")
            )
    return names


def duplicated(sources: dict[str, str]) -> dict[str, int]:
    counts = Counter(name for source in sources.values() for name in names_in(source))
    return {name: count for name, count in counts.items() if count > 1}


def suite_sources() -> dict[str, str]:
    return {path.name: path.read_text() for path in sorted(TESTS_DIR.glob("test_*.py"))}


def test_no_two_tests_share_a_name():
    assert duplicated(suite_sources()) == {}


def test_a_name_shared_across_two_files_is_found():
    sources = {"test_a.py": "def test_same():\n    pass\n", "test_b.py": "def test_same():\n    pass\n"}
    assert duplicated(sources) == {"test_same": 2}


def test_a_name_redefined_in_one_file_is_found():
    sources = {"test_a.py": "def test_twice():\n    pass\n\n\ndef test_twice():\n    pass\n"}
    assert duplicated(sources) == {"test_twice": 2}
