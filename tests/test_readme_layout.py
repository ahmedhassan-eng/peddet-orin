"""The README "Repository layout" tree matches the files git tracks."""
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
BEGIN = "<!-- layout:begin -->"
END = "<!-- layout:end -->"
# A tree line: guides, a branch glyph, the name, then an optional description.
ENTRY = re.compile(r"^(?P<guides>(?:│   |    )*)(?:├── |└── )(?P<name>\S+)(?:\s{2,}.*)?$")
INDENT = 4


def layout_entries(readme: str) -> list[str]:
    """Return the tree's paths; a directory ends with ``/``."""
    start = readme.index(BEGIN) + len(BEGIN)
    fence = readme[start : readme.index(END)].split("```")
    assert len(fence) == 3, "the layout block must hold exactly one fenced code block"
    lines = [line for line in fence[1].splitlines()[1:] if line.strip()]
    stack: list[str] = []
    paths = []
    for line in lines[1:]:  # The first line names the repository root.
        match = ENTRY.match(line)
        if not match:
            raise ValueError(f"not a tree line: {line!r}")
        depth = len(match.group("guides")) // INDENT
        if depth > len(stack):
            raise ValueError(f"entry is nested under a file or skips a level: {line!r}")
        del stack[depth:]
        path = "".join(stack) + match.group("name")
        paths.append(path)
        if path.endswith("/"):
            stack.append(match.group("name"))
    return paths


def layout_problems(paths: list[str], tracked: list[str]) -> list[str]:
    """Compare the tree with the tracked files, both ways."""
    directories = [path for path in paths if path.endswith("/")]
    files = {path for path in paths if not path.endswith("/")}
    # A directory with no entries under it stands for everything it holds.
    leaves = [d for d in directories if not any(p != d and p.startswith(d) for p in paths)]
    problems = []
    for path in sorted(files):
        if path not in tracked:
            problems.append(f"listed but not tracked: {path}")
    for directory in directories:
        if not any(name.startswith(directory) for name in tracked):
            problems.append(f"listed directory holds no tracked file: {directory}")
    for name in tracked:
        if name not in files and not any(name.startswith(leaf) for leaf in leaves):
            problems.append(f"tracked but not in the layout: {name}")
    duplicates = sorted({path for path in paths if paths.count(path) > 1})
    problems.extend(f"listed twice: {path}" for path in duplicates)
    return problems


def tracked_files() -> list[str]:
    done = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return done.stdout.splitlines()


def test_layout_matches_tracked_files():
    assert layout_problems(layout_entries(README.read_text()), tracked_files()) == []


def test_an_untracked_listed_file_is_reported():
    paths = layout_entries(README.read_text())
    assert layout_problems([*paths, "MISSING.md"], tracked_files()) == ["listed but not tracked: MISSING.md"]


def test_a_tracked_file_missing_from_the_layout_is_reported():
    problems = layout_problems(layout_entries(README.read_text()), [*tracked_files(), "extra.txt"])
    assert problems == ["tracked but not in the layout: extra.txt"]


def test_a_malformed_tree_line_is_rejected():
    text = README.read_text()
    assert "├── LICENSE" in text
    with pytest.raises(ValueError):
        layout_entries(text.replace("├── LICENSE", "├─ LICENSE"))
