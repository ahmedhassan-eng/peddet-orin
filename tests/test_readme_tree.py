# SPDX-License-Identifier: Apache-2.0
"""The README "Repository layout" tree is exactly what scripts/generate_tree.py makes of git ls-files."""

import pytest

import generate_tree
from strict_skips import checkout_skip_reason

CHECKOUT_SKIP = checkout_skip_reason(generate_tree.REPO_ROOT)
# Every test here reads git ls-files, which a release tarball (no .git) cannot answer.
pytestmark = pytest.mark.skipif(CHECKOUT_SKIP is not None, reason=CHECKOUT_SKIP or "")
README_TEXT = generate_tree.README.read_text()


def test_readme_tree_equals_generated_tree():
    files = generate_tree.tracked_files()
    assert generate_tree.readme_block(README_TEXT) == generate_tree.build_tree(files)


def test_every_tree_note_names_a_shown_path():
    assert generate_tree.unknown_descriptions(generate_tree.tracked_files()) == []


def test_a_stale_tree_is_detected():
    # Negative control: the tree as it was before a file was added no longer matches.
    files = generate_tree.tracked_files()
    stale = generate_tree.build_tree([name for name in files if name != "LICENSE"])
    assert stale != generate_tree.build_tree(files)
    planted = README_TEXT.replace(generate_tree.readme_block(README_TEXT), stale)
    assert generate_tree.readme_block(planted) != generate_tree.build_tree(files)


def test_a_new_tracked_file_changes_the_tree():
    files = generate_tree.tracked_files()
    assert "├── NEW.md" in generate_tree.build_tree([*files, "NEW.md"])


def test_a_note_for_a_removed_file_is_reported():
    files = [name for name in generate_tree.tracked_files() if name != "PRD.md"]
    assert generate_tree.unknown_descriptions(files) == ["PRD.md"]
