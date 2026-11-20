# SPDX-License-Identifier: Apache-2.0
"""AGENTS.md states the commit-message rule, and every commit keeps its checkable parts: no trailer, and no
subject that ends with a period."""

import subprocess
from pathlib import Path

import pytest

from strict_skips import checkout_skip_reason, release_skip_reason

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS = REPO_ROOT / "AGENTS.md"
RULE_LINES = (
    "- Subject: imperative mood, no trailing period, at most 72 characters where the change can be named in that.",
    "- Commit messages carry no trailers.",
)
HISTORY_SKIP = checkout_skip_reason(REPO_ROOT, history=True)
needs_history = pytest.mark.skipif(HISTORY_SKIP is not None, reason=HISTORY_SKIP or "")


def trailers(message: str) -> list[str]:
    """The trailers git itself parses from a commit message."""
    done = subprocess.run(
        ["git", "interpret-trailers", "--parse"],
        input=message,
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return done.stdout.splitlines()


def message_problems(commit: str, message: str) -> list[str]:
    """Each break of the rule's checkable parts in one message, as "<commit>: <problem>"."""
    subject = message.split("\n", 1)[0]
    problems = [f"{commit}: trailer {line}" for line in trailers(message)]
    return problems + ([f"{commit}: subject ends with a period"] if subject.rstrip().endswith(".") else [])


def history() -> list[tuple[str, str]]:
    done = subprocess.run(
        ["git", "log", "--format=%h%x00%B%x00"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    fields = done.stdout.split("\0")
    return [(fields[i].strip(), fields[i + 1]) for i in range(0, len(fields) - 1, 2)]


def test_agents_md_states_the_commit_rule():
    text = AGENTS.read_text()
    assert "## Commit messages" in text
    assert all(line in text.splitlines() for line in RULE_LINES)


@needs_history
def test_no_commit_carries_a_trailer_or_a_subject_ending_in_a_period():
    # A merged commit with a trailer would fail every later CI run, so the scan of all history runs at release only.
    if release_skip_reason():
        pytest.skip(release_skip_reason())
    commits = history()
    assert commits
    assert [problem for commit, message in commits for problem in message_problems(commit, message)] == []


# Built from parts, so this file holds no address literal.
ADDRESS = "a" + "@" + "example.org"


def test_a_trailer_or_a_period_at_the_end_of_the_subject_is_detected():
    body = "Fix the thing\n\n- Change one file\n"
    assert message_problems("abc1234", body) == []
    assert message_problems("abc1234", body + f"\nSigned-off-by: A Person <{ADDRESS}>\n") == [
        f"abc1234: trailer Signed-off-by: A Person <{ADDRESS}>"
    ]
    assert message_problems("abc1234", body.replace("thing", "thing.", 1)) == ["abc1234: subject ends with a period"]
