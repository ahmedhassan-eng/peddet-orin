"""AGENTS.md carries the shared commit-message rule byte for byte, in the shared-rules v2 block.

The block is a declared-digest line, then BEGIN and END markers, the shape the shared-rules checker reads.
"""
import hashlib
from pathlib import Path

import pytest

AGENTS = Path(__file__).resolve().parents[1] / "AGENTS.md"
DIGEST_LINE = b"<!-- shared-rules v2 sha256=%s -->\n"
BEGIN = b"<!-- BEGIN SHARED RULES v2 -->\n"
END = b"<!-- END SHARED RULES v2 -->"
# sha256 and size of commit-rule.md as shared by this project's repositories.
RULE_SHA256 = "1900994bbf2c4f6c7090742d7457c66e7988fd0293d6b91023924c3cca987686"
RULE_BYTES = 2174


def rule_bytes(document: bytes) -> bytes:
    """Return the bytes between the markers, refusing a missing or repeated marker."""
    for marker in (BEGIN, END):
        if document.count(marker) != 1:
            raise ValueError(f"expected one {marker!r}, found {document.count(marker)}")
    start = document.index(BEGIN) + len(BEGIN)
    end = document.index(END)
    if end < start:
        raise ValueError("END marker comes before BEGIN marker")
    return document[start:end]


def rule_matches(document: bytes) -> bool:
    copy = rule_bytes(document)
    digest = hashlib.sha256(copy).hexdigest()
    declared = DIGEST_LINE % digest.encode() + BEGIN
    return len(copy) == RULE_BYTES and digest == RULE_SHA256 and document.count(declared) == 1


def test_agents_md_carries_the_commit_rule_verbatim():
    assert rule_matches(AGENTS.read_bytes())


def test_one_changed_byte_in_the_rule_is_detected():
    document = AGENTS.read_bytes()
    position = document.index(BEGIN) + len(BEGIN) + 10
    edited = document[:position] + bytes([document[position] ^ 0x20]) + document[position + 1 :]
    assert not rule_matches(edited)


def test_a_missing_marker_is_rejected():
    with pytest.raises(ValueError):
        rule_bytes(AGENTS.read_bytes().replace(END, b""))


def test_a_wrong_declared_digest_is_detected():
    document = AGENTS.read_bytes()
    edited = document.replace(b"sha256=" + RULE_SHA256.encode(), b"sha256=" + b"0" * 64)
    assert edited != document
    assert not rule_matches(edited)
