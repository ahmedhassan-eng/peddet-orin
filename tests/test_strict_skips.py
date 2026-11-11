"""Under PDORIN_STRICT_SKIPS=1, only the named GPU and TensorRT tests may skip."""
from pathlib import Path

from strict_skips import EXPECTED_SKIPS, unexpected_skips

TESTS_DIR = Path(__file__).resolve().parent
CONFTEST = "from strict_skips import pytest_configure\n"
PLANTED = """
import pytest


def test_planted_skip():
    pytest.skip("planted")
"""


def test_the_expected_skips_are_allowed():
    assert unexpected_skips(sorted(EXPECTED_SKIPS)) == []


def test_any_other_skip_is_reported():
    assert unexpected_skips(["test_cuda_forward_matches_cpu", "test_other"]) == ["test_other"]


def _planted_run(pytester, monkeypatch, strict: bool):
    monkeypatch.syspath_prepend(str(TESTS_DIR))
    if strict:
        monkeypatch.setenv("PDORIN_STRICT_SKIPS", "1")
    else:
        monkeypatch.delenv("PDORIN_STRICT_SKIPS", raising=False)
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(PLANTED)
    return pytester.runpytest_inprocess()


def test_an_unexpected_skip_fails_a_strict_run(pytester, monkeypatch):
    result = _planted_run(pytester, monkeypatch, strict=True)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*unexpected skips: test_planted_skip*"])


def test_the_same_skip_passes_a_lenient_run(pytester, monkeypatch):
    assert _planted_run(pytester, monkeypatch, strict=False).ret == 0
