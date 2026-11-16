# SPDX-License-Identifier: Apache-2.0
"""Under PDORIN_STRICT_SKIPS=1, only the named GPU and TensorRT tests may skip."""
from pathlib import Path

import pytest

from strict_skips import EXPECTED_SKIPS, NO_GIT_REASON, SHALLOW_REASON, unexpected_skips

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


def _planted_run(pytester, monkeypatch, strict: bool, source: str = PLANTED, ci: bool = False, gpu: bool = False):
    monkeypatch.syspath_prepend(str(TESTS_DIR))
    if ci:
        monkeypatch.setenv("CI", "true")
    else:
        monkeypatch.delenv("CI", raising=False)
    if strict:
        monkeypatch.setenv("PDORIN_STRICT_SKIPS", "1")
    else:
        monkeypatch.delenv("PDORIN_STRICT_SKIPS", raising=False)
    if gpu:
        monkeypatch.setenv("PDORIN_REQUIRE_GPU", "1")
    else:
        monkeypatch.delenv("PDORIN_REQUIRE_GPU", raising=False)
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(source)
    return pytester.runpytest_inprocess()


def test_an_unexpected_skip_fails_a_strict_run(pytester, monkeypatch):
    result = _planted_run(pytester, monkeypatch, strict=True)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*unexpected skips: test_planted_skip*"])


def test_the_same_skip_passes_a_lenient_run(pytester, monkeypatch):
    assert _planted_run(pytester, monkeypatch, strict=False).ret == 0


GPU_PLANTED = """
import pytest


def test_cuda_forward_matches_cpu():
    pytest.skip("no CUDA device")
"""


def test_a_gpu_skip_passes_a_strict_run_but_fails_when_the_gpu_is_required(pytester, monkeypatch):
    assert _planted_run(pytester, monkeypatch, strict=True, source=GPU_PLANTED).ret == 0
    result = _planted_run(pytester, monkeypatch, strict=False, source=GPU_PLANTED, gpu=True)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*unexpected skips: test_cuda_forward_matches_cpu*"])
    assert unexpected_skips(sorted(EXPECTED_SKIPS), gpu=True) == sorted(EXPECTED_SKIPS)


CHECKOUT_PLANTED = """
import pytest

from strict_skips import {constant}


def test_planted_checkout_skip():
    pytest.skip({constant})
"""
NEAR_MISS_PLANTED = """
import pytest

from strict_skips import NO_GIT_REASON


def test_planted_near_miss_skip():
    pytest.skip(NO_GIT_REASON + ".")
"""


@pytest.mark.parametrize("constant", ["NO_GIT_REASON", "SHALLOW_REASON"])
def test_a_checkout_skip_passes_a_strict_run_outside_ci(pytester, monkeypatch, constant):
    result = _planted_run(pytester, monkeypatch, strict=True, source=CHECKOUT_PLANTED.format(constant=constant))
    assert result.ret == 0
    result.assert_outcomes(skipped=1)


@pytest.mark.parametrize("constant", ["NO_GIT_REASON", "SHALLOW_REASON"])
def test_a_checkout_skip_fails_a_strict_run_in_ci(pytester, monkeypatch, constant):
    result = _planted_run(pytester, monkeypatch, strict=True, source=CHECKOUT_PLANTED.format(constant=constant),
                          ci=True)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*unexpected skips: test_planted_checkout_skip*"])


def test_a_skip_whose_reason_only_resembles_a_checkout_reason_fails(pytester, monkeypatch):
    result = _planted_run(pytester, monkeypatch, strict=True, source=NEAR_MISS_PLANTED)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*unexpected skips: test_planted_near_miss_skip*"])


def test_checkout_reasons_are_expected_only_outside_ci():
    skipped = [("test_history_is_complete", SHALLOW_REASON), ("test_readme_tree_equals_generated_tree", NO_GIT_REASON),
               ("test_other", "some other reason")]
    assert unexpected_skips(skipped, ci=False) == ["test_other"]
    assert unexpected_skips(skipped, ci=True) == ["test_history_is_complete", "test_other",
                                                  "test_readme_tree_equals_generated_tree"]
