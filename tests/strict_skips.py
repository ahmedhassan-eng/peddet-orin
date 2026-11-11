"""A pytest plugin: under PDORIN_STRICT_SKIPS=1, any skip but the expected ones fails the run."""
from __future__ import annotations

import os

import pytest

# The only tests allowed to skip: they need a CUDA device or TensorRT, which CI does not have.
EXPECTED_SKIPS = frozenset(
    {
        "test_cuda_forward_matches_cpu",
        "test_tensorrt_fp16_engine_matches_cpu",
        "test_tensorrt_check_fails_on_unnormalised_input",
    }
)
# CI sets this, so that any other skip fails the run instead of passing quietly.
VARIABLE = "PDORIN_STRICT_SKIPS"


def unexpected_skips(skipped: list[str]) -> list[str]:
    """Return the skipped test names that are not expected to skip."""
    return sorted({name for name in skipped if name.split("[")[0] not in EXPECTED_SKIPS})


class StrictSkips:
    """Collect one session's skips; each pytest session gets its own instance."""

    def __init__(self) -> None:
        self.skipped: list[str] = []

    def pytest_runtest_logreport(self, report) -> None:
        if report.skipped:
            self.skipped.append(report.nodeid.rsplit("::", 1)[-1])

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus) -> None:
        if os.environ.get(VARIABLE) != "1":
            return
        unexpected = unexpected_skips(self.skipped)
        if unexpected:
            print(f"\nunexpected skips: {', '.join(unexpected)}")
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_configure(config) -> None:
    config.pluginmanager.register(StrictSkips(), "peddet-orin-strict-skips")
