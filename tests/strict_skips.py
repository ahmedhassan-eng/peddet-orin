# SPDX-License-Identifier: Apache-2.0
"""A pytest plugin: under PDORIN_STRICT_SKIPS=1, any skip but the expected ones fails the run.

Two kinds of skip are expected:

* the GPU and TensorRT tests in EXPECTED_SKIPS, which need hardware CI does not have;
* the repository-rule tests, when the tree is not a full git clone: a release tarball
  (no .git) or a shallow clone. Such a skip is allowed only with exactly one of the
  reasons in CHECKOUT_REASONS, and only outside CI. CI checks out the full history, so
  there a checkout skip still fails the run.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

# The only tests allowed to skip in CI: they need a CUDA device or TensorRT, which CI does not have.
EXPECTED_SKIPS = frozenset(
    {
        "test_cuda_forward_matches_cpu",
        "test_tensorrt_fp16_engine_matches_cpu",
        "test_tensorrt_check_fails_on_unnormalised_input",
        "test_tensorrt_int8_entropy_engine_builds_runs_and_logs_its_path",
        "test_tensorrt_int8_qdq_fallback_engine_builds_runs_and_logs_its_path",
    }
)
# CI sets this, so that any other skip fails the run instead of passing quietly.
VARIABLE = "PDORIN_STRICT_SKIPS"
# The reasons a repository-rule test skips with outside a full clone; nothing else may use them.
NO_GIT_REASON = "repository rule: no .git here (a release tarball), so git cannot list the tracked files"
SHALLOW_REASON = "repository rule: a shallow clone lacks the history this test looks commits up in"
CHECKOUT_REASONS = frozenset({NO_GIT_REASON, SHALLOW_REASON})
SKIPPED_PREFIX = "Skipped: "


def in_ci() -> bool:
    """Return whether this is a CI run; GitHub Actions and most CI services set CI=true."""
    return os.environ.get("CI", "").strip().lower() not in ("", "0", "false")


def checkout_skip_reason(root: Path, history: bool = False) -> str | None:
    """Return why a repository-rule test cannot run in ``root``, or None if it can.

    ``root`` itself must hold .git: a tarball unpacked inside some other repository must
    not borrow that repository's files. With ``history``, a shallow clone is not enough.
    """
    if not (root / ".git").exists():
        return NO_GIT_REASON
    if history:
        done = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=root,
                              capture_output=True, text=True, check=False)
        if done.stdout.strip() == "true":
            return SHALLOW_REASON
    return None


def unexpected_skips(skipped: list[str | tuple[str, str]], ci: bool = True) -> list[str]:
    """Return the skipped test names that are not expected to skip.

    Each entry is a test name, or a ``(name, reason)`` pair. A checkout reason is
    expected only when ``ci`` is false.
    """
    unexpected = set()
    for entry in skipped:
        name, reason = (entry, "") if isinstance(entry, str) else entry
        if name.split("[")[0] in EXPECTED_SKIPS:
            continue
        if not ci and reason in CHECKOUT_REASONS:
            continue
        unexpected.add(name)
    return sorted(unexpected)


def skip_reason(report) -> str:
    """Return the reason text of a skip report, without pytest's "Skipped: " prefix."""
    longrepr = report.longrepr
    message = longrepr[2] if isinstance(longrepr, tuple) and len(longrepr) == 3 else str(longrepr)
    return message.removeprefix(SKIPPED_PREFIX)


class StrictSkips:
    """Collect one session's skips; each pytest session gets its own instance."""

    def __init__(self) -> None:
        self.skipped: list[tuple[str, str]] = []

    def pytest_runtest_logreport(self, report) -> None:
        if report.skipped:
            self.skipped.append((report.nodeid.rsplit("::", 1)[-1], skip_reason(report)))

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus) -> None:
        if os.environ.get(VARIABLE) != "1":
            return
        unexpected = unexpected_skips(self.skipped, ci=in_ci())
        if unexpected:
            print(f"\nunexpected skips: {', '.join(unexpected)}")
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_configure(config) -> None:
    config.pluginmanager.register(StrictSkips(), "peddet-orin-strict-skips")
