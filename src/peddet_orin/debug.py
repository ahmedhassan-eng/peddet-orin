# SPDX-License-Identifier: Apache-2.0
"""Logging and profiling helpers, silent unless the ``PDORIN_DEBUG`` environment variable is set.

Production paths call these freely: with ``PDORIN_DEBUG`` unset, ``log`` returns at once
and ``profile`` measures nothing, so timed code is not perturbed by diagnostics.
"""
from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

# Prefixed, so a DEBUG that another tool reads does not turn on this package's diagnostics.
DEBUG_ENV_VAR = "PDORIN_DEBUG"
FALSE_VALUES = ("", "0", "false", "no")


def enabled() -> bool:
    """Report whether diagnostics are on. Read on each call so tests can toggle it."""
    return os.environ.get(DEBUG_ENV_VAR, "").strip().lower() not in FALSE_VALUES


def log(message: str) -> None:
    """Write one diagnostic line to stderr when ``PDORIN_DEBUG`` is set."""
    if enabled():
        print(f"[debug] {message}", file=sys.stderr, flush=True)


@contextmanager
def profile(label: str) -> Iterator[None]:
    """Log the wall time of the enclosed block when ``PDORIN_DEBUG`` is set."""
    if not enabled():
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        log(f"{label}: {(time.perf_counter() - started) * 1e3:.2f} ms")
