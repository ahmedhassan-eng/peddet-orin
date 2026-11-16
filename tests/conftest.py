# SPDX-License-Identifier: Apache-2.0
"""Shared pytest setup: the whole session runs under the network guard, and strict skips apply in CI."""

from network_guard import network_guard  # noqa: F401  (pytest collects the fixture by name)
from network_guard import pytest_configure as configure_network_guard
from strict_skips import pytest_configure as configure_strict_skips

pytest_plugins = ["pytester"]


def pytest_configure(config) -> None:
    configure_network_guard(config)
    configure_strict_skips(config)
