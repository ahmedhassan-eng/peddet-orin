"""Shared pytest setup: every test runs under the network guard, and strict skips apply in CI."""
from network_guard import network_guard  # noqa: F401  (pytest collects the fixture by name)
from strict_skips import pytest_configure  # noqa: F401  (pytest calls the hook by name)

pytest_plugins = ["pytester"]
