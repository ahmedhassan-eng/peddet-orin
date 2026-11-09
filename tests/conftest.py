"""Shared pytest setup: every test runs under the network guard."""
from network_guard import network_guard  # noqa: F401  (pytest collects the fixture by name)

pytest_plugins = ["pytester"]
