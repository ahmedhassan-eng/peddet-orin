# SPDX-License-Identifier: Apache-2.0
"""A pytest plugin that fails any test which tries to use the network.

The guard is installed once for the whole session, at pytest_configure, before any
fixture of any scope runs, so session- and module-scoped fixtures and test-module
imports are guarded too. It blocks name lookups (getaddrinfo, create_connection) and
every way an internet socket sends or connects: connect, connect_ex, sendto and
sendmsg. Local (AF_UNIX) sockets remain usable, because they never leave the machine.

Every attempt is also remembered, so a test or fixture that swallows the exception
still fails: the phase (setup, call or teardown) in which it happened is reported as
failed.
"""
from __future__ import annotations

import socket

import pytest

NETWORK_FAMILIES = (socket.AF_INET, socket.AF_INET6)
PLUGIN_NAME = "peddet-orin-network-guard"
# The guarded socket methods, which refuse only for internet families.
SOCKET_METHODS = ("connect", "connect_ex", "sendto", "sendmsg")
# The guarded module functions, which always refuse.
MODULE_FUNCTIONS = ("getaddrinfo", "create_connection")


class NetworkBlocked(RuntimeError):
    """A test attempted to use the network."""


class NetworkGuard:
    """Patch the socket module for one pytest session and fail whatever touches the network."""

    def __init__(self) -> None:
        self.attempts: list[str] = []
        self._saved: list[tuple[object, str, object]] = []

    def refuse(self, what: str):
        self.attempts.append(what)
        raise NetworkBlocked(f"network use is not allowed in tests: {what}")

    def _socket_method(self, name: str, real):
        def guarded(sock, *arguments, **keywords):
            if sock.family in NETWORK_FAMILIES:
                self.refuse(name)
            return real(sock, *arguments, **keywords)

        guarded.__name__ = name
        return guarded

    def _module_function(self, name: str):
        def guarded(*arguments, **keywords):
            self.refuse(name)

        guarded.__name__ = name
        return guarded

    def install(self) -> None:
        for name in SOCKET_METHODS:
            real = getattr(socket.socket, name)
            self._saved.append((socket.socket, name, real))
            setattr(socket.socket, name, self._socket_method(name, real))
        for name in MODULE_FUNCTIONS:
            self._saved.append((socket, name, getattr(socket, name)))
            setattr(socket, name, self._module_function(name))

    def uninstall(self) -> None:
        while self._saved:
            owner, name, real = self._saved.pop()
            setattr(owner, name, real)

    def _take(self) -> list[str]:
        attempts, self.attempts[:] = list(self.attempts), []
        return attempts

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        outcome = yield
        attempts = self._take()
        report = outcome.get_result()
        if attempts and not report.failed:
            report.outcome = "failed"
            report.longrepr = f"{report.when} opened a network connection: {attempts}"

    @pytest.hookimpl(hookwrapper=True)
    def pytest_make_collect_report(self, collector):
        outcome = yield
        attempts = self._take()
        report = outcome.get_result()
        if attempts and not report.failed:
            report.outcome = "failed"
            report.longrepr = f"collecting {collector.nodeid} opened a network connection: {attempts}"

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus) -> None:
        attempts = self._take()
        if attempts:
            print(f"\nnetwork use outside any test: {attempts}")
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_unconfigure(self, config) -> None:
        self.uninstall()


def pytest_configure(config) -> None:
    """Install the guard for the session; a nested pytester session stacks its own on top."""
    guard = NetworkGuard()
    guard.install()
    config.pluginmanager.register(guard, PLUGIN_NAME)


@pytest.fixture
def network_guard(request) -> list[str]:
    """Return the session guard's record of attempts, so a test that trips it on purpose can clear it."""
    return request.config.pluginmanager.get_plugin(PLUGIN_NAME).attempts
