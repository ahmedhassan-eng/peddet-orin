"""A pytest guard that fails any test which tries to open a network connection.

Local (AF_UNIX) sockets remain usable, because they never leave the machine. The
violation is also remembered, so a test that swallows the exception still fails.
"""
from __future__ import annotations

import socket

import pytest

NETWORK_FAMILIES = (socket.AF_INET, socket.AF_INET6)


class NetworkBlocked(RuntimeError):
    """A test attempted to use the network."""


@pytest.fixture(autouse=True)
def network_guard(monkeypatch):
    attempts: list[str] = []
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def refuse(what: str):
        attempts.append(what)
        raise NetworkBlocked(f"network use is not allowed in tests: {what}")

    def guarded_connect(self, address):
        if self.family in NETWORK_FAMILIES:
            refuse("connect")
        return real_connect(self, address)

    def guarded_connect_ex(self, address):
        if self.family in NETWORK_FAMILIES:
            refuse("connect_ex")
        return real_connect_ex(self, address)

    def guarded_getaddrinfo(*arguments, **keywords):
        refuse("getaddrinfo")

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", guarded_connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    yield attempts
    if attempts:
        pytest.fail(f"test opened a network connection: {attempts}", pytrace=False)
