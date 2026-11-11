"""The guard must actually trip; a guard that never fires proves nothing."""
import socket
from pathlib import Path

import pytest

from network_guard import NetworkBlocked

TESTS_DIR = Path(__file__).resolve().parent
DISCARD_PORT = 9
# Derived from the socket module so the test source carries no address literal.
LOOPBACK_ADDRESS = (socket.inet_ntoa(socket.INADDR_LOOPBACK.to_bytes(4, "big")), DISCARD_PORT)
OFFENDING_TEST = """
import socket

def test_opens_socket():
    address = (socket.inet_ntoa(socket.INADDR_LOOPBACK.to_bytes(4, "big")), 9)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(address)
"""
SWALLOWING_TEST = """
import socket

def test_hides_socket_failure():
    try:
        address = (socket.inet_ntoa(socket.INADDR_LOOPBACK.to_bytes(4, "big")), 9)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(address)
    except Exception:
        pass
"""
CONFTEST = "from network_guard import network_guard\n"


def test_guard_raises_inside_a_test(network_guard):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        with pytest.raises(NetworkBlocked):
            connection.connect(LOOPBACK_ADDRESS)
    # This test tripped the guard on purpose, so clear the record for teardown.
    network_guard.clear()


def _outcome(pytester, monkeypatch, source: str):
    monkeypatch.syspath_prepend(str(TESTS_DIR))
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(source)
    return pytester.runpytest_inprocess()


def test_a_test_that_opens_a_socket_fails(pytester, monkeypatch):
    result = _outcome(pytester, monkeypatch, OFFENDING_TEST)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*network*"])


def test_swallowing_the_exception_does_not_help(pytester, monkeypatch):
    result = _outcome(pytester, monkeypatch, SWALLOWING_TEST)
    assert result.ret != 0
