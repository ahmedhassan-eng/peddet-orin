# SPDX-License-Identifier: Apache-2.0
"""The guard must actually trip; a guard that never fires proves nothing."""

import os
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
MODULE_FIXTURE_TEST = """
import socket

import pytest


@pytest.fixture(scope="module")
def resolved():
    return socket.getaddrinfo("localhost", 9)


def test_uses_a_resolved_name(resolved):
    assert resolved
"""
SWALLOWING_MODULE_FIXTURE_TEST = """
import socket

import pytest


@pytest.fixture(scope="module")
def resolved():
    try:
        return socket.getaddrinfo("localhost", 9)
    except Exception:
        return None


def test_uses_a_quietly_unresolved_name(resolved):
    pass
"""
# The suite's own conftest, so the planted tests run under exactly the guard every test gets.
CONFTEST = (TESTS_DIR / "conftest.py").read_text()


def test_guard_raises_inside_a_test(network_guard):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        with pytest.raises(NetworkBlocked):
            connection.connect(LOOPBACK_ADDRESS)
    # This test tripped the guard on purpose, so clear the record for teardown.
    network_guard.clear()


def _outcome(pytester, monkeypatch, source: str, isolated: bool = False):
    """Run ``source`` under the suite's conftest; ``isolated`` uses a fresh process.

    In process, this test's own guard is still active underneath, so only a separate
    process shows what the planted test's guard alone blocks.
    """
    monkeypatch.syspath_prepend(str(TESTS_DIR))
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(source)
    if not isolated:
        return pytester.runpytest_inprocess()
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [str(TESTS_DIR), os.environ.get("PYTHONPATH")])))
    return pytester.runpytest_subprocess("-p", "no:cacheprovider")


def test_a_test_that_opens_a_socket_fails(pytester, monkeypatch):
    result = _outcome(pytester, monkeypatch, OFFENDING_TEST)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*network*"])


def test_swallowing_the_exception_does_not_help(pytester, monkeypatch):
    result = _outcome(pytester, monkeypatch, SWALLOWING_TEST)
    assert result.ret != 0


def test_a_module_scoped_fixture_that_resolves_a_name_is_blocked(pytester, monkeypatch):
    result = _outcome(pytester, monkeypatch, MODULE_FIXTURE_TEST, isolated=True)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["*NetworkBlocked: network use is not allowed in tests: getaddrinfo*"])


def test_a_module_scoped_fixture_that_swallows_the_block_still_fails(pytester, monkeypatch):
    result = _outcome(pytester, monkeypatch, SWALLOWING_MODULE_FIXTURE_TEST, isolated=True)
    assert result.ret != 0
    result.stdout.fnmatch_lines(["setup opened a network connection: *getaddrinfo*"])


def test_a_udp_sendto_is_blocked(network_guard):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as datagram:
        with pytest.raises(NetworkBlocked, match="sendto"):
            datagram.sendto(b"x", LOOPBACK_ADDRESS)
    assert network_guard == ["sendto"]
    network_guard.clear()


def test_a_udp_sendmsg_is_blocked(network_guard):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as datagram:
        with pytest.raises(NetworkBlocked, match="sendmsg"):
            datagram.sendmsg([b"x"], [], 0, LOOPBACK_ADDRESS)
    assert network_guard == ["sendmsg"]
    network_guard.clear()


def test_create_connection_is_blocked_before_any_lookup(network_guard):
    with pytest.raises(NetworkBlocked, match="create_connection"):
        socket.create_connection(LOOPBACK_ADDRESS, timeout=1)
    assert network_guard == ["create_connection"]
    network_guard.clear()


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("gethostbyname", ("localhost",)),
        ("gethostbyname_ex", ("localhost",)),
        ("gethostbyaddr", (LOOPBACK_ADDRESS[0],)),
        ("getnameinfo", ((LOOPBACK_ADDRESS[0], 80), 0)),
    ],
)
def test_every_name_lookup_is_blocked(network_guard, name, arguments):
    with pytest.raises(NetworkBlocked, match=name):
        getattr(socket, name)(*arguments)
    assert network_guard == [name]
    network_guard.clear()


def test_a_local_unix_datagram_still_works():
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    with left, right:
        left.sendmsg([b"local"])
        assert right.recv(16) == b"local"
