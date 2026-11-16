# SPDX-License-Identifier: Apache-2.0
"""Diagnostics are on only when PDORIN_DEBUG holds a true value; a bare DEBUG turns nothing on."""
from peddet_orin import debug


def test_pdorin_debug_turns_logging_on(monkeypatch, capsys):
    monkeypatch.delenv("DEBUG", raising=False)
    monkeypatch.setenv("PDORIN_DEBUG", "1")
    debug.log("hello")
    assert capsys.readouterr().err == "[debug] hello\n"


def test_a_false_pdorin_debug_value_leaves_logging_off(monkeypatch, capsys):
    for value in ("", "0", "false", "No"):
        monkeypatch.setenv("PDORIN_DEBUG", value)
        debug.log("hello")
    assert capsys.readouterr().err == ""


def test_the_generic_debug_variable_turns_nothing_on(monkeypatch, capsys):
    monkeypatch.delenv("PDORIN_DEBUG", raising=False)
    monkeypatch.setenv("DEBUG", "1")
    assert not debug.enabled()
    debug.log("hello")
    with debug.profile("block"):
        pass
    assert capsys.readouterr().err == ""
