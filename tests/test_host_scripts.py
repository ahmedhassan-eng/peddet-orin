# SPDX-License-Identifier: Apache-2.0
"""scripts/host/quiet.py (the QUIET gate and sampler) and scripts/host/host_setup.py (the root setup), on CPU,
with every host file, systemctl call and command replaced by a stand-in."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

HOST = Path(__file__).resolve().parents[1] / "scripts" / "host"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HOST / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


quiet = _load("quiet")
host_setup = _load("host_setup")


def _host(tmp_path, available=6_000_000, swap_free=1024, pswpout=7):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(f"MemAvailable: {available} kB\nSwapTotal: 1024 kB\nSwapFree: {swap_free} kB\n")
    vmstat = tmp_path / "vmstat"
    vmstat.write_text(f"pswpin 3\npswpout {pswpout}\n")
    thermal = tmp_path / "thermal"
    for number, (kind, temp) in enumerate([("cpu-thermal", "51000"), ("dead", "x"), ("gpu-thermal", "52000")]):
        zone = thermal / f"thermal_zone{number}"
        zone.mkdir(parents=True)
        (zone / "type").write_text(kind + "\n")
        (zone / "temp").write_text(temp + "\n")
    return meminfo, vmstat, thermal


def test_the_sampler_writes_numbers_only_from_every_zone_that_reads(tmp_path):
    meminfo, vmstat, thermal = _host(tmp_path)
    zones = quiet.zones(thermal)
    assert [name for name, _ in zones] == ["cpu-thermal", "gpu-thermal"]
    line = quiet.sample(meminfo, vmstat, zones)
    values = quiet.values(line)
    assert line.startswith("SYS ") and values["mem_available_kb"] == "6000000" and values["swap_used_kb"] == "0"
    assert values["pswpout"] == "7" and values["cpu-thermal_mc"] == "51000" and values["gpu-thermal_mc"] == "52000"


def test_a_units_file_is_parsed_and_a_bad_line_is_refused(tmp_path):
    units = tmp_path / "units"
    units.write_text("watch x-*.timer\nallow x-beat.timer  # kept\nbusy x-tick*.service\nscope x-judge*.scope 5000\n")
    rules = quiet.read_rules(units)
    assert (rules.watch, rules.allow, rules.busy, rules.scopes) == (
        ["x-*.timer"],
        ["x-beat.timer"],
        ["x-tick*.service"],
        [("x-judge*.scope", 5000)],
    )
    units.write_text("watch\n")
    with pytest.raises(ValueError, match="line 1"):
        quiet.read_rules(units)


def _terms(**changed):
    terms = {
        "held_active": 0,
        "allowed_timers": 1,
        "scopes_over": 0,
        "busy_services": 0,
        "load1": "0.20",
        "swap_used_kb": "0",
        "pswpout_delta": "0",
        "mem_available_kb": "6000000",
        "temps": {"cpu-thermal_mc": "51000"},
    }
    return {**terms, **changed}


@pytest.mark.parametrize(
    ("changed", "term"),
    [
        ({"held_active": 1}, "held_active"),
        ({"scopes_over": 1}, "scopes_over"),
        ({"busy_services": 2}, "busy_services"),
        ({"swap_used_kb": "256"}, "swap_used"),
        ({"pswpout_delta": "11"}, "pswpout"),
        ({"mem_available_kb": "5000000"}, "mem_available"),
        ({"load1": "1.00"}, "load1"),
    ],
)
def test_each_gate_term_fails_alone(changed, term):
    assert quiet.judge(_terms(), 5_859_375, 1.0) == []
    assert quiet.judge(_terms(**changed), 5_859_375, 1.0) == [term]


def test_one_try_counts_held_allowed_scopes_and_busy_units_and_the_swap_out_delta():
    rules = quiet.Rules(watch=["x-*"], allow=["x-beat.timer"], busy=["x-tick*"], scopes=[("x-judge*", 5000)])
    active = {"active": {("x-*",): ["x-beat.timer", "x-push.timer"], ("x-tick*",): ["x-tick.service"]}}

    def units(patterns, state):
        if state == "running":
            return ["x-judge-1.scope"]
        return active["active"].get(tuple(patterns), [])

    cpu = iter([0, 6_000_000_000])
    lines = iter(["SYS t pswpout=5 swap_used_kb=0", "SYS t mem_available_kb=6000000 swap_used_kb=0 pswpout=9"])
    terms = quiet.one_try(
        rules, 0, units=units, cpu_ns=lambda unit: next(cpu), take=lambda: next(lines), load1=lambda: "0.10"
    )
    assert (terms["held_active"], terms["allowed_timers"], terms["scopes_over"], terms["busy_services"]) == (1, 1, 1, 1)
    assert terms["pswpout_delta"] == "4" and terms["mem_available_kb"] == "6000000"


def test_the_gate_writes_counts_only_and_passes_or_fails(tmp_path):
    quiet_file = tmp_path / "quiet.txt"
    tries = iter([_terms(load1="1.50"), _terms()])
    assert quiet.gate("start", quiet.Rules(), quiet_file, 0, 3, 5_859_375, 1.0, measure=lambda r, s: next(tries)) == 0
    lines = quiet_file.read_text().splitlines()
    assert lines[0].startswith("QUIET start try=1 held_active=0 allowed_timers=1") and "fail=load1" in lines[0]
    assert "fail=none" in lines[1] and lines[2] == "QUIET start PASS"
    again = tmp_path / "again.txt"
    assert quiet.gate("b", quiet.Rules(), again, 0, 2, 5_859_375, 1.0, measure=lambda r, s: _terms(held_active=1)) == 1
    assert again.read_text().splitlines()[-1] == "QUIET b FAIL terms=held_active"


def test_the_gates_dry_plan_names_counts_not_units(tmp_path, capsys):
    units = tmp_path / "units"
    units.write_text("watch secret-unit.timer\nallow secret-beat.timer\n")
    assert quiet.main(["gate", "start", str(units), str(tmp_path / "q.txt"), "--dry"]) == 0
    out = capsys.readouterr().out
    assert "watch=1 allow=1" in out and "secret" not in out and not (tmp_path / "q.txt").exists()


def test_the_setup_reads_exact_unit_names_only():
    assert host_setup.read_units("stop x-tick.timer\nstop docker.service  # the container runtime\n") == [
        "x-tick.timer",
        "docker.service",
    ]
    for bad in ("stop x-*.timer", "start x.service", "stop x.service; reboot", "stop"):
        with pytest.raises(ValueError, match="line 1"):
            host_setup.read_units(bad + "\n")


def _units_dir(tmp_path):
    directory = tmp_path / "root-units"
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)
    units = directory / "units"
    units.write_text("stop x.service\n")
    units.chmod(0o644)
    return directory, units


def test_root_reads_a_units_file_it_can_trust(tmp_path):
    _, units = _units_dir(tmp_path)
    assert host_setup.read_trusted(units, owner=os.getuid()) == "stop x.service\n"


def test_root_refuses_a_units_file_it_should_not_trust(tmp_path):
    directory, units = _units_dir(tmp_path)
    # A file this test user owns is not root's, so root refuses it.
    with pytest.raises(ValueError, match="not owned by root"):
        host_setup.read_trusted(units)
    link = directory / "link"
    link.symlink_to(units)
    with pytest.raises(ValueError, match="cannot be opened safely"):
        host_setup.read_trusted(link, owner=os.getuid())
    fifo = directory / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="not a regular file"):
        host_setup.read_trusted(fifo, owner=os.getuid())
    big = directory / "big"
    big.write_text("# x\n" * (host_setup.MAX_UNITS_BYTES // 4 + 1))
    big.chmod(0o644)
    with pytest.raises(ValueError, match="is over 65536 bytes"):
        host_setup.read_trusted(big, owner=os.getuid())
    units.chmod(0o664)
    with pytest.raises(ValueError, match="the units file is writable"):
        host_setup.read_trusted(units, owner=os.getuid())
    units.chmod(0o644)
    directory.chmod(0o775)
    with pytest.raises(ValueError, match="directory is writable"):
        host_setup.read_trusted(units, owner=os.getuid())
    directory.chmod(0o755)


def test_the_setup_runs_only_as_root_except_dry_and_prints_counts_only(tmp_path, capsys):
    text = "stop secret-a.service\nstop secret-b.timer\n"
    units = tmp_path / "units"
    read = lambda path: text  # noqa: E731
    assert host_setup.main(["stop", str(units), "--dry"], is_root=lambda: False, read=read) == 0
    assert "SETUP-PLAN stop units=2 swap=off swappiness=1" in capsys.readouterr().out
    assert host_setup.main(["stop", str(units)], is_root=lambda: False, read=read) == 2
    with pytest.raises(SystemExit):
        host_setup.main(["start", str(units), "--dry"], is_root=lambda: True, read=read)
    assert "start needs --swappiness-after" in capsys.readouterr().err
    commands = []
    swappiness = tmp_path / "swappiness"
    swappiness.write_text("30\n")
    counts = host_setup.apply(
        "stop",
        ["secret-a.service", "secret-b.timer"],
        60,
        run=lambda command: commands.append(command) or 0,
        swappiness=swappiness,
    )
    assert commands == [
        ["systemctl", "stop", "secret-a.service"],
        ["systemctl", "stop", "secret-b.timer"],
        ["swapoff", "-a"],
    ]
    assert swappiness.read_text() == "1\n"
    line = host_setup.record_line("stop", counts)
    assert line.startswith("SETUP stop units=2 done=2 failed=0 swap=off swappiness=1 found_swappiness=30")
    assert "secret" not in line


def test_the_setup_writes_no_file(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    swappiness = tmp_path / "swappiness"
    swappiness.write_text("30\n")
    monkeypatch.setattr(host_setup, "SWAPPINESS", swappiness)
    monkeypatch.setattr(host_setup.apply, "__defaults__", (host_setup._run, swappiness))
    code = host_setup.main(
        ["start", "units", "--swappiness-after", "30"],
        is_root=lambda: True,
        read=lambda path: "stop x.service\n",
        run=lambda command: 0,
    )
    assert code == 0 and sorted(path.name for path in tmp_path.iterdir()) == ["swappiness"]
    assert capsys.readouterr().out.startswith("SETUP start units=1 done=1 failed=0 swap=on swappiness=30")


def test_the_gate_and_sampler_refuse_to_run_as_root(tmp_path, capsys):
    for argv in (["once"], ["sample", str(tmp_path / "s.txt")], ["gate", "n", "u", str(tmp_path / "q.txt"), "--dry"]):
        assert quiet.main(argv, is_root=lambda: True) == 2
    assert "refusing to run as root" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_a_failed_systemctl_call_fails_the_gate_instead_of_passing_it(tmp_path, monkeypatch):
    class Done:
        returncode, stdout = 1, ""

    monkeypatch.setattr(quiet.subprocess, "run", lambda *arguments, **options: Done())
    with pytest.raises(quiet.SystemctlFailed):
        quiet._units(["x-*"], "active")

    def measure(rules, settle_s):
        raise quiet.SystemctlFailed(1)

    quiet_file = tmp_path / "quiet.txt"
    assert quiet.gate("start", quiet.Rules(watch=["x-*"]), quiet_file, 0, 3, 5_859_375, 1.0, measure=measure) == 1
    assert quiet_file.read_text() == "QUIET start FAIL systemctl rc=1\n"
