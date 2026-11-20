# SPDX-License-Identifier: Apache-2.0
"""A generic QUIET gate and host sampler any host can run, written after the judged runs; no judged run used it.

Usage:
  python scripts/host/quiet.py sample <out file> [--interval S]   one SYS line per interval until SIGTERM
  python scripts/host/quiet.py once                                one SYS line to stdout
  python scripts/host/quiet.py gate <name> <units file> <quiet file>
                                   [--settle S] [--tries N] [--mem-min-kb K]
                                   [--load-bound B] [--dry]

The sampler reads the kernel's meminfo and vmstat files and every thermal zone that reports a temperature, and
writes numbers only: ``SYS <UTC> mem_available_kb=.. swap_used_kb=.. pswpin=.. pswpout=.. <zone>_mc=..``.
Its file's first line names this script's sha256.

The gate runs before each benchmark. Each try takes a sample, waits ``--settle`` seconds, takes another, and checks:
no unit the units file watches is active apart from those it allows; no scope it budgets used more CPU over the
settle than its budget; no busy service it lists is active; 0 kB of swap in use; no page swapped out during the
settle; available memory at least ``--mem-min-kb``; a one-minute load below ``--load-bound``. It appends one line
per try to the quiet file in the committed form, counts only, with no unit names:

  QUIET <name> try=<n> held_active=.. allowed_timers=.. scopes_over=.. busy_services=.. load1=.. bound=.. \\
    swap_used_kb=.. pswpout_delta=.. mem_available_kb=.. mem_min_kb=.. <zone>_mc=.. fail=<terms or none> <UTC>

then ``QUIET <name> PASS`` and exit 0, or after ``--tries`` failing tries ``QUIET <name> FAIL terms=..`` and exit 1.
``--dry`` prints what the gate would check and runs nothing. It refuses to run as root: it opens its output files
by name, so it is never a root path.

The units file stays outside the repository, since its unit names are the host's. One rule per line, # comments:
  watch <glob>          a systemd unit pattern that must not be active (timers, paths, services)
  allow <glob>          a watched unit that may stay active; counted as allowed_timers
  busy <glob>           a service pattern that must not be active
  scope <glob> <ms>     a running scope pattern, allowed at most <ms> of CPU over the settle
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# System paths, built from parts so that the source names no absolute path.
ROOT = Path(os.sep)
MEMINFO = ROOT / "proc" / "meminfo"
VMSTAT = ROOT / "proc" / "vmstat"
LOADAVG = ROOT / "proc" / "loadavg"
THERMAL = ROOT / "sys" / "class" / "thermal"
STAMP = "%Y-%m-%dT%H:%M:%SZ"


def _read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def _field(text: str, key: str) -> str:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[0] == key and parts[1].isdigit():
            return parts[1]
    return "NA"


def zones(thermal: Path = THERMAL) -> list[tuple[str, Path]]:
    """Each thermal zone that reads a temperature now, as (type, temp file), in zone order."""
    found = []
    for zone in sorted(thermal.glob("thermal_zone*"), key=lambda path: int(path.name.removeprefix("thermal_zone"))):
        if _read(zone / "temp").strip().isdigit():
            found.append((_read(zone / "type").strip() or "NA", zone / "temp"))
    return found


def sample(meminfo: Path = MEMINFO, vmstat: Path = VMSTAT, temps: list[tuple[str, Path]] | None = None) -> str:
    """One SYS line, numbers only."""
    memory, counters = _read(meminfo), _read(vmstat)
    total, free = _field(memory, "SwapTotal:"), _field(memory, "SwapFree:")
    used = str(int(total) - int(free)) if total.isdigit() and free.isdigit() else "NA"
    readings = []
    for name, path in zones() if temps is None else temps:
        value = _read(path).strip()
        readings.append(f"{name}_mc={value if value.isdigit() else 'NA'}")
    return (
        f"SYS {time.strftime(STAMP, time.gmtime())} mem_available_kb={_field(memory, 'MemAvailable:')} "
        f"swap_used_kb={used} pswpin={_field(counters, 'pswpin')} pswpout={_field(counters, 'pswpout')} "
        + " ".join(readings)
    ).rstrip()


def values(line: str) -> dict[str, str]:
    """The key=value pairs of a SYS line."""
    return dict(part.split("=", 1) for part in line.split() if "=" in part)


@dataclass
class Rules:
    watch: list[str] = field(default_factory=list)
    allow: list[str] = field(default_factory=list)
    busy: list[str] = field(default_factory=list)
    scopes: list[tuple[str, int]] = field(default_factory=list)


def read_rules(path: Path) -> Rules:
    """Parse a units file; an unknown rule or a malformed line stops with ValueError naming its line."""
    rules = Rules()
    for number, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        words = line.split()
        if words[0] in ("watch", "allow", "busy") and len(words) == 2:
            getattr(rules, words[0]).append(words[1])
        elif words[0] == "scope" and len(words) == 3 and words[2].isdigit():
            rules.scopes.append((words[1], int(words[2])))
        else:
            raise ValueError(f"units file line {number}: not a rule: {raw!r}")
    return rules


class SystemctlFailed(Exception):
    """systemctl list-units exited non-zero, so its unit lists cannot be trusted: the gate fails, not passes."""

    def __init__(self, returncode: int) -> None:
        super().__init__(f"systemctl rc={returncode}")
        self.returncode = returncode


def _units(patterns: list[str], state: str) -> list[str]:
    if not patterns:
        return []
    done = subprocess.run(
        ["systemctl", "list-units", f"--state={state}", "--no-legend", "--plain", *patterns],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise SystemctlFailed(done.returncode)
    return [line.split()[0] for line in done.stdout.splitlines() if line.split()]


def _cpu_ns(unit: str) -> int | None:
    done = subprocess.run(
        ["systemctl", "show", "-p", "CPUUsageNSec", "--value", unit], capture_output=True, text=True, check=False
    )
    text = done.stdout.strip()
    return int(text) if text.isdigit() else None


def _load1(loadavg: Path = LOADAVG) -> str:
    words = _read(loadavg).split()
    return f"{float(words[0]):.2f}" if words else "NA"


def judge(terms: dict, mem_min_kb: int, load_bound: float) -> list[str]:
    """The failing terms of one try, by name."""
    fails = []
    for name in ("held_active", "scopes_over", "busy_services"):
        if terms[name]:
            fails.append(name)
    if terms["swap_used_kb"] != "0":
        fails.append("swap_used")
    if terms["pswpout_delta"] != "0":
        fails.append("pswpout")
    memory = terms["mem_available_kb"]
    if not (memory.isdigit() and int(memory) >= mem_min_kb):
        fails.append("mem_available")
    try:
        if not float(terms["load1"]) < load_bound:
            fails.append("load1")
    except ValueError:
        fails.append("load1")
    return fails


def one_try(rules: Rules, settle_s: float, units=_units, cpu_ns=_cpu_ns, take=sample, load1=_load1) -> dict:
    """Measure one try's terms: the counts and numbers its QUIET line records."""
    before = {
        unit: cpu_ns(unit)
        for _, unit in ((pattern, unit) for pattern, _ in rules.scopes for unit in units([pattern], "running"))
    }
    first = values(take())
    time.sleep(settle_s)
    second_line = take()
    second = values(second_line)
    active = units(rules.watch, "active")
    allowed = [unit for unit in active if any(fnmatch.fnmatchcase(unit, pattern) for pattern in rules.allow)]
    held = [unit for unit in active if unit not in allowed]
    over = []
    for pattern, budget_ms in rules.scopes:
        for unit in units([pattern], "running"):
            start, end = before.get(unit), cpu_ns(unit)
            if start is None or end is None or (end - start) / 1_000_000 > budget_ms:
                over.append(unit)
    busy = units(rules.busy, "active")
    pout0, pout1 = first.get("pswpout", "NA"), second.get("pswpout", "NA")
    delta = str(int(pout1) - int(pout0)) if pout0.isdigit() and pout1.isdigit() else "NA"
    temps = {key: value for key, value in second.items() if key.endswith("_mc")}
    return {
        "held_active": len(held),
        "allowed_timers": len(allowed),
        "scopes_over": len(over),
        "busy_services": len(busy),
        "load1": load1(),
        "swap_used_kb": second.get("swap_used_kb", "NA"),
        "pswpout_delta": delta,
        "mem_available_kb": second.get("mem_available_kb", "NA"),
        "temps": temps,
    }


def quiet_line(name: str, number: int, terms: dict, fails: list[str], mem_min_kb: int, load_bound: float) -> str:
    temps = "".join(f"{key}={value} " for key, value in terms["temps"].items())
    return (
        f"QUIET {name} try={number} held_active={terms['held_active']} allowed_timers={terms['allowed_timers']} "
        f"scopes_over={terms['scopes_over']} busy_services={terms['busy_services']} load1={terms['load1']} "
        f"bound={load_bound} swap_used_kb={terms['swap_used_kb']} pswpout_delta={terms['pswpout_delta']} "
        f"mem_available_kb={terms['mem_available_kb']} mem_min_kb={mem_min_kb} {temps}"
        f"fail={','.join(fails) or 'none'} {time.strftime(STAMP, time.gmtime())}"
    )


def gate(
    name: str,
    rules: Rules,
    quiet_file: Path,
    settle_s: float,
    tries: int,
    mem_min_kb: int,
    load_bound: float,
    measure=one_try,
) -> int:
    """Run up to ``tries`` tries; append each try's line and the verdict to ``quiet_file``; 0 on PASS, else 1."""
    fails: list[str] = []
    with Path(quiet_file).open("a") as out:
        for number in range(1, tries + 1):
            try:
                terms = measure(rules, settle_s)
            except SystemctlFailed as failure:
                out.write(f"QUIET {name} FAIL {failure}\n")
                return 1
            fails = judge(terms, mem_min_kb, load_bound)
            out.write(quiet_line(name, number, terms, fails, mem_min_kb, load_bound) + "\n")
            if not fails:
                out.write(f"QUIET {name} PASS\n")
                return 0
        out.write(f"QUIET {name} FAIL terms={','.join(fails)}\n")
    return 1


def plan(name: str, rules: Rules, settle_s: float, tries: int, mem_min_kb: int, load_bound: float) -> list[str]:
    """What the gate would check, for --dry: counts of rules, never their names."""
    return [
        f"QUIET-PLAN {name} tries={tries} settle_s={settle_s} watch={len(rules.watch)} allow={len(rules.allow)} "
        f"busy={len(rules.busy)} scopes={len(rules.scopes)} mem_min_kb={mem_min_kb} load_bound={load_bound}",
        "QUIET-PLAN terms: held_active scopes_over busy_services swap_used pswpout mem_available load1",
    ]


def run_sampler(out_path: Path, interval_s: float) -> None:
    out = Path(out_path).open("a", buffering=1)
    if out.tell() == 0:
        out.write(f"SYSMON sha256={hashlib.sha256(Path(__file__).read_bytes()).hexdigest()} interval_s={interval_s}\n")

    def stop(signum, frame):
        out.write(sample() + "\n")
        out.close()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    while True:
        out.write(sample() + "\n")
        time.sleep(interval_s)


def main(argv: list[str] | None = None, is_root=lambda: os.geteuid() == 0) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    sampler = commands.add_parser("sample", help="write one SYS line per interval to a file until SIGTERM")
    sampler.add_argument("out", type=Path)
    sampler.add_argument("--interval", type=float, default=1.0)
    commands.add_parser("once", help="print one SYS line")
    gater = commands.add_parser("gate", help="run the QUIET gate and append its lines to a quiet file")
    gater.add_argument("name")
    gater.add_argument("units_file", type=Path)
    gater.add_argument("quiet_file", type=Path)
    gater.add_argument("--settle", type=float, default=120.0)
    gater.add_argument("--tries", type=int, default=10)
    gater.add_argument("--mem-min-kb", type=int, default=5859375)
    gater.add_argument("--load-bound", type=float, default=1.0)
    gater.add_argument("--dry", action="store_true", help="print what the gate would check; run nothing")
    args = parser.parse_args(argv)
    if is_root():
        print("quiet: error: refusing to run as root; run it as the measuring user", file=sys.stderr)
        return 2
    if args.command == "once":
        print(sample())
        return 0
    if args.command == "sample":
        run_sampler(args.out, args.interval)
        return 0
    try:
        rules = read_rules(args.units_file)
    except (OSError, ValueError) as error:
        print(f"quiet: error: {error}", file=sys.stderr)
        return 2
    if args.dry:
        print("\n".join(plan(args.name, rules, args.settle, args.tries, args.mem_min_kb, args.load_bound)))
        return 0
    return gate(args.name, rules, args.quiet_file, args.settle, args.tries, args.mem_min_kb, args.load_bound)


if __name__ == "__main__":
    sys.exit(main())
