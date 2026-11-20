# SPDX-License-Identifier: Apache-2.0
"""Quiet a host for a measurement window, and restore it after: a generic root setup, written after the judged runs.

Usage (as root):
  python scripts/host/host_setup.py stop  <units file> [--dry]
  python scripts/host/host_setup.py start <units file> --swappiness-after N [--dry]

``stop`` stops every unit the units file names, turns swap off and sets vm.swappiness to 1. ``start`` starts the
same units again, turns swap on and sets vm.swappiness to ``--swappiness-after``, which is required: give it the
value ``stop`` found. Each prints one line to stdout, with counts only and no unit names, and writes no file:

  SETUP stop units=<n> done=<n> failed=<n> swap=off swappiness=1 found_swappiness=<n> <UTC>

``--dry`` prints the plan, also as counts, runs nothing, and needs no root.

The units file stays outside the repository. Because root reads it, the script opens it once, without following a
link and without blocking, and checks the open file itself: a regular file, owned by root, writable by root only,
at most 64 KiB, in a directory owned by root and writable by root only. So an unprivileged user can neither change
what root stops nor swap the file between the check and the read. The checkout's user can still change this
script itself, so root runs a root-owned copy of it, not the checkout's file. One rule per
line, # comments:
  stop <unit>      an exact systemd unit name, such as name.service or name.timer; no patterns
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.sep)
SWAPPINESS = ROOT / "proc" / "sys" / "vm" / "swappiness"
UNIT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9@._:-]*\.(service|timer|path|socket|target)")
STAMP = "%Y-%m-%dT%H:%M:%SZ"


MAX_UNITS_BYTES = 64 * 1024


def _trusted(info: os.stat_result, what: str, owner: int) -> None:
    if info.st_uid != owner:
        raise ValueError(f"{what} is not owned by root")
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError(f"{what} is writable by its group or by others")


def read_trusted(path: Path, owner: int = 0) -> str:
    """The units file's text, from one open checked on its descriptor; refuse a file root should not trust.

    The open follows no link and does not block, so a symlink or a FIFO is refused at once; the checks read the open
    file, so a file swapped in after them is never read. ``owner`` is root's uid except in tests.
    """
    parent = os.stat(Path(path).parent)
    _trusted(parent, "the units file's directory", owner)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as error:
        raise ValueError(f"the units file cannot be opened safely: {error.strerror}") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("the units file is not a regular file")
        _trusted(info, "the units file", owner)
        if info.st_size > MAX_UNITS_BYTES:
            raise ValueError(f"the units file is over {MAX_UNITS_BYTES} bytes")
        data = os.read(fd, MAX_UNITS_BYTES + 1)
    finally:
        os.close(fd)
    if len(data) > MAX_UNITS_BYTES:
        raise ValueError(f"the units file grew past {MAX_UNITS_BYTES} bytes while it was read")
    return data.decode()


def read_units(text: str) -> list[str]:
    """The exact unit names a units file's text stops; a malformed line or name raises ValueError naming its line."""
    names = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        words = line.split()
        if len(words) != 2 or words[0] != "stop" or not UNIT_NAME.fullmatch(words[1]):
            raise ValueError(f"units file line {number}: not a stop rule with an exact unit name")
        names.append(words[1])
    return names


def _run(command: list[str]) -> int:
    return subprocess.run(command, capture_output=True, check=False).returncode


def apply(action: str, units: list[str], swappiness_after: int, run=_run, swappiness: Path = SWAPPINESS) -> dict:
    """Stop or start the units and set swap; return the counts the record line holds."""
    found = swappiness.read_text().strip()
    verb = "stop" if action == "stop" else "start"
    done = sum(run(["systemctl", verb, unit]) == 0 for unit in units)
    swap_ok = run(["swapoff" if action == "stop" else "swapon", "-a"]) == 0
    value = 1 if action == "stop" else swappiness_after
    swappiness.write_text(f"{value}\n")
    return {
        "units": len(units),
        "done": done,
        "failed": len(units) - done,
        "swap": ("off" if action == "stop" else "on") if swap_ok else "unchanged",
        "swappiness": value,
        "found_swappiness": found if found.isdigit() else "NA",
    }


def record_line(action: str, counts: dict) -> str:
    return (
        f"SETUP {action} units={counts['units']} done={counts['done']} failed={counts['failed']} "
        f"swap={counts['swap']} swappiness={counts['swappiness']} found_swappiness={counts['found_swappiness']} "
        f"{time.strftime(STAMP, time.gmtime())}"
    )


def main(argv: list[str] | None = None, is_root=lambda: os.geteuid() == 0, read=read_trusted, run=_run) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=("stop", "start"))
    parser.add_argument("units_file", type=Path)
    parser.add_argument("--swappiness-after", type=int, help="required for start: the value stop found")
    parser.add_argument("--dry", action="store_true", help="print the plan as counts; run nothing")
    args = parser.parse_args(argv)
    if args.action == "start" and args.swappiness_after is None:
        parser.error("start needs --swappiness-after, the value stop found")
    try:
        # Also in --dry, so a dry pass checks the real file as root would.
        units = read_units(read(args.units_file))
    except (OSError, ValueError) as error:
        print(f"setup: error: {error}", file=sys.stderr)
        return 2
    if args.dry:
        swap = "off" if args.action == "stop" else "on"
        value = 1 if args.action == "stop" else args.swappiness_after
        print(f"SETUP-PLAN {args.action} units={len(units)} swap={swap} swappiness={value}")
        return 0
    if not is_root():
        print("setup: error: run as root, or with --dry", file=sys.stderr)
        return 2
    counts = apply(args.action, units, args.swappiness_after, run=run)
    print(record_line(args.action, counts))
    return 0 if counts["failed"] == 0 and counts["swap"] != "unchanged" else 1


if __name__ == "__main__":
    sys.exit(main())
