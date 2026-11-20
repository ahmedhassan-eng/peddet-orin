# SPDX-License-Identifier: Apache-2.0
"""Recompute a benchmark log's timing and power figures from its raw side file, and compare them with the log.

Usage: python scripts/recompute_stats.py <benchmark log> [...]

A benchmark log written from the 0.1.1 release commit on (the committed logs, written at 8427133, carry none) names a
raw side file beside it (``results.raw_file``, with ``results.raw_sha256``):
every timed iteration's stage and end-to-end times in ms, to 4 decimals, and the VDD_IN readings in mW. This script
recomputes the engine, end-to-end and per-stage percentiles and FPS, the per-repeat p50s and the mean power, and
prints one line per figure:

    RECOMPUTE <log> <figure> logged=.. recomputed=.. OK|DIFF

It compares at the precision the README publishes: ms to 2 decimals, FPS to 1, watts to 2, since the raw times
are rounded to 4 decimals. The last line is ``RECOMPUTE DONE PASS`` (exit 0) or ``RECOMPUTE DONE FAIL <n>`` (exit 1).
A log that names no raw side file, as every committed log, is refused with one line and exit 2.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PERCENTILES = (50, 90, 99)
MS_PER_S = 1000.0
MW_PER_W = 1000.0
# The published precision of each kind of figure, as an allowed absolute difference.
TOLERANCE = {"ms": 0.006, "fps": 0.06, "w": 0.006}


def summary(ms: list[float]) -> dict[str, float]:
    """Percentiles in ms and FPS, as the benchmark computes them from seconds."""
    samples = np.array(ms)
    result = {f"p{q}_ms": float(np.percentile(samples, q)) for q in PERCENTILES}
    result["fps"] = float(MS_PER_S / samples.mean())
    return result


def recompute(raw: dict) -> dict:
    """The figures the log publishes, recomputed from the raw side file."""
    columns = raw["columns"]
    rows = [row for repeat in raw["repeats"] for row in repeat]

    def column(name: str, source: list) -> list[float]:
        return [row[columns.index(name)] for row in source]

    figures = {name: summary(column(name, rows)) for name in ("engine", "end_to_end")}
    figures["stages"] = {name: summary(column(name, rows)) for name in columns if name != "end_to_end"}
    for name in ("engine", "end_to_end"):
        figures[f"{name}_p50_ms_per_repeat"] = [summary(column(name, repeat))["p50_ms"] for repeat in raw["repeats"]]
    power = raw["vdd_in_mw"]
    figures["avg_power_w"] = sum(power) / len(power) / MW_PER_W if power else None
    return figures


def _pairs(logged: dict, figures: dict) -> list[tuple[str, float, float, str]]:
    pairs = []
    for name in ("engine", "end_to_end"):
        for key, value in figures[name].items():
            pairs.append((f"{name}.{key}", logged[name][key], value, "fps" if key == "fps" else "ms"))
    for stage, values in figures["stages"].items():
        for key, value in values.items():
            pairs.append(
                (f"stages.{stage}.{key}", logged["stages"][stage][key], value, "fps" if key == "fps" else "ms")
            )
    for name in ("engine", "end_to_end"):
        key = f"{name}_p50_ms_per_repeat"
        for index, value in enumerate(figures[key]):
            pairs.append((f"{key}[{index}]", logged[key][index], value, "ms"))
    pairs.append(("avg_power_w", logged["avg_power_w"], figures["avg_power_w"], "w"))
    return pairs


class Refused(Exception):
    """A log this script cannot recompute, with the one line that says why."""


def compare(log_path: Path) -> list[str]:
    """The RECOMPUTE lines for one log, ending with its DONE line; Refused if the log names no raw side file."""
    log = json.loads(Path(log_path).read_text())
    results = log["results"]
    if "raw_file" not in results:
        raise Refused(
            f"{Path(log_path).name} names no raw_file: it was written before the code that writes one "
            "(the committed logs carry none)"
        )
    raw_path = Path(log_path).parent / results["raw_file"]
    if not raw_path.is_file():
        raise Refused(f"{Path(log_path).name} names raw_file {results['raw_file']}, which is not beside it")
    data = raw_path.read_bytes()
    if hashlib.sha256(data).hexdigest() != results["raw_sha256"]:
        return [
            f"RECOMPUTE {Path(log_path).name} raw_sha256 logged=match recomputed=differs DIFF",
            "RECOMPUTE DONE FAIL 1",
        ]
    figures = recompute(json.loads(data))
    if figures["avg_power_w"] is None:
        raise Refused(f"{Path(log_path).name}'s raw file {results['raw_file']} holds no power readings")
    lines, failures = [], 0
    for name, logged, recomputed, kind in _pairs(results, figures):
        ok = isinstance(logged, (int, float)) and recomputed is not None and abs(logged - recomputed) <= TOLERANCE[kind]
        failures += not ok
        verdict = "OK" if ok else "DIFF"
        lines.append(f"RECOMPUTE {Path(log_path).name} {name} logged={logged} recomputed={recomputed:.4f} {verdict}")
    lines.append("RECOMPUTE DONE PASS" if failures == 0 else f"RECOMPUTE DONE FAIL {failures}")
    return lines


def main(argv: list[str] | None = None) -> int:
    paths = sys.argv[1:] if argv is None else argv
    if not paths:
        print("usage: python scripts/recompute_stats.py <benchmark log> [...]", file=sys.stderr)
        return 2
    status = 0
    for path in paths:
        try:
            lines = compare(Path(path))
        except Refused as refusal:
            print(f"recompute_stats: error: {refusal}", file=sys.stderr)
            status = 2
            continue
        print("\n".join(lines))
        status = status or int(lines[-1] != "RECOMPUTE DONE PASS")
    return status


if __name__ == "__main__":
    sys.exit(main())
