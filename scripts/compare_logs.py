# SPDX-License-Identifier: Apache-2.0
"""Compare a new run's logs with the committed ones, metric by metric, against fixed tolerances.

Usage: python scripts/compare_logs.py <new_runs_dir> [--against runs/]

For each kind and precision with an eval or benchmark log in the new run, the newest
clean log there is compared with the newest clean log in ``--against``. Each metric
prints one line:

    COMPARE <kind> <precision> <metric> published=.. new=.. delta=.. tol=.. OK|OUT

Metrics in ``REPORTED_ONLY`` end in ``REPORTED`` instead: they are shown, not judged.
The last line is ``COMPARE DONE PASS`` (exit 0) or ``COMPARE DONE FAIL <n>`` (exit 1),
where n counts the OUT lines.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import generate_table

# The agreement a re-measure must reach for each judged metric. "points" is an
# absolute difference in AP points (0-100 scale), per precision; "relative" is a
# fraction of the published value.
TOLERANCES: dict[str, dict] = {
    "ap": {"points": {"fp32": 0.10, "fp16": 0.10, "int8": 0.50}},
    "ap50": {"points": {"fp32": 0.10, "fp16": 0.10, "int8": 0.50}},
    "engine_p50_ms": {"relative": 0.10},
    "end_to_end_p50_ms": {"relative": 0.15},
    "avg_power_w": {"relative": 0.10},
}
REPORTED_ONLY = ("engine_p99_ms", "end_to_end_p99_ms", "system_ram_rise_mib")
KINDS = ("eval", "benchmark")
# Floating-point slack, so that a difference equal to its tolerance is within it.
EPSILON = 1e-9


def _points(key: str) -> Callable[[dict], object]:
    return lambda results: results[key] * generate_table.POINTS if _is_number(results.get(key)) else results.get(key)


METRICS: dict[str, tuple[tuple[str, Callable[[dict], object]], ...]] = {
    "eval": (("ap", _points("ap")), ("ap50", _points("ap50"))),
    "benchmark": (
        ("engine_p50_ms", lambda results: results["engine"]["p50_ms"]),
        ("end_to_end_p50_ms", lambda results: results["end_to_end"]["p50_ms"]),
        ("avg_power_w", lambda results: results["avg_power_w"]),
        ("engine_p99_ms", lambda results: results["engine"]["p99_ms"]),
        ("end_to_end_p99_ms", lambda results: results["end_to_end"]["p99_ms"]),
        ("system_ram_rise_mib", lambda results: results["system_ram_rise_mib"]),
    ),
}


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _shown(value: object) -> str:
    return f"{value:.2f}" if _is_number(value) else str(value)


def judge(metric: str, precision: str, published: object, new: object) -> tuple[str, str, str]:
    """Return the delta text, the tolerance text and the verdict for one metric."""
    numbers = _is_number(published) and _is_number(new)
    if metric in REPORTED_ONLY:
        return (f"{new - published:+.2f}" if numbers else "n/a"), "-", "REPORTED"
    rule = TOLERANCES[metric]
    if "points" in rule:
        tolerance = rule["points"][precision]
        if not numbers:
            return "n/a", f"{tolerance:.2f}", "OUT"
        delta = new - published
        return f"{delta:+.2f}", f"{tolerance:.2f}", "OK" if abs(delta) <= tolerance + EPSILON else "OUT"
    tolerance = rule["relative"]
    tolerance_text = f"{tolerance * 100:.0f}%"
    if not numbers or published == 0:
        return "n/a", tolerance_text, "OUT"
    delta = (new - published) / abs(published)
    return f"{delta * 100:+.1f}%", tolerance_text, "OK" if abs(delta) <= tolerance + EPSILON else "OUT"


def compare(new_dir: Path, against_dir: Path) -> list[str]:
    """Return the COMPARE lines, the final DONE line included."""
    new_logs = generate_table.load_logs(new_dir)
    published_logs = generate_table.load_logs(against_dir)
    lines, failures = [], 0
    for kind in KINDS:
        for precision in generate_table.PRECISIONS:
            new = generate_table.latest(new_logs, kind, precision)
            if new is None:
                continue
            published = generate_table.latest(published_logs, kind, precision)
            for metric, read in METRICS[kind]:
                new_value = read(new["results"])
                published_value = read(published["results"]) if published else "missing"
                delta, tolerance, verdict = judge(metric, precision, published_value, new_value)
                failures += verdict == "OUT"
                lines.append(
                    f"COMPARE {kind} {precision} {metric} published={_shown(published_value)} "
                    f"new={_shown(new_value)} delta={delta} tol={tolerance} {verdict}"
                )
    if not lines:
        failures += 1
        lines.append("COMPARE - - logs published=- new=none delta=n/a tol=- OUT")
    lines.append("COMPARE DONE PASS" if failures == 0 else f"COMPARE DONE FAIL {failures}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("new_runs_dir", type=Path, help="directory holding the new run's logs")
    parser.add_argument("--against", type=Path, default=generate_table.RUNS_DIR,
                        help="directory of the committed logs (default: runs/)")
    args = parser.parse_args(argv)
    lines = compare(args.new_runs_dir, args.against)
    print("\n".join(lines))
    return 0 if lines[-1] == "COMPARE DONE PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
