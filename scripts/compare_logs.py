# SPDX-License-Identifier: Apache-2.0
"""Compare a re-measure with the published table, metric by metric, against fixed tolerances.

Usage: python scripts/compare_logs.py <runs dir>, for example runs after a re-measure

The reference is what the README's results table is built from: with judged runs under ``runs-repeat/``, their
median; without them, the newest clean logs in ``runs/``. Logs that are committed under ``runs/`` or
``runs-repeat/`` are left out of the new set, so a fresh clone's ``runs/`` can be compared after a re-measure and
the committed logs never stand in for missing ones. The rest must come from one commit and hold an eval and a
benchmark log for every precision; for each, the newest is compared, and it must be clean and run with the same
runtime and iteration counts as the reference. Each metric prints one line:

    COMPARE <kind> <precision> <metric> published=.. new=.. delta=.. tol=.. OK|OUT

Metrics in ``REPORTED_ONLY`` end in ``REPORTED`` instead: they are shown, not judged. A kind and precision with no
new log prints one ``COMPARE <kind> <precision> log published=- new=missing delta=n/a tol=- OUT`` line; a dirty
newest log, a different runtime or iteration count, or new logs from more than one commit print their own OUT line.
The last line is ``COMPARE DONE PASS`` (exit 0) or ``COMPARE DONE FAIL <n>`` (exit 1), where n counts the OUT lines.
"""

from __future__ import annotations

import argparse
import json
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


def reference() -> dict:
    """The logs the README's results table is built from, keyed by kind and precision."""
    return generate_table.headline()[0]


def _identity(log: dict) -> tuple:
    return (log.get("kind"), log.get("precision"), log.get("commit"), log.get("start_utc"), log.get("end_utc"))


def committed_identities() -> set[tuple]:
    """The identity of every log committed under runs/ and runs-repeat/."""
    logs = generate_table.load_logs(generate_table.RUNS_DIR)
    for directory in sorted(path for path in generate_table.REPEATS_DIR.glob("*") if path.is_dir()):
        logs += generate_table.load_logs(directory)
    return {_identity(log) for log in logs}


def _newest(logs: list[dict], kind: str, precision: str) -> dict | None:
    matching = [log for log in logs if log["kind"] == kind and log["precision"] == precision]
    return max(matching, key=lambda log: log["end_utc"]) if matching else None


def _setup(log: dict) -> str:
    """The runtime and iteration counts a log ran with, as one space-free token."""
    counts = json.dumps(log.get("iterations"), sort_keys=True, separators=(",", ":"))
    return f"{log['results'].get('runtime')}/{counts}"


def compare(new_dir: Path, published_logs: dict | None = None, committed: set[tuple] | None = None) -> list[str]:
    """Return the COMPARE lines, the final DONE line included; ``published_logs`` defaults to ``reference()`` and
    ``committed`` to ``committed_identities()``."""
    published_logs = reference() if published_logs is None else published_logs
    committed = committed_identities() if committed is None else committed
    found = generate_table.load_logs(new_dir)
    new_logs = [log for log in found if _identity(log) not in committed]
    if len(new_logs) < len(found):
        print(f"compare_logs: leaving out {len(found) - len(new_logs)} committed logs", file=sys.stderr)
    lines, failures = [], 0
    commits = sorted({log["commit"] for log in new_logs if log["kind"] in KINDS})
    if len(commits) > 1:
        failures += 1
        lines.append(f"COMPARE - - commit published=- new={','.join(c[:7] for c in commits)} delta=n/a tol=- OUT")
    for kind in KINDS:
        for precision in generate_table.PRECISIONS:
            new = _newest(new_logs, kind, precision)
            if new is None:
                failures += 1
                lines.append(f"COMPARE {kind} {precision} log published=- new=missing delta=n/a tol=- OUT")
                continue
            if new.get("dirty") is not False:
                failures += 1
                lines.append(f"COMPARE {kind} {precision} log published=- new=dirty delta=n/a tol=- OUT")
                continue
            published = published_logs.get((kind, precision))
            if published and _setup(new) != _setup(published):
                failures += 1
                shown = f"published={_setup(published)} new={_setup(new)}"
                lines.append(f"COMPARE {kind} {precision} setup {shown} delta=n/a tol=- OUT")
            for metric, read in METRICS[kind]:
                new_value = read(new["results"])
                published_value = read(published["results"]) if published else "missing"
                delta, tolerance, verdict = judge(metric, precision, published_value, new_value)
                failures += verdict == "OUT"
                lines.append(
                    f"COMPARE {kind} {precision} {metric} published={_shown(published_value)} "
                    f"new={_shown(new_value)} delta={delta} tol={tolerance} {verdict}"
                )
    lines.append("COMPARE DONE PASS" if failures == 0 else f"COMPARE DONE FAIL {failures}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("new_runs_dir", type=Path, help="directory holding the re-measure's logs")
    args = parser.parse_args(argv)
    try:
        lines = compare(args.new_runs_dir)
    except generate_table.TableError as error:
        print(f"compare_logs: {error}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0 if lines[-1] == "COMPARE DONE PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
