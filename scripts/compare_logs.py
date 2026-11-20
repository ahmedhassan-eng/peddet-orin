# SPDX-License-Identifier: Apache-2.0
"""Compare a re-measure with the published table, metric by metric, against fixed tolerances.

Usage: python scripts/compare_logs.py <runs dir>, for example runs after a re-measure

The reference is what the README's results table is built from: with judged runs under ``runs-repeat/``, their median;
without them, the newest clean logs in ``runs/``. Logs that are committed under ``runs/``, ``runs-repeat/`` or
``runs-superseded/`` are left out of the new set, so a fresh clone's ``runs/`` can be compared after a re-measure and
the committed logs never stand in for missing ones. The rest must come from one commit and hold an eval and a benchmark
log for every precision; for each, the newest is compared, and it must be clean and run with the same runtime and
iteration counts as the reference. Each metric prints one line:

    COMPARE <kind> <precision> <metric> published=.. new=.. delta=.. tol=.. OK|OUT

Metrics in ``REPORTED_ONLY`` end in ``REPORTED`` instead: they are shown, not judged. So does every line for a
precision the reference predates, until the reference has it; the judged runs carry all four. The deployment-style
benchmark (``benchmark_deploy``) is never judged: when the new logs carry it, each precision prints one REPORTED line
per metric in ``DEPLOY_METRICS``, preceded by a REPORTED line when its newest log is dirty (then no metrics), or has
another commit or another setup than the newest main benchmark of its precision. A kind and precision with no
new log prints one ``COMPARE <kind> <precision> log published=- new=missing delta=n/a tol=- OUT`` line; a dirty
newest log, a different runtime or iteration count, new logs from more than one commit, a different torch,
TensorRT or CUDA version, and a missing or different engine hash print their own OUT line.
The last line is ``COMPARE DONE PASS`` (exit 0) or ``COMPARE DONE FAIL <n>`` (exit 1), where n counts the OUT lines.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import generate_table

# The agreement a re-measure must reach for each judged metric. "points" is an
# absolute difference in AP points (0-100 scale), per precision; "relative" is a
# fraction of the published value.
TOLERANCES: dict[str, dict] = {
    "ap": {"points": {"fp32": 0.10, "fp32-trt": 0.10, "fp16": 0.10, "int8": 0.50}},
    "ap50": {"points": {"fp32": 0.10, "fp32-trt": 0.10, "fp16": 0.10, "int8": 0.50}},
    "engine_p50_ms": {"relative": 0.10},
    "end_to_end_p50_ms": {"relative": 0.15},
    "avg_power_w": {"relative": 0.10},
}
REPORTED_ONLY = ("engine_p99_ms", "end_to_end_p99_ms", "system_ram_rise_mib")
KINDS = ("eval", "benchmark")
# The deployment-style benchmark is not judged by R-11: each of its metrics prints one REPORTED line when the new
# logs carry it, and a missing one is not a failure. A deploy log from another commit than the rest still fails the
# one-commit check, as a log of any kind does.
DEPLOY_KIND = generate_table.DEPLOY
DEPLOY_METRICS: tuple[tuple[str, Callable[[dict], object]], ...] = (
    ("end_to_end_p50_ms", lambda results: results["end_to_end"]["p50_ms"]),
    ("end_to_end_p90_ms", lambda results: results["end_to_end"]["p90_ms"]),
    ("end_to_end_p99_ms", lambda results: results["end_to_end"]["p99_ms"]),
    ("end_to_end_fps", lambda results: results["end_to_end"]["fps"]),
    ("avg_power_w", lambda results: results["avg_power_w"]),
    ("end_to_end_energy_per_frame_j", generate_table.energy_per_frame_j),
)
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


# The directories whose committed logs never stand in for a re-measure.
COMMITTED_DIRS = ("runs", "runs-attempts", "runs-repeat", "runs-superseded")
# git with no system or global configuration, no pager and no prompt, so the committed set depends on the checkout only.
GIT_ENV = {
    "PATH": os.environ.get("PATH", os.defpath),
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_PAGER": "cat",
    "GIT_TERMINAL_PROMPT": "0",
    "LC_ALL": "C",
}
NO_GIT = "not a git checkout, so the committed logs cannot be told apart from a re-measure's; run it in a clone"


def committed_identities(root: Path | None = None) -> set[tuple]:
    """The identity of every log committed at HEAD under runs/, runs-repeat/ and runs-superseded/, read from git.

    A re-measure writes its logs into runs/, beside the committed ones, so only git can tell which are committed: a
    log that is not in HEAD's tree is new, wherever it lies. Outside a git checkout this stops with ``TableError``."""
    root = generate_table.REPO_ROOT if root is None else root
    if not (root / ".git").exists():
        raise generate_table.TableError(NO_GIT)
    listed = subprocess.run(
        ["git", "ls-tree", "-r", "-z", "--name-only", "HEAD", "--", *COMMITTED_DIRS],
        cwd=root,
        env=GIT_ENV,
        capture_output=True,
        check=True,
    ).stdout.decode()
    names = [name for name in listed.split("\0") if name.endswith(".json")]
    if not names:
        return set()
    batch = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=root,
        input="".join(f"HEAD:{name}\n" for name in names).encode(),
        env=GIT_ENV,
        capture_output=True,
        check=True,
    ).stdout
    identities, offset = set(), 0
    for _ in names:
        header_end = batch.index(b"\n", offset)
        size = int(batch[offset:header_end].split()[2])
        body = batch[header_end + 1 : header_end + 1 + size]
        offset = header_end + 1 + size + 1
        identities.add(_identity(json.loads(body)))
    # A committed log edited in the work tree is still a committed log, not a re-measure.
    for name in names:
        path = root / name
        if path.is_file():
            identities.add(_identity(json.loads(path.read_text())))
    return identities


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
    # Every kind counts here, the build, deployment-style and other logs included: a re-measure is one commit.
    commits = sorted({log["commit"] for log in new_logs})
    if len(commits) > 1:
        failures += 1
        lines.append(f"COMPARE - - commit published=- new={','.join(c[:7] for c in commits)} delta=n/a tol=- OUT")
    for kind in KINDS:
        for precision in generate_table.TABLE_PRECISIONS:
            new = _newest(new_logs, kind, precision)
            published = published_logs.get((kind, precision))
            # A precision the reference predates is reported when new logs carry it and is
            # judged once the reference has it.
            judged = precision in generate_table.PRECISIONS or published is not None
            if not judged and new is None:
                continue
            if new is None:
                failures += 1
                lines.append(f"COMPARE {kind} {precision} log published=- new=missing delta=n/a tol=- OUT")
                continue
            if new.get("dirty") is not False:
                failures += judged
                lines.append(
                    f"COMPARE {kind} {precision} log published=- new=dirty delta=n/a tol=- "
                    f"{'OUT' if judged else 'REPORTED'}"
                )
                continue
            if published and _setup(new) != _setup(published):
                failures += 1
                shown = f"published={_setup(published)} new={_setup(new)}"
                lines.append(f"COMPARE {kind} {precision} setup {shown} delta=n/a tol=- OUT")
            for metric, read in METRICS[kind]:
                new_value = read(new["results"])
                published_value = read(published["results"]) if published else "missing"
                delta, tolerance, verdict = judge(metric, precision, published_value, new_value)
                verdict = verdict if judged else "REPORTED"
                failures += verdict == "OUT"
                lines.append(
                    f"COMPARE {kind} {precision} {metric} published={_shown(published_value)} "
                    f"new={_shown(new_value)} delta={delta} tol={tolerance} {verdict}"
                )
    path_lines = int8_path_lines(new_logs, published_logs)
    failures += sum(line.endswith(" OUT") for line in path_lines)
    lines += path_lines
    flag_lines = builder_flags_lines(new_logs, published_logs)
    failures += sum(line.endswith(" OUT") for line in flag_lines)
    lines += flag_lines
    hash_lines = engine_hash_lines(new_logs)
    failures += sum(line.endswith(" OUT") for line in hash_lines)
    lines += hash_lines
    environment = environment_lines(new_logs, published_logs)
    failures += sum(line.endswith(" OUT") for line in environment)
    lines += environment
    lines += deploy_lines(new_logs, published_logs)
    lines.append("COMPARE DONE PASS" if failures == 0 else f"COMPARE DONE FAIL {failures}")
    return lines


def int8_path_lines(new_logs: list[dict], published_logs: dict) -> list[str]:
    """One line for the INT8 build's calibration path: the newest INT8 build log must take the reference's path, so a
    fallback build is never judged against an entropy-calibrated reference. No line when the reference has no
    INT8 build log."""
    published = published_logs.get(("build", "int8"))
    if published is None:
        return []
    new = _newest(new_logs, "build", "int8")
    if new is not None and new.get("dirty") is not False:
        return ["COMPARE build int8 int8_path published=- new=dirty delta=n/a tol=- OUT"]
    want = published["results"].get("int8_path")
    got = new["results"].get("int8_path") if new else "missing"
    verdict = "OK" if got == want else "OUT"
    return [f"COMPARE build int8 int8_path published={want} new={got} delta=n/a tol=- {verdict}"]


def builder_flags_lines(new_logs: list[dict], published_logs: dict) -> list[str]:
    """One line per TensorRT precision whose reference has a build log: the newest build log of that precision must
    record the reference's builder flags, so an engine built another way is never judged against it. A reference
    that records no flags gives a REPORTED line, not judged."""
    lines = []
    for precision in generate_table.TABLE_PRECISIONS:
        published = published_logs.get(("build", precision))
        if published is None:
            continue
        new = _newest(new_logs, "build", precision)
        if new is not None and new.get("dirty") is not False:
            lines.append(f"COMPARE build {precision} builder_flags published=- new=dirty delta=n/a tol=- OUT")
            continue
        want = published["results"].get("builder_flags")
        got = new["results"].get("builder_flags", "missing") if new else "missing"
        shown = ",".join(got) if isinstance(got, list) else got
        if want is None:
            lines.append(
                f"COMPARE build {precision} builder_flags published=unrecorded new={shown or 'none'} delta=n/a tol=- "
                "REPORTED"
            )
            continue
        verdict = "OK" if got == want else "OUT"
        lines.append(
            f"COMPARE build {precision} builder_flags published={','.join(want) or 'none'} new={shown or 'none'} "
            f"delta=n/a tol=- {verdict}"
        )
    return lines


# The environment facts a re-measure is checked against the reference's; a torch, TensorRT or CUDA difference is OUT.
ENVIRONMENT_FIELDS = ("torch", "tensorrt", "cuda")


def engine_hash_lines(new_logs: list[dict]) -> list[str]:
    """One line per TensorRT precision and kind: the newest eval and benchmark log must name the engine its
    precision's newest build log wrote, so an engine from another build is never judged as this one. A log or build
    without a hash gives an OUT line; a dirty one gives a REPORTED line, since the metric and build lines fail it."""
    lines = []
    for precision in generate_table.TABLE_PRECISIONS:
        build = _newest(new_logs, "build", precision)
        for kind in KINDS:
            new = _newest(new_logs, kind, precision)
            if new is None or build is None:
                continue
            want = build["results"].get("engine_sha256")
            got = new["results"].get("engine_sha256")
            if new.get("dirty") is not False or build.get("dirty") is not False:
                lines.append(f"COMPARE {kind} {precision} engine_sha256 build=- new=- delta=n/a tol=- REPORTED")
                continue
            if not want or not got:
                lines.append(
                    f"COMPARE {kind} {precision} engine_sha256 build={(want or 'missing')[:12]} "
                    f"new={(got or 'missing')[:12]} delta=n/a tol=- OUT"
                )
                continue
            verdict = "OK" if got == want else "OUT"
            lines.append(
                f"COMPARE {kind} {precision} engine_sha256 build={want[:12]} new={got[:12]} delta=n/a tol=- {verdict}"
            )
    return lines


def environment_lines(new_logs: list[dict], published_logs: dict) -> list[str]:
    """One OUT line per environment fact in ENVIRONMENT_FIELDS whose value in the newest new benchmark log differs
    from the reference benchmark log's: a re-measure on another torch, TensorRT or CUDA is not comparable."""
    new = max((log for log in new_logs if log["kind"] == "benchmark"), key=lambda log: log["end_utc"], default=None)
    reference = next((published_logs[key] for key in sorted(published_logs) if key[0] == "benchmark"), None)
    if new is None or reference is None:
        return []
    lines = []
    for field in ENVIRONMENT_FIELDS:
        want = reference.get("environment", {}).get(field)
        got = new.get("environment", {}).get(field)
        if want != got:
            lines.append(f"COMPARE environment - {field} published={want} new={got} delta=n/a tol=- OUT")
    return lines


def deploy_lines(new_logs: list[dict], published_logs: dict) -> list[str]:
    """One REPORTED line per deployment-style metric, for each precision the new logs carry it for. A dirty newest
    deploy log gives one REPORTED line instead; one from another commit or with another setup than the newest main
    benchmark of its precision gives a REPORTED line saying so before its metrics. None of them is judged."""
    lines = []
    for precision in generate_table.TABLE_PRECISIONS:
        new = _newest(new_logs, DEPLOY_KIND, precision)
        if new is None:
            continue
        if new.get("dirty") is not False:
            lines.append(f"COMPARE {DEPLOY_KIND} {precision} log published=- new=dirty delta=n/a tol=- REPORTED")
            continue
        main = _newest(new_logs, "benchmark", precision)
        if main is None:
            lines.append(f"COMPARE {DEPLOY_KIND} {precision} main published=- new=missing delta=n/a tol=- REPORTED")
        else:
            if main.get("commit") != new.get("commit"):
                shown = f"main={main.get('commit', '-')[:7]} new={new.get('commit', '-')[:7]}"
                lines.append(f"COMPARE {DEPLOY_KIND} {precision} commit {shown} delta=n/a tol=- REPORTED")
            if _setup(main) != _setup(new):
                shown = f"main={_setup(main)} new={_setup(new)}"
                lines.append(f"COMPARE {DEPLOY_KIND} {precision} setup {shown} delta=n/a tol=- REPORTED")
        published = published_logs.get((DEPLOY_KIND, precision))
        for metric, read in DEPLOY_METRICS:
            new_value = read(new["results"])
            published_value = read(published["results"]) if published else "missing"
            numbers = _is_number(published_value) and _is_number(new_value)
            delta = f"{new_value - published_value:+.2f}" if numbers else "n/a"
            lines.append(
                f"COMPARE {DEPLOY_KIND} {precision} {metric} published={_shown(published_value)} "
                f"new={_shown(new_value)} delta={delta} tol=- REPORTED"
            )
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
