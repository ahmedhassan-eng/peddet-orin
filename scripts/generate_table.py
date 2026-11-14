# SPDX-License-Identifier: Apache-2.0
"""Generate the README results block from the JSON logs under ``runs/``.

The block sits between two HTML comment markers in the README. ``--write`` replaces
it, and two smaller generated blocks: the INT8 comparison and the repeat-run check of
PRD R-11 against the further runs under ``runs-repeat/``. A test calls
``readme_matches`` so that the README cannot drift from the logs.
For each precision and log kind, the newest clean log by end time supplies the
numbers: a log made from a dirty tree is skipped, with a line on stderr. Every log
that feeds the block must come from one commit, and each row's evaluation and
benchmark must name the same runtime; otherwise the script exits non-zero.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
# Further full runs at the release code, one directory of logs per run, used only for the R-11 check.
REPEATS_DIR = REPO_ROOT / "runs-repeat"
README = REPO_ROOT / "README.md"
BEGIN_MARKER = "<!-- results:begin -->"
END_MARKER = "<!-- results:end -->"
INT8_MARKERS = ("<!-- int8:begin -->", "<!-- int8:end -->")
REPEAT_MARKERS = ("<!-- repeats:begin -->", "<!-- repeats:end -->")
PRECISIONS = ("fp32", "fp16", "int8")
PRECISION_NAMES = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8"}
INT8_DROP_LIMIT_POINTS = 2.0
POINTS = 100.0
HZ_PER_MHZ = 1_000_000
KHZ_PER_MHZ = 1_000
MISSING = "n/a"
UNREADABLE = "UNREADABLE"
# Each row's first cell names its device and its precision, e.g. "GPU, PyTorch FP32".
# The key is the ``runtime`` field the benchmark and evaluation logs record.
DEVICE_AND_FRAMEWORK = {
    "pytorch-cuda": "GPU, PyTorch",
    "pytorch-cpu": "CPU, PyTorch",
    "tensorrt-gpu": "GPU, TensorRT",
}
HEADER = (
    "| Device and precision | Engine p50/p90/p99 (ms) | End-to-end p50/p90/p99 (ms) | Engine FPS | "
    "End-to-end FPS | Peak process RSS (MiB) | Avg power VDD_IN (W) | "
    "Person AP@[.5:.95] | Person AP50 | INT8 path |"
)


class TableError(ValueError):
    """The logs cannot make one consistent table; the message is one line."""


def load_logs(runs_dir: Path) -> list[dict]:
    """Read every run log in the directory."""
    return [json.loads(path.read_text()) for path in sorted(runs_dir.glob("*.json"))]


def latest(logs: list[dict], kind: str, precision: str) -> dict | None:
    """Return the newest clean log of a kind and precision, or ``None``; say on stderr which dirty ones it skips."""
    matching = [log for log in logs if log["kind"] == kind and log["precision"] == precision]
    for log in matching:
        if log.get("dirty") is not False:
            print(f"skipping dirty log: {kind} {precision} ending {log.get('end_utc')}", file=sys.stderr)
    clean = [log for log in matching if log.get("dirty") is False]
    return max(clean, key=lambda log: log["end_utc"]) if clean else None


def select(logs: list[dict]) -> dict[tuple[str, str], dict]:
    """Pick the logs that feed the block, keyed by kind and precision, and check they agree."""
    wanted = [(kind, precision) for precision in PRECISIONS for kind in ("benchmark", "eval")]
    wanted += [("build", "fp16"), ("build", "int8"), ("gpu_check", "fp32")]
    chosen = {key: log for key in wanted if (log := latest(logs, *key)) is not None}
    commits = sorted({log["commit"] for log in chosen.values()})
    if len(commits) > 1:
        raise TableError(f"the logs feeding the table come from {len(commits)} commits: {', '.join(commits)}")
    for precision in PRECISIONS:
        benchmark, evaluation = chosen.get(("benchmark", precision)), chosen.get(("eval", precision))
        if benchmark and evaluation and benchmark["results"]["runtime"] != evaluation["results"]["runtime"]:
            raise TableError(
                f"{PRECISION_NAMES[precision]} eval runtime {evaluation['results']['runtime']} differs from "
                f"benchmark runtime {benchmark['results']['runtime']}"
            )
    return chosen


def _number(value, digits: int) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def _percent(value) -> str:
    return f"{_number(value, 0)}%" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def _latency(summary: dict) -> str:
    return " / ".join(f"{summary[key]:.1f}" for key in ("p50_ms", "p90_ms", "p99_ms"))


def _row(chosen: dict, precision: str) -> str:
    benchmark = chosen.get(("benchmark", precision))
    evaluation = chosen.get(("eval", precision))
    build = chosen.get(("build", precision))
    timing = benchmark["results"] if benchmark else None
    scores = evaluation["results"] if evaluation else None
    if precision == "int8":
        int8_path = build["results"]["int8_path"] if build else MISSING
    else:
        int8_path = "-"
    runtime = (timing or scores or {}).get("runtime", MISSING)
    cells = [
        f"{DEVICE_AND_FRAMEWORK.get(runtime, runtime)} {PRECISION_NAMES[precision]}",
        _latency(timing["engine"]) if timing else MISSING,
        _latency(timing["end_to_end"]) if timing else MISSING,
        _number(timing["engine"]["fps"], 1) if timing else MISSING,
        _number(timing["end_to_end"]["fps"], 1) if timing else MISSING,
        _number(timing["peak_process_rss_mib"], 0) if timing else MISSING,
        _number(timing["avg_power_w"], 1) if timing else MISSING,
        _number(scores["ap"] * POINTS, 2) if scores else MISSING,
        _number(scores["ap50"] * POINTS, 2) if scores else MISSING,
        int8_path,
    ]
    return "| " + " | ".join(cells) + " |"


def _at_max(clock: dict, unit: str, per_mhz: int) -> str:
    """The governor, and how many timed samples were at the maximum, as counts."""
    ceiling = clock[f"max_{unit}"]
    top = f"{ceiling / per_mhz:.0f} MHz" if isinstance(ceiling, int) else str(ceiling)
    return f"governor `{clock['governor']}`, at max {top} in {clock['at_max_samples']} of {clock['samples']} samples"


def _clocks(results: dict) -> str:
    cpus = "; ".join(
        f"{name} {_at_max(clock, 'khz', KHZ_PER_MHZ)}" for name, clock in sorted(results["cpu_clocks"].items())
    )
    return f"GPU clock {_at_max(results['gpu_clock'], 'hz', HZ_PER_MHZ)}; CPU clocks {cpus}"


def _spread(values: list[float]) -> str:
    return f"{min(values):.2f}-{max(values):.2f} ms over {len(values)} repeats"


def _first_line(text: str) -> str:
    lines = text.strip().splitlines()
    return lines[0].strip() if lines else ""


def _notes(chosen: dict) -> list[str]:
    """State the measurement conditions and any INT8 accuracy drop over the limit."""
    notes = []
    fp16, int8 = chosen.get(("eval", "fp16")), chosen.get(("eval", "int8"))
    if fp16 and int8:
        # The drop is judged as printed, to 2 decimals, so a drop shown as 2.00 is within the limit.
        drop = round((fp16["results"]["ap"] - int8["results"]["ap"]) * POINTS, 2)
        if drop > INT8_DROP_LIMIT_POINTS:
            notes.append(
                f"INT8 person AP is {drop:.2f} points below FP16, "
                f"more than the {INT8_DROP_LIMIT_POINTS:.0f}-point limit."
            )
        else:
            notes.append(f"INT8 person AP differs from FP16 by {-drop:+.2f} points.")
    for precision in PRECISIONS:
        benchmark = chosen.get(("benchmark", precision))
        if benchmark:
            results = benchmark["results"]
            mode = " ".join(str(results["power_mode"]).split())
            name = PRECISION_NAMES[precision]
            notes.append(
                f"{name} benchmark: power mode `{mode}`, "
                f"{_clocks(results)}, "
                f"one-minute CPU load before timing: {_number(results['cpu_load_1min_before_timing'], 1)}, "
                f"GPU busy before timing: {_percent(results['gpu_busy_pct_before_timing'])}, "
                f"other GPU processes: {benchmark['other_gpu_processes']}."
            )
            notes.append(
                f"{name} p50 across repeats: engine {_spread(results['engine_p50_ms_per_repeat'])}, "
                f"end-to-end {_spread(results['end_to_end_p50_ms_per_repeat'])}."
            )
    check = chosen.get(("gpu_check", "fp32"))
    if check:
        results = check["results"]
        warning = _first_line(results["torch_warning"])
        said = (
            f'PyTorch emitted an unsupported-GPU warning at first CUDA use: "{warning}"'
            if warning
            else "PyTorch emitted no unsupported-GPU warning at first CUDA use"
        )
        notes.append(
            f"PyTorch GPU check: usable {results['gpu_torch_usable']}, device compute capability "
            f"{results['device_compute_capability']}, wheel slices {', '.join(results['arch_list'])}. "
            f"{said}. FP32 ran with cudnn.allow_tf32 {results['cudnn_allow_tf32']} and "
            f"cuda.matmul.allow_tf32 {results['matmul_allow_tf32']}."
        )
    return notes


def build_block(runs_dir: Path = RUNS_DIR) -> str:
    """Render the full results block, markers excluded. Raises ``TableError`` if the logs disagree."""
    chosen = select(load_logs(runs_dir))
    divider = "|" + "---|" * HEADER.count(" | ") + "---|"
    rows = [_row(chosen, precision) for precision in PRECISIONS]
    notes = [f"- {note}" for note in _notes(chosen)]
    return "\n".join([HEADER, divider, *rows, "", *notes]).strip()


def _than(a: float, b: float, more: str, less: str) -> str:
    return more if a > b else less if a < b else "the same as"


def build_int8_block(runs_dir: Path = RUNS_DIR) -> str:
    """The INT8 against FP16 comparison, in words and numbers taken from the logs."""
    chosen = select(load_logs(runs_dir))
    needed = [("benchmark", "fp16"), ("benchmark", "int8"), ("eval", "fp16"), ("eval", "int8")]
    if any(key not in chosen for key in needed):
        return "The INT8 and FP16 logs needed for this comparison are not all present."
    fp16, int8 = chosen[("benchmark", "fp16")]["results"], chosen[("benchmark", "int8")]["results"]
    e16, e8 = fp16["engine"]["p50_ms"], int8["engine"]["p50_ms"]
    t16, t8 = fp16["end_to_end"]["p50_ms"], int8["end_to_end"]["p50_ms"]
    r16, r8 = fp16["end_to_end_p50_ms_per_repeat"], int8["end_to_end_p50_ms_per_repeat"]
    overlap = max(min(r16), min(r8)) <= min(max(r16), max(r8))
    d16 = chosen[("eval", "fp16")]["results"]["detections"]
    d8 = chosen[("eval", "int8")]["results"]["detections"]
    return (
        f"In this run the INT8 engine p50 is {_than(e8, e16, 'higher than', 'lower than')} FP16's, "
        f"{e8:.2f} against {e16:.2f} ms, and its end-to-end p50 is {_than(t8, t16, 'higher than', 'lower than')} "
        f"FP16's, {t8:.2f} against {t16:.2f} ms; the per-repeat end-to-end ranges in the notes "
        f"{'overlap' if overlap else 'do not overlap'}. The FP16 engine p50 is {e16 / t16 * 100:.0f}% of its "
        f"end-to-end p50; the rest is image decoding, preprocessing and postprocessing on the CPU. INT8 yields "
        f"{_than(d8, d16, 'more', 'fewer')} person detections after NMS on val2017 than FP16, {d8} against {d16}, "
        f"the `detections` count in the eval logs."
    )


def repeat_dirs(repeats_dir: Path = REPEATS_DIR) -> list[Path]:
    """The further runs' log directories, in name order."""
    return sorted(path for path in repeats_dir.glob("run*") if path.is_dir()) if repeats_dir.is_dir() else []


# The QUIET gate's load bound: the gate requires load1 below it after a 120 s settle. Every committed QUIET try
# line names the bound its gate used, and a line naming another bound stops the script, so the label and the gate
# cannot drift apart.
QUIET_LOAD_BOUND = 1.0


def _quiet_gates(path: Path) -> dict[str, list[str]]:
    """Each gate's verdicts in a run's ``quiet.txt``, from "QUIET <gate> try=..." and "QUIET <gate> PASS|FAIL" lines."""
    gates: dict[str, list[str]] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] == "QUIET":
            gates.setdefault(fields[1], [])
            if fields[2] in ("PASS", "FAIL"):
                gates[fields[1]].append(fields[2])
            else:
                bounds = [field.split("=", 1)[1] for field in fields if field.startswith("bound=")]
                if len(bounds) != 1 or float(bounds[0]) != QUIET_LOAD_BOUND:
                    raise TableError(f"{path.parent.name}/quiet.txt names a load bound other than {QUIET_LOAD_BOUND}")
    return gates


def _counts_allowed_timers(path: Path) -> bool:
    """Whether every try line carries ``allowed_timers=``; an earlier gate wrote the same line without that count."""
    tries = [line.split() for line in path.read_text().splitlines() if line.startswith("QUIET ") and "try=" in line]
    return all(any(field.startswith("allowed_timers=") for field in fields) for fields in tries)


def host_state(run_dir: Path | None, chosen: dict) -> str:
    """A run's host state, derived from data only.

    With a ``quiet.txt``, the run is quieted when every gate in it ends in exactly one PASS and none in FAIL.
    Without one, it is not quieted when every benchmark's load before timing is at or above the gate's bound;
    otherwise its state is not recorded.
    """
    loads = [
        chosen[("benchmark", precision)]["results"]["cpu_load_1min_before_timing"]
        for precision in PRECISIONS
        if ("benchmark", precision) in chosen
    ]
    quiet = run_dir / "quiet.txt" if run_dir is not None else None
    if quiet is not None and quiet.is_file():
        gates = _quiet_gates(quiet)
        passed = bool(gates) and all(verdicts == ["PASS"] for verdicts in gates.values())
        state = "quieted, every QUIET gate passed" if passed else "not quieted, a QUIET gate did not pass"
        if not _counts_allowed_timers(quiet):
            state += " (its gate lines predate the count of allowed timers)"
    elif loads and all(load >= QUIET_LOAD_BOUND for load in loads):
        state = f"not quieted, load before every benchmark at or above the QUIET gate's {QUIET_LOAD_BOUND:.1f}"
    else:
        state = "host state not recorded"
    return f"{state}; one-minute CPU load before timing {_range(loads, 1)}" if loads else state


def _range(values: list[float], digits: int) -> str:
    return f"{min(values):.{digits}f}-{max(values):.{digits}f}"


def build_repeat_block(runs_dir: Path = RUNS_DIR, repeats_dir: Path = REPEATS_DIR) -> str:
    """The R-11 check: each judged metric's values over the published run and every further run, IN or OUT."""
    import compare_logs  # compare_logs imports this module, so it is imported here, at use.

    runs = repeat_dirs(repeats_dir)
    if not runs:
        return "No further runs are committed under `runs-repeat/`, so R-11 has not been checked."
    published = select(load_logs(runs_dir))
    further = [select(load_logs(path)) for path in runs]
    commits = sorted({log["commit"] for chosen in further for log in chosen.values()})
    names = [path.name for path in runs]
    lines = [
        f"R-11 is checked against {len(runs)} further full runs, at commit{'s' if len(commits) > 1 else ''} "
        f"{', '.join(f'`{commit[:7]}`' for commit in commits)}, whose logs are in `runs-repeat/`. The tolerances are "
        f"the `PRD.md` values in `scripts/compare_logs.py`, unchanged. A metric is IN when every further run is within "
        f"its tolerance of the published value, and OUT otherwise. The spread comes from {len(runs) + 1} runs on one "
        f"board, which is not a population. A further run's QUIET gate, at its start and before each benchmark, "
        f"requires that none of the board's other scheduled jobs is active, that each of at most two long-running "
        f"sessions uses under 5 s of CPU in a 120 s settle, and that load1 is below {QUIET_LOAD_BOUND:.1f} after it.",
        "",
        f"- published (`runs/`): {host_state(None, published)}.",
        *[f"- {path.name}: {host_state(path, chosen)}." for path, chosen in zip(runs, further, strict=True)],
        "",
        f"| Metric | Precision | Published | {' | '.join(names)} | Min-max | Spread | Tolerance | R-11 |",
        "|---|---|---|" + "---|" * len(names) + "---|---|---|---|",
    ]
    missed = []
    for kind in compare_logs.KINDS:
        for precision in PRECISIONS:
            base = published.get((kind, precision))
            if base is None:
                continue
            for metric, read in compare_logs.METRICS[kind]:
                if metric in compare_logs.REPORTED_ONLY:
                    continue
                value = read(base["results"])
                others = [read(chosen[(kind, precision)]["results"]) if (kind, precision) in chosen else None
                          for chosen in further]
                judged = [compare_logs.judge(metric, precision, value, other) for other in others]
                verdict = "OUT" if any(item[2] == "OUT" for item in judged) else "IN"
                if verdict == "OUT":
                    missed.append(f"{metric} {PRECISION_NAMES[precision]}")
                values = [value, *(other for other in others if other is not None)]
                cells = " | ".join(f"{other:.2f}" if other is not None else "missing" for other in others)
                lines.append(
                    f"| {metric} | {PRECISION_NAMES[precision]} | {value:.2f} | {cells} | {_range(values, 2)} "
                    f"| {max(values) - min(values):.2f} | {judged[0][1]} | {verdict} |"
                )
    lines += [
        "",
        "R-11 is met for every judged metric."
        if not missed
        else "; ".join(f"R-11 not met for {name}" for name in missed) + ".",
    ]
    return "\n".join(lines)


BLOCKS = (
    ((BEGIN_MARKER, END_MARKER), lambda runs_dir, repeats_dir: build_block(runs_dir)),
    (INT8_MARKERS, lambda runs_dir, repeats_dir: build_int8_block(runs_dir)),
    (REPEAT_MARKERS, lambda runs_dir, repeats_dir: build_repeat_block(runs_dir, repeats_dir)),
)


def _between(text: str, markers: tuple[str, str]) -> tuple[str, str, str]:
    before, separator, rest = text.partition(markers[0])
    block, closing, after = rest.partition(markers[1])
    if not separator or not closing:
        raise ValueError(f"README has no {markers[0]} markers")
    return before, block, after


def readme_block(text: str, markers: tuple[str, str] = (BEGIN_MARKER, END_MARKER)) -> str:
    """Extract one current block from README text."""
    return _between(text, markers)[1].strip()


def readme_matches(text: str, runs_dir: Path = RUNS_DIR, repeats_dir: Path = REPEATS_DIR) -> bool:
    """Report whether every generated block whose markers the text carries is exactly what the logs make.

    The results markers are required; ``test_readme_table`` checks that the README carries all three pairs.
    """
    readme_block(text)
    return all(
        readme_block(text, markers) == make(runs_dir, repeats_dir)
        for markers, make in BLOCKS
        if markers[0] in text
    )


def write_readme(path: Path = README, runs_dir: Path = RUNS_DIR, repeats_dir: Path = REPEATS_DIR) -> None:
    """Replace everything between each pair of README markers with its generated block."""
    text = path.read_text()
    for markers, make in BLOCKS:
        before, _, after = _between(text, markers)
        text = f"{before}{markers[0]}\n{make(runs_dir, repeats_dir)}\n{markers[1]}{after}"
    path.write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="update the README in place")
    parser.add_argument("--runs", type=Path, default=RUNS_DIR, help="directory of run logs (default: runs/)")
    parser.add_argument("--readme", type=Path, default=README, help="README to update with --write")
    args = parser.parse_args(argv)
    try:
        if args.write:
            write_readme(args.readme, args.runs)
        else:
            print(build_block(args.runs))
    except TableError as error:
        print(f"generate_table: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
