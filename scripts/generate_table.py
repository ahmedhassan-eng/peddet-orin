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
import copy
import json
import re
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
# Further full runs at the release code, one directory of logs per run, used only for the R-11 check.
REPEATS_DIR = REPO_ROOT / "runs-repeat"
# The sampler that wrote the runs' memory, swap and temperature files; each file's first line names its sha256.
# The sha256 of the host's sampler script, as every judged sampler file records it in its first line.
SYSMON_SHA256 = "aa30652bf11859de67420a4e6b64f53cc9348254f03b8a5dea424b94ed2a93ce"
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


# The logs that feed the block, by kind and precision; a judged repeat run must carry every one.
WANTED = (
    *[(kind, precision) for precision in PRECISIONS for kind in ("benchmark", "eval")],
    ("build", "fp16"),
    ("build", "int8"),
    ("gpu_check", "fp32"),
)


def select(logs: list[dict]) -> dict[tuple[str, str], dict]:
    """Pick the logs that feed the block, keyed by kind and precision, and check they agree."""
    chosen = {key: log for key in WANTED if (log := latest(logs, *key)) is not None}
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
        # The drop is taken from the two APs as the table prints them, to 2 decimals, and judged as printed, so the
        # note's difference follows from the shown figures and a drop shown as 2.00 is within the limit.
        drop = round(_shown(fp16["results"]["ap"] * POINTS) - _shown(int8["results"]["ap"] * POINTS), 2)
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


def _shown(value: float) -> float:
    """A number as the README prints it, to 2 decimals."""
    return float(f"{value:.2f}")


def _signed(delta: float, digits: int, suffix: str = "") -> str:
    """A signed difference, with no sign on a zero as printed: "+0.12", "-0.12", "0.00"."""
    text = f"{delta:+.{digits}f}"
    return f"{text[1:]}{suffix}" if float(text) == 0 else f"{text}{suffix}"


def _is_plain_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _median(values: list):
    """The median of numbers (an int when every value is an int), field by field for dicts with equal keys.

    Lists are pooled in run order. Any other value is kept when every run agrees and is "varies" otherwise.
    """
    if all(_is_plain_number(value) for value in values):
        middle = statistics.median(values)
        return int(middle) if all(isinstance(value, int) for value in values) and middle == int(middle) else middle
    if all(isinstance(value, dict) for value in values) and all(set(value) == set(values[0]) for value in values):
        return {key: _median([value[key] for value in values]) for key in values[0]}
    if all(isinstance(value, list) for value in values):
        return [item for value in values for item in value]
    return values[0] if all(value == values[0] for value in values) else "varies"


def median_chosen(runs: list[dict]) -> dict:
    """One log per kind and precision whose results are the medians over the runs; all runs must share a commit."""
    commits = sorted({log["commit"] for chosen in runs for log in chosen.values()})
    if len(commits) > 1:
        raise TableError(f"the repeat runs come from {len(commits)} commits: {', '.join(commits)}")
    keys = [key for key in runs[0] if all(key in chosen for chosen in runs)]
    medians = {}
    for key in keys:
        log = copy.deepcopy(runs[0][key])
        log["results"] = _median([chosen[key]["results"] for chosen in runs])
        medians[key] = log
    return medians


def _resolve_repeats(runs_dir: Path, repeats_dir: Path | None) -> Path | None:
    """The repo's own runs-repeat/ goes with its own runs/; any other runs directory has no repeats unless named."""
    if repeats_dir is None and runs_dir == RUNS_DIR:
        return REPEATS_DIR
    return repeats_dir


def require_quieted(path: Path, chosen: dict) -> None:
    """Stop with ``TableError`` when a judged run's host state is not quieted: such a run never enters the median
    and is parked by hand as ``run<N>-failed``, which ``repeat_dirs`` never reads."""
    state = host_state(path, chosen)
    if not state.startswith("quieted"):
        shown = state.split(";")[0]
        raise TableError(f"judged run {path.name} is not quieted ({shown}); park it as {path.name}-failed")


def judged_runs(repeats_dir: Path) -> list[tuple[Path, dict]]:
    """Each judged run with its selected logs; every one must be quieted."""
    runs = [(path, select(load_logs(path))) for path in judged_and_apart(repeats_dir)[0]]
    for path, chosen in runs:
        missing = [f"{kind} {precision}" for kind, precision in WANTED if (kind, precision) not in chosen]
        if missing:
            raise TableError(f"judged run {path.name} has no clean log for {', '.join(missing)}")
        require_quieted(path, chosen)
    return runs


def headline(runs_dir: Path = RUNS_DIR, repeats_dir: Path | None = None) -> tuple[dict, list[tuple[str, dict]]]:
    """The logs the headline uses: the median over the repeat runs when there are any, else ``runs_dir``'s logs."""
    repeats = _resolve_repeats(runs_dir, repeats_dir)
    runs = [(path.name, chosen) for path, chosen in judged_runs(repeats)] if repeats else []
    if not runs:
        return select(load_logs(runs_dir)), []
    return median_chosen([chosen for _, chosen in runs]), runs


def _ranges(runs: list[tuple[str, dict]], precision: str) -> str | None:
    """One precision's min-max over the runs for the headline metrics."""
    import compare_logs  # compare_logs imports this module, so it is imported here, at use.

    parts = []
    for kind in compare_logs.KINDS:
        if not all((kind, precision) in chosen for _, chosen in runs):
            continue
        for metric, read in compare_logs.METRICS[kind]:
            if metric in compare_logs.REPORTED_ONLY:
                continue
            values = [read(chosen[(kind, precision)]["results"]) for _, chosen in runs]
            if all(_is_plain_number(value) for value in values):
                parts.append(f"{metric} {_range(values, 2)}")
    return f"{PRECISION_NAMES[precision]} min-max over the {len(runs)} runs: {', '.join(parts)}." if parts else None


def _benchmarks(chosen: dict) -> list[dict]:
    return [chosen[key]["results"] for precision in PRECISIONS if (key := ("benchmark", precision)) in chosen]


def _maxima(results: dict) -> tuple:
    """A benchmark's GPU maximum clock and each CPU policy's, as its clock record gives them."""
    return (results["gpu_clock"]["max_hz"],
            tuple(sorted((name, clock["max_khz"]) for name, clock in results["cpu_clocks"].items())))


def power_mode_sentence(runs: list[tuple[str, dict]], earlier: dict) -> str:
    """Whether the judged runs' power mode was read, and, when it was not, the clock maxima the logs hold as
    evidence, against the earlier run's; stops when any judged benchmark's maxima differ from the earlier run's."""
    judged = [results for _, chosen in runs for results in _benchmarks(chosen)]
    modes = sorted({" ".join(str(results["power_mode"]).split()) for results in judged})
    if UNREADABLE not in modes:
        return f"The judged runs' power mode, as `nvpmodel -q` reported it: {', '.join(f'`{m}`' for m in modes)}."
    before = _benchmarks(earlier)
    earlier_maxima = sorted({_maxima(results) for results in before})
    earlier_modes = sorted({" ".join(str(results["power_mode"]).split()) for results in before})
    if len(earlier_maxima) != 1 or any(_maxima(results) != earlier_maxima[0] for results in judged):
        raise TableError("a judged benchmark's GPU or CPU maximum clock differs from the earlier run's")
    unread = sum(" ".join(str(results["power_mode"]).split()) == UNREADABLE for results in judged)
    gpu, cpus = earlier_maxima[0]
    policies = ", ".join(f"{name} {khz / KHZ_PER_MHZ:.0f} MHz" for name, khz in cpus)
    return (
        f"The judged runs' power mode was not read: {unread} of their {len(judged)} benchmark logs record "
        f"`{UNREADABLE}`, which a log records when `nvpmodel` is not found on PATH or `nvpmodel -q` fails, and "
        f"the host's run script adds no system sbin directory to PATH. The evidence the logs hold is the "
        f"clock maxima of every judged benchmark: GPU {gpu / HZ_PER_MHZ:.0f} MHz, CPU {policies}, the same as in "
        f"each of the earlier run's {len(before)} benchmarks, whose power mode reads "
        f"{', '.join(f'`{mode}`' for mode in earlier_modes)}."
    )


def int8_ap_sentence(runs: list[tuple[str, dict]], earlier: dict) -> str:
    """INT8 person AP in the judged runs and in the earlier run; each run builds its own INT8 engine."""
    judged = sorted({f"{chosen[('eval', 'int8')]['results']['ap'] * POINTS:.2f}" for _, chosen in runs})
    shown = (f"{judged[0]} in each of the {len(runs)} judged runs" if len(judged) == 1
             else f"{judged[0]}-{judged[-1]} across the {len(runs)} judged runs")
    before = earlier.get(("eval", "int8"))
    then = f" and {before['results']['ap'] * POINTS:.2f} in the earlier run" if before else ""
    return (f"INT8 person AP is {shown}{then}; each run builds its own INT8 engine, so each score comes from a "
            "separate build.")


def provenance(runs: list[tuple[str, dict]]) -> tuple[str, str]:
    """The one commit and the one peddet-orin version that every judged run's logs record."""
    logs = [log for _, chosen in runs for log in chosen.values()]
    commits = sorted({log["commit"] for log in logs})
    versions = sorted({str(log.get("environment", {}).get("peddet_orin", MISSING)) for log in logs})
    if len(commits) != 1 or len(versions) != 1:
        raise TableError(
            f"the judged runs record {len(commits)} commits and {len(versions)} package versions: "
            f"{', '.join(commits + versions)}"
        )
    return commits[0], versions[0]


def build_block(runs_dir: Path = RUNS_DIR, repeats_dir: Path | None = None) -> str:
    """Render the full results block, markers excluded. Raises ``TableError`` if the logs disagree.

    With repeat runs, each cell is the median over them, the notes give each precision's min-max, and each run's
    benchmark conditions are listed under its name.
    """
    chosen, runs = headline(runs_dir, repeats_dir)
    divider = "|" + "---|" * HEADER.count(" | ") + "---|"
    rows = [_row(chosen, precision) for precision in PRECISIONS]
    if not runs:
        notes = [f"- {note}" for note in _notes(chosen)]
        return "\n".join([HEADER, divider, *rows, "", *notes]).strip()
    commit, version = provenance(runs)
    notes = [
        f"Each cell is the median over the n = {len(runs)} quieted runs in `runs-repeat/` "
        f"({', '.join(n for n, _ in runs)}),"
        f" this release's count; 0.2.0 plans 5 or more runs.",
        f"The judged runs were made at commit `{commit[:7]}` with package version {version}, as their logs record.",
        "Memory, swap and temperature are recorded for the judged runs only, over each whole benchmark job, warm-up "
        "included; the earlier runs have no such record.",
        *[note for note in _notes(chosen) if note.startswith("INT8 person AP")],
        int8_ap_sentence(runs, select(load_logs(runs_dir))),
        power_mode_sentence(runs, select(load_logs(runs_dir))),
        *[line for precision in PRECISIONS if (line := _ranges(runs, precision))],
    ]
    for name, run in runs:
        notes += [f"{name}: {note}" for note in _notes(run) if not note.startswith("INT8 person AP")]
    return "\n".join([HEADER, divider, *rows, "", *[f"- {note}" for note in notes]]).strip()


def _than(a: float, b: float, more: str, less: str, same: str = "the same as") -> str:
    """``more`` when a > b, ``less`` when a < b, else ``same``: each call passes the words that fit its sentence."""
    return more if a > b else less if a < b else same


def build_int8_block(runs_dir: Path = RUNS_DIR, repeats_dir: Path | None = None) -> str:
    """The INT8 against FP16 comparison, in words and numbers taken from the headline's logs."""
    chosen, runs = headline(runs_dir, repeats_dir)
    needed = [("benchmark", "fp16"), ("benchmark", "int8"), ("eval", "fp16"), ("eval", "int8")]
    if any(key not in chosen for key in needed):
        return "The INT8 and FP16 logs needed for this comparison are not all present."
    fp16, int8 = chosen[("benchmark", "fp16")]["results"], chosen[("benchmark", "int8")]["results"]
    e16, e8 = fp16["engine"]["p50_ms"], int8["engine"]["p50_ms"]
    t16, t8 = fp16["end_to_end"]["p50_ms"], int8["end_to_end"]["p50_ms"]
    opening = f"Over the median of the {len(runs)} quieted runs" if runs else "In this run"
    per_run = [chosen for _, chosen in runs] or [chosen]
    overlaps = sum(_repeats_overlap(run) for run in per_run)
    if not runs:
        ranges = "overlap" if overlaps else "do not overlap"
    elif overlaps == len(per_run):
        ranges = f"overlap in each of the {len(per_run)} runs"
    elif overlaps == 0:
        ranges = f"do not overlap in any of the {len(per_run)} runs"
    else:
        ranges = f"overlap in {overlaps} of the {len(per_run)} runs"
    d16 = chosen[("eval", "fp16")]["results"]["detections"]
    d8 = chosen[("eval", "int8")]["results"]["detections"]
    return (
        f"{opening}, the INT8 engine p50 is {_than(e8, e16, 'higher than', 'lower than')} FP16's, "
        f"{e8:.2f} against {e16:.2f} ms, and its end-to-end p50 is {_than(t8, t16, 'higher than', 'lower than')} "
        f"FP16's, {t8:.2f} against {t16:.2f} ms; FP16's and INT8's per-repeat end-to-end ranges in the notes "
        f"{ranges}. The FP16 engine p50 is {e16 / t16 * 100:.0f}% of its "
        f"end-to-end p50; the rest is image decoding, preprocessing and postprocessing on the CPU. INT8 yields "
        f"{_than(d8, d16, 'more', 'fewer', 'as many')} person detections after NMS on val2017 "
        f"{'as' if d8 == d16 else 'than'} FP16, {d8} against {d16}, "
        f"the `detections` count in the eval logs."
        + (
            f" Whether {_than(d8, d16, 'more', 'fewer')} detections give the postprocess "
            f"{_than(d8, d16, 'more', 'less')} work during the benchmark is a hypothesis: no profile here separates "
            f"it from run-to-run variation."
            if d8 != d16
            else ""
        )
    )


def _repeats_overlap(chosen: dict) -> bool:
    """Whether one run's FP16 and INT8 per-repeat end-to-end p50 ranges overlap."""
    r16 = chosen[("benchmark", "fp16")]["results"]["end_to_end_p50_ms_per_repeat"]
    r8 = chosen[("benchmark", "int8")]["results"]["end_to_end_p50_ms_per_repeat"]
    return max(min(r16), min(r8)) <= min(max(r16), max(r8))


def repeat_dirs(repeats_dir: Path = REPEATS_DIR) -> list[Path]:
    """The further runs' log directories in name order, ``run<digits>`` only: failed and attempt dirs are never read."""
    if not repeats_dir.is_dir():
        return []
    return sorted(path for path in repeats_dir.iterdir() if path.is_dir() and re.fullmatch(r"run[0-9]+", path.name))


def is_recorded(run_dir: Path) -> bool:
    """A run is recorded when it carries the memory, swap and temperature samples of its GPU jobs."""
    return (run_dir / "sysmon").is_dir()


def judged_and_apart(repeats_dir: Path) -> tuple[list[Path], list[Path]]:
    """The judged runs are the recorded ones when any run is recorded, else every run; the rest are shown apart."""
    runs = repeat_dirs(repeats_dir)
    recorded = [path for path in runs if is_recorded(path)]
    if not recorded:
        return runs, []
    return recorded, [path for path in runs if not is_recorded(path)]


def _samples(path: Path) -> list[dict[str, str]]:
    """The ``SYS`` lines of one sysmon file, each as its key=value fields.

    The file's first line must be ``SYSMON sha256=<hash>`` naming the recorded sampler, ``SYSMON_SHA256``.
    """
    lines = path.read_text().splitlines()
    want = SYSMON_SHA256
    first = lines[0].split() if lines else []
    if first[:1] != ["SYSMON"] or f"sha256={want}" not in first:
        raise TableError(f"sampler file {path.name} of {path.parent.parent.name} does not name the recorded sampler")
    rows = []
    for line in lines[1:]:
        fields = line.split()
        if fields[:1] == ["SYS"] and len(fields) > 1:
            row = dict(field.split("=", 1) for field in fields[2:] if "=" in field)
            row["utc"] = fields[1]
            rows.append(row)
    return rows


def system_record(run_dir: Path) -> dict[str, dict]:
    """Per benchmark job of a recorded run: swap used at its start, its swap-out delta, its lowest available memory
    and the highest reading of each thermal zone, from ``sysmon/*bench_<precision>.txt``."""
    record = {}
    for precision in PRECISIONS:
        files = sorted((run_dir / "sysmon").glob(f"*bench_{precision}.txt"))
        rows = _samples(files[0]) if len(files) == 1 else []
        if not rows:
            continue
        def number(row: dict, key: str) -> int | None:
            value = row.get(key, "")
            return int(value) if value.isdigit() else None
        first, last = rows[0], rows[-1]
        start, end = number(first, "pswpout"), number(last, "pswpout")
        memory = [value for row in rows if (value := number(row, "mem_available_kb")) is not None]
        zones = sorted({key for row in rows for key in row if key.endswith("_mc")})
        record[precision] = {
            "samples": len(rows),
            "swap_used_kb_at_start": number(first, "swap_used_kb"),
            "pswpout_delta": end - start if start is not None and end is not None else None,
            "min_mem_available_mib": min(memory) // 1024 if memory else None,
            "max_temp_c": {
                zone[: -len("_mc")]: max(int(row[zone]) for row in rows if row.get(zone, "").isdigit()) / 1000
                for zone in zones
                if any(row.get(zone, "").isdigit() for row in rows)
            },
        }
    return record


def job_memory(run_dir: Path) -> list[tuple[str, int | None, int | None]]:
    """Per GPU job of a recorded run, in name order: the job, its lowest available memory in MiB over the samples
    taken during it, and its swap-out delta from its first sample to its last, from ``sysmon/<job>.txt``."""
    jobs = []
    for path in sorted((run_dir / "sysmon").glob("*.txt")):
        rows = _samples(path)
        memory = [int(row["mem_available_kb"]) for row in rows if row.get("mem_available_kb", "").isdigit()]
        counts = [int(row["pswpout"]) for row in rows if row.get("pswpout", "").isdigit()]
        job = path.stem.split("-", 1)[1] if "-" in path.stem else path.stem
        jobs.append((job, min(memory) // 1024 if memory else None, counts[-1] - counts[0] if counts else None))
    return jobs


def ram_lines(runs: list[Path]) -> list[str]:
    """Each judged run's GPU jobs with their lowest available memory and pages swapped out, and one sentence: RAM
    was not a constraint, or each job that swapped pages out, with its count."""
    rows, lowest, swapped, unread = [], [], [], []
    for path in runs:
        for job, memory, delta in job_memory(path):
            rows.append(f"| {path.name} | {job} | {MISSING if memory is None else memory} "
                        f"| {MISSING if delta is None else delta} |")
            if memory is None or delta is None:
                unread.append(f"{path.name} {job}")
                continue
            lowest.append(memory)
            if delta > 0:
                swapped.append(f"{path.name} {job} swapped out {delta} pages")
    if not rows:
        return []
    numbers = [int(path.name.removeprefix("run")) for path in runs]
    if len(numbers) == 1:
        names = f"run {numbers[0]}"
    elif numbers == list(range(numbers[0], numbers[-1] + 1)):
        names = f"runs {numbers[0]}-{numbers[-1]}"
    else:
        names = "runs " + ", ".join(str(number) for number in numbers)
    if unread:
        verdict = f"Available memory or swap-outs could not be read for {', '.join(unread)}."
    elif swapped:
        verdict = "; ".join(swapped) + "."
    else:
        verdict = (f"RAM was not a constraint in {names}: the lowest available memory during any job was "
                   f"{min(lowest)} MiB, and 0 pages were swapped out.")
    return [
        "",
        "Each judged run's GPU jobs, from the samples taken during each job, from its start to its end:",
        "",
        "| Run | Job | Lowest available memory (MiB) | Pages swapped out |",
        "|---|---|---|---|",
        *rows,
        "",
        verdict,
    ]


def sampler_intervals(runs: list[Path]) -> set[str]:
    """Every ``interval_s`` value named by the judged runs' sysmon header lines."""
    found = set()
    for path in runs:
        for file in (path / "sysmon").glob("*.txt"):
            first = file.read_text().splitlines()[:1]
            fields = first[0].split() if first else []
            found |= {field.split("=", 1)[1] for field in fields if field.startswith("interval_s=")}
    return found


def first_sample(run_dir: Path) -> dict[str, str] | None:
    """The run's first memory, swap and temperature sample: the first SYS line of its earliest GPU job."""
    files = sorted((run_dir / "sysmon").glob("*.txt"), key=lambda path: (_samples(path) or [{}])[0].get("utc", ""))
    for path in files:
        rows = _samples(path)
        if rows:
            return rows[0]
    return None


def _swap_clean(record: dict[str, dict]) -> bool:
    """Swap used 0 at each benchmark job's start and no page swapped out during it, for all three precisions.

    A precision missing from the record is not clean either; ``host_state`` names it as missing before calling this.
    """
    return len(record) == len(PRECISIONS) and all(
        entry["swap_used_kb_at_start"] == 0 and entry["pswpout_delta"] == 0 for entry in record.values()
    )


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


# The QUIET try-line fields that each judged gate must show, as (field, value it must equal, what it counts).
GATE_ZEROS = (
    ("held_active", "0", "scheduled jobs or path units active other than the ones allowed by name"),
    ("scopes_over", "0", "sessions over their CPU budget or not allowed"),
    ("busy_services", "0", "loop services active"),
    ("swap_used_kb", "0", "kB of swap used"),
    ("pswpout_delta", "0", "pages swapped out during the settle"),
)


def _passing_tries(path: Path) -> list[dict[str, str]]:
    """The fields of the try line just before each PASS in a run's ``quiet.txt``, with the gate's name."""
    found, last = [], None
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[0] == "QUIET":
            if fields[2] == "PASS" and last is not None and last["gate"] == fields[1]:
                found.append(last)
            elif fields[2].startswith("try="):
                last = dict(field.split("=", 1) for field in fields[2:] if "=" in field)
                last["gate"] = fields[1]
    return found


def gate_lines(runs: list[Path]) -> list[str]:
    """The QUIET gate's enforced terms and each judged run's values, from the fields of its passing try lines."""
    tries = {path.name: _passing_tries(path / "quiet.txt") for path in runs}
    every = [entry for entries in tries.values() for entry in entries]
    for field in ("bound", "mem_min_kb", "allowed_timers"):
        values = sorted({entry.get(field, MISSING) for entry in every})
        if len(values) != 1 or values[0] == MISSING:
            raise TableError(f"the judged runs' passing QUIET lines name {len(values)} values of {field}")
    for field, want, _ in GATE_ZEROS:
        if any(entry.get(field) != want for entry in every):
            raise TableError(f"a judged run's passing QUIET line has {field} other than {want}")
    bound, floor, allowed = every[0]["bound"], every[0]["mem_min_kb"], every[0]["allowed_timers"]
    zeros = ", ".join(f"{want} {what}" for _, want, what in GATE_ZEROS)
    lines = [
        f"Each judged run's QUIET gate ran at its start and before each benchmark, after a settle; each gate passed "
        f"on the try shown, and its passing line records: {zeros}; {allowed} scheduled jobs allowed by name; load1 "
        f"below {bound}; available memory at least {floor} kB.",
        "",
        "| Run | Gate | Try | load1 | Available memory (kB) |",
        "|---|---|---|---|---|",
    ]
    for name, entries in tries.items():
        lines += [
            f"| {name} | {entry['gate']} | {entry.get('try', MISSING)} | {entry.get('load1', MISSING)} "
            f"| {entry.get('mem_available_kb', MISSING)} |"
            for entry in entries
        ]
    return lines


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
        record = system_record(run_dir) if passed and is_recorded(run_dir) else None
        unsampled = [PRECISION_NAMES[precision] for precision in PRECISIONS if record is not None
                     and precision not in record]
        if unsampled:
            state = f"not quieted, no single sysmon file with samples for the {', '.join(unsampled)} benchmark job"
        elif record is not None and not _swap_clean(record):
            state = "not quieted, swap was in use or pages were swapped out during a benchmark job"
        elif not is_recorded(run_dir):
            state += ", recorded before the memory, swap and temperature reads"
        if not _counts_allowed_timers(quiet):
            state += " (its gate lines predate the count of allowed timers)"
    elif loads and all(load >= QUIET_LOAD_BOUND for load in loads):
        state = f"not quieted, load before every benchmark at or above the QUIET gate's {QUIET_LOAD_BOUND:.1f}"
    else:
        state = "host state not recorded"
    return f"{state}; one-minute CPU load before timing {_range(loads, 1)}" if loads else state


def _range(values: list[float], digits: int) -> str:
    return f"{min(values):.{digits}f}-{max(values):.{digits}f}"


def _deviation(compare_logs, metric: str, precision: str, median: object, value: object) -> str:
    if not _is_plain_number(value):
        return "missing"
    if "points" in compare_logs.TOLERANCES.get(metric, {}):
        return f"{value:.2f} ({_signed(_shown(value) - _shown(median), 2)})"
    delta = compare_logs.judge(metric, precision, median, value)[0]
    return f"{value:.2f} ({_signed(float(delta.rstrip('%')), 1, '%') if delta.endswith('%') else delta})"


def _system_lines(runs: list[Path]) -> list[str]:
    """Per judged run and benchmark: swap at the job's start, swap-outs during it, lowest free memory, zone peaks."""
    rows = []
    for path in runs:
        record = system_record(path)
        for precision in PRECISIONS:
            entry = record.get(precision)
            if entry is None:
                rows.append(f"| {path.name} | {PRECISION_NAMES[precision]} | missing | missing | missing | missing |")
                continue
            peaks = ", ".join(f"{zone} {value:.1f}" for zone, value in entry["max_temp_c"].items())
            rows.append(
                f"| {path.name} | {PRECISION_NAMES[precision]} | {entry['swap_used_kb_at_start']} "
                f"| {entry['pswpout_delta']} | {entry['min_mem_available_mib']} | {peaks} |"
            )
    if not rows:
        return []
    starts = []
    for path in runs:
        sample = first_sample(path)
        if sample is None:
            continue
        zones = ", ".join(
            f"{key[: -len('_mc')]} {int(value) / 1000:.1f}" for key, value in sample.items()
            if key.endswith("_mc") and value.isdigit()
        )
        memory = sample.get("mem_available_kb", "")
        shown = f"{int(memory) // 1024} MiB" if memory.isdigit() else "NA"
        starts.append(
            f"- {path.name} first sample, {sample['utc']}: available memory {shown}, swap used "
            f"{sample.get('swap_used_kb', 'NA')} kB, temperatures (C) {zones}."
        )
    intervals = sampler_intervals(runs)
    if len(intervals) != 1:
        raise TableError(f"the judged runs' samplers name {len(intervals)} intervals: {', '.join(sorted(intervals))}")
    interval = intervals.pop()
    zones = sorted({zone for path in runs for entry in system_record(path).values() for zone in entry["max_temp_c"]})
    return [
        "",
        "Each judged run's state before any measurement, from the first sample of its first GPU job:",
        "",
        *starts,
        "",
        f"Memory, swap and temperature of each judged run's benchmark jobs, sampled every {interval} s by "
        f"the host's sampler script, a separate process, over the whole job, warm-up included; runs shown apart had "
        f"no sampler. The thermal zones are the ones whose readings the records hold: {', '.join(zones)}.",
        "",
        "| Run | Precision | Swap used at start (kB) | Pages swapped out | Lowest available memory (MiB) "
        "| Highest temperature (C) |",
        "|---|---|---|---|---|---|",
        *rows,
        *ram_lines(runs),
    ]


def build_repeat_block(runs_dir: Path = RUNS_DIR, repeats_dir: Path | None = None) -> str:
    """The R-11 check among the judged runs, each against their median, with every other run shown apart."""
    import compare_logs  # compare_logs imports this module, so it is imported here, at use.

    repeats = _resolve_repeats(runs_dir, repeats_dir)
    runs, apart_runs = judged_and_apart(repeats) if repeats else ([], [])
    if not runs:
        return "No further runs are committed under `runs-repeat/`, so R-11 has not been checked."
    further = [chosen for _, chosen in judged_runs(repeats)]
    apart = [(path.name, select(load_logs(path))) for path in apart_runs]
    median = median_chosen(further)
    earlier = select(load_logs(runs_dir))
    apart.append(("earlier run", earlier))
    commits = sorted({log["commit"] for chosen in further for log in chosen.values()})
    earlier_commits = sorted({log["commit"][:7] for log in earlier.values()})
    names = [path.name for path in runs]
    also = (
        f" and runs {', '.join(path.name for path in apart_runs)} were quieted but recorded before the memory, swap "
        f"and temperature reads"
        if apart_runs
        else ""
    )
    lines = [
        f"R-11 is judged among the {len(runs)} quieted runs {', '.join(names)} in `runs-repeat/`, at commit "
        f"{', '.join(f'`{commit[:7]}`' for commit in commits)}: a metric is IN when every judged run is within its "
        f"tolerance of their median, and OUT otherwise. The tolerances are the `PRD.md` values in "
        f"`scripts/compare_logs.py`, unchanged. The spread comes from {len(runs)} runs on one board, which is not a "
        f"population. avg_power_w is the board's input power, VDD_IN, as its sensor measures it. "
        f"The earlier run in `runs/`, at {', '.join(f'`{c}`' for c in earlier_commits)}, gave 0.1.0's table"
        f"{also}; "
        f"{'they are' if apart_runs else 'it is'} shown apart with the deviation from the median and not judged.",
        "",
        *gate_lines(runs),
        "",
        *[f"- {path.name}: {host_state(path, chosen)}." for path, chosen in zip(runs, further, strict=True)],
        *[f"- {path.name}: {host_state(path, chosen)}." for path, (_, chosen) in zip(apart_runs, apart, strict=False)],
        f"- earlier run (`runs/`): {host_state(None, earlier)}.",
        "",
        f"| Metric | Precision | Median | {' | '.join(names)} | Min-max | Spread | Tolerance | R-11 "
        f"| {' | '.join(name for name, _ in apart)} |",
        "|---|---|---|" + "---|" * len(names) + "---|---|---|---|" + "---|" * len(apart),
    ]
    missed = []
    for kind in compare_logs.KINDS:
        for precision in PRECISIONS:
            base = median.get((kind, precision))
            if base is None:
                continue
            for metric, read in compare_logs.METRICS[kind]:
                if metric in compare_logs.REPORTED_ONLY:
                    continue
                value = read(base["results"])
                others = [
                    read(chosen[(kind, precision)]["results"]) if (kind, precision) in chosen else None
                    for chosen in further
                ]
                judged = [compare_logs.judge(metric, precision, value, other) for other in others]
                verdict = "OUT" if any(item[2] == "OUT" for item in judged) else "IN"
                if verdict == "OUT":
                    missed.append(f"{metric} {PRECISION_NAMES[precision]}")
                values = [other for other in others if other is not None]
                cells = " | ".join(f"{other:.2f}" if other is not None else "missing" for other in others)
                shown = " | ".join(
                    _deviation(compare_logs, metric, precision, value,
                               read(chosen[(kind, precision)]["results"]) if (kind, precision) in chosen else None)
                    for _, chosen in apart
                )
                lines.append(
                    f"| {metric} | {PRECISION_NAMES[precision]} | {value:.2f} | {cells} | {_range(values, 2)} "
                    f"| {max(values) - min(values):.2f} | {judged[0][1]} | {verdict} | {shown} |"
                )
    lines += [
        "",
        "R-11 is met for every judged metric among the quieted runs."
        if not missed
        else "; ".join(f"R-11 not met for {name}" for name in missed) + ".",
    ]
    lines += _system_lines([path for path in runs if is_recorded(path)])
    return "\n".join(lines)


BLOCKS = (
    ((BEGIN_MARKER, END_MARKER), lambda runs_dir, repeats_dir: build_block(runs_dir, repeats_dir)),
    (INT8_MARKERS, lambda runs_dir, repeats_dir: build_int8_block(runs_dir, repeats_dir)),
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


def readme_matches(text: str, runs_dir: Path = RUNS_DIR, repeats_dir: Path | None = None) -> bool:
    """Report whether every generated block whose markers the text carries is exactly what the logs make.

    The results markers are required; ``test_readme_table`` checks that the README carries all three pairs.
    """
    readme_block(text)
    return all(
        readme_block(text, markers) == make(runs_dir, repeats_dir)
        for markers, make in BLOCKS
        if markers[0] in text
    )


def write_readme(path: Path = README, runs_dir: Path = RUNS_DIR, repeats_dir: Path | None = None) -> None:
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
