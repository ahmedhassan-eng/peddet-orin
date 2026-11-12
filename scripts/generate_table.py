"""Generate the README results block from the JSON logs under ``runs/``.

The block sits between two HTML comment markers in the README. ``--write`` replaces
it. A test calls ``readme_matches`` so that the README cannot drift from the logs.
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
README = REPO_ROOT / "README.md"
BEGIN_MARKER = "<!-- results:begin -->"
END_MARKER = "<!-- results:end -->"
PRECISIONS = ("fp32", "fp16", "int8")
PRECISION_NAMES = {"fp32": "FP32", "fp16": "FP16", "int8": "INT8"}
INT8_DROP_LIMIT_POINTS = 2.0
POINTS = 100.0
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
                f"clocks locked: {results['clocks_locked']}, "
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


def readme_block(text: str) -> str:
    """Extract the current block from README text."""
    before, separator, rest = text.partition(BEGIN_MARKER)
    block, closing, _ = rest.partition(END_MARKER)
    if not separator or not closing:
        raise ValueError("README has no results markers")
    return block.strip()


def readme_matches(text: str, runs_dir: Path = RUNS_DIR) -> bool:
    """Report whether README text carries exactly the generated block."""
    return readme_block(text) == build_block(runs_dir)


def write_readme(path: Path = README, runs_dir: Path = RUNS_DIR) -> None:
    """Replace everything between the README's markers with the generated block."""
    text = path.read_text()
    before, separator, rest = text.partition(BEGIN_MARKER)
    _, closing, after = rest.partition(END_MARKER)
    if not separator or not closing:
        raise ValueError("README has no results markers")
    path.write_text(f"{before}{BEGIN_MARKER}\n{build_block(runs_dir)}\n{END_MARKER}{after}")


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
