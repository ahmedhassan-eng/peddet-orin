"""Generate the README results block from the JSON logs under ``runs/``.

The block sits between two HTML comment markers in the README. ``--write`` replaces
it. A test calls ``readme_matches`` so that the README cannot drift from the logs.
For each precision and log kind, the newest log by end time supplies the numbers.
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


def load_logs(runs_dir: Path) -> list[dict]:
    """Read every run log in the directory."""
    return [json.loads(path.read_text()) for path in sorted(runs_dir.glob("*.json"))]


def latest(logs: list[dict], kind: str, precision: str) -> dict | None:
    """Return the newest log of a kind and precision, or ``None``."""
    matching = [log for log in logs if log["kind"] == kind and log["precision"] == precision]
    return max(matching, key=lambda log: log["end_utc"]) if matching else None


def _number(value, digits: int) -> str:
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else str(value)


def _latency(summary: dict) -> str:
    return " / ".join(f"{summary[key]:.1f}" for key in ("p50_ms", "p90_ms", "p99_ms"))


def _row(logs: list[dict], precision: str) -> str:
    benchmark = latest(logs, "benchmark", precision)
    evaluation = latest(logs, "eval", precision)
    build = latest(logs, "build", precision)
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
        _number(timing["system_ram_rise_mib"], 0) if timing else MISSING,
        _number(timing["avg_power_w"], 2) if timing else MISSING,
        _number(scores["ap"] * POINTS, 2) if scores else MISSING,
        _number(scores["ap50"] * POINTS, 2) if scores else MISSING,
        int8_path,
    ]
    return "| " + " | ".join(cells) + " |"


def _notes(logs: list[dict]) -> list[str]:
    """State the measurement conditions and any INT8 accuracy drop over the limit."""
    notes = []
    fp16, int8 = latest(logs, "eval", "fp16"), latest(logs, "eval", "int8")
    if fp16 and int8:
        drop = (fp16["results"]["ap"] - int8["results"]["ap"]) * POINTS
        if drop > INT8_DROP_LIMIT_POINTS:
            notes.append(
                f"INT8 person AP is {drop:.2f} points below FP16, "
                f"more than the {INT8_DROP_LIMIT_POINTS:.0f}-point limit."
            )
        else:
            notes.append(f"INT8 person AP differs from FP16 by {-drop:+.2f} points.")
    for precision in PRECISIONS:
        benchmark = latest(logs, "benchmark", precision)
        if benchmark:
            results = benchmark["results"]
            mode = " ".join(str(results["power_mode"]).split())
            notes.append(
                f"{PRECISION_NAMES[precision]} benchmark: power mode `{mode}`, "
                f"clocks locked: {results['clocks_locked']}, "
                f"one-minute CPU load after timing: {results['cpu_load_1min']:.1f}, "
                f"other GPU processes: {benchmark['other_gpu_processes']}."
            )
    check = latest(logs, "gpu_check", "fp32")
    if check:
        results = check["results"]
        notes.append(
            f"PyTorch GPU check: usable {results['gpu_torch_usable']}, device compute capability "
            f"{results['device_compute_capability']}, wheel slices {', '.join(results['arch_list'])}, "
            f"slice that runs by CUDA's binary-compatibility rule: {results['inferred_slice']}."
        )
    return notes


def build_block(runs_dir: Path = RUNS_DIR) -> str:
    """Render the full results block, markers excluded."""
    logs = load_logs(runs_dir)
    header = (
        "| Device and precision | Engine p50/p90/p99 (ms) | End-to-end p50/p90/p99 (ms) | Engine FPS | "
        "End-to-end FPS | Peak process RSS (MiB) | System RAM rise (MiB) | Avg power VDD_IN (W) | "
        "Person AP@[.5:.95] | Person AP50 | INT8 path |"
    )
    divider = "|" + "---|" * header.count(" | ") + "---|"
    rows = [_row(logs, precision) for precision in PRECISIONS]
    notes = [f"- {note}" for note in _notes(logs)]
    return "\n".join([header, divider, *rows, "", *notes]).strip()


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="update the README in place")
    args = parser.parse_args()
    if args.write:
        write_readme()
    else:
        print(build_block())
    return 0


if __name__ == "__main__":
    sys.exit(main())
