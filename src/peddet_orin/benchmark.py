# SPDX-License-Identifier: Apache-2.0
"""Latency, throughput, memory and power for one precision, at batch size 1.

Each timed iteration decodes and preprocesses an image from memory, runs inference
and postprocesses. Engine-only latency is the inference step alone, including the
copies to and from device memory. End-to-end latency is the whole iteration.
Warm-up passes come first and are excluded, which also absorbs the first-call JIT.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from peddet_orin import debug, runlog
from peddet_orin.evaluate import (
    IMAGES,
    IMAGES_RELATIVE,
    HelpFormatter,
    add_common_options,
    check_engine,
    load_backend,
    release_backend,
    usage_error,
)
from peddet_orin.model import load_detector
from peddet_orin.postprocess import Postprocessor
from peddet_orin.preprocess import PIXEL_MAX, preprocess

# Nothing in this project tries to read or set the clock state, so the log says so.
CLOCKS_LOCKED_STATUS = "NOT_PROBED"
WARMUP_ITERATIONS = 50
TIMED_ITERATIONS = 1000
REPEATS = 3
IMAGE_POOL_SIZE = 200
TEGRASTATS = "tegrastats"
BENCHMARK_SEED = 7
TEGRASTATS_INTERVAL_MS = 200
BASELINE_SAMPLE_S = 3.0
PERCENTILES = (50, 90, 99)
KIB_PER_MIB = 1024.0
MS_PER_S = 1e3
MW_PER_W = 1e3
VDD_IN_PATTERN = re.compile(r"VDD_IN (\d+)mW")
GPU_BUSY_PATTERN = re.compile(r"GR3D_FREQ (\d+)%")
RAM_PATTERN = re.compile(r"RAM (\d+)/\d+MB")


class TegrastatsSampler:
    """Collect tegrastats lines in a thread between ``start`` and ``stop``."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.process: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        try:
            self.process = subprocess.Popen(
                [TEGRASTATS, "--interval", str(TEGRASTATS_INTERVAL_MS)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            self.process = None
            return
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self) -> None:
        for line in self.process.stdout:
            self.lines.append(line)

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
            self.thread.join()

    def values(self, pattern: re.Pattern[str]) -> list[int]:
        return tegrastats_values(self.lines, pattern)


def tegrastats_values(lines: list[str], pattern: re.Pattern[str]) -> list[int]:
    """The integer the pattern captures in each tegrastats line that has it."""
    return [int(match.group(1)) for line in lines if (match := pattern.search(line))]


def average_power_w(lines: list[str]) -> float | str:
    """Mean VDD_IN board input power in watts; tegrastats reports milliwatts."""
    return _mean_or_unreadable(tegrastats_values(lines, VDD_IN_PATTERN), MW_PER_W)


def gpu_busy_pct(lines: list[str]) -> float | str:
    """Mean GR3D_FREQ load in percent, with or without the @frequency suffix."""
    return _mean_or_unreadable(tegrastats_values(lines, GPU_BUSY_PATTERN))


def system_ram_rise_mib(timed_lines: list[str], baseline_lines: list[str]) -> float | str:
    """Peak used RAM while timing minus the mean used RAM of the idle baseline, in MiB."""
    ram = tegrastats_values(timed_lines, RAM_PATTERN)
    base_ram = tegrastats_values(baseline_lines, RAM_PATTERN)
    return (max(ram) - float(np.mean(base_ram))) if ram and base_ram else runlog.UNREADABLE


def peak_rss_mib() -> float:
    """This process's peak resident set size; Linux reports ru_maxrss in KiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KIB_PER_MIB


def image_pool(images_dir: Path = IMAGES) -> list[bytes]:
    """Load the encoded bytes of the first validation images, in file-name order."""
    return [path.read_bytes() for path in sorted(images_dir.glob("*.jpg"))[:IMAGE_POOL_SIZE]]


def _decode(encoded: bytes) -> torch.Tensor:
    with Image.open(io.BytesIO(encoded)) as image:
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).float() / PIXEL_MAX


def _iteration(backend, postprocess, encoded: bytes) -> tuple[float, float]:
    """Run one full iteration; returns (engine seconds, end-to-end seconds)."""
    started = time.perf_counter()
    image = _decode(encoded)
    batch = preprocess(image)
    before_engine = time.perf_counter()
    box_regression, class_logits = backend(batch)
    after_engine = time.perf_counter()
    postprocess(box_regression, class_logits, image.shape[-2:])
    finished = time.perf_counter()
    return after_engine - before_engine, finished - started


def _summary(seconds: list[float]) -> dict[str, float]:
    samples = np.array(seconds)
    summary = {f"p{q}_ms": float(np.percentile(samples, q) * MS_PER_S) for q in PERCENTILES}
    summary["fps"] = float(1.0 / samples.mean())
    return summary


def _mean_or_unreadable(values: list[int], scale: float = 1.0) -> float | str:
    return float(np.mean(values) / scale) if values else runlog.UNREADABLE


def _count(minimum: int):
    """An argparse type for an integer of at least ``minimum``."""

    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"{value} is below the minimum of {minimum}")
        return value

    return parse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m peddet_orin.benchmark", description=__doc__.splitlines()[0], formatter_class=HelpFormatter
    )
    add_common_options(parser)
    parser.add_argument("--iterations", type=_count(1), default=TIMED_ITERATIONS, help="timed iterations per repeat")
    parser.add_argument("--warmup", type=_count(0), default=WARMUP_ITERATIONS, help="untimed iterations before timing")
    parser.add_argument("--repeats", type=_count(1), default=REPEATS, help="times the timed loop runs after warm-up")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    images_dir = args.data_dir / IMAGES_RELATIVE
    if not images_dir.is_dir():
        usage_error(parser.prog, f"image directory {images_dir} is missing; run python scripts/fetch_coco.py val")
    engine_path = check_engine(parser.prog, args.precision, args.engine)
    pool = image_pool(images_dir)
    if not pool:
        usage_error(parser.prog, f"{images_dir} holds no .jpg images; run python scripts/fetch_coco.py val")

    counts = {"warmup": args.warmup, "timed": args.iterations, "repeats": args.repeats, "batch": 1}
    with runlog.RunLog("benchmark", args.precision, BENCHMARK_SEED, counts) as log:
        order = np.random.default_rng(BENCHMARK_SEED)
        detector = load_detector(pretrained=True)
        postprocess = Postprocessor.from_torchvision(detector)
        backend = load_backend(parser.prog, args.precision, args.device, detector, engine_path)

        for index in range(args.warmup):
            _iteration(backend, postprocess, pool[index % len(pool)])

        baseline = TegrastatsSampler()
        baseline.start()
        time.sleep(BASELINE_SAMPLE_S)
        baseline.stop()

        picks = order.integers(0, len(pool), size=(args.repeats, args.iterations))
        # The one-minute load average shows whether other CPU work was running as timing began.
        load_before_timing = os.getloadavg()[0]
        sampler = TegrastatsSampler()
        sampler.start()
        engine_repeats, total_repeats = [], []
        for repeat_picks in picks:
            engine_seconds, total_seconds = [], []
            for pick in repeat_picks:
                engine, total = _iteration(backend, postprocess, pool[int(pick)])
                engine_seconds.append(engine)
                total_seconds.append(total)
            engine_repeats.append(engine_seconds)
            total_repeats.append(total_seconds)
        sampler.stop()
        release_backend(backend)

        power_samples = len(sampler.values(VDD_IN_PATTERN))
        debug.log(f"tegrastats samples: {power_samples}")
        log.results.update(
            {
                "runtime": backend.label,
                "engine": _summary([sample for repeat in engine_repeats for sample in repeat]),
                "end_to_end": _summary([sample for repeat in total_repeats for sample in repeat]),
                "repeats": args.repeats,
                "engine_p50_ms_per_repeat": [_summary(repeat)["p50_ms"] for repeat in engine_repeats],
                "end_to_end_p50_ms_per_repeat": [_summary(repeat)["p50_ms"] for repeat in total_repeats],
                "peak_process_rss_mib": peak_rss_mib(),
                "system_ram_rise_mib": system_ram_rise_mib(sampler.lines, baseline.lines),
                "avg_power_w": average_power_w(sampler.lines),
                "power_samples": power_samples,
                "power_mode": runlog._command_output(["nvpmodel", "-q"]),
                "cpu_load_1min_before_timing": load_before_timing,
                "gpu_busy_pct_before_timing": gpu_busy_pct(baseline.lines),
                "clocks_locked": CLOCKS_LOCKED_STATUS,
            }
        )
    print(f"{args.precision}: end-to-end p50 {log.results['end_to_end']['p50_ms']:.2f} ms")
    return 0

if __name__ == "__main__":
    sys.exit(main())
