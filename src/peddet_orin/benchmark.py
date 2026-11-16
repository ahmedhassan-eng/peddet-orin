# SPDX-License-Identifier: Apache-2.0
"""Latency, throughput, memory and power for one precision, at batch size 1.

Each timed iteration decodes and preprocesses an image from memory, runs inference
and postprocesses. Engine-only latency is the inference step alone, including the
copies to and from device memory. End-to-end latency is the whole iteration.
Each of the four stages is also timed on its own, from the same perf_counter
readings plus one between decoding and preprocessing. Warm-up passes come first
and are excluded, which also absorbs the first-call JIT.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import resource
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image

from peddet_orin import debug, runlog
from peddet_orin.cli import (
    IMAGES,
    IMAGES_RELATIVE,
    HelpFormatter,
    add_common_options,
    check_device,
    check_engine,
    engine_sha256,
    load_backend,
    release_backend,
    usage_error,
)
from peddet_orin.model import load_detector
from peddet_orin.postprocess import Postprocessor
from peddet_orin.preprocess import PIXEL_MAX, preprocess

# The clocks are read from sysfs, read-only; nothing here sets them. Each file is named in full,
# so the absolute paths in the source are exactly these ten.
GPU_GOVERNOR_FILE = "/sys/class/devfreq/17000000.gpu/governor"
GPU_CUR_FREQ_FILE = "/sys/class/devfreq/17000000.gpu/cur_freq"
GPU_MIN_FREQ_FILE = "/sys/class/devfreq/17000000.gpu/min_freq"
GPU_MAX_FREQ_FILE = "/sys/class/devfreq/17000000.gpu/max_freq"
# Each CPU frequency policy as (name, governor file, current-clock file, maximum file); cpufreq counts in kHz.
CPU_POLICIES: tuple[tuple[str, str, str, str], ...] = (
    (
        "policy0",
        "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor",
        "/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq",
        "/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq",
    ),
    (
        "policy4",
        "/sys/devices/system/cpu/cpufreq/policy4/scaling_governor",
        "/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq",
        "/sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq",
    ),
)
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


# The clocks are sampled by a separate small process, so the timed process does no extra work.
# Each interval it prints one line with a reading per file, or UNREADABLE, until it is terminated.
CLOCK_READER = """
import sys, time
interval, paths = float(sys.argv[1]), sys.argv[2:]
def read(path):
    try:
        with open(path) as handle:
            text = handle.read().strip()
    except OSError:
        text = ""
    return text if text.isdigit() else "UNREADABLE"
while True:
    print(" ".join(read(path) for path in paths), flush=True)
    time.sleep(interval)
"""


class ClockSampler:
    """Sample several clock files from a separate process, between ``start`` and ``stop``."""

    def __init__(self, cur_files: list[str]) -> None:
        self.cur_files = list(cur_files)
        self.samples: list[list[int]] = [[] for _ in self.cur_files]
        self.unreadable = [0 for _ in self.cur_files]
        self.process: subprocess.Popen | None = None
        self.output = None

    def start(self) -> None:
        self.output = tempfile.TemporaryFile(mode="w+")
        interval = str(TEGRASTATS_INTERVAL_MS / MS_PER_S)
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-I", "-c", CLOCK_READER, interval, *self.cur_files],
                stdout=self.output,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            self.process = None

    def stop(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.wait()
        if self.output is not None:
            self.output.seek(0)
            self.add_lines(self.output.read().splitlines())
            self.output.close()
            self.output = None
        if not any(self.samples) and not any(self.unreadable):
            self.read_once()

    def add_lines(self, lines: list[str]) -> None:
        """Take each whole line; a line cut short when the reader was stopped is dropped."""
        for line in lines:
            fields = line.split()
            if len(fields) != len(self.cur_files):
                continue
            for index, field in enumerate(fields):
                if field.isdigit():
                    self.samples[index].append(int(field))
                else:
                    self.unreadable[index] += 1

    def read_once(self) -> None:
        for index, path in enumerate(self.cur_files):
            value = read_int(path)
            if value == runlog.UNREADABLE:
                self.unreadable[index] += 1
            else:
                self.samples[index].append(value)


def read_text(path: str) -> str:
    """A sysfs file's stripped text, or ``UNREADABLE``."""
    try:
        text = Path(path).read_text().strip()
    except OSError:
        return runlog.UNREADABLE
    return text or runlog.UNREADABLE


def read_int(path: str) -> int | str:
    """A sysfs file's integer value, or ``UNREADABLE``."""
    text = read_text(path)
    return int(text) if text.isdigit() else runlog.UNREADABLE


def clock_record(governor_file: str, max_file: str, samples: list[int], unreadable: int, unit: str) -> dict:
    """The governor, the maximum, how often each clock was seen, and how many samples were at the maximum."""
    ceiling = read_int(max_file)
    counts: dict[str, int] = {}
    for value in samples:
        counts[str(value)] = counts.get(str(value), 0) + 1
    at_max = sum(value == ceiling for value in samples) if isinstance(ceiling, int) else runlog.UNREADABLE
    return {
        "governor": read_text(governor_file),
        f"max_{unit}": ceiling,
        f"cur_{unit}_counts": counts,
        "samples": len(samples) + unreadable,
        "unreadable_samples": unreadable,
        "at_max_samples": at_max,
    }


def clock_files() -> list[str]:
    """The current-clock files the reader samples: the GPU's first, then each CPU policy's."""
    return [GPU_CUR_FREQ_FILE, *(cur for _, _, cur, _ in CPU_POLICIES)]


def clock_records(sampler: ClockSampler) -> tuple[dict, dict]:
    """The GPU's record, with its devfreq floor, and one record per CPU policy."""
    gpu = clock_record(GPU_GOVERNOR_FILE, GPU_MAX_FREQ_FILE, sampler.samples[0], sampler.unreadable[0], "hz")
    gpu["min_hz"] = read_int(GPU_MIN_FREQ_FILE)
    cpus = {
        name: clock_record(governor, ceiling, sampler.samples[index], sampler.unreadable[index], "khz")
        for index, (name, governor, _, ceiling) in enumerate(CPU_POLICIES, start=1)
    }
    return gpu, cpus


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


class IterationTimes(NamedTuple):
    """One iteration's seconds: each stage in ``runlog.BENCHMARK_STAGES`` order, then the whole."""

    decode: float
    preprocess: float
    engine: float
    postprocess: float
    end_to_end: float


def _iteration(backend, postprocess, encoded: bytes) -> IterationTimes:
    """Run one full iteration and return its stage and end-to-end seconds."""
    started = time.perf_counter()
    image = _decode(encoded)
    decoded = time.perf_counter()
    batch = preprocess(image)
    before_engine = time.perf_counter()
    box_regression, class_logits = backend(batch)
    after_engine = time.perf_counter()
    postprocess(box_regression, class_logits, image.shape[-2:])
    finished = time.perf_counter()
    return IterationTimes(
        decoded - started,
        before_engine - decoded,
        after_engine - before_engine,
        finished - after_engine,
        finished - started,
    )


def _summary(seconds: list[float]) -> dict[str, float]:
    samples = np.array(seconds)
    summary = {f"p{q}_ms": float(np.percentile(samples, q) * MS_PER_S) for q in PERCENTILES}
    summary["fps"] = float(1.0 / samples.mean())
    return summary


def stage_summaries(timed: list[IterationTimes]) -> dict[str, dict[str, float]]:
    """Each stage's summary over every timed iteration, in the shape of the engine and end-to-end summaries."""
    return {stage: _summary([getattr(times, stage) for times in timed]) for stage in runlog.BENCHMARK_STAGES}


def postprocess_settings(postprocess: Postprocessor) -> dict[str, float | int]:
    """The postprocess settings every timed iteration ran with."""
    return {
        "score_threshold": float(postprocess.score_threshold),
        "nms_iou_threshold": float(postprocess.nms_iou_threshold),
        "top_k_per_class": int(postprocess.top_k_per_class),
        "detections_per_image": int(postprocess.detections_per_image),
        "num_classes": int(postprocess.num_classes),
    }


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
    check_device(parser.prog, args.precision, args.device)

    counts = {"warmup": args.warmup, "timed": args.iterations, "repeats": args.repeats, "batch": 1}
    with runlog.RunLog("benchmark", args.precision, BENCHMARK_SEED, counts) as log:
        engine_digest = engine_sha256(engine_path)
        threads = torch.get_num_threads()
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
        # The one-minute load average shows whether other CPU work was running as timing began. The power mode is
        # read here too, so the log records the mode in force as timing starts.
        load_before_timing = os.getloadavg()[0]
        power_mode = runlog.command_output(["nvpmodel", "-q"])
        sampler = TegrastatsSampler()
        sampler.start()
        clock = ClockSampler(clock_files())
        clock.start()
        engine_repeats, total_repeats, timed = [], [], []
        for repeat_picks in picks:
            engine_seconds, total_seconds = [], []
            for pick in repeat_picks:
                times = _iteration(backend, postprocess, pool[int(pick)])
                engine_seconds.append(times.engine)
                total_seconds.append(times.end_to_end)
                timed.append(times)
            engine_repeats.append(engine_seconds)
            total_repeats.append(total_seconds)
        clock.stop()
        sampler.stop()
        release_backend(backend)
        gpu_clock, cpu_clocks = clock_records(clock)

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
                "power_mode": power_mode,
                "cpu_load_1min_before_timing": load_before_timing,
                "gpu_busy_pct_before_timing": gpu_busy_pct(baseline.lines),
                "gpu_clock": gpu_clock,
                "cpu_clocks": cpu_clocks,
                "stages": stage_summaries(timed),
                "postprocess_settings": postprocess_settings(postprocess),
                "engine_sha256": engine_digest,
                "torch_num_threads": threads,
            }
        )
    print(f"{args.precision}: end-to-end p50 {log.results['end_to_end']['p50_ms']:.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
