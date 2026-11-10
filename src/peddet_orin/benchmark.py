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

import numpy as np
import torch
from PIL import Image

from peddet_orin import debug, runlog
from peddet_orin.evaluate import CUDA, IMAGES, PRECISIONS, make_backend
from peddet_orin.model import load_detector
from peddet_orin.postprocess import Postprocessor
from peddet_orin.preprocess import PIXEL_MAX, preprocess

# Reading or setting the clock state needs root, which this project never uses, so
# this run cannot say whether something else locked them, and it locks nothing itself.
CLOCKS_LOCKED_STATUS = runlog.UNREADABLE
WARMUP_ITERATIONS = 50
TIMED_ITERATIONS = 1000
IMAGE_POOL_SIZE = 200
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
                ["tegrastats", "--interval", str(TEGRASTATS_INTERVAL_MS)],
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
        return [int(match.group(1)) for line in self.lines if (match := pattern.search(line))]


def image_pool() -> list[bytes]:
    """Load the encoded bytes of the first validation images, in file-name order."""
    return [path.read_bytes() for path in sorted(IMAGES.glob("*.jpg"))[:IMAGE_POOL_SIZE]]


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--precision", choices=PRECISIONS, required=True)
    parser.add_argument("--device", default=CUDA, help="torch device for fp32; TensorRT always uses the GPU")
    parser.add_argument("--iterations", type=int, default=TIMED_ITERATIONS)
    parser.add_argument("--warmup", type=int, default=WARMUP_ITERATIONS)
    args = parser.parse_args()

    counts = {"warmup": args.warmup, "timed": args.iterations, "batch": 1}
    with runlog.RunLog("benchmark", args.precision, BENCHMARK_SEED, counts) as log:
        order = np.random.default_rng(BENCHMARK_SEED)
        pool = image_pool()
        detector = load_detector(pretrained=True)
        postprocess = Postprocessor.from_torchvision(detector)
        backend = make_backend(args.precision, args.device, detector)

        for index in range(args.warmup):
            _iteration(backend, postprocess, pool[index % len(pool)])

        baseline = TegrastatsSampler()
        baseline.start()
        time.sleep(BASELINE_SAMPLE_S)
        baseline.stop()

        picks = order.integers(0, len(pool), size=args.iterations)
        sampler = TegrastatsSampler()
        sampler.start()
        engine_seconds, total_seconds = [], []
        for pick in picks:
            engine, total = _iteration(backend, postprocess, pool[int(pick)])
            engine_seconds.append(engine)
            total_seconds.append(total)
        sampler.stop()
        # The one-minute load average shows whether other CPU work shared the timed run.
        one_minute_load = os.getloadavg()[0]

        ram = sampler.values(RAM_PATTERN)
        base_ram = baseline.values(RAM_PATTERN)
        power = sampler.values(VDD_IN_PATTERN)
        debug.log(f"tegrastats samples: {len(power)}")
        log.results.update(
            {
                "runtime": backend.label,
                "engine": _summary(engine_seconds),
                "end_to_end": _summary(total_seconds),
                "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / KIB_PER_MIB,
                "system_ram_rise_mib": (max(ram) - float(np.mean(base_ram))) if ram and base_ram else runlog.UNREADABLE,
                "avg_power_w": _mean_or_unreadable(power, MW_PER_W),
                "power_samples": len(power),
                "gpu_busy_pct_before_timing": _mean_or_unreadable(baseline.values(GPU_BUSY_PATTERN)),
                "cpu_load_1min": one_minute_load,
                "clocks_locked": CLOCKS_LOCKED_STATUS,
                "power_mode": runlog._command_output(["nvpmodel", "-q"]),
            }
        )
    print(f"{args.precision}: end-to-end p50 {log.results['end_to_end']['p50_ms']:.2f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
