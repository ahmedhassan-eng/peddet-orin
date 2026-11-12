# SPDX-License-Identifier: Apache-2.0
"""Benchmark statistics, tegrastats parsing and the result record, checked on CPU with known answers."""

import itertools
import resource
import types

import pytest
import torch
from PIL import Image

from peddet_orin import benchmark, runlog
from peddet_orin.model import load_detector
from peddet_orin.postprocess import Postprocessor

# Samples 1, 2, 3, 4 and 10 ms: numpy's linear percentiles are 3.0, 7.6 and 9.76 ms, the mean is 4 ms.
SAMPLES_S = [0.001, 0.002, 0.003, 0.004, 0.010]
# tegrastats lines from two releases: GR3D_FREQ with and without the @[frequency] suffix.
TEGRASTATS_LINES = [
    "RAM 3000/7620MB (lfb 8x4MB) SWAP 0/3810MB CPU [5%@1510] GR3D_FREQ 40%@[624] VDD_IN 5000mW/5000mW\n",
    "RAM 3300/7620MB (lfb 8x4MB) SWAP 0/3810MB CPU [5%@1510] GR3D_FREQ 60% VDD_IN 7000mW/6000mW\n",
    "RAM 3100/7620MB (lfb 8x4MB) SWAP 0/3810MB CPU [5%@1510] GR3D_FREQ 80%@[624,624] VDD_IN 6000mW/6000mW\n",
]
BASELINE_LINES = [
    "RAM 2900/7620MB (lfb 8x4MB) GR3D_FREQ 0% VDD_IN 4000mW/4000mW\n",
    "RAM 3000/7620MB (lfb 8x4MB) GR3D_FREQ 10%@[305] VDD_IN 4000mW/4000mW\n",
]
RESULT_KEYS = {
    "runtime",
    "engine",
    "end_to_end",
    "repeats",
    "engine_p50_ms_per_repeat",
    "end_to_end_p50_ms_per_repeat",
    "peak_process_rss_mib",
    "system_ram_rise_mib",
    "avg_power_w",
    "power_samples",
    "power_mode",
    "cpu_load_1min_before_timing",
    "gpu_busy_pct_before_timing",
    "clocks_locked",
}
SUMMARY_KEYS = {"p50_ms", "p90_ms", "p99_ms", "fps"}


def test_benchmark_summary_matches_hand_computed_values():
    summary = benchmark._summary(SAMPLES_S)
    assert summary == {
        "p50_ms": pytest.approx(3.0),
        "p90_ms": pytest.approx(7.6),
        "p99_ms": pytest.approx(9.76),
        "fps": pytest.approx(250.0),  # 1 / mean; the minimum would give 1000.
    }


def test_benchmark_power_is_mean_vdd_in_in_watts():
    assert benchmark.average_power_w(TEGRASTATS_LINES) == pytest.approx(6.0)
    assert benchmark.average_power_w([]) == runlog.UNREADABLE


def test_benchmark_gpu_busy_reads_both_gr3d_formats():
    assert benchmark.tegrastats_values(TEGRASTATS_LINES, benchmark.GPU_BUSY_PATTERN) == [40, 60, 80]
    assert benchmark.gpu_busy_pct(BASELINE_LINES) == pytest.approx(5.0)


def test_benchmark_ram_rise_is_timed_peak_minus_baseline_mean():
    # Peak 3300 MB while timing, baseline mean 2950 MB.
    assert benchmark.system_ram_rise_mib(TEGRASTATS_LINES, BASELINE_LINES) == pytest.approx(350.0)
    assert benchmark.system_ram_rise_mib([], BASELINE_LINES) == runlog.UNREADABLE


def test_benchmark_peak_rss_converts_kib_to_mib(monkeypatch):
    monkeypatch.setattr(resource, "getrusage", lambda who: types.SimpleNamespace(ru_maxrss=3 * 1024 * 1024))
    assert benchmark.peak_rss_mib() == pytest.approx(3072.0)


class FakeBackend:
    """Raw outputs that postprocess to no detections, shaped for SSDLite's anchors."""

    label = "fake-cpu"

    def __init__(self, detector) -> None:
        anchors = Postprocessor.from_torchvision(detector).anchors.shape[0]
        self.box_regression = torch.zeros(1, anchors, 4)
        self.class_logits = torch.zeros(1, anchors, 91)
        self.class_logits[..., 0] = 20.0  # Background everywhere.

    def __call__(self, batch):
        return self.box_regression, self.class_logits


@pytest.fixture
def cpu_benchmark(tmp_path, monkeypatch):
    """Run benchmark.main on CPU with tiny images, no tegrastats and no log file; return the results."""
    images = tmp_path / "val2017"
    images.mkdir()
    for number in range(3):
        Image.new("RGB", (40 + number, 30)).save(images / f"{number:012d}.jpg")
    written = {}

    def capture(self):
        written["record"] = self
        return tmp_path / "log.json"

    monkeypatch.setattr(runlog.RunLog, "_write", capture)
    monkeypatch.setattr(runlog, "other_gpu_process_count", lambda: 0)
    monkeypatch.setattr(runlog, "_command_output", lambda arguments: "MODE TEST")
    monkeypatch.setattr(benchmark, "TEGRASTATS", "peddet-orin-absent-tegrastats")
    monkeypatch.setattr(benchmark, "BASELINE_SAMPLE_S", 0.0)
    monkeypatch.setattr(benchmark, "load_detector", lambda pretrained: load_detector(pretrained=False))
    monkeypatch.setattr(
        benchmark, "load_backend", lambda prog, precision, device, detector, engine: FakeBackend(detector)
    )

    def run(*extra: str) -> dict:
        arguments = ["--precision", "fp32", "--device", "cpu", "--data-dir", str(tmp_path)]
        assert benchmark.main([*arguments, "--warmup", "1", "--iterations", "2", "--repeats", "2", *extra]) == 0
        return written["record"].results

    return run


def test_benchmark_result_record_has_the_schema_keys(cpu_benchmark):
    results = cpu_benchmark()
    assert set(results) == RESULT_KEYS
    assert set(results["engine"]) == SUMMARY_KEYS
    assert set(results["end_to_end"]) == SUMMARY_KEYS
    assert results["repeats"] == 2
    for key in ("engine_p50_ms_per_repeat", "end_to_end_p50_ms_per_repeat"):
        assert len(results[key]) == 2
        assert all(isinstance(value, float) for value in results[key])
    assert results["runtime"] == "fake-cpu"
    assert results["clocks_locked"] == "NOT_PROBED"
    assert isinstance(results["cpu_load_1min_before_timing"], float)
    assert isinstance(results["peak_process_rss_mib"], float)
    assert results["power_samples"] == 0
    assert results["avg_power_w"] == runlog.UNREADABLE
    assert results["power_mode"] == "MODE TEST"


def test_benchmark_pools_repeats_and_reads_load_before_timing(cpu_benchmark, monkeypatch):
    events = []
    # One warm-up at 100 ms, then four timed iterations at 1, 2, 3 and 4 ms (end-to-end twice that).
    durations = itertools.chain([0.1], [0.001, 0.002, 0.003, 0.004])

    def fake_iteration(backend, postprocess, encoded):
        events.append("iteration")
        seconds = next(durations)
        return seconds, 2 * seconds

    def fake_load():
        events.append("load")
        return (1.25, 1.0, 1.0)

    monkeypatch.setattr(benchmark, "_iteration", fake_iteration)
    monkeypatch.setattr(benchmark.os, "getloadavg", fake_load)
    results = cpu_benchmark()
    assert results["engine"]["p50_ms"] == pytest.approx(2.5)
    assert results["engine"]["fps"] == pytest.approx(400.0)
    assert results["end_to_end"]["p99_ms"] == pytest.approx(2 * (3 + 0.97))
    assert results["engine_p50_ms_per_repeat"] == [pytest.approx(1.5), pytest.approx(3.5)]
    assert results["end_to_end_p50_ms_per_repeat"] == [pytest.approx(3.0), pytest.approx(7.0)]
    assert events == ["iteration", "load", "iteration", "iteration", "iteration", "iteration"]
    assert results["cpu_load_1min_before_timing"] == 1.25
