# SPDX-License-Identifier: Apache-2.0
"""Benchmark statistics, tegrastats parsing and the result record, checked on CPU with known answers."""

import hashlib
import itertools
import os
import resource
import time
import types
from pathlib import Path

import pytest
import torch
from PIL import Image

from peddet_orin import benchmark, runlog
from peddet_orin import preprocess as preprocess_module
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
    "gpu_clock",
    "cpu_clocks",
    "stages",
    "postprocess_settings",
    "engine_sha256",
    "torch_num_threads",
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


GPU_FILES = ("GPU_GOVERNOR_FILE", "GPU_CUR_FREQ_FILE", "GPU_MIN_FREQ_FILE", "GPU_MAX_FREQ_FILE")


def fake_clocks(tmp_path, gpu=(None, None, None, None), cpus=None) -> dict:
    """Stand-in sysfs files under tmp_path; None leaves a file missing. Returns the names to patch."""
    root = tmp_path / "clocks"
    root.mkdir(exist_ok=True)
    patched = {}
    for name, value in zip(GPU_FILES, gpu, strict=True):
        path = root / name.lower()
        if value is not None:
            path.write_text(f"{value}\n")
        patched[name] = str(path)
    policies = []
    for policy, values in (cpus or {}).items():
        files = []
        for kind, value in zip(("governor", "cur", "max"), values, strict=True):
            path = root / f"{policy}_{kind}"
            if value is not None:
                path.write_text(f"{value}\n")
            files.append(str(path))
        policies.append((policy, *files))
    patched["CPU_POLICIES"] = tuple(policies)
    return patched


def _patch(monkeypatch, patched):
    for name, value in patched.items():
        monkeypatch.setattr(benchmark, name, value)


BOARD = {
    "gpu": ("performance", 1020000000, 306000000, 1020000000),
    "cpus": {"policy0": ("schedutil", 1728000, 1728000), "policy4": ("schedutil", 729600, 1728000)},
}


def test_clock_records_count_samples_at_max_per_clock(tmp_path, monkeypatch):
    _patch(monkeypatch, fake_clocks(tmp_path, BOARD["gpu"], BOARD["cpus"]))
    sampler = benchmark.ClockSampler(benchmark.clock_files())
    sampler.read_once()
    Path(benchmark.GPU_CUR_FREQ_FILE).write_text("624000000\n")
    sampler.read_once()
    gpu, cpus = benchmark.clock_records(sampler)
    assert gpu == {
        "governor": "performance",
        "max_hz": 1020000000,
        "min_hz": 306000000,
        "cur_hz_counts": {"1020000000": 1, "624000000": 1},
        "samples": 2,
        "unreadable_samples": 0,
        "at_max_samples": 1,
    }
    assert cpus["policy0"]["at_max_samples"] == 2 and cpus["policy0"]["max_khz"] == 1728000
    assert cpus["policy4"]["at_max_samples"] == 0 and cpus["policy4"]["cur_khz_counts"] == {"729600": 2}
    assert runlog.RESULT_CHECKS["benchmark"]["gpu_clock"](gpu)
    assert runlog.RESULT_CHECKS["benchmark"]["cpu_clocks"](cpus)


def test_clock_records_missing_files_are_unreadable(tmp_path, monkeypatch):
    _patch(monkeypatch, fake_clocks(tmp_path, cpus={"policy0": (None, None, None)}))
    sampler = benchmark.ClockSampler(benchmark.clock_files())
    sampler.start()
    sampler.stop()
    gpu, cpus = benchmark.clock_records(sampler)
    for record, unit in ((gpu, "hz"), (cpus["policy0"], "khz")):
        assert record["governor"] == record[f"max_{unit}"] == runlog.UNREADABLE
        assert record[f"cur_{unit}_counts"] == {} and record["unreadable_samples"] == record["samples"] >= 1
        assert record["at_max_samples"] == runlog.UNREADABLE
    assert runlog.RESULT_CHECKS["benchmark"]["gpu_clock"](gpu)
    assert runlog.RESULT_CHECKS["benchmark"]["cpu_clocks"](cpus)


def test_clock_reader_process_samples_every_file_until_stopped(tmp_path, monkeypatch):
    """The reader runs in its own process and keeps sampling while the caller only waits."""
    _patch(monkeypatch, fake_clocks(tmp_path, BOARD["gpu"], BOARD["cpus"]))
    monkeypatch.setattr(benchmark, "TEGRASTATS_INTERVAL_MS", 20)
    sampler = benchmark.ClockSampler(benchmark.clock_files())
    sampler.start()
    assert sampler.process is not None and sampler.process.pid != os.getpid()
    time.sleep(1.0)
    sampler.stop()
    assert len(sampler.samples) == 3 and all(len(samples) >= 5 for samples in sampler.samples)
    assert set(sampler.samples[0]) == {1020000000} and set(sampler.samples[2]) == {729600}
    assert sampler.unreadable == [0, 0, 0]


def test_clock_reader_lines_are_parsed_and_short_lines_dropped():
    sampler = benchmark.ClockSampler(["a", "b"])
    sampler.add_lines(["1020000000 1728000", "UNREADABLE 1728000", "624000000", ""])
    assert sampler.samples == [[1020000000], [1728000, 1728000]] and sampler.unreadable == [1, 0]


def test_clock_paths_are_the_ten_sysfs_files():
    """The source names exactly these ten absolute paths, and no other form of them."""
    assert benchmark.GPU_GOVERNOR_FILE == "/sys/class/devfreq/17000000.gpu/governor"
    assert benchmark.GPU_CUR_FREQ_FILE == "/sys/class/devfreq/17000000.gpu/cur_freq"
    assert benchmark.GPU_MIN_FREQ_FILE == "/sys/class/devfreq/17000000.gpu/min_freq"
    assert benchmark.GPU_MAX_FREQ_FILE == "/sys/class/devfreq/17000000.gpu/max_freq"
    assert benchmark.CPU_POLICIES == (
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
    monkeypatch.setattr(runlog, "command_output", lambda arguments: "MODE TEST")
    monkeypatch.setattr(benchmark, "TEGRASTATS", "peddet-orin-absent-tegrastats")
    monkeypatch.setattr(benchmark, "BASELINE_SAMPLE_S", 0.0)
    # Never read the host's own devfreq node in a test: point all four files into tmp_path.
    _patch(monkeypatch, fake_clocks(tmp_path, BOARD["gpu"], BOARD["cpus"]))
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
    clock = results["gpu_clock"]
    assert clock["governor"] == "performance" and (clock["min_hz"], clock["max_hz"]) == (306000000, 1020000000)
    assert clock["at_max_samples"] == clock["samples"] >= 1 and clock["unreadable_samples"] == 0
    assert set(results["cpu_clocks"]) == {"policy0", "policy4"}
    assert results["cpu_clocks"]["policy4"]["at_max_samples"] == 0
    assert isinstance(results["cpu_load_1min_before_timing"], float)
    assert isinstance(results["peak_process_rss_mib"], float)
    assert results["power_samples"] == 0
    assert results["avg_power_w"] == runlog.UNREADABLE
    assert results["power_mode"] == "MODE TEST"


def test_benchmark_pools_repeats_and_reads_load_and_power_mode_before_timing(cpu_benchmark, monkeypatch):
    events = []
    # One warm-up at 100 ms, then four timed iterations at 1, 2, 3 and 4 ms (end-to-end twice that).
    durations = itertools.chain([0.1], [0.001, 0.002, 0.003, 0.004])

    def fake_iteration(backend, postprocess, encoded):
        events.append("iteration")
        seconds = next(durations)
        return benchmark.IterationTimes(0.5 * seconds, 0.25 * seconds, seconds, 0.25 * seconds, 2 * seconds)

    def fake_load():
        events.append("load")
        return (1.25, 1.0, 1.0)

    monkeypatch.setattr(benchmark, "_iteration", fake_iteration)

    def fake_power_mode(arguments):
        events.append("power_mode")
        return "MODE TEST"

    monkeypatch.setattr(benchmark.os, "getloadavg", fake_load)
    monkeypatch.setattr(runlog, "command_output", fake_power_mode)
    results = cpu_benchmark()
    assert results["engine"]["p50_ms"] == pytest.approx(2.5)
    assert results["engine"]["fps"] == pytest.approx(400.0)
    assert results["end_to_end"]["p99_ms"] == pytest.approx(2 * (3 + 0.97))
    assert results["engine_p50_ms_per_repeat"] == [pytest.approx(1.5), pytest.approx(3.5)]
    assert results["end_to_end_p50_ms_per_repeat"] == [pytest.approx(3.0), pytest.approx(7.0)]
    # The load average and the power mode are both read after warm-up and before the first timed iteration.
    assert events == ["iteration", "load", "power_mode", "iteration", "iteration", "iteration", "iteration"]
    assert results["cpu_load_1min_before_timing"] == 1.25
    # Each stage is summarised over all four timed iterations; the engine stage is the engine summary itself.
    stages = results["stages"]
    assert list(stages) == ["decode", "preprocess", "engine", "postprocess"]
    assert stages["engine"] == results["engine"]
    assert stages["decode"]["p50_ms"] == pytest.approx(1.25)
    assert stages["preprocess"]["p50_ms"] == stages["postprocess"]["p50_ms"] == pytest.approx(0.625)
    assert stages["postprocess"]["fps"] == pytest.approx(1600.0)


def test_each_stage_is_timed_between_its_own_perf_counter_readings(monkeypatch):
    """Five readings bound the four stages; engine and end-to-end are computed as before."""
    readings = iter([10.0, 11.0, 13.0, 16.0, 20.0])
    monkeypatch.setattr(benchmark, "time", types.SimpleNamespace(perf_counter=lambda: next(readings)))
    calls = []
    monkeypatch.setattr(benchmark, "_decode", lambda encoded: calls.append("decode") or torch.zeros(3, 4, 5))
    monkeypatch.setattr(benchmark, "preprocess", lambda image, **options: calls.append("preprocess") or image)

    def backend(batch):
        calls.append("engine")
        return "boxes", "logits"

    times = benchmark._iteration(backend, lambda *outputs: calls.append("postprocess"), b"jpeg")
    assert calls == ["decode", "preprocess", "engine", "postprocess"]
    assert times == benchmark.IterationTimes(1.0, 2.0, 3.0, 4.0, 10.0)
    assert times.decode + times.preprocess + times.engine + times.postprocess == times.end_to_end


def test_benchmark_records_stage_summaries_and_postprocess_settings(cpu_benchmark):
    results = cpu_benchmark()
    assert set(results["stages"]) == set(runlog.BENCHMARK_STAGES)
    assert all(set(summary) == SUMMARY_KEYS for summary in results["stages"].values())
    detector = load_detector(pretrained=False)
    assert results["postprocess_settings"] == {
        "score_threshold": detector.score_thresh,
        "nms_iou_threshold": detector.nms_thresh,
        "top_k_per_class": detector.topk_candidates,
        "detections_per_image": detector.detections_per_img,
        "num_classes": 91,
    }
    for key in ("stages", "postprocess_settings"):
        assert runlog.OPTIONAL_CHECKS["benchmark"][key](results[key])


def test_benchmark_logs_the_engine_hash_for_tensorrt_and_none_for_fp32(cpu_benchmark, tmp_path):
    assert cpu_benchmark()["engine_sha256"] == "-"
    engine = tmp_path / "fp16.engine"
    engine.write_bytes(b"fp16 engine")
    results = cpu_benchmark("--precision", "fp16", "--engine", str(engine))
    assert results["engine_sha256"] == hashlib.sha256(b"fp16 engine").hexdigest()
    results = cpu_benchmark("--precision", "fp32-trt", "--engine", str(engine))
    assert results["engine_sha256"] == hashlib.sha256(b"fp16 engine").hexdigest()


def test_benchmark_logs_torchs_cpu_thread_count(cpu_benchmark, monkeypatch):
    monkeypatch.setattr(benchmark.torch, "get_num_threads", lambda: 3)
    assert cpu_benchmark()["torch_num_threads"] == 3


def test_the_full_image_check_runs_once_per_pooled_image_and_never_while_timing(cpu_benchmark, monkeypatch):
    """Three pooled images, one warm-up and four timed iterations: the range check runs three times."""
    checked = []
    full_check = preprocess_module.check_pixel_range

    def counting_check(image):
        checked.append(image)
        full_check(image)

    monkeypatch.setattr(preprocess_module, "check_pixel_range", counting_check)
    iterations = []
    real_iteration = benchmark._iteration

    def counting_iteration(*arguments):
        iterations.append(len(checked))
        return real_iteration(*arguments)

    monkeypatch.setattr(benchmark, "_iteration", counting_iteration)
    cpu_benchmark()
    assert len(checked) == 3
    assert iterations == [3] * 5


def test_the_timed_iteration_checks_only_the_layout(tmp_path, monkeypatch):
    def refuse(image):
        raise AssertionError("the full-frame range check ran inside the timed iteration")

    monkeypatch.setattr(preprocess_module, "check_image", refuse)
    monkeypatch.setattr(preprocess_module, "check_pixel_range", refuse)
    detector = load_detector(pretrained=False)
    Image.new("RGB", (40, 30)).save(tmp_path / "image.jpg")
    encoded = (tmp_path / "image.jpg").read_bytes()
    times = benchmark._iteration(FakeBackend(detector), Postprocessor.from_torchvision(detector), encoded)
    assert times.end_to_end > 0
    monkeypatch.setattr(benchmark, "_decode", lambda encoded: torch.zeros(30, 40, 3))
    with pytest.raises(ValueError, match="3 channels first"):
        benchmark._iteration(FakeBackend(detector), Postprocessor.from_torchvision(detector), encoded)


def test_a_pooled_image_out_of_range_is_named(monkeypatch):
    monkeypatch.setattr(benchmark, "_decode", lambda encoded: torch.full((3, 4, 5), float(len(encoded))))
    benchmark.check_pool([b"x"])
    with pytest.raises(ValueError, match=r"pooled image 1: expected values in \[0, 1\], got values from 2 to 2"):
        benchmark.check_pool([b"x", b"xy"])


def test_the_deploy_postprocess_writes_its_own_log_kind(cpu_benchmark, monkeypatch):
    written = []
    capture = runlog.RunLog._write

    def recording(self):
        written.append(self)
        return capture(self)

    monkeypatch.setattr(runlog.RunLog, "_write", recording)
    results = cpu_benchmark("--postprocess", "deploy")
    assert written[0].kind == "benchmark_deploy" and set(results) == RESULT_KEYS
    assert results["postprocess_settings"]["kept_labels"] == [1]
    assert results["postprocess_settings"]["score_threshold"] == 0.5
    assert runlog.RESULT_CHECKS["benchmark_deploy"]["postprocess_settings"](results["postprocess_settings"])
    cpu_benchmark()
    assert written[1].kind == "benchmark" and "kept_labels" not in written[1].results["postprocess_settings"]
    assert benchmark.build_parser().parse_args(["--precision", "fp16"]).postprocess == "standard"
