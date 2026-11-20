# SPDX-License-Identifier: Apache-2.0
"""scripts/recompute_stats.py regenerates a benchmark log's published figures from its raw side file."""

import hashlib
import json
from pathlib import Path

import numpy as np

import recompute_stats
from peddet_orin import benchmark

REPO_ROOT = Path(__file__).resolve().parents[1]


def _log_with_raw(tmp_path, tamper=False, power=(5000, 5200, 4900)):
    rng = np.random.default_rng(3)
    repeats = [
        [benchmark.IterationTimes(*(float(value) for value in rng.uniform(0.001, 0.05, 4)), 0.0) for _ in range(40)]
        for _ in range(2)
    ]
    repeats = [[times._replace(end_to_end=sum(times[:4])) for times in repeat] for repeat in repeats]
    timed = [times for repeat in repeats for times in repeat]
    power = list(power)
    results = {
        "engine": benchmark._summary([times.engine for times in timed]),
        "end_to_end": benchmark._summary([times.end_to_end for times in timed]),
        "stages": benchmark.stage_summaries(timed),
        "engine_p50_ms_per_repeat": [benchmark._summary([t.engine for t in r])["p50_ms"] for r in repeats],
        "end_to_end_p50_ms_per_repeat": [benchmark._summary([t.end_to_end for t in r])["p50_ms"] for r in repeats],
        "avg_power_w": sum(power) / len(power) / 1000 if power else 7.0,
    }
    raw = benchmark.raw_record(repeats, power)
    if tamper:
        raw["repeats"][0][0][2] += 25.0
    data = json.dumps(raw, separators=(",", ":")).encode()
    (tmp_path / "benchmark_fp16_x.raw").write_bytes(data)
    results.update(raw_file="benchmark_fp16_x.raw", raw_sha256=hashlib.sha256(data).hexdigest())
    path = tmp_path / "benchmark_fp16_x.json"
    path.write_text(json.dumps({"results": results}))
    return path


def test_every_published_figure_recomputes_from_the_raw_file(tmp_path):
    lines = recompute_stats.compare(_log_with_raw(tmp_path))
    assert lines[-1] == "RECOMPUTE DONE PASS", [line for line in lines if line.endswith("DIFF")]
    assert sum(" OK" in line for line in lines) == 2 * 4 + 4 * 4 + 2 * 2 + 1


def test_a_raw_file_that_disagrees_with_the_log_fails(tmp_path):
    path = _log_with_raw(tmp_path, tamper=True)
    lines = recompute_stats.compare(path)
    assert lines[-1].startswith("RECOMPUTE DONE FAIL")
    (tmp_path / "benchmark_fp16_x.raw").write_bytes(b"{}")
    assert recompute_stats.compare(path)[-1] == "RECOMPUTE DONE FAIL 1"


def test_a_committed_log_without_a_raw_file_is_refused_in_one_line(capsys):
    log = sorted((REPO_ROOT / "runs-repeat" / "run6").glob("benchmark_fp16_*.json"))[0]
    assert recompute_stats.main([str(log)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1 and "names no raw_file" in captured.err


def test_a_raw_file_without_power_readings_is_refused_in_one_line(tmp_path, capsys):
    path = _log_with_raw(tmp_path, power=())
    assert recompute_stats.main([str(path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1 and "holds no power readings" in captured.err
