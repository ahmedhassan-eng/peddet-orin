# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the PRD requirements that the committed run logs and manifests must meet."""
import hashlib
import json
import re

import pytest

import fetch_coco
import generate_table

LOGS = generate_table.load_logs(generate_table.RUNS_DIR)
PRECISIONS = ("fp32", "fp16", "int8")
SHA256_LINE = re.compile(r"^[0-9a-f]{64}  \d{12}\.jpg$")


def test_r1_an_eval_log_exists_for_each_precision():
    for precision in PRECISIONS:
        log = generate_table.latest(LOGS, "eval", precision)
        assert log is not None, precision
        assert 0 < log["results"]["ap"] <= log["results"]["ap50"] <= 1, precision


def test_r2_the_export_log_is_within_its_tolerance():
    log = generate_table.latest(LOGS, "export", "onnx")
    assert log is not None
    assert log["results"]["max_abs_difference"] <= log["results"]["tolerance"]


def test_r4_build_logs_exist_and_int8_names_its_calibration_path():
    for precision in ("fp16", "int8"):
        log = generate_table.latest(LOGS, "build", precision)
        assert log is not None and log["results"]["engine_bytes"] > 0, precision
    assert generate_table.latest(LOGS, "build", "int8")["results"]["int8_path"]


def test_r5_the_manifest_lists_500_train2017_images_with_hashes():
    lines = fetch_coco.MANIFEST_PATH.read_text().splitlines()
    assert len(lines) == fetch_coco.CALIBRATION_COUNT == 500
    assert all(SHA256_LINE.match(line) for line in lines)
    assert len({line.split()[1] for line in lines}) == 500


def test_r5_a_calibration_hash_mismatch_is_rejected(tmp_path, monkeypatch):
    image = tmp_path / fetch_coco.CALIBRATION_DIR_NAME / "000000000001.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not the recorded image")
    manifest = tmp_path / "calibration.sha256"
    manifest.write_text(f"{hashlib.sha256(b'the recorded image').hexdigest()}  000000000001.jpg\n")
    monkeypatch.setattr(fetch_coco, "DATA_DIR", tmp_path)
    monkeypatch.setattr(fetch_coco, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(fetch_coco, "CALIBRATION_COUNT", 1)
    monkeypatch.setattr(fetch_coco, "_image_names", lambda name: [] if name == fetch_coco.VAL_IMAGE_LIST
                        else ["000000000001.jpg"])
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        fetch_coco.fetch_calibration()


def test_r6_a_benchmark_log_exists_for_each_precision():
    for precision in PRECISIONS:
        log = generate_table.latest(LOGS, "benchmark", precision)
        assert log is not None, precision
        results = log["results"]
        for part in ("engine", "end_to_end"):
            assert {"p50_ms", "p90_ms", "p99_ms", "fps"} <= set(results[part]), (precision, part)
        assert results["peak_process_rss_mib"] > 0 and results["power_samples"] > 0, precision


def test_r10_an_int8_drop_over_two_points_is_stated(tmp_path):
    logs = [dict(log) for log in LOGS]
    fp16 = generate_table.latest(logs, "eval", "fp16")
    int8 = generate_table.latest(logs, "eval", "int8")
    for index, log in enumerate(logs):
        (tmp_path / f"{index:02d}.json").write_text(json.dumps(log))
    assert "more than the" not in generate_table.build_block(tmp_path)
    dropped = dict(int8, results=dict(int8["results"], ap=fp16["results"]["ap"] - 0.025))
    (tmp_path / f"{logs.index(int8):02d}.json").write_text(json.dumps(dropped))
    assert "more than the 2-point limit" in generate_table.build_block(tmp_path)
