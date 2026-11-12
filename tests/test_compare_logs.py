# SPDX-License-Identifier: Apache-2.0
"""scripts/compare_logs.py: a re-measure against the committed logs, metric by metric."""
import json

import compare_logs
from test_run_logs import latency, make_log


def write(directory, logs):
    directory.mkdir(exist_ok=True)
    for index, log in enumerate(logs):
        (directory / f"{index:02d}.json").write_text(json.dumps(log))
    return directory


def published(tmp_path):
    return write(tmp_path / "published", [
        make_log("eval", "fp16", results={"ap": 0.3000, "ap50": 0.5000}),
        make_log("eval", "int8", results={"ap": 0.2900, "ap50": 0.4900}),
        make_log("benchmark", "fp16", results={"engine": latency(4.0), "end_to_end": latency(10.0),
                                               "avg_power_w": 8.0}),
    ])


def run(tmp_path, new_logs, capsys):
    new = write(tmp_path / "new", new_logs)
    status = compare_logs.main([str(new), "--against", str(published(tmp_path))])
    return status, capsys.readouterr().out.splitlines()


def line(lines, kind, precision, metric):
    return next(item for item in lines if item.startswith(f"COMPARE {kind} {precision} {metric} "))


def test_the_tolerances_are_one_dict_with_the_stated_values():
    assert compare_logs.TOLERANCES == {
        "ap": {"points": {"fp32": 0.10, "fp16": 0.10, "int8": 0.50}},
        "ap50": {"points": {"fp32": 0.10, "fp16": 0.10, "int8": 0.50}},
        "engine_p50_ms": {"relative": 0.10},
        "end_to_end_p50_ms": {"relative": 0.15},
        "avg_power_w": {"relative": 0.10},
    }


def test_a_re_measure_within_every_tolerance_passes(tmp_path, capsys):
    status, lines = run(tmp_path, [
        make_log("eval", "fp16", results={"ap": 0.3010, "ap50": 0.4990}),
        make_log("eval", "int8", results={"ap": 0.2850, "ap50": 0.4900}),
        make_log("benchmark", "fp16", results={"engine": latency(4.4), "end_to_end": latency(8.5),
                                               "avg_power_w": 7.2, "system_ram_rise_mib": 900.0}),
    ], capsys)
    assert status == 0 and lines[-1] == "COMPARE DONE PASS"
    assert line(lines, "eval", "fp16", "ap") == "COMPARE eval fp16 ap published=30.00 new=30.10 delta=+0.10 tol=0.10 OK"
    assert line(lines, "eval", "int8", "ap").endswith("delta=-0.50 tol=0.50 OK")
    assert line(lines, "benchmark", "fp16", "engine_p50_ms").endswith("delta=+10.0% tol=10% OK")
    assert line(lines, "benchmark", "fp16", "end_to_end_p50_ms").endswith("delta=-15.0% tol=15% OK")
    assert line(lines, "benchmark", "fp16", "avg_power_w").endswith("delta=-10.0% tol=10% OK")
    # p99 and RAM rise are shown, never judged, however far they move.
    assert line(lines, "benchmark", "fp16", "system_ram_rise_mib").endswith("tol=- REPORTED")
    assert line(lines, "benchmark", "fp16", "engine_p99_ms").endswith("tol=- REPORTED")


def test_each_metric_out_of_tolerance_fails(tmp_path, capsys):
    status, lines = run(tmp_path, [
        make_log("eval", "fp16", results={"ap": 0.3011, "ap50": 0.5000}),
        make_log("eval", "int8", results={"ap": 0.2900, "ap50": 0.4840}),
        make_log("benchmark", "fp16", results={"engine": latency(4.41), "end_to_end": latency(11.6),
                                               "avg_power_w": 8.9}),
    ], capsys)
    assert status == 1 and lines[-1] == "COMPARE DONE FAIL 5"
    for kind, precision, metric in [("eval", "fp16", "ap"), ("eval", "int8", "ap50"),
                                    ("benchmark", "fp16", "engine_p50_ms"),
                                    ("benchmark", "fp16", "end_to_end_p50_ms"), ("benchmark", "fp16", "avg_power_w")]:
        assert line(lines, kind, precision, metric).endswith(" OUT"), metric


def test_a_log_with_no_published_counterpart_is_out(tmp_path, capsys):
    status, lines = run(tmp_path, [make_log("eval", "fp32")], capsys)
    assert status == 1
    expected = "COMPARE eval fp32 ap published=missing new=25.00 delta=n/a tol=0.10 OUT"
    assert line(lines, "eval", "fp32", "ap") == expected


def test_the_newest_clean_new_log_is_the_one_compared(tmp_path, capsys):
    status, lines = run(tmp_path, [
        make_log("eval", "fp16", results={"ap": 0.2000}, end_utc="2026-01-01T00:09:00Z"),
        make_log("eval", "fp16", results={"ap": 0.3000, "ap50": 0.5000}),
        make_log("eval", "fp16", results={"ap": 0.1000}, end_utc="2026-01-01T00:10:00Z", dirty=True),
    ], capsys)
    assert status == 0 and line(lines, "eval", "fp16", "ap").endswith("new=30.00 delta=+0.00 tol=0.10 OK")


def test_an_empty_new_run_fails(tmp_path, capsys):
    status, lines = run(tmp_path, [], capsys)
    assert status == 1 and lines[-1] == "COMPARE DONE FAIL 1"
