# SPDX-License-Identifier: Apache-2.0
"""scripts/compare_logs.py: a re-measure against the published table, metric by metric."""
import json

import compare_logs
import generate_table
from test_run_logs import latency, make_log

PRECISIONS = ("fp32", "fp16", "int8")


def write(directory, logs):
    directory.mkdir(exist_ok=True)
    for index, log in enumerate(logs):
        (directory / f"{index:02d}.json").write_text(json.dumps(log))
    return directory


def complete(eval_results=None, benchmark_results=None, skip=()):
    """One eval and one benchmark log per precision, with per-precision results merged in."""
    eval_results, benchmark_results = eval_results or {}, benchmark_results or {}
    logs = []
    for precision in PRECISIONS:
        if ("eval", precision) not in skip:
            results = {"ap": 0.3000, "ap50": 0.5000, **eval_results.get(precision, {})}
            logs.append(make_log("eval", precision, results=results))
        if ("benchmark", precision) not in skip:
            results = {"engine": latency(4.0), "end_to_end": latency(10.0), "avg_power_w": 8.0,
                       **benchmark_results.get(precision, {})}
            logs.append(make_log("benchmark", precision, results=results))
    return logs


def published(tmp_path, logs=None):
    """The reference compare() takes, picked from logs as the table picks them."""
    directory = write(tmp_path / "published", complete() if logs is None else logs)
    return generate_table.select(generate_table.load_logs(directory))


def run(tmp_path, new_logs, reference=None):
    new = write(tmp_path / "new", new_logs)
    lines = compare_logs.compare(new, published(tmp_path) if reference is None else reference)
    return (0 if lines[-1] == "COMPARE DONE PASS" else 1), lines


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


def test_a_re_measure_within_every_tolerance_passes(tmp_path):
    status, lines = run(tmp_path, complete(
        {"fp16": {"ap": 0.3010, "ap50": 0.4990}, "int8": {"ap": 0.2950}},
        {"fp16": {"engine": latency(4.4), "end_to_end": latency(8.5), "avg_power_w": 7.2,
                  "system_ram_rise_mib": 900.0}},
    ))
    assert status == 0 and lines[-1] == "COMPARE DONE PASS"
    assert line(lines, "eval", "fp16", "ap") == "COMPARE eval fp16 ap published=30.00 new=30.10 delta=+0.10 tol=0.10 OK"
    assert line(lines, "eval", "int8", "ap").endswith("delta=-0.50 tol=0.50 OK")
    assert line(lines, "benchmark", "fp16", "engine_p50_ms").endswith("delta=+10.0% tol=10% OK")
    assert line(lines, "benchmark", "fp16", "end_to_end_p50_ms").endswith("delta=-15.0% tol=15% OK")
    assert line(lines, "benchmark", "fp16", "avg_power_w").endswith("delta=-10.0% tol=10% OK")
    # p99 and RAM rise are shown, never judged, however far they move.
    assert line(lines, "benchmark", "fp16", "system_ram_rise_mib").endswith("tol=- REPORTED")
    assert line(lines, "benchmark", "fp16", "engine_p99_ms").endswith("tol=- REPORTED")


def test_each_metric_out_of_tolerance_fails(tmp_path):
    status, lines = run(tmp_path, complete(
        {"fp16": {"ap": 0.3011}, "int8": {"ap50": 0.4940}},
        {"fp16": {"engine": latency(4.41), "end_to_end": latency(11.6), "avg_power_w": 8.9}},
    ))
    assert status == 1 and lines[-1] == "COMPARE DONE FAIL 5"
    for kind, precision, metric in [("eval", "fp16", "ap"), ("eval", "int8", "ap50"),
                                    ("benchmark", "fp16", "engine_p50_ms"),
                                    ("benchmark", "fp16", "end_to_end_p50_ms"), ("benchmark", "fp16", "avg_power_w")]:
        assert line(lines, kind, precision, metric).endswith(" OUT"), metric


def test_a_log_with_no_published_counterpart_is_out(tmp_path):
    reference = published(tmp_path, complete(skip={("eval", "fp32")}))
    status, lines = run(tmp_path, complete({"fp32": {"ap": 0.25}}), reference)
    assert status == 1
    expected = "COMPARE eval fp32 ap published=missing new=25.00 delta=n/a tol=0.10 OUT"
    assert line(lines, "eval", "fp32", "ap") == expected


def test_the_newest_clean_new_log_is_the_one_compared(tmp_path):
    status, lines = run(tmp_path, [
        make_log("eval", "fp16", results={"ap": 0.2000}, end_utc="2026-01-01T00:09:00Z"),
        *complete(),
        make_log("eval", "fp16", results={"ap": 0.1000}, end_utc="2026-01-01T00:11:00Z", dirty=True),
    ])
    assert status == 0 and line(lines, "eval", "fp16", "ap").endswith("new=30.00 delta=+0.00 tol=0.10 OK")


def test_an_incomplete_new_set_fails(tmp_path):
    status, lines = run(tmp_path, complete(skip={("benchmark", "int8")}))
    assert status == 1 and lines[-1] == "COMPARE DONE FAIL 1"
    assert "COMPARE benchmark int8 log published=- new=missing delta=n/a tol=- OUT" in lines


def test_an_empty_new_run_fails_once_per_missing_log(tmp_path):
    status, lines = run(tmp_path, [])
    assert status == 1 and lines[-1] == "COMPARE DONE FAIL 6"


def test_the_default_reference_is_what_the_published_table_is_built_from():
    reference = compare_logs.reference()
    block = generate_table.readme_block(generate_table.README.read_text())
    for precision in PRECISIONS:
        assert generate_table._row(reference, precision) in block.splitlines(), precision
    # The earlier run in runs/ gives other cells, so a reference taken from it would fail the loop above.
    earlier = generate_table.select(generate_table.load_logs(generate_table.RUNS_DIR))
    assert any(generate_table._row(earlier, precision) not in block for precision in PRECISIONS)


def test_main_compares_against_the_reference_and_sets_the_status(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(compare_logs, "reference", lambda: published(tmp_path))
    new = write(tmp_path / "new", complete())
    assert compare_logs.main([str(new)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "COMPARE DONE PASS"
    assert compare_logs.main([str(write(tmp_path / "partial", complete(skip={("eval", "fp32")})))]) == 1
