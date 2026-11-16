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
            results = {
                "engine": latency(4.0),
                "end_to_end": latency(10.0),
                "avg_power_w": 8.0,
                **benchmark_results.get(precision, {}),
            }
            logs.append(make_log("benchmark", precision, results=results))
    return logs


def published(tmp_path, logs=None):
    """The reference compare() takes, picked from logs as the table picks them."""
    directory = write(tmp_path / "published", complete() if logs is None else logs)
    return generate_table.select(generate_table.load_logs(directory))


def run(tmp_path, new_logs, reference=None, committed=frozenset()):
    new = write(tmp_path / "new", new_logs)
    lines = compare_logs.compare(new, published(tmp_path) if reference is None else reference, set(committed))
    return (0 if lines[-1] == "COMPARE DONE PASS" else 1), lines


def line(lines, kind, precision, metric):
    return next(item for item in lines if item.startswith(f"COMPARE {kind} {precision} {metric} "))


def test_the_tolerances_are_one_dict_with_the_stated_values():
    assert compare_logs.TOLERANCES == {
        "ap": {"points": {"fp32": 0.10, "fp32-trt": 0.10, "fp16": 0.10, "int8": 0.50}},
        "ap50": {"points": {"fp32": 0.10, "fp32-trt": 0.10, "fp16": 0.10, "int8": 0.50}},
        "engine_p50_ms": {"relative": 0.10},
        "end_to_end_p50_ms": {"relative": 0.15},
        "avg_power_w": {"relative": 0.10},
    }


def test_a_re_measure_within_every_tolerance_passes(tmp_path):
    status, lines = run(
        tmp_path,
        complete(
            {"fp16": {"ap": 0.3010, "ap50": 0.4990}, "int8": {"ap": 0.2950}},
            {
                "fp16": {
                    "engine": latency(4.4),
                    "end_to_end": latency(8.5),
                    "avg_power_w": 7.2,
                    "system_ram_rise_mib": 900.0,
                }
            },
        ),
    )
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
    status, lines = run(
        tmp_path,
        complete(
            {"fp16": {"ap": 0.3011}, "int8": {"ap50": 0.4940}},
            {"fp16": {"engine": latency(4.41), "end_to_end": latency(11.6), "avg_power_w": 8.9}},
        ),
    )
    assert status == 1 and lines[-1] == "COMPARE DONE FAIL 5"
    for kind, precision, metric in [
        ("eval", "fp16", "ap"),
        ("eval", "int8", "ap50"),
        ("benchmark", "fp16", "engine_p50_ms"),
        ("benchmark", "fp16", "end_to_end_p50_ms"),
        ("benchmark", "fp16", "avg_power_w"),
    ]:
        assert line(lines, kind, precision, metric).endswith(" OUT"), metric


def test_a_log_with_no_published_counterpart_is_out(tmp_path):
    reference = published(tmp_path, complete(skip={("eval", "fp32")}))
    status, lines = run(tmp_path, complete({"fp32": {"ap": 0.25}}), reference)
    assert status == 1
    expected = "COMPARE eval fp32 ap published=missing new=25.00 delta=n/a tol=0.10 OUT"
    assert line(lines, "eval", "fp32", "ap") == expected


def test_the_newest_new_log_is_the_one_compared(tmp_path):
    status, lines = run(
        tmp_path,
        [
            make_log("eval", "fp16", results={"ap": 0.2000}, end_utc="2026-01-01T00:09:00Z"),
            *complete(),
        ],
    )
    assert status == 0 and line(lines, "eval", "fp16", "ap").endswith("new=30.00 delta=+0.00 tol=0.10 OK")


def test_a_dirty_newest_log_fails_instead_of_falling_back_to_an_older_one(tmp_path):
    dirty = make_log("eval", "fp16", results={"ap": 0.1000}, end_utc="2026-01-01T00:11:00Z", dirty=True)
    status, lines = run(tmp_path, [*complete(), dirty])
    assert status == 1 and lines[-1] == "COMPARE DONE FAIL 1"
    assert "COMPARE eval fp16 log published=- new=dirty delta=n/a tol=- OUT" in lines


def test_committed_logs_never_stand_in_for_a_re_measure(tmp_path):
    committed_logs = complete()
    identities = {compare_logs._identity(log) for log in committed_logs}
    # A fresh clone's runs/ holding only the committed logs: every one is left out, so every log is missing.
    status, lines = run(tmp_path, committed_logs, committed=identities)
    assert status == 1 and lines[-1] == "COMPARE DONE FAIL 6"
    # With a re-measure beside them, only the re-measure is compared.
    remeasure = [dict(log, end_utc="2026-02-01T00:10:00Z", start_utc="2026-02-01T00:00:00Z") for log in complete()]
    status, lines = run(tmp_path, [*committed_logs, *remeasure], committed=identities)
    assert status == 0 and lines[-1] == "COMPARE DONE PASS"


def test_the_repository_committed_logs_are_all_known():
    identities = compare_logs.committed_identities()
    # runs/ holds 10 logs and each judged run under runs-repeat/ holds 10 more.
    assert len(identities) >= 40


def test_new_logs_from_two_commits_fail(tmp_path):
    logs = complete()
    logs[0] = dict(logs[0], commit="f" * 40)
    status, lines = run(tmp_path, logs)
    assert status == 1 and any(item.startswith("COMPARE - - commit ") and item.endswith(" OUT") for item in lines)


def test_a_re_measure_with_other_counts_or_another_runtime_fails(tmp_path):
    fewer = [
        dict(log, iterations=dict(log["iterations"], repeats=1)) if log["kind"] == "benchmark" else log
        for log in complete()
    ]
    status, lines = run(tmp_path, fewer)
    assert status == 1 and lines[-1] == "COMPARE DONE FAIL 3"
    assert all(line(lines, "benchmark", p, "setup").endswith(" OUT") for p in PRECISIONS)
    status, lines = run(tmp_path, complete({"fp32": {"runtime": "pytorch-cpu"}}))
    assert status == 1 and line(lines, "eval", "fp32", "setup").endswith(" OUT")


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


def tensorrt_fp32(ap=0.3000, end_to_end=10.0) -> list[dict]:
    """An eval and a benchmark log for TensorRT FP32."""
    return [
        make_log("eval", "fp32-trt", results={"ap": ap, "ap50": 0.5000}),
        make_log(
            "benchmark",
            "fp32-trt",
            results={"engine": latency(8.0), "end_to_end": latency(end_to_end), "avg_power_w": 8.0},
        ),
    ]


def test_tensorrt_fp32_is_reported_not_judged_while_the_reference_predates_it(tmp_path):
    (tmp_path / "with").mkdir()
    status, lines = run(tmp_path, complete())
    assert status == 0 and not any(" fp32-trt " in item for item in lines)
    status, lines = run(tmp_path / "with", [*complete(), *tensorrt_fp32(ap=0.10)])
    assert status == 0
    assert line(lines, "eval", "fp32-trt", "ap").endswith(" REPORTED")
    assert line(lines, "benchmark", "fp32-trt", "end_to_end_p50_ms").startswith(
        "COMPARE benchmark fp32-trt end_to_end_p50_ms published=missing new=10.00"
    )


def test_tensorrt_fp32_is_judged_once_the_reference_has_it(tmp_path):
    for name in ("within", "outside", "without"):
        (tmp_path / name).mkdir()
    reference = published(tmp_path, [*complete(), *tensorrt_fp32()])
    status, lines = run(tmp_path / "within", [*complete(), *tensorrt_fp32(ap=0.3009)], reference)
    assert status == 0 and line(lines, "eval", "fp32-trt", "ap").endswith(" tol=0.10 OK")
    status, lines = run(tmp_path / "outside", [*complete(), *tensorrt_fp32(ap=0.3020)], reference)
    assert status == 1 and line(lines, "eval", "fp32-trt", "ap").endswith(" tol=0.10 OUT")
    status, lines = run(tmp_path / "without", complete(), reference)
    assert status == 1 and line(lines, "eval", "fp32-trt", "log").endswith("new=missing delta=n/a tol=- OUT")


def test_deploy_logs_print_reported_lines_and_are_never_judged(tmp_path):
    (tmp_path / "with").mkdir()
    deploy = make_log("benchmark", "fp16", results={"end_to_end": latency(9.0), "avg_power_w": 8.0})
    deploy["kind"] = "benchmark_deploy"
    status, lines = run(tmp_path / "with", [*complete(), deploy])
    assert status == 0
    reported = [item for item in lines if item.startswith("COMPARE benchmark_deploy ")]
    assert [item.split()[3] for item in reported] == [metric for metric, _ in compare_logs.DEPLOY_METRICS]
    assert all(item.endswith(" tol=- REPORTED") for item in reported)
    assert (
        "COMPARE benchmark_deploy fp16 end_to_end_p50_ms published=missing new=9.00 delta=n/a tol=- REPORTED" in lines
    )
    energy = "COMPARE benchmark_deploy fp16 end_to_end_energy_per_frame_j published=missing new=0.07 "
    assert any(item.startswith(energy) for item in lines)
    status, lines = run(tmp_path, complete())
    assert status == 0 and not any("benchmark_deploy" in item for item in lines)


def deploy_of(precision="fp16", **fields):
    log = make_log("benchmark", precision, results={"end_to_end": latency(9.0), "avg_power_w": 8.0}, **fields)
    log["kind"] = "benchmark_deploy"
    return log


def test_a_dirty_deploy_log_is_shown_and_not_judged(tmp_path):
    (tmp_path / "with").mkdir()
    status, lines = run(tmp_path / "with", [*complete(), deploy_of(dirty=True)])
    assert status == 0
    assert "COMPARE benchmark_deploy fp16 log published=- new=dirty delta=n/a tol=- REPORTED" in lines
    assert not any(item.startswith("COMPARE benchmark_deploy fp16 end_to_end") for item in lines)


def test_a_deploy_log_with_another_setup_is_shown_and_not_judged(tmp_path):
    (tmp_path / "with").mkdir()
    status, lines = run(tmp_path / "with", [*complete(), deploy_of(iterations={"timed": 7})])
    assert status == 0
    setup = line(lines, "benchmark_deploy", "fp16", "setup")
    assert setup.endswith(" REPORTED") and 'new=tensorrt-gpu/{"timed":7}' in setup


def test_logs_of_any_kind_from_another_commit_fail_and_the_deploy_log_says_so(tmp_path):
    (tmp_path / "deploy").mkdir()
    status, lines = run(tmp_path / "deploy", [*complete(), deploy_of(commit="f" * 40)])
    assert status == 1 and line(lines, "-", "-", "commit").endswith(" OUT")
    assert line(lines, "benchmark_deploy", "fp16", "commit").endswith(" REPORTED")
    (tmp_path / "build").mkdir()
    status, lines = run(tmp_path / "build", [*complete(), make_log("build", "fp16", commit="f" * 40)])
    assert status == 1 and line(lines, "-", "-", "commit").endswith(" OUT")
