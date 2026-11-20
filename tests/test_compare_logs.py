# SPDX-License-Identifier: Apache-2.0
"""scripts/compare_logs.py: a re-measure against the published table, metric by metric."""

import datetime
import json
import subprocess

import pytest

import compare_logs
import generate_table
from strict_skips import checkout_skip_reason
from test_run_logs import latency, make_log

PRECISIONS = ("fp32", "fp16", "int8")
CHECKOUT_SKIP = checkout_skip_reason(generate_table.REPO_ROOT)
needs_checkout = pytest.mark.skipif(CHECKOUT_SKIP is not None, reason=CHECKOUT_SKIP or "")


def write(directory, logs):
    directory.mkdir(exist_ok=True)
    for index, log in enumerate(logs):
        (directory / f"{index:02d}.json").write_text(json.dumps(log))
    return directory


# The one INT8 engine the fixture's build, eval and benchmark logs name, so the engine-hash check passes.
ENGINE = "e" * 64


def complete(eval_results=None, benchmark_results=None, skip=()):
    """One eval and one benchmark log per precision, with per-precision results merged in."""
    eval_results, benchmark_results = eval_results or {}, benchmark_results or {}
    logs = []
    for precision in PRECISIONS:
        if ("eval", precision) not in skip:
            results = {"ap": 0.3000, "ap50": 0.5000, "engine_sha256": ENGINE, **eval_results.get(precision, {})}
            logs.append(make_log("eval", precision, results=results))
        if ("benchmark", precision) not in skip:
            results = {
                "engine": latency(4.0),
                "end_to_end": latency(10.0),
                "avg_power_w": 8.0,
                "engine_sha256": ENGINE,
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


@needs_checkout
def test_the_repository_committed_logs_are_all_known():
    identities = compare_logs.committed_identities()
    # runs/ holds 10 logs and each judged run under runs-repeat/ holds 17 more.
    assert len(identities) >= 40


def test_a_committed_attempt_log_never_stands_in_for_a_re_measure():
    attempt = sorted((compare_logs.generate_table.REPO_ROOT / "runs-attempts" / "run6-window1").glob("*.json"))[0]
    assert compare_logs._identity(json.loads(attempt.read_text())) in compare_logs.committed_identities()


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


def test_an_int8_build_on_another_calibration_path_fails(tmp_path):
    build = make_log("build", "int8", results={"engine_sha256": ENGINE, "int8_path": "IInt8EntropyCalibrator2"})
    reference = published(tmp_path, [*complete(), build])
    fallback = make_log("build", "int8", results={"engine_sha256": ENGINE, "int8_path": "qdq-fallback"})
    for name in ("same", "fallback", "none"):
        (tmp_path / name).mkdir()
    status, lines = run(tmp_path / "same", [*complete(), build], reference)
    assert status == 0 and line(lines, "build", "int8", "int8_path").endswith(" OK")
    status, lines = run(tmp_path / "fallback", [*complete(), fallback], reference)
    assert status == 1 and "new=qdq-fallback" in line(lines, "build", "int8", "int8_path")
    status, lines = run(tmp_path / "none", complete(), reference)
    assert status == 1 and "new=missing" in line(lines, "build", "int8", "int8_path")


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
    monkeypatch.setattr(compare_logs, "committed_identities", lambda: set())
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


def test_a_build_with_other_builder_flags_fails_and_an_unrecorded_reference_is_reported(tmp_path):
    build = make_log(
        "build",
        "int8",
        results={"engine_sha256": ENGINE, "int8_path": "IInt8EntropyCalibrator2", "builder_flags": ["FP16", "INT8"]},
    )
    reference = published(tmp_path, [*complete(), build])
    alone = make_log(
        "build",
        "int8",
        results={"engine_sha256": ENGINE, "int8_path": "IInt8EntropyCalibrator2", "builder_flags": ["INT8"]},
    )
    for name in ("same", "other", "unrecorded"):
        (tmp_path / name).mkdir()
    status, lines = run(tmp_path / "same", [*complete(), build], reference)
    assert status == 0 and line(lines, "build", "int8", "builder_flags").endswith(
        "published=FP16,INT8 new=FP16,INT8 delta=n/a tol=- OK"
    )
    status, lines = run(tmp_path / "other", [*complete(), alone], reference)
    assert status == 1 and line(lines, "build", "int8", "builder_flags").endswith("new=INT8 delta=n/a tol=- OUT")
    old = make_log("build", "int8", results={"engine_sha256": ENGINE, "int8_path": "IInt8EntropyCalibrator2"})
    status, lines = run(
        tmp_path / "unrecorded", [*complete(), build], published(tmp_path / "unrecorded", [*complete(), old])
    )
    assert status == 0 and line(lines, "build", "int8", "builder_flags").endswith(" REPORTED")


def _redated(log: dict) -> dict:
    """A copy of a log one day later, so it is no committed log's identity."""
    shifted = dict(log)
    for field in ("start_utc", "end_utc"):
        moment = datetime.datetime.fromisoformat(log[field].replace("Z", "+00:00")) + datetime.timedelta(days=1)
        shifted[field] = moment.strftime("%Y-%m-%dT%H:%M:%SZ")
    return shifted


@needs_checkout
def test_a_judged_runs_own_logs_redated_pass_against_the_three_run_reference(tmp_path):
    # The reference is the median of the three judged runs. A list compared by equality, such as builder_flags,
    # must stay one run's list in it, so a rerun that matches the judged runs reads OK, not OUT.
    run7 = generate_table.REPEATS_DIR / "run7"
    new = write(tmp_path / "new", [_redated(log) for log in generate_table.load_logs(run7)])
    lines = compare_logs.compare(new)
    flags = [item for item in lines if " builder_flags " in item]
    assert len(flags) == 3 and all(item.endswith(" OK") for item in flags), flags
    assert lines[-1] == "COMPARE DONE PASS", [item for item in lines if item.endswith(" OUT")]


def test_the_median_keeps_a_list_the_runs_agree_on_once_and_pools_per_repeat_samples(tmp_path):
    runs = []
    for index in range(3):
        flags = ["FP16", "INT8", "TF32"]
        build = make_log(
            "build",
            "int8",
            results={"engine_sha256": ENGINE, "int8_path": "IInt8EntropyCalibrator2", "builder_flags": flags},
        )
        bench = make_log(
            "benchmark", "int8", results={"engine_p50_ms_per_repeat": [3.0 + index, 3.1, 3.2], "engine_sha256": ENGINE}
        )
        runs.append(generate_table.select(generate_table.load_logs(write(tmp_path / f"run{index}", [build, bench]))))
    reference = generate_table.median_chosen(runs)
    assert reference[("build", "int8")]["results"]["builder_flags"] == ["FP16", "INT8", "TF32"]
    assert reference[("benchmark", "int8")]["results"]["engine_p50_ms_per_repeat"] == [
        3.0,
        3.1,
        3.2,
        4.0,
        3.1,
        3.2,
        5.0,
        3.1,
        3.2,
    ]
    (tmp_path / "rerun").mkdir()
    reference = {**published(tmp_path), ("build", "int8"): reference[("build", "int8")]}
    status, lines = run(tmp_path / "rerun", [*complete(), runs[0][("build", "int8")]], reference)
    assert status == 0 and line(lines, "build", "int8", "builder_flags").endswith(
        "published=FP16,INT8,TF32 new=FP16,INT8,TF32 delta=n/a tol=- OK"
    )


def test_judged_runs_that_disagree_on_builder_flags_stop_the_median(tmp_path):
    runs = []
    for index, flags in enumerate((["FP16", "INT8", "TF32"], ["FP16", "INT8", "TF32"], ["INT8"])):
        build = make_log(
            "build",
            "int8",
            results={"engine_sha256": ENGINE, "int8_path": "IInt8EntropyCalibrator2", "builder_flags": flags},
        )
        runs.append(generate_table.select(generate_table.load_logs(write(tmp_path / f"run{index}", [build]))))
    with pytest.raises(generate_table.TableError, match="the judged runs disagree on builder_flags"):
        generate_table.median_chosen(runs)


def test_the_reproduce_block_makes_every_judged_log_before_it_compares():
    # A reader who runs the block top to bottom must have every log compare_logs judges before it runs.
    text = generate_table.README.read_text()
    blocks = [block.split("```", 1)[0] for block in text.split("```sh\n")[1:]]
    block = next(block for block in blocks if "python scripts/compare_logs.py runs" in block)
    commands = [command for command in block.splitlines() if command.startswith("python ")]
    assert commands[-1] == "python scripts/compare_logs.py runs"
    for step in ("engine_build", "evaluate", "benchmark"):
        assert f"python -m peddet_orin.{step} --precision fp32-trt" in commands, step
    measuring = {command for other in blocks if other is not block for command in other.splitlines()}
    assert not {command for command in measuring if "fp32-trt" in command or "--postprocess deploy" in command}


def test_a_rerun_with_other_builder_flags_fails_against_the_three_run_median(tmp_path):
    # The planted mismatch: the reference is a three-run median, and the rerun's INT8 build drops the FP16 flag.
    runs = []
    for index in range(3):
        build = make_log(
            "build", "int8", results={"int8_path": "IInt8EntropyCalibrator2", "builder_flags": ["FP16", "INT8", "TF32"]}
        )
        runs.append(generate_table.select(generate_table.load_logs(write(tmp_path / f"run{index}", [build]))))
    reference = {**published(tmp_path), ("build", "int8"): generate_table.median_chosen(runs)[("build", "int8")]}
    planted = make_log(
        "build",
        "int8",
        results={"engine_sha256": ENGINE, "int8_path": "IInt8EntropyCalibrator2", "builder_flags": ["INT8"]},
    )
    (tmp_path / "rerun").mkdir()
    status, lines = run(tmp_path / "rerun", [*complete(), planted], reference)
    assert status == 1 and line(lines, "build", "int8", "builder_flags").endswith(
        "published=FP16,INT8,TF32 new=INT8 delta=n/a tol=- OUT"
    )


def _git(root, *arguments):
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
    )


def test_a_re_measure_written_into_runs_is_compared_and_the_committed_logs_are_left_out(tmp_path, monkeypatch):
    # The README's reproduce block writes the re-measure into runs/, beside the committed logs, and compares runs/:
    # only the logs in HEAD's tree count as committed, so the re-measure is compared and the committed logs are not.
    root = tmp_path / "clone"
    root.mkdir()
    runs = write(root / "runs", complete())
    _git(root, "init", "-q")
    _git(root, "add", "runs")
    _git(root, "commit", "-q", "-m", "committed logs")
    remeasure = [dict(log, start_utc="2026-02-01T00:00:00Z", end_utc="2026-02-01T00:10:00Z") for log in complete()]
    for index, log in enumerate(remeasure):
        (runs / f"new-{index:02d}.json").write_text(json.dumps(log))
    monkeypatch.setattr(generate_table, "REPO_ROOT", root)
    monkeypatch.setattr(generate_table, "RUNS_DIR", runs)
    monkeypatch.setattr(generate_table, "REPEATS_DIR", root / "runs-repeat")
    identities = compare_logs.committed_identities()
    assert identities == {compare_logs._identity(log) for log in complete()}
    lines = compare_logs.compare(runs, published(tmp_path), identities)
    assert lines[-1] == "COMPARE DONE PASS", [item for item in lines if item.endswith(" OUT")]
    # Without the re-measure, the committed logs alone are all left out: every judged log is missing.
    for index in range(len(remeasure)):
        (runs / f"new-{index:02d}.json").unlink()
    assert compare_logs.compare(runs, published(tmp_path), compare_logs.committed_identities())[-1] == (
        "COMPARE DONE FAIL 6"
    )


def test_a_committed_log_edited_in_the_work_tree_is_still_left_out(tmp_path, monkeypatch):
    root = tmp_path / "clone"
    root.mkdir()
    runs = write(root / "runs", complete())
    _git(root, "init", "-q")
    _git(root, "add", "runs")
    _git(root, "commit", "-q", "-m", "committed logs")
    edited = json.loads((runs / "00.json").read_text())
    edited["end_utc"] = "2026-03-01T00:10:00Z"
    (runs / "00.json").write_text(json.dumps(edited))
    monkeypatch.setattr(generate_table, "REPO_ROOT", root)
    assert compare_logs._identity(edited) in compare_logs.committed_identities()


def test_outside_a_git_checkout_the_committed_logs_stop_with_one_line(tmp_path):
    write(tmp_path / "runs", complete())
    with pytest.raises(generate_table.TableError) as raised:
        compare_logs.committed_identities(tmp_path)
    assert str(raised.value) == compare_logs.NO_GIT


def _int8_build(**results):
    base = {"int8_path": "IInt8EntropyCalibrator2", "builder_flags": ["FP16", "INT8", "TF32"], "engine_sha256": ENGINE}
    return make_log("build", "int8", results={**base, **results})


def test_a_dirty_newest_build_log_fails_its_int8_path_and_builder_flags_lines(tmp_path):
    reference = published(tmp_path, [*complete(), _int8_build()])
    dirty = dict(_int8_build(), dirty=True)
    (tmp_path / "dirty").mkdir()
    status, lines = run(tmp_path / "dirty", [*complete(), dirty], reference)
    assert status == 1
    assert "COMPARE build int8 int8_path published=- new=dirty delta=n/a tol=- OUT" in lines
    assert "COMPARE build int8 builder_flags published=- new=dirty delta=n/a tol=- OUT" in lines


def test_an_eval_or_benchmark_from_another_engine_than_its_build_fails(tmp_path):
    reference = published(tmp_path, [*complete(), _int8_build()])
    build = _int8_build(engine_sha256="a" * 64)
    same = {"engine_sha256": "a" * 64}
    other = {"engine_sha256": "b" * 64}
    (tmp_path / "same").mkdir()
    status, lines = run(tmp_path / "same", [*complete({"int8": same}, {"int8": same}), build], reference)
    assert status == 0 and line(lines, "eval", "int8", "engine_sha256").endswith(" OK")
    (tmp_path / "other").mkdir()
    status, lines = run(tmp_path / "other", [*complete({"int8": same}, {"int8": other}), build], reference)
    assert status == 1 and line(lines, "benchmark", "int8", "engine_sha256").endswith(" OUT")
    assert line(lines, "eval", "int8", "engine_sha256").endswith(" OK")


def test_another_environment_fails(tmp_path):
    reference = published(tmp_path)
    moved = [
        dict(log, environment={**log["environment"], "tensorrt": "10.99"}) if log["kind"] == "benchmark" else log
        for log in complete()
    ]
    (tmp_path / "moved").mkdir()
    status, lines = run(tmp_path / "moved", moved, reference)
    assert status == 1
    assert "COMPARE environment - tensorrt published=value new=10.99 delta=n/a tol=- OUT" in lines


def test_a_missing_engine_hash_fails(tmp_path):
    reference = published(tmp_path, [*complete(), _int8_build()])
    unhashed = [
        {**log, "results": {key: value for key, value in log["results"].items() if key != "engine_sha256"}}
        if log["kind"] == "benchmark" and log["precision"] == "int8"
        else log
        for log in complete()
    ]
    (tmp_path / "unhashed").mkdir()
    status, lines = run(tmp_path / "unhashed", [*unhashed, _int8_build()], reference)
    assert status == 1
    assert line(lines, "benchmark", "int8", "engine_sha256").endswith(" OUT")
