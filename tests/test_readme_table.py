# SPDX-License-Identifier: Apache-2.0
"""The README results block must be exactly what the logs generate."""

import json
import re

import pytest

import compare_logs
import generate_table
from test_run_logs import make_log

NUMERIC_CELL = re.compile(r"^\d+\.\d+$")


def test_readme_table_equals_generated_table():
    assert generate_table.readme_matches(generate_table.README.read_text())


def test_the_readme_carries_every_generated_block():
    text = generate_table.README.read_text()
    for markers, _ in generate_table.BLOCKS:
        assert text.count(markers[0]) == 1 and text.count(markers[1]) == 1, markers


def test_editing_one_number_is_detected(tmp_path):
    # The committed README may carry only n/a cells, so the block is generated from synthetic logs.
    for index, log in enumerate([make_log("eval", "fp16"), make_log("benchmark", "fp16")]):
        (tmp_path / f"{index}.json").write_text(json.dumps(log))
    block = generate_table.build_block(tmp_path)
    text = f"intro\n{generate_table.BEGIN_MARKER}\n{block}\n{generate_table.END_MARKER}\nend\n"
    assert generate_table.readme_matches(text, tmp_path)
    lines = block.splitlines()
    row = next(index for index, line in enumerate(lines) if line.startswith("| GPU, TensorRT FP16"))
    cells = lines[row].split(" | ")
    column = next(index for index, cell in enumerate(cells) if NUMERIC_CELL.match(cell))
    original = cells[column]
    cells[column] = original[:-1] + ("1" if original[-1] != "1" else "2")
    edited = "\n".join([*lines[:row], " | ".join(cells), *lines[row + 1 :]])
    assert NUMERIC_CELL.match(cells[column]) and cells[column] != original
    assert not generate_table.readme_matches(text.replace(block, edited), tmp_path)


QUIET_PASSED = "".join(
    f"QUIET {gate} try=1 held_active=0 allowed_timers=3 scopes_over=0 busy_services=0 load1=0.80 bound=1.0 "
    "swap_used_kb=0 pswpout_delta=0 mem_available_kb=6000000 mem_min_kb=5859375 fail=none 2026-01-01T00:00:00Z\n"
    f"QUIET {gate} PASS\n"
    for gate in ("start", "bench_fp32", "bench_fp16", "bench_int8")
)


def _copy_runs(source, destination, scale_power=1.0):
    destination.mkdir(parents=True)
    for path in source.glob("*.json"):
        log = json.loads(path.read_text())
        if log["kind"] == "benchmark" and isinstance(log["results"]["avg_power_w"], float):
            log["results"]["avg_power_w"] *= scale_power
        (destination / path.name).write_text(json.dumps(log))


def test_the_repeat_block_judges_every_further_run_against_the_prd_tolerances(tmp_path):
    for name in ("run1", "run2", "run3"):
        _copy_runs(generate_table.RUNS_DIR, tmp_path / name)
    for name in ("run1", "run2", "run3"):
        (tmp_path / name / "quiet.txt").write_text(QUIET_PASSED)
    block = generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    assert "R-11 checks that the 3 quieted runs" in block and "| OUT |" not in block
    assert "3 runs on one board, which is not a population" in block
    assert "- earlier run (`runs/`): not quieted, load before every benchmark at or above" in block
    assert "- run1: quieted, every QUIET gate passed, recorded before the memory, swap and temperature reads;" in block
    assert "not quieted" not in block.split("- earlier run")[0]
    assert block.endswith("R-11 is met for every judged metric among the quieted runs.")


def test_deleting_one_pass_stops_the_generator(tmp_path):
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run1")
    quiet = tmp_path / "run1" / "quiet.txt"
    quiet.write_text(QUIET_PASSED)
    assert "- run1: quieted," in generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    quiet.write_text(QUIET_PASSED.replace("QUIET bench_fp16 PASS\n", ""))
    for build in (generate_table.build_repeat_block, generate_table.build_block, generate_table.build_int8_block):
        with pytest.raises(generate_table.TableError, match="run run1 is not quieted .not quieted, a QUIET gate did"):
            build(generate_table.RUNS_DIR, tmp_path)


def test_a_judged_run_without_a_quiet_file_stops_the_generator(tmp_path):
    for name in ("run1", "run2"):
        _copy_runs(generate_table.RUNS_DIR, tmp_path / name)
    (tmp_path / "run1" / "quiet.txt").write_text(QUIET_PASSED)
    with pytest.raises(generate_table.TableError, match="judged run run2 is not quieted"):
        generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)


# The earlier gate wrote no count of allowed timers and no swap or memory fields.
QUIET_EARLIER = re.sub(
    r" allowed_timers=3| swap_used_kb=0 pswpout_delta=0 mem_available_kb=\d+ mem_min_kb=\d+ fail=none", "", QUIET_PASSED
)


def test_both_quiet_line_forms_are_read_and_the_earlier_one_is_noted(tmp_path):
    for name in ("run1", "run2", "run3"):
        _copy_runs(generate_table.RUNS_DIR, tmp_path / name)
    (tmp_path / "run1" / "quiet.txt").write_text(QUIET_EARLIER)
    (tmp_path / "run2" / "quiet.txt").write_text(QUIET_PASSED.replace(" swap_used_kb=0 pswpout_delta=0", ""))
    (tmp_path / "run3" / "quiet.txt").write_text(QUIET_PASSED)
    _sysmon(tmp_path / "run3")
    block = generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    assert (
        "- run1: quieted, every QUIET gate passed, recorded before the memory, swap and temperature reads "
        "(its gate lines predate the count of allowed timers);" in block
    )
    assert "- run2: quieted, every QUIET gate passed, recorded before the memory, swap and temperature reads;" in block


def test_the_gate_terms_and_each_runs_values_come_from_the_passing_lines(tmp_path):
    for name, memory in (("run3", "6100000"), ("run4", "6200000")):
        _copy_runs(generate_table.RUNS_DIR, tmp_path / name)
        (tmp_path / name / "quiet.txt").write_text(
            QUIET_PASSED.replace("mem_available_kb=6000000", f"mem_available_kb={memory}")
        )
        _sysmon(tmp_path / name)
    block = generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    assert (
        "0 kB of swap used, 0 pages swapped out during the settle; 3 scheduled jobs allowed by name; load1 below "
        "1.0; available memory at least 5859375 kB." in block
    )
    assert "| run4 | bench_int8 | 1 | 0.80 | 6200000 |" in block and "| run3 | start | 1 | 0.80 | 6100000 |" in block
    assert "long-running sessions" not in block and "5 s of CPU" not in block


@pytest.mark.parametrize(
    "change",
    [
        ("swap_used_kb=0", "swap_used_kb=256"),
        ("mem_min_kb=5859375", "mem_min_kb=1"),
        ("busy_services=0", "busy_services=1"),
    ],
)
def test_a_passing_line_that_contradicts_a_term_stops_the_generator(tmp_path, change):
    for name in ("run3", "run4"):
        _copy_runs(generate_table.RUNS_DIR, tmp_path / name)
        _sysmon(tmp_path / name)
    (tmp_path / "run3" / "quiet.txt").write_text(QUIET_PASSED)
    (tmp_path / "run4" / "quiet.txt").write_text(QUIET_PASSED.replace(*change, 1))
    with pytest.raises(generate_table.TableError, match="passing QUIET line"):
        generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)


def test_the_headline_is_the_median_of_the_repeat_runs(tmp_path):
    for name, scale in (("run1", 0.8), ("run2", 1.0), ("run3", 1.3)):
        _copy_runs(generate_table.RUNS_DIR, tmp_path / name, scale_power=scale)
        (tmp_path / name / "quiet.txt").write_text(QUIET_PASSED)
    chosen, runs = generate_table.headline(generate_table.RUNS_DIR, tmp_path)
    middle = generate_table.select(generate_table.load_logs(tmp_path / "run2"))
    for precision in generate_table.PRECISIONS:
        key = ("benchmark", precision)
        assert chosen[key]["results"]["avg_power_w"] == middle[key]["results"]["avg_power_w"]
    block = generate_table.build_block(generate_table.RUNS_DIR, tmp_path)
    assert "Each cell is the median over the n = 3 quieted runs" in block and "0.2.0 plans 5 or more" in block
    assert "- run1: FP16 benchmark:" in block and "FP16 min-max over the 3 runs:" in block


def _benchmark_dir(directory, power, fps):
    directory.mkdir()
    end_to_end = {"p50_ms": 1000.0 / fps, "p90_ms": 1000.0 / fps, "p99_ms": 1000.0 / fps, "fps": fps}
    log = make_log("benchmark", "fp16", results={"avg_power_w": power, "end_to_end": end_to_end})
    (directory / "benchmark.json").write_text(json.dumps(log))
    return generate_table.select(generate_table.load_logs(directory))


def test_energy_per_frame_is_power_over_the_end_to_end_rate(tmp_path):
    chosen = _benchmark_dir(tmp_path / "one", 7.25, 20.0)
    assert chosen[("benchmark", "fp16")]["results"][generate_table.ENERGY] == pytest.approx(0.3625)
    row = generate_table._row(chosen, "fp16").split(" | ")
    header = generate_table.HEADER.split(" | ")
    assert row[header.index("End-to-end energy per frame (J)")] == "0.362"
    # Engine FPS stays beside the engine latency, apart from power, which sits with the end-to-end columns.
    assert header.index("Engine FPS") == header.index("Engine p50/p90/p99 (ms)") + 1
    assert header.index("Avg power VDD_IN (W)") == header.index("End-to-end FPS") + 1
    unread = _benchmark_dir(tmp_path / "two", "UNREADABLE", 20.0)
    assert unread[("benchmark", "fp16")]["results"][generate_table.ENERGY] == "UNREADABLE"


def test_the_energy_median_is_the_median_of_each_runs_figure(tmp_path):
    # Per run: 6/10 = 0.6, 7/30 = 0.233 and 9/14 = 0.643, median 0.6; the medians' ratio, 7/14 = 0.5, is not it.
    runs = [
        _benchmark_dir(tmp_path / f"run{n}", power, fps)
        for n, (power, fps) in enumerate([(6.0, 10.0), (7.0, 30.0), (9.0, 14.0)])
    ]
    medians = generate_table.median_chosen(runs)
    assert medians[("benchmark", "fp16")]["results"][generate_table.ENERGY] == pytest.approx(0.6)


def test_a_runs_directory_without_repeats_gives_the_single_run_table(tmp_path):
    for index, log in enumerate([make_log("eval", "fp16"), make_log("benchmark", "fp16")]):
        (tmp_path / f"{index}.json").write_text(json.dumps(log))
    assert "Each cell is the median" not in generate_table.build_block(tmp_path)


def _sysmon(run, swap_at_start=0, pswpout=(100, 100), sha=None):
    folder = run / "sysmon"
    folder.mkdir()
    sha = sha or generate_table.SYSMON_SHA256
    for precision in generate_table.PRECISIONS:
        (folder / f"r9-bench_{precision}.txt").write_text(
            f"SYSMON sha256={sha} interval_s=1\n"
            f"SYS 2026-01-01T00:00:00Z mem_available_kb=4096000 swap_used_kb={swap_at_start} pswpin=5 "
            f"pswpout={pswpout[0]} cpu-thermal_mc=45000 gpu-thermal_mc=44000\n"
            f"SYS 2026-01-01T00:00:01Z mem_available_kb=3072000 swap_used_kb={swap_at_start} pswpin=5 "
            f"pswpout={pswpout[1]} cpu-thermal_mc=51500 gpu-thermal_mc=50000\n"
        )


def test_recorded_runs_are_judged_and_unrecorded_ones_shown_apart(tmp_path):
    for name in ("run1", "run2", "run3", "run4", "run5"):
        _copy_runs(generate_table.RUNS_DIR, tmp_path / name)
        (tmp_path / name / "quiet.txt").write_text(QUIET_PASSED)
    for name in ("run3", "run4", "run5"):
        _sysmon(tmp_path / name)
    block = generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    assert "R-11 checks that the 3 quieted runs run3, run4, run5" in block
    assert "runs run1, run2 were quieted but recorded before the memory, swap and temperature reads" in block
    assert "- run3: quieted, every QUIET gate passed;" in block
    header = next(line for line in block.splitlines() if line.startswith("| Metric |"))
    assert header.endswith("| R-11 | run1 | run2 | earlier run |")
    assert "| run3 | FP16 | 0 | 0 | 3000 | cpu-thermal 51.5, gpu-thermal 50.0 |" in block
    chosen, runs = generate_table.headline(generate_table.RUNS_DIR, tmp_path)
    assert [name for name, _ in runs] == ["run3", "run4", "run5"]


def test_a_failed_or_attempt_directory_is_never_read(tmp_path):
    for name in ("run1", "run2", "run3-failed", "run3-attempt1", "run10"):
        (tmp_path / name).mkdir()
    (tmp_path / "run3-attempt1-quiet.txt").write_text("QUIET start FAIL\n")
    assert [path.name for path in generate_table.repeat_dirs(tmp_path)] == ["run1", "run10", "run2"]


@pytest.mark.parametrize("sysmon", [{"swap_at_start": 2048}, {"pswpout": (100, 103)}])
def test_swap_in_use_or_a_swap_out_stops_the_generator(tmp_path, sysmon):
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run3")
    (tmp_path / "run3" / "quiet.txt").write_text(QUIET_PASSED)
    _sysmon(tmp_path / "run3", **sysmon)
    with pytest.raises(generate_table.TableError, match="judged run run3 is not quieted .not quieted, swap was in use"):
        generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)


def test_each_run_shows_its_first_sample(tmp_path):
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run3")
    (tmp_path / "run3" / "quiet.txt").write_text(QUIET_PASSED)
    _sysmon(tmp_path / "run3")
    block = generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    assert "| run3 | FP16 | 0 | 0 | 3000 |" in block
    assert (
        "- run3 first sample, 2026-01-01T00:00:00Z: available memory 4000 MiB, swap used 0 kB, temperatures (C) "
        "cpu-thermal 45.0, gpu-thermal 44.0." in block
    )


def test_a_sysmon_file_from_another_sampler_is_refused(tmp_path):
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run3")
    (tmp_path / "run3" / "quiet.txt").write_text(QUIET_PASSED)
    _sysmon(tmp_path / "run3", sha="0" * 64)
    with pytest.raises(generate_table.TableError, match="does not name the recorded sampler"):
        generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)


def test_a_quiet_line_naming_another_load_bound_is_refused(tmp_path):
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run1")
    (tmp_path / "run1" / "quiet.txt").write_text(QUIET_PASSED.replace("bound=1.0", "bound=3.0", 1))
    with pytest.raises(generate_table.TableError, match="load bound other than 1.0"):
        generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)


def test_a_further_run_outside_a_tolerance_is_reported_out_and_not_met(tmp_path):
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run1")
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run2", scale_power=1.5)
    for name in ("run1", "run2"):
        (tmp_path / name / "quiet.txt").write_text(QUIET_PASSED)
    block = generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    row = next(line for line in block.splitlines() if line.startswith("| avg_power_w | FP16 |"))
    assert "| 10% | OUT |" in row
    assert "R-11 not met for avg_power_w FP16" in block and "avg_power_w FP32" in block
    assert compare_logs.TOLERANCES["avg_power_w"] == {"relative": 0.10}


def test_editing_a_number_in_the_int8_comparison_is_detected():
    text = generate_table.README.read_text()
    block = generate_table.readme_block(text, generate_table.INT8_MARKERS)
    number = re.search(r"\d+\.\d\d", block).group()
    edited = block.replace(number, number[:-1] + ("1" if number[-1] != "1" else "2"), 1)
    assert generate_table.readme_matches(text)
    assert not generate_table.readme_matches(text.replace(block, edited))


@pytest.mark.parametrize("missing", ["benchmark_int8", "eval_fp16", "build_int8", "gpu_check_fp32"])
def test_a_judged_run_missing_a_log_stops_the_generator(tmp_path, missing):
    for name in ("run1", "run2"):
        _copy_runs(generate_table.RUNS_DIR, tmp_path / name)
        (tmp_path / name / "quiet.txt").write_text(QUIET_PASSED)
    removed = sorted((tmp_path / "run2").glob(f"{missing}_*.json"))
    assert len(removed) == 1
    removed[0].unlink()
    kind, precision = missing.rsplit("_", 1)
    for build in (generate_table.build_repeat_block, generate_table.build_block):
        with pytest.raises(generate_table.TableError, match=f"judged run run2 has no clean log for {kind} {precision}"):
            build(generate_table.RUNS_DIR, tmp_path)


def _set_version(run, version):
    for path in run.glob("*.json"):
        log = json.loads(path.read_text())
        log["environment"]["peddet_orin"] = version
        path.write_text(json.dumps(log))


def test_the_commit_and_package_version_come_from_the_judged_logs(tmp_path):
    for name in ("run1", "run2"):
        _copy_runs(generate_table.RUNS_DIR, tmp_path / name)
        (tmp_path / name / "quiet.txt").write_text(QUIET_PASSED)
        _set_version(tmp_path / name, "9.8.7")
    commit = generate_table.select(generate_table.load_logs(tmp_path / "run1"))[("eval", "fp16")]["commit"]
    block = generate_table.build_block(generate_table.RUNS_DIR, tmp_path)
    assert (
        f"The judged runs were made at commit `{commit[:7]}` with package version 9.8.7, as their logs record." in block
    )
    _set_version(tmp_path / "run2", "9.8.8")
    with pytest.raises(generate_table.TableError, match="1 commits and 2 package versions"):
        generate_table.build_block(generate_table.RUNS_DIR, tmp_path)


def test_the_readme_names_no_commit_or_version_of_the_judged_runs_by_hand():
    text = generate_table.README.read_text()
    outside = text.replace(generate_table.readme_block(text), "")
    assert "made at commit `" not in outside and "with package version" not in outside


def test_equal_detection_counts_read_as_many_as(tmp_path):
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run1")
    (tmp_path / "run1" / "quiet.txt").write_text(QUIET_PASSED)
    logs = {path: json.loads(path.read_text()) for path in (tmp_path / "run1").glob("eval_*.json")}
    count = next(log for log in logs.values() if log["precision"] == "fp16")["results"]["detections"]
    for path, log in logs.items():
        log["results"]["detections"] = count
        path.write_text(json.dumps(log))
    block = generate_table.build_int8_block(generate_table.RUNS_DIR, tmp_path)
    assert f"INT8 yields as many person detections after NMS on val2017 as FP16, {count} against {count}," in block
    assert "Whether" not in block


def _edit_logs(run, kind, precision, edit):
    for path in run.glob(f"{kind}_{precision}_*.json"):
        log = json.loads(path.read_text())
        edit(log["results"])
        path.write_text(json.dumps(log))


def test_the_overlap_is_judged_per_run_and_said_so(tmp_path):
    for name in ("run1", "run2"):
        _copy_runs(generate_table.RUNS_DIR, tmp_path / name)
        (tmp_path / name / "quiet.txt").write_text(QUIET_PASSED)
        _edit_logs(
            tmp_path / name,
            "benchmark",
            "fp16",
            lambda results: results.update(end_to_end_p50_ms_per_repeat=[60.0, 61.0, 62.0]),
        )
    _edit_logs(
        tmp_path / "run1",
        "benchmark",
        "int8",
        lambda results: results.update(end_to_end_p50_ms_per_repeat=[61.5, 63.0, 64.0]),
    )
    _edit_logs(
        tmp_path / "run2",
        "benchmark",
        "int8",
        lambda results: results.update(end_to_end_p50_ms_per_repeat=[63.0, 64.0, 65.0]),
    )
    block = generate_table.build_int8_block(generate_table.RUNS_DIR, tmp_path)
    assert "FP16's and INT8's per-repeat end-to-end ranges in the notes overlap in 1 of the 2 runs." in block
    _edit_logs(
        tmp_path / "run2",
        "benchmark",
        "int8",
        lambda results: results.update(end_to_end_p50_ms_per_repeat=[61.0, 64.0, 65.0]),
    )
    block = generate_table.build_int8_block(generate_table.RUNS_DIR, tmp_path)
    assert "ranges in the notes overlap in each of the 2 runs." in block
    _edit_logs(
        tmp_path / "run1",
        "benchmark",
        "int8",
        lambda results: results.update(end_to_end_p50_ms_per_repeat=[63.0, 64.0, 65.0]),
    )
    _edit_logs(
        tmp_path / "run2",
        "benchmark",
        "int8",
        lambda results: results.update(end_to_end_p50_ms_per_repeat=[63.0, 64.0, 65.0]),
    )
    block = generate_table.build_int8_block(generate_table.RUNS_DIR, tmp_path)
    assert "ranges in the notes do not overlap in any of the 2 runs." in block


def test_the_postprocess_hypothesis_follows_the_detection_counts(tmp_path):
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run1")
    (tmp_path / "run1" / "quiet.txt").write_text(QUIET_PASSED)
    _edit_logs(tmp_path / "run1", "eval", "int8", lambda results: results.update(detections=1))
    block = generate_table.build_int8_block(generate_table.RUNS_DIR, tmp_path)
    assert "INT8 yields fewer person detections" in block
    assert "Whether fewer detections give the postprocess less work during the benchmark is a hypothesis" in block


def test_the_thermal_zone_names_come_from_the_records(tmp_path):
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run3")
    (tmp_path / "run3" / "quiet.txt").write_text(QUIET_PASSED)
    _sysmon(tmp_path / "run3")
    block = generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    assert "The thermal zones are the ones whose readings the records hold: cpu-thermal, gpu-thermal." in block
    assert "cv0" not in block


def test_judged_runs_whose_samplers_name_two_intervals_stop_the_generator(tmp_path):
    for name in ("run3", "run4"):
        _copy_runs(generate_table.RUNS_DIR, tmp_path / name)
        (tmp_path / name / "quiet.txt").write_text(QUIET_PASSED)
        _sysmon(tmp_path / name)
    for path in (tmp_path / "run4" / "sysmon").glob("*.txt"):
        path.write_text(path.read_text().replace("interval_s=1", "interval_s=2", 1))
    assert generate_table.sampler_intervals([tmp_path / "run3", tmp_path / "run4"]) == {"1", "2"}
    with pytest.raises(generate_table.TableError, match="samplers name 2 intervals: 1, 2"):
        generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)


def _job_sysmon(run, job, memory_kb, pswpout):
    folder = run / "sysmon"
    folder.mkdir(exist_ok=True)
    sha = generate_table.SYSMON_SHA256
    lines = [f"SYSMON sha256={sha} interval_s=1"]
    for second, (memory, count) in enumerate(zip(memory_kb, pswpout, strict=True)):
        lines.append(
            f"SYS 2026-01-01T00:00:0{second}Z mem_available_kb={memory} swap_used_kb=0 pswpin=5 "
            f"pswpout={count} cpu-thermal_mc=45000"
        )
    (folder / f"{run.name.replace('run', 'r')}-{job}.txt").write_text("\n".join(lines) + "\n")


def _ram_runs(tmp_path, build_pswpout=(7, 7, 7)):
    for name in ("run3", "run4", "run5"):
        _copy_runs(generate_table.RUNS_DIR, tmp_path / name)
        (tmp_path / name / "quiet.txt").write_text(QUIET_PASSED)
        _sysmon(tmp_path / name)
    _job_sysmon(tmp_path / "run4", "build_int8", (5242880, 4194304, 5000000), build_pswpout)
    _job_sysmon(tmp_path / "run5", "eval_fp32", (4300000, 4400000, 4500000), (9, 9, 9))


def _job_rows(block):
    return {
        tuple(line.split(" | ")[:2]): line.split(" | ")[2:]
        for line in block.splitlines()
        if line.startswith("| run") and line.count(" | ") == 3
    }


def test_each_jobs_lowest_memory_is_the_lowest_of_the_samples_during_it(tmp_path):
    _ram_runs(tmp_path)
    rows = _job_rows(generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path))
    assert rows[("| run4", "build_int8")][0] == "4096"
    assert rows[("| run5", "eval_fp32")][0] == "4199"
    assert rows[("| run3", "bench_fp16")][0] == "3000"
    assert len(rows) == 3 + 3 + 3 + 2


def test_each_jobs_swap_outs_are_its_pswpout_delta(tmp_path):
    _ram_runs(tmp_path, build_pswpout=(7, 9, 12))
    rows = _job_rows(generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path))
    assert rows[("| run4", "build_int8")][1] == "5 |"
    assert rows[("| run5", "eval_fp32")][1] == "0 |"


def test_the_ram_sentence_gives_the_lowest_job_or_each_swap_out(tmp_path):
    _ram_runs(tmp_path)
    block = generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    assert block.endswith(
        "RAM was not a constraint in runs 3-5: the lowest available memory during any job was "
        "3000 MiB, and 0 pages were swapped out."
    )
    _job_sysmon(tmp_path / "run4", "build_int8", (5242880, 4194304, 5000000), (7, 9, 12))
    block = generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    assert block.endswith("run4 build_int8 swapped out 5 pages.") and "RAM was not a constraint" not in block


def test_a_missing_sysmon_file_is_named_as_missing_and_not_as_swap_in_use(tmp_path):
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run3")
    (tmp_path / "run3" / "quiet.txt").write_text(QUIET_PASSED)
    _sysmon(tmp_path / "run3")
    (tmp_path / "run3" / "sysmon" / "r9-bench_int8.txt").unlink()
    with pytest.raises(generate_table.TableError) as raised:
        generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    assert "no single sysmon file with samples for the INT8 benchmark job" in str(raised.value)
    assert "swap was in use" not in str(raised.value)


def _repeat_runs(tmp_path, names=("run3", "run4")):
    for name in names:
        _copy_runs(generate_table.RUNS_DIR, tmp_path / name)
        (tmp_path / name / "quiet.txt").write_text(QUIET_PASSED)


def test_an_unread_power_mode_is_said_with_the_clock_maxima_as_evidence(tmp_path):
    _repeat_runs(tmp_path)
    for name in ("run3", "run4"):
        for precision in generate_table.PRECISIONS:
            _edit_logs(tmp_path / name, "benchmark", precision, lambda results: results.update(power_mode="UNREADABLE"))
    block = generate_table.build_block(generate_table.RUNS_DIR, tmp_path)
    earlier = generate_table.select(generate_table.load_logs(generate_table.RUNS_DIR))[("benchmark", "fp16")]
    gpu = earlier["results"]["gpu_clock"]["max_hz"] // 1_000_000
    assert "The judged runs' power mode was not read: 6 of their 6 benchmark logs record `UNREADABLE`" in block
    assert f"clock maxima of every judged benchmark: GPU {gpu} MHz, CPU policy0 " in block
    assert "MAXN_SUPER" not in block.split("whose power mode reads")[0]


def test_a_judged_benchmark_whose_clock_maxima_differ_from_the_earlier_runs_stops_the_generator(tmp_path):
    _repeat_runs(tmp_path)
    _edit_logs(tmp_path / "run4", "benchmark", "int8", lambda results: results.update(power_mode="UNREADABLE"))
    _edit_logs(tmp_path / "run4", "benchmark", "fp32", lambda results: results["gpu_clock"].update(max_hz=624000000))
    with pytest.raises(generate_table.TableError, match="maximum clock differs from the earlier run's"):
        generate_table.build_block(generate_table.RUNS_DIR, tmp_path)


def test_a_read_power_mode_is_quoted_from_the_judged_logs(tmp_path):
    _repeat_runs(tmp_path)
    block = generate_table.build_block(generate_table.RUNS_DIR, tmp_path)
    assert "The judged runs' power mode, as `nvpmodel -q` reported it:" in block


def test_the_int8_ap_difference_follows_from_the_shown_figures(tmp_path):
    _repeat_runs(tmp_path, ("run3",))
    _edit_logs(tmp_path / "run3", "eval", "fp16", lambda results: results.update(ap=0.280059))
    _edit_logs(tmp_path / "run3", "eval", "int8", lambda results: results.update(ap=0.267430))
    block = generate_table.build_block(generate_table.RUNS_DIR, tmp_path)
    assert "| 28.01 |" in block and "| 26.74 |" in block
    assert "INT8 person AP differs from FP16 by -1.27 points." in block


def test_a_zero_deviation_has_no_sign_and_no_line_ends_in_a_space(tmp_path):
    _repeat_runs(tmp_path)
    block = generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    assert "(-0.00)" not in block and "(+0.00)" not in block and "(0.00)" in block
    assert "(-0.0%)" not in block and "(+0.0%)" not in block
    assert not any(line.endswith(" ") for line in block.splitlines())
    assert generate_table._signed(-0.001, 2) == "0.00" and generate_table._signed(-0.012, 2) == "-0.01"


def test_the_int8_ap_sentence_names_each_runs_own_engine(tmp_path):
    _repeat_runs(tmp_path)
    _edit_logs(tmp_path / "run4", "eval", "int8", lambda results: results.update(ap=0.2690))
    block = generate_table.build_block(generate_table.RUNS_DIR, tmp_path)
    assert (
        "INT8 person AP is 26.83-26.90 across the 2 judged runs and 26.83 in the earlier run; each run builds its "
        "own INT8 engine, so each score comes from a separate build." in block
    )


def test_the_changelog_names_the_commits_the_logs_record():
    entry = generate_table.README.with_name("CHANGELOG.md").read_text().split("## [0.1.1]", 1)[1].split("\n## [", 1)[0]
    _, runs = generate_table.headline()
    commit, _ = generate_table.provenance(runs)
    logs = generate_table.select(generate_table.load_logs(generate_table.RUNS_DIR))
    earlier = {log["commit"][:7] for log in logs.values()}
    assert re.findall(r"judged quieted runs at commit `([0-9a-f]{7})`", entry) == [commit[:7]]
    assert re.findall(r"one run at `([0-9a-f]{7})`", entry) == sorted(earlier)


def test_the_judged_runs_paragraph_ends_before_the_gate_paragraph():
    block = generate_table.build_repeat_block()
    first, second, *_ = block.split("\n\n")
    assert first.endswith("and not judged.") and second.startswith("Each judged run's QUIET gate")
