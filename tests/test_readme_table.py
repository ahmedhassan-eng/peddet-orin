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
    "2026-01-01T00:00:00Z\n"
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
    for name in ("run1", "run2"):
        (tmp_path / name / "quiet.txt").write_text(QUIET_PASSED)
    block = generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    assert "checked against 3 further full runs" in block and "| OUT |" not in block
    assert "4 runs on one board, which is not a population" in block
    assert "- published (`runs/`): not quieted, load before every benchmark at or above" in block
    assert "- run1: quieted, every QUIET gate passed; one-minute" in block
    assert "- run3: not quieted, load before every benchmark" in block
    assert block.endswith("R-11 is met for every judged metric.")


def test_deleting_one_pass_flips_a_run_to_not_quieted(tmp_path):
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run1")
    quiet = tmp_path / "run1" / "quiet.txt"
    quiet.write_text(QUIET_PASSED)
    assert "- run1: quieted," in generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    quiet.write_text(QUIET_PASSED.replace("QUIET bench_fp16 PASS\n", ""))
    assert "- run1: not quieted, a QUIET gate did not pass" in generate_table.build_repeat_block(
        generate_table.RUNS_DIR, tmp_path
    )


def test_both_quiet_line_forms_are_read_and_the_earlier_one_is_noted(tmp_path):
    for name in ("run1", "run2"):
        _copy_runs(generate_table.RUNS_DIR, tmp_path / name)
    (tmp_path / "run1" / "quiet.txt").write_text(QUIET_PASSED.replace(" allowed_timers=3", ""))
    (tmp_path / "run2" / "quiet.txt").write_text(QUIET_PASSED)
    block = generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    assert "- run1: quieted, every QUIET gate passed (its gate lines predate the count of allowed timers);" in block
    assert "- run2: quieted, every QUIET gate passed; one-minute" in block


def test_a_quiet_line_naming_another_load_bound_is_refused(tmp_path):
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run1")
    (tmp_path / "run1" / "quiet.txt").write_text(QUIET_PASSED.replace("bound=1.0", "bound=3.0", 1))
    with pytest.raises(generate_table.TableError, match="load bound other than 1.0"):
        generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)


def test_a_further_run_outside_a_tolerance_is_reported_out_and_not_met(tmp_path):
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run1")
    _copy_runs(generate_table.RUNS_DIR, tmp_path / "run2", scale_power=1.5)
    block = generate_table.build_repeat_block(generate_table.RUNS_DIR, tmp_path)
    row = next(line for line in block.splitlines() if line.startswith("| avg_power_w | FP16 |"))
    assert row.endswith("| 10% | OUT |")
    assert "R-11 not met for avg_power_w FP16" in block and "avg_power_w FP32" in block
    assert compare_logs.TOLERANCES["avg_power_w"] == {"relative": 0.10}


def test_editing_a_number_in_the_int8_comparison_is_detected():
    text = generate_table.README.read_text()
    block = generate_table.readme_block(text, generate_table.INT8_MARKERS)
    number = re.search(r"\d+\.\d\d", block).group()
    edited = block.replace(number, number[:-1] + ("1" if number[-1] != "1" else "2"), 1)
    assert generate_table.readme_matches(text)
    assert not generate_table.readme_matches(text.replace(block, edited))
