"""scripts/generate_table.py: which logs it picks, what it refuses, and what the block says."""
import json

import generate_table
from test_run_logs import COMMIT, make_log

OTHER_COMMIT = "f" * 40
EMPTY_BLOCK = "\n".join(
    [
        "| Device and precision | Engine p50/p90/p99 (ms) | End-to-end p50/p90/p99 (ms) | Engine FPS | "
        "End-to-end FPS | Peak process RSS (MiB) | Avg power VDD_IN (W) | Person AP@[.5:.95] | Person AP50 | "
        "INT8 path |",
        "|---|---|---|---|---|---|---|---|---|---|",
        "| n/a FP32 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | - |",
        "| n/a FP16 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | - |",
        "| n/a INT8 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |",
    ]
)


def write_logs(directory, logs):
    for index, log in enumerate(logs):
        (directory / f"{index:02d}.json").write_text(json.dumps(log))
    return directory


def test_latest_picks_the_newer_of_two_clean_logs():
    older = make_log("eval", "fp16", end_utc="2026-01-01T00:10:00Z", results={"ap": 0.20})
    newer = make_log("eval", "fp16", end_utc="2026-01-02T00:10:00Z", results={"ap": 0.30})
    assert generate_table.latest([older, newer], "eval", "fp16") is newer
    assert generate_table.latest([newer, older], "eval", "fp16") is newer


def test_latest_skips_a_newer_dirty_log_and_says_so(capsys):
    clean = make_log("eval", "fp16", end_utc="2026-01-01T00:10:00Z")
    dirty = make_log("eval", "fp16", end_utc="2026-01-02T00:10:00Z", dirty=True)
    assert generate_table.latest([clean, dirty], "eval", "fp16") is clean
    assert "skipping dirty log: eval fp16 ending 2026-01-02T00:10:00Z" in capsys.readouterr().err


def test_only_dirty_logs_give_the_empty_table(tmp_path, capsys):
    write_logs(tmp_path, [make_log("eval", "fp16", dirty=True)])
    assert generate_table.build_block(tmp_path) == EMPTY_BLOCK
    assert "skipping dirty log" in capsys.readouterr().err


def test_no_logs_print_the_empty_table_and_exit_zero(tmp_path, capsys):
    assert generate_table.main(["--runs", str(tmp_path)]) == 0
    assert capsys.readouterr().out == EMPTY_BLOCK + "\n"


def test_logs_from_two_commits_exit_nonzero_with_one_line(tmp_path, capsys):
    write_logs(tmp_path, [make_log("eval", "fp16"), make_log("benchmark", "fp32", commit=OTHER_COMMIT)])
    assert generate_table.main(["--runs", str(tmp_path)]) == 1
    error = capsys.readouterr().err
    assert error.count("\n") == 1 and "2 commits" in error and COMMIT in error and OTHER_COMMIT in error


def test_an_older_log_from_another_commit_does_not_count(tmp_path):
    old = make_log("eval", "fp16", commit=OTHER_COMMIT, end_utc="2026-01-01T00:05:00Z")
    write_logs(tmp_path, [old, make_log("eval", "fp16"), make_log("benchmark", "fp16")])
    assert "| GPU, TensorRT FP16 |" in generate_table.build_block(tmp_path)


def test_an_eval_and_benchmark_runtime_mismatch_exits_nonzero(tmp_path, capsys):
    write_logs(tmp_path, [make_log("eval", "fp32", results={"runtime": "pytorch-cpu"}),
                          make_log("benchmark", "fp32", results={"runtime": "pytorch-cuda"})])
    assert generate_table.main(["--runs", str(tmp_path)]) == 1
    error = capsys.readouterr().err
    assert error.count("\n") == 1 and "FP32" in error and "pytorch-cpu" in error and "pytorch-cuda" in error


def test_the_ram_rise_column_is_gone_and_power_has_one_decimal(tmp_path):
    write_logs(tmp_path, [make_log("benchmark", "fp16", results={"system_ram_rise_mib": 123.0, "avg_power_w": 7.26})])
    block = generate_table.build_block(tmp_path)
    assert "RAM rise" not in block and "123" not in block
    row = next(line for line in block.splitlines() if line.startswith("| GPU, TensorRT FP16"))
    assert " | 7.3 | " in row and "7.26" not in row


def test_each_benchmark_row_has_a_note_with_its_repeat_ranges(tmp_path):
    write_logs(tmp_path, [make_log("benchmark", precision) for precision in ("fp16", "int8")])
    block = generate_table.build_block(tmp_path)
    for name in ("FP16", "INT8"):
        assert (f"- {name} p50 across repeats: engine 3.90-4.10 ms over 3 repeats, "
                "end-to-end 8.80-9.30 ms over 3 repeats.") in block
    assert "one-minute CPU load before timing: 0.4" in block


def int8_drop_block(tmp_path, fp16_ap, int8_ap):
    write_logs(tmp_path, [make_log("eval", "fp16", results={"ap": fp16_ap}),
                          make_log("eval", "int8", results={"ap": int8_ap})])
    return generate_table.build_block(tmp_path)


def test_an_int8_drop_of_exactly_two_points_is_within_the_limit(tmp_path):
    # (0.32 - 0.30) * 100 is 2.0000000000000018 in binary floating point.
    block = int8_drop_block(tmp_path, 0.32, 0.30)
    assert "more than the" not in block and "INT8 person AP differs from FP16 by -2.00 points." in block


def test_an_int8_drop_just_over_two_points_is_called_out(tmp_path):
    block = int8_drop_block(tmp_path, 0.30, 0.2799)
    assert "INT8 person AP is 2.01 points below FP16, more than the 2-point limit." in block


def test_the_gpu_check_note_quotes_the_warning_and_states_tf32(tmp_path):
    warning = "Found GPU0 Orin which is of compute capability 8.7.\nMinimum and Maximum cuda capability ..."
    write_logs(tmp_path, [make_log("gpu_check", results={"torch_warning": warning, "cudnn_allow_tf32": False,
                                                         "matmul_allow_tf32": True})])
    block = generate_table.build_block(tmp_path)
    assert ('PyTorch emitted an unsupported-GPU warning at first CUDA use: '
            '"Found GPU0 Orin which is of compute capability 8.7."') in block
    assert "Minimum and Maximum" not in block and "binary-compatibility" not in block
    assert "cudnn.allow_tf32 False and cuda.matmul.allow_tf32 True" in block


def test_the_gpu_check_note_says_when_there_was_no_warning(tmp_path):
    write_logs(tmp_path, [make_log("gpu_check")])
    assert "PyTorch emitted no unsupported-GPU warning at first CUDA use." in generate_table.build_block(tmp_path)
