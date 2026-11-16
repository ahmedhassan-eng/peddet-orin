# SPDX-License-Identifier: Apache-2.0
"""The shape of run logs: required fields, types, per-kind results, path hygiene, and writing them."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from peddet_orin import runlog

RUNS = sorted(runlog.RUNS_DIR.glob("*.json"))
COMMIT = "0123456789abcdef0123456789abcdef01234567"
MANIFEST_SHA256 = "ab" * 32
# Paths are built at run time, so that no absolute or home path is stored in the repository.
HOME = str(Path.home())
FILE_URL = "file://" + str(Path.home() / "model.onnx")
TILDE_PATH = "~" + "/".join(("", "models", "x.onnx"))


def latency(p50: float) -> dict:
    return {"p50_ms": p50, "p90_ms": p50 * 1.1, "p99_ms": p50 * 1.3, "fps": 1000.0 / p50}


RESULTS = {
    "eval": {"ap": 0.25, "ap50": 0.45, "detections": 12000, "runtime": "tensorrt-gpu"},
    "benchmark": {
        "runtime": "tensorrt-gpu",
        "engine": latency(4.0),
        "end_to_end": latency(9.0),
        "repeats": 3,
        "engine_p50_ms_per_repeat": [3.9, 4.0, 4.1],
        "end_to_end_p50_ms_per_repeat": [8.8, 9.0, 9.3],
        "peak_process_rss_mib": 900.0,
        "system_ram_rise_mib": 120.0,
        "avg_power_w": 7.25,
        "power_samples": 40,
        "power_mode": "NV Power Mode: 25W",
        "cpu_load_1min_before_timing": 0.4,
        "gpu_busy_pct_before_timing": 0.0,
        "gpu_clock": {
            "governor": "performance",
            "max_hz": 1020000000,
            "min_hz": 306000000,
            "cur_hz_counts": {"1020000000": 300},
            "samples": 300,
            "unreadable_samples": 0,
            "at_max_samples": 300,
        },
        "cpu_clocks": {
            "policy0": {
                "governor": "schedutil",
                "max_khz": 1728000,
                "cur_khz_counts": {"1728000": 290, "729600": 10},
                "samples": 300,
                "unreadable_samples": 0,
                "at_max_samples": 290,
            },
        },
    },
    "build": {
        "engine_bytes": 9000000,
        "build_seconds": 120.5,
        "int8_path": "-",
        "entropy_failure": "",
        "calibration_manifest_sha256": "-",
    },
    "export": {"max_abs_difference": 1e-5, "tolerance": 1e-4, "opset": 17, "input_shape": [1, 3, 320, 320]},
    "gpu_check": {
        "cuda_available": True,
        "arch_list": ["sm_80", "sm_90"],
        "device_compute_capability": "8.7",
        "matmul_max_abs_difference": 0.0,
        "matmul_tolerance": 1e-3,
        "matmul_matches": True,
        "matmul_error": "",
        "nms_comparison": {"nms_identical": True},
        "nms_matches": True,
        "nms_error": "",
        "gpu_torch_usable": True,
        "torch_warning": "",
        "cudnn_allow_tf32": False,
        "matmul_allow_tf32": False,
    },
}
DEFAULT_PRECISION = {"eval": "fp16", "benchmark": "fp16", "build": "fp16", "export": "onnx", "gpu_check": "fp32"}


def make_log(kind: str = "eval", precision: str | None = None, **fields) -> dict:
    """Return a valid schema-2 log of one kind; ``results=`` entries are merged, other fields replace."""
    results = copy.deepcopy(RESULTS[kind])
    results.update(fields.pop("results", {}))
    log = {
        "schema_version": 2,
        "kind": kind,
        "precision": precision or DEFAULT_PRECISION[kind],
        "commit": COMMIT,
        "dirty": False,
        "environment": {name: "value" for name in runlog.REQUIRED_ENVIRONMENT},
        "command": f"python -m peddet_orin.{kind}",
        "seed": 0,
        "iterations": {"images": 5000},
        "start_utc": "2026-01-01T00:00:00Z",
        "end_utc": "2026-01-01T00:10:00Z",
        "other_gpu_processes": 0,
        "results": results,
    }
    log.update(fields)
    return log


VALID_LOG = make_log()


def test_valid_log_passes():
    assert runlog.validate_log(VALID_LOG) == []


@pytest.mark.parametrize("kind", sorted(RESULTS))
def test_a_valid_log_of_each_kind_is_accepted(kind):
    assert runlog.validate_log(make_log(kind)) == []


def test_a_benchmark_without_tegrastats_records_power_and_ram_as_unreadable():
    log = make_log("benchmark", "fp16", results={"avg_power_w": "UNREADABLE", "system_ram_rise_mib": "UNREADABLE"})
    assert runlog.validate_log(log) == []


def test_a_valid_int8_build_log_is_accepted():
    log = make_log("build", "int8", results={"int8_path": "entropy", "calibration_manifest_sha256": MANIFEST_SHA256})
    assert runlog.validate_log(log) == []


@pytest.mark.parametrize("field", sorted(runlog.REQUIRED_FIELDS))
def test_each_required_field_is_enforced(field):
    broken = copy.deepcopy(VALID_LOG)
    del broken[field]
    assert runlog.validate_log(broken)


def test_wrong_type_is_rejected():
    broken = copy.deepcopy(VALID_LOG)
    broken["seed"] = "zero"
    assert runlog.validate_log(broken)


BAD_FIELDS = {
    "short commit": {"commit": "abc123"},
    "empty commit": {"commit": ""},
    "uppercase commit": {"commit": COMMIT.upper()},
    "bool seed": {"seed": True},
    "float seed": {"seed": 0.0},
    "start after end": {"start_utc": "2026-01-01T00:10:01Z"},
    "schema version 1": {"schema_version": 1},
    "schema version 3": {"schema_version": 3},
    "bool schema version": {"schema_version": True},
}


@pytest.mark.parametrize("fields", BAD_FIELDS.values(), ids=BAD_FIELDS.keys())
def test_each_bad_top_level_field_is_rejected(fields):
    assert runlog.validate_log(make_log(**fields))


def test_an_unknown_kind_is_rejected():
    assert "unknown kind: train" in runlog.validate_log(dict(make_log(), kind="train"))


@pytest.mark.parametrize("fact", runlog.REQUIRED_ENVIRONMENT)
def test_each_environment_fact_is_required(fact):
    broken = make_log()
    del broken["environment"][fact]
    assert f"missing fact: {fact}" in runlog.validate_log(broken)


def test_the_new_environment_facts_are_required_and_collected():
    assert {"python", "onnx", "cuda", "peddet_orin"} <= set(runlog.REQUIRED_ENVIRONMENT)
    facts = runlog.environment_facts()
    assert set(runlog.REQUIRED_ENVIRONMENT) <= set(facts)
    assert facts["python"].count(".") == 2


def test_the_decode_facts_are_recorded_as_strings():
    facts = runlog.environment_facts()
    assert set(runlog.DECODE_ENVIRONMENT) <= set(facts)
    assert all(isinstance(facts[name], str) and facts[name] for name in runlog.DECODE_ENVIRONMENT)
    import numpy
    import PIL

    assert (facts["numpy"], facts["pillow"]) == (numpy.__version__, PIL.__version__)
    from PIL import features

    turbo = features.version_feature("libjpeg_turbo")
    assert facts["libjpeg"] == (f"libjpeg-turbo {turbo}" if turbo else f"libjpeg {features.version('jpg')}")


@pytest.mark.parametrize("fact", runlog.DECODE_ENVIRONMENT)
def test_a_decode_fact_that_is_not_a_string_is_rejected(fact):
    broken = make_log()
    broken["environment"][fact] = 3
    assert f"fact is not a string: {fact}" in runlog.validate_log(broken)
    del broken["environment"][fact]
    assert runlog.validate_log(broken) == []


BAD_RESULTS = [
    ("eval", {"ap": 1.5}),
    ("eval", {"ap": -0.1}),
    ("eval", {"ap50": "0.4"}),
    ("eval", {"ap": 0.5, "ap50": 0.4}),
    ("eval", {"detections": -1}),
    ("eval", {"detections": 1.0}),
    ("eval", {"runtime": 3}),
    ("benchmark", {"engine": {"p50_ms": 1.0}}),
    ("benchmark", {"end_to_end": dict(latency(9.0), p99_ms="slow")}),
    ("benchmark", {"repeats": 0}),
    ("benchmark", {"repeats": True}),
    ("benchmark", {"engine_p50_ms_per_repeat": [4.0, 4.1]}),
    ("benchmark", {"end_to_end_p50_ms_per_repeat": [9.0, 9.0, "x"]}),
    ("benchmark", {"avg_power_w": "watts"}),
    ("benchmark", {"system_ram_rise_mib": "lots"}),
    ("benchmark", {"system_ram_rise_mib": float("nan")}),
    ("benchmark", {"power_samples": 1.5}),
    ("benchmark", {"gpu_busy_pct_before_timing": "busy"}),
    ("benchmark", {"gpu_clock": "UNREADABLE"}),
    ("benchmark", {"gpu_clock": {"governor": "performance"}}),
    ("benchmark", {"cpu_clocks": {}}),
    ("benchmark", {"cpu_clocks": {"policy0": {"governor": "schedutil"}}}),
    ("benchmark", {"cpu_load_1min": 0.4}),
    ("build", {"engine_bytes": 0}),
    ("build", {"build_seconds": "2m"}),
    ("build", {"int8_path": "entropy"}),
    ("build", {"calibration_manifest_sha256": MANIFEST_SHA256}),
    ("export", {"opset": 17.0}),
    ("export", {"input_shape": [1, 3, 320.0, 320]}),
    ("export", {"tolerance": None}),
    ("gpu_check", {"torch_warning": None}),
    ("gpu_check", {"cudnn_allow_tf32": "False"}),
    ("gpu_check", {"matmul_allow_tf32": 0}),
    ("gpu_check", {"arch_list": "sm_87"}),
]


@pytest.mark.parametrize(("kind", "results"), BAD_RESULTS, ids=[f"{k}-{sorted(r)[0]}" for k, r in BAD_RESULTS])
def test_each_bad_result_is_rejected(kind, results):
    assert runlog.validate_log(make_log(kind, results=results))


@pytest.mark.parametrize("kind", sorted(RESULTS))
def test_each_required_result_is_enforced(kind):
    for key in RESULTS[kind]:
        broken = make_log(kind)
        del broken["results"][key]
        assert f"missing result: {kind} {key}" in runlog.validate_log(broken), key


@pytest.mark.parametrize(
    "results",
    [{"calibration_manifest_sha256": "-", "int8_path": "entropy"}, {"int8_path": "-", "calibration_manifest_sha256":
                                                                    MANIFEST_SHA256}],
    ids=["no-manifest-hash", "no-route"],
)
def test_a_bad_int8_build_log_is_rejected(results):
    assert runlog.validate_log(make_log("build", "int8", results=results))


def test_absolute_home_path_in_command_is_rejected():
    broken = copy.deepcopy(VALID_LOG)
    # The path is computed so that no such literal is stored in the repository.
    broken["command"] = f"python -m peddet_orin.evaluate --out {Path.home() / 'results.json'}"
    assert any("forbidden path" in problem for problem in runlog.validate_log(broken))


def test_nested_absolute_path_is_rejected():
    broken = copy.deepcopy(VALID_LOG)
    broken["results"]["runtime"] = str(Path.home())
    assert runlog.validate_log(broken)


@pytest.mark.parametrize("text", [FILE_URL, f"see {TILDE_PATH}", f"a:{HOME}"], ids=["file-url", "tilde", "colon"])
def test_file_urls_home_and_colon_paths_are_rejected(text):
    broken = make_log(results={"runtime": text})
    assert any("forbidden path" in problem for problem in runlog.validate_log(broken))


def test_allowlisted_file_and_urls_are_accepted():
    allowed = make_log("gpu_check")
    allowed["results"]["matmul_error"] = runlog.TEGRA_RELEASE_FILE
    allowed["results"]["nms_error"] = "https://docs.python.org/3/library"
    allowed["results"]["torch_warning"] = "usr/lib/thing"
    assert runlog.validate_log(allowed) == []


def test_redaction_replaces_every_path_kind_and_keeps_urls():
    text = (
        f"[TRT] error: cannot open {HOME}/engine.plan; also {FILE_URL} and {TILDE_PATH}, "
        f"see https://docs.python.org/3/x and {runlog.TEGRA_RELEASE_FILE}"
    )
    out = runlog.redact_paths(text)
    assert out == (
        "[TRT] error: cannot open <path>; also <path> and <path>, "
        f"see https://docs.python.org/3/x and {runlog.TEGRA_RELEASE_FILE}"
    )
    assert runlog.forbidden_paths(out) == []


# Writing a log: the git and host facts are replaced so that the test needs neither a board nor history.


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(runlog, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(runlog, "_git", lambda *arguments: COMMIT if arguments[0] == "rev-parse" else "")
    monkeypatch.setattr(runlog, "environment_facts", lambda: {name: "value" for name in runlog.REQUIRED_ENVIRONMENT})
    monkeypatch.setattr(runlog, "other_gpu_process_count", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--precision", "fp16", "--out", str(tmp_path / "x.json")])
    return tmp_path


def test_runlog_writes_one_valid_log_with_relative_paths(fake_repo):
    with runlog.RunLog("eval", "fp16", 0, {"images": 5000}) as log:
        log.results.update(RESULTS["eval"])
    record = json.loads(log.path.read_text())
    assert log.path.parent == fake_repo / "runs" and log.path.name.startswith("eval_fp16_")
    assert runlog.validate_log(record) == []
    assert record["commit"] == COMMIT and record["dirty"] is False and record["schema_version"] == 2
    assert record["command"].endswith("--precision fp16 --out x.json")


def test_runlog_redacts_an_absolute_path_in_a_result_instead_of_raising(fake_repo):
    error = f"TensorRT: failed to open {Path.home() / 'cache' / 'timing.cache'}: no such file"
    with runlog.RunLog("eval", "fp16", 0, {"images": 1}) as log:
        log.results.update(RESULTS["eval"], runtime=error)
    record = json.loads(log.path.read_text())
    assert record["results"]["runtime"] == "TensorRT: failed to open <path>: no such file"


def test_runlog_redacts_a_file_url_and_a_home_path_instead_of_raising(fake_repo):
    with runlog.RunLog("eval", "fp16", 0, {"images": 1}) as log:
        log.results.update(RESULTS["eval"], runtime=f"{FILE_URL} or {TILDE_PATH}")
    assert json.loads(log.path.read_text())["results"]["runtime"] == "<path> or <path>"


def test_runlog_writes_nothing_when_the_block_raises(fake_repo):
    with pytest.raises(KeyError), runlog.RunLog("eval", "fp16", 0, {}):
        raise KeyError("crash")
    assert not (fake_repo / "runs").exists()


def test_runlog_refuses_invalid_results(fake_repo):
    with pytest.raises(ValueError, match="invalid log"), runlog.RunLog("eval", "fp16", 0, {}) as log:
        log.results.update(RESULTS["eval"], ap=2.0)
    assert not (fake_repo / "runs").exists()


def test_runlog_fails_on_entry_without_a_commit(fake_repo, monkeypatch):
    monkeypatch.setattr(runlog, "_git", lambda *arguments: "")
    with pytest.raises(RuntimeError, match="git commit"), runlog.RunLog("eval", "fp16", 0, {}):
        pass


def test_relative_command_makes_repository_paths_relative(monkeypatch, tmp_path):
    monkeypatch.setattr(runlog, "REPO_ROOT", tmp_path)
    arguments = ["python -m peddet_orin.export", "--out", str(tmp_path / "artifacts" / "a.onnx"),
                 f"--onnx={tmp_path / 'b.onnx'}", str(tmp_path)]
    assert runlog.relative_command(arguments) == "python -m peddet_orin.export --out artifacts/a.onnx --onnx=b.onnx ."


def test_relative_command_leaves_other_paths_for_redaction(monkeypatch, tmp_path):
    monkeypatch.setattr(runlog, "REPO_ROOT", tmp_path / "repo")
    elsewhere = str(tmp_path / "elsewhere" / "a.onnx")
    rendered = runlog.relative_command(["python", elsewhere, str(tmp_path / "repository")])
    assert rendered == f"python {elsewhere} {tmp_path / 'repository'}"
    assert runlog.redact_paths(rendered) == "python <path> <path>"


# The dirty flag, in a real git repository.


GIT_IDENTITY = ("-c", "user.name=Ahmed Hassan", "-c", "user.email=1659721+ahmedhassan-eng@users.noreply.github.com")


def _git_in(repo: Path, *arguments: str) -> None:
    subprocess.run(["git", *GIT_IDENTITY, *arguments], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    _git_in(tmp_path, "init", "-q")
    (tmp_path / "README.md").write_text("table\n")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "old.json").write_text("{}\n")
    _git_in(tmp_path, "add", ".")
    _git_in(tmp_path, "commit", "-q", "-m", "start")
    monkeypatch.setattr(runlog, "REPO_ROOT", tmp_path)
    return tmp_path


def test_a_clean_tree_is_not_dirty_and_its_commit_is_read(git_repo):
    assert runlog.tree_is_dirty() is False
    assert runlog.COMMIT_PATTERN.fullmatch(runlog._git("rev-parse", "HEAD"))


def test_a_hand_edited_readme_makes_the_tree_dirty(git_repo):
    (git_repo / "README.md").write_text("table, edited by hand\n")
    assert runlog.tree_is_dirty() is True


def test_runlog_records_a_hand_edited_readme_as_dirty(git_repo, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", git_repo / "runs")
    monkeypatch.setattr(runlog, "environment_facts", lambda: {name: "value" for name in runlog.REQUIRED_ENVIRONMENT})
    monkeypatch.setattr(runlog, "other_gpu_process_count", lambda: 0)
    monkeypatch.setattr(sys, "argv", ["evaluate.py"])
    (git_repo / "README.md").write_text("table, edited by hand\n")
    with runlog.RunLog("eval", "fp16", 0, {"images": 1}) as log:
        log.results.update(RESULTS["eval"])
    assert json.loads(log.path.read_text())["dirty"] is True


def test_new_and_changed_logs_under_runs_leave_the_tree_clean(git_repo):
    (git_repo / "runs" / "old.json").write_text('{"changed": 1}\n')
    (git_repo / "runs" / "new.json").write_text("{}\n")
    assert runlog.tree_is_dirty() is False


@pytest.mark.parametrize("path", RUNS, ids=lambda item: item.name)
def test_committed_logs_are_valid(path):
    assert runlog.validate_log(json.loads(path.read_text())) == []
