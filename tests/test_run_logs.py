"""The shape of run logs: required fields, types, and path hygiene."""
import copy
import json
from pathlib import Path

import pytest

from peddet_orin import runlog

RUNS = sorted(runlog.RUNS_DIR.glob("*.json"))
VALID_LOG = {
    "schema_version": 1,
    "kind": "eval",
    "precision": "fp16",
    "commit": "abc123",
    "dirty": False,
    "environment": {name: "value" for name in runlog.REQUIRED_ENVIRONMENT},
    "command": "python -m peddet_orin.evaluate --precision fp16",
    "seed": 0,
    "iterations": {"images": 5000},
    "start_utc": "2026-01-01T00:00:00Z",
    "end_utc": "2026-01-01T00:10:00Z",
    "other_gpu_processes": 0,
    "results": {"ap": 0.5},
}


def test_valid_log_passes():
    assert runlog.validate_log(VALID_LOG) == []


@pytest.mark.parametrize("field", sorted(runlog.REQUIRED_FIELDS))
def test_each_required_field_is_enforced(field):
    broken = copy.deepcopy(VALID_LOG)
    del broken[field]
    assert runlog.validate_log(broken)


def test_wrong_type_is_rejected():
    broken = copy.deepcopy(VALID_LOG)
    broken["seed"] = "zero"
    assert runlog.validate_log(broken)


def test_absolute_home_path_in_command_is_rejected():
    broken = copy.deepcopy(VALID_LOG)
    # The path is computed so that no such literal is stored in the repository.
    broken["command"] = f"python -m peddet_orin.evaluate --out {Path.home() / 'results.json'}"
    assert any("forbidden path" in problem for problem in runlog.validate_log(broken))


def test_nested_absolute_path_is_rejected():
    broken = copy.deepcopy(VALID_LOG)
    broken["results"]["note"] = str(Path.home())
    assert runlog.validate_log(broken)


def test_allowlisted_file_and_urls_are_accepted():
    allowed = copy.deepcopy(VALID_LOG)
    allowed["results"]["source"] = runlog.TEGRA_RELEASE_FILE
    allowed["results"]["url"] = "https://docs.python.org/3/library"
    allowed["results"]["relative"] = "usr/lib/thing"
    assert runlog.validate_log(allowed) == []


@pytest.mark.parametrize("path", RUNS, ids=lambda item: item.name)
def test_committed_logs_are_valid(path):
    assert runlog.validate_log(json.loads(path.read_text())) == []
