# SPDX-License-Identifier: Apache-2.0
"""The board's pins equal the versions every judged run log records.

constraints/jetson-orin.txt pins the packages the board ran with. The judged logs under runs-repeat/ record the
versions of torch, torchvision, onnx and onnxruntime, so each of those pins must equal what every judged log
records. numpy, pycocotools and pillow are pinned too but no log records them, so they cannot be checked here.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = REPO_ROOT / "constraints" / "jetson-orin.txt"
JUDGED = ("run3", "run4", "run5")
LOGGED = ("torch", "torchvision", "onnx", "onnxruntime")


def pins(text: str) -> dict[str, str]:
    return dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)$", text, re.M))


def judged_logs() -> list[Path]:
    return sorted(path for run in JUDGED for path in (REPO_ROOT / "runs-repeat" / run).glob("*.json"))


def pin_problems(pinned: dict[str, str], logs: dict[str, dict]) -> list[str]:
    """Each ``log: package logged <v>, pinned <p>`` where a judged log records another version than the pin."""
    return [
        f"{name}: {package} logged {environment.get(package)}, pinned {pinned.get(package)}"
        for name, environment in sorted(logs.items())
        for package in LOGGED
        if environment.get(package) != pinned.get(package)
    ]


def test_there_are_judged_logs_to_compare():
    assert len(judged_logs()) == 30


def test_every_logged_pin_equals_what_every_judged_log_records():
    logs = {path.name: json.loads(path.read_text())["environment"] for path in judged_logs()}
    assert pin_problems(pins(CONSTRAINTS.read_text()), logs) == []


def test_a_pin_that_differs_from_the_logs_is_detected():
    board = pins(CONSTRAINTS.read_text())
    logs = {"a.json": {package: board[package] for package in LOGGED}}
    assert pin_problems({**board, "onnx": "0.0.1"}, logs) == [f"a.json: onnx logged {board['onnx']}, pinned 0.0.1"]
