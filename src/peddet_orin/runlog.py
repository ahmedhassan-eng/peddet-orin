"""JSON run logs: writing them, recording host facts, and validating their shape.

Every number published by this project comes from a file under ``runs/``. A log
is written only if the run completes, so a crashed run cannot supply a number.
Paths are written relative to the repository root, and validation rejects any
absolute path that is not in the system-file allowlist.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO_ROOT / "runs"
UNREADABLE = "UNREADABLE"
SCHEMA_VERSION = 1
SUBPROCESS_TIMEOUT_S = 20
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
FILE_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
TEGRA_RELEASE_FILE = "/etc/nv_tegra_release"
ABSOLUTE_PATH_ALLOWLIST = frozenset({TEGRA_RELEASE_FILE})
# A slash that begins a token, so URLs and relative paths such as lib/x do not match.
ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![\w:/.~-])/[\w.+-]+(?:/[\w.+-]+)*")
HOME_PATH_PATTERN = re.compile(r"~/")

REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "schema_version": int,
    "kind": str,
    "precision": str,
    "commit": str,
    "dirty": bool,
    "environment": dict,
    "command": str,
    "seed": int,
    "iterations": dict,
    "start_utc": str,
    "end_utc": str,
    "other_gpu_processes": (int, str),
    "results": dict,
}
REQUIRED_ENVIRONMENT = (
    "nv_tegra_release",
    "tensorrt",
    "torch",
    "torchvision",
    "onnxruntime",
    "nvpmodel",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)


def _command_output(arguments: list[str]) -> str:
    """Return a command's stripped stdout, or ``UNREADABLE`` if it cannot be run."""
    if shutil.which(arguments[0]) is None:
        return UNREADABLE
    try:
        done = subprocess.run(arguments, capture_output=True, text=True, timeout=SUBPROCESS_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return UNREADABLE
    return done.stdout.strip() if done.returncode == 0 and done.stdout.strip() else UNREADABLE


def _module_version(name: str) -> str:
    """Return an importable module's version string, or ``UNREADABLE``."""
    try:
        return str(__import__(name).__version__)
    except (ImportError, AttributeError):
        return UNREADABLE


def environment_facts() -> dict[str, str]:
    """Collect the host's own strings. Anything unavailable is recorded as such."""
    try:
        release = Path(TEGRA_RELEASE_FILE).read_text().strip()
    except OSError:
        release = UNREADABLE
    facts = {"nv_tegra_release": release, "nvpmodel": _command_output(["nvpmodel", "-q"])}
    for name in ("tensorrt", "torch", "torchvision", "onnxruntime"):
        facts[name] = _module_version(name)
    return facts


def other_gpu_process_count() -> int | str:
    """Count compute processes through NVML; only the number is ever kept."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return len(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # NVML raises its own error family, and may be absent on Jetson.
        return UNREADABLE


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def relative_command(arguments: list[str]) -> str:
    """Render an invocation with any path under the repository made relative."""
    root = f"{REPO_ROOT}{os.sep}"
    return " ".join(argument.replace(root, "") for argument in arguments)


class RunLog:
    """Context manager that writes one JSON log when its block finishes cleanly."""

    def __init__(self, kind: str, precision: str, seed: int, iterations: dict[str, int]) -> None:
        self.kind = kind
        self.precision = precision
        self.seed = seed
        self.iterations = iterations
        self.results: dict[str, Any] = {}
        self.path: Path | None = None

    def __enter__(self) -> "RunLog":
        self.start_utc = _utc_now()
        self.gpu_processes = other_gpu_process_count()
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        if exception_type is None:
            self.path = self._write()
        return False

    def _write(self) -> Path:
        module = sys.modules["__main__"].__spec__
        program = f"python -m {module.name}" if module else "python"
        record = {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "precision": self.precision,
            "commit": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--", ".", ":!runs", ":!README.md")),
            "environment": environment_facts(),
            "command": relative_command([program, *sys.argv[1:]]),
            "seed": self.seed,
            "iterations": self.iterations,
            "start_utc": self.start_utc,
            "end_utc": _utc_now(),
            "other_gpu_processes": self.gpu_processes,
            "results": self.results,
        }
        problems = validate_log(record)
        if problems:
            raise ValueError(f"refusing to write an invalid log: {problems}")
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime(FILE_STAMP_FORMAT)
        path = RUNS_DIR / f"{self.kind}_{self.precision}_{stamp}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"log written: {path.relative_to(REPO_ROOT)}")
        return path


def _strings_in(value: Any) -> list[str]:
    """Collect every string in a nested structure, dictionary keys included."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for key, item in value.items() for text in _strings_in(key) + _strings_in(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _strings_in(item)]
    return []


def forbidden_paths(text: str) -> list[str]:
    """Return the absolute or home-relative paths in ``text`` outside the allowlist."""
    found = [path for path in ABSOLUTE_PATH_PATTERN.findall(text) if path not in ABSOLUTE_PATH_ALLOWLIST]
    return found + HOME_PATH_PATTERN.findall(text)


def validate_log(record: Any) -> list[str]:
    """Return every schema problem found in a log; an empty list means it is valid."""
    if not isinstance(record, dict):
        return ["log is not an object"]
    problems = []
    for field, expected in REQUIRED_FIELDS.items():
        if field not in record:
            problems.append(f"missing field: {field}")
        elif not isinstance(record[field], expected):
            problems.append(f"wrong type for field: {field}")
    environment = record.get("environment")
    if isinstance(environment, dict):
        problems += [f"missing fact: {name}" for name in REQUIRED_ENVIRONMENT if name not in environment]
    for field in ("start_utc", "end_utc"):
        try:
            datetime.strptime(str(record.get(field)), TIMESTAMP_FORMAT)
        except ValueError:
            problems.append(f"bad timestamp: {field}")
    for text in _strings_in(record):
        problems += [f"forbidden path: {path}" for path in forbidden_paths(text)]
    return problems
