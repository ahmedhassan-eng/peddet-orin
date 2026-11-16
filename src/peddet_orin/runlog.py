# SPDX-License-Identifier: Apache-2.0
"""JSON run logs: writing them, recording host facts, and validating their shape.

Every number published by this project comes from a log: the judged runs' under ``runs-repeat/``, the earlier
run's under ``runs/``. A log is written only if the run completes, so a crashed run cannot supply a number.
Paths under the repository root are written relative to it, and every other
absolute, home or ``file:`` path in a log string is replaced by ``<path>``
before the log is written. Validation still rejects any such path that is not
in the system-file allowlist, so a committed log cannot carry one.

The repository root is the source checkout the package runs from. The
environment variable ``PDORIN_ROOT`` overrides it, and must name a directory
whose ``pyproject.toml`` is peddet-orin's. With neither, importing this module
prints one error line and exits with status 2: logs, data and artifacts belong
to a checkout, never to the interpreter's installation tree.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from peddet_orin.root import (  # noqa: F401  (re-exported: tests and callers read them from runlog)
    BAD_OVERRIDE_MESSAGE,
    NO_CHECKOUT_MESSAGE,
    PROJECT_NAME,
    ROOT_ENVIRONMENT_VARIABLE,
    is_checkout,
    project_field,
    resolve_root,
)

# PRD R-13: outside a checkout every command refuses at import with one line and exit status 2, never a traceback.
try:
    REPO_ROOT = resolve_root()
except RuntimeError as error:
    print(f"peddet-orin: error: {error}", file=sys.stderr)
    raise SystemExit(2) from None
RUNS_DIR = REPO_ROOT / "runs"
UNREADABLE = "UNREADABLE"
SCHEMA_VERSION = 2
SUBPROCESS_TIMEOUT_S = 20
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
FILE_STAMP_FORMAT = "%Y%m%dT%H%M%SZ"
TEGRA_RELEASE_FILE = "/etc/nv_tegra_release"
ABSOLUTE_PATH_ALLOWLIST = frozenset({TEGRA_RELEASE_FILE})
REDACTED_PATH = "<path>"
# A slash that begins a token, so URLs and relative paths such as lib/x do not match.
# "https://host/x" never matches: its first slash is not followed by a name, and the
# others follow a slash or a letter. A slash right after a colon does match, so each
# entry of a colon-separated path list is found.
ABSOLUTE_PATH_PATTERN = re.compile(r"(?<![\w/.~-])/[\w.+-]+(?:/[\w.+-]+)*")
HOME_PATH_PATTERN = re.compile(r"(?<![\w/.~-])~[\w.+-]*/[\w.+/-]*")
FILE_URL_PATTERN = re.compile(r"\bfile:[^\s'\"<>]*", re.IGNORECASE)
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
# Paths that never make a run's tree dirty: the logs the runs themselves write.
DIRTY_EXCLUDES = (":!runs",)

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
# Fields typed int where JSON's true and false (Python bools) do not count.
INTEGER_FIELDS = frozenset({"schema_version", "seed"})
REQUIRED_ENVIRONMENT = (
    "nv_tegra_release",
    "tensorrt",
    "torch",
    "torchvision",
    "onnxruntime",
    "nvpmodel",
    "python",
    "onnx",
    "cuda",
    "peddet_orin",
)
# Recorded by every log from 0.1.1 on, and checked as strings when present; not required, so the committed logs
# written before them stay valid. JPEG decoding is inside every end-to-end iteration, so its library is recorded.
DECODE_ENVIRONMENT = ("numpy", "pillow", "libjpeg", "pycocotools")
# The CUDA runtime and driver versions as the runtime library in the measuring process reports them. Facts are
# collected when the log is written, after the run, so the library is the one the run already loaded: torch's for
# PyTorch, the one TensorRT runs with for engines. Recorded from 0.1.1 on, not required.
RUNTIME_ENVIRONMENT = ("cuda_runtime", "cuda_driver")
# The stages of a timed benchmark iteration, in the order they run; their times add up to the end-to-end time.
BENCHMARK_STAGES = ("decode", "preprocess", "engine", "postprocess")


def _utc_now() -> str:
    return datetime.now(UTC).strftime(TIMESTAMP_FORMAT)


def command_output(arguments: list[str]) -> str:
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


def _torch_cuda_version() -> str:
    """Return the CUDA version PyTorch was built with, or ``UNREADABLE`` (a CPU build has none)."""
    try:
        import torch

        return str(torch.version.cuda) if torch.version.cuda else UNREADABLE
    except (ImportError, AttributeError):
        return UNREADABLE


def _distribution_version(name: str) -> str:
    """Return an installed distribution's version, or ``UNREADABLE``."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return UNREADABLE


def _libjpeg_version() -> str:
    """Return the libjpeg Pillow decodes JPEG with, as "libjpeg-turbo <release>" or "libjpeg <version>", or
    ``UNREADABLE``. Pillow's plain "jpg" version is the libjpeg API level, which many turbo releases share."""
    try:
        from PIL import features

        turbo = features.version_feature("libjpeg_turbo") if features.check_feature("libjpeg_turbo") else None
        version = features.version("jpg")
    except Exception:  # Pillow raises its own errors for a missing codec; any failure means it was not read.
        return UNREADABLE
    if turbo:
        return f"libjpeg-turbo {turbo}"
    return f"libjpeg {version}" if version else UNREADABLE


def _cuda_version_text(value: int) -> str:
    """CUDA's integer version, 1000 * major + 10 * minor, as "major.minor"."""
    return f"{value // 1000}.{value % 1000 // 10}"


def _cuda_versions() -> tuple[str, str]:
    """The CUDA runtime and driver versions, or ``UNREADABLE`` for each that cannot be read (as on a CPU host)."""
    try:
        from peddet_orin import cuda_native

        library = cuda_native._runtime()
    except (ImportError, OSError):
        return UNREADABLE, UNREADABLE
    versions = []
    for function in ("cudaRuntimeGetVersion", "cudaDriverGetVersion"):
        value = ctypes.c_int(0)
        try:
            status = getattr(library, function)(ctypes.byref(value))
        except AttributeError:
            status = -1
        versions.append(_cuda_version_text(value.value) if status == 0 and value.value > 0 else UNREADABLE)
    return versions[0], versions[1]


def _package_version() -> str:
    """The version in the checkout's pyproject.toml, which is the code that runs; an install's metadata can be stale
    or absent (as under PYTHONPATH), so it is only the fallback."""
    version = project_field(REPO_ROOT / "pyproject.toml", "version")
    if version:
        return version
    try:
        return metadata.version(PROJECT_NAME)
    except metadata.PackageNotFoundError:
        return UNREADABLE


def environment_facts() -> dict[str, str]:
    """Collect the host's own strings. Anything unavailable is recorded as such."""
    try:
        release = Path(TEGRA_RELEASE_FILE).read_text().strip()
    except OSError:
        release = UNREADABLE
    facts = {"nv_tegra_release": release, "nvpmodel": command_output(["nvpmodel", "-q"])}
    for name in ("tensorrt", "torch", "torchvision", "onnxruntime", "onnx"):
        facts[name] = _module_version(name)
    facts["python"] = platform.python_version()
    facts["cuda"] = _torch_cuda_version()
    facts["peddet_orin"] = _package_version()
    facts["numpy"] = _module_version("numpy")
    facts["pillow"] = _module_version("PIL")
    facts["libjpeg"] = _libjpeg_version()
    facts["pycocotools"] = _distribution_version("pycocotools")
    facts["cuda_runtime"], facts["cuda_driver"] = _cuda_versions()
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
    """Return a git command's stripped stdout in the repository root, or "" if it fails."""
    try:
        done = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=SUBPROCESS_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def tree_is_dirty() -> bool:
    """Report whether any file outside ``runs/`` differs from the commit, untracked files included."""
    return bool(_git("status", "--porcelain", "--", ".", *DIRTY_EXCLUDES))


def relative_command(arguments: list[str]) -> str:
    """Render an invocation with any path under the repository made relative."""
    root = str(REPO_ROOT)
    prefix = f"{root}{os.sep}"
    return " ".join("." if argument == root else argument.replace(prefix, "") for argument in arguments)


def redact_paths(text: str) -> str:
    """Replace every absolute, home or ``file:`` path outside the allowlist with ``<path>``."""
    text = FILE_URL_PATTERN.sub(REDACTED_PATH, text)
    text = ABSOLUTE_PATH_PATTERN.sub(
        lambda match: match.group() if match.group() in ABSOLUTE_PATH_ALLOWLIST else REDACTED_PATH, text
    )
    return HOME_PATH_PATTERN.sub(REDACTED_PATH, text)


def redacted(value: Any) -> Any:
    """Return a copy of a nested structure with every string, dictionary keys included, redacted."""
    if isinstance(value, str):
        return redact_paths(value)
    if isinstance(value, dict):
        return {redacted(key): redacted(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redacted(item) for item in value]
    return value


class RunLog:
    """Context manager that writes one JSON log when its block finishes cleanly.

    The commit and the dirty flag are read on entry, when the code that runs is the
    code on disk. Entry fails at once if the commit cannot be read, rather than
    after a long run.
    """

    def __init__(self, kind: str, precision: str, seed: int, iterations: dict[str, int]) -> None:
        self.kind = kind
        self.precision = precision
        self.seed = seed
        self.iterations = iterations
        self.results: dict[str, Any] = {}
        self.path: Path | None = None

    def __enter__(self) -> RunLog:
        self.commit = _git("rev-parse", "HEAD")
        if not COMMIT_PATTERN.fullmatch(self.commit):
            raise RuntimeError("cannot read the git commit of the repository root; run from a git checkout")
        self.dirty = tree_is_dirty()
        self.start_utc = _utc_now()
        self.gpu_processes = other_gpu_process_count()
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        if exception_type is None:
            self.path = self._write()
        return False

    def _write(self) -> Path:
        module = getattr(sys.modules["__main__"], "__spec__", None)
        program = f"python -m {module.name}" if module else "python"
        record = redacted(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": self.kind,
                "precision": self.precision,
                "commit": self.commit,
                "dirty": self.dirty,
                "environment": environment_facts(),
                "command": relative_command([program, *sys.argv[1:]]),
                "seed": self.seed,
                "iterations": self.iterations,
                "start_utc": self.start_utc,
                "end_utc": _utc_now(),
                "other_gpu_processes": self.gpu_processes,
                "results": self.results,
            }
        )
        problems = validate_log(record)
        if problems:
            raise ValueError(f"refusing to write an invalid log: {problems}")
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime(FILE_STAMP_FORMAT)
        path = RUNS_DIR / f"{self.kind}_{self.precision}_{stamp}.json"
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"log written: {relative_command([str(path)])}")
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
    """Return the absolute, home-relative or ``file:`` paths in ``text`` outside the allowlist."""
    found = FILE_URL_PATTERN.findall(text)
    text = FILE_URL_PATTERN.sub(" ", text)
    found += [path for path in ABSOLUTE_PATH_PATTERN.findall(text) if path not in ABSOLUTE_PATH_ALLOWLIST]
    return found + HOME_PATH_PATTERN.findall(text)


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_string(value: Any) -> bool:
    return isinstance(value, str)


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _is_fraction(value: Any) -> bool:
    return _is_number(value) and 0 <= value <= 1


def _is_latency(value: Any) -> bool:
    keys = ("p50_ms", "p90_ms", "p99_ms", "fps")
    return isinstance(value, dict) and set(value) == set(keys) and all(_is_number(value[key]) for key in keys)


def _is_stages(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == set(BENCHMARK_STAGES)
        and all(_is_latency(item) for item in value.values())
    )


def _is_postprocess_settings(value: Any) -> bool:
    """The settings, with ``kept_labels``, a non-empty list of distinct labels, only for a postprocess that keeps
    some classes."""
    thresholds = ("score_threshold", "nms_iou_threshold")
    counts = ("top_k_per_class", "detections_per_image", "num_classes")
    if not isinstance(value, dict):
        return False
    labels = value.get("kept_labels", [1])
    return (
        set(value) - {"kept_labels"} == {*thresholds, *counts}
        and all(_is_fraction(value[key]) for key in thresholds)
        and all(_is_integer(value[key]) and value[key] >= 1 for key in counts)
        and isinstance(labels, list)
        and bool(labels)
        and all(_is_integer(label) and label >= 1 for label in labels)
        and len(set(labels)) == len(labels)
    )


def _is_deploy_settings(value: Any) -> bool:
    return _is_postprocess_settings(value) and "kept_labels" in value


def _is_flag_list(value: Any) -> bool:
    return isinstance(value, list) and all(map(_is_string, value)) and len(set(value)) == len(value)


def _is_number_list(value: Any) -> bool:
    return isinstance(value, list) and all(_is_number(item) for item in value)


def _is_reading(value: Any) -> bool:
    return (_is_integer(value) and value > 0) or value == UNREADABLE


def _is_count(value: Any) -> bool:
    return _is_integer(value) and value >= 0


def _is_clock(value: Any, unit: str, extra: tuple[str, ...] = ()) -> bool:
    """One clock's record: governor, maximum, counts per clock seen, and the sample tallies."""
    keys = ("governor", f"max_{unit}", f"cur_{unit}_counts", "samples", "unreadable_samples", "at_max_samples", *extra)
    if not isinstance(value, dict) or set(value) != set(keys):
        return False
    counts = value[f"cur_{unit}_counts"]
    return (
        _is_string(value["governor"])
        and _is_reading(value[f"max_{unit}"])
        and all(_is_reading(value[key]) for key in extra)
        and isinstance(counts, dict)
        and all(key.isdigit() and _is_integer(count) and count > 0 for key, count in counts.items())
        and _is_count(value["samples"])
        and _is_count(value["unreadable_samples"])
        and value["samples"] == sum(counts.values()) + value["unreadable_samples"]
        and (_is_count(value["at_max_samples"]) or value["at_max_samples"] == UNREADABLE)
    )


def _is_cpu_clocks(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(_is_clock(record, "khz") for record in value.values())


# The keys each kind of log carries in ``results``, and the test each value must pass.
RESULT_CHECKS: dict[str, dict[str, Callable[[Any], bool]]] = {
    "eval": {
        "ap": _is_fraction,
        "ap50": _is_fraction,
        "detections": lambda value: _is_integer(value) and value >= 0,
        "runtime": _is_string,
    },
    "benchmark": {
        "runtime": _is_string,
        "engine": _is_latency,
        "end_to_end": _is_latency,
        "repeats": lambda value: _is_integer(value) and value >= 1,
        "engine_p50_ms_per_repeat": _is_number_list,
        "end_to_end_p50_ms_per_repeat": _is_number_list,
        "peak_process_rss_mib": _is_number,
        # Both come from tegrastats, so a host without it records them as UNREADABLE.
        "system_ram_rise_mib": lambda value: _is_number(value) or value == UNREADABLE,
        "avg_power_w": lambda value: _is_number(value) or value == UNREADABLE,
        "power_samples": lambda value: _is_integer(value) and value >= 0,
        "power_mode": _is_string,
        "cpu_load_1min_before_timing": _is_number,
        "gpu_busy_pct_before_timing": lambda value: _is_number(value) or value == UNREADABLE,
        # Read from sysfs during timing; a host without a file records UNREADABLE for it.
        "gpu_clock": lambda value: _is_clock(value, "hz", ("min_hz",)),
        "cpu_clocks": _is_cpu_clocks,
    },
    "build": {
        "engine_bytes": lambda value: _is_integer(value) and value > 0,
        "build_seconds": _is_number,
        "int8_path": _is_string,
        "entropy_failure": _is_string,
        "calibration_manifest_sha256": _is_string,
    },
    "export": {
        "max_abs_difference": _is_number,
        "tolerance": _is_number,
        "opset": _is_integer,
        "input_shape": lambda value: isinstance(value, list) and bool(value) and all(map(_is_integer, value)),
    },
    "gpu_check": {
        "cuda_available": _is_bool,
        "arch_list": lambda value: isinstance(value, list) and all(map(_is_string, value)),
        "device_compute_capability": _is_string,
        "matmul_max_abs_difference": lambda value: _is_number(value) or _is_string(value),
        "matmul_tolerance": _is_number,
        "matmul_matches": _is_bool,
        "matmul_error": _is_string,
        "nms_comparison": lambda value: isinstance(value, (dict, str)),
        "nms_matches": _is_bool,
        "nms_error": _is_string,
        "gpu_torch_usable": _is_bool,
        "torch_warning": _is_string,
        "cudnn_allow_tf32": _is_bool,
        "matmul_allow_tf32": _is_bool,
    },
}
# Keys a kind may carry beyond its required ones, with the test each value must pass when present. Logs written
# from 0.1.1 on carry them; the committed logs predate them and stay valid without them.
OPTIONAL_CHECKS: dict[str, dict[str, Callable[[Any], bool]]] = {
    "eval": {
        # The sha256 of the TensorRT engine file that ran, or "-" for fp32, which runs no engine file.
        "engine_sha256": _is_string,
        # torch.get_num_threads() as the run began: PyTorch's intra-op CPU threads, which preprocessing and
        # postprocessing use at every precision, and FP32 inference uses on the CPU.
        "torch_num_threads": lambda value: _is_integer(value) and value >= 1,
    },
    "benchmark": {
        "engine_sha256": _is_string,
        "torch_num_threads": lambda value: _is_integer(value) and value >= 1,
        # One summary per stage of the timed iteration, each over every timed iteration of every repeat.
        "stages": _is_stages,
        # The postprocess's score and NMS IoU thresholds, per-class top-k, detection cap and class count.
        "postprocess_settings": _is_postprocess_settings,
    },
    "build": {
        # The sha256 of the engine file the build wrote.
        "engine_sha256": _is_string,
        # The sha256 of the INT8 calibration cache this build wrote, or "-" when it wrote none.
        "calibration_cache_sha256": _is_string,
        # The precision flags set on the builder config, read back from it after configuring, by BuilderFlag name.
        "builder_flags": lambda value: _is_flag_list(value),
    },
}
# The deployment-style benchmark: the benchmark's results, from a postprocess that keeps only some classes. It is new
# in 0.1.1, so the results the main benchmark carries as optional are required of it.
DEPLOY_BENCHMARK_KIND = "benchmark_deploy"
RESULT_CHECKS[DEPLOY_BENCHMARK_KIND] = {
    **RESULT_CHECKS["benchmark"],
    **OPTIONAL_CHECKS["benchmark"],
    "postprocess_settings": _is_deploy_settings,
}
BENCHMARK_KINDS = ("benchmark", DEPLOY_BENCHMARK_KIND)
# Logs written before 0.1.1 carry an inferred_slice, which gpu_check no longer records: no slice can be inferred
# for a device that PyTorch itself says its build excludes. It is allowed unchecked.
OPTIONAL_RESULTS: dict[str, frozenset[str]] = {
    kind: frozenset(OPTIONAL_CHECKS.get(kind, {})) | ({"inferred_slice"} if kind == "gpu_check" else set())
    for kind in RESULT_CHECKS
}


# A build log's sentinels: "-" for a field that does not apply to its precision, and NO_ENTROPY_FAILURE for an INT8
# build whose entropy calibration succeeded. Each INT8 path has exactly one valid entropy_failure form.
NOT_APPLICABLE = "-"
NO_ENTROPY_FAILURE = ""
INT8_ENTROPY_PATH = "IInt8EntropyCalibrator2"
# The precision that runs in PyTorch, with no engine file: the PyTorch baseline.
PYTORCH_PRECISION = "fp32"
# The TensorRT engine built in FP32 throughout: the precision baseline, against which FP16 and INT8 differ in
# reduced-precision flags. "fp32" stays PyTorch's key, so the committed logs keep their meaning.
TENSORRT_FP32_PRECISION = "fp32-trt"
# The TensorRT precision flags a build controls, in the order a build log lists them.
PRECISION_FLAG_NAMES = ("FP16", "INT8", "TF32", "BF16", "FP8", "INT4", "FP4")
# The precision flags each engine precision has set, as its build log reads them back from the config. INT8 has FP16
# too, so a layer TensorRT does not run in INT8 runs in FP16, not FP32. TF32 is TensorRT's default and stays on for
# FP16 and INT8; TensorRT FP32 has none, so it is FP32 throughout.
BUILDER_FLAG_NAMES: dict[str, list[str]] = {
    TENSORRT_FP32_PRECISION: [],
    "fp16": ["FP16", "TF32"],
    "int8": ["FP16", "INT8", "TF32"],
}


def _identity_problems(kind: str, precision: Any, results: dict) -> list[str]:
    """Return the problems with the engine and calibration cache sha256s, where a log records them."""
    problems = []
    engine = results.get("engine_sha256")
    if isinstance(engine, str):
        if kind in ("eval", *BENCHMARK_KINDS) and precision == PYTORCH_PRECISION:
            if engine != NOT_APPLICABLE:
                problems.append(f"bad result: {kind} engine_sha256 must be - for {PYTORCH_PRECISION}")
        elif not SHA256_PATTERN.fullmatch(engine):
            problems.append(f"bad result: {kind} engine_sha256 is not a sha256")
    cache = results.get("calibration_cache_sha256")
    if kind == "build" and isinstance(cache, str) and cache != NOT_APPLICABLE:
        if precision != "int8" or results.get("int8_path") != INT8_ENTROPY_PATH:
            problems.append("bad result: build calibration_cache_sha256 must be - without the int8 entropy path")
        elif not SHA256_PATTERN.fullmatch(cache):
            problems.append("bad result: build calibration_cache_sha256 is not a sha256")
    return problems


def _cross_field_problems(kind: str, precision: Any, results: dict) -> list[str]:
    """Return the problems that involve more than one value of a log's results."""
    problems = _identity_problems(kind, precision, results)
    if kind == "eval" and _is_number(results.get("ap")) and _is_number(results.get("ap50")):
        if results["ap"] > results["ap50"]:
            problems.append("bad result: eval ap is above ap50")
    if kind in BENCHMARK_KINDS and _is_integer(results.get("repeats")):
        for key in ("engine_p50_ms_per_repeat", "end_to_end_p50_ms_per_repeat"):
            if isinstance(results.get(key), list) and len(results[key]) != results["repeats"]:
                problems.append(f"bad result: {kind} {key} does not hold one value per repeat")
    if kind == "build":
        manifest = results.get("calibration_manifest_sha256")
        route = results.get("int8_path")
        failure = results.get("entropy_failure")
        if precision == "int8":
            if isinstance(manifest, str) and not SHA256_PATTERN.fullmatch(manifest):
                problems.append("bad result: build calibration_manifest_sha256 is not a sha256")
            if route in ("", NOT_APPLICABLE):
                problems.append("bad result: build int8_path is empty for int8")
            elif route == INT8_ENTROPY_PATH and failure not in (None, NO_ENTROPY_FAILURE):
                problems.append("bad result: build entropy_failure must be empty when the entropy path was taken")
            elif route != INT8_ENTROPY_PATH and failure in (NO_ENTROPY_FAILURE, NOT_APPLICABLE):
                problems.append("bad result: build entropy_failure must name the failure when the fallback was taken")
        else:
            if manifest not in (None, NOT_APPLICABLE):
                problems.append("bad result: build calibration_manifest_sha256 must be - without int8")
            if route not in (None, NOT_APPLICABLE):
                problems.append("bad result: build int8_path must be - without int8")
            if failure not in (None, NOT_APPLICABLE):
                problems.append("bad result: build entropy_failure must be - without int8")
        flags = results.get("builder_flags")
        # No committed log predates the TensorRT FP32 build, so its log must say what it was built with.
        if precision == TENSORRT_FP32_PRECISION:
            problems += [
                f"missing result: build {key} for {precision}"
                for key in ("builder_flags", "engine_sha256")
                if key not in results
            ]
        if isinstance(flags, list) and precision in BUILDER_FLAG_NAMES and flags != BUILDER_FLAG_NAMES[precision]:
            problems.append(f"bad result: build builder_flags must be {BUILDER_FLAG_NAMES[precision]} for {precision}")
    return problems


# The precisions each kind of log may name. gpu_check and export name what the committed logs name.
MEASURED_PRECISIONS = (PYTORCH_PRECISION, TENSORRT_FP32_PRECISION, "fp16", "int8")
KIND_PRECISIONS: dict[str, tuple[str, ...]] = {
    "build": (TENSORRT_FP32_PRECISION, "fp16", "int8"),
    "eval": MEASURED_PRECISIONS,
    "benchmark": MEASURED_PRECISIONS,
    DEPLOY_BENCHMARK_KIND: MEASURED_PRECISIONS,
    "gpu_check": (PYTORCH_PRECISION,),
    "export": ("onnx",),
}
# The slack allowed between the sum of the stage means and the end-to-end mean, relative to the latter: each
# iteration's stages add up to its end-to-end time exactly, so only rounding separates them.
STAGE_SUM_RELATIVE_TOLERANCE = 1e-6
MS_PER_S = 1e3


def _stage_problems(kind: str, results: dict) -> list[str]:
    """The stages must hold the engine summary itself, and their means must add up to the end-to-end mean."""
    stages, engine, end_to_end = results.get("stages"), results.get("engine"), results.get("end_to_end")
    if not (_is_stages(stages) and _is_latency(engine) and _is_latency(end_to_end)):
        return []
    problems = []
    if stages["engine"] != engine:
        problems.append(f"bad result: {kind} stages engine differs from the engine summary")
    means = [MS_PER_S / stage["fps"] for stage in stages.values() if stage["fps"] > 0]
    total = MS_PER_S / end_to_end["fps"] if end_to_end["fps"] > 0 else None
    if total is None or len(means) != len(stages) or abs(sum(means) - total) > STAGE_SUM_RELATIVE_TOLERANCE * total:
        problems.append(f"bad result: {kind} stage means do not add up to the end-to-end mean")
    return problems


def _deploy_problems(kind: str, results: dict) -> list[str]:
    """The main benchmark keeps every class; the deployment-style one keeps the deploy labels at its threshold."""
    settings = results.get("postprocess_settings")
    if not isinstance(settings, dict):
        return []
    if kind == "benchmark" and "kept_labels" in settings:
        return ["bad result: benchmark postprocess_settings keeps only some labels"]
    # A deploy log without kept_labels is already refused by its settings check.
    if kind != DEPLOY_BENCHMARK_KIND or "kept_labels" not in settings:
        return []
    # Imported here: postprocess needs torch, and this module must import without it, so that an installed copy
    # refuses with one line (PRD R-13). Where torch is missing, the check is a problem, not an ImportError.
    try:
        from peddet_orin.postprocess import DEPLOY_LABELS, DEPLOY_SCORE_THRESHOLD
    except ImportError:
        return ["bad result: benchmark_deploy settings cannot be checked without torch"]

    if settings["kept_labels"] != list(DEPLOY_LABELS) or settings.get("score_threshold") != DEPLOY_SCORE_THRESHOLD:
        return [
            f"bad result: {kind} postprocess_settings must keep labels {list(DEPLOY_LABELS)} at score threshold "
            f"{DEPLOY_SCORE_THRESHOLD}"
        ]
    return []


def _results_problems(kind: Any, precision: Any, results: Any) -> list[str]:
    """Return the problems in ``results`` for a log's kind: missing, unexpected or bad values."""
    if kind not in RESULT_CHECKS:
        return [f"unknown kind: {kind}"]
    precision_problems = [] if precision in KIND_PRECISIONS[kind] else [f"bad precision for {kind}: {precision}"]
    if not isinstance(results, dict):
        return precision_problems
    checks = RESULT_CHECKS[kind]
    problems = [f"missing result: {kind} {key}" for key in checks if key not in results]
    allowed = set(checks) | OPTIONAL_RESULTS.get(kind, frozenset())
    problems += [f"unexpected result: {kind} {key}" for key in sorted(map(str, set(results) - allowed))]
    checks = {**checks, **OPTIONAL_CHECKS.get(kind, {})}
    problems += [
        f"bad result: {kind} {key}" for key, check in checks.items() if key in results and not check(results[key])
    ]
    if kind in BENCHMARK_KINDS:
        problems += _stage_problems(kind, results) + _deploy_problems(kind, results)
    return precision_problems + problems + _cross_field_problems(kind, precision, results)


def validate_log(record: Any) -> list[str]:
    """Return every schema problem found in a log; an empty list means it is valid."""
    if not isinstance(record, dict):
        return ["log is not an object"]
    problems = []
    for field, expected in REQUIRED_FIELDS.items():
        if field not in record:
            problems.append(f"missing field: {field}")
        elif not isinstance(record[field], expected) or (field in INTEGER_FIELDS and isinstance(record[field], bool)):
            problems.append(f"wrong type for field: {field}")
    version = record.get("schema_version")
    if version != SCHEMA_VERSION or isinstance(version, bool):
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    if isinstance(record.get("commit"), str) and not COMMIT_PATTERN.fullmatch(record["commit"]):
        problems.append("commit is not a 40-character hex sha")
    environment = record.get("environment")
    if isinstance(environment, dict):
        problems += [f"missing fact: {name}" for name in REQUIRED_ENVIRONMENT if name not in environment]
        problems += [
            f"fact is not a string: {name}"
            for name in (*REQUIRED_ENVIRONMENT, *DECODE_ENVIRONMENT, *RUNTIME_ENVIRONMENT)
            if name in environment and not isinstance(environment[name], str)
        ]
    times = {}
    for field in ("start_utc", "end_utc"):
        try:
            times[field] = datetime.strptime(str(record.get(field)), TIMESTAMP_FORMAT)
        except ValueError:
            problems.append(f"bad timestamp: {field}")
    if len(times) == 2 and times["start_utc"] > times["end_utc"]:
        problems.append("start_utc is after end_utc")
    if "kind" in record:
        problems += _results_problems(record["kind"], record.get("precision"), record.get("results"))
    for text in _strings_in(record):
        problems += [f"forbidden path: {path}" for path in forbidden_paths(text)]
    return problems
