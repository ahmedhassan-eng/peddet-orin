# SPDX-License-Identifier: Apache-2.0
"""Up-front checks and options of the evaluate and benchmark commands, without data or a GPU."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from peddet_orin import benchmark, cli, evaluate, runlog
from peddet_orin.model import load_detector
from peddet_orin.postprocess import Postprocessor

SRC_DIR = Path(evaluate.__file__).resolve().parents[1]
MODULES = {"evaluate": evaluate, "benchmark": benchmark}


def fails_with_one_line(capsys, main, arguments: list[str], usage_first: bool = False) -> str:
    """Run a main and return its stderr: exit status 2, no traceback, one error line.

    Argparse's own type errors print the usage before that line; the commands' checks print the line alone.
    """
    with pytest.raises(SystemExit) as stopped:
        main(arguments)
    assert stopped.value.code == 2
    error = capsys.readouterr().err
    assert "Traceback" not in error
    lines = error.strip().splitlines()
    assert lines[-1].startswith("python -m peddet_orin.") and ": error: " in lines[-1]
    if usage_first:
        assert lines[0].startswith("usage: ")
    else:
        assert len(lines) == 1
    return lines[-1]


def make_data_dir(root: Path, images: int = 1) -> Path:
    (root / "annotations").mkdir(parents=True)
    (root / "val2017").mkdir()
    record = {
        "images": [{"id": n, "file_name": f"{n:012d}.jpg", "width": 32, "height": 24} for n in range(1, images + 1)],
        "annotations": [],
        "categories": [{"id": 1, "name": "person"}],
    }
    (root / "annotations" / "instances_val2017.json").write_text(json.dumps(record))
    for n in range(1, images + 1):
        Image.new("RGB", (32, 24)).save(root / "val2017" / f"{n:012d}.jpg")
    return root


@pytest.fixture
def offline_run(monkeypatch):
    """Keep main from loading weights, asking NVML or writing a log."""
    monkeypatch.setattr(runlog, "other_gpu_process_count", lambda: 0)
    monkeypatch.setattr(runlog.RunLog, "_write", lambda self: None)
    for module in MODULES.values():
        monkeypatch.setattr(module, "load_detector", lambda pretrained: load_detector(pretrained=False))


@pytest.mark.parametrize("name", sorted(MODULES))
def test_measure_missing_images_name_the_fetch_command(name, tmp_path, capsys):
    error = fails_with_one_line(capsys, MODULES[name].main, ["--precision", "fp32", "--data-dir", str(tmp_path)])
    assert "python scripts/fetch_coco.py" in error


def test_evaluate_missing_annotations_name_the_fetch_command(tmp_path, capsys):
    (tmp_path / "val2017").mkdir()
    error = fails_with_one_line(capsys, evaluate.main, ["--precision", "fp32", "--data-dir", str(tmp_path)])
    assert "python scripts/fetch_coco.py annotations" in error


def test_evaluate_missing_image_dir_names_the_val_fetch(tmp_path, capsys):
    make_data_dir(tmp_path)
    (tmp_path / "val2017" / "000000000001.jpg").unlink()
    (tmp_path / "val2017").rmdir()
    error = fails_with_one_line(capsys, evaluate.main, ["--precision", "fp32", "--data-dir", str(tmp_path)])
    assert "python scripts/fetch_coco.py val" in error


@pytest.mark.parametrize("name", sorted(MODULES))
@pytest.mark.parametrize("device", ["cuda", "cuda:0", "nonsense"])
def test_measure_fp32_on_an_unusable_device_is_one_line(name, device, tmp_path, capsys, monkeypatch):
    make_data_dir(tmp_path)
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)
    arguments = ["--precision", "fp32", "--device", device, "--data-dir", str(tmp_path)]
    error = fails_with_one_line(capsys, MODULES[name].main, arguments)
    assert f"--device {device}" in error


@pytest.mark.parametrize("name", sorted(MODULES))
def test_measure_missing_engine_names_the_build_command(name, tmp_path, capsys):
    make_data_dir(tmp_path)
    arguments = ["--precision", "int8", "--data-dir", str(tmp_path), "--engine", str(tmp_path / "none.engine")]
    error = fails_with_one_line(capsys, MODULES[name].main, arguments)
    assert "python -m peddet_orin.engine_build --precision int8" in error


def test_benchmark_empty_image_pool_is_refused(tmp_path, capsys):
    (tmp_path / "val2017").mkdir()
    error = fails_with_one_line(capsys, benchmark.main, ["--precision", "fp32", "--data-dir", str(tmp_path)])
    assert "no .jpg images" in error


@pytest.mark.parametrize(
    "option, value", [("--iterations", "0"), ("--warmup", "-1"), ("--repeats", "0"), ("--iterations", "x")]
)
def test_benchmark_counts_are_range_checked(option, value, tmp_path, capsys):
    error = fails_with_one_line(capsys, benchmark.main, ["--precision", "fp32", option, value], usage_first=True)
    assert f"argument {option}" in error


def _unloadable_engine(path):
    raise AttributeError("'NoneType' object has no attribute 'create_execution_context'")


class _NoEngine:
    label = "tensorrt-gpu"
    engine = None

    def __init__(self, path) -> None:
        pass


@pytest.mark.parametrize("name", sorted(MODULES))
@pytest.mark.parametrize("backend", [_unloadable_engine, _NoEngine, lambda path: None])
def test_measure_corrupt_engine_is_refused(name, backend, tmp_path, capsys, monkeypatch, offline_run):
    make_data_dir(tmp_path)
    engine = tmp_path / "corrupt.engine"
    engine.write_bytes(b"not an engine")
    monkeypatch.setattr(cli, "TensorRtBackend", backend)
    arguments = ["--precision", "fp16", "--data-dir", str(tmp_path), "--engine", str(engine)]
    error = fails_with_one_line(capsys, MODULES[name].main, arguments)
    assert "could not load engine" in error
    assert "python -m peddet_orin.engine_build --precision fp16" in error


def test_evaluate_engine_option_reaches_the_backend(tmp_path, capsys, monkeypatch, offline_run):
    make_data_dir(tmp_path)
    engine = tmp_path / "custom.engine"
    engine.write_bytes(b"x")
    loaded = []
    monkeypatch.setattr(cli, "TensorRtBackend", lambda path: loaded.append(path))
    fails_with_one_line(
        capsys, evaluate.main, ["--precision", "fp16", "--data-dir", str(tmp_path), "--engine", str(engine)]
    )
    assert loaded == [engine]


class _QuietBackend:
    """Raw outputs that postprocess to no detections, shaped for SSDLite's anchors."""

    label = "fake-cpu"

    def __init__(self, detector) -> None:
        anchors = Postprocessor.from_torchvision(detector).anchors.shape[0]
        self.outputs = torch.zeros(1, anchors, 4), torch.zeros(1, anchors, 91)
        self.outputs[1][..., 0] = 20.0

    def __call__(self, batch):
        return self.outputs


def test_engine_sha256_hashes_the_engine_file_and_is_absent_for_fp32(tmp_path):
    engine = tmp_path / "fp16.engine"
    engine.write_bytes(b"engine bytes")
    assert cli.engine_sha256(engine) == hashlib.sha256(b"engine bytes").hexdigest()
    assert cli.engine_sha256(None) == "-"


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_evaluate_logs_the_engine_hash(precision, tmp_path, monkeypatch, offline_run):
    make_data_dir(tmp_path)
    engine = tmp_path / "custom.engine"
    engine.write_bytes(b"x")
    written = []
    monkeypatch.setattr(runlog.RunLog, "_write", lambda self: written.append(self))
    monkeypatch.setattr(evaluate, "load_backend", lambda *arguments: _QuietBackend(arguments[3]))
    arguments = ["--precision", precision, "--device", "cpu", "--data-dir", str(tmp_path), "--engine", str(engine)]
    assert evaluate.main(arguments) == 0
    expected = hashlib.sha256(b"x").hexdigest() if precision == "fp16" else "-"
    assert written[0].results["engine_sha256"] == expected


def test_measure_default_paths_are_under_the_repository_root():
    for module in MODULES.values():
        defaults = {action.dest: action.default for action in module.build_parser()._actions}
        assert defaults["data_dir"] == runlog.REPO_ROOT / "data"
        assert defaults["engine"] is None
    assert cli.check_engine("prog", "fp32", None) is None
    assert cli.default_engine("fp16") == runlog.REPO_ROOT / "artifacts" / "fp16.engine"


@pytest.mark.parametrize("name", sorted(MODULES))
def test_measure_options_have_prog_and_help_with_defaults(name):
    parser = MODULES[name].build_parser()
    assert parser.prog == f"python -m peddet_orin.{name}"
    formatter = parser.formatter_class(parser.prog)
    for action in parser._actions:
        assert action.help, action.dest
        if action.default not in (None, "==SUPPRESS=="):
            assert "(default: %(default)s)" in formatter._get_help_string(action), action.dest
    assert "(default: cuda)" in parser.format_help()


def test_benchmark_defaults_to_three_repeats():
    assert benchmark.build_parser().parse_args(["--precision", "fp16"]).repeats == 3


def test_measure_bad_input_exits_two_without_traceback_in_a_process(tmp_path):
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC_DIR), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run(
        [sys.executable, "-m", "peddet_orin.benchmark", "--precision", "fp16", "--data-dir", str(tmp_path)],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    assert done.returncode == 2
    assert "Traceback" not in done.stderr
    assert done.stderr.strip().count("\n") == 0
    assert "python scripts/fetch_coco.py val" in done.stderr


# The commands whose input checks must not load ONNX Runtime: it can print its own warnings to stderr on import
# (on a host with no PCI bus id, for example), and a bad input must stay one line on stderr (PRD R-14).
CLI_MODULES = ("peddet_orin.benchmark", "peddet_orin.evaluate", "peddet_orin.export", "peddet_orin.engine_build")


def loaded_after(code: str) -> set[str]:
    """Run ``code`` in a fresh interpreter and return the top-level module names it left loaded."""
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SRC_DIR), os.environ.get("PYTHONPATH", "")])}
    report = "\nimport sys\nprint(' '.join(sorted({name.split('.')[0] for name in sys.modules})))"
    done = subprocess.run(
        [sys.executable, "-c", code + report], capture_output=True, text=True, env=environment, timeout=300, check=True
    )
    return set(done.stdout.split())


def test_a_bad_input_command_never_loads_onnxruntime(tmp_path):
    code = (
        "import importlib\n"
        f"for name in {CLI_MODULES!r}:\n"
        "    importlib.import_module(name)\n"
        "from peddet_orin import benchmark\n"
        "try:\n"
        f"    benchmark.main(['--precision', 'fp16', '--data-dir', {str(tmp_path)!r}])\n"
        "except SystemExit:\n"
        "    pass\n"
    )
    assert "onnxruntime" not in loaded_after(code)


def test_a_module_that_loads_onnxruntime_is_detected():
    assert "onnxruntime" in loaded_after("import onnxruntime")
