# SPDX-License-Identifier: Apache-2.0
"""The repository root: PDORIN_ROOT, else the source checkout; else resolve_root raises a one-line RuntimeError,
and importing the package prints that line and exits 2."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import peddet_orin
from peddet_orin import runlog

PACKAGE_DIR = Path(runlog.__file__).resolve().parent


def fake_checkout(directory: Path, name: str = "peddet-orin", version: str = "9.8.7") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pyproject.toml").write_text(
        f'[build-system]\nrequires = ["setuptools"]\n\n[project]\nname = "{name}"\nversion = "{version}"\n'
    )
    return directory


def installed_copy(tmp_path: Path) -> Path:
    """Copy the package into a site directory, as a non-editable install would place it."""
    site = tmp_path / "interpreter" / "lib" / "site-packages"
    shutil.copytree(PACKAGE_DIR, site / "peddet_orin", ignore=shutil.ignore_patterns("__pycache__"))
    return site


def run_installed(site: Path, code: str, root: Path | None) -> subprocess.CompletedProcess:
    """Run code in an isolated interpreter that sees only the copied package and the standard library."""
    environment = {"PATH": "", "CUDA_VISIBLE_DEVICES": ""}
    if root is not None:
        environment["PDORIN_ROOT"] = str(root)
    script = f"import sys; sys.path.insert(0, {str(site)!r}); {code}"
    return subprocess.run(
        [sys.executable, "-I", "-S", "-c", script], env=environment, capture_output=True, text=True, timeout=60
    )


def test_pdorin_root_naming_a_checkout_is_used(tmp_path):
    root = fake_checkout(tmp_path / "clone")
    assert runlog.resolve_root({"PDORIN_ROOT": str(root)}) == root.resolve()


def test_pdorin_root_without_a_pyproject_is_refused(tmp_path):
    with pytest.raises(RuntimeError) as raised:
        runlog.resolve_root({"PDORIN_ROOT": str(tmp_path)})
    message = str(raised.value)
    assert "\n" not in message and "source checkout" in message and "PDORIN_ROOT" in message


def test_pdorin_root_with_another_projects_pyproject_is_refused(tmp_path):
    root = fake_checkout(tmp_path / "other", name="other-project")
    with pytest.raises(RuntimeError, match="PDORIN_ROOT"):
        runlog.resolve_root({"PDORIN_ROOT": str(root)})


def test_unset_pdorin_root_uses_the_checkout_above_the_package():
    assert runlog.resolve_root({}) == PACKAGE_DIR.parents[1]
    assert runlog.REPO_ROOT == PACKAGE_DIR.parents[1] and runlog.RUNS_DIR == runlog.REPO_ROOT / "runs"


def test_unset_pdorin_root_outside_a_checkout_is_refused(tmp_path):
    module_file = tmp_path / "lib" / "site-packages" / "peddet_orin" / "runlog.py"
    with pytest.raises(RuntimeError) as raised:
        runlog.resolve_root({}, module_file)
    message = str(raised.value)
    assert "\n" not in message and "source checkout" in message and "PDORIN_ROOT" in message


def test_project_field_reads_only_the_project_table(tmp_path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[tool.x]\nname = "wrong"\n\n[project]\nname = "peddet-orin"  # note\nversion = "1.2.3"\n')
    assert runlog.project_field(pyproject, "name") == "peddet-orin"
    assert runlog.project_field(pyproject, "version") == "1.2.3"
    assert runlog.project_field(tmp_path / "missing.toml", "name") is None


def test_an_installed_copy_without_pdorin_root_refuses_to_import(tmp_path):
    done = run_installed(installed_copy(tmp_path), "import peddet_orin.runlog", root=None)
    assert done.returncode == 2
    assert done.stderr == f"peddet-orin: error: {runlog.NO_CHECKOUT_MESSAGE}\n"


def test_an_installed_copy_with_pdorin_root_writes_under_it(tmp_path):
    root = fake_checkout(tmp_path / "clone")
    code = "import peddet_orin, peddet_orin.runlog as r; print(r.RUNS_DIR); print(peddet_orin.__version__)"
    done = run_installed(installed_copy(tmp_path), code, root=root)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [str(root.resolve() / "runs"), "9.8.7"]


def test_an_installed_copy_with_a_bad_pdorin_root_refuses_to_import(tmp_path):
    done = run_installed(installed_copy(tmp_path), "import peddet_orin.runlog", root=tmp_path / "empty")
    assert done.returncode == 2
    assert done.stderr == f"peddet-orin: error: {runlog.BAD_OVERRIDE_MESSAGE}\n"


def test_a_command_with_a_bad_pdorin_root_prints_one_line_and_no_traceback(tmp_path):
    environment = {
        "PYTHONPATH": str(PACKAGE_DIR.parent),
        "CUDA_VISIBLE_DEVICES": "",
        "PATH": "",
        "PDORIN_ROOT": str(tmp_path / "empty"),
    }
    done = subprocess.run(
        [sys.executable, "-m", "peddet_orin.export"], capture_output=True, text=True, timeout=120, env=environment
    )
    assert done.returncode == 2 and "Traceback" not in done.stderr
    assert done.stderr.count("\n") == 1 and "PDORIN_ROOT" in done.stderr


def test_package_version_falls_back_to_the_checkout_pyproject(monkeypatch, tmp_path):
    def not_installed(name):
        raise peddet_orin.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(peddet_orin.metadata, "version", not_installed)
    monkeypatch.setenv("PDORIN_ROOT", str(fake_checkout(tmp_path / "clone", version="4.5.6")))
    assert peddet_orin._read_version() == "4.5.6"


def test_python_m_peddet_orin_prints_the_version_and_the_subcommands():
    source = str(PACKAGE_DIR.parent)
    done = subprocess.run(
        [sys.executable, "-m", "peddet_orin"],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PYTHONPATH": source, "CUDA_VISIBLE_DEVICES": "", "PATH": ""},
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines() == [
        f"peddet-orin {peddet_orin.__version__}",
        "subcommands:",
        *(
            f"  python -m peddet_orin.{name}"
            for name in ("benchmark", "engine_build", "evaluate", "export", "gpu_check")
        ),
    ]
