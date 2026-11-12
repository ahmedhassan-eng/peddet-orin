"""pyproject.toml, requirements/ci-cpu.txt and constraints/jetson-orin.txt agree with each other and the code."""
import ast
import re
import sys
from importlib import metadata
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import Requirement

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: pytest's own dependency provides the same parser.
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = REPO_ROOT / "pyproject.toml"
CI_PINS = REPO_ROOT / "requirements" / "ci-cpu.txt"
JETSON_PINS = REPO_ROOT / "constraints" / "jetson-orin.txt"
PACKAGE = REPO_ROOT / "src" / "peddet_orin"
PIN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^;\s]+)\s*(?:;\s*(?P<marker>.+))?$")
# pillow 10.3.0 is the first release with the fix for CVE-2024-28219.
FIRST_FIXED_PILLOW = (10, 3, 0)


def project() -> dict:
    return tomllib.loads(PYPROJECT.read_text())["project"]


def pins(path: Path) -> dict[str, list[tuple[str, str | None]]]:
    """Return ``{normalised name: [(version, marker or None)]}`` for every ``name==version`` line."""
    found: dict[str, list[tuple[str, str | None]]] = {}
    for line in path.read_text().splitlines():
        match = PIN.match(line.strip())
        if match:
            name = re.sub(r"[-_.]+", "-", match.group("name")).lower()
            found.setdefault(name, []).append((match.group("version"), match.group("marker")))
    return found


def cli_modules(package: Path = PACKAGE) -> list[str]:
    """Return the package modules that define a top-level ``main``, which is how a module is a CLI."""
    found = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text())
        if any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body):
            found.append(path.stem)
    return found


def expected_scripts(modules: list[str]) -> dict[str, str]:
    return {f"peddet-orin-{module.replace('_', '-')}": f"peddet_orin.{module}:main" for module in modules}


def test_every_cli_module_has_a_console_script():
    modules = cli_modules()
    assert {"benchmark", "engine_build", "evaluate", "export", "gpu_check"} <= set(modules)
    assert project()["scripts"] == expected_scripts(modules)


def test_a_cli_module_without_a_console_script_is_detected(tmp_path):
    (tmp_path / "newtool.py").write_text("def main() -> int:\n    return 0\n")
    (tmp_path / "helper.py").write_text("def helper() -> int:\n    return 0\n")
    assert cli_modules(tmp_path) == ["newtool"]
    assert expected_scripts(cli_modules(tmp_path)) == {"peddet-orin-newtool": "peddet_orin.newtool:main"}


def test_requires_python_stops_before_the_first_python_without_numpy_1_wheels():
    # numpy<2 has no wheels for Python 3.13.
    assert "numpy<2" in project()["dependencies"]
    assert project()["requires-python"] == ">=3.10,<3.13"


def test_torch_and_torchvision_are_bounded_above_for_the_torchscript_exporter():
    dependencies = project()["dependencies"]
    assert "torch>=2.9,<2.15" in dependencies
    assert "torchvision>=0.24,<0.30" in dependencies


def test_ruff_is_pinned_alike_in_the_test_extra_ci_and_the_jetson_constraints():
    ci = pins(CI_PINS)["ruff"]
    assert len(ci) == 1 and ci[0][1] is None
    version = ci[0][0]
    assert f"ruff=={version}" in project()["optional-dependencies"]["test"]
    assert pins(JETSON_PINS)["ruff"] == [(version, None)]


def test_pillow_includes_the_cve_2024_28219_fix_and_agrees_across_pin_files():
    ci, orin = pins(CI_PINS)["pillow"], pins(JETSON_PINS)["pillow"]
    assert ci == orin and len(ci) == 1
    assert tuple(int(part) for part in ci[0][0].split(".")) >= FIRST_FIXED_PILLOW


def applicable_pins(path: Path, python_version: str) -> dict[str, str]:
    """Return ``{name: version}`` for the pins whose marker holds on the given Python version."""
    chosen: dict[str, str] = {}
    for name, entries in pins(path).items():
        for version, marker in entries:
            if marker is None or Marker(marker).evaluate({"python_version": python_version}):
                assert name not in chosen, f"{name} is pinned twice for Python {python_version}"
                chosen[name] = version
    return chosen


def test_every_ci_requirement_is_an_exact_pin_once_per_python():
    lines = [line.strip() for line in CI_PINS.read_text().splitlines()]
    requirements = [line for line in lines if line and not line.startswith(("#", "-"))]
    assert requirements and all(PIN.match(line) for line in requirements), requirements
    names = set(pins(CI_PINS))
    for python_version in ("3.10", "3.11", "3.12"):
        assert set(applicable_pins(CI_PINS, python_version)) <= names
    assert {"exceptiongroup", "tomli"} <= set(applicable_pins(CI_PINS, "3.10"))
    assert not {"exceptiongroup", "tomli"} & set(applicable_pins(CI_PINS, "3.12"))


def unpinned_dependencies(chosen: dict[str, str]) -> list[str]:
    """Return the dependencies of installed pinned packages that the lock leaves unpinned.

    Only a package installed at exactly its pinned version is checked, so in CI, which
    installs the lock, this covers every package; in another venv it covers the matching ones.
    """
    missing = set()
    for name, version in chosen.items():
        try:
            installed = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        if installed.version != version:
            continue
        for text in installed.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                if re.sub(r"[-_.]+", "-", requirement.name).lower() not in chosen:
                    missing.add(f"{requirement.name} (needed by {name})")
    return sorted(missing)


def test_the_ci_lock_pins_the_dependencies_of_what_it_installs():
    python_version = ".".join(map(str, sys.version_info[:2]))
    assert unpinned_dependencies(applicable_pins(CI_PINS, python_version)) == []


def test_an_unpinned_transitive_dependency_is_detected():
    chosen = applicable_pins(CI_PINS, ".".join(map(str, sys.version_info[:2])))
    pytest_version = metadata.version("pytest")
    planted = {name: version for name, version in chosen.items() if name != "pluggy"} | {"pytest": pytest_version}
    assert "pluggy (needed by pytest)" in unpinned_dependencies(planted)
