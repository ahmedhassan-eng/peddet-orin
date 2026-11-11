"""The CI workflow runs exactly what the README says it runs, on the platforms it names."""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
README = REPO_ROOT / "README.md"
BEGIN = "<!-- ci:begin -->"
END = "<!-- ci:end -->"
RUN_KEY = re.compile(r"^(?P<indent>\s*)(?:- )?run:\s*(?P<value>.*)$")
MATRIX = re.compile(r"^\s*python-version:\s*\[(?P<versions>[^\]]*)\]\s*$")
RUNS_ON = re.compile(r"^\s*runs-on:\s*(?P<runner>\S+)\s*$")
PLATFORM = re.compile(r"^# CI: (?P<runner>\S+), Python (?P<versions>.+)$")


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def workflow_commands(text: str) -> list[str]:
    """Return every shell command in the workflow's ``run`` steps, in order."""
    lines = text.splitlines()
    commands = []
    index = 0
    while index < len(lines):
        match = RUN_KEY.match(lines[index])
        index += 1
        if not match:
            continue
        value = match.group("value").strip()
        if value not in ("|", ">"):
            commands.append(value)
            continue
        key_indent = len(match.group("indent"))
        while index < len(lines) and (not lines[index].strip() or indent_of(lines[index]) > key_indent):
            if lines[index].strip():
                commands.append(lines[index].strip())
            index += 1
    return commands


def workflow_platform(text: str) -> tuple[list[str], list[str]]:
    runners = [m.group("runner") for m in map(RUNS_ON.match, text.splitlines()) if m]
    versions = [
        version.strip().strip("\"'")
        for m in map(MATRIX.match, text.splitlines())
        if m
        for version in m.group("versions").split(",")
    ]
    return runners, versions


def readme_block(text: str) -> list[str]:
    start = text.index(BEGIN) + len(BEGIN)
    block = text[start : text.index(END)]
    fence = block.split("```")
    assert len(fence) == 3, "the CI block must hold exactly one fenced code block"
    return [line.strip() for line in fence[1].splitlines()[1:] if line.strip()]


def readme_commands(text: str) -> list[str]:
    return [line for line in readme_block(text) if not line.startswith("#")]


def readme_platform(text: str) -> tuple[list[str], list[str]]:
    matches = [m for m in map(PLATFORM.match, readme_block(text)) if m]
    assert len(matches) == 1, "the CI block must name its platform once"
    versions = re.split(r",\s*|\s+and\s+", matches[0].group("versions"))
    return [matches[0].group("runner")], versions


def test_workflow_runs_what_the_readme_says():
    assert workflow_commands(WORKFLOW.read_text()) == readme_commands(README.read_text())


def test_workflow_platform_is_what_the_readme_says():
    assert workflow_platform(WORKFLOW.read_text()) == readme_platform(README.read_text())


def test_a_command_missing_from_the_workflow_is_detected():
    text = WORKFLOW.read_text()
    assert "ruff check ." in text
    assert workflow_commands(text.replace("ruff check .", "true")) != readme_commands(README.read_text())


def test_a_dropped_python_version_is_detected():
    text = WORKFLOW.read_text()
    assert '"3.10", ' in text
    assert workflow_platform(text.replace('"3.10", ', "")) != readme_platform(README.read_text())


STRICT = "PDORIN_STRICT_SKIPS=1 "


def pytest_commands(commands: list[str]) -> list[str]:
    return [command for command in commands if "pytest" in command]


def test_ci_makes_an_unexpected_skip_fail_the_run():
    commands = pytest_commands(workflow_commands(WORKFLOW.read_text()))
    assert commands and all(command.startswith(STRICT) for command in commands)


def test_dropping_the_strict_skip_variable_is_detected():
    text = WORKFLOW.read_text()
    assert text.count(STRICT) == 1
    commands = pytest_commands(workflow_commands(text.replace(STRICT, "")))
    assert not all(command.startswith(STRICT) for command in commands)
