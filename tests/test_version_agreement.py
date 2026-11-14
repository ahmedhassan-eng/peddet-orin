# SPDX-License-Identifier: Apache-2.0
"""pyproject.toml, CHANGELOG.md, CITATION.cff and peddet_orin.__version__ name the same version."""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = re.compile(r'^version = "(?P<version>[^"]+)"$', re.M)
CHANGELOG = re.compile(r"^## \[(?P<version>\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$", re.M)
CITATION = re.compile(r"^version: (?P<version>\S+)$", re.M)


def versions(pyproject: str, changelog: str, citation: str) -> dict[str, str | None]:
    """Return each file's version; the changelog's is its newest released heading."""
    found = {}
    for name, pattern, text in (("pyproject", PYPROJECT, pyproject), ("changelog", CHANGELOG, changelog),
                                ("citation", CITATION, citation)):
        match = pattern.search(text)
        found[name] = match.group("version") if match else None
    return found


def current() -> tuple[str, str, str]:
    return tuple((REPO_ROOT / name).read_text() for name in ("pyproject.toml", "CHANGELOG.md", "CITATION.cff"))


def test_versions_agree():
    found = versions(*current())
    assert None not in found.values() and len(set(found.values())) == 1, found


def test_package_version_agrees_with_pyproject():
    import peddet_orin

    assert peddet_orin.__version__ == versions(*current())["pyproject"]


def test_a_bumped_pyproject_alone_is_detected():
    pyproject, changelog, citation = current()
    assert 'version = "0.1.1"' in pyproject
    found = versions(pyproject.replace('version = "0.1.1"', 'version = "0.1.2"'), changelog, citation)
    assert len(set(found.values())) == 2
