# SPDX-License-Identifier: Apache-2.0
"""CHANGELOG.md has a dated, non-empty entry with a release link for every version."""

import hashlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HEADING = re.compile(r"^## \[(?P<version>\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$", re.M)
# A release link: "[x.y.z]: <repository URL>" followed by "releases", "tag" and "vx.y.z", slash-separated.
LINK = re.compile(r"^\[(?P<version>\d+\.\d+\.\d+)\]: \S+[/]releases[/]tag[/]v(?P<tag>\d+\.\d+\.\d+)$", re.M)
PYPROJECT = re.compile(r'^version = "(?P<version>[^"]+)"$', re.M)


def changelog_problems(changelog: str, version: str) -> list[str]:
    """Return what is missing: an entry for the current version, a body per entry, a link per entry."""
    problems = []
    headings = list(HEADING.finditer(changelog))
    versions = [m.group("version") for m in headings]
    links = {m.group("version"): m.group("tag") for m in LINK.finditer(changelog)}
    if version not in versions:
        problems.append(f"no entry for {version}")
    for index, match in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(changelog)
        body = changelog[match.end() : end]
        if not re.search(r"^- \S", body, re.M):
            problems.append(f"empty entry: {match.group('version')}")
        if links.get(match.group("version")) != match.group("version"):
            problems.append(f"no release link: {match.group('version')}")
    if len(set(versions)) != len(versions):
        problems.append("a version has two entries")
    return problems


def current() -> tuple[str, str]:
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
    version = PYPROJECT.search((REPO_ROOT / "pyproject.toml").read_text()).group("version")
    return changelog, version


def test_every_version_has_a_changelog_entry():
    assert changelog_problems(*current()) == []


def test_a_version_without_an_entry_is_detected():
    changelog, _ = current()
    assert changelog_problems(changelog, "9.9.9") == ["no entry for 9.9.9"]


def test_an_entry_without_a_release_link_is_detected():
    changelog, version = current()
    planted = LINK.sub("", changelog)
    assert f"no release link: {version}" in changelog_problems(planted, version)


def test_an_empty_entry_is_detected():
    changelog, version = current()
    heading = HEADING.search(changelog)
    link = LINK.search(changelog).group()
    planted = changelog[: heading.end()] + "\n\n" + link + "\n"
    assert f"empty entry: {version}" in changelog_problems(planted, version)


# The bytes of the 0.1.0 section at 8f3a50bd73887a0207bcde81bbbd4ee1df352202, the commit tag v0.1.0 names: from its
# heading up to the next heading or the release links. A released section is never edited afterwards.
RELEASED_0_1_0_SHA256 = "17f73f526145aa4a8334f78e238220480e28c576a28468d3377e4f369629e04f"
SECTION_END = re.compile(r"^(## \[|\[\d+\.\d+\.\d+\]: )", re.M)


def section(changelog: str, version: str) -> str:
    """Return one version's section, from its heading up to the next heading or the release links."""
    start = changelog.index(f"## [{version}]")
    return changelog[start : SECTION_END.search(changelog, start + 1).start()]


def test_the_0_1_0_section_equals_the_one_at_the_tag():
    released = section((REPO_ROOT / "CHANGELOG.md").read_text(), "0.1.0")
    assert hashlib.sha256(released.encode()).hexdigest() == RELEASED_0_1_0_SHA256
