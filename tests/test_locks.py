# SPDX-License-Identifier: Apache-2.0
"""Each hashed lock holds exactly its pin file's pins, each with at least one sha256, so an install from the lock
with --require-hashes is the pinned set and nothing else. scripts/lock_hashes.py writes the locks."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Each lock and the pin file it is made from.
LOCKS = {
    REPO_ROOT / "requirements" / "ci-cpu.lock": REPO_ROOT / "requirements" / "ci-cpu.txt",
    REPO_ROOT / "constraints" / "jetson-orin.lock": REPO_ROOT / "constraints" / "jetson-orin.txt",
}
AUDIT_LOCK = REPO_ROOT / "requirements" / "pip-audit.lock"
PIN = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^;\s]+)\s*(?:;\s*(?P<marker>[^\\]+?))?\s*$")
HASH = re.compile(r"--hash=sha256:[0-9a-f]{64}")
INDEX = re.compile(r"^--extra-index-url\s+\S+$")


def normalised(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirements(text: str) -> list[str]:
    """The requirement lines of a requirements file, continuation lines joined and comments dropped."""
    joined, lines = "", []
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if line.endswith("\\"):
            joined += line[:-1] + " "
            continue
        joined += line
        if joined.strip():
            lines.append(" ".join(joined.split()))
        joined = ""
    return lines


def pinned(text: str) -> dict[tuple[str, str | None], str]:
    """``{(name, marker): version}`` for each pin, with any --hash options removed."""
    found = {}
    for line in requirements(text):
        bare = HASH.sub("", line).strip()
        match = PIN.match(bare)
        if match:
            marker = " ".join(match.group("marker").split()) if match.group("marker") else None
            found[(normalised(match.group("name")), marker)] = match.group("version")
    return found


def lock_problems(lock_text: str, source_text: str) -> list[str]:
    """Each way a lock differs from its pin file: a pin added, missing or at another version, a pin without a
    hash, or an index the pin file names that the lock does not."""
    lock, source = pinned(lock_text), pinned(source_text)
    problems = [
        f"{name}: {version} in the lock, not pinned" for (name, _), version in lock.items() if (name, _) not in source
    ]
    problems += [
        f"{name}: pinned {version}, lock {lock.get((name, marker), 'missing')}"
        for (name, marker), version in source.items()
        if lock.get((name, marker)) != version
    ]
    for line in requirements(lock_text):
        if PIN.match(HASH.sub("", line).strip()) and not HASH.search(line):
            problems.append(f"no hash: {line}")
    indexes = {line for line in requirements(source_text) if INDEX.match(line)}
    problems += [f"index missing: {line}" for line in sorted(indexes - set(requirements(lock_text)))]
    return problems


def test_each_lock_holds_exactly_its_pin_files_pins_with_hashes():
    for lock, source in LOCKS.items():
        assert lock_problems(lock.read_text(), source.read_text()) == [], lock.name
        assert len(pinned(lock.read_text())) >= 25


def test_the_board_lock_names_the_index_of_its_cuda_builds():
    lines = requirements((REPO_ROOT / "constraints" / "jetson-orin.lock").read_text())
    assert "--extra-index-url https://download.pytorch.org/whl/cu130" in lines


def test_the_audit_lock_pins_pip_audit_and_hashes_every_package():
    text = AUDIT_LOCK.read_text()
    assert pinned(text)[("pip-audit", None)] == "2.10.1"
    unhashed = [line for line in requirements(text) if PIN.match(HASH.sub("", line).strip()) and not HASH.search(line)]
    assert unhashed == []


def test_a_lock_that_drifts_from_its_pin_file_is_detected():
    lock, source = next(iter(LOCKS.items()))
    lock_text, source_text = lock.read_text(), source.read_text()
    first = next(line for line in requirements(source_text) if PIN.match(line))
    name, version = PIN.match(first).group("name"), PIN.match(first).group("version")
    bumped = source_text.replace(f"{name}=={version}", f"{name}=={version}9", 1)
    assert lock_problems(lock_text, bumped) == [f"{normalised(name)}: pinned {version}9, lock {version}"]
    # The first pin's hash lines removed, and its line continuation with them.
    block = re.compile(rf"(?m)^({re.escape(name)}=={re.escape(version)}[^\n]*?) \\\n(?:    --hash=[^\n]*\n)+")
    hashless = block.sub(lambda match: match.group(1) + "\n", lock_text, count=1)
    assert hashless != lock_text
    assert lock_problems(hashless, source_text) == [f"no hash: {name}=={version}"]


def test_lock_lines_hash_every_file_of_the_pinned_release_from_each_index(monkeypatch):
    import json

    import lock_hashes

    pages = {
        "https://pypi.org/pypi/demo/1.0/json": json.dumps({"urls": [{"digests": {"sha256": "a" * 64}}]}).encode(),
        "https://index.example/simple/demo/": (
            b'<a href="demo-1.0-py3-none-any.whl#sha256=' + b"b" * 64 + b'">w</a>'
            b'<a href="demo-1.1-py3-none-any.whl#sha256=' + b"c" * 64 + b'">other release</a>'
        ),
    }
    monkeypatch.setattr(lock_hashes, "_get", lambda url: pages.get(url))
    lines = lock_hashes.lock_lines("--extra-index-url https://index.example/simple\ndemo==1.0\n")
    assert lines == [
        "--extra-index-url https://index.example/simple",
        "demo==1.0 \\",
        f"    --hash=sha256:{'a' * 64} \\",
        f"    --hash=sha256:{'b' * 64}",
    ]
    monkeypatch.setattr(lock_hashes, "_get", lambda url: None)
    try:
        lock_hashes.lock_lines("demo==1.0\n")
    except ValueError as error:
        assert "no file found to hash for demo==1.0" in str(error)
    else:
        raise AssertionError("a pin with no file to hash was locked")
