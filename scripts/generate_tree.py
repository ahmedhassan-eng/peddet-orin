# SPDX-License-Identifier: Apache-2.0
"""Write the README "Repository layout" tree from the files git tracks.

Run with --write to update README.md in place; with no flag it prints the tree.
"""

import argparse
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
BEGIN_MARKER = "<!-- tree:start -->"
END_MARKER = "<!-- tree:end -->"
ROOT_NAME = "peddet-orin/"
# Directories shown as one line, without their files.
COLLAPSED = (
    "runs/",
    "runs-attempts/",
    "runs-repeat/",
    "runs-superseded/",
    "src/peddet_orin/",
    "tests/",
    ".github/ISSUE_TEMPLATE/",
)
# A short note after a path; every key must be a tracked file or a shown directory.
DESCRIPTIONS = {
    ".github/ISSUE_TEMPLATE/": "bug report, feature request and the private security link",
    ".github/workflows/ci.yml": "pip-audit, ruff and the CPU tests on Python 3.11 and 3.12",
    "constraints/jetson-orin.txt": "every pin for the board venv",
    "constraints/jetson-orin.lock": "the same pins with the sha256 of every file",
    "manifests/calibration_train2017.sha256": "the 500 calibration images and their sha256 hashes",
    "requirements/ci-cpu.txt": "the CPU-only set for CI",
    "requirements/ci-cpu.lock": "the same pins with the sha256 of every file",
    "requirements/pip-audit.txt": "the pip-audit release CI audits with",
    "requirements/pip-audit.lock": "pip-audit and its dependencies, with hashes",
    "runs/": "the earlier run's JSON logs, one per measurement, shown apart from the judged runs",
    "runs-attempts/": "the measurement attempts that are not judged runs, each with where it stopped",
    "runs-repeat/": "the judged runs the results table is built from, with their QUIET lines and samples",
    "runs-superseded/": "earlier judged runs, replaced by those under runs-repeat/ and kept with their files",
    "scripts/check_planted_defects.py": "plants each registered defect and checks that its test fails",
    "scripts/ci_local.sh": "runs the CI matrix locally, one venv per Python version",
    "scripts/compare_logs.py": "checks a re-measure against the judged median within the PRD tolerances",
    "scripts/host/": (
        "generic QUIET gate, sampler and root setup, written after the judged runs; not the scripts that ran them"
    ),
    "scripts/host/host_setup.py": "quiets a host for a window as root, and restores it, from a units file",
    "scripts/host/quiet.py": "the QUIET gate and the host sampler",
    "scripts/fetch_coco.py": "fetches COCO 2017 over HTTPS from COCO's S3 bucket, path-style, into data/",
    "scripts/generate_table.py": "writes the README's generated blocks from runs/, runs-repeat/ and runs-superseded/",
    "scripts/generate_tree.py": "writes this tree from git ls-files",
    "scripts/gpu_bounds.py": "measures on the board what the GPU tests' bounds are set from",
    "scripts/lock_hashes.py": "writes a hashed .lock beside a pin file",
    "scripts/recompute_ap.py": (
        "recomputes an eval log's AP from its detections file; logs written from the 0.1.1 release commit on, "
        "the committed ones carry none"
    ),
    "scripts/recompute_stats.py": (
        "recomputes a benchmark log's figures from its raw side file; logs written from the 0.1.1 release commit on, "
        "the committed ones carry none"
    ),
    "src/peddet_orin/": "export, engine builds, evaluation, benchmarks and run logs",
    "tests/": "CPU tests, GPU and TensorRT tests, and the repository-rule tests",
    "AGENTS.md": "rules for changes, with the commit-message rule",
    "PRD.md": "requirements, their acceptance checks and the tests that cover them",
    "QUEUE.md": "open and not-shipped work",
}
NOTE_COLUMN = 37


def tracked_files(root: Path = REPO_ROOT) -> list[str]:
    # Without its own .git, git would answer for whatever repository encloses root, or fail.
    if not (root / ".git").exists():
        raise RuntimeError(f"{root.name} has no .git: the tree is generated from a git checkout")
    done = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True)
    return done.stdout.splitlines()


def shown_paths(files: list[str]) -> set[str]:
    """Return every path the tree shows: files outside collapsed directories, and their directories."""
    shown = set()
    for name in files:
        collapsed = next((c for c in COLLAPSED if name.startswith(c)), None)
        if collapsed:
            name = collapsed
        parts = name.rstrip("/").split("/")
        for depth in range(1, len(parts)):
            shown.add("/".join(parts[:depth]) + "/")
        shown.add(name)
    return shown


def _children(shown: set[str], parent: str) -> list[str]:
    """Return the direct children of a directory: directories first, then files, each sorted."""
    depth = parent.count("/")
    kids = [p for p in shown if p.startswith(parent) and p != parent and p.rstrip("/").count("/") == depth]
    return sorted(kids, key=lambda p: (not p.endswith("/"), p.lower()))


def build_tree(files: list[str]) -> str:
    """Render the tree as the text inside the README's fenced block."""
    shown = shown_paths(files)
    lines = [ROOT_NAME]

    def walk(parent: str, guides: str) -> None:
        kids = _children(shown, parent)
        for index, path in enumerate(kids):
            last = index == len(kids) - 1
            name = path[len(parent) :]
            entry = f"{guides}{'└── ' if last else '├── '}{name}"
            note = DESCRIPTIONS.get(path)
            if note:
                entry = entry.ljust(max(NOTE_COLUMN, len(entry) + 2)) + note
            lines.append(entry)
            if path.endswith("/") and path not in COLLAPSED:
                walk(path, guides + ("    " if last else "│   "))

    walk("", "")
    return "\n".join(lines)


def unknown_descriptions(files: list[str]) -> list[str]:
    """Return description keys that the tree does not show, so a stale note cannot linger."""
    shown = shown_paths(files)
    return sorted(key for key in DESCRIPTIONS if key not in shown)


def readme_block(text: str) -> str:
    """Return the text inside the fenced block between the README's tree markers."""
    before, separator, rest = text.partition(BEGIN_MARKER)
    block, closing, _ = rest.partition(END_MARKER)
    if not separator or not closing:
        raise ValueError("README has no tree markers")
    fence = block.split("```")
    if len(fence) != 3:
        raise ValueError("the tree block must hold exactly one fenced code block")
    return fence[1].removeprefix("text").strip("\n")


def write_readme(path: Path = README, files: list[str] | None = None) -> None:
    text = path.read_text()
    before, separator, rest = text.partition(BEGIN_MARKER)
    _, closing, after = rest.partition(END_MARKER)
    if not separator or not closing:
        raise ValueError("README has no tree markers")
    tree = build_tree(tracked_files() if files is None else files)
    path.write_text(f"{before}{BEGIN_MARKER}\n```text\n{tree}\n```\n{END_MARKER}{after}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="update the README in place")
    args = parser.parse_args()
    if args.write:
        write_readme()
    else:
        print(build_tree(tracked_files()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
