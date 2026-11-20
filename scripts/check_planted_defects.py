# SPDX-License-Identifier: Apache-2.0
"""Plant each registered defect in a fresh clone of HEAD and check that its named test catches it.

For every entry in tests/planted_defects.py: the test must pass in the clean clone and fail once the defect is
planted, with pytest's "tests failed" status. Entries that need a GPU run only where CUDA is available. Prints one
PLANT line per entry, then "PLANTED DONE PASS" when every entry ran and was caught, "PLANTED DONE INCOMPLETE <n> not
run" when none failed but some needed a GPU it did not have, or "PLANTED DONE FAIL"; only PASS exits 0.
Usage: python scripts/check_planted_defects.py
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tests"))
from planted_defects import DEFECTS  # noqa: E402


def run_test(clone: Path, node: str) -> int:
    env = {key: value for key, value in os.environ.items() if key != "PDORIN_STRICT_SKIPS"}
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", node],
        cwd=clone,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return done.returncode


def fresh_clone(work: Path, name: str) -> Path:
    clone = work / name
    subprocess.run(["git", "clone", "-q", "--no-hardlinks", str(REPO_ROOT), str(clone)], check=True)
    return clone


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


# pytest's exit status when tests ran and some failed; a usage error, an unknown test or nothing collected is not
# a caught defect.
TESTS_FAILED = 1


def verdict(failures: int, not_run: int) -> str:
    """The last line: PASS only when every entry ran and was caught."""
    if failures:
        return "PLANTED DONE FAIL"
    return f"PLANTED DONE INCOMPLETE {not_run} not run" if not_run else "PLANTED DONE PASS"


def cpu_only_status(line: str, not_run: int, gpu_entries: int) -> int:
    """CI's status for a CPU-only run: 0 when no entry failed and exactly the GPU-only entries were not run."""
    return 0 if line == f"PLANTED DONE INCOMPLETE {gpu_entries} not run" and not_run == gpu_entries else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plant each registered defect and check that its test catches it.")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="run the CPU entries only, as CI does; pass when each is caught and only the GPU entries are not run",
    )
    args = parser.parse_args(argv)
    gpu = cuda_available() and not args.cpu_only
    failures = not_run = 0
    (REPO_ROOT / ".cache").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="planted-", dir=REPO_ROOT / ".cache"))
    try:
        clean = fresh_clone(work, "clean")
        for defect in DEFECTS:
            if defect.get("needs") == "gpu" and not gpu:
                print(f"PLANT {defect['id']} not-run (needs a GPU)")
                not_run += 1
                continue
            before = run_test(clean, defect["catches"])
            clone = fresh_clone(work, defect["id"])
            path = clone / defect["file"]
            text = path.read_text()
            if text.count(defect["old"]) != 1:
                print(f"PLANT {defect['id']} FAIL old text occurs {text.count(defect['old'])} times")
                failures += 1
                continue
            path.write_text(text.replace(defect["old"], defect["new"]))
            after = run_test(clone, defect["catches"])
            caught = before == 0 and after == TESTS_FAILED
            failures += not caught
            print(f"PLANT {defect['id']} clean-rc={before} planted-rc={after} {'caught' if caught else 'FAIL'}")
            shutil.rmtree(clone)
    finally:
        shutil.rmtree(work)
    line = verdict(failures, not_run)
    print(line)
    if args.cpu_only:
        return cpu_only_status(line, not_run, sum(defect.get("needs") == "gpu" for defect in DEFECTS))
    return 0 if line == "PLANTED DONE PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
