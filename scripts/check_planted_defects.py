"""Plant each registered defect in a fresh clone of HEAD and check that its named test catches it.

For every entry in tests/planted_defects.py: the test must pass in the clean clone and fail once the defect is
planted. Entries that need a GPU run only where CUDA is available. Prints one PLANT line per entry, then
"PLANTED DONE PASS" or "PLANTED DONE FAIL". Usage: python scripts/check_planted_defects.py
"""
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
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", node],
                          cwd=clone, capture_output=True, text=True, env=env, check=False)
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


def main() -> int:
    gpu = cuda_available()
    failures = 0
    (REPO_ROOT / ".cache").mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="planted-", dir=REPO_ROOT / ".cache"))
    try:
        clean = fresh_clone(work, "clean")
        for defect in DEFECTS:
            if defect.get("needs") == "gpu" and not gpu:
                print(f"PLANT {defect['id']} not-run (needs a GPU)")
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
            caught = before == 0 and after != 0
            failures += not caught
            print(f"PLANT {defect['id']} clean-rc={before} planted-rc={after} {'caught' if caught else 'FAIL'}")
            shutil.rmtree(clone)
    finally:
        shutil.rmtree(work)
    print(f"PLANTED DONE {'PASS' if failures == 0 else 'FAIL'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
