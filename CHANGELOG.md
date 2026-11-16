# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-11-16

### Added

- `CODE_OF_CONDUCT.md`, the Contributor Covenant 2.1 unchanged except for its contact line, which points at the
  repository's private vulnerability reporting.
- `runs-repeat/`: further full runs on a quieted host, each with its QUIET gate lines and, for the judged runs,
  memory, swap and temperature samples of every GPU job from the host's sampler script; the README gives each
  judged GPU job's lowest available memory and pages swapped out.
- README Provenance names the three host scripts that ran the judged runs, outside this repository, by sha256;
  `scripts/generate_table.py` refuses a sampler file that does not name the recorded sampler's sha256.
- Generated README text for the INT8 against FP16 comparison and for the check of `PRD.md` R-11 among the judged
  runs, so every number in it comes from the logs.

### Changed

- The published numbers changed. The results table is the median of the judged quieted runs at commit `025171d`;
  0.1.0's table came from one run at `17a1ee2` on a host that was not quieted. That run stays in `runs/` and is
  shown apart, with its deviation from the median, and is not judged.
- No file under `src/` changes.

### Security

- `requirements/ci-cpu.txt` pins pillow 13.0.0, pytest 9.0.3 and filelock 4.2.10 on every Python, the first
  releases with the fixes pip-audit lists against pillow 12.3.0, pytest 7.4.4 and filelock 4.1.0, and pins pygments,
  which pytest 9 needs. `constraints/jetson-orin.txt` pins pillow 13.0.0 too.
- The judged runs used Pillow 12.3.0. The pins name 13.0.0, the first release with the fixes pip-audit lists, and
  the numbers were not re-measured with it; JPEG decoding by Pillow is inside the end-to-end timings, not the
  engine timings.

### Removed

- Python 3.10. No release of Pillow or filelock with the fixes pip-audit lists installs on it, so
  `requires-python` is `>=3.11,<3.13` and CI runs on Python 3.11 and 3.12.

### Fixed

- `scripts/fetch_coco.py` keeps a COCO archive whose MD5 does not match and names it in the error. 0.1.0 deleted
  it, so 0.1.0's Security note, which says a file that fails its check is kept, was wrong for archives. A second
  run downloads the archive again and replaces the kept one only once the new download is complete.
- The README defines FPS as 1 / mean latency over serial batch-1 iterations, says where `nvpmodel` lives, and
  marks INT8's effect on postprocess time as a hypothesis.
- `CONTRIBUTING.md` gives the Jetson suite's time as about fifteen minutes.
- `pyproject.toml` requires pillow 10.3 or later, the first release with the fix for CVE-2024-28219.

## [0.1.0] - 2026-11-14

### Added

- Person detection with torchvision's SSDLite320 MobileNetV3-Large and its COCO weights, at FP32 in PyTorch and at
  FP16 and INT8 in TensorRT, on a Jetson Orin Nano Super.
- ONNX export, TensorRT engine builds with entropy-calibrated INT8, COCO val2017 person evaluation and batch-1
  benchmarks, each writing a JSON run log under `runs/`.
- A README results table generated from those logs, with a test that fails if it drifts.
- Each benchmark log records the governor and maximum of the GPU's devfreq clock and of each CPU frequency
  policy, and their clocks sampled every 200 ms during timing, read-only.
- `constraints/jetson-orin.txt`, which pins the Jetson venv, and `requirements/ci-cpu.txt`, the CPU-only set for CI.
- CI on ubuntu-latest for Python 3.10, 3.11 and 3.12, running ruff and the CPU tests.
- GPU and TensorRT tests, which skip where there is no CUDA device or no TensorRT.
- Tests for the repository's rules: the README layout tree, the CI contract, unique test names, the commit rule
  copied in `AGENTS.md`, version agreement, and strict skips in CI.
- `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff` and issue and pull request templates.
- A README layout tree generated from `git ls-files` by `scripts/generate_tree.py`, with a test that fails if it
  drifts.
- Tests that every version has a changelog entry, every test a document cites is collected, every PRD requirement
  names a CI test, and every cited commit id is in the history.
- A planted-defect registry, `tests/planted_defects.py`, naming the test that must catch each defect, and
  `scripts/check_planted_defects.py`, which plants each one.
- GitHub Actions pinned by commit SHA, and `scripts/ci_local.sh`, which runs the CI matrix locally.
- `scripts/compare_logs.py`, which checks a re-measure against the committed logs within the tolerances of
  `PRD.md` R-11.
- Benchmarks that time three repeats of their loop and log each repeat's median; the table notes give their range.
- `--data-dir`, `--engine`, `--onnx`, `--output`, `--calibration-dir` and `--repeats` options, `python -m peddet_orin`,
  `peddet-orin-<command>` console scripts and `peddet_orin.__version__`.
- `python -m peddet_orin.gpu_check`, which decides by running whether PyTorch's GPU build is usable and records
  PyTorch's own warning about this GPU, and the `DEBUG=1` switch for diagnostic logging.
- `scripts/fetch_coco.py train`, which fetches all of train2017 for experiments; the published results do not need it.
- Known-answer tests for the AP scoring, the latency percentiles, FPS, power and RAM figures, the calibration
  selection and the ONNX graph, and INT8 engine tests on both build routes for the Jetson.
- `pip-audit` in CI, a full lock of the CPU set for Python 3.10, 3.11 and 3.12, a 30-minute job timeout, and
  cancellation of superseded runs.

### Security

- Download COCO over HTTPS only, and keep a file whose hash does not match instead of deleting it.
- Load the pretrained weights as tensors only, after their full sha256 matches a pinned value.
- Pin pillow 12.3.0, which includes the fix for CVE-2024-28219.
- Replace any absolute path in a run log with `<path>` before writing it.

### Notes

- The FP32 row runs PyTorch with TF32 off for cuDNN and matmul. PyTorch's published wheels declare this GPU
  unsupported; the warning is recorded and shown, not filtered.
- The package runs from a source checkout, or the one `PDORIN_ROOT` names, on Python 3.10 to 3.12.

[0.1.1]: https://github.com/ahmedhassan-eng/peddet-orin/releases/tag/v0.1.1
[0.1.0]: https://github.com/ahmedhassan-eng/peddet-orin/releases/tag/v0.1.0
