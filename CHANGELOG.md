# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- `CODE_OF_CONDUCT.md`, the Contributor Covenant 2.1 unchanged except for its contact line.
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

[0.1.0]: https://github.com/ahmedhassan-eng/peddet-orin/releases/tag/v0.1.0
