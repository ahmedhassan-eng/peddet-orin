# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-11

### Added

- Person detection with torchvision's SSDLite320 MobileNetV3-Large and its COCO weights, at FP32 in PyTorch and at
  FP16 and INT8 in TensorRT, on a Jetson Orin Nano Super.
- ONNX export, TensorRT engine builds with entropy-calibrated INT8, COCO val2017 person evaluation and batch-1
  benchmarks, each writing a JSON run log under `runs/`.
- A README results table generated from those logs, with a test that fails if it drifts.
- `constraints/jetson-orin.txt`, which pins the Jetson venv, and `requirements/ci-cpu.txt`, the CPU-only set for CI.
- CI on ubuntu-latest for Python 3.10, 3.11 and 3.12, running ruff and the CPU tests.
- GPU and TensorRT tests, which skip where there is no CUDA device or no TensorRT.
- Tests for the repository's rules: the README layout tree, the CI contract, unique test names, the commit rule
  copied in `AGENTS.md`, version agreement, and strict skips in CI.
- `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff` and issue and pull request templates.

[0.1.0]: https://github.com/ahmedhassan-eng/peddet-orin/releases/tag/v0.1.0
