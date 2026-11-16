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
- `src/` changes in the ways below. Two change what a run measures: INT8 engines are built with the FP16 flag as
  well, and the benchmark's full image check moves out of its timed loop. The published numbers come from the
  code before both and were not re-measured. Each run log also records the numpy, Pillow, pycocotools and libjpeg versions,
  since JPEG decoding is inside every end-to-end iteration; the committed logs predate these facts and stay valid.
  And `peddet_orin.export` and `peddet_orin.engine_build` import ONNX Runtime where
  they use it, not at the top. On a host with no PCI bus id ONNX Runtime prints its own warning when it loads, so a
  command's bad-input error was two lines on stderr there, against PRD R-14; it is one line again. The judged runs
  ran the code before this change, with ONNX Runtime loaded in the measuring process; the peak process RSS in the
  table includes it and was not re-measured; the timings and accuracy do not depend on it. The run log still loads
  ONNX Runtime to record its version, so a benchmark run of the released code loads it too, once its input checks
  have passed.
- `scripts/compare_logs.py` compares a re-measure with what the results table is built from, the median of the
  judged runs, not with the earlier run in `runs/`, and fails a re-measure that lacks a clean eval or benchmark log
  for any precision; it skipped those before. Its `--against` option is gone. Two planted defects cover the change.
  It leaves out the logs committed under `runs/` and `runs-repeat/`, so they never stand in for a missing or
  failed re-measure log, and it fails a dirty newest log, a log run with another runtime or other iteration counts,
  and new logs from more than one commit. The README's reproduce steps end with `python scripts/compare_logs.py runs`
  instead of `generate_table.py --write`, which would show a re-measure as the earlier run.
- `PRD.md` R-11 says what it checks: that the judged runs agree with their median within its tolerances, which is
  their consistency, not a re-measure from a fresh clone. The README lists the quieting, the power mode and the GPU
  governor as prerequisites for a comparable re-measure, describes how the judged runs applied them, and states
  the earlier unquieted run's deviation as what the quieting is worth.
- The results table gives end-to-end energy per frame, each run's mean power over the timed loop divided by its
  end-to-end FPS, then the median. Power and energy sit with the end-to-end columns and engine FPS with the engine
  latency: power was sampled over the end-to-end loop, never at the engine rate.
- The README leads with Results and Limitations. The hardware and software follow, then a Judged runs section
  with the generated repeat block (gates, memory, swap, temperature and sampler records) and Provenance, then the
  Quickstart and how to reproduce the results.
- `AGENTS.md` states this repository's own short commit-message rule in place of a block copied from other
  projects; `tests/test_agents_rule.py` now checks that every commit keeps its checkable parts, no trailer and no
  subject ending with a period, and refuses a message that breaks either.
- `AGENTS.md` says the published numbers come from logs under `runs/` or `runs-repeat/`; it named `runs/` only.
- `tests/test_negative_controls.py` counts the in-file negative tests from a named list, one test per file,
  and fails when a named test does not exist; a test-name pattern counted them before.
- `tests/test_pins_match_logs.py` fails when a pin in `constraints/jetson-orin.txt` for torch, torchvision,
  onnx or onnxruntime differs from what any judged run log records.
- The command-line pieces that evaluate and benchmark share (data and engine paths, common options, the
  one-line usage error, and the backend for each precision) move from `evaluate.py` to
  `src/peddet_orin/cli.py`, so benchmark no longer imports them from evaluate.
- A bad input to `python -m peddet_orin.engine_build` prints `python -m peddet_orin.engine_build: error: ...`,
  naming the command as evaluate and benchmark do; it printed `error: ...` before.
- Diagnostic logging and profiling turn on with `PDORIN_DEBUG=1`; the generic `DEBUG` variable, which other
  tools also read, no longer turns them on. `tests/test_debug.py` checks both.
- A command run outside a checkout, or with a `PDORIN_ROOT` that holds no peddet-orin `pyproject.toml`, prints
  one `peddet-orin: error: ...` line and exits 2, as PRD R-13 says; it printed an 11-line traceback before. The root
  lookup moves to `src/peddet_orin/root.py`, which reads `pyproject.toml` with `tomllib`, and `import peddet_orin`
  still works outside a checkout.
- ruff's target is Python 3.11, the oldest supported version.
- The benchmark reads the power mode with `nvpmodel -q` after warm-up and before timing, beside the load
  average, so the log records the mode in force as timing starts; it was read after timing ended. The
  command-output helper in `runlog` is public as `command_output`.
- The Quickstart names the system packages it relies on and their versions on the measured board:
  `python3.12-venv`, `python3-libnvinfer`, `nvidia-l4t-nvpmodel`, `nvidia-l4t-tools` and `python3-numpy`.
- `PDORIN_REQUIRE_GPU=1` makes a GPU or TensorRT test that skips fail the run; the README's board test command
  sets it, so a missing CUDA device or TensorRT no longer passes quietly.
- `gpu_check` no longer records an `inferred_slice`: PyTorch's own warning, which the log keeps, says its build's
  sm_80 slice excludes this device, so no slice can be inferred from the build's list. A planted defect for its
  usable verdict covers `tests/test_gpu_check.py`, the last test file with no negative control; `tests/test_negative_controls.py` now fails if any test file has neither.
- evaluate and benchmark refuse `--precision fp32` with a `--device` that is not a torch device, or is CUDA on
  a host where PyTorch finds none, with one line and exit 2 (PRD R-14); it was a traceback.
- The TensorRT backend refuses an engine whose input or outputs are not FLOAT before it allocates a buffer, since
  its buffers hold float32; the engines `engine_build` writes are FLOAT at both ends.
- A build log's `entropy_failure` has one form per case, and the log check enforces it: `-` without INT8,
  empty when the INT8 entropy path was taken, and the failure text when the Q/DQ fallback was; the committed logs
  already follow it.
- The tree is formatted with `ruff format`, and CI and `scripts/ci_local.sh` run `ruff format --check .` after
  `ruff check .`.
- Both INT8 build routes, `IInt8EntropyCalibrator2` and the Q/DQ fallback, set TensorRT's `FP16` builder flag as
  well as `INT8`, so a layer TensorRT does not run in INT8 runs in FP16, not FP32. Each build sets its precision
  flags and clears the others among `FP16`, `INT8`, `TF32`, `BF16` and `FP8`; `TF32`, TensorRT's default, stays on
  for FP16 and INT8 as before. The build log records as `builder_flags` the flags read back from the config after
  configuring, `["FP16", "TF32"]` for FP16 and `["FP16", "INT8", "TF32"]` for INT8, and the log check refuses any
  other set when the key is present; the committed logs predate the key and stay valid.
- After a failed entropy attempt, the INT8 build removes the calibration cache beside the output before the Q/DQ
  fallback builds, since the fallback engine is not built from it.
- Each benchmark log records `stages`, a summary for each stage of the timed iteration (decode, preprocess, engine
  and postprocess) in the shape of `engine` and `end_to_end`, over every timed iteration of every repeat, and
  `postprocess_settings`, the score threshold, NMS IoU threshold, top-k per class, detections per image and class
  count the postprocess ran with. The stage times come from the same `perf_counter` readings as before plus one
  between decoding and preprocessing, so the end-to-end time holds one more `perf_counter` call; `engine` and
  `end_to_end` are computed as before. Both keys are optional in the log check, and the committed logs predate them.
- Each build log records `engine_sha256`, the sha256 of the engine file it wrote, and `calibration_cache_sha256`,
  the sha256 of the INT8 calibration cache the entropy route wrote, or `-` when the build wrote none (FP16, or the
  Q/DQ fallback). The entropy route removes a cache left by an earlier build before it calibrates, since the
  calibrator never reads one, so a cache present afterwards is that build's. Each eval and benchmark log records
  the `engine_sha256` of the engine it ran, or `-` for FP32, so a measurement can be traced to its build. The log
  check enforces each form when present; the committed logs predate the keys and stay valid.
- Each eval and benchmark log records `torch_num_threads`, `torch.get_num_threads()` as the run begins: PyTorch's
  CPU threads run preprocessing and postprocessing at every precision, and FP32 inference on the CPU. The key is
  optional in the log check, a positive integer, and the committed logs predate it.
- The benchmark runs the full `check_image` once on each pooled image, before warm-up, and refuses a bad one with
  one line; the timed iteration calls `preprocess(image, range_checked=True)`, which checks the decoded image's
  layout but not every pixel's range, so the full-frame `min()` and `max()` are out of the timed loop. evaluate and
  calibration keep the full check on every image. The judged runs ran the check inside each timed iteration, so
  their end-to-end times include it and were not re-measured.
- `engine_build --precision fp32-trt` builds a TensorRT FP32 engine, `artifacts/fp32-trt.engine`, the precision
  baseline: it sets neither `FP16` nor `INT8` and clears `TF32`, so it is FP32 throughout. Its build log records
  `builder_flags` `[]`, read back from the config. evaluate and
  benchmark accept `--precision fp32-trt`; `fp32` stays PyTorch, the PyTorch baseline, so the committed logs keep
  their meaning. `tests/test_gpu.py` gains a sixth GPU test, which builds the engine and checks it against the CPU.
- The results table names the PyTorch FP32 row `GPU, PyTorch FP32 (PyTorch baseline)` and gains a
  `GPU, TensorRT FP32 (precision baseline)` row when the judged runs carry TensorRT FP32 logs. The committed judged
  runs have none, so the row is left out and one generated note says it is measured from the next judged runs on;
  judged runs that carry it in some runs and not others, or only some of its logs, stop `scripts/generate_table.py`.
  `scripts/compare_logs.py` judges TensorRT FP32 against R-11's FP32 tolerances once the reference has it, AP within
  0.10 points, and reports it without judging until then. The README's sentence on 0.1.1's `src/` changes no longer
  says none changes a computation: two do, and neither was re-measured.
- `python -m peddet_orin.benchmark --precision <p> --postprocess deploy` runs a deployment-style benchmark: the
  postprocess keeps the person class only, COCO category 1, at one fixed score threshold, 0.5, a fixed midpoint
  chosen before any deployment-style run and not tuned on the results; the rest is torchvision's settings. It
  writes a log of its own kind, `benchmark_deploy`, whose results are the benchmark's with `stages`,
  `postprocess_settings` (with `kept_labels`), `engine_sha256` and `torch_num_threads` required. `--postprocess`
  defaults to `standard`, the main benchmark, which is unchanged.
- The README gains a generated deployment-style block right after the results table: per precision, end-to-end
  p50/p90/p99, end-to-end FPS, power and energy per frame from the `benchmark_deploy` logs, and the postprocess
  settings they record. The committed judged runs have no such logs, so the block says in one line that they are
  measured from the next judged runs on; judged runs that carry them in some runs and not others stop
  `scripts/generate_table.py`. `scripts/compare_logs.py` never judges them, and prints one REPORTED line per
  deployment-style metric when the new logs carry it. The README's reproduce steps run it for all four precisions.

### Security

- `requirements/ci-cpu.txt` pins pillow 13.0.0, pytest 9.0.3 and filelock 4.2.10 on every Python, the first
  releases with the fixes pip-audit lists against pillow 12.3.0, pytest 7.4.4 and filelock 4.1.0, and pins pygments,
  which pytest 9 needs.
- `constraints/jetson-orin.txt` pins the same versions as `requirements/ci-cpu.txt` for every package in both:
  pillow 13.0.0, pytest 9.0.3, pygments 2.21.0, filelock 4.2.10, fsspec 2026.9.0, networkx 3.7, protobuf 7.36.2 and
  setuptools 84.2.1; it also pins iniconfig, jinja2, markupsafe, packaging and pluggy, which only the CI file pinned.
  `tests/test_packaging_metadata.py` fails when a package pinned in both differs, or when the CI file pins one the
  board file does not; torch and torchvision are compared before the "+", since the board uses CUDA builds and CI
  CPU builds.
- CI audits both pin files with `pip-audit --strict`, so a package it cannot find fails the run. Before, torch and
  torchvision were skipped unaudited, since PyPI does not know their `+cpu` builds, and the board file was not
  audited. The audit now reads copies with the local suffix stripped, so torch 2.14.1 and torchvision 0.29.1 are
  audited as their public releases.
- CI, `scripts/ci_local.sh` and the README Quickstart install from hashed locks with `--require-hashes`:
  `requirements/ci-cpu.lock`, `constraints/jetson-orin.lock` and `requirements/pip-audit.lock`, so every file pip
  installs, from PyPI or the PyTorch index, must match a recorded sha256. The package itself installs with
  `--no-build-isolation`, so its build backend is the pinned setuptools. `scripts/lock_hashes.py` writes the two pin
  locks from their pin files without resolving anything, and `tests/test_locks.py` checks they agree.
- `pyproject.toml` declares `pillow>=13.0.0` and, for the tests, `pytest>=9.0.3`, so an install outside the pinned
  sets cannot pick the releases pip-audit lists; the floors were 10.3 and 7.4.
- CI also runs weekly and on demand, so a new advisory against an unchanged pin fails the audit; it runs on
  `ubuntu-24.04`, not the moving `ubuntu-latest`; checkout keeps no token in the clone; and
  `tests/test_ci_contract.py` refuses an `if`, `continue-on-error`, `shell`, `env` or `working-directory` key that
  could let a step pass without running what it lists.
- `scripts/fetch_coco.py` takes an archive as present only once its extraction finished, recorded in a stamp
  file beside it; one extracted image alone made an interrupted extraction look complete.
- Each run log also records the CUDA runtime and driver versions, as the runtime library the run loaded reports
  them, beside the TensorRT version it already recorded; `UNREADABLE` on a host with no CUDA runtime.
- README Limitations states what the measurements and checks do not cover: idle power inside energy per frame,
  no GPU-only kernel time, the deprecated INT8 calibrator, what the INT8 GPU tests check, ONNX parity on random
  input, MD5 for the COCO archives, and compare_logs' two-decimal print.
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
