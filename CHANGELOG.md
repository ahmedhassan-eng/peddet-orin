# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-11-20

### Release commit

- The judged runs, run6, run7 and run8 under `runs-repeat/`, were measured at commit `8427133`, and the published
  numbers are those runs' figures, unchanged.
- The published logs were written by `8427133`'s code, so they lack the fields this release's code adds.
- The benchmark's timed loop and the evaluation's scoring are unchanged: by syntax tree, with docstrings stripped,
  they are identical to 8427133's.
- Since this release, the benchmark's timed loop runs inside try/finally blocks that release the samplers and the
  backend.
- From `8427133` to this release, the code changes file by file as follows.
- `src/peddet_orin/benchmark.py` records the input hashes, a raw side file and the pool count, refuses a pool under
  200, stops without a log when it reads no power samples, and releases its samplers and backend in finally
  blocks.
- `src/peddet_orin/evaluate.py` records the input hashes and writes a detections file, and its score line is in
  percentage points.
- `src/peddet_orin/export.py`'s ONNX check fails on a NaN or infinite output, and the export logs the ONNX sha256.
- `src/peddet_orin/engine_build.py` logs the ONNX sha256.
- `src/peddet_orin/runlog.py` adds optional result checks, the board model string, the raw side file, the reason an
  unread GPU-process count was not read, a dirty tree when git fails, a refusal of a benchmark log with 0 power
  samples, and refusals of non-positive latencies and FPS and of unordered percentiles.
- `src/peddet_orin/__init__.py`, `gpu_check.py`, `model.py` and `postprocess.py` change only in comments and
  docstrings.
- `src/peddet_orin/postprocess.py`'s changed comments include its SPDX licence line.
- Under `scripts/`, `check_planted_defects.py`, `ci_local.sh`, `compare_logs.py`, `fetch_coco.py`,
  `generate_table.py` and `generate_tree.py` change, and `gpu_bounds.py`, `recompute_ap.py`, `recompute_stats.py`,
  `host/quiet.py` and `host/host_setup.py` are new.
- Under `tests/`, 30 files change and 3 are new: `test_host_scripts.py`, `test_recompute_ap.py` and
  `test_recompute_stats.py`.
- The changed test files are `planted_defects.py`, `planted_results.txt`, `strict_skips.py`, `test_agents_rule.py`,
  `test_benchmark_measurement.py`, `test_checkout_kinds.py`, `test_ci_contract.py`, `test_commit_ids.py`,
  `test_compare_logs.py`, `test_engine_build.py`, `test_evaluate_scoring.py`, `test_fetch_coco.py`,
  `test_generate_table.py`, `test_gpu.py`, `test_internal_ids.py`, `test_locks.py`, `test_measure_cli_checks.py`,
  `test_negative_controls.py`, `test_network_guard.py`, `test_onnx_export.py`, `test_packaging_metadata.py`,
  `test_pins_match_logs.py`, `test_planted_defects.py`, `test_postprocess.py`, `test_readme_numbers.py`,
  `test_readme_table.py`, `test_readme_tree.py`, `test_requirements.py`, `test_run_logs.py` and
  `test_strict_skips.py`.
- `.github/workflows/ci.yml` adds the step that plants each CPU defect.
- `pyproject.toml` changes in its description, which names the Jetson Orin Nano Super, and in its licence,
  which is now `Apache-2.0 AND BSD-3-Clause`.
- `constraints/` and `requirements/` change only in comments, and no pin changes.

### Since the pre-release audit

The pre-release audit is the author's checks of the tree before publishing it.

- The benchmark stops the clock reader and the tegrastats sampler and releases the backend in finally blocks when
  timing raises, and writes no log then.
- The log check refuses a benchmark log with a zero or negative latency or FPS, or with percentiles out of the
  order p50 <= p90 <= p99.
- `scripts/recompute_stats.py` refuses, in one line, a raw file with no power readings.
- `scripts/host/quiet.py` fails its QUIET gate when `systemctl` exits non-zero, instead of passing on an empty unit
  list.
- The committed-log check covers the logs under `runs-attempts/`, and `scripts/compare_logs.py` never takes an
  attempt log as a re-measure.
- `scripts/recompute_stats.py` and `scripts/recompute_ap.py` refuse, with one line and exit 2, a log that names no
  raw side file or detections file, as every committed log.
- The tests that only a release commit can pass, the planted-output freshness test and the scan of every commit
  message for trailers, run only when `PDORIN_RELEASE_CHECK=1` is set; CI and contributors skip them.
- A run log records its tree as dirty when `git status` fails or times out.
- The log check refuses a benchmark log with 0 power samples.
- The benchmark stops with one error line and writes no log when `tegrastats` gives no power samples.
- The benchmark refuses to run when the image directory holds fewer than the 200 images it times.
- Each benchmark log records `pool_images`, the number of images it timed, and the log check requires it to be 200
  where it is present.
- Each eval and benchmark log records the sha256 of the annotations file and of the names of the images it read.
- Each export and build log records the sha256 of the ONNX file.
- Each eval writes its raw detections to a file under the git-ignored data directory and logs the file's name and
  sha256.
- evaluate prints person AP and AP50 in percentage points, as the README's table gives them.
- Each benchmark writes every timed iteration's stage and end-to-end times and its VDD_IN readings to a `.raw`
  side file beside its log, and the log records the side file's name and sha256.
- `scripts/recompute_stats.py` recomputes a benchmark log's figures from its raw side file.
- `scripts/recompute_ap.py` recomputes an eval log's AP from its detections file.
- Each log records the board's device-tree model string as `board_model`.
- When the count of other GPU processes cannot be read, the log records why: `UNREADABLE:import` when `pynvml`
  cannot be imported, else `UNREADABLE:` followed by the NVML error's class name.
- The log check treats the new input hashes, raw side file fields and pool count as optional, so the committed
  logs, which predate them, stay valid.
- `scripts/compare_logs.py` checks that each TensorRT eval and benchmark log names the engine that its precision's
  newest build log wrote, and fails one that names no engine.
- `scripts/compare_logs.py` fails a dirty build log on its INT8 path and builder flags lines.
- `scripts/compare_logs.py` fails a re-measure whose torch, TensorRT or CUDA version differs from the reference
  benchmark log's.
- `scripts/fetch_coco.py` checks each archive's sha256 as well as its MD5.
- `scripts/fetch_coco.py` refuses a download whose final URL, after redirects, is not HTTPS.
- `scripts/fetch_coco.py` refuses an annotations file that lists an empty, absolute, `..` or separator-carrying
  image file name.
- `scripts/fetch_coco.py` retries a download whose connection is cut mid-body.
- `scripts/fetch_coco.py` no longer has a `train` subcommand, as nothing here reads train2017.
- CI runs `scripts/check_planted_defects.py --cpu-only`, which passes only when every CPU defect is caught and only
  the GPU entries are not run.
- `scripts/host/quiet.py` holds a generic equivalent of the QUIET gate and host sampler, written after the judged
  runs; no judged run used it.
- `scripts/host/host_setup.py` quiets a host as root from a units file kept outside the repository, and restores it.
- `scripts/host/host_setup.py` opens its units file once and refuses it unless the open file and its directory
  are owned by root and writable by root only.
- `scripts/gpu_bounds.py` measures on the board the figures that the GPU tests' bounds are set from.
- `tests/test_gpu.py` holds PyTorch's CUDA FP32 output to a relative error of 1e-4 against the CPU, set from a
  board run; the bound was 1e-2.
- `tests/test_gpu.py` states that its CUDA FP32 bound cannot tell TF32 from FP32: one board run measured 1.13e-5
  with TF32 on and 1.35e-5 with it off.
- `tests/test_gpu.py` checks the TensorRT FP16 and INT8 engines' outputs against the TensorRT FP32 engine's by
  cosine similarity.
- `tests/test_gpu.py` checks that the TensorRT backend synchronizes its stream before it reads the outputs.
- Three new planted defects target the TensorRT path: a dropped stream synchronize, an FP32 engine built with the
  FP16 flag, and INT8 calibration on unnormalised input.
- The generated summary quotes the PyTorch GPU check's recorded warning that the wheel does not support this GPU.
- The generated summary names each TensorRT row's builder flags.
- The generated summary gives each precision's deployment-style end-to-end p50 beside the evaluation setting's.
- The generated summary gives the postprocess share of the end-to-end p50, rounded to nearest at one decimal.
- The results table labels its engine FPS column as the engine-only bound, 1 / engine mean latency.
- The deployment-style block gives each precision's end-to-end p50 range across repeats.
- The deployment-style block says so when the FP16 and INT8 ranges overlap.
- The deployment-style block's range line for the PyTorch baseline names PyTorch.
- The package's docstring names the Jetson Orin Nano Super.
- `src/peddet_orin/postprocess.py` carries the licence Apache-2.0 AND BSD-3-Clause, with torchvision's copyright
  line.
- `tests/planted_results.txt` holds the planted run of this release's files: the GPU entries on the board, then the
  CPU run.
- `runs-attempts/` holds the files from 6 runs and attempts that are not judged runs, each with the run script's
  lines to where it ended.

### Errata in the history

- Commit `9317a75`'s subject, "Release 0.1.1, with no change under src/", is untrue of 0.1.1 as published, which
  changes `src/`.
- Commit `148cc11`'s subject is 77 characters, over the 72 that `AGENTS.md` sets.
- Commit `7664f72` published 0.1.0 measurements made at `ac10495` on 2026-11-13; `8f3a50b` deleted them and
  published the measurements made at `17a1ee2` instead, and no document named the `ac10495` set until this note.

0.1.1 was not made public before this release. Its first judged runs, at `025171d` and now under
`runs-superseded/`, were made before the version was raised to 0.1.1, so their logs record package version 0.1.0;
they were replaced by run6, run7 and run8 at `8427133` before publication.

### Added

- `CODE_OF_CONDUCT.md`, the Contributor Covenant 2.1 unchanged except for its contact line, which sends reports to the
  maintainer on GitHub or to GitHub's Report abuse route.
- `runs-repeat/`: further full runs on a quieted host, each with its QUIET gate lines and, for the judged runs,
  memory, swap and temperature samples of every GPU job from the host's sampler script; the README gives each
  judged GPU job's lowest available memory and pages swapped out.
- `scripts/generate_table.py` refuses a sampler file that does not name the recorded sampler's sha256: a consistency check
  between the files, not a check of the sampler script.
- Generated README text for the INT8 against FP16 comparison and for the check of `PRD.md` R-11 among the judged
  runs, so every number in it comes from the logs.
- `engine_build --precision fp32-trt` builds a TensorRT FP32 engine, `artifacts/fp32-trt.engine`, the precision
  baseline: it sets neither `FP16` nor `INT8` and clears `TF32`, so it is FP32 throughout. Its build log records
  `builder_flags` `[]`, read back from the config. evaluate and
  benchmark accept `--precision fp32-trt`; `fp32` stays PyTorch, the PyTorch baseline, so the committed logs keep
  their meaning. `tests/test_gpu.py` gains a sixth GPU test, which builds the engine and checks it against the CPU; the file now
  holds eight GPU tests.
- The results table names the PyTorch FP32 row `GPU, PyTorch FP32 (PyTorch baseline)` and gains a
  `GPU, TensorRT FP32 (precision baseline)` row when the judged runs carry TensorRT FP32 logs. The judged runs carry
  them, so the row is shown;
  judged runs that carry it in some runs and not others, or only some of its logs, stop `scripts/generate_table.py`.
  `scripts/compare_logs.py` judges TensorRT FP32 against R-11's FP32 tolerances once the reference has it, AP within
  0.10 points, and reports it without judging until then.
- `python -m peddet_orin.benchmark --precision <p> --postprocess deploy` runs a deployment-style benchmark: the
  postprocess keeps the person class only, COCO category 1, at one fixed score threshold, 0.5, a fixed midpoint
  (by the author's record set before any deployment-style run and not tuned on the results), with no recall or AP reported at it; the rest
  is torchvision's settings. It
  writes a log of its own kind, `benchmark_deploy`, whose results are the benchmark's with `stages`,
  `postprocess_settings` (with `kept_labels`), `engine_sha256` and `torch_num_threads` required. `--postprocess`
  defaults to `standard`, the main benchmark, which is unchanged.
- The README gains a generated deployment-style block right after the results table: per precision, end-to-end
  p50/p90/p99, end-to-end FPS, power and energy per frame from the `benchmark_deploy` logs, and the postprocess
  settings they record. The judged runs carry such logs, so the block shows them; judged runs that carry them in some runs and not others stop
  `scripts/generate_table.py`. `scripts/compare_logs.py` never judges them, and prints one REPORTED line per
  deployment-style metric when the new logs carry it. The README's reproduce steps list its commands for all four
  precisions, with the TensorRT FP32 row's.
- `tests/planted_results.txt` holds the recorded lines of the GPU entries and the CPU run, assembled by the release helper (the author's script outside this repository that assembles the file), and
  `tests/test_negative_controls.py` fails unless it shows every registry entry caught and no other id.

### Changed

- `scripts/generate_table.py` no longer takes a QUIET gate's PASS on trust: a judged run is quieted only when it
  has a start gate and a gate before each benchmark it publishes, and when each passing line's own load is below its
  bound and its available memory at least its floor. A TensorRT precision whose logs name an engine in some logs
  and not in others stops the generator, as such a log cannot be tied to the others' engine.
- `scripts/compare_logs.py` judges each TensorRT build's `builder_flags` against the reference's, so an engine
  built another way is never judged against it; a reference that records no flags gives a line that is reported,
  not judged.
- The README says before the table which columns are what the board delivers, names the scripts as installed
  (`peddet-orin-engine-build`), says JetPack 6 is not supported and gives the `apt` line for the system packages. It states each benchmark's
  sample counts, image pool and seed, and that the precisions run in one fixed order every run.
- The published numbers changed; for example, the FP16 end-to-end p50 went from 75.1 ms in
  0.1.0 to 58.7 ms. The results table is the median of the judged quieted runs at commit `8427133`;
  0.1.0's table came from one run at `17a1ee2` on a host that was not quieted. That run stays in `runs/` and is
  shown apart, with its deviation from the median, and is not judged.
- `src/` changes in the ways below. Two change what a run measures: INT8 engines are built with the FP16 flag as
  well, and the benchmark's full image check moves out of its timed loop. The judged runs at `8427133` ran the code
  after both. Each run log also records the numpy, Pillow, pycocotools and libjpeg versions,
  since JPEG decoding is inside every end-to-end iteration; the judged runs' logs record them.
  And `peddet_orin.export` and `peddet_orin.engine_build` import ONNX Runtime where
  they use it, not at the top. On a host with no PCI bus id ONNX Runtime prints its own warning when it loads, so a
  command's bad-input error was two lines on stderr there, against PRD R-14; it is one line again. The judged runs
  ran the code after this change. The run log still loads
  ONNX Runtime to record its version, so a benchmark run of the released code loads it too, once its input checks
  have passed.
- `scripts/compare_logs.py` compares a re-measure with what the results table is built from, the median of the
  judged runs, not with the earlier run in `runs/`, and fails a re-measure that lacks a clean eval or benchmark log
  for any precision; it skipped those before. Its `--against` option is gone. Two planted defects cover the change.
  It leaves out the logs committed under `runs/`, `runs-repeat/` and `runs-superseded/`, so they never stand in for a missing or
  failed re-measure log, and it fails a dirty newest log, a log run with another runtime or other iteration counts,
  and new logs from more than one commit. The README's reproduce steps end with `python scripts/compare_logs.py runs`
  instead of `generate_table.py --write`, which would show a re-measure as the earlier run.
- `PRD.md` R-11 says what it checks: that the judged runs agree with their median within its tolerances, which is
  their consistency, not a re-measure from a fresh clone. The README lists the quieting, the power mode and the GPU
  governor as prerequisites for a comparable re-measure, describes how the judged runs applied them, and shows
  the earlier unquieted run's deviation as an uncontrolled before-and-after difference that also takes in code and library changes.
- The results table gives end-to-end energy per frame, each run's mean power over the timed loop divided by its
  end-to-end FPS, then the median. Power and energy sit with the end-to-end columns and engine FPS with the engine
  latency: power was sampled over the end-to-end loop, never at the engine rate.
- The README leads with Results and Limitations. The hardware and software follow, then a Judged runs section
  with the generated repeat block (gates, memory, swap, temperature and sampler records) and Provenance, then the
  Quickstart and how to reproduce the results.
- `AGENTS.md` states this repository's own short commit-message rule in place of a block copied from other
  projects; `tests/test_agents_rule.py` now checks that every commit keeps its checkable parts, no trailer and no
  subject ending with a period, and refuses a message that breaks either.
- `AGENTS.md` says the published numbers come from logs under `runs/`, `runs-repeat/` or `runs-superseded/`, except
  figures the README marks as the author's record; it named `runs/` only.
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
  other set when the key is present; the judged runs' build logs record it.
- After a failed entropy attempt, the INT8 build removes the calibration cache beside the output before the Q/DQ
  fallback builds, since the fallback engine is not built from it.
- Each benchmark log records `stages`, a summary for each stage of the timed iteration (decode, preprocess, engine
  and postprocess) in the shape of `engine` and `end_to_end`, over every timed iteration of every repeat, and
  `postprocess_settings`, the score threshold, NMS IoU threshold, top-k per class, detections per image and class
  count the postprocess ran with. The stage times come from the same `perf_counter` readings as before plus one
  between decoding and preprocessing, so the end-to-end time holds one more `perf_counter` call; `engine` and
  `end_to_end` are computed as before. Both keys are optional in the log check; the judged runs' benchmark logs record both.
- Each build log records `engine_sha256`, the sha256 of the engine file it wrote, and `calibration_cache_sha256`,
  the sha256 of the INT8 calibration cache the entropy route wrote, or `-` when the build wrote none (FP16, or the
  Q/DQ fallback). The entropy route removes a cache left by an earlier build before it calibrates, since the
  calibrator never reads one, so a cache present afterwards is that build's. Each eval and benchmark log records
  the `engine_sha256` of the engine it ran, or `-` for FP32, so a measurement can be traced to its build. The log
  check enforces each form when present; the judged runs' logs record them.
- Each eval and benchmark log records `torch_num_threads`, `torch.get_num_threads()` as the run begins: PyTorch's
  CPU threads run preprocessing and postprocessing at every precision, and FP32 inference on the CPU. The key is
  optional in the log check, a positive integer, and the judged runs' eval and benchmark logs record it.
- The benchmark runs the full `check_image` once on each pooled image, before warm-up, and refuses a bad one with
  one line; the timed iteration calls `preprocess(image, range_checked=True)`, which checks the decoded image's
  layout but not every pixel's range, so the full-frame `min()` and `max()` are out of the timed loop. evaluate and
  calibration keep the full check on every image. The judged runs ran this
  code, so their end-to-end times do not include the full check.
- The README's sentence on 0.1.1's `src/` changes no longer says none changes a computation: two do, and the judged runs
  ran both. It also says the timed loop holds one more `perf_counter` call, between decode and preprocess,
  and that the TensorRT FP32 baseline differs from FP16 and INT8 by their reduced-precision flags, not precision
  alone: tactics, TF32 in fallback layers and the Q/DQ fallback's graph can differ too.
- The log check refuses a precision a kind does not have: `build` takes `fp16`, `int8` and `fp32-trt`; `eval`,
  `benchmark` and `benchmark_deploy` take `fp32`, `fp16`, `int8` and `fp32-trt`; `gpu_check` takes `fp32` and
  `export` `onnx`, as the committed logs name them. It requires a benchmark's `stages` to hold the `engine` summary
  itself and their means to add up to the end-to-end mean within a relative 1e-6; it refuses `kept_labels` in a
  `benchmark` log, and requires a `benchmark_deploy` log to keep labels `[1]` at the score threshold
  `postprocess.DEPLOY_SCORE_THRESHOLD`, 0.5. The standard benchmark prints `<precision>: end-to-end p50 ...` as
  before; the deploy run prints `<precision> deploy: ...`.
- `scripts/generate_table.py` stops unless each deployment-style log ran with the runtime and iteration counts of
  the main benchmark of its precision, keeps labels `[1]` at the deploy score threshold, and has the main benchmark's
  NMS IoU threshold, top-k, per-image cap and class count. The deploy block adds the postprocess-stage p50 of the
  deploy run and of the main benchmark, and says the QUIET gates, sampler records and power-mode evidence are read
  for the main benchmark jobs. The all-or-none check of the TensorRT FP32 and deployment-style logs also applies to
  a `runs/` read by its newest logs.
- `scripts/compare_logs.py` fails new logs of any kind, build and deployment-style included, from more than one
  commit. A dirty newest deployment-style log, or one with another commit or setup than the main benchmark of its
  precision, prints a REPORTED line saying so.
- `scripts/fetch_coco.py` takes an archive as present only once its extraction finished, recorded in a stamp
  file beside it; one extracted image alone made an interrupted extraction look complete.
- Each run log also records the CUDA runtime and driver versions, as the runtime library the run loaded reports
  them, beside the TensorRT version it already recorded; `UNREADABLE` on a host with no CUDA runtime.
- A run log's `peddet_orin` version is the checkout's `pyproject.toml` version, the code that ran; it was the
  installed metadata's, which reads `UNREADABLE` when the package runs from `PYTHONPATH` and can be stale.
- README Limitations states what the measurements and checks do not cover: idle power inside energy per frame,
  no GPU-only kernel time, the deprecated INT8 calibrator, what the INT8 GPU tests check, ONNX parity on random
  input, MD5 for the COCO archives, and compare_logs' two-decimal print.

### Security

- `requirements/ci-cpu.txt` pins pillow 13.0.0, pytest 9.0.3 and filelock 4.2.10 on every Python, which by the
  author's pip-audit run when they were set are the first releases with fixes for the advisories it listed against
  pillow 12.3.0, pytest 7.4.4 and filelock 4.1.0, and pins pygments, which pytest 9 needs.
- `constraints/jetson-orin.txt` pins the same versions as `requirements/ci-cpu.txt` for every package in both:
  pillow 13.0.0, pytest 9.0.3, pygments 2.21.0, filelock 4.2.10, fsspec 2026.9.0, networkx 3.7 (the CI file pins 3.6.1 for Python 3.11), protobuf 7.36.2 and
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
- The judged runs used Pillow 13.0.0, the pinned release.

### Removed

- Python 3.10. By the author's pip-audit run, no release of Pillow or filelock with the fixes it listed installs
  on it, so
  `requires-python` is `>=3.11,<3.13` and CI runs on Python 3.11 and 3.12.

### Fixed

- `scripts/compare_logs.py` reads which logs are committed from git at HEAD. It had taken every log in `runs/`,
  `runs-repeat/` and `runs-superseded/` as committed, so a re-measure written into `runs/`, as the README's
  reproduce block does, was left out and every log read as missing. Outside a git checkout it stops with one line.
- `scripts/fetch_coco.py` downloads from `https://s3.amazonaws.com/images.cocodataset.org/`. The host name
  `images.cocodataset.org` serves the S3 certificate, which does not cover it, so HTTPS downloads by that name fail
  certificate verification (read 2026-11-19). The MD5 and sha256 checks are unchanged.
- The export check fails on a NaN or infinite output: `peddet_orin.export.largest_difference` returns
  infinity when either side is not finite, where `max()` had dropped a NaN, so a broken backend no longer passes.
  The GPU tests' relative-error check does the same. The judged runs ran the export before this change.
- `scripts/fetch_coco.py` keeps a COCO archive whose MD5 does not match and names it in the error. 0.1.0 deleted
  it, so 0.1.0's Security note, which says a file that fails its check is kept, was wrong for archives. A second
  run downloads the archive again and replaces the kept one only once the new download is complete.
- The README defines FPS as 1 / mean latency over serial batch-1 iterations, says where `nvpmodel` lives, and
  marks INT8's effect on postprocess time as a hypothesis.
- `CONTRIBUTING.md` says the Jetson suite takes longer, because it builds TensorRT engines.
- The pillow floor in `pyproject.toml` also covers CVE-2024-28219, fixed in pillow 10.3; the floor is 13.0.0, as above.
- `scripts/generate_table.py` refuses a benchmark log that times fewer than 3 repeats, as `PRD.md` R-12 requires
  of a published one; `--repeats` still accepts fewer for a quick run. Without `--write` it prints every generated
  block between its markers, not the results block alone.
- `scripts/compare_logs.py` fails an INT8 build on another calibration path than the reference's.
- `scripts/fetch_coco.py` removes the partial file when every download attempt fails.
- The README states the main benchmark's postprocess settings, which the end-to-end time includes, and the 3 s
  window the RAM rise is measured against; the builder flags sentence says the build logs record them from 0.1.1
  on. `constraints/jetson-orin.txt` names the hashed install line the README uses.
- `scripts/ci_local.sh` waits for its log's `tee` before it exits, so the log is never cut short.
- `tests/test_readme_numbers.py` ties the figures typed into the README's and CHANGELOG's prose to their sources: the
  repeat count, the baseline window, the torch wheel, the compute capability, the quieting figure, the postprocess
  settings and the FP16 change. A planted defect checks the first.
- `tests/test_internal_ids.py` also finds review-finding ids and scans every commit message.
- `tests/planted_results.txt` names the files it was made from as those of the commit that last changed it, and
  `tests/test_negative_controls.py` fails once any other file has changed since, so the output is never stale.

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
