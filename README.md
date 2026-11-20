# peddet-orin

<!-- summary:begin -->
Person detection with torchvision's SSDLite320 MobileNetV3-Large and its COCO weights on a Jetson Orin Nano Super, batch size 1; the median of the 3 judged runs at commit `8427133`:

| Device and precision | Engine p50 (ms) | End-to-end p50, evaluation setting (ms) | End-to-end p50, deploy-style (ms) | Person AP@[.5:.95], evaluation setting | Postprocess share of end-to-end p50, evaluation setting | From the logs |
|---|---|---|---|---|---|---|
| GPU, PyTorch FP32 (PyTorch baseline) | 30.3 | 88.2 | 41.1 | 28.01 | 54.1% | PyTorch warned at first CUDA use: "No published PyTorch CUDA builds for release 2.14.1+cu130 support this GPU." |
| GPU, TensorRT FP32 (precision baseline) | 5.8 | 60.9 | 16.4 | 28.01 | 74.8% | TensorRT builder flags none |
| GPU, TensorRT FP16 | 3.8 | 58.7 | 14.2 | 28.01 | 77.4% | TensorRT builder flags FP16, TF32 |
| GPU, TensorRT INT8 | 3.6 | 63.4 | 14.1 | 26.83 | 80.4% | TensorRT builder flags FP16, INT8, TF32 |

The millisecond and watt cells are rounded to one decimal; the ranges in the notes below are given to two, so a cell can read just outside the range of the runs it is the median of. The engine is at most 34.4% of a frame's end-to-end p50 at the evaluation setting in the median cells. The rest is image decoding, preprocessing and the postprocess on the CPU, at torchvision's evaluation setting (score threshold 0.001, 91 classes), so the end-to-end time is set by the CPU, not the engine; the CPU clocks were not locked, so the end-to-end figures are measurements under the `schedutil` governor. The deploy-style column is each precision's `benchmark_deploy` log, the same frames with the postprocess the deployment-style block gives, keeping labels 1 at score threshold 0.5; no accuracy is measured at that setting.
<!-- summary:end -->

The judged runs were measured at commit `8427133`, and this release changes code since then: under `src/peddet_orin/`, `__init__.py`, `benchmark.py`, `engine_build.py`, `evaluate.py`, `export.py`, `gpu_check.py`, `model.py`, `postprocess.py` and `runlog.py`; under `scripts/`, `check_planted_defects.py`, `ci_local.sh`, `compare_logs.py`, `fetch_coco.py`, `generate_table.py`, `generate_tree.py`, `gpu_bounds.py`, `host/host_setup.py`, `host/quiet.py`, `recompute_ap.py` and `recompute_stats.py`; 33 files under `tests/`, 30 changed and 3 new; and `.github/workflows/ci.yml`. In `__init__.py`, `gpu_check.py`, `model.py` and `postprocess.py` only comments and docstrings changed. None of the changes alters what a benchmark times or how an eval scores: the benchmark's timed loop and its `_iteration`, and the eval's scoring functions, are unchanged apart from docstrings (the timed loop itself is unchanged; since this release it runs inside try/finally blocks that release the samplers and the backend), and `benchmark.py`, `evaluate.py`, `export.py` and `engine_build.py` change only what a log records and when a run refuses: the benchmark refuses a pool under 200 images and writes no log when it reads no power samples, and the export's ONNX check now fails on a NaN or infinite output. `CHANGELOG.md` lists each change. The published logs were written by the code at `8427133`, so they lack the fields this release's code adds to a log: the raw side file, the input hashes, the eval's detections file, the ONNX sha256, the image pool count, the board model string and the reason an unread GPU-process count was not read.

The judged runs' host quieting was done by scripts outside this repository. `scripts/host/` holds generic equivalents of the QUIET gate, the sampler and the root setup, written after the judged runs; they are not the scripts that ran them. The generated blocks read only sampler records that carry the as-run sampler's sha256 (beginning `aa30652b`), so records written by `scripts/host/quiet.py` are not read as a judged run's. [Reproduce](#reproduce-the-results) says what a re-measure needs.

Each QUIET gate was read before its job, not during it. Two timing windows opened with the one-minute load at or above the gate's bound of 1.0: run6's TensorRT FP32 benchmark at 1.51 and run8's INT8 benchmark at 1.03.

How the judged runs were chosen. No rule for choosing them was fixed before the runs. Five complete runs came first: runs 1 and 2 on 2026-11-14 at `025171d`, whose gates checked load only; runs 3, 4 and 5 on 2026-11-15 at `025171d`, which measured code this release no longer runs (the INT8 engine built without the FP16 flag, and the full-frame image check inside the timed loop); and run 6 on 2026-11-17 at `8ce8f5c`, which completed alone when run 7 did not start. Runs 6, 7 and 8 on 2026-11-18 at `8427133` were the first three runs at one commit, after those changes, in which all three completed with every QUIET gate passing; the author named them the judged runs after they completed, and no run of that window was left out. The other complete runs' figures are under Provenance; their logs are in `runs-superseded/` and `runs-attempts/run6-window1/`, and runs 1 and 2's are on the host only. No rule for the memory bound was fixed in advance either: it was lowered from 6.5e9 to 6.0e9 bytes on 2026-11-15, after five attempts had stopped on it.

Across the judged runs the TensorRT rows' end-to-end p50s overlap (FP16 58.13-62.64 ms, TensorRT FP32 60.17-62.60 ms, INT8 61.91-65.78 ms, as the Results section gives them), so the table does not order them end to end; their engine p50s do separate. Energy per frame is one figure per run.

Person detection on a Jetson Orin Nano Super with torchvision's SSDLite320 MobileNetV3-Large and its COCO weights, measured at FP32 in PyTorch, the PyTorch baseline, and at FP16 and INT8 in TensorRT; TensorRT FP32, the precision baseline, is measured in the same judged runs. The benchmark reports accuracy on COCO 2017 val, latency, frames per second, memory and power. The headline numbers are the median of the judged runs' JSON logs under `runs-repeat/`, which also hold each run's QUIET gate and sampler files; the earlier run's logs are in `runs/`, and a test fails if the table and the logs disagree. [Results](#results) and [Limitations](#limitations) follow; the hardware and software, how the judged runs were checked, the Quickstart and how to reproduce the results come after them.

## Results

The end-to-end columns time a whole frame: decoded, preprocessed, run and postprocessed, and the CPU steps take most of it. Their postprocess is torchvision's evaluation setting, a score threshold of 0.001 over all 91 class columns, which keeps far more candidates than a detector deployed for people would; the deployment-style benchmark below times a person-only setting, and no accuracy is measured at that setting. The engine columns time the inference backend alone: the TensorRT engine, or the PyTorch network in the PyTorch FP32 row; the board never ran a frame at the engine FPS.

Latencies are milliseconds, at batch size 1. "Engine" covers inference, including the copies to and from device memory. "End-to-end" adds image decoding, preprocessing and postprocessing. FPS is 1 / mean latency over the serial batch-1 iterations, not a sustained throughput with several requests in flight. Power is the mean board input over the whole timed loop, which ran at the end-to-end rate, so it sits with the end-to-end columns: energy per frame is each run's power divided by its end-to-end FPS, and the board never ran at the engine FPS while power was read. The energy cell is the median of those per-run figures, so the displayed power divided by the displayed FPS need not equal it. Each run builds its own engines, so each TensorRT row's figures come from three builds whose sizes differ even at one flag set (TensorRT FP32 15056116-15067292 bytes, FP16 8412724-8482108, INT8 6071876-6179412), and the three INT8 builds share one calibration cache, whose sha256 begins `d440b41a`. The CPU clocks were not locked: the end-to-end figures were measured under the `schedutil` governor, and Limitations gives what the CPU clock samples show. The PyTorch FP32 row is the PyTorch baseline: the same network in PyTorch. The TensorRT FP32 row is the precision baseline: the same TensorRT runtime and ONNX graph as FP16 and INT8, built with neither reduced-precision flag and with TF32 cleared. FP16 and INT8 differ from it by their reduced-precision flags, and with them TensorRT may choose other tactics, layers it keeps out of FP16 or INT8 may run in TF32, and the INT8 Q/DQ fallback builds from another graph, so the difference is not precision alone. The builder flags each TensorRT build sets, as every build log from 0.1.1 on records them in `builder_flags`: FP16 sets FP16 and TF32; INT8 sets FP16, INT8 and TF32; TensorRT FP32 sets none. Every judged INT8 build log records the flags FP16, INT8 and TF32. All precisions use the same postprocess, which the end-to-end time includes: torchvision's settings for this model, a score threshold of 0.001, the top 300 candidates per class, NMS at IoU 0.55 and at most 300 detections per image, over all 91 class columns, run on the CPU. Each benchmark pools three timed repeats of its loop, and the notes give the range of the per-repeat medians. Each repeat times 1000 iterations, after 50 untimed warm-up iterations, so each percentile pools 3000 samples; the images are the first 200 val2017 images in file-name order, taken by 3000 draws with replacement from that pool, seeded with 7, so every run times the same image sequence. The warm-up runs once per benchmark, then the board is sampled idle for 3.0 s, then the three repeats run; there is no warm-up between repeats. The pool size and order are fixed in `src/peddet_orin/benchmark.py` (`IMAGE_POOL_SIZE` and a sorted file list) and are not written into the judged logs, which record the seed; logs written from the 0.1.1 release commit on also record the pool count and the sha256 of the image names. Each percentile is NumPy's `percentile` with its default linear interpolation, computed by the run's logger from samples the judged logs do not keep, so the published percentiles cannot be recomputed from them. Every run benchmarks the precisions in one fixed order, the PyTorch baseline first; the order is not randomised, so an order effect is not separated from a difference between precisions; counterbalancing the order is listed in `QUEUE.md`. The timed loop holds one `perf_counter` call between decode and preprocess, to time each stage; the judged runs' benchmark logs record those stage times. Accuracy is person AP on the 5000 COCO val2017 images, from pycocotools with `catIds=[1]`. No all-class COCO mAP is reported, so the numbers are not anchored to torchvision's published all-class figure for these weights.

<!-- results:begin -->
| Device and precision | Engine p50/p90/p99 (ms) | Engine-only bound FPS (1 / engine mean latency) | End-to-end p50/p90/p99 (ms) | End-to-end FPS | Avg power VDD_IN (W) | End-to-end energy per frame (J) | Peak process RSS (MiB) | Person AP@[.5:.95] | Person AP50 | INT8 path |
|---|---|---|---|---|---|---|---|---|---|---|
| GPU, PyTorch FP32 (PyTorch baseline) | 30.3 / 30.9 / 32.5 | 32.9 | 88.2 / 101.4 / 109.4 | 11.4 | 7.5 | 0.655 | 1151 | 28.01 | 49.50 | - |
| GPU, TensorRT FP32 (precision baseline) | 5.8 / 5.9 / 6.0 | 171.5 | 60.9 / 73.5 / 81.3 | 16.6 | 7.3 | 0.443 | 1154 | 28.01 | 49.50 | - |
| GPU, TensorRT FP16 | 3.8 / 3.9 / 4.0 | 260.4 | 58.7 / 71.3 / 78.4 | 17.2 | 7.1 | 0.412 | 1148 | 28.01 | 49.49 | - |
| GPU, TensorRT INT8 | 3.6 / 3.7 / 3.8 | 274.2 | 63.4 / 76.1 / 82.8 | 16.0 | 7.0 | 0.439 | 1057 | 26.83 | 48.15 | IInt8EntropyCalibrator2 |

- Each cell is the median over the n = 3 quieted runs in `runs-repeat/` (run6, run7, run8), this release's count; 0.2.0 plans 5 or more runs.
- The judged runs were made at commit `8427133` with package version 0.1.1, as their logs record.
- Memory, swap and temperature are recorded for the judged runs and the superseded runs 3-5, over each whole benchmark job, warm-up included; the earlier run in `runs/` has no such record.
- INT8 person AP differs from FP16 by -1.18 points.
- INT8 person AP is 26.83 in each of the 3 judged runs, 26.74 in each of the 3 superseded runs at `025171d` and 26.83 in the earlier run; each run builds its own INT8 engine, so each score comes from a separate build. Why the superseded runs score differently from the judged runs is not known. The attempts' own INT8 builds scored 26.74 in `runs-attempts/run3-attempt2/` at `025171d`, 26.83 in `runs-attempts/run6-window1/` at `8ce8f5c`.
- The judged runs' power mode, as `nvpmodel -q` reported it: `NV Power Mode: MAXN_SUPER 2`.
- FP32 min-max over the 3 runs: ap 28.01-28.01, ap50 49.50-49.50, engine_p50_ms 30.04-30.66, end_to_end_p50_ms 85.62-89.21, avg_power_w 7.46-7.49.
- TensorRT FP32 min-max over the 3 runs: ap 28.01-28.01, ap50 49.50-49.50, engine_p50_ms 5.79-5.85, end_to_end_p50_ms 60.47-62.46, avg_power_w 7.31-7.34.
- FP16 min-max over the 3 runs: ap 28.00-28.01, ap50 49.48-49.52, engine_p50_ms 3.83-3.86, end_to_end_p50_ms 58.50-62.18, avg_power_w 7.06-7.08.
- INT8 min-max over the 3 runs: ap 26.83-26.83, ap50 48.14-48.15, engine_p50_ms 3.63-3.65, end_to_end_p50_ms 62.11-65.67, avg_power_w 7.01-7.03.
- run6: FP32 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 1310 of 1310 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 1299 of 1310 samples; policy4 governor `schedutil`, at max 1728 MHz in 7 of 1310 samples, one-minute CPU load before timing: 0.42, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run6: FP32 p50 across repeats: engine 30.30-30.34 ms over 3 repeats, end-to-end 87.94-88.50 ms over 3 repeats.
- run6: TensorRT FP32 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 907 of 907 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 901 of 907 samples; policy4 governor `schedutil`, at max 1728 MHz in 5 of 907 samples, one-minute CPU load before timing: 1.51, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run6: TensorRT FP32 p50 across repeats: engine 5.85-5.85 ms over 3 repeats, end-to-end 60.17-60.95 ms over 3 repeats.
- run6: FP16 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 930 of 930 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 921 of 930 samples; policy4 governor `schedutil`, at max 1728 MHz in 11 of 930 samples, one-minute CPU load before timing: 0.32, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run6: FP16 p50 across repeats: engine 3.83-3.84 ms over 3 repeats, end-to-end 61.73-62.64 ms over 3 repeats.
- run6: INT8 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 942 of 942 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 938 of 942 samples; policy4 governor `schedutil`, at max 1728 MHz in 2 of 942 samples, one-minute CPU load before timing: 0.41, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run6: INT8 p50 across repeats: engine 3.64-3.65 ms over 3 repeats, end-to-end 63.01-63.65 ms over 3 repeats.
- run6: PyTorch GPU check: usable True, device compute capability 8.7, wheel slices sm_80, sm_90, sm_100, sm_110, sm_120. PyTorch emitted an unsupported-GPU warning at first CUDA use: "Found GPU0 Orin which is of compute capability (CC) 8.7."; the benchmark's PyTorch backend turns TF32 off for cuDNN and matmul when it is built, and the GPU check process set both flags itself and read back cudnn.allow_tf32 False and cuda.matmul.allow_tf32 False; the eval and benchmark logs record neither flag.
- run7: FP32 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 1328 of 1328 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 1310 of 1328 samples; policy4 governor `schedutil`, at max 1728 MHz in 14 of 1328 samples, one-minute CPU load before timing: 0.69, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run7: FP32 p50 across repeats: engine 30.63-30.69 ms over 3 repeats, end-to-end 88.97-89.52 ms over 3 repeats.
- run7: TensorRT FP32 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 910 of 910 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 901 of 910 samples; policy4 governor `schedutil`, at max 1728 MHz in 11 of 910 samples, one-minute CPU load before timing: 0.87, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run7: TensorRT FP32 p50 across repeats: engine 5.82-5.83 ms over 3 repeats, end-to-end 60.55-61.09 ms over 3 repeats.
- run7: FP16 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 876 of 876 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 873 of 876 samples; policy4 governor `schedutil`, at max 1728 MHz in 3 of 876 samples, one-minute CPU load before timing: 0.49, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run7: FP16 p50 across repeats: engine 3.85-3.86 ms over 3 repeats, end-to-end 58.13-58.84 ms over 3 repeats.
- run7: INT8 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 923 of 923 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 921 of 923 samples; policy4 governor `schedutil`, at max 1728 MHz in 2 of 923 samples, one-minute CPU load before timing: 0.29, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run7: INT8 p50 across repeats: engine 3.64-3.64 ms over 3 repeats, end-to-end 61.91-62.21 ms over 3 repeats.
- run7: PyTorch GPU check: usable True, device compute capability 8.7, wheel slices sm_80, sm_90, sm_100, sm_110, sm_120. PyTorch emitted an unsupported-GPU warning at first CUDA use: "Found GPU0 Orin which is of compute capability (CC) 8.7."; the benchmark's PyTorch backend turns TF32 off for cuDNN and matmul when it is built, and the GPU check process set both flags itself and read back cudnn.allow_tf32 False and cuda.matmul.allow_tf32 False; the eval and benchmark logs record neither flag.
- run8: FP32 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 1275 of 1275 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 1268 of 1275 samples; policy4 governor `schedutil`, at max 1728 MHz in 6 of 1275 samples, one-minute CPU load before timing: 0.30, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run8: FP32 p50 across repeats: engine 29.99-30.10 ms over 3 repeats, end-to-end 85.14-86.02 ms over 3 repeats.
- run8: TensorRT FP32 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 931 of 931 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 929 of 931 samples; policy4 governor `schedutil`, at max 1728 MHz in 2 of 931 samples, one-minute CPU load before timing: 0.29, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run8: TensorRT FP32 p50 across repeats: engine 5.78-5.79 ms over 3 repeats, end-to-end 62.26-62.60 ms over 3 repeats.
- run8: FP16 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 874 of 874 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 874 of 874 samples; policy4 governor `schedutil`, at max 1728 MHz in 1 of 874 samples, one-minute CPU load before timing: 0.54, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run8: FP16 p50 across repeats: engine 3.82-3.83 ms over 3 repeats, end-to-end 58.43-58.89 ms over 3 repeats.
- run8: INT8 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 973 of 973 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 960 of 973 samples; policy4 governor `schedutil`, at max 1728 MHz in 15 of 973 samples, one-minute CPU load before timing: 1.03, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run8: INT8 p50 across repeats: engine 3.63-3.63 ms over 3 repeats, end-to-end 65.61-65.78 ms over 3 repeats.
- run8: PyTorch GPU check: usable True, device compute capability 8.7, wheel slices sm_80, sm_90, sm_100, sm_110, sm_120. PyTorch emitted an unsupported-GPU warning at first CUDA use: "Found GPU0 Orin which is of compute capability (CC) 8.7."; the benchmark's PyTorch backend turns TF32 off for cuDNN and matmul when it is built, and the GPU check process set both flags itself and read back cudnn.allow_tf32 False and cuda.matmul.allow_tf32 False; the eval and benchmark logs record neither flag.
<!-- results:end -->

A second, deployment-style benchmark runs the same timed loop with a postprocess that keeps the person class only, COCO category 1, at one fixed score threshold of 0.5: a fixed midpoint; the author's record is that it was set before any deployment-style run and not tuned on the results, and no recall or AP is reported at this threshold. Its other settings are torchvision's. It writes its own log kind, `benchmark_deploy`, which the results table and R-11 never use. Each judged run carries a `benchmark_deploy` log for each of the four precisions.

<!-- deploy:begin -->
Each cell is the median over the 3 judged runs, from the `benchmark_deploy` logs, which the results table and R-11 never use.

| Device and precision | End-to-end p50/p90/p99 (ms) | End-to-end FPS | Avg power VDD_IN (W) | End-to-end energy per frame (J) | Postprocess p50 (ms) | Main benchmark postprocess p50 (ms) |
|---|---|---|---|---|---|---|
| GPU, PyTorch FP32 (PyTorch baseline) | 41.1 / 43.2 / 45.5 | 24.3 | 8.3 | 0.342 | 3.20 | 47.72 |
| GPU, TensorRT FP32 (precision baseline) | 16.4 / 18.7 / 21.5 | 61.0 | 8.9 | 0.145 | 3.02 | 45.58 |
| GPU, TensorRT FP16 | 14.2 / 16.6 / 19.4 | 69.9 | 8.0 | 0.114 | 2.92 | 45.45 |
| GPU, TensorRT INT8 | 14.1 / 16.3 / 19.2 | 70.9 | 7.8 | 0.110 | 2.88 | 50.97 |

- Postprocess, as the logs record it: labels kept 1 (COCO category 1, person), score threshold 0.5, NMS IoU threshold 0.55, top-k 300 per class, at most 300 detections per image, over 91 classes. The score threshold is a fixed midpoint; the author's record is that it was set before any deployment-style run and not tuned on the results. The other settings are the main benchmark's of the same precision.
- Only end-to-end figures and the postprocess stage are shown: the postprocess runs after the engine, whose time the results table gives. The main benchmark's postprocess p50, from its own log, stands beside the deploy run's. No recall or AP is reported at this threshold.
- Each deployment-style log ran with the runtime and iteration counts of the main benchmark of its precision, which the script checks. The gate and sampler tables below list the deployment-style jobs' gates and samples beside the main jobs'.
- PyTorch FP32 deploy end-to-end p50 across repeats: 39.68-41.20 ms over 9 repeats of the 3 judged runs.
- TensorRT FP32 deploy end-to-end p50 across repeats: 16.11-16.47 ms over 9 repeats of the 3 judged runs.
- FP16 deploy end-to-end p50 across repeats: 14.12-14.37 ms over 9 repeats of the 3 judged runs.
- INT8 deploy end-to-end p50 across repeats: 14.00-14.21 ms over 9 repeats of the 3 judged runs.
- FP16 and INT8 deploy end-to-end p50 ranges overlap (14.12-14.37 ms vs 14.00-14.21 ms), so their order is not settled by these runs.
<!-- deploy:end -->

The PyTorch FP32 row runs PyTorch on the GPU with TF32 turned off for cuDNN and matmul: the benchmark's PyTorch backend turns both off when it is built, and the GPU check process set both itself and read them back as False in every judged run, as the notes show; the eval and benchmark logs record neither flag. On this board a numerical test cannot tell TF32 from FP32: by the author's record of one board run of `scripts/gpu_bounds.py` before its TF32 line was dropped (the log's sha256 begins `0fb14a18dc74f700`, as the bounds' comments in `tests/test_gpu.py` cite), the seeded network's relative error against the CPU was 1.13e-5 with TF32 on and 1.35e-5 with it off. So the flags are the record that TF32 was off, the test that was meant to show it is dropped, and `QUEUE.md` lists a check that can fail. By the author's record, the GPU suite, 14 tests, passed on the board with 0 skipped at code equal to this release's (every Python file equal by syntax tree, docstrings and comments aside, and every other non-document file byte-equal), and the GPU bounds were measured three times, with INT8 cosines of 0.917264, 0.917473 and 0.917557; those logs are not in this repository. PyTorch warns on first CUDA use that its published 2.14.1+cu130 wheels do not support this board's compute capability, 8.7; the summary quotes the warning's line that says no published build supports this GPU; the notes quote its first line; the `gpu_check` log holds all of it, and the code does not hide it. Which kernel image the driver runs is not reported. `python -m peddet_orin.gpu_check` decides by running whether the GPU build is usable: it compares a matmul and NMS against the CPU, and `tests/test_gpu.py` compares a full forward pass.

Peak process RSS is each benchmark process's own. Every benchmark process loads the PyTorch detector, the TensorRT ones included, to take its postprocess settings (`src/peddet_orin/benchmark.py`), so the TensorRT rows' RSS includes the PyTorch model. Peak process RSS does not fully capture memory allocated on the device. System RAM rise, the highest system-wide RAM during timing minus the mean over the 3 s just before timing, is in the logs but not in the table: it includes any other activity on the board and does not reproduce between runs.

<!-- int8:begin -->
Over the median of the 3 quieted runs, the INT8 engine p50 is lower than FP16's, 3.64 against 3.84 ms, and its end-to-end p50 is higher than FP16's, 63.38 against 58.71 ms; FP16's and INT8's per-repeat end-to-end ranges in the notes do not overlap in any of the 3 runs. The FP16 engine p50 is 6.5% of its end-to-end p50; the rest is image decoding, preprocessing and postprocessing on the CPU. INT8 yields more person detections after NMS on val2017 than FP16, 398824 against 364926, the medians of the judged runs' `detections` counts in the eval logs. Whether more detections give the postprocess more work during the benchmark is a hypothesis: no profile here separates it from run-to-run variation. R-11 cannot order INT8 and FP16: the engine p50 tolerance, 10%, is wider than their 5.5% difference, computed from the rounded figures shown and the end-to-end p50 tolerance, 15%, is wider than their 8.0% difference, computed from the rounded figures shown, so a re-measure within the tolerances does not confirm which of the two is faster.
<!-- int8:end -->

Across the 3 judged runs, the per-repeat end-to-end p50s of the three TensorRT rows overlap pairwise (FP16 58.13-62.64 ms, TensorRT FP32 60.17-62.60 ms, INT8 61.91-65.78 ms), so the order of their end-to-end columns can change between runs; within each run FP16's is below INT8's. Energy per frame is one figure per run, and the table gives no range for it.

## Limitations

- The INT8 row's engine is built with the FP16, INT8 and TF32 flags, so TensorRT may run layers in FP16 or TF32 where it finds them faster; no per-layer report shows how much of the engine runs in INT8.
- One model, one dataset and one board. The numbers do not transfer to other Jetson modules, power modes or JetPack versions.
- The export uses PyTorch's TorchScript-based ONNX exporter, which PyTorch has deprecated; `pyproject.toml` bounds torch below 2.15 for that reason.
- CI runs on x86_64 CPUs only. The aarch64 build, the GPU tests and TensorRT run only on the Jetson, by hand, before a release.
- Batch size 1 only.
- PyTorch FP32 runs in PyTorch and FP16 and INT8 in TensorRT, so the gap between the PyTorch baseline and the other two is runtime and precision together, not precision alone. TensorRT FP32, the precision baseline, runs in the same runtime as FP16 and INT8, so it separates runtime from precision more closely, though not entirely (see Results). The PyTorch FP32 row runs on a PyTorch wheel without support for this board's compute capability, 8.7, so the TensorRT FP32 row is the FP32 reference for FP16 and INT8. Which kernel image the driver runs for the PyTorch FP32 row was not identified.
- `jetson_clocks` is not run, memory clocks are not read, and a tegrastats reader thread runs in the timed process. For the judged runs, the QUIET gates and the measurement window (a span with the board quieted and its other workloads stopped for the runs) excluded, at the time each gate was read before its job, what the repeat block and the Provenance notes name: the board's other scheduled jobs and other host services, other processes over their CPU budget or not allowed, swap use and swap-outs, available memory under the bound, and a load1 at or above the gate's bound. During the jobs the gates did not hold: the one-minute load before timing reached 1.51 (run6 TensorRT FP32) and 1.03 (run8 INT8), and available memory fell below the 5859375 kB bound in every benchmark job, to 5462372 kB at the lowest, and to 4402472 kB in a build job. Swap stayed at 0 kB in every sampler line. They did not exclude the operating system's own processes or the tegrastats reader, so latencies still vary between runs, as the min-max notes show. The earlier run in `runs/` had no such gate.
- The system RAM rise in each benchmark log is reported per run and not judged.
- Power is the average of `tegrastats` VDD_IN readings during the timed end-to-end loop. It is the whole board's input power, not the GPU's alone, and no power was read for inference alone, so there is no energy figure at the engine rate.
- Energy per frame divides that whole-board power by the end-to-end rate, so it includes the board's idle draw. The logs do not record the idle power, so no energy above idle is given.
- Engine time is measured on the CPU around each inference call and includes the synchronous copies to and from device memory; no GPU-only kernel time is recorded. The per-stage engine time in each judged run's benchmark log is the nearest figure.
- INT8 is calibrated on one set, 500 seeded train2017 images. INT8 accuracy with another calibration set was not measured here.
- INT8 uses TensorRT's `IInt8EntropyCalibrator2`, which TensorRT deprecated in 10.1 in favour of explicit quantization; it still builds with 10.16.2.10, and its deprecation warning is filtered from the build output.
- The INT8 GPU tests check that each engine builds, runs and gives outputs of the expected shapes with finite values; INT8 accuracy is measured by the COCO evaluation, not by those tests.
- The ONNX export is checked against PyTorch on seeded random input, not on real images.
- The COCO archives are checked against the MD5 and sha256 values pinned in `scripts/fetch_coco.py`; MD5 alone is not collision-resistant, sha256 is; the 500 calibration images are each checked by sha256 against `manifests/calibration_train2017.sha256`.
- `scripts/compare_logs.py` prints each value to two decimals but judges the unrounded values, so a line can read as exactly at its tolerance and still be `OUT`.
- The judged logs hold summaries (percentiles, means and counts), not the raw per-iteration timings or power readings; the sampler files under `runs-repeat/` hold one line per second, and their `pswpin` and `pswpout` fields count pages from the board's boot, so a run's swap-outs are the change between its lines. A benchmark run from the 0.1.1 release commit on also writes a raw side file beside its log.
- Each run exports the ONNX graph once and builds its engines from it; the judged logs record each engine's sha256, not the ONNX file's, which logs written from the 0.1.1 release commit on also record.
- The end-to-end columns are bound by the CPU: in the TensorRT rows of the judged runs the postprocess stage alone is 74.8-80.7% of a frame's end-to-end p50, and the CPU clock samples of the main benchmarks show policy0 at its 1728 MHz maximum in 98.6-100% of samples and policy4 at 729.6 MHz, its lowest recorded clock, in 98.4-99.8%, both under the `schedutil` governor; the logs do not record which cores each policy covers.
- By the author's records, swap was off for the judged runs, so the QUIET gates' swap terms could not fail. The logs show 0 kB of swap in use at every gate and in every sampler line, and no page swapped out during the runs; no log records whether swap was on.
- R-11 judges 3 runs, each against a median it is part of, with tolerances far wider than their spread: the largest end-to-end p50 deviation from the median is 5.9% (FP16, run6) against a 15% tolerance. Passing it shows that the runs agree, not how precise the numbers are.
- The Quickstart's `apt install` line pins no versions: it installs what the JetPack 7 repositories serve at the time. The versions the measured board had are named beside it.
- Responsible use: this measures a pretrained COCO person detector on one board; it is not a tested product. It is not meant for surveillance or for monitoring people without their consent. Its accuracy was not measured across age, skin tone, body type, clothing, pose or disability. It misses people and reports false detections, at rates the AP figures only summarise. Its output is not fit for safety- or security-critical use, or for decisions about individuals, without a person who checks each result.

## Hardware and software

The results were measured with this stack. Each run log records the Jetson Linux release string, the power mode (or `UNREADABLE` when it could not be read) and the Python, PyTorch, torchvision, CUDA, TensorRT, ONNX, ONNX Runtime and peddet-orin versions it ran with; the judged runs' logs also record the numpy, Pillow, pycocotools and libjpeg versions. The board and JetPack rows come from the board's documentation, not from the logs.

| Component | Version |
|---|---|
| Board | Jetson Orin Nano Super by the author's record (the logs' Jetson Linux release string reads `BOARD: generic`), GPU compute capability 8.7 |
| JetPack | 7 by the author's record; Jetson Linux R39 revision 2.1, as the logs' release string records |
| Power mode | `MAXN_SUPER` (mode 2), as every judged benchmark log records |
| CUDA | runtime 13.0, from the PyTorch `cu130` wheels; driver 13.2, as the judged runs' `gpu_check` logs record it |
| TensorRT | 10.16.2.10, from the JetPack system Python |
| Python | 3.12 |
| PyTorch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 |
| ONNX / ONNX Runtime | 1.23.2 / 1.31.1 |

The judged runs used Pillow 13.0.0, the pinned release; JPEG decoding by Pillow is inside the end-to-end timings, not the engine timings.

CI runs the CPU tests on Ubuntu 24.04 x86_64 with Python 3.11 and 3.12; other Linux systems are not tested. See "Continuous integration" below.

## Judged runs

How the median in [Results](#results) was judged, with the gate, memory, swap, temperature and sampler records of each judged run, all generated from the logs and host records in `runs-repeat/`. By the author's record, each gate line records 3 scheduled jobs allowed by name.

<!-- repeats:begin -->
R-11 checks that the 3 quieted runs run6, run7, run8 in `runs-repeat/`, at commit `8427133`, agree with their median: a metric is IN when every judged run is within its tolerance of the median, and OUT otherwise. Each run is part of the median it is judged against, so this is the judged runs' consistency, not a re-measure from a fresh clone; `scripts/compare_logs.py` checks a re-measure against the same median and tolerances. The tolerances are the `PRD.md` values in `scripts/compare_logs.py`, unchanged. The spread comes from 3 runs on one board, which is not a population. avg_power_w is the board's input power, VDD_IN, as its sensor measures it. The earlier run in `runs/`, at `17a1ee2`, gave 0.1.0's table without the quieting; it is shown apart with the deviation from the median, an uncontrolled before-and-after difference that also takes in code and library changes, and not judged.

Each judged run's QUIET gate ran at its start and before each benchmark, after a settle; each gate passed on the try shown, and its passing line records: 0 scheduled jobs or path units active other than the ones allowed by name, 0 other processes over their CPU budget or not allowed, 0 other host services active, 0 kB of swap used, 0 pages swapped out during the settle; 3 scheduled jobs allowed by name; load1 below 1.0; available memory at least 5859375 kB.
2 of the 3 judged runs had a gate below the earlier attempts' bound of 6.5e9 bytes, 6347657 kB, and would not have run under it.

| Run | Gate | Try | load1 | Available memory (kB) |
|---|---|---|---|---|
| run6 | start | 1 | 0.35 | 6576500 |
| run6 | bench_fp32 | 1 | 0.17 | 6413556 |
| run6 | bench_fp16 | 1 | 0.23 | 6415120 |
| run6 | bench_int8 | 1 | 0.24 | 6411324 |
| run6 | bench_fp32-trt | 1 | 0.55 | 6408572 |
| run6 | bench_deploy_fp32 | 1 | 0.23 | 6406120 |
| run6 | bench_deploy_fp16 | 1 | 0.15 | 6424148 |
| run6 | bench_deploy_int8 | 1 | 0.17 | 6416548 |
| run6 | bench_deploy_fp32-trt | 1 | 0.78 | 6411680 |
| run7 | start | 1 | 0.04 | 6412132 |
| run7 | bench_fp32 | 1 | 0.61 | 6297200 |
| run7 | bench_fp16 | 1 | 0.21 | 6295388 |
| run7 | bench_int8 | 1 | 0.19 | 6289976 |
| run7 | bench_fp32-trt | 1 | 0.22 | 6294948 |
| run7 | bench_deploy_fp32 | 1 | 0.20 | 6295604 |
| run7 | bench_deploy_fp16 | 1 | 0.25 | 6294052 |
| run7 | bench_deploy_int8 | 1 | 0.55 | 6299944 |
| run7 | bench_deploy_fp32-trt | 1 | 0.32 | 6302068 |
| run8 | start | 1 | 0.14 | 6293360 |
| run8 | bench_fp32 | 1 | 0.20 | 6326300 |
| run8 | bench_fp16 | 1 | 0.30 | 6332076 |
| run8 | bench_int8 | 1 | 0.56 | 6331516 |
| run8 | bench_fp32-trt | 1 | 0.19 | 6327948 |
| run8 | bench_deploy_fp32 | 1 | 0.26 | 6322816 |
| run8 | bench_deploy_fp16 | 1 | 0.25 | 6321732 |
| run8 | bench_deploy_int8 | 1 | 0.33 | 6326624 |
| run8 | bench_deploy_fp32-trt | 1 | 0.26 | 6328736 |

- run6: quieted, every QUIET gate passed; one-minute CPU load before each main benchmark's timing 0.32-1.51.
- run7: quieted, every QUIET gate passed; one-minute CPU load before each main benchmark's timing 0.29-0.87.
- run8: quieted, every QUIET gate passed; one-minute CPU load before each main benchmark's timing 0.29-1.03.
- earlier run (`runs/`): not quieted, load before every benchmark at or above the QUIET gate's 1.0; one-minute CPU load before each main benchmark's timing 3.85-5.54.

| Metric | Precision | Median | run6 | run7 | run8 | Min-max | Spread | Tolerance | R-11 | earlier run |
|---|---|---|---|---|---|---|---|---|---|---|
| ap | FP32 | 28.01 | 28.01 | 28.01 | 28.01 | 28.01-28.01 | 0.00 | 0.10 | IN | 28.01 (0.00) |
| ap50 | FP32 | 49.50 | 49.50 | 49.50 | 49.50 | 49.50-49.50 | 0.00 | 0.10 | IN | 49.50 (0.00) |
| ap | TensorRT FP32 | 28.01 | 28.01 | 28.01 | 28.01 | 28.01-28.01 | 0.00 | 0.10 | IN | missing |
| ap50 | TensorRT FP32 | 49.50 | 49.50 | 49.50 | 49.50 | 49.50-49.50 | 0.00 | 0.10 | IN | missing |
| ap | FP16 | 28.01 | 28.00 | 28.01 | 28.01 | 28.00-28.01 | 0.01 | 0.10 | IN | 28.00 (-0.01) |
| ap50 | FP16 | 49.49 | 49.48 | 49.52 | 49.49 | 49.48-49.52 | 0.04 | 0.10 | IN | 49.48 (-0.01) |
| ap | INT8 | 26.83 | 26.83 | 26.83 | 26.83 | 26.83-26.83 | 0.00 | 0.50 | IN | 26.83 (0.00) |
| ap50 | INT8 | 48.15 | 48.14 | 48.15 | 48.15 | 48.14-48.15 | 0.01 | 0.50 | IN | 48.15 (0.00) |
| engine_p50_ms | FP32 | 30.32 | 30.32 | 30.66 | 30.04 | 30.04-30.66 | 0.62 | 10% | IN | 30.31 (0.0%) |
| end_to_end_p50_ms | FP32 | 88.22 | 88.22 | 89.21 | 85.62 | 85.62-89.21 | 3.59 | 15% | IN | 96.61 (+9.5%) |
| avg_power_w | FP32 | 7.47 | 7.47 | 7.46 | 7.49 | 7.46-7.49 | 0.03 | 10% | IN | 8.63 (+15.5%) |
| engine_p50_ms | TensorRT FP32 | 5.82 | 5.85 | 5.82 | 5.79 | 5.79-5.85 | 0.06 | 10% | IN | missing |
| end_to_end_p50_ms | TensorRT FP32 | 60.90 | 60.47 | 60.90 | 62.46 | 60.47-62.46 | 1.99 | 15% | IN | missing |
| avg_power_w | TensorRT FP32 | 7.33 | 7.33 | 7.34 | 7.31 | 7.31-7.34 | 0.03 | 10% | IN | missing |
| engine_p50_ms | FP16 | 3.84 | 3.84 | 3.86 | 3.83 | 3.83-3.86 | 0.03 | 10% | IN | 3.88 (+1.1%) |
| end_to_end_p50_ms | FP16 | 58.71 | 62.18 | 58.50 | 58.71 | 58.50-62.18 | 3.68 | 15% | IN | 75.13 (+28.0%) |
| avg_power_w | FP16 | 7.08 | 7.06 | 7.08 | 7.08 | 7.06-7.08 | 0.02 | 10% | IN | 8.75 (+23.6%) |
| engine_p50_ms | INT8 | 3.64 | 3.65 | 3.64 | 3.63 | 3.63-3.65 | 0.02 | 10% | IN | 3.69 (+1.4%) |
| end_to_end_p50_ms | INT8 | 63.38 | 63.38 | 62.11 | 65.67 | 62.11-65.67 | 3.56 | 15% | IN | 78.33 (+23.6%) |
| avg_power_w | INT8 | 7.01 | 7.01 | 7.03 | 7.01 | 7.01-7.03 | 0.02 | 10% | IN | 8.66 (+23.5%) |

R-11 is met for every judged metric among the quieted runs.

Each judged run's state before any measurement, from the first sample of its first GPU job:

- run6 first sample, 2026-11-18T09:37:29Z: available memory 6411 MiB, swap used 0 kB, temperatures (C) cpu-thermal 51.4, gpu-thermal 51.8, soc0-thermal 50.8, soc1-thermal 51.8, soc2-thermal 50.1, tj-thermal 51.8.
- run7 first sample, 2026-11-18T10:59:43Z: available memory 6249 MiB, swap used 0 kB, temperatures (C) cpu-thermal 52.1, gpu-thermal 52.4, soc0-thermal 51.6, soc1-thermal 52.6, soc2-thermal 50.8, tj-thermal 52.6.
- run8 first sample, 2026-11-18T12:21:30Z: available memory 6139 MiB, swap used 0 kB, temperatures (C) cpu-thermal 51.7, gpu-thermal 52.3, soc0-thermal 51.4, soc1-thermal 52.4, soc2-thermal 50.8, tj-thermal 52.5.

Memory, swap and temperature of each judged run's main benchmark jobs, sampled every 1 s by the host's sampler script, a separate process, over the whole job, warm-up included; runs shown apart had no sampler. The thermal zones are the ones whose readings the records hold: cpu-thermal, gpu-thermal, soc0-thermal, soc1-thermal, soc2-thermal, tj-thermal.

| Run | Precision | Swap used at start (kB) | Pages swapped out | Lowest available memory (MiB) | Highest temperature (C) |
|---|---|---|---|---|---|
| run6 | FP32 | 0 | 0 | 5464 | cpu-thermal 54.5, gpu-thermal 55.0, soc0-thermal 53.8, soc1-thermal 54.9, soc2-thermal 53.3, tj-thermal 55.0 |
| run6 | TensorRT FP32 | 0 | 0 | 5483 | cpu-thermal 54.2, gpu-thermal 54.8, soc0-thermal 53.5, soc1-thermal 54.7, soc2-thermal 53.1, tj-thermal 54.8 |
| run6 | FP16 | 0 | 0 | 5418 | cpu-thermal 54.0, gpu-thermal 54.3, soc0-thermal 53.1, soc1-thermal 54.2, soc2-thermal 52.8, tj-thermal 54.3 |
| run6 | INT8 | 0 | 0 | 5592 | cpu-thermal 53.8, gpu-thermal 54.2, soc0-thermal 53.0, soc1-thermal 54.1, soc2-thermal 52.5, tj-thermal 54.2 |
| run7 | FP32 | 0 | 0 | 5334 | cpu-thermal 54.8, gpu-thermal 55.3, soc0-thermal 54.1, soc1-thermal 55.2, soc2-thermal 53.7, tj-thermal 55.3 |
| run7 | TensorRT FP32 | 0 | 0 | 5334 | cpu-thermal 54.3, gpu-thermal 54.8, soc0-thermal 53.6, soc1-thermal 54.7, soc2-thermal 53.2, tj-thermal 54.8 |
| run7 | FP16 | 0 | 0 | 5350 | cpu-thermal 54.2, gpu-thermal 54.6, soc0-thermal 53.4, soc1-thermal 54.6, soc2-thermal 52.9, tj-thermal 54.6 |
| run7 | INT8 | 0 | 0 | 5478 | cpu-thermal 53.8, gpu-thermal 54.2, soc0-thermal 53.2, soc1-thermal 54.3, soc2-thermal 52.7, tj-thermal 54.3 |
| run8 | FP32 | 0 | 0 | 5371 | cpu-thermal 54.8, gpu-thermal 55.2, soc0-thermal 54.0, soc1-thermal 55.1, soc2-thermal 53.5, tj-thermal 55.2 |
| run8 | TensorRT FP32 | 0 | 0 | 5364 | cpu-thermal 54.3, gpu-thermal 54.9, soc0-thermal 53.7, soc1-thermal 54.8, soc2-thermal 53.3, tj-thermal 54.9 |
| run8 | FP16 | 0 | 0 | 5369 | cpu-thermal 54.1, gpu-thermal 54.6, soc0-thermal 53.4, soc1-thermal 54.6, soc2-thermal 52.9, tj-thermal 54.6 |
| run8 | INT8 | 0 | 0 | 5475 | cpu-thermal 53.9, gpu-thermal 54.5, soc0-thermal 53.3, soc1-thermal 54.5, soc2-thermal 52.8, tj-thermal 54.5 |

Each judged run's GPU jobs, from the samples taken during each job, from its start to its end:

| Run | Job | Lowest available memory (MiB) | Pages swapped out |
|---|---|---|---|
| run6 | bench_deploy_fp16 | 5527 | 0 |
| run6 | bench_deploy_fp32-trt | 5543 | 0 |
| run6 | bench_deploy_fp32 | 5515 | 0 |
| run6 | bench_deploy_int8 | 5653 | 0 |
| run6 | bench_fp16 | 5418 | 0 |
| run6 | bench_fp32-trt | 5483 | 0 |
| run6 | bench_fp32 | 5464 | 0 |
| run6 | bench_int8 | 5592 | 0 |
| run6 | build_fp16 | 4882 | 0 |
| run6 | build_fp32-trt | 4966 | 0 |
| run6 | build_int8 | 4322 | 0 |
| run6 | eval_fp16 | 4907 | 0 |
| run6 | eval_fp32-trt | 4973 | 0 |
| run6 | eval_fp32 | 4807 | 0 |
| run6 | eval_int8 | 4980 | 0 |
| run6 | export | 5708 | 0 |
| run6 | gpu_check | 5627 | 0 |
| run7 | bench_deploy_fp16 | 5475 | 0 |
| run7 | bench_deploy_fp32-trt | 5466 | 0 |
| run7 | bench_deploy_fp32 | 5398 | 0 |
| run7 | bench_deploy_int8 | 5578 | 0 |
| run7 | bench_fp16 | 5350 | 0 |
| run7 | bench_fp32-trt | 5334 | 0 |
| run7 | bench_fp32 | 5334 | 0 |
| run7 | bench_int8 | 5478 | 0 |
| run7 | build_fp16 | 4857 | 0 |
| run7 | build_fp32-trt | 5062 | 0 |
| run7 | build_int8 | 4393 | 0 |
| run7 | eval_fp16 | 4796 | 0 |
| run7 | eval_fp32-trt | 4795 | 0 |
| run7 | eval_fp32 | 4775 | 0 |
| run7 | eval_int8 | 4849 | 0 |
| run7 | export | 5712 | 0 |
| run7 | gpu_check | 5749 | 0 |
| run8 | bench_deploy_fp16 | 5468 | 0 |
| run8 | bench_deploy_fp32-trt | 5487 | 0 |
| run8 | bench_deploy_fp32 | 5409 | 0 |
| run8 | bench_deploy_int8 | 5553 | 0 |
| run8 | bench_fp16 | 5369 | 0 |
| run8 | bench_fp32-trt | 5364 | 0 |
| run8 | bench_fp32 | 5371 | 0 |
| run8 | bench_int8 | 5475 | 0 |
| run8 | build_fp16 | 4836 | 0 |
| run8 | build_fp32-trt | 4951 | 0 |
| run8 | build_int8 | 4299 | 0 |
| run8 | eval_fp16 | 4832 | 0 |
| run8 | eval_fp32-trt | 4806 | 0 |
| run8 | eval_fp32 | 4754 | 0 |
| run8 | eval_int8 | 4869 | 0 |
| run8 | export | 5594 | 0 |
| run8 | gpu_check | 5602 | 0 |

RAM was not a constraint in runs 6-8: the lowest available memory during any job was 4299 MiB, and 0 pages were swapped out.
<!-- repeats:end -->

### Provenance

- Each cell is the median over the judged quieted runs listed in the notes, on one board; the notes name the commit and package version their logs record. Each cell's median is taken on its own, so the cells of one row can come from different runs and their engines. Every environment field the logs record is identical across the 51 judged logs; the generator does not require it, and `scripts/compare_logs.py` fails a re-measure whose torch, TensorRT or CUDA version differs. A benchmark pools three timed repeats of its loop. At that commit, INT8 engines are built with the FP16 flag as well as INT8, the benchmark's full-frame image check is outside its timed loop, ONNX Runtime is imported where it is used, and the logs record per-stage times, engine hashes, the postprocess settings and the numpy, Pillow, pycocotools and libjpeg versions. The release commit changes code since that commit, as the first screen names and `CHANGELOG.md` lists. The run log loads ONNX Runtime to record its version when it writes the log, after the benchmark has read its peak process RSS, so the RSS in the table does not include it. 0.1.0's table was measured at `17a1ee2` from one run on a host that was not quieted.
- Every log records its commit, whether the tree was dirty, the start and end times, the Jetson Linux release string, and the Python, PyTorch, torchvision, CUDA, TensorRT, ONNX, ONNX Runtime and peddet-orin versions, and from 0.1.1 on the numpy, Pillow, pycocotools and libjpeg versions. The table uses only clean logs, all from one commit.
- The power mode in each benchmark note is what `nvpmodel -q` reported, or `UNREADABLE` when it was not read; the notes say which for the judged runs. `jetson_clocks` was not run, since it needs root. During timing a separate process samples the GPU's devfreq clock and each CPU frequency policy's clock every 200 ms; the notes give each governor and how many samples were at that clock's maximum. Memory clocks are not read.
- The one-minute load average in the notes is read just before timing starts, so it shows what else was running then. It is read after the benchmark's warm-up, so it can be higher than the load the benchmark's QUIET gate read before the job began. The number of other GPU processes could not be read, and is recorded as `UNREADABLE`.
- By the run script's record, the measurements ran one at a time, at normal priority. Every log records the number of other GPU processes as `UNREADABLE`: the count was not read. The code at `8427133` asked NVML through `pynvml` and recorded any error, an import error included, as a bare `UNREADABLE`, so the judged logs do not say why; this release's code records the reason, as `UNREADABLE:import` or `UNREADABLE:` and the NVML error's class (`src/peddet_orin/runlog.py`). By the author's record, `pynvml` was importable on the board, from the system Python's packages, so the NVML query raised; `pynvml` is not in the pinned locks; the QUIET gates checked scheduled jobs, other processes over their CPU budget, other host services, swap, available memory and load, not the GPU.
- `tests/test_readme_table.py` fails if the results block differs from what `scripts/generate_table.py` makes of the logs.
- The judged runs are run6, run7 and run8 of one measurement window on 2026-11-18 (UTC); their QUIET and sampler records run from 09:37:29 to 10:55:42 (run6), 10:59:43 to 12:17:29 (run7) and 12:21:30 to 13:39:45 (run8). By the author's host records, which are not in this repository, each run ran once in this window.
- The judged runs were measured at commit `8427133`, the parent of the release commit. The release commit changes code since then: the first screen names each changed file, and `CHANGELOG.md` lists each change.
- `runs-superseded/` holds run3, run4, run5, the judged runs of an earlier window at `025171d`, kept with their QUIET and sampler files and not judged: they were measured on 2026-11-15, before the INT8 engine was built with the FP16 flag on 2026-11-16 and the full-frame image check was moved out of the benchmark's timed loop the same day, so their INT8 numbers and all their end-to-end times measure code that this release no longer runs. Why their INT8 person AP is 26.74, not 26.83, is not known: the earlier run in `runs/`, whose INT8 engine was also built without the FP16 flag, scored 26.83.
- The measurement attempts that are not judged runs are in `runs-attempts/`: the run logs, QUIET lines and sampler files each attempt wrote, as the host kept them, and a `STOP.txt` with the run script's lines from the attempt's start to where it stopped; a `RUN... DONE` line's `logs=` counts the run directory's entries, so `run6-window1`'s `logs=12` is its 10 logs with `quiet.txt` and `sysmon/`. Their console transcripts are not included. `run6-attempt1` stopped at its GPU check because the lock that reserves the board for a measurement was not held, so the check refused to start, as its `STOP.txt` records with `gpu_check rc=75`. The attempts' 17 run logs, from `025171d` and `8ce8f5c`, pass the same log check as the committed runs (`tests/test_run_logs.py`). The root setup block and the setting of the memory bound are host records outside this repository.
- By the same records, the bound was 6.0e9 bytes for every attempt of this window.
- By the same records: five attempts stopped at a gate on available memory under the 6.5e9-byte bound (6347657 kB): run99 on 2026-11-14 at 23:46 UTC, and four on 2026-11-15 at 10:27, 10:35, 17:42 and 17:45 UTC. The bound was then set to 6.0e9 bytes (5859375 kB) at 18:15 UTC, before any judged run, and every later attempt ran under it. The judged runs' lowest gate readings were 6406120 kB (run6), 6289976 kB (run7) and 6293360 kB (run8).
- By the same records, every quieted measurement attempt, in order. Each start of a measurement block is a row, including those that stopped before any run.

  | Date (UTC) | Attempt | Commit | Memory bound | Outcome | Logs |
  |---|---|---|---|---|---|
  | 2026-11-14 | runs 1 and 2 | `025171d` | none | each wrote its 10 logs; their gate lines record load terms only, with no swap or memory term; not judged | on the host only |
  | 2026-11-14 | run 3, first attempt, 21:39 UTC | `025171d` | none | stopped at its first gate after 10 tries, with 256 kB of swap in use, before any run | `runs-attempts/run3-attempt1/`: gate lines only |
  | 2026-11-14 | run99, 23:46 UTC | `025171d` | 6.5e9 bytes | stopped at its first gate after 10 tries, on swap in use and available memory, before any run | `runs-attempts/run99/`: gate lines only |
  | 2026-11-15 | three launches, 10:24, 10:27 and 10:35 UTC | - | 6.5e9 bytes | the first refused at its start, another workload running; the other two stopped at the first gate on available memory; all before any run | none |
  | 2026-11-15 | one launch, 14:27 UTC | - | 6.5e9 bytes | refused at its start, another workload running, before any run | none |
  | 2026-11-15 | run 3, 14:38 UTC | `025171d` | 6.5e9 bytes | stopped at a QUIET gate that read 1024 kB of swap in use; runs 4 and 5 did not start | `runs-attempts/run3-attempt2/`: 7 logs |
  | 2026-11-15 | run 3, two attempts, 17:42 and 17:45 UTC | - | 6.5e9 bytes | stopped at the first gate on available memory, before any run | none |
  | 2026-11-15 | runs 3, 4 and 5 | `025171d` | 6.0e9 bytes | each completed; superseded | `runs-superseded/` |
  | 2026-11-17 | run 6, two attempts | `8ce8f5c` | 6.0e9 bytes | stopped at the GPU check, then at the first attempt's leftover directory | `runs-attempts/run6-attempt1/` and `run6-attempt2/`: no run logs |
  | 2026-11-17 | run 6, third attempt, and run 7 | `8ce8f5c` | 6.0e9 bytes | run 6 completed; run 7 did not start; not judged, since one run is not a set | `runs-attempts/run6-window1/`: run 6's 10 logs |
  | 2026-11-17 | runs 6, 7 and 8 in one window | - | 6.0e9 bytes | stopped while stopping the board's other workloads, before any run | none |
  | 2026-11-18 | runs 6, 7 and 8 in one window | `8427133` | 6.0e9 bytes | each completed; judged | `runs-repeat/` |

  The earlier run in `runs/`, at `17a1ee2` on 2026-11-14, was not quieted and is not in this table.
- By the author's records, the completed runs that were not judged gave, from their logs on the host (engine p50 / end-to-end p50 in ms, person AP): runs 1 and 2 at `025171d`, FP32 30.27 / 88.02 and 30.15 / 89.40, 28.01 and 28.01; FP16 3.83 / 63.74 and 3.82 / 60.79, 27.99 and 28.02; INT8 3.65 / 66.27 and 3.65 / 64.74, 26.74 and 26.74. Run 6 of 2026-11-17 at `8ce8f5c`: FP32 29.96 / 86.41, 28.01; FP16 3.83 / 58.95, 28.01; INT8 3.66 / 61.85, 26.83. Runs 1 and 2's logs are not in this repository; that run 6's are in `runs-attempts/run6-window1/`.
- By the same records, the window was set up by a root shell block that is not in this repository. It stopped the board's other workloads before the runs, 16 scheduled timer and path units at 09:33:11 among them, which it started again at 13:39:55, after the third run; it stopped eight host services (the container runtime, snap, firmware update and jtop) before the runs and started them again after them; it turned swap off, set swappiness to 1 and waited 120 s before each run, and no process was killed for memory. Each run's first QUIET gate read 6576500, 6412132 and 6293360 kB of available memory, above the bound of 5859375 kB, and no gate or sampler line shows a page swapped out.

## Quickstart

This needs a Jetson Orin on JetPack 7; JetPack 6 is not supported, as its Python is older than the package's `requires-python` and the board lock pins CUDA 13 wheels. Install these system packages with `sudo apt install python3.12-venv python3-libnvinfer nvidia-l4t-nvpmodel nvidia-l4t-tools`, at the versions the measured board had: `python3.12-venv` 3.12.3 for the venv; from JetPack, `python3-libnvinfer` 10.16.2.10 for TensorRT, and `nvidia-l4t-nvpmodel` and `nvidia-l4t-tools` 39.2.1 for `nvpmodel` and `tegrastats`. The board's `python3-numpy` 1.26.4 equals the numpy pin, so the venv uses it; without it, pip installs the pinned numpy. Run the commands from the repository root:

```sh
git clone https://github.com/ahmedhassan-eng/peddet-orin.git
cd peddet-orin
export TORCH_HOME="$PWD/.cache/torch" PIP_CACHE_DIR="$PWD/.cache/pip" XDG_CACHE_HOME="$PWD/.cache/xdg"
export MPLCONFIGDIR="$PWD/.cache/mpl" CUDA_CACHE_PATH="$PWD/.cache/cuda"
python3.12 -m venv .venv --system-site-packages
. .venv/bin/activate
pip install --require-hashes -r constraints/jetson-orin.lock
pip install --no-deps --no-build-isolation -e .
```

The venv uses `--system-site-packages` because TensorRT 10.16.2.10 comes from the JetPack system Python. `constraints/jetson-orin.txt` pins every package that pip installs into the venv, and `constraints/jetson-orin.lock`, made from it by `scripts/lock_hashes.py`, adds the sha256 of every file of each pinned release, so `--require-hashes` installs exactly that set and refuses any other file for them; the system Python's packages, which `--system-site-packages` exposes, are not checked. TensorRT, and numpy, whose system version equals its pin, are used from the system.

The package runs from this source checkout, not from an installed wheel. Its commands find the checkout two levels above the package, or the directory `PDORIN_ROOT` names, and refuse to start anywhere else. `python -m peddet_orin` lists the commands; each is also installed as a script named `peddet-orin-` and the command, with `_` written as `-`, for example `peddet-orin-engine-build`. Set `PDORIN_DEBUG=1` to turn on diagnostic logging and profiling.

Run the tests. `tests/network_guard.py` refuses an internet connection from the test process; it does not guard a subprocess a test starts or direct use of the C-level `_socket` module. On the Jetson, the GPU and TensorRT tests in `tests/test_gpu.py` run too, so the suite takes longer while it builds TensorRT engines. `PDORIN_REQUIRE_GPU=1` makes a GPU or TensorRT test that skips, for want of a CUDA device or TensorRT, fail the run instead:

```sh
PDORIN_REQUIRE_GPU=1 python -m pytest -rs
```

Export the network and build an FP16 engine. This downloads the pretrained weights once, and checks their sha256 before every load:

```sh
python -m peddet_orin.export
python -m peddet_orin.engine_build --precision fp16
```

## Reproduce the results

After the quickstart, fetch the data over HTTPS, check the GPU, then build and measure. `nvpmodel` and `tegrastats` must be on `PATH`; `nvpmodel` lives in the system sbin directory, which a user's `PATH` may leave out. A log records `UNREADABLE` for anything it cannot read, so a `power_mode` of `UNREADABLE` usually means `nvpmodel` was not found; it is also recorded when `nvpmodel -q` fails, times out or prints nothing. Run the measurements from a clean tree, one at a time, with nothing else using the GPU.

The judged runs were measured on a quieted board, and a re-measure is comparable with them only under the same conditions. These are prerequisites, not steps the commands take:

- Power mode `MAXN_SUPER` (`sudo nvpmodel -m 2`), with `nvpmodel` on `PATH` so every log records it.
- The GPU's devfreq governor reads `performance`, as every judged benchmark log records; the CPU governors are left at `schedutil`. `jetson_clocks` is not run.
- Swap off; the docker, containerd, snapd, fwupd and jtop services stopped; no scheduled job, other workload or other GPU process running.
- Before each benchmark, after a 120 s settle: a one-minute load average below 1.0, no swap in use and none swapped out, and at least 6.0 x 10^9 bytes of available memory.

The judged runs were made that way: a root setup block applied the first three items (that no other GPU process ran is the author's record, since every log records the other-GPU-process count as `UNREADABLE`), and a run script outside this repository ran the commands below once per run, checked the last item before each benchmark as a QUIET gate, and recorded each gate's terms in `runs-repeat/<run>/quiet.txt`. This repository does not include or run those host scripts. `scripts/host/quiet.py` (the QUIET gate and the sampler) and `scripts/host/host_setup.py` (the root setup) are generic equivalents written after the judged runs, not the scripts that ran them, and no judged run used them. `compare_logs.py` does not check the quieting, so a re-measure made without it is not comparable with the judged runs. The earlier run in `runs/` was made without the quieting; its deviation from the judged median, shown in the repeat block, is up to 28% on the FP16 end-to-end p50; it ran on another day and commit at a one-minute load of 3.9-5.5, so the deviation is an uncontrolled before-and-after difference that also takes in code and library changes: between them the full-frame image check left the timed loop (`1c21082`) and the board's Pillow went from 12.3.0 to 13.0.0, as `constraints/jetson-orin.txt` pinned at each commit. It does not measure what the quieting is worth.

```sh
python scripts/fetch_coco.py annotations
python scripts/fetch_coco.py val
python scripts/fetch_coco.py calibration

python -m peddet_orin.gpu_check
python -m peddet_orin.export
python -m peddet_orin.engine_build --precision fp16
python -m peddet_orin.engine_build --precision int8
python -m peddet_orin.engine_build --precision fp32-trt

python -m peddet_orin.evaluate --precision fp32
python -m peddet_orin.evaluate --precision fp16
python -m peddet_orin.evaluate --precision int8
python -m peddet_orin.evaluate --precision fp32-trt

python -m peddet_orin.benchmark --precision fp32
python -m peddet_orin.benchmark --precision fp16
python -m peddet_orin.benchmark --precision int8
python -m peddet_orin.benchmark --precision fp32-trt

python -m peddet_orin.benchmark --precision fp32 --postprocess deploy
python -m peddet_orin.benchmark --precision fp16 --postprocess deploy
python -m peddet_orin.benchmark --precision int8 --postprocess deploy
python -m peddet_orin.benchmark --precision fp32-trt --postprocess deploy

python scripts/compare_logs.py runs
```

By the author's record, the three `fetch_coco.py` lines were run as written on 2026-11-19 in a fresh clone on the measured board, at a commit whose `scripts/fetch_coco.py` is the released one, and each completed. The commands are in the order the judged runs' logs record, TensorRT FP32 after INT8 at each step and the deployment-style benchmark last. `compare_logs.py` judges the TensorRT FP32 evaluation and benchmark logs and reports the deployment-style ones without judging them. On a board that is not quieted, the power and end-to-end lines may read OUT.

Each command writes a JSON log to `runs/`. A re-measure in this checkout therefore changes what the suite reads: with a new log in `runs/`, `tests/test_readme_table.py` fails, since the README's blocks are built from the committed logs. Run a re-measure in a separate clone, or move its logs out of `runs/` before running the tests. The results table is built from the judged runs committed under `runs-repeat/`, and `python scripts/generate_table.py --write` rebuilds it from them, refusing a benchmark log that times fewer than 3 repeats (`PRD.md` R-12); a re-measure does not run it, since it would show the re-measure's logs in `runs/` as the earlier run. `--data-dir`, `--engine`, `--onnx`, `--output` and `--calibration-dir` override the default locations, and `python -m peddet_orin.<command> --help` lists each command's options. A missing input stops a command with one line naming the command that makes it.

The last command checks the re-measure against the published numbers. `python scripts/compare_logs.py <runs dir>` leaves out the logs committed under `runs/`, `runs-repeat/` and `runs-superseded/`, so the committed logs never stand in for a missing or failed one. It compares the newest eval and benchmark log of each precision with what the results table is built from, the median of the judged runs, applies the tolerances of `PRD.md` R-11 and prints one line per metric. TensorRT FP32 is judged, since the judged runs carry its logs. The deployment-style benchmark is never judged: it prints one REPORTED line per metric when the new logs carry it. It fails a precision with no new log, a dirty newest log, a log run with another runtime or other iteration counts, an INT8 build on another calibration path than the reference's, an eval or benchmark log whose engine sha256 differs from its precision's newest build's or that names no engine, a newest benchmark log whose torch, TensorRT or CUDA version differs from the reference's, and new logs from more than one commit. It also checks engine hashes: each new TensorRT eval and benchmark log must name the engine its precision's newest build log wrote, and a log or build that names none fails. It reads which logs are committed from git, at HEAD, so it runs in a git checkout and stops with one line in an unpacked archive.

## Continuous integration

CI runs on every push to `main`, on every pull request, weekly on a schedule, so an advisory published against an unchanged pin fails the audit, and on demand; cancels a run that a newer push supersedes, and stops a job after 30 minutes. It uses only the CPU, so the GPU and TensorRT tests in `tests/test_gpu.py` skip there, and `-rs` names each skip in the log. `PDORIN_STRICT_SKIPS=1` makes any other skip fail the run. `pip-audit` checks the pinned CPU set and the board's pin file for known vulnerabilities: it strips each local version suffix first, so `torch` 2.14.1+cu130 is audited as 2.14.1, and runs with `--no-deps` on the pin files; the system TensorRT the board uses is in neither file. `tests/test_ci_contract.py` fails if `.github/workflows/ci.yml` runs anything other than this:

<!-- ci:begin -->
```sh
# CI: ubuntu-24.04, Python 3.11 and 3.12
python -m pip install --require-hashes -r requirements/ci-cpu.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m venv --clear .cache/pip-audit
.cache/pip-audit/bin/python -m pip install --require-hashes -r requirements/pip-audit.lock
sed -E 's/^([A-Za-z0-9_.-]+==[^+ ;]+)[+][A-Za-z0-9.]+/\1/' requirements/ci-cpu.txt > .cache/pip-audit/ci-cpu.txt
sed -E 's/^([A-Za-z0-9_.-]+==[^+ ;]+)[+][A-Za-z0-9.]+/\1/' constraints/jetson-orin.txt > .cache/pip-audit/jetson-orin.txt
.cache/pip-audit/bin/pip-audit --strict --no-deps --disable-pip -r .cache/pip-audit/ci-cpu.txt
.cache/pip-audit/bin/pip-audit --strict --no-deps --disable-pip -r .cache/pip-audit/jetson-orin.txt
ruff check .
ruff format --check .
PDORIN_STRICT_SKIPS=1 python -m pytest -rs
python scripts/check_planted_defects.py --cpu-only
```
<!-- ci:end -->

Negative controls: `tests/planted_defects.py` registers 42 planted defects, and `python scripts/check_planted_defects.py` plants each one in a fresh clone and checks that its test fails; the release check requires all 42 to be caught, and `tests/planted_results.txt` holds the recorded lines of the GPU entries and of the CPU run, assembled by the release helper, the author's script outside this repository that assembles the file, whose header states when it ran and that it was made from the files of the commit that last changed it; a test fails once any other file has changed since, so the release check, which sets `PDORIN_RELEASE_CHECK=1`, fails on a commit that changes any other file until the planted run is redone on the board; CI and contributors' runs skip that test. The check does not prove that a run happened: it reads only the file's header and the file's git history, so an edit of the file alone passes it with no run; CI plants every CPU defect with `python scripts/check_planted_defects.py --cpu-only` and fails unless each is caught; the GPU entries' lines are the author's recorded run on the board. They cover 23 of the 41 test files, 18 more each define a named negative test, which `tests/test_negative_controls.py` checks exists in its file, and no test file has neither. `tests/test_negative_controls.py` fails if these numbers differ from the registry and the test files.

## Repository layout

`scripts/generate_tree.py --write` generates this tree from `git ls-files`, and `tests/test_readme_tree.py` fails if it differs from what the script generates.

<!-- tree:start -->
```text
peddet-orin/
├── .github/
│   ├── ISSUE_TEMPLATE/              bug report, feature request and the private security link
│   ├── workflows/
│   │   └── ci.yml                   pip-audit, ruff and the CPU tests on Python 3.11 and 3.12
│   └── PULL_REQUEST_TEMPLATE.md
├── constraints/
│   ├── jetson-orin.lock             the same pins with the sha256 of every file
│   └── jetson-orin.txt              every pin for the board venv
├── manifests/
│   └── calibration_train2017.sha256  the 500 calibration images and their sha256 hashes
├── requirements/
│   ├── ci-cpu.lock                  the same pins with the sha256 of every file
│   ├── ci-cpu.txt                   the CPU-only set for CI
│   ├── pip-audit.lock               pip-audit and its dependencies, with hashes
│   └── pip-audit.txt                the pip-audit release CI audits with
├── runs-attempts/                   the measurement attempts that are not judged runs, each with where it stopped
├── runs-repeat/                     the judged runs the results table is built from, with their QUIET lines and samples
├── runs-superseded/                 earlier judged runs, replaced by those under runs-repeat/ and kept with their files
├── runs/                            the earlier run's JSON logs, one per measurement, shown apart from the judged runs
├── scripts/
│   ├── host/                        generic QUIET gate, sampler and root setup, written after the judged runs; not the scripts that ran them
│   │   ├── host_setup.py            quiets a host for a window as root, and restores it, from a units file
│   │   └── quiet.py                 the QUIET gate and the host sampler
│   ├── check_planted_defects.py     plants each registered defect and checks that its test fails
│   ├── ci_local.sh                  runs the CI matrix locally, one venv per Python version
│   ├── compare_logs.py              checks a re-measure against the judged median within the PRD tolerances
│   ├── fetch_coco.py                fetches COCO 2017 over HTTPS from COCO's S3 bucket, path-style, into data/
│   ├── generate_table.py            writes the README's generated blocks from runs/, runs-repeat/ and runs-superseded/
│   ├── generate_tree.py             writes this tree from git ls-files
│   ├── gpu_bounds.py                measures on the board what the GPU tests' bounds are set from
│   ├── lock_hashes.py               writes a hashed .lock beside a pin file
│   ├── recompute_ap.py              recomputes an eval log's AP from its detections file; logs written from the 0.1.1 release commit on, the committed ones carry none
│   └── recompute_stats.py           recomputes a benchmark log's figures from its raw side file; logs written from the 0.1.1 release commit on, the committed ones carry none
├── src/
│   └── peddet_orin/                 export, engine builds, evaluation, benchmarks and run logs
├── tests/                           CPU tests, GPU and TensorRT tests, and the repository-rule tests
├── .editorconfig
├── .gitignore
├── AGENTS.md                        rules for changes, with the commit-message rule
├── CHANGELOG.md
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── NOTICE
├── PRD.md                           requirements, their acceptance checks and the tests that cover them
├── pyproject.toml
├── QUEUE.md                         open and not-shipped work
├── README.md
└── SECURITY.md
```
<!-- tree:end -->

## Data, weights and licences

This project uses the [COCO 2017](https://cocodataset.org) dataset. The COCO annotations are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), and the images remain under their own Flickr terms ([COCO terms of use](https://cocodataset.org/#termsofuse)). The images and annotations are fetched over HTTPS from COCO's S3 bucket, by its path-style address, into `data/` and are not redistributed here. The calibration set is 500 train2017 images chosen by a fixed seed, listed with their sha256 hashes in `manifests/calibration_train2017.sha256`.

The pretrained weights, torchvision's `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1`, are downloaded at run time and are not committed. They load only after their full sha256 matches the value pinned in `src/peddet_orin/model.py`, and only as tensors. torchvision's code is [BSD 3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE). The weights were trained on COCO; their source is [torchvision's model documentation](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html). Check it and the COCO terms before you reuse them.

The code is licensed under Apache-2.0 AND BSD-3-Clause: Apache-2.0 for the project, and BSD-3-Clause for the parts of `src/peddet_orin/postprocess.py` that follow torchvision's code, as its SPDX line says. See `LICENSE` and `NOTICE`.

## How to cite

If you use this software or its results, please cite it. `CITATION.cff` holds the metadata, and GitHub shows it as "Cite this repository". In BibTeX:

```bibtex
@software{hassan_peddet_orin_2026,
  author  = {Hassan, Ahmed},
  title   = {peddet-orin: person detection on a Jetson Orin Nano Super in FP32, FP16 and INT8},
  version = {0.1.1},
  year    = {2026},
  url     = {https://github.com/ahmedhassan-eng/peddet-orin}
}
```

## Authorship

Built by AI coding agents under my direction ([Ahmed Hassan](https://github.com/ahmedhassan-eng)); I set the acceptance bar and made the release decisions.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), the [code of conduct](CODE_OF_CONDUCT.md) and, for vulnerabilities, [SECURITY.md](SECURITY.md).
