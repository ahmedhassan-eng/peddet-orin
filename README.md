# peddet-orin

Person detection on a Jetson Orin Nano Super with torchvision's SSDLite320 MobileNetV3-Large and its COCO weights, measured at three precisions: FP32 in PyTorch, and FP16 and INT8 in TensorRT. The benchmark reports accuracy on COCO 2017 val, latency, frames per second, memory and power. The headline numbers are the median of the judged runs' JSON logs under `runs-repeat/`, which also hold each run's QUIET gate and sampler files; the earlier run's logs are in `runs/`, and a test fails if the table and the logs disagree. [Results](#results) and [Limitations](#limitations) follow; the hardware and software, how the judged runs were checked, the Quickstart and how to reproduce the results come after them.

## Results

Latencies are milliseconds, at batch size 1. "Engine" covers inference, including the copies to and from device memory. "End-to-end" adds image decoding, preprocessing and postprocessing. FPS is 1 / mean latency over the serial batch-1 iterations, not a sustained throughput with several requests in flight. Power is the mean board input over the whole timed loop, which ran at the end-to-end rate, so it sits with the end-to-end columns: energy per frame is each run's power divided by its end-to-end FPS, and the board never ran at the engine FPS while power was read. All precisions use the same postprocess. Each benchmark pools three timed repeats of its loop, and the notes give the range of the per-repeat medians. Accuracy is person AP on the 5000 COCO val2017 images, from pycocotools with `catIds=[1]`.

<!-- results:begin -->
| Device and precision | Engine p50/p90/p99 (ms) | Engine FPS | End-to-end p50/p90/p99 (ms) | End-to-end FPS | Avg power VDD_IN (W) | End-to-end energy per frame (J) | Peak process RSS (MiB) | Person AP@[.5:.95] | Person AP50 | INT8 path |
|---|---|---|---|---|---|---|---|---|---|---|
| GPU, PyTorch FP32 | 29.8 / 30.2 / 31.9 | 33.5 | 87.1 / 99.9 / 106.2 | 11.5 | 7.6 | 0.655 | 1219 | 28.01 | 49.50 | - |
| GPU, TensorRT FP16 | 3.8 / 3.9 / 4.0 | 263.0 | 60.6 / 73.8 / 80.4 | 16.7 | 7.1 | 0.428 | 1195 | 28.01 | 49.49 | - |
| GPU, TensorRT INT8 | 3.6 / 3.7 / 3.8 | 276.2 | 64.8 / 77.5 / 84.5 | 15.7 | 7.1 | 0.453 | 1102 | 26.74 | 48.14 | IInt8EntropyCalibrator2 |

- Each cell is the median over the n = 3 quieted runs in `runs-repeat/` (run3, run4, run5), this release's count; 0.2.0 plans 5 or more runs.
- The judged runs were made at commit `025171d` with package version 0.1.0, as their logs record.
- Memory, swap and temperature are recorded for the judged runs only, over each whole benchmark job, warm-up included; the earlier runs have no such record.
- INT8 person AP differs from FP16 by -1.27 points.
- INT8 person AP is 26.74 in each of the 3 judged runs and 26.83 in the earlier run; each run builds its own INT8 engine, so each score comes from a separate build.
- The judged runs' power mode was not read: 9 of their 9 benchmark logs record `UNREADABLE`, which a log records when `nvpmodel` is not found on PATH or `nvpmodel -q` fails, and the host's run script adds no system sbin directory to PATH. The evidence the logs hold is the clock maxima of every judged benchmark: GPU 1020 MHz, CPU policy0 1728 MHz, policy4 1728 MHz, the same as in each of the earlier run's 3 benchmarks, whose power mode reads `NV Power Mode: MAXN_SUPER 2`.
- FP32 min-max over the 3 runs: ap 28.01-28.01, ap50 49.50-49.50, engine_p50_ms 29.31-30.21, end_to_end_p50_ms 86.28-87.76, avg_power_w 7.56-7.58.
- FP16 min-max over the 3 runs: ap 28.00-28.01, ap50 49.46-49.51, engine_p50_ms 3.78-3.84, end_to_end_p50_ms 59.67-61.22, avg_power_w 7.13-7.18.
- INT8 min-max over the 3 runs: ap 26.74-26.74, ap50 48.14-48.14, engine_p50_ms 3.60-3.63, end_to_end_p50_ms 63.33-66.22, avg_power_w 7.05-7.10.
- run3: FP32 benchmark: power mode `UNREADABLE`, GPU clock governor `performance`, at max 1020 MHz in 1305 of 1305 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 1299 of 1305 samples; policy4 governor `schedutil`, at max 1728 MHz in 4 of 1305 samples, one-minute CPU load before timing: 0.5, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run3: FP32 p50 across repeats: engine 29.72-29.81 ms over 3 repeats, end-to-end 87.54-88.14 ms over 3 repeats.
- run3: FP16 benchmark: power mode `UNREADABLE`, GPU clock governor `performance`, at max 1020 MHz in 888 of 888 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 884 of 888 samples; policy4 governor `schedutil`, at max 1728 MHz in 3 of 888 samples, one-minute CPU load before timing: 0.5, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run3: FP16 p50 across repeats: engine 3.79-3.80 ms over 3 repeats, end-to-end 59.35-60.08 ms over 3 repeats.
- run3: INT8 benchmark: power mode `UNREADABLE`, GPU clock governor `performance`, at max 1020 MHz in 960 of 960 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 956 of 960 samples; policy4 governor `schedutil`, at max 1728 MHz in 10 of 960 samples, one-minute CPU load before timing: 0.5, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run3: INT8 p50 across repeats: engine 3.61-3.62 ms over 3 repeats, end-to-end 64.45-65.51 ms over 3 repeats.
- run3: PyTorch GPU check: usable True, device compute capability 8.7, wheel slices sm_80, sm_90, sm_100, sm_110, sm_120. PyTorch emitted an unsupported-GPU warning at first CUDA use: "Found GPU0 Orin which is of compute capability (CC) 8.7.". FP32 ran with cudnn.allow_tf32 False and cuda.matmul.allow_tf32 False.
- run4: FP32 benchmark: power mode `UNREADABLE`, GPU clock governor `performance`, at max 1020 MHz in 1294 of 1294 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 1286 of 1294 samples; policy4 governor `schedutil`, at max 1728 MHz in 5 of 1294 samples, one-minute CPU load before timing: 0.6, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run4: FP32 p50 across repeats: engine 30.18-30.28 ms over 3 repeats, end-to-end 86.87-87.25 ms over 3 repeats.
- run4: FP16 benchmark: power mode `UNREADABLE`, GPU clock governor `performance`, at max 1020 MHz in 913 of 913 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 902 of 913 samples; policy4 governor `schedutil`, at max 1728 MHz in 13 of 913 samples, one-minute CPU load before timing: 0.3, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run4: FP16 p50 across repeats: engine 3.78-3.78 ms over 3 repeats, end-to-end 60.72-61.39 ms over 3 repeats.
- run4: INT8 benchmark: power mode `UNREADABLE`, GPU clock governor `performance`, at max 1020 MHz in 983 of 983 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 979 of 983 samples; policy4 governor `schedutil`, at max 1728 MHz in 2 of 983 samples, one-minute CPU load before timing: 0.8, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run4: INT8 p50 across repeats: engine 3.60-3.61 ms over 3 repeats, end-to-end 65.82-66.65 ms over 3 repeats.
- run4: PyTorch GPU check: usable True, device compute capability 8.7, wheel slices sm_80, sm_90, sm_100, sm_110, sm_120. PyTorch emitted an unsupported-GPU warning at first CUDA use: "Found GPU0 Orin which is of compute capability (CC) 8.7.". FP32 ran with cudnn.allow_tf32 False and cuda.matmul.allow_tf32 False.
- run5: FP32 benchmark: power mode `UNREADABLE`, GPU clock governor `performance`, at max 1020 MHz in 1283 of 1283 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 1276 of 1283 samples; policy4 governor `schedutil`, at max 1728 MHz in 10 of 1283 samples, one-minute CPU load before timing: 0.3, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run5: FP32 p50 across repeats: engine 29.26-29.37 ms over 3 repeats, end-to-end 85.96-86.48 ms over 3 repeats.
- run5: FP16 benchmark: power mode `UNREADABLE`, GPU clock governor `performance`, at max 1020 MHz in 902 of 902 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 895 of 902 samples; policy4 governor `schedutil`, at max 1728 MHz in 5 of 902 samples, one-minute CPU load before timing: 0.7, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run5: FP16 p50 across repeats: engine 3.83-3.84 ms over 3 repeats, end-to-end 60.23-61.03 ms over 3 repeats.
- run5: INT8 benchmark: power mode `UNREADABLE`, GPU clock governor `performance`, at max 1020 MHz in 936 of 936 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 930 of 936 samples; policy4 governor `schedutil`, at max 1728 MHz in 5 of 936 samples, one-minute CPU load before timing: 0.8, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- run5: INT8 p50 across repeats: engine 3.62-3.63 ms over 3 repeats, end-to-end 62.96-63.64 ms over 3 repeats.
- run5: PyTorch GPU check: usable True, device compute capability 8.7, wheel slices sm_80, sm_90, sm_100, sm_110, sm_120. PyTorch emitted an unsupported-GPU warning at first CUDA use: "Found GPU0 Orin which is of compute capability (CC) 8.7.". FP32 ran with cudnn.allow_tf32 False and cuda.matmul.allow_tf32 False.
<!-- results:end -->

The FP32 row runs PyTorch on the GPU with TF32 turned off for cuDNN and matmul, so it is FP32 throughout. PyTorch warns on first CUDA use that its published 2.14.1+cu130 wheels do not support this board's compute capability, 8.7; the notes quote the warning's first line, the `gpu_check` log holds all of it, and the code does not hide it. Which kernel image the driver runs is not reported. `python -m peddet_orin.gpu_check` decides by running whether the GPU build is usable: it compares a matmul and NMS against the CPU, and `tests/test_gpu.py` compares a full forward pass.

Peak process RSS does not fully capture memory allocated on the device. System RAM rise, the highest system-wide RAM during timing minus the mean just before, is in the logs but not in the table: it includes any other activity on the board and does not reproduce between runs.

<!-- int8:begin -->
Over the median of the 3 quieted runs, the INT8 engine p50 is lower than FP16's, 3.61 against 3.80 ms, and its end-to-end p50 is higher than FP16's, 64.83 against 60.60 ms; FP16's and INT8's per-repeat end-to-end ranges in the notes do not overlap in any of the 3 runs. The FP16 engine p50 is 6% of its end-to-end p50; the rest is image decoding, preprocessing and postprocessing on the CPU. INT8 yields more person detections after NMS on val2017 than FP16, 398837 against 365070, the `detections` count in the eval logs. Whether more detections give the postprocess more work during the benchmark is a hypothesis: no profile here separates it from run-to-run variation.
<!-- int8:end -->

## Limitations

- One model, one dataset and one board. The numbers do not transfer to other Jetson modules, power modes or JetPack versions.
- The export uses PyTorch's TorchScript-based ONNX exporter, which PyTorch has deprecated; `pyproject.toml` bounds torch below 2.15 for that reason.
- CI runs on x86_64 CPUs only. The aarch64 build, the GPU tests and TensorRT run only on the Jetson, by hand, before a release.
- Batch size 1 only.
- FP32 runs in PyTorch and FP16 and INT8 in TensorRT, so the gap between FP32 and the other two is runtime and precision together, not precision alone.
- `jetson_clocks` is not run, memory clocks are not read, and a tegrastats reader thread runs in the timed process. For the judged runs, the QUIET gates and the measurement window excluded what the repeat block and the Provenance notes name: the board's other scheduled jobs and loop services, sessions over their CPU budget or not allowed, swap use and swap-outs, available memory under the bound, and a load1 at or above the gate's bound. They did not exclude the operating system's own processes or the tegrastats reader, so latencies still vary between runs, as the min-max notes show. The earlier run in `runs/` had no such gate.
- The system RAM rise in each benchmark log is reported per run and not judged.
- Power is the average of `tegrastats` VDD_IN readings during the timed end-to-end loop. It is the whole board's input power, not the GPU's alone, and no power was read for inference alone, so there is no energy figure at the engine rate.
- INT8 is calibrated on one set, 500 seeded train2017 images. INT8 accuracy with another calibration set was not measured here.
- Person-only fine-tuning was tried and not shipped. See `QUEUE.md`.

## Hardware and software

The results were measured with this stack. Each run log records the Jetson Linux release string, the power mode (or `UNREADABLE` when it could not be read) and the Python, PyTorch, torchvision, CUDA, TensorRT, ONNX, ONNX Runtime and peddet-orin versions it ran with; from 0.1.1 on it also records the numpy, Pillow, pycocotools and libjpeg versions, which the judged runs' logs predate. The board and JetPack rows come from the board's documentation, not from the logs.

| Component | Version |
|---|---|
| Board | Jetson Orin Nano Super, GPU compute capability 8.7 |
| JetPack | 7, Jetson Linux R39 revision 2.1 |
| Power mode | Not read for the judged runs; the results notes give the clock evidence. `MAXN_SUPER` (mode 2) in the earlier run's logs |
| CUDA | 13.0, from the PyTorch `cu130` wheels |
| TensorRT | 10.16.2.10, from the JetPack system Python |
| Python | 3.12 |
| PyTorch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 |
| ONNX / ONNX Runtime | 1.23.2 / 1.31.1 |

The judged runs used Pillow 12.3.0. The pins name 13.0.0, the first release with the fixes pip-audit lists, and the numbers were not re-measured with it; JPEG decoding by Pillow is inside the end-to-end timings, not the engine timings.

The CPU tests also run on any Linux machine with Python 3.11 or 3.12. See "Continuous integration" below.

## Judged runs

How the median in [Results](#results) was judged, with the gate, memory, swap, temperature and sampler records of each judged run, all generated from the logs and host records in `runs-repeat/`.

<!-- repeats:begin -->
R-11 is judged among the 3 quieted runs run3, run4, run5 in `runs-repeat/`, at commit `025171d`: a metric is IN when every judged run is within its tolerance of their median, and OUT otherwise. The tolerances are the `PRD.md` values in `scripts/compare_logs.py`, unchanged. The spread comes from 3 runs on one board, which is not a population. avg_power_w is the board's input power, VDD_IN, as its sensor measures it. The earlier run in `runs/`, at `17a1ee2`, gave 0.1.0's table; it is shown apart with the deviation from the median and not judged.

Each judged run's QUIET gate ran at its start and before each benchmark, after a settle; each gate passed on the try shown, and its passing line records: 0 scheduled jobs or path units active other than the ones allowed by name, 0 sessions over their CPU budget or not allowed, 0 loop services active, 0 kB of swap used, 0 pages swapped out during the settle; 3 scheduled jobs allowed by name; load1 below 1.0; available memory at least 5859375 kB.

| Run | Gate | Try | load1 | Available memory (kB) |
|---|---|---|---|---|
| run3 | start | 1 | 0.04 | 6451156 |
| run3 | bench_fp32 | 1 | 0.38 | 6353292 |
| run3 | bench_fp16 | 1 | 0.50 | 6346020 |
| run3 | bench_int8 | 1 | 0.26 | 6353664 |
| run4 | start | 1 | 0.03 | 6358472 |
| run4 | bench_fp32 | 1 | 0.41 | 6233060 |
| run4 | bench_fp16 | 1 | 0.19 | 6218600 |
| run4 | bench_int8 | 1 | 0.24 | 6234984 |
| run5 | start | 1 | 0.09 | 6236208 |
| run5 | bench_fp32 | 1 | 0.17 | 6252984 |
| run5 | bench_fp16 | 1 | 0.20 | 6255428 |
| run5 | bench_int8 | 1 | 0.57 | 6252944 |

- run3: quieted, every QUIET gate passed; one-minute CPU load before timing 0.5-0.5.
- run4: quieted, every QUIET gate passed; one-minute CPU load before timing 0.3-0.8.
- run5: quieted, every QUIET gate passed; one-minute CPU load before timing 0.3-0.8.
- earlier run (`runs/`): not quieted, load before every benchmark at or above the QUIET gate's 1.0; one-minute CPU load before timing 3.9-5.5.

| Metric | Precision | Median | run3 | run4 | run5 | Min-max | Spread | Tolerance | R-11 | earlier run |
|---|---|---|---|---|---|---|---|---|---|---|
| ap | FP32 | 28.01 | 28.01 | 28.01 | 28.01 | 28.01-28.01 | 0.00 | 0.10 | IN | 28.01 (0.00) |
| ap50 | FP32 | 49.50 | 49.50 | 49.50 | 49.50 | 49.50-49.50 | 0.00 | 0.10 | IN | 49.50 (0.00) |
| ap | FP16 | 28.01 | 28.01 | 28.00 | 28.01 | 28.00-28.01 | 0.01 | 0.10 | IN | 28.00 (-0.01) |
| ap50 | FP16 | 49.49 | 49.46 | 49.51 | 49.49 | 49.46-49.51 | 0.05 | 0.10 | IN | 49.48 (-0.01) |
| ap | INT8 | 26.74 | 26.74 | 26.74 | 26.74 | 26.74-26.74 | 0.00 | 0.50 | IN | 26.83 (+0.09) |
| ap50 | INT8 | 48.14 | 48.14 | 48.14 | 48.14 | 48.14-48.14 | 0.00 | 0.50 | IN | 48.15 (+0.01) |
| engine_p50_ms | FP32 | 29.78 | 29.78 | 30.21 | 29.31 | 29.31-30.21 | 0.90 | 10% | IN | 30.31 (+1.8%) |
| end_to_end_p50_ms | FP32 | 87.10 | 87.76 | 87.10 | 86.28 | 86.28-87.76 | 1.48 | 15% | IN | 96.61 (+10.9%) |
| avg_power_w | FP32 | 7.56 | 7.56 | 7.56 | 7.58 | 7.56-7.58 | 0.02 | 10% | IN | 8.63 (+14.2%) |
| engine_p50_ms | FP16 | 3.80 | 3.80 | 3.78 | 3.84 | 3.78-3.84 | 0.06 | 10% | IN | 3.88 (+2.2%) |
| end_to_end_p50_ms | FP16 | 60.60 | 59.67 | 61.22 | 60.60 | 59.67-61.22 | 1.54 | 15% | IN | 75.13 (+24.0%) |
| avg_power_w | FP16 | 7.14 | 7.18 | 7.14 | 7.13 | 7.13-7.18 | 0.05 | 10% | IN | 8.75 (+22.6%) |
| engine_p50_ms | INT8 | 3.61 | 3.61 | 3.60 | 3.63 | 3.60-3.63 | 0.03 | 10% | IN | 3.69 (+2.1%) |
| end_to_end_p50_ms | INT8 | 64.83 | 64.83 | 66.22 | 63.33 | 63.33-66.22 | 2.89 | 15% | IN | 78.33 (+20.8%) |
| avg_power_w | INT8 | 7.08 | 7.10 | 7.05 | 7.08 | 7.05-7.10 | 0.05 | 10% | IN | 8.66 (+22.4%) |

R-11 is met for every judged metric among the quieted runs.

Each judged run's state before any measurement, from the first sample of its first GPU job:

- run3 first sample, 2026-11-15T18:50:58Z: available memory 6293 MiB, swap used 0 kB, temperatures (C) cpu-thermal 49.8, gpu-thermal 50.3, soc0-thermal 49.8, soc1-thermal 50.5, soc2-thermal 48.9, tj-thermal 50.6.
- run4 first sample, 2026-11-15T19:43:30Z: available memory 6203 MiB, swap used 0 kB, temperatures (C) cpu-thermal 50.5, gpu-thermal 50.7, soc0-thermal 50.0, soc1-thermal 51.0, soc2-thermal 49.1, tj-thermal 51.0.
- run5 first sample, 2026-11-15T20:35:52Z: available memory 6084 MiB, swap used 0 kB, temperatures (C) cpu-thermal 50.7, gpu-thermal 51.0, soc0-thermal 50.3, soc1-thermal 51.5, soc2-thermal 49.5, tj-thermal 51.5.

Memory, swap and temperature of each judged run's benchmark jobs, sampled every 1 s by the host's sampler script, a separate process, over the whole job, warm-up included; runs shown apart had no sampler. The thermal zones are the ones whose readings the records hold: cpu-thermal, gpu-thermal, soc0-thermal, soc1-thermal, soc2-thermal, tj-thermal.

| Run | Precision | Swap used at start (kB) | Pages swapped out | Lowest available memory (MiB) | Highest temperature (C) |
|---|---|---|---|---|---|
| run3 | FP32 | 0 | 0 | 5360 | cpu-thermal 53.0, gpu-thermal 53.4, soc0-thermal 52.3, soc1-thermal 53.3, soc2-thermal 51.7, tj-thermal 53.4 |
| run3 | FP16 | 0 | 0 | 5406 | cpu-thermal 52.6, gpu-thermal 52.8, soc0-thermal 51.7, soc1-thermal 52.7, soc2-thermal 51.2, tj-thermal 52.8 |
| run3 | INT8 | 0 | 0 | 5487 | cpu-thermal 52.5, gpu-thermal 52.8, soc0-thermal 51.8, soc1-thermal 52.8, soc2-thermal 51.2, tj-thermal 52.7 |
| run4 | FP32 | 0 | 0 | 5261 | cpu-thermal 53.6, gpu-thermal 54.0, soc0-thermal 52.8, soc1-thermal 53.8, soc2-thermal 52.2, tj-thermal 54.0 |
| run4 | FP16 | 0 | 0 | 5249 | cpu-thermal 52.9, gpu-thermal 53.2, soc0-thermal 52.1, soc1-thermal 53.2, soc2-thermal 51.5, tj-thermal 53.2 |
| run4 | INT8 | 0 | 0 | 5383 | cpu-thermal 52.7, gpu-thermal 53.0, soc0-thermal 51.9, soc1-thermal 52.9, soc2-thermal 51.4, tj-thermal 53.0 |
| run5 | FP32 | 0 | 0 | 5257 | cpu-thermal 53.9, gpu-thermal 54.3, soc0-thermal 53.1, soc1-thermal 54.1, soc2-thermal 52.6, tj-thermal 54.3 |
| run5 | FP16 | 0 | 0 | 5359 | cpu-thermal 53.1, gpu-thermal 53.6, soc0-thermal 52.4, soc1-thermal 53.5, soc2-thermal 51.9, tj-thermal 53.6 |
| run5 | INT8 | 0 | 0 | 5385 | cpu-thermal 52.9, gpu-thermal 53.2, soc0-thermal 52.1, soc1-thermal 53.2, soc2-thermal 51.6, tj-thermal 53.2 |

Each judged run's GPU jobs, from the samples taken during each job, from its start to its end:

| Run | Job | Lowest available memory (MiB) | Pages swapped out |
|---|---|---|---|
| run3 | bench_fp16 | 5406 | 0 |
| run3 | bench_fp32 | 5360 | 0 |
| run3 | bench_int8 | 5487 | 0 |
| run3 | build_fp16 | 4841 | 0 |
| run3 | build_int8 | 4610 | 0 |
| run3 | eval_fp16 | 4820 | 0 |
| run3 | eval_fp32 | 4752 | 0 |
| run3 | eval_int8 | 4884 | 0 |
| run3 | export | 5702 | 0 |
| run3 | gpu_check | 5723 | 0 |
| run4 | bench_fp16 | 5249 | 0 |
| run4 | bench_fp32 | 5261 | 0 |
| run4 | bench_int8 | 5383 | 0 |
| run4 | build_fp16 | 4721 | 0 |
| run4 | build_int8 | 4568 | 0 |
| run4 | eval_fp16 | 4714 | 0 |
| run4 | eval_fp32 | 4651 | 0 |
| run4 | eval_int8 | 4759 | 0 |
| run4 | export | 5599 | 0 |
| run4 | gpu_check | 5638 | 0 |
| run5 | bench_fp16 | 5359 | 0 |
| run5 | bench_fp32 | 5257 | 0 |
| run5 | bench_int8 | 5385 | 0 |
| run5 | build_fp16 | 4774 | 0 |
| run5 | build_int8 | 4527 | 0 |
| run5 | eval_fp16 | 4721 | 0 |
| run5 | eval_fp32 | 4665 | 0 |
| run5 | eval_int8 | 4774 | 0 |
| run5 | export | 5486 | 0 |
| run5 | gpu_check | 5551 | 0 |

RAM was not a constraint in runs 3-5: the lowest available memory during any job was 4527 MiB, and 0 pages were swapped out.
<!-- repeats:end -->

### Provenance

- Each cell is the median over the judged quieted runs listed in the notes, on one board; the notes name the commit and package version their logs record. A benchmark pools three timed repeats of its loop. That commit's `src/` is the same as at `17a1ee2`, where 0.1.0's table was measured from one run on a host that was not quieted, and 0.1.1's changes to `src/`, each listed in `CHANGELOG.md`, change no computation: among them, it loads ONNX Runtime where it is used, and each run log also records the numpy, Pillow, pycocotools and libjpeg versions. So the code that produced the measured numbers computes what the released code computes. The judged runs ran the code before this change, with ONNX Runtime loaded in the measuring process; the peak process RSS in the table includes it and was not re-measured; the timings and accuracy do not depend on it. The run log still loads ONNX Runtime to record its version, so a benchmark run of the released code loads it too, once its input checks have passed.
- Every log records its commit, whether the tree was dirty, the start and end times, the Jetson Linux release string, and the Python, PyTorch, torchvision, CUDA, TensorRT, ONNX, ONNX Runtime and peddet-orin versions, and from 0.1.1 on the numpy, Pillow, pycocotools and libjpeg versions. The table uses only clean logs, all from one commit.
- The power mode in each benchmark note is what `nvpmodel -q` reported, or `UNREADABLE` when it was not read; the notes say which for the judged runs. `jetson_clocks` was not run, since it needs root. During timing a separate process samples the GPU's devfreq clock and each CPU frequency policy's clock every 200 ms; the notes give each governor and how many samples were at that clock's maximum. Memory clocks are not read.
- The one-minute load average in the notes is read just before timing starts, so it shows what else was running then. The number of other GPU processes could not be read, and is recorded as `UNREADABLE`.
- The measurements ran one at a time, at normal priority, while no other GPU or benchmark job ran on the board.
- `tests/test_readme_table.py` fails if the results block differs from what `scripts/generate_table.py` makes of the logs.
- The judged runs are run3, run4 and run5 of one measurement window, which started at 2026-11-15T18:46Z; the memory bound, 6.0e9 bytes, was set before any judged run. Nine earlier attempts did not run or did not finish, and their logs are not committed. By reason: 2 stopped at the window's check that no automated session was running, 4 at its check of available memory, and 3 at a run's QUIET gate (swap in use in all 3, available memory under the bound in 1 of them as well).
- The earlier attempts, the root setup block and the setting of the memory bound are host records outside this repository.
- The bound was 6.5e9 bytes in the earlier attempts, four of which stopped on it, and was lowered to 6.0e9 before the judged runs; no judged run was made under another bound.
- The window was set up by a root shell block that is not in this repository. It turned swap off, stopped the docker, containerd, snapd, fwupd and jtop services and every automated session, and logged for each judged run 0 swap entries, swappiness 1, 0 pages swapped out and 0 OOM kills.
- The judged runs were run by three host scripts outside this repository: a run script that ran each run and its QUIET gates (sha256 `bef51001f5f3ff431bccb60eabf294f2afcb41aefd49daffa45b86dae45628d5`), a job script that ran each GPU job under a lock with the sampler alongside (`2f36e2682c0faed445168adeed5b046820a8494307a6600113075c1f88e5a25d`), and the sampler `sysmon.sh` (`aa30652bf11859de67420a4e6b64f53cc9348254f03b8a5dea424b94ed2a93ce`). The sampler's sha256 heads every judged sampler file, and `scripts/generate_table.py` refuses a sampler file that names another; the run and job scripts' sha256s are not recorded in the logs. What the scripts did is what the gate terms and the repeat block state, generated from the logs. The scripts are specific to this host and do not run elsewhere.

## Quickstart

This needs a Jetson Orin on JetPack 7, with these system packages, at the versions the measured board had: `python3.12-venv` 3.12.3 for the venv; from JetPack, `python3-libnvinfer` 10.16.2.10 for TensorRT, and `nvidia-l4t-nvpmodel` and `nvidia-l4t-tools` 39.2.1 for `nvpmodel` and `tegrastats`. The board's `python3-numpy` 1.26.4 equals the numpy pin, so the venv uses it; without it, pip installs the pinned numpy. Run the commands from the repository root:

```sh
git clone https://github.com/ahmedhassan-eng/peddet-orin.git
cd peddet-orin
export TORCH_HOME="$PWD/.cache/torch" PIP_CACHE_DIR="$PWD/.cache/pip" XDG_CACHE_HOME="$PWD/.cache/xdg"
export MPLCONFIGDIR="$PWD/.cache/mpl" CUDA_CACHE_PATH="$PWD/.cache/cuda"
python3.12 -m venv .venv --system-site-packages
. .venv/bin/activate
pip install -c constraints/jetson-orin.txt --extra-index-url https://download.pytorch.org/whl/cu130 "torch==2.14.1+cu130" "torchvision==0.29.1+cu130"
pip install -c constraints/jetson-orin.txt -e ".[test]"
```

The venv uses `--system-site-packages` because TensorRT 10.16.2.10 comes from the JetPack system Python. `constraints/jetson-orin.txt` pins every package that pip installs into the venv; TensorRT, and any package the system Python already provides at a compatible version, are used from the system.

The package runs from this source checkout, not from an installed wheel. Its commands find the checkout two levels above the package, or the directory `PDORIN_ROOT` names, and refuse to start anywhere else. `python -m peddet_orin` lists the commands; each is also installed as a `peddet-orin-<command>` script.

Run the tests. They need no network. On the Jetson, the GPU and TensorRT tests in `tests/test_gpu.py` run too, so the suite takes about fifteen minutes while it builds TensorRT engines. `PDORIN_REQUIRE_GPU=1` makes a GPU or TensorRT test that skips, for want of a CUDA device or TensorRT, fail the run instead:

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

```sh
python scripts/fetch_coco.py annotations
python scripts/fetch_coco.py val
python scripts/fetch_coco.py calibration

python -m peddet_orin.gpu_check
python -m peddet_orin.export
python -m peddet_orin.engine_build --precision fp16
python -m peddet_orin.engine_build --precision int8

python -m peddet_orin.evaluate --precision fp32
python -m peddet_orin.evaluate --precision fp16
python -m peddet_orin.evaluate --precision int8

python -m peddet_orin.benchmark --precision fp32
python -m peddet_orin.benchmark --precision fp16
python -m peddet_orin.benchmark --precision int8

python scripts/generate_table.py --write
```

Each command writes a JSON log to `runs/`, and the last one rewrites the README's generated blocks. With quieted runs committed under `runs-repeat/`, the results table is their median; without them, it is the newest clean logs in `runs/`, which must all come from one commit. `--data-dir`, `--engine`, `--onnx`, `--output` and `--calibration-dir` override the default locations, and `python -m peddet_orin.<command> --help` lists each command's options. A missing input stops a command with one line naming the command that makes it.

To check a re-measure against the published numbers, write its logs to another directory, for example a fresh clone's `runs/`, and compare: `python scripts/compare_logs.py <that runs dir>`. It compares the newest clean eval and benchmark log of each precision with what the results table is built from, the median of the judged runs in `runs-repeat/`, applies the tolerances of `PRD.md` R-11 and prints one line per metric. A set without a clean eval and benchmark log for every precision fails. Set `PDORIN_DEBUG=1` to turn on diagnostic logging and profiling.

## Continuous integration

CI runs on every push to `main` and on every pull request, cancels a run that a newer push supersedes, and stops a job after 30 minutes. It uses only the CPU, so the GPU and TensorRT tests in `tests/test_gpu.py` skip there, and `-rs` names each skip in the log. `PDORIN_STRICT_SKIPS=1` makes any other skip fail the run. `pip-audit` checks the pinned CPU set for known vulnerabilities. `tests/test_ci_contract.py` fails if `.github/workflows/ci.yml` runs anything other than this:

<!-- ci:begin -->
```sh
# CI: ubuntu-latest, Python 3.11 and 3.12
python -m pip install -r requirements/ci-cpu.txt
python -m pip install --no-deps -e .
python -m venv --clear .cache/pip-audit
.cache/pip-audit/bin/python -m pip install pip-audit==2.10.1
.cache/pip-audit/bin/pip-audit --no-deps --disable-pip -r requirements/ci-cpu.txt
ruff check .
PDORIN_STRICT_SKIPS=1 python -m pytest -rs
```
<!-- ci:end -->

Negative controls: `tests/planted_defects.py` registers 37 planted defects, and `python scripts/check_planted_defects.py` plants each one in a fresh clone and checks that its test fails; the release check requires all 37 to be caught. They cover 21 of the 36 test files, 14 more carry in-file negative tests, and `QUEUE.md` names the rest. `tests/test_negative_controls.py` fails if these numbers differ from the registry and the test files.

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
│   └── jetson-orin.txt              every pin for the Jetson venv
├── manifests/
│   └── calibration_train2017.sha256  the 500 calibration images and their sha256 hashes
├── requirements/
│   └── ci-cpu.txt                   the CPU-only set for CI
├── runs-repeat/                     further full runs, one directory each, with their QUIET lines and samples
├── runs/                            one JSON log per measurement; the results table is built from them
├── scripts/
│   ├── check_planted_defects.py     plants each registered defect and checks that its test fails
│   ├── ci_local.sh                  runs the CI matrix locally, one venv per Python version
│   ├── compare_logs.py              checks a re-measure against the judged median within the PRD tolerances
│   ├── fetch_coco.py                fetches COCO 2017 over HTTPS from the official URLs into data/
│   ├── generate_table.py            writes the README's generated blocks from runs/ and runs-repeat/
│   └── generate_tree.py             writes this tree from git ls-files
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

This project uses the [COCO 2017](https://cocodataset.org) dataset. The COCO annotations are licensed under CC BY 4.0, and the images remain under their own Flickr terms. The images and annotations are fetched over HTTPS from the official COCO URLs into `data/` and are not redistributed here. The calibration set is 500 train2017 images chosen by a fixed seed, listed with their sha256 hashes in `manifests/calibration_train2017.sha256`.

The pretrained weights, torchvision's `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1`, are downloaded at run time and are not committed. They load only after their full sha256 matches the value pinned in `src/peddet_orin/model.py`, and only as tensors. torchvision's code is BSD 3-Clause. The weights were trained on COCO, so check torchvision's documentation and the COCO terms before you reuse them.

The code is licensed under Apache-2.0. See `LICENSE` and `NOTICE`.

## How to cite

If you use this software or its results, please cite it. `CITATION.cff` holds the metadata, and GitHub shows it as "Cite this repository". In BibTeX:

```bibtex
@software{hassan_peddet_orin_2026,
  author  = {Hassan, Ahmed},
  title   = {peddet-orin: person detection on the Jetson Orin in FP32, FP16 and INT8},
  version = {0.1.1},
  year    = {2026},
  url     = {https://github.com/ahmedhassan-eng/peddet-orin}
}
```

## Authorship

Built with AI coding agents under the direction of [Ahmed Hassan](https://github.com/ahmedhassan-eng). The design, the benchmark method and the review of every result are his.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), the [code of conduct](CODE_OF_CONDUCT.md) and, for vulnerabilities, [SECURITY.md](SECURITY.md).
