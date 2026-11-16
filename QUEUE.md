# Queue

| Item | Status |
|---|---|
| Person-only fine-tuning on COCO train2017 | not shipped: the one run that fit the time budget scored below the FP32 reference; it left no run log and its code is not in this repository, so no number is published for it |
| Locked-clock benchmark with `jetson_clocks`, which needs root; today the GPU and CPU clocks are sampled during timing but not locked | planned for 0.2.0 |
| Count of other GPU processes on this board, which NVML does not report here | planned for 0.2.0 |
| Move the `tegrastats` reader out of the timed process, as the clock reader already is | planned for 0.2.0 |
| Drop `inferred_slice` from the `gpu_check` log, or record `none` when PyTorch's warning excludes this capability | planned for 0.2.0 |
| Check each TensorRT I/O tensor's dtype is FLOAT before allocating its buffer | planned for 0.2.0 |
| One sentinel for "no entropy failure" in build logs, enforced by the log check | planned for 0.2.0 |
| Format the tree with `ruff format` and check it in CI | planned for 0.2.0 |
| Re-measure with the pinned set, the peak process RSS included, and record the Pillow, numpy and libjpeg versions in every run log | planned for 0.1.2 |
| The job script puts the system sbin directory on PATH, or fails when the power mode is `UNREADABLE` | planned for 0.1.2 |
| A guard in the test suite that refuses the GPU tests unless the run holds the board's GPU lock; today none exists | planned for 0.1.2 |
