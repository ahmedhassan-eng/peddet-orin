# Queue

| Item | Status |
|---|---|
| Person-only fine-tuning on COCO train2017 | not shipped: the one run that fit the time budget scored below the FP32 reference; it left no run log and its code is not in this repository, so no number is published for it |
| Locked-clock benchmark with `jetson_clocks`, which needs root; today the GPU and CPU clocks are sampled during timing but not locked | open |
| Count of other GPU processes on this board, which NVML does not report here | open |
| Move the `tegrastats` reader out of the timed process, as the clock reader already is | open |
| Repeat runs on a quiet host to set the R-11 power tolerance from data; the one comparison so far is outside it on FP32 power | open |
| Drop `inferred_slice` from the `gpu_check` log, or record `none` when PyTorch's warning excludes this capability | open |
| A one-line error, not a traceback, for `--precision fp32 --device cuda` on a host with no usable GPU | open |
| Check each TensorRT I/O tensor's dtype is FLOAT before allocating its buffer | open |
| One sentinel for "no entropy failure" in build logs, enforced by the log check | open |
| Make the command-output helper public, and read the power mode before timing starts | open |
| Format the tree with `ruff format` and check it in CI | open |
