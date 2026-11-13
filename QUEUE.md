# Queue

| Item | Status |
|---|---|
| Person-only fine-tuning on COCO train2017 | not shipped: the one run that fit the time budget scored below the FP32 reference; it left no run log and its code is not in this repository, so no number is published for it |
| Locked-clock benchmark with `jetson_clocks`, which needs root; today the GPU and CPU clocks are sampled during timing but not locked | open |
| Count of other GPU processes on this board, which NVML does not report here | open |
| Move the `tegrastats` reader out of the timed process, as the clock reader already is | open |
