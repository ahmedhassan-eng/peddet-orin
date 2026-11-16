# Queue

| Item | Status |
|---|---|
| Person-only fine-tuning on COCO train2017 | not shipped: the one run that fit the time budget scored below the FP32 reference; it left no run log and its code is not in this repository, so no number is published for it |
| Locked-clock benchmark with `jetson_clocks`, which needs root; today the GPU and CPU clocks are sampled during timing but not locked | planned for 0.2.0 |
| Count of other GPU processes on this board, which NVML does not report here | planned for 0.2.0 |
| Move the `tegrastats` reader out of the timed process, as the clock reader already is | planned for 0.2.0 |
| Re-measure the judged runs with the pinned set and this release's code, so that every log records the power mode, the peak process RSS with ONNX Runtime loaded only where it is used, and the numpy, Pillow, pycocotools and libjpeg versions | before this release is published |
| The TensorRT FP32 row, the precision baseline, measured in judged runs; its code ships and no judged run carries its logs | planned for 0.2.0 |
| The deployment-style benchmark (`--postprocess deploy`) measured in judged runs; its code ships and no deployment setting was measured | not yet measured |
| A guard in the test suite that refuses the GPU tests unless the run holds the board's GPU lock; today none exists | planned for 0.1.2 |
