# peddet-orin: product requirements

**Goal.** Measure a person detector, torchvision's SSDLite320 MobileNetV3-Large with its COCO weights, on a Jetson Orin Nano, at FP32, FP16 and INT8, and publish accuracy and performance numbers that can be traced to run logs and reproduced from a fresh clone.

**Non-goals.** Other models or datasets, multi-object tracking, a serving layer, and redistributing images, labels or weights.

| Id | Requirement | Acceptance check |
|---|---|---|
| R-1 | Report person AP@[.5:.95] and AP50 on COCO val2017 from pycocotools with `catIds=[1]`. | An `eval` log exists under `runs/` for each precision. |
| R-2 | Export only the backbone and head to ONNX, with a 1x3x320x320 normalised input and raw outputs. | `python -m peddet_orin.export` logs a difference within the tolerance. |
| R-3 | All precisions share one postprocess, using torchvision's anchor generator and box coder. | `tests/test_postprocess.py` passes, including equality with torchvision's detector. |
| R-4 | Build FP16 and calibrated INT8 TensorRT engines on the device. | `build` logs exist; the INT8 log names the calibration path. |
| R-5 | Calibrate on 500 seeded train2017 images, disjoint from val2017, with recorded hashes. | `manifests/calibration_train2017.sha256` lists 500 files and the fetch script verifies them. |
| R-6 | Benchmark batch 1 with warm-up: latency percentiles, FPS, peak memory and power. | A `benchmark` log exists for each precision. |
| R-7 | Every published number comes from a JSON log with a defined schema. | `tests/test_run_logs.py` passes. |
| R-8 | The README table is generated from the logs. | `tests/test_readme_table.py` passes and fails on an edited number. |
| R-9 | Tests run on CPU with no GPU, TensorRT or network. | `python -m pytest` passes under the network guard. |
| R-10 | State any INT8 AP drop of more than 2 points against FP16. | The generated README block contains the statement when it applies. |
