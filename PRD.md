# peddet-orin: product requirements

**Goal.** Measure a person detector, torchvision's SSDLite320 MobileNetV3-Large with its COCO weights, on a Jetson Orin Nano, at FP32, FP16 and INT8, and publish accuracy and performance numbers that can be traced to run logs and reproduced from a fresh clone.

**Non-goals.** Other models or datasets, multi-object tracking, a serving layer, and redistributing images, labels or weights.

| Id | Requirement | Acceptance check | Covered in CI by | Orin-run |
|---|---|---|---|---|
| R-1 | Report person AP@[.5:.95] and AP50 on COCO val2017 from pycocotools with `catIds=[1]`. | An `eval` log exists under `runs/` for each precision. | `tests/test_requirements.py::test_r1_an_eval_log_exists_for_each_precision` | `python -m peddet_orin.evaluate` for each precision |
| R-2 | Export only the backbone and head to ONNX, with a 1x3x320x320 normalised input and raw outputs. | `python -m peddet_orin.export` logs a difference within the tolerance. | `tests/test_requirements.py::test_r2_the_export_log_is_within_its_tolerance`, `tests/test_onnx_export.py::test_onnx_matches_pytorch`, `tests/test_onnx_export.py::test_graph_outputs_are_raw_regression_and_logits` | `python -m peddet_orin.export` |
| R-3 | All precisions share one postprocess, using torchvision's anchor generator and box coder. | `tests/test_postprocess.py` passes, including equality with torchvision's detector. | `tests/test_postprocess.py::test_pipeline_equals_torchvision_detector` | - |
| R-4 | Build FP16 and calibrated INT8 TensorRT engines on the device. | `build` logs exist; the INT8 log names the calibration path. | `tests/test_requirements.py::test_r4_build_logs_exist_and_int8_names_its_calibration_path` | `python -m peddet_orin.engine_build`; `tests/test_gpu.py::test_tensorrt_fp16_engine_matches_cpu` |
| R-5 | Calibrate on 500 seeded train2017 images, disjoint from val2017, with recorded hashes. | `manifests/calibration_train2017.sha256` lists 500 files and the fetch script verifies them. | `tests/test_requirements.py::test_r5_the_manifest_lists_500_train2017_images_with_hashes`, `tests/test_requirements.py::test_r5_a_calibration_hash_mismatch_is_rejected` | `python scripts/fetch_coco.py calibration` |
| R-6 | Benchmark batch 1 with warm-up: latency percentiles, FPS, peak memory and power. | A `benchmark` log exists for each precision. | `tests/test_requirements.py::test_r6_a_benchmark_log_exists_for_each_precision` | `python -m peddet_orin.benchmark` for each precision |
| R-7 | Every published number comes from a JSON log with a defined schema. | `tests/test_run_logs.py` passes. | `tests/test_run_logs.py::test_committed_logs_are_valid`, `tests/test_run_logs.py::test_each_required_field_is_enforced` | - |
| R-8 | The README table is generated from the logs. | `tests/test_readme_table.py` passes and fails on an edited number. | `tests/test_readme_table.py::test_readme_table_equals_generated_table`, `tests/test_readme_table.py::test_editing_one_number_is_detected` | - |
| R-9 | Tests run on CPU with no GPU, TensorRT or network. | `python -m pytest` passes under the network guard. | `tests/test_network_guard.py::test_guard_raises_inside_a_test`, `tests/test_strict_skips.py::test_an_unexpected_skip_fails_a_strict_run` | - |
| R-10 | State any INT8 AP drop of more than 2 points against FP16. | The generated README block contains the statement when it applies. | `tests/test_requirements.py::test_r10_an_int8_drop_over_two_points_is_stated` | - |

`tests/test_prd_coverage.py` fails if a requirement names no CI test, or names one that is not collected or that skips in CI. The Orin-run column lists what runs only on the Jetson: the measurements behind the committed logs and the GPU tests.
