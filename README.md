# peddet-orin

Person detection on a Jetson Orin Nano with torchvision's SSDLite320 MobileNetV3-Large and its COCO weights, measured at three precisions: FP32 in PyTorch, and FP16 and INT8 in TensorRT. Every number below comes from a JSON log in `runs/`.

## Results

Latencies are milliseconds, at batch size 1. "Engine" covers inference, including the copies to and from device memory. "End-to-end" adds image decoding, preprocessing and postprocessing. All precisions use the same postprocess.

<!-- results:begin -->
| Precision | Runtime | Engine p50/p90/p99 (ms) | End-to-end p50/p90/p99 (ms) | Engine FPS | End-to-end FPS | Peak process RSS (MiB) | System RAM rise (MiB) | Avg power VDD_IN (W) | Person AP@[.5:.95] | Person AP50 | INT8 path |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP32 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | - |
| FP16 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | - |
| INT8 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
<!-- results:end -->

The FP32 row runs PyTorch on the CPU, so its timing is not comparable with the TensorRT rows. The PyTorch wheels available for this board have no kernels for its GPU. Its accuracy is the reference for the other rows.

Peak process RSS does not fully capture memory allocated on the device. System RAM rise is the highest system-wide RAM during timing minus the mean just before, so it also includes any other activity on the board.

## Reproduce from a fresh clone

The steps below need a Jetson Orin Nano Super on JetPack 7. The CPU tests need only the Python packages. `nvpmodel` and `tegrastats` must be on `PATH`; a log records `UNREADABLE` for anything it cannot read.

Set `INDEX` to the Jetson AI Lab pip index for SBSA and CUDA 13.0, whose path ends in `sbsa/cu130`. Then run these from the repository root:

```sh
export TORCH_HOME="$PWD/.cache/torch" PIP_CACHE_DIR="$PWD/.cache/pip" XDG_CACHE_HOME="$PWD/.cache/xdg"
export MPLCONFIGDIR="$PWD/.cache/mpl" CUDA_CACHE_PATH="$PWD/.cache/cuda"
export PYTHONPATH="$PWD/src:$PWD/scripts"
python3 -m venv .venv --system-site-packages
. .venv/bin/activate
pip install --index-url "$INDEX" torch==2.10.0 torchvision==0.25.0 nvpl "nvidia-cudss-cu13<1" "numpy<2"
pip install "numpy<2" onnx onnxruntime pycocotools
export LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/nvpl/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
```

Run the tests, which use the CPU and no network:

```sh
python -m pytest
```

Fetch the data, then build and measure. `FP32` runs on the CPU here, as explained above.

```sh
python scripts/fetch_coco.py annotations
python scripts/fetch_coco.py val
python scripts/fetch_coco.py calibration

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

Set `DEBUG=1` to turn on diagnostic logging and profiling.

## Data, weights and licences

This project uses the [COCO 2017](https://cocodataset.org) dataset. The COCO annotations are licensed under CC BY 4.0, and the images remain under their own Flickr terms. The images and annotations are fetched from the official COCO URLs into `data/` and are not redistributed here. The calibration set is 500 train2017 images chosen by a fixed seed, listed with their sha256 hashes in `manifests/calibration_train2017.sha256`.

The pretrained weights are downloaded by torchvision at run time and are not committed.

The code is licensed under Apache-2.0. See `LICENSE` and `NOTICE`.
