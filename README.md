# peddet-orin

Person detection on a Jetson Orin Nano Super with torchvision's SSDLite320 MobileNetV3-Large and its COCO weights, measured at three precisions: FP32 in PyTorch, and FP16 and INT8 in TensorRT. The benchmark reports accuracy on COCO 2017 val, latency, throughput, memory and power. Every number below comes from a JSON log in `runs/`, and a test fails if the table and the logs disagree.

## Hardware and software

The results were measured with this stack. The run logs record the versions each run used.

| Component | Version |
|---|---|
| Board | Jetson Orin Nano Super, GPU compute capability 8.7 |
| JetPack | 7, Jetson Linux R39 revision 2.1 |
| Power mode | `MAXN_SUPER` (mode 2) |
| CUDA | 13.0, from the PyTorch `cu130` wheels |
| TensorRT | 10.16.2.10, from the JetPack system Python |
| Python | 3.12 |
| PyTorch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 |
| ONNX / ONNX Runtime | 1.23.2 / 1.31.1 |

The CPU tests also run on any Linux machine with Python 3.10, 3.11 or 3.12. See "Continuous integration" below.

## Quickstart

This needs a Jetson Orin on JetPack 7. Run the commands from the repository root:

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

The venv uses `--system-site-packages` because TensorRT 10.16.2.10 comes from the JetPack system Python. `constraints/jetson-orin.txt` pins every package version.

Run the tests. They need no network. On the Jetson, the GPU and TensorRT tests in `tests/test_gpu.py` run too:

```sh
python -m pytest -rs
```

Export the network and build an FP16 engine. This downloads the pretrained weights through torchvision:

```sh
python -m peddet_orin.export
python -m peddet_orin.engine_build --precision fp16
```

## Reproduce the results

After the quickstart, fetch the data, check the GPU, then build and measure. `nvpmodel` and `tegrastats` must be on `PATH`; a log records `UNREADABLE` for anything it cannot read.

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

Each command writes a JSON log to `runs/`, and the last one rewrites the results table from the newest logs. Set `DEBUG=1` to turn on diagnostic logging and profiling.

## Results

Latencies are milliseconds, at batch size 1. "Engine" covers inference, including the copies to and from device memory. "End-to-end" adds image decoding, preprocessing and postprocessing. All precisions use the same postprocess. Accuracy is person AP on the 5000 COCO val2017 images, from pycocotools with `catIds=[1]`.

<!-- results:begin -->
| Device and precision | Engine p50/p90/p99 (ms) | End-to-end p50/p90/p99 (ms) | Engine FPS | End-to-end FPS | Peak process RSS (MiB) | System RAM rise (MiB) | Avg power VDD_IN (W) | Person AP@[.5:.95] | Person AP50 | INT8 path |
|---|---|---|---|---|---|---|---|---|---|---|
| n/a FP32 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | - |
| n/a FP16 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | - |
| n/a INT8 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
<!-- results:end -->

### Provenance

- Each row is one run per precision, on one board. The numbers are as run, with no averaging across runs.
- Every log records its commit, whether the tree was dirty, the start and end times, and the JetPack, TensorRT, PyTorch, torchvision and ONNX Runtime versions.
- The power mode in the notes is what `nvpmodel -q` reported. The clocks were not locked, since that needs root, and their state could not be read, so it is recorded as `UNREADABLE`. Timings therefore include any clock scaling.
- The number of other GPU processes could not be read either, and is recorded as `UNREADABLE`.
- `tests/test_readme_table.py` fails if the block above differs from what `scripts/generate_table.py` makes of the logs.

The FP32 row runs PyTorch on the GPU. The PyTorch wheels carry no slice built for this board's compute capability, so the GPU runs the compatible slice named in the notes above. The driver does not report which slice it loaded, so the notes give the one CUDA's binary-compatibility rule selects. `python -m peddet_orin.gpu_check` decides by running whether the GPU build is usable.

Peak process RSS does not fully capture memory allocated on the device. System RAM rise is the highest system-wide RAM during timing minus the mean just before, so it also includes any other activity on the board.

## Limitations

- One model, one dataset and one board. The numbers do not transfer to other Jetson modules, power modes or JetPack versions.
- Batch size 1 only.
- Clocks are not locked, and other work on the board is not excluded, so latencies can vary between runs.
- Power is the average of `tegrastats` VDD_IN readings during timing. It is the whole board's input power, not the GPU's alone.
- INT8 is calibrated on 500 seeded train2017 images. A different calibration set gives a different INT8 accuracy.
- Person-only fine-tuning was tried and not shipped. See `QUEUE.md`.

## Continuous integration

CI runs on every push and pull request. It uses only the CPU, so the GPU and TensorRT tests in `tests/test_gpu.py` skip there, and `-rs` names each skip in the log. `PDORIN_STRICT_SKIPS=1` makes any other skip fail the run. `tests/test_ci_contract.py` fails if `.github/workflows/ci.yml` runs anything other than this:

<!-- ci:begin -->
```sh
# CI: ubuntu-latest, Python 3.10, 3.11 and 3.12
python -m pip install -r requirements/ci-cpu.txt
python -m pip install --no-deps -e .
ruff check .
PDORIN_STRICT_SKIPS=1 python -m pytest -rs
```
<!-- ci:end -->

## Repository layout

`scripts/generate_tree.py --write` generates this tree from `git ls-files`, and `tests/test_readme_tree.py` fails if it differs from what the script generates.

<!-- tree:start -->
```text
peddet-orin/
├── .github/
│   ├── ISSUE_TEMPLATE/              bug report, feature request and the private security link
│   ├── workflows/
│   │   └── ci.yml                   ruff and the CPU tests on Python 3.10, 3.11 and 3.12
│   └── PULL_REQUEST_TEMPLATE.md
├── constraints/
│   └── jetson-orin.txt              every pin for the Jetson venv
├── manifests/
│   └── calibration_train2017.sha256  the 500 calibration images and their sha256 hashes
├── requirements/
│   └── ci-cpu.txt                   the CPU-only set for CI
├── runs/                            one JSON log per measurement; the results table is built from them
├── scripts/
│   ├── check_planted_defects.py     plants each registered defect and checks that its test fails
│   ├── ci_local.sh                  runs the CI matrix locally, one venv per Python version
│   ├── fetch_coco.py                fetches COCO 2017 from the official URLs into data/
│   ├── generate_table.py            writes the README results block from runs/
│   └── generate_tree.py             writes this tree from git ls-files
├── src/
│   └── peddet_orin/                 export, engine builds, evaluation, benchmarks and run logs
├── tests/                           CPU tests, GPU and TensorRT tests, and the repository-rule tests
├── .editorconfig
├── .gitignore
├── AGENTS.md                        rules for changes, with the shared commit-message rule
├── CHANGELOG.md
├── CITATION.cff
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

This project uses the [COCO 2017](https://cocodataset.org) dataset. The COCO annotations are licensed under CC BY 4.0, and the images remain under their own Flickr terms. The images and annotations are fetched from the official COCO URLs into `data/` and are not redistributed here. The calibration set is 500 train2017 images chosen by a fixed seed, listed with their sha256 hashes in `manifests/calibration_train2017.sha256`.

The pretrained weights, torchvision's `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1`, are downloaded by torchvision at run time and are not committed. torchvision's code is BSD 3-Clause. The weights were trained on COCO, so check torchvision's documentation and the COCO terms before you reuse them.

The code is licensed under Apache-2.0. See `LICENSE` and `NOTICE`.

## How to cite

If you use this software or its results, please cite it. `CITATION.cff` holds the metadata, and GitHub shows it as "Cite this repository". In BibTeX:

```bibtex
@software{hassan_peddet_orin_2026,
  author  = {Hassan, Ahmed},
  title   = {peddet-orin: person detection on the Jetson Orin in FP32, FP16 and INT8},
  version = {0.1.0},
  year    = {2026},
  url     = {https://github.com/ahmedhassan-eng/peddet-orin}
}
```

## Authorship

Built with AI coding agents under the direction of [Ahmed Hassan](https://github.com/ahmedhassan-eng). The design, the benchmark method and the review of every result are his.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and, for vulnerabilities, [SECURITY.md](SECURITY.md).
