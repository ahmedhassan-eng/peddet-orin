# peddet-orin

Person detection on a Jetson Orin Nano Super with torchvision's SSDLite320 MobileNetV3-Large and its COCO weights, measured at three precisions: FP32 in PyTorch, and FP16 and INT8 in TensorRT. The benchmark reports accuracy on COCO 2017 val, latency, throughput, memory and power. Every number below comes from a JSON log in `runs/`, and a test fails if the table and the logs disagree.

## Hardware and software

The results were measured with this stack. Each run log records the Jetson Linux release string, the power mode and the Python, PyTorch, torchvision, CUDA, TensorRT, ONNX, ONNX Runtime and peddet-orin versions it ran with. The board and JetPack rows come from the board's documentation, not from the logs.

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

The venv uses `--system-site-packages` because TensorRT 10.16.2.10 comes from the JetPack system Python. `constraints/jetson-orin.txt` pins every package that pip installs into the venv; TensorRT, and any package the system Python already provides at a compatible version, are used from the system.

The package runs from this source checkout, not from an installed wheel. Its commands find the checkout two levels above the package, or the directory `PDORIN_ROOT` names, and refuse to start anywhere else. `python -m peddet_orin` lists the commands; each is also installed as a `peddet-orin-<command>` script.

Run the tests. They need no network. On the Jetson, the GPU and TensorRT tests in `tests/test_gpu.py` run too, so the suite takes about fifteen minutes while it builds TensorRT engines:

```sh
python -m pytest -rs
```

Export the network and build an FP16 engine. This downloads the pretrained weights once, and checks their sha256 before every load:

```sh
python -m peddet_orin.export
python -m peddet_orin.engine_build --precision fp16
```

## Reproduce the results

After the quickstart, fetch the data over HTTPS, check the GPU, then build and measure. `nvpmodel` and `tegrastats` must be on `PATH`; a log records `UNREADABLE` for anything it cannot read. Run the measurements from a clean tree, one at a time, with nothing else using the GPU.

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

Each command writes a JSON log to `runs/`, and the last one rewrites the results table from the newest clean logs, which must all come from one commit. `--data-dir`, `--engine`, `--onnx`, `--output` and `--calibration-dir` override the default locations, and `python -m peddet_orin.<command> --help` lists each command's options. A missing input stops a command with one line naming the command that makes it.

To check a re-measure against the committed numbers, write its logs to another directory, for example a fresh clone's `runs/`, and compare: `python scripts/compare_logs.py <that runs dir>`. It applies the tolerances of `PRD.md` R-11 and prints one line per metric. Set `DEBUG=1` to turn on diagnostic logging and profiling.

## Results

Latencies are milliseconds, at batch size 1. "Engine" covers inference, including the copies to and from device memory. "End-to-end" adds image decoding, preprocessing and postprocessing. All precisions use the same postprocess. Each benchmark pools three timed repeats of its loop, and the notes give the range of the per-repeat medians. Accuracy is person AP on the 5000 COCO val2017 images, from pycocotools with `catIds=[1]`.

<!-- results:begin -->
| Device and precision | Engine p50/p90/p99 (ms) | End-to-end p50/p90/p99 (ms) | Engine FPS | End-to-end FPS | Peak process RSS (MiB) | Avg power VDD_IN (W) | Person AP@[.5:.95] | Person AP50 | INT8 path |
|---|---|---|---|---|---|---|---|---|---|
| GPU, PyTorch FP32 | 30.3 / 33.2 / 37.2 | 96.6 / 119.8 / 147.9 | 32.2 | 10.2 | 1271 | 8.6 | 28.01 | 49.50 | - |
| GPU, TensorRT FP16 | 3.9 / 4.2 / 6.2 | 75.1 / 102.3 / 146.7 | 250.6 | 12.9 | 1305 | 8.8 | 28.00 | 49.48 | - |
| GPU, TensorRT INT8 | 3.7 / 3.9 / 5.4 | 78.3 / 101.7 / 137.9 | 265.0 | 12.5 | 1075 | 8.7 | 26.83 | 48.15 | IInt8EntropyCalibrator2 |

- INT8 person AP differs from FP16 by -1.17 points.
- FP32 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 1467 of 1467 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 1317 of 1467 samples; policy4 governor `schedutil`, at max 1728 MHz in 1058 of 1467 samples, one-minute CPU load before timing: 3.9, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- FP32 p50 across repeats: engine 30.05-31.29 ms over 3 repeats, end-to-end 93.43-103.34 ms over 3 repeats.
- FP16 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 1165 of 1165 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 1121 of 1165 samples; policy4 governor `schedutil`, at max 1728 MHz in 959 of 1165 samples, one-minute CPU load before timing: 4.4, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- FP16 p50 across repeats: engine 3.84-3.98 ms over 3 repeats, end-to-end 69.35-86.09 ms over 3 repeats.
- INT8 benchmark: power mode `NV Power Mode: MAXN_SUPER 2`, GPU clock governor `performance`, at max 1020 MHz in 1202 of 1202 samples; CPU clocks policy0 governor `schedutil`, at max 1728 MHz in 1159 of 1202 samples; policy4 governor `schedutil`, at max 1728 MHz in 972 of 1202 samples, one-minute CPU load before timing: 5.5, GPU busy before timing: 0%, other GPU processes: UNREADABLE.
- INT8 p50 across repeats: engine 3.66-3.74 ms over 3 repeats, end-to-end 73.59-86.84 ms over 3 repeats.
- PyTorch GPU check: usable True, device compute capability 8.7, wheel slices sm_80, sm_90, sm_100, sm_110, sm_120. PyTorch emitted an unsupported-GPU warning at first CUDA use: "Found GPU0 Orin which is of compute capability (CC) 8.7.". FP32 ran with cudnn.allow_tf32 False and cuda.matmul.allow_tf32 False.
<!-- results:end -->

### Provenance

- Each row is one run per precision, on one board. A benchmark pools three timed repeats of its loop; there is no averaging across runs.
- Every log records its commit, whether the tree was dirty, the start and end times, the Jetson Linux release string, and the Python, PyTorch, torchvision, CUDA, TensorRT, ONNX, ONNX Runtime and peddet-orin versions. The table uses only clean logs, all from one commit.
- The power mode in the notes is what `nvpmodel -q` reported. `jetson_clocks` was not run, since it needs root. During timing a separate process samples the GPU's devfreq clock and each CPU frequency policy's clock every 200 ms; the notes give each governor and how many samples were at that clock's maximum. Memory clocks are not read.
- The one-minute load average in the notes is read just before timing starts, so it shows what else was running then. The number of other GPU processes could not be read, and is recorded as `UNREADABLE`.
- The measurements ran one at a time, at normal priority, while no other GPU or benchmark job ran on the board.
- `tests/test_readme_table.py` fails if the block above differs from what `scripts/generate_table.py` makes of the logs.

The FP32 row runs PyTorch on the GPU with TF32 turned off for cuDNN and matmul, so it is FP32 throughout. PyTorch warns on first CUDA use that its published 2.14.1+cu130 wheels do not support this board's compute capability, 8.7; the notes quote the warning's first line, the `gpu_check` log holds all of it, and the code does not hide it. Which kernel image the driver runs is not reported. `python -m peddet_orin.gpu_check` decides by running whether the GPU build is usable: it compares a matmul and NMS against the CPU, and `tests/test_gpu.py` compares a full forward pass.

Peak process RSS does not fully capture memory allocated on the device. System RAM rise, the highest system-wide RAM during timing minus the mean just before, is in the logs but not in the table: it includes any other activity on the board and does not reproduce between runs.

In this run the INT8 engine is faster than the FP16 one, but its end-to-end median is slower, and the per-repeat ranges in the notes overlap. End-to-end time is mostly image decoding, preprocessing and postprocessing on the CPU, so the engine is a small share of it. INT8 also keeps more boxes above the score threshold on val2017 (the `detections` count in the eval logs), which gives the postprocess more work. No profile here separates that from run-to-run variation.

## Limitations

- One model, one dataset and one board. The numbers do not transfer to other Jetson modules, power modes or JetPack versions.
- The export uses PyTorch's TorchScript-based ONNX exporter, which PyTorch has deprecated; `pyproject.toml` bounds torch below 2.15 for that reason.
- CI runs on x86_64 CPUs only. The aarch64 build, the GPU tests and TensorRT run only on the Jetson, by hand, before a release.
- Batch size 1 only.
- `jetson_clocks` is not run, memory clocks are not read, a tegrastats reader thread runs in the timed process, and other work on the board is not excluded, so latencies can vary between runs.
- Power is the average of `tegrastats` VDD_IN readings during timing. It is the whole board's input power, not the GPU's alone.
- INT8 is calibrated on 500 seeded train2017 images. A different calibration set gives a different INT8 accuracy.
- Person-only fine-tuning was tried and not shipped. See `QUEUE.md`.

## Continuous integration

CI runs on every push to `main` and on every pull request, cancels a run that a newer push supersedes, and stops a job after 30 minutes. It uses only the CPU, so the GPU and TensorRT tests in `tests/test_gpu.py` skip there, and `-rs` names each skip in the log. `PDORIN_STRICT_SKIPS=1` makes any other skip fail the run. `pip-audit` checks the pinned CPU set for known vulnerabilities. `tests/test_ci_contract.py` fails if `.github/workflows/ci.yml` runs anything other than this:

<!-- ci:begin -->
```sh
# CI: ubuntu-latest, Python 3.10, 3.11 and 3.12
python -m pip install -r requirements/ci-cpu.txt
python -m pip install --no-deps -e .
python -m venv --clear .cache/pip-audit
.cache/pip-audit/bin/python -m pip install pip-audit==2.10.1
.cache/pip-audit/bin/pip-audit --no-deps --disable-pip -r requirements/ci-cpu.txt
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
│   │   └── ci.yml                   pip-audit, ruff and the CPU tests on Python 3.10, 3.11 and 3.12
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
│   ├── compare_logs.py              checks a re-measure against runs/ within the PRD tolerances
│   ├── fetch_coco.py                fetches COCO 2017 over HTTPS from the official URLs into data/
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

This project uses the [COCO 2017](https://cocodataset.org) dataset. The COCO annotations are licensed under CC BY 4.0, and the images remain under their own Flickr terms. The images and annotations are fetched over HTTPS from the official COCO URLs into `data/` and are not redistributed here. The calibration set is 500 train2017 images chosen by a fixed seed, listed with their sha256 hashes in `manifests/calibration_train2017.sha256`.

The pretrained weights, torchvision's `SSDLite320_MobileNet_V3_Large_Weights.COCO_V1`, are downloaded at run time and are not committed. They load only after their full sha256 matches the value pinned in `src/peddet_orin/model.py`, and only as tensors. torchvision's code is BSD 3-Clause. The weights were trained on COCO, so check torchvision's documentation and the COCO terms before you reuse them.

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
