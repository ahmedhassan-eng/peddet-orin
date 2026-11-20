# Queue

| Item | Status |
|---|---|
| A new judged window at one frozen commit, under this rule: each QUIET gate requires at least 6.0e9 bytes (5859375 kB) of available memory, a one-minute load below 1.0 after a 120 s settle, no swap in use and none swapped out, and no other workload; the first three runs at that commit in which every gate passes are the judged runs, and no run is chosen or dropped after it is seen; the log of every attempt, passed or stopped, is committed under runs-attempts/ with the gate line that stopped it. | planned for 0.1.2 |
| Re-measure so that each published benchmark log has its raw timings and power side file; the code that writes it ships, and the published logs, which predate it, lack it | planned for 0.1.2 |
| Re-measure so that each published log records the sha256 of its inputs (annotations, image names, ONNX file); the code that records them ships, and the published logs, which predate it, lack them | planned for 0.1.2 |
| Publish each eval's raw detections file as a release asset, so AP can be recomputed from the published runs with scripts/recompute_ap.py | planned for 0.1.2 |
| Re-measure so that each published benchmark log records its image pool count; the code that records it ships, and the published logs, which predate it, lack it | planned for 0.1.2 |
| Re-measure so that each published log records the board's device-tree model string and why an unread GPU-process count was not read; the code that records both ships, and the published logs, which predate it, lack them | planned for 0.1.2 |
| Move the long README sections to docs/, with their generated blocks and tests | planned for 0.1.2 |
| A PyTorch wheel built for the GPU's compute capability 8.7; the published PyTorch rows ran on a wheel that warns it does not support this GPU | planned for 0.1.2 |
| Show TF32 is off by a check that can fail: on this board the seeded network's error is no larger with TF32 on (1.13e-5 against 1.35e-5), so a tolerance cannot tell them apart | planned for 0.1.2 |
| Locked-clock benchmark with `jetson_clocks`, which needs root; this repository sets no clock lock. In the judged runs the GPU governor read `performance` with the GPU clock at its maximum in every sample, and the CPU clocks ran under `schedutil` | planned for 0.2.0 |
| Count of other GPU processes on this board; today the NVML query raises here, the count is recorded as `UNREADABLE`, and `pynvml` is not in the pinned locks | planned for 0.2.0 |
| Move the `tegrastats` reader out of the timed process, as the clock reader already is | planned for 0.2.0 |
| One shared image decode for the benchmark's timed loop and `preprocess.load_rgb`, which today hold two copies of the same decode | planned for 0.1.2 |
| A guard in the test suite that refuses the GPU tests unless the run holds the board's GPU lock; today none exists | planned for 0.1.2 |
| Benchmark the precisions in an order randomised per run, so a drift within a run does not fall on the later precisions; today every run uses one fixed order | planned for 0.2.0 |
| Hold a GPU lock in the run script and record it in each log; today other GPU processes read as `UNREADABLE` | planned for 0.1.2 |
| Check the extracted COCO files against a manifest; today the archives are checked by MD5 and sha256 and the files after extraction are trusted | planned for 0.1.2 |
| Refuse to run a source checkout on a Python outside 3.11 and 3.12 at run time; today only `requires-python` limits installs | planned for 0.1.2 |
| Write each run log atomically, by a rename, so a stopped job leaves no partial log | planned for 0.1.2 |
| Refuse negative power in the log check; zero or negative latencies and FPS, unordered percentiles and negative counts are refused already | planned for 0.1.2 |
| Compare the judged runs' environments before taking the median | planned for 0.1.2 |
| Measure the TensorRT processes' peak RSS without the PyTorch model loaded in them | planned for 0.1.2 |
| Refuse `http` and credential-carrying URLs in `scripts/lock_hashes.py` | planned for 0.1.2 |
| Check the model's class count against COCO's 91 when the weights load | planned for 0.1.2 |
| Judge 5 or more quieted runs per release; this release judges 3 | planned for 0.2.0 |
| Report all-class COCO mAP beside person AP, as an anchor against torchvision's published figure for these weights | planned for 0.1.2 |
| Measure accuracy at the deployment-style postprocess setting; its timing ships in 0.1.1 | planned for 0.1.2 |
| Give the end-to-end energy per frame a range across runs | planned for 0.1.2 |
| Tie the planted-results freshness check to a recorded planted run, not to the commit that last touched `tests/planted_results.txt`; today a one-character edit to that file satisfies it, and the release check fails on every commit that does not touch it | planned for 0.1.2 |
| Scan only this repository's own commits for trailers, so one merged commit with a sign-off or co-author trailer line does not fail the release check, and check the 72-character subject rule, which commit 148cc11 breaks with 77 | planned for 0.1.2 |
| Write a re-measure into its own directory, not `runs/`, which the repeat block reads by its newest clean logs; `scripts/compare_logs.py` now tells a re-measure in `runs/` from the committed logs by git, but a re-measure in `runs/` still makes `tests/test_readme_table.py` fail in the reproducer's checkout | planned for 0.1.2 |
| Mark in `src/peddet_orin/postprocess.py` the steps that follow torchvision's SSD postprocess; `NOTICE` holds torchvision's copyright notice and BSD-3-Clause text | planned for 0.1.2 |
| Give `requirements/pip-audit.lock` a header naming the tool and command that made it; today it has `# via` comments only | planned for 0.1.2 |
| Return to COCO's own host name for downloads if it serves a certificate that covers it; today `scripts/fetch_coco.py` uses the path-style S3 address, which AWS calls legacy, and the MD5 and sha256 checks pin the data whichever host serves it | planned for 0.1.2 |
| Measure the effect of the 3 s idle between warm-up and timing (`BASELINE_SAMPLE_S` in `src/peddet_orin/benchmark.py`) on the first timed iterations; today it is not measured | planned for 0.1.2 |
| Tag `8427133`, the judged runs' commit; today it is named only by its sha | planned for 0.1.2 |
| Read `__version__` from the checkout's `pyproject.toml` before an installed distribution's metadata when running from a source checkout; today an installed version wins | planned for 0.1.2 |
| Read the peak process RSS after the run log has loaded ONNX Runtime, or keep stating that the RSS excludes it | planned for 0.1.2 |
| Let R-11 fail on a real outlier: judge a held-out run against the others' median, or report the spread beside the tolerances | planned for 0.1.2 |
| Explain the INT8 person AP difference between builds, 26.74 at `025171d` against 26.83 at `17a1ee2` and `8427133`, or give it as a range; report INT8 AP as a cross-build range | planned for 0.1.2 |
| Put the responsible-use note under the summary, not last in Limitations; expand the name "peddet" where it first appears | planned for 0.1.2 |
| Move the judged runs to `runs/` and keep the earlier and superseded runs apart, so the shipped logs lead with the judged set | planned for 0.1.2 |
| Add a cooldown between benchmarks, so a thermal drift within a run does not fall on the later precisions | planned for 0.1.2 |
| Summarise the sampler files or ship them as release assets; today they are most of the tree's size | planned for 0.1.2 |
| Plant defects in the device copies and the timed-region boundaries, each with a test that catches it; today the TensorRT path has 3 planted defects (the stream synchronize, the FP32 build's flags and the INT8 calibration input) | planned for 0.1.2 |
| Report how much of the INT8 engine runs in INT8, from a per-layer precision summary of one build (IEngineInspector) with a trtexec cross-check; today no log shows it | planned for 0.1.2 |
| Say that the stage logs put the whole INT8-FP16 end-to-end gap in the postprocess stage, and that INT8 always ran after FP16, so no log separates the detection count from run order | planned for 0.1.2 |
| Log the benchmark's idle-baseline power, which its baseline tegrastats lines hold, and report energy above idle | planned for 0.1.2 |
| Decide whether to redact the GCID that every log's `nv_tegra_release` string carries | planned for 0.1.2 |
| State in SECURITY.md that the network guard refuses internet connections from the test process only, not from a subprocess a test starts | planned for 0.1.2 |
| Name the base of the INT8-FP16 engine gap in the README's tolerance sentence (5.5% of INT8's p50, 5.2% of FP16's) | planned for 0.1.2 |
| Re-add a train2017 archive subcommand to `scripts/fetch_coco.py`, with its sha256 pinned from a verified fetch, if a full train2017 fetch is ever needed; it was dropped because nothing here reads it | planned for 0.2.0 |
| Load the pretrained weights from the bytes whose sha256 was checked, not by name after the check, in `src/peddet_orin/model.py` | planned for 0.1.2 |
| Generate the repeat block's RAM sentence with the swap evidence it rests on, and move the fixed prose inside the generated markers into the generator's tests | planned for 0.1.2 |
| Exit `scripts/compare_logs.py` non-zero on any OUT line it prints outside the DONE count | planned for 0.1.2 |
| Test the clock sampler against a real devfreq file, not only fake clock files | planned for 0.1.2 |
| Check a convolution, not only a matmul and NMS, in `python -m peddet_orin.gpu_check` | planned for 0.1.2 |
| Split the INT8 comparison's joined sentence in the generated INT8 block into one sentence per tolerance | planned for 0.1.2 |
| Widen the run log's absolute-path pattern so a path that follows a word character, `.`, `-` or `~` is replaced too | planned for 0.1.2 |
