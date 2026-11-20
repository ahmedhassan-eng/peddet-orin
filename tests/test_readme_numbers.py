# SPDX-License-Identifier: Apache-2.0
"""Numbers typed into the README's and CHANGELOG's prose match the code, pins and logs they describe.

The generated blocks are checked by test_readme_table.py; this file covers the figures written by hand around them.
One is not checked here: the 5000 val2017 images are COCO's own count, fixed by the archive whose MD5
scripts/fetch_coco.py checks; every eval log records it as "images": 5000, but no test reads that field yet.
"""

import ast
import re
from pathlib import Path

import generate_table

REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}
# How the README names the judged metric that deviates most from the median.
LARGEST_LABELS = {"end_to_end_p50_ms": "end-to-end p50", "engine_p50_ms": "engine p50", "avg_power_w": "average power"}


def constant(path: Path, name: str):
    """The literal value of a module-level ``NAME = <literal>`` in a source file, read without importing it."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", None) == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} not found in {path}")


def flat(text: str) -> str:
    """Text with every run of whitespace, line breaks included, made one space."""
    return " ".join(text.split())


def test_the_repeat_count_and_the_baseline_window_match_the_benchmark():
    benchmark = REPO_ROOT / "src" / "peddet_orin" / "benchmark.py"
    text = flat(README.read_text())
    stated = re.findall(r"pools (\w+) timed repeats", text)
    assert len(stated) >= 2 and set(stated) == {WORDS[constant(benchmark, "REPEATS")]}, stated
    assert f"the {constant(benchmark, 'BASELINE_SAMPLE_S'):g} s just before timing" in text


def test_the_torch_wheel_and_the_compute_capability_match_the_pin_and_the_logs():
    text = flat(README.read_text())
    pin = re.search(r"^torch==(\S+)", (REPO_ROOT / "constraints" / "jetson-orin.txt").read_text(), re.M).group(1)
    assert f"its published {pin} wheels do not support" in text
    logs = [
        log
        for directory in (generate_table.RUNS_DIR, *sorted(generate_table.REPEATS_DIR.glob("*")))
        for log in generate_table.load_logs(directory)
        if log["kind"] == "gpu_check"
    ]
    capabilities = {log["results"]["device_compute_capability"] for log in logs}
    assert len(logs) >= 4 and capabilities == {"8.7"}
    assert text.count("this board's compute capability, 8.7") == 2
    assert "GPU compute capability 8.7" in text


def test_the_quieting_figure_is_the_largest_deviation_in_the_repeat_block():
    readme = README.read_text()
    block = generate_table.readme_block(readme, generate_table.REPEAT_MARKERS)
    deviations = [
        (abs(float(value)), line) for line in block.splitlines() for value in re.findall(r"\(([+-][0-9.]+)%\)", line)
    ]
    largest, line = max(deviations)
    metric, precision = (cell.strip() for cell in line.strip("|").split("|")[:2])
    assert f"up to {largest:.0f}% on the {precision} {LARGEST_LABELS[metric]}" in flat(readme)


def test_the_postprocess_settings_sentence_matches_the_model():
    from peddet_orin.model import load_detector

    model = load_detector(pretrained=False)
    expected = (
        f"a score threshold of {model.score_thresh:g}, the top {model.topk_candidates} candidates per class, "
        f"NMS at IoU {model.nms_thresh:g} and at most {model.detections_per_img} detections per image, "
        f"over all {model.head.classification_head.num_columns} class columns"
    )
    assert expected in flat(README.read_text())


def test_the_changelog_fp16_change_matches_the_logs():
    earlier = generate_table.select(generate_table.load_logs(generate_table.RUNS_DIR))
    published, _ = generate_table.headline()
    before = earlier[("benchmark", "fp16")]["results"]["end_to_end"]["p50_ms"]
    after = published[("benchmark", "fp16")]["results"]["end_to_end"]["p50_ms"]
    sentence = f"the FP16 end-to-end p50 went from {before:.1f} ms in 0.1.0 to {after:.1f} ms"
    assert sentence in flat(CHANGELOG.read_text())


def test_the_sample_counts_image_pool_and_seed_match_the_benchmark():
    benchmark = REPO_ROOT / "src" / "peddet_orin" / "benchmark.py"
    timed, warmup = constant(benchmark, "TIMED_ITERATIONS"), constant(benchmark, "WARMUP_ITERATIONS")
    pool, seed, repeats = (
        constant(benchmark, "IMAGE_POOL_SIZE"),
        constant(benchmark, "BENCHMARK_SEED"),
        constant(benchmark, "REPEATS"),
    )
    assert (
        f"Each repeat times {timed} iterations, after {warmup} untimed warm-up iterations, so each percentile pools "
        f"{timed * repeats} samples; the images are the first {pool} val2017 images in file-name order, taken by "
        f"{timed * repeats} draws with replacement from that pool, seeded with {seed}," in flat(README.read_text())
    )
