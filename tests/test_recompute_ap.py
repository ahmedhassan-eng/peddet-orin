# SPDX-License-Identifier: Apache-2.0
"""scripts/recompute_ap.py rescores an eval's detections file and matches its logged AP."""

import hashlib
import json
from pathlib import Path

import recompute_ap
from peddet_orin import evaluate
from test_evaluate_scoring import PERSON_BOXES, as_detections, synthetic_coco

REPO_ROOT = Path(__file__).resolve().parents[1]


def _eval_log(tmp_path, ap_shift=0.0):
    ground_truth = synthetic_coco(tmp_path)
    detections = as_detections(PERSON_BOXES)
    scores = evaluate.person_ap(ground_truth, detections, sorted(ground_truth.getImgIds()))
    path = tmp_path / "detections" / "eval_fp16_20261120T000000Z.detections.json"
    hashes = evaluate.input_hashes(tmp_path / "instances.json", ["a.jpg"], detections, path)
    results = {**hashes, "ap": scores["ap"] + ap_shift, "ap50": scores["ap50"], "detections_file": path.name}
    log = tmp_path / "eval_fp16.json"
    log.write_text(json.dumps({"results": results}))
    return log, path, tmp_path / "instances.json"


def test_an_eval_logs_ap_recomputes_from_its_detections_file(tmp_path):
    lines = recompute_ap.recompute(*_eval_log(tmp_path))
    assert lines[-1] == "RECOMPUTE-AP DONE PASS", lines
    assert [line.split()[1] for line in lines[:-1]] == ["detections_sha256", "annotations_sha256", "ap", "ap50"]


def test_a_wrong_ap_or_another_detections_file_fails(tmp_path):
    log, detections, annotations = _eval_log(tmp_path, ap_shift=0.01)
    assert recompute_ap.recompute(log, detections, annotations)[-1] == "RECOMPUTE-AP DONE FAIL 1"
    detections.write_bytes(detections.read_bytes() + b" ")
    lines = recompute_ap.recompute(log, detections, annotations)
    assert lines[-1] == "RECOMPUTE-AP DONE FAIL 1" and lines[0].endswith(" DIFF")
    assert hashlib.sha256(detections.read_bytes()).hexdigest() in lines[0]


def test_a_committed_log_without_a_detections_file_is_refused_in_one_line(tmp_path, capsys):
    log = sorted((REPO_ROOT / "runs-repeat" / "run6").glob("eval_fp16_*.json"))[0]
    assert recompute_ap.main([str(log), str(tmp_path / "none.json"), str(tmp_path / "none.json")]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.count("\n") == 1 and "names no detections_file" in captured.err
