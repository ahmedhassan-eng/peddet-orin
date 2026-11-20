# SPDX-License-Identifier: Apache-2.0
"""Recompute an eval log's person AP and AP50 from its detections file and the COCO annotations.

Usage: python scripts/recompute_ap.py <eval log> <detections file> <instances_val2017.json>

Each eval run from the 0.1.1 release commit on (the committed logs, written at 8427133, carry none) writes its raw
detections outside the tree and names the file and its sha256 in its log
(``results.detections_file``, ``results.detections_sha256``); a later release is to carry the files as assets.
This script checks the file against the logged sha256 and the annotations against
``results.annotations_sha256``, scores the detections with pycocotools as the eval did, and prints one line per
figure:

    RECOMPUTE-AP <figure> logged=.. recomputed=.. OK|DIFF

AP and AP50 agree when they match to 4 decimals of the fraction (the README prints percentage points to 2). The last
line is ``RECOMPUTE-AP DONE PASS`` (exit 0) or ``RECOMPUTE-AP DONE FAIL <n>`` (exit 1). A log that names no
detections file, as every committed log, is refused with one line and exit 2.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

from pycocotools.coco import COCO

from peddet_orin.evaluate import person_ap

TOLERANCE = 0.00005


def recompute(log_path: Path, detections_path: Path, annotations_path: Path) -> list[str]:
    """The RECOMPUTE-AP lines for one eval log, ending with its DONE line."""
    results = json.loads(Path(log_path).read_text())["results"]
    lines, failures = [], 0
    for name, path in (("detections_sha256", detections_path), ("annotations_sha256", annotations_path)):
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        ok = digest == results.get(name)
        failures += not ok
        lines.append(f"RECOMPUTE-AP {name} logged={results.get(name)} recomputed={digest} {'OK' if ok else 'DIFF'}")
    if failures:
        return [*lines, f"RECOMPUTE-AP DONE FAIL {failures}"]
    with contextlib.redirect_stdout(io.StringIO()):
        ground_truth = COCO(str(annotations_path))
    scores = person_ap(ground_truth, json.loads(Path(detections_path).read_bytes()), sorted(ground_truth.getImgIds()))
    for name in ("ap", "ap50"):
        ok = abs(scores[name] - results[name]) <= TOLERANCE
        failures += not ok
        lines.append(
            f"RECOMPUTE-AP {name} logged={results[name]} recomputed={scores[name]:.6f} {'OK' if ok else 'DIFF'}"
        )
    lines.append("RECOMPUTE-AP DONE PASS" if failures == 0 else f"RECOMPUTE-AP DONE FAIL {failures}")
    return lines


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 3:
        print(
            "usage: python scripts/recompute_ap.py <eval log> <detections file> <instances_val2017.json>",
            file=sys.stderr,
        )
        return 2
    log, detections, annotations = (Path(argument) for argument in arguments)
    results = json.loads(log.read_text())["results"]
    if "detections_file" not in results:
        print(
            f"recompute_ap: error: {log.name} names no detections_file: it was written before the code that writes one "
            "(the committed logs carry none)",
            file=sys.stderr,
        )
        return 2
    for path in (detections, annotations):
        if not path.is_file():
            print(f"recompute_ap: error: {path} is not a file", file=sys.stderr)
            return 2
    lines = recompute(log, detections, annotations)
    print("\n".join(lines))
    return 0 if lines[-1] == "RECOMPUTE-AP DONE PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
