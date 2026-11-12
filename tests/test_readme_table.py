# SPDX-License-Identifier: Apache-2.0
"""The README results block must be exactly what the logs generate."""
import json
import re

import generate_table
from test_run_logs import make_log

NUMERIC_CELL = re.compile(r"^\d+\.\d+$")


def test_readme_table_equals_generated_table():
    assert generate_table.readme_matches(generate_table.README.read_text())


def test_editing_one_number_is_detected(tmp_path):
    # The committed README may carry only n/a cells, so the block is generated from synthetic logs.
    for index, log in enumerate([make_log("eval", "fp16"), make_log("benchmark", "fp16")]):
        (tmp_path / f"{index}.json").write_text(json.dumps(log))
    block = generate_table.build_block(tmp_path)
    text = f"intro\n{generate_table.BEGIN_MARKER}\n{block}\n{generate_table.END_MARKER}\nend\n"
    assert generate_table.readme_matches(text, tmp_path)
    lines = block.splitlines()
    row = next(index for index, line in enumerate(lines) if line.startswith("| GPU, TensorRT FP16"))
    cells = lines[row].split(" | ")
    column = next(index for index, cell in enumerate(cells) if NUMERIC_CELL.match(cell))
    original = cells[column]
    cells[column] = original[:-1] + ("1" if original[-1] != "1" else "2")
    edited = "\n".join([*lines[:row], " | ".join(cells), *lines[row + 1 :]])
    assert NUMERIC_CELL.match(cells[column]) and cells[column] != original
    assert not generate_table.readme_matches(text.replace(block, edited), tmp_path)
