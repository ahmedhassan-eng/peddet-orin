"""The README results block must be exactly what the logs generate."""
import re

import generate_table

DIGIT = re.compile(r"\d")


def test_readme_table_equals_generated_table():
    assert generate_table.readme_matches(generate_table.README.read_text())


def test_editing_one_number_is_detected():
    text = generate_table.README.read_text()
    block = generate_table.readme_block(text)
    # Prefer a digit in a data row, after the header divider.
    match = DIGIT.search(block, block.index("---")) or DIGIT.search(block)
    replacement = "1" if match.group() != "1" else "2"
    edited = block[: match.start()] + replacement + block[match.end() :]
    assert not generate_table.readme_matches(text.replace(block, edited))
