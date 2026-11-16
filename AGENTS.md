# Rules for changes to this repository

These rules apply to everyone who changes this repository, whether a person or an automated agent.

## Build and test

- Set up as the README's "Quickstart" says on a Jetson Orin, or as `CONTRIBUTING.md` says on any other Linux machine.
- Run `ruff check .`, `ruff format --check .` and `python -m pytest -rs` before opening a pull request. CI runs them, the tests with `PDORIN_STRICT_SKIPS=1`.
- After adding or removing a file, run `python scripts/generate_tree.py --write`.

## Network and GPU

- The tests need no network: `tests/network_guard.py` fails any test that looks up a name or connects or sends on an internet socket. Only `pip`, the PyTorch index and torchvision's weights download need a network, and `scripts/fetch_coco.py` fetches the data.
- The GPU and TensorRT tests in `tests/test_gpu.py` skip without a CUDA device or TensorRT; on the board, run the suite with `PDORIN_REQUIRE_GPU=1` so such a skip fails. Run one GPU job at a time: two at once spoil each other's timings.

## Measurements

- Every published number comes from a JSON log under `runs/` or `runs-repeat/`; regenerate the README table with `python scripts/generate_table.py --write` and never edit it by hand.
- Measure from a clean tree, so the log records `dirty: false`. The table skips dirty logs and needs all its logs from one commit.
- Check a re-measure against the published table with `python scripts/compare_logs.py runs`.

## Commit messages

- Subject: imperative mood, no trailing period, at most 72 characters where the change can be named in that.
- Body: bullets, one claim each, saying what changed and why, scoped to a file or component where that helps.
- Keep every number, file path, symbol and quoted string the change turns on; leave out command output and logs.
- Commit messages carry no trailers.

`tests/test_agents_rule.py` fails if any commit carries a trailer or ends its subject with a period.
