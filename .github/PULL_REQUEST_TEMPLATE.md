## What this changes

## Why

## How it was checked

- [ ] `ruff check .` and `ruff format --check .` pass.
- [ ] `python -m pytest -rs` passes, and any skipped tests are named here.
- [ ] If the GPU path changed: the tests also ran on a Jetson, with the board and JetPack version named here.
- [ ] If a result changed: the board, its logs and what moved are named here. The README's results come from the
      judged runs under `runs-repeat/`; do not add logs under `runs/` or run `python scripts/generate_table.py
      --write`, since a new log in `runs/` would show as the earlier run.
- [ ] New files are listed in the README "Repository layout".
- [ ] Commit messages follow the rule in `AGENTS.md`, with no trailers.
