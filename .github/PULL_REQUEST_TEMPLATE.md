## What this changes

## Why

## How it was checked

- [ ] `ruff check .` and `ruff format --check .` pass.
- [ ] `python -m pytest -rs` passes, and any skipped tests are named here.
- [ ] If the GPU path changed: the tests also ran on a Jetson, with the board and JetPack version named here.
- [ ] If a result changed: its JSON log is under `runs/`, and the table was regenerated with
      `python scripts/generate_table.py --write`.
- [ ] New files are listed in the README "Repository layout".
- [ ] Commit messages follow the rule in `AGENTS.md`, with no trailers.
