# Contributing

Thank you for helping. This project measures one detector on one board, so a change is welcome when it keeps every
published number traceable to a run log.

## Before you start

- Read the [Code of Conduct](CODE_OF_CONDUCT.md).
- Report a vulnerability privately, as [SECURITY.md](SECURITY.md) describes, not in an issue.
- Open an issue before a large change, so that the approach can be agreed first.

## Set up

On any Linux machine, the CPU-only set runs the linter and the tests:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements/ci-cpu.txt
python -m pip install --no-deps -e .
```

On a Jetson Orin, follow the README's "Quickstart" instead, which pins every package with
`constraints/jetson-orin.txt`.

## Check your change

```sh
ruff check .
python -m pytest -rs
```

The GPU and TensorRT tests in `tests/test_gpu.py` skip without a CUDA device or TensorRT. If your change touches the
GPU path, run the tests on a Jetson as well and say so in the pull request.

The suite also enforces the repository's own rules:

- `tests/test_readme_layout.py`: the README "Repository layout" tree matches the files git tracks. Add new files to
  it.
- `tests/test_ci_contract.py`: `.github/workflows/ci.yml` runs exactly the commands the README's CI section lists.
- `tests/test_test_names.py`: no two tests share a name.
- `tests/test_agents_rule.py`: the commit-message rule in `AGENTS.md` is unchanged, with its declared digest.
- `tests/test_version_agreement.py`: `pyproject.toml`, `CHANGELOG.md` and `CITATION.cff` name the same version.
- `tests/test_strict_skips.py`: with `PDORIN_STRICT_SKIPS=1`, as in CI, any skip other than the two GPU and
  TensorRT tests fails the run.
- `tests/test_readme_table.py`: the README results table is exactly what the logs in `runs/` generate.

## Results

Never edit the results table by hand. Run the measurement, commit its JSON log under `runs/`, then run
`python scripts/generate_table.py --write`. A new result must name its board, JetPack, power mode and package
versions, which the run log records.

## Commit messages

Follow the commit-message rule in [AGENTS.md](AGENTS.md): an imperative subject of 72 characters or fewer, and a
body of bullets with one claim each. This repository uses no commit trailers.

## Licence

By contributing, you agree that your contribution is licensed under the Apache License 2.0, as in [LICENSE](LICENSE).
