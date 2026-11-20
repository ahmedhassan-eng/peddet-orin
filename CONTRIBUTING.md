# Contributing

Thank you for helping. This project measures one detector on one board, so a change is welcome when it keeps every
published number traceable to a run log.

## Before you start

- Follow the [code of conduct](CODE_OF_CONDUCT.md).
- Report a vulnerability privately, as [SECURITY.md](SECURITY.md) describes, not in an issue.
- Open an issue before a large change, so that the approach can be agreed first.

## Set up

On a Linux x86_64 machine whose `python3` is 3.11 or 3.12, the CPU-only set runs the linter and the tests:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --require-hashes -r requirements/ci-cpu.lock
python -m pip install --no-deps --no-build-isolation -e .
```

On a Jetson Orin, follow the README's "Quickstart" instead, which installs the board's set from
`constraints/jetson-orin.lock`. `requirements/ci-cpu.lock` and `constraints/jetson-orin.lock` are made from the `.txt` pin files beside them with
`python scripts/lock_hashes.py <pin file> <lock file>` (add `--extra-index-url https://download.pytorch.org/whl/cu130`
for the board's); after changing a pin, regenerate its lock. `requirements/pip-audit.lock` is not made by that script: it is a hashed
lock of pip-audit and its dependencies in pip-compile's format, from `requirements/pip-audit.txt`. `tests/test_locks.py`
fails if a lock and its pin file differ.

The package runs from a source checkout. Its commands find the checkout two levels above the package, or the one
`PDORIN_ROOT` names, and refuse to start anywhere else.

## Check your change

```sh
ruff check .
ruff format --check .
python -m pytest -rs
```

Two checks run only in the release check, which sets `PDORIN_RELEASE_CHECK=1`; CI and a contributor's run skip them. `tests/test_negative_controls.py::test_the_planted_output_was_made_from_the_current_files` needs `tests/planted_results.txt` to come from a planted run on the board after the last change; the check does not prove that a run happened, since an edit of that file alone passes it. `tests/test_agents_rule.py::test_no_commit_carries_a_trailer_or_a_subject_ending_in_a_period` scans every commit message, so one merged commit with a trailer fails the release check. CI does run the CPU planted defects, with `python scripts/check_planted_defects.py --cpu-only`, and fails unless each is caught.

The GPU and TensorRT tests in `tests/test_gpu.py` skip without a CUDA device or TensorRT. CI has neither, so they
run only on a Jetson. If your change touches the GPU path, run them there and say so in the pull request. On the
Jetson the whole suite takes longer, because it builds TensorRT engines.

The suite also enforces the repository's own rules:

- `tests/test_readme_tree.py`: the README "Repository layout" tree is what `scripts/generate_tree.py` makes of
  `git ls-files`. Run `python scripts/generate_tree.py --write` after adding or removing a file.
- `tests/test_ci_contract.py`: `.github/workflows/ci.yml` runs exactly the commands the README's CI section lists,
  pins every action by commit SHA, and matches `scripts/ci_local.sh`, which runs the same matrix locally.
- `tests/test_test_names.py`: no two tests share a name.
- `tests/test_agents_rule.py`: `AGENTS.md` states the commit-message rule, and no commit carries a trailer or
  ends its subject with a period.
- `tests/test_version_agreement.py`: `pyproject.toml`, `CHANGELOG.md` and `CITATION.cff` name the same version.
- `tests/test_strict_skips.py`: with `PDORIN_STRICT_SKIPS=1`, as in CI, any skip other than the eight GPU and
  TensorRT tests and the release-only tests, which skip with `tests/strict_skips.py`'s release reason unless
  `PDORIN_RELEASE_CHECK=1` is set, fails the run. Outside CI, the repository-rule tests may also skip in a release tarball (no `.git`)
  or a shallow clone, with exactly the reason `tests/strict_skips.py` names.
  With `PDORIN_REQUIRE_GPU=1`, as on the board, the GPU and TensorRT tests may not skip either.
- `tests/test_readme_table.py`: the README results table is exactly what the logs in `runs/`, `runs-repeat/` and `runs-superseded/` generate.
- `tests/test_changelog.py`: `CHANGELOG.md` has a dated entry with a release link for every version.
- `tests/test_doc_test_refs.py`: every test file and test name a tracked document cites exists and is collected.
- `tests/test_prd_coverage.py`: every requirement in `PRD.md` names a test that runs in CI.
- `tests/test_commit_ids.py`: every commit id a tracked file cites, such as a run log's, is in the history.
- `tests/test_planted_defects.py`: every entry in `tests/planted_defects.py` still applies. Run
  `python scripts/check_planted_defects.py` to plant each defect and check that its test fails.

## Results

Never edit the results table by hand. A re-measure from a clean tree writes its JSON logs under `runs/`; check it
against the published table with `python scripts/compare_logs.py runs`, which applies the tolerances of `PRD.md` R-11
and leaves the committed logs out. New judged runs go under `runs-repeat/`, one directory per run with its gate and
sampler records, and `python scripts/generate_table.py --write` then rebuilds the README's generated blocks. The run log records
the Jetson Linux release, power mode and package versions; name the board and JetPack version in the pull request.

## Commit messages

Follow the commit-message rule in [AGENTS.md](AGENTS.md): an imperative subject of at most 72 characters where the
change can be named in that, and a body of bullets with one claim each. This repository uses no commit trailers.

## Licence

By contributing, you agree that your contribution is licensed under the Apache License 2.0, as in [LICENSE](LICENSE).
