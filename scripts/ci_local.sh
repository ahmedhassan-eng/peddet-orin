#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Run the CI matrix of .github/workflows/ci.yml on this machine: one fresh venv per Python version.
# tests/test_ci_contract.py fails if the versions or the commands below differ from the workflow's.
# Usage: scripts/ci_local.sh [log]   (default log: .cache/ci-local/ci-local.log)
cd "$(dirname "$0")" && cd .. || exit 1
LOG=${1:-.cache/ci-local/ci-local.log}
mkdir -p .cache/ci-local "$(dirname "$LOG")"
exec > >(tee "$LOG") 2>&1
# ci:versions
VERSIONS="3.11 3.12"
status=0
for version in $VERSIONS; do
  if ! command -v "python$version" > /dev/null; then
    echo "CI-LOCAL python$version MISSING"
    status=1
    continue
  fi
  venv=".cache/ci-local/py$version"
  rm -rf "$venv"
  "python$version" -m venv "$venv" || { echo "CI-LOCAL python$version VENV-FAIL"; status=1; continue; }
  if (
    . "$venv/bin/activate"
    set -e
    # ci:commands:begin
    python -m pip install -r requirements/ci-cpu.txt
    python -m pip install --no-deps -e .
    python -m venv --clear .cache/pip-audit
    .cache/pip-audit/bin/python -m pip install pip-audit==2.10.1
    .cache/pip-audit/bin/pip-audit --no-deps --disable-pip -r requirements/ci-cpu.txt
    ruff check .
    PDORIN_STRICT_SKIPS=1 python -m pytest -rs
    # ci:commands:end
  ); then
    echo "CI-LOCAL python$version PASS"
  else
    echo "CI-LOCAL python$version FAIL"
    status=1
  fi
done
echo "CI-LOCAL DONE $([ $status = 0 ] && echo PASS || echo FAIL)"
exit $status
