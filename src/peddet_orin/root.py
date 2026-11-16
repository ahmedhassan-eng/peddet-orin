# SPDX-License-Identifier: Apache-2.0
"""Find the repository root, with no side effect on import: ``PDORIN_ROOT`` if set, else the checkout above the
package. ``runlog`` resolves it once and refuses to start without one; the package's version lookup only reads it."""
from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path

PROJECT_NAME = "peddet-orin"
ROOT_ENVIRONMENT_VARIABLE = "PDORIN_ROOT"
NO_CHECKOUT_MESSAGE = (
    "peddet-orin runs from a source checkout: run it from a clone of the repository, "
    "or set PDORIN_ROOT to the clone's directory"
)
BAD_OVERRIDE_MESSAGE = (
    "PDORIN_ROOT does not hold peddet-orin's pyproject.toml; peddet-orin runs from a source checkout, "
    "so set PDORIN_ROOT to the clone's directory"
)


def project_field(pyproject: Path, key: str) -> str | None:
    """Return a string value from the ``[project]`` table of a pyproject.toml, or ``None``."""
    try:
        table = tomllib.loads(pyproject.read_text()).get("project", {})
    except (OSError, tomllib.TOMLDecodeError):
        return None
    value = table.get(key) if isinstance(table, dict) else None
    return value if isinstance(value, str) else None


def is_checkout(directory: Path) -> bool:
    """Report whether a directory holds peddet-orin's pyproject.toml."""
    return project_field(directory / "pyproject.toml", "name") == PROJECT_NAME


def resolve_root(environ: Mapping[str, str] | None = None, module_file: str | Path = __file__) -> Path:
    """Return the repository root: ``PDORIN_ROOT`` if set, else the checkout above the package; else raise
    ``RuntimeError`` with a one-line reason."""
    environ = os.environ if environ is None else environ
    override = environ.get(ROOT_ENVIRONMENT_VARIABLE, "")
    if override:
        root = Path(override).expanduser().resolve()
        if not is_checkout(root):
            raise RuntimeError(BAD_OVERRIDE_MESSAGE)
        return root
    # src/peddet_orin/root.py: the checkout is two levels above the package directory.
    checkout = Path(module_file).resolve().parents[2]
    if is_checkout(checkout):
        return checkout
    raise RuntimeError(NO_CHECKOUT_MESSAGE)
