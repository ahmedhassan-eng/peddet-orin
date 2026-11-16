# SPDX-License-Identifier: Apache-2.0
"""Print the package version and the subcommands, each run as ``python -m peddet_orin.<name>``.

The subcommands are found by reading each module's source for a ``__main__`` guard,
without importing it, so this touches neither CUDA nor TensorRT.
"""

from __future__ import annotations

import pkgutil
import sys
from pathlib import Path

import peddet_orin

MAIN_GUARD = 'if __name__ == "__main__":'


def subcommands() -> list[str]:
    """Return the names of the package's modules that run as a program, sorted."""
    names = []
    for module in pkgutil.iter_modules(peddet_orin.__path__):
        if module.ispkg or module.name.startswith("_"):
            continue
        source = Path(module.module_finder.path, f"{module.name}.py")
        if source.is_file() and MAIN_GUARD in source.read_text():
            names.append(module.name)
    return sorted(names)


def main() -> int:
    print(f"peddet-orin {peddet_orin.__version__}")
    print("subcommands:")
    for name in subcommands():
        print(f"  python -m peddet_orin.{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
