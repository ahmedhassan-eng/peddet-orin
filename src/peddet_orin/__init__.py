# SPDX-License-Identifier: Apache-2.0
"""Person detection with torchvision SSDLite on a Jetson Orin Nano: PyTorch and TensorRT."""
from importlib import metadata

UNKNOWN_VERSION = "0+unknown"


def _read_version() -> str:
    """Return the installed distribution's version, else the one in the checkout's pyproject.toml."""
    try:
        return metadata.version("peddet-orin")
    except metadata.PackageNotFoundError:
        pass
    from peddet_orin import root

    try:
        checkout = root.resolve_root()
    except RuntimeError:  # Not installed and not in a checkout: there is no version to read.
        return UNKNOWN_VERSION
    return root.project_field(checkout / "pyproject.toml", "version") or UNKNOWN_VERSION


__version__ = _read_version()
