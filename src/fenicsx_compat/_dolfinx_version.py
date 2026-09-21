from packaging.version import Version

import dolfinx

DOLFINX_VERSION = Version(dolfinx.__version__)


def at_least(v: str) -> bool:
    """Whether the installed dolfinx is at least version `v`."""
    return DOLFINX_VERSION >= Version(v)


def before(v: str) -> bool:
    """Whether the installed dolfinx is strictly before version `v`."""
    return DOLFINX_VERSION < Version(v)
