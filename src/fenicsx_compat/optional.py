import importlib
from types import ModuleType


def require_module(name: str, *, extra: str | None = None) -> ModuleType:
    """Import `name`, raising a clear, actionable ImportError if it's missing."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise ImportError(
            f"{name} is required for this feature: pip install {extra or name}"
        ) from e
