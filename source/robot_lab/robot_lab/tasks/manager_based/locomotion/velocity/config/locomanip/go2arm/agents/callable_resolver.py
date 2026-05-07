from __future__ import annotations

import importlib
from typing import Any


def _resolve_from_path(path: str) -> Any:
    module_path, separator, attr_path = path.partition(":")
    if not separator:
        module_path, separator, attr_path = path.rpartition(".")
    if not separator or not module_path or not attr_path:
        raise ValueError(
            f"Could not resolve callable path '{path}'. Expected 'module.submodule:callable' or "
            "'module.submodule.callable'."
        )

    resolved = importlib.import_module(module_path)
    for attr_name in attr_path.split("."):
        resolved = getattr(resolved, attr_name)
    return resolved


def resolve_callable(name_or_callable: Any) -> Any:
    """Resolve a callable without requiring newer rsl_rl helper utilities.

    rsl_rl >= 3.3.0 exports ``rsl_rl.utils.resolve_callable`` for cross-repo custom classes.
    Older releases such as 3.1.2 do not, so RoboDuet falls back to local importlib-based
    resolution while preserving the same ``"pkg.module:Symbol"`` config format.
    """

    if callable(name_or_callable):
        return name_or_callable
    if not isinstance(name_or_callable, str):
        raise TypeError(f"Expected callable or string path, got {type(name_or_callable)!r}.")

    try:
        from rsl_rl.utils import resolve_callable as rsl_resolve_callable
    except (ImportError, ModuleNotFoundError):
        rsl_resolve_callable = None

    resolved = rsl_resolve_callable(name_or_callable) if callable(rsl_resolve_callable) else _resolve_from_path(name_or_callable)
    if not callable(resolved):
        raise TypeError(f"Resolved object for '{name_or_callable}' is not callable: {type(resolved)!r}.")
    return resolved
