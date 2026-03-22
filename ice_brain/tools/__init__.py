"""
Tool registry for ice_brain.

Phase 1: Registry is wired up but tools are never called automatically.
Phase 5+: The orchestrator will look up tools here and execute them.

Usage
-----
    from tools import register_tool, get_available_tools

    @register_tool("weather")
    def my_weather_tool(location: str) -> dict:
        ...
"""

from __future__ import annotations

from typing import Any, Callable, Dict

_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_tool(name: str) -> Callable:
    """Decorator that registers a callable under *name*."""
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = fn
        return fn
    return decorator


def get_available_tools() -> Dict[str, Callable[..., Any]]:
    """Return a shallow copy of the current tool registry."""
    return dict(_REGISTRY)


def call_tool(name: str, **kwargs: Any) -> Any:
    """Call a registered tool by name.  Raises KeyError if unknown."""
    if name not in _REGISTRY:
        raise KeyError(f"Unknown tool: '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)
