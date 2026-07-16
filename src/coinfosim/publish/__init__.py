"""Publishing utilities for coinfosim."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["publish_to_pages", "publish_pages", "PublishError", "PublishResult"]

_LAZY_ATTRIBUTES: dict[str, tuple[str, str]] = {
    "publish_to_pages": ("coinfosim.publish.publisher", "publish_to_pages"),
    "publish_pages": ("coinfosim.publish.publisher", "publish_pages"),
    "PublishError": ("coinfosim.publish.publisher", "PublishError"),
    "PublishResult": ("coinfosim.publish.publisher", "PublishResult"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _LAZY_ATTRIBUTES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
