"""Publishing utilities for coinfosim."""

__all__ = ["publish_to_pages"]


def __getattr__(name):
    if name == "publish_to_pages":
        from .publisher import publish_to_pages

        return publish_to_pages
    raise AttributeError(name)
