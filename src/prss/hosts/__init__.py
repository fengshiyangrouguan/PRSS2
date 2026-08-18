"""Host adapters: all host-specific coupling lives here, never in the core package."""

from prss.hosts.base import HostAdapter, OutsideBridge  # noqa: F401

__all__ = ["HostAdapter", "OutsideBridge"]
