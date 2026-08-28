"""Python interface and Claude-powered agent for the Internet2 Periscope
Looking Glass MCP server (https://periscope.ns.internet2.edu/mcp)."""

from .agent import Internet2Agent
from .config import Settings
from .periscope import AsyncPeriscopeClient, PeriscopeClient, PeriscopeError

__version__ = "0.1.0"

__all__ = [
    "AsyncPeriscopeClient",
    "Internet2Agent",
    "PeriscopeClient",
    "PeriscopeError",
    "Settings",
    "__version__",
]
