"""Assemble the registry the agent is offered, from named toolsets.

Keep the default small: a 1.2B model picks tools worse as the list grows.
"""

from __future__ import annotations

from . import example_tools, workspace_tools
from .config import settings
from .tools import Registry
from .workspace import Workspace

TOOLSETS = ("workspace", "demo")


def build_registry(spec: str | None = None, workspace: str | None = None) -> Registry:
    """`spec` like "workspace,demo" (default LFM_TOOLSETS)."""
    names = [n.strip() for n in (settings.toolsets if spec is None else spec).split(",") if n.strip()]
    unknown = sorted(set(names) - set(TOOLSETS))
    if unknown:
        raise ValueError(f"unknown toolset(s) {unknown}; choose from {list(TOOLSETS)}")
    r = Registry()
    if "workspace" in names:
        workspace_tools.register(r, Workspace(workspace or settings.workspace))
    if "demo" in names:
        example_tools.register(r)
    return r
