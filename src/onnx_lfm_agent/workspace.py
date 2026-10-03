"""The one directory tree the agent's file tools may touch, plus git helpers."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

GIT_AUTHOR = ("onnx-lfm-agent", "onnx-lfm-agent@localhost")


class WorkspaceError(Exception):
    """A tool asked for something outside the sandbox or the workspace is unusable.
    The message goes back to the model, so keep it short and actionable."""


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        # Serializes git-mutating operations (the HTTP service runs requests
        # in worker threads).
        self.lock = threading.Lock()

    def resolve(self, path: str) -> Path:
        """Map a model-supplied path to a real path inside the workspace.

        Relative paths are taken from the root; absolute ones must already be
        inside it. Symlinks are resolved first, so links pointing out are
        refused too. `.git` is off-limits.
        """
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace directory not found: {self.root}")
        p = (self.root / (path or ".")).resolve()
        if not p.is_relative_to(self.root):
            raise WorkspaceError(f"path is outside the workspace: {path}")
        if ".git" in p.relative_to(self.root).parts:
            raise WorkspaceError(f"path is not accessible: {path}")
        return p

    def rel(self, p: Path) -> str:
        """Workspace-relative display form of a resolved path."""
        return p.relative_to(self.root).as_posix() or "."

    def git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        name, email = GIT_AUTHOR
        cmd = ["git", "-c", f"user.name={name}", "-c", f"user.email={email}",
               "-C", str(self.root), *args]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if check and proc.returncode != 0:
            msg = (proc.stderr or proc.stdout).strip().splitlines()
            raise WorkspaceError(f"git {args[0]} failed: {msg[-1] if msg else proc.returncode}")
        return proc

    def require_git(self) -> None:
        if not self.root.is_dir():
            raise WorkspaceError(f"workspace directory not found: {self.root}")
        proc = self.git("rev-parse", "--show-toplevel", check=False)
        if proc.returncode != 0 or Path(proc.stdout.strip()).resolve() != self.root:
            raise WorkspaceError("workspace is not the root of a git repository; "
                                 "config changes need git for commit/rollback")
