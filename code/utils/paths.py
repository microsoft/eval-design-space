"""Resolve project files independently of the caller's working directory."""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | os.PathLike[str]) -> str:
    """Anchor relative paths to the repository; preserve absolute overrides."""
    normalized = os.fspath(path).replace("\\", os.sep).replace("/", os.sep)
    expanded = os.path.expanduser(normalized)
    return os.path.normpath(os.path.join(REPO_ROOT, expanded))
