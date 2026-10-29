"""User-writable cache paths for simulation resources."""

from __future__ import annotations

import os
from pathlib import Path


def cache_root() -> Path:
    """Return the simulator cache directory."""
    base = os.environ.get("XDG_CACHE_HOME") or os.environ.get("LOCALAPPDATA")
    path = (Path(base) if base else Path.home() / ".cache") / "mobile-network-sim"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()