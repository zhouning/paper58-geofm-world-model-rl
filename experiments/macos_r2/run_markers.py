# -*- coding: utf-8 -*-
"""Completion marker helpers for macOS R2 experiments."""
from __future__ import annotations

from pathlib import Path


def mark_done(results_dir: Path, smoke: bool = False) -> Path:
    """Write the right sentinel for smoke vs full experiment runs.

    ``run_all_macos.sh --resume`` keys off the canonical ``.done`` marker, so
    smoke runs must not create it.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    if smoke:
        marker = results_dir / ".done_smoke"
        marker.touch()
        return marker
    smoke_marker = results_dir / ".done_smoke"
    if smoke_marker.exists():
        smoke_marker.unlink()
    (results_dir / ".done").touch()
    marker = results_dir / ".done_full"
    marker.touch()
    return marker
