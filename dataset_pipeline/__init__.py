"""Dataset creation pipeline for ball and goal+cone YOLO models."""

from pathlib import Path
from typing import Any

__all__ = ["run_pipeline"]


def run_pipeline(config_path: str | Path, *args: Any, **kwargs: Any) -> None:
    """Lazily import and execute the heavy pipeline module."""
    from .pipeline import run_pipeline as _run_pipeline

    _run_pipeline(config_path, *args, **kwargs)
