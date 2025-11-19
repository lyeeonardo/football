from __future__ import annotations

import torch
from rich.console import Console


def resolve_device(preferred: str, console: Console | None = None) -> str:
    """Match the dataset pipeline's device resolution logic for consistency."""
    normalized = preferred.lower()
    logger = console or Console()
    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        logger.log("[yellow]CUDA requested but not available. Falling back to CPU.[/yellow]")
        return "cpu"
    if normalized == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if not (mps_backend and torch.backends.mps.is_available()):
            logger.log("[yellow]MPS requested but not available. Falling back to CPU.[/yellow]")
            return "cpu"
    return preferred
