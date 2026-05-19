"""Checkpointing interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union


def save_checkpoint(path: Union[Path, str], state: Any) -> None:
    """Save a training checkpoint."""

    raise NotImplementedError("save_checkpoint is not implemented yet")


def load_checkpoint(path: Union[Path, str]) -> Any:
    """Load a training checkpoint."""

    raise NotImplementedError("load_checkpoint is not implemented yet")
