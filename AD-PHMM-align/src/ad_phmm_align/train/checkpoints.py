"""Checkpointing interfaces."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Union


def save_checkpoint(path: Union[Path, str], state: Any) -> None:
    """Save a training checkpoint."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        torch.save(state, target)
        return
    except ImportError:
        pass
    with target.open("wb") as handle:
        pickle.dump(state, handle)


def load_checkpoint(path: Union[Path, str]) -> Any:
    """Load a training checkpoint."""

    source = Path(path)
    try:
        import torch

        return torch.load(source, map_location="cpu", weights_only=False)
    except ImportError:
        pass
    with source.open("rb") as handle:
        return pickle.load(handle)
