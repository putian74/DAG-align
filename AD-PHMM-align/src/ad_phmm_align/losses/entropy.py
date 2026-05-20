"""Compatibility wrapper for soft alignment entropy."""

from __future__ import annotations

from typing import Any

from ad_phmm_align.losses.soft_alignment import (
    soft_alignment_entropy,
    soft_column_counts,
)

_column_counts = soft_column_counts


def posterior_entropy(posterior: Any, normalize: bool = True, eps: float = 1e-12) -> Any:
    """Backward-compatible alias for soft alignment entropy."""

    return soft_alignment_entropy(posterior, normalize=normalize, eps=eps)
