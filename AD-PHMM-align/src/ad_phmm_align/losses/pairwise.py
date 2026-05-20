"""Compatibility wrapper for soft pairwise alignment scoring."""

from __future__ import annotations

from typing import Any

def soft_pairwise_score(occupancy: Any, score_matrix: Any, eps: float = 1e-12) -> Any:
    """Backward-compatible alias for the soft pairwise alignment score."""

    from ad_phmm_align.losses.soft_alignment import soft_pairwise_score as _impl

    return _impl(occupancy, score_matrix, eps=eps)
