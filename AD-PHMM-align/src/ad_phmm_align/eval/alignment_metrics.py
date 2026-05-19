"""Alignment metric containers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class AlignmentMetrics:
    """Summary metrics for decoded alignments."""

    log_likelihood: Optional[float] = None
    entropy: Optional[float] = None
    pairwise_score: Optional[float] = None
