"""Global PHMM state-mask interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence


@dataclass(frozen=True)
class CandidateStateRange:
    """Half-open global state range proposed for active-state sampling."""

    left: int
    right: int
    score: float = 0.0
    reason: str = "unspecified"

    @property
    def length(self) -> int:
        """Number of global states in the half-open range."""

        return self.right - self.left


@dataclass(frozen=True)
class StateMaskSpec:
    """Global state mask or candidate range configuration."""

    global_state_count: int
    mode: str = "fixed"
    candidate_ranges: Optional[Sequence[CandidateStateRange]] = None
    metadata: Optional[Mapping[str, object]] = None
