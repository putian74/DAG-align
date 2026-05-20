"""Global PHMM state-mask interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np


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
    mode: str = "full"
    candidate_ranges: Optional[Sequence[CandidateStateRange]] = None
    metadata: Optional[Mapping[str, object]] = None

    def validate(self) -> None:
        """Validate state-mask configuration against the global state axis."""

        if self.global_state_count < 0:
            raise ValueError("global_state_count must be non-negative")
        if self.mode not in {"full", "fixed"}:
            raise ValueError(f"unsupported state-mask mode: {self.mode}")
        if self.mode == "full":
            return
        if not self.candidate_ranges:
            raise ValueError("fixed state-mask mode requires candidate_ranges")
        for candidate in self.candidate_ranges:
            if candidate.left < 0:
                raise ValueError("candidate range left bound is negative")
            if candidate.right < candidate.left:
                raise ValueError("candidate range right bound is before left")
            if candidate.right > self.global_state_count:
                raise ValueError("candidate range exceeds global_state_count")

    def active_global_state_ids(self) -> np.ndarray:
        """Materialize sorted active global state IDs from the configured ranges."""

        self.validate()
        if self.mode == "full":
            return np.arange(self.global_state_count, dtype=np.int64)
        active_ranges = [
            np.arange(candidate.left, candidate.right, dtype=np.int64)
            for candidate in (self.candidate_ranges or ())
            if candidate.right > candidate.left
        ]
        if not active_ranges:
            return np.zeros((0,), dtype=np.int64)
        return np.unique(np.concatenate(active_ranges))
