"""Soft and hard inference result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class SoftInferenceSummary:
    """Outputs from the differentiable/soft inference path."""

    forward_result: Any
    backward_result: Any
    posterior: Any
    soft_viterbi_result: Any
    negative_log_likelihood: float
    entropy: float
    pairwise_score: float
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HardInferenceSummary:
    """Outputs from the hard/max-product inference path."""

    viterbi_result: Any
    decoded_alignment: Optional[Any] = None
    alignment_metrics: Optional[Any] = None
    decode_status: str = "viterbi_only"
    decode_error: Optional[str] = None
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceSummary:
    """Combined soft and hard inference outputs for one graph/batch."""

    soft: SoftInferenceSummary
    hard: HardInferenceSummary
