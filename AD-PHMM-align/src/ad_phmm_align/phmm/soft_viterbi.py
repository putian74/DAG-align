"""Soft-Viterbi / log-sum-exp relaxation interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.phmm.parameters import PhmmParameterSet
from ad_phmm_align.phmm.ranges import EffectiveStateMask
from ad_phmm_align.phmm.viterbi import ViterbiInput, prepare_viterbi_input
from ad_phmm_align.phmm.wavefront import WavefrontSchedule


@dataclass(frozen=True)
class SoftViterbiResult:
    """Temperature-smoothed Viterbi score over packed DAG windows."""

    input: ViterbiInput
    temperature: float
    score: Any
    metadata: Mapping[str, object] = field(default_factory=dict)


def soft_viterbi_score(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    temperature: float = 1.0,
    effective_support: Optional[EffectiveStateMask] = None,
    schedule: Optional[WavefrontSchedule] = None,
) -> SoftViterbiResult:
    """Compute a temperature-smoothed Viterbi score."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    problem = prepare_viterbi_input(
        graph,
        parameters,
        effective_support=effective_support,
        schedule=schedule,
    )
    raise NotImplementedError(
        "soft_viterbi_score is not implemented yet; "
        "use prepare_viterbi_input for validated graph/support setup"
    )
