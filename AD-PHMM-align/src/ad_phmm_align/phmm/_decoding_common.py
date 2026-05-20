"""Narrow shared helpers for hard and soft decoding-style DP paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.phmm.parameters import PhmmParameterSet
from ad_phmm_align.phmm.ranges import EffectiveStateMask, propagate_forward_support
from ad_phmm_align.phmm.wavefront import WavefrontSchedule, build_wavefront_schedule


@dataclass(frozen=True)
class ViterbiInput:
    """Validated decoding input shared by hard- and soft-Viterbi paths."""

    graph: TensorDag
    parameters: PhmmParameterSet
    schedule: WavefrontSchedule
    effective_support: EffectiveStateMask
    metadata: Mapping[str, object] = field(default_factory=dict)


def prepare_viterbi_input(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    effective_support: Optional[EffectiveStateMask] = None,
    schedule: Optional[WavefrontSchedule] = None,
) -> ViterbiInput:
    """Validate graph/parameter contracts and derive decoding support defaults."""

    graph.validate()
    if effective_support is None:
        effective_support = propagate_forward_support(graph)
    schedule = build_wavefront_schedule(graph) if schedule is None else schedule
    schedule.validate(graph.node_count)
    return ViterbiInput(
        graph=graph,
        parameters=parameters,
        schedule=schedule,
        effective_support=effective_support,
        metadata={
            "graph_id": graph.metadata.graph_id,
            "global_state_count": graph.metadata.global_state_count,
        },
    )
