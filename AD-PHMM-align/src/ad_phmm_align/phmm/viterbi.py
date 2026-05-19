"""Hard Viterbi decoding interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.phmm.parameters import PhmmParameterSet
from ad_phmm_align.phmm.ranges import EffectiveStateMask, propagate_forward_support
from ad_phmm_align.phmm.wavefront import WavefrontSchedule, build_wavefront_schedule


@dataclass(frozen=True)
class ViterbiInput:
    """Validated inputs shared by hard- and soft-Viterbi kernels."""

    graph: TensorDag
    parameters: PhmmParameterSet
    schedule: WavefrontSchedule
    effective_support: EffectiveStateMask
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ViterbiBackpointerTable:
    """Sparse backpointer storage for reachable packed cells only."""

    packed_state_index: Any
    predecessor_edge_id: Any
    predecessor_channel: Any
    predecessor_packed_state_index: Any


@dataclass(frozen=True)
class ViterbiDecodeResult:
    """Decoded path/state output for evaluation."""

    score: float
    states: Any
    node_ids: Optional[Any] = None
    global_state_ids: Optional[Any] = None
    backpointers: Optional[ViterbiBackpointerTable] = None
    effective_support: Optional[EffectiveStateMask] = None
    metadata: Optional[Mapping[str, Any]] = None


def prepare_viterbi_input(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    effective_support: Optional[EffectiveStateMask] = None,
    schedule: Optional[WavefrontSchedule] = None,
) -> ViterbiInput:
    """Validate graph/parameter contracts and derive default effective support."""

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


def viterbi_decode(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    effective_support: Optional[EffectiveStateMask] = None,
    schedule: Optional[WavefrontSchedule] = None,
) -> ViterbiDecodeResult:
    """Decode hard Viterbi states."""

    problem = prepare_viterbi_input(
        graph,
        parameters,
        effective_support=effective_support,
        schedule=schedule,
    )
    raise NotImplementedError(
        "viterbi_decode is not implemented yet; "
        "use prepare_viterbi_input for validated graph/support setup"
    )
