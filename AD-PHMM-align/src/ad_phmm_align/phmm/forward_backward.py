"""Differentiable forward/backward dynamic-program scaffolds."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.phmm.parameters import PhmmParameterSet
from ad_phmm_align.phmm.ranges import (
    EffectiveStateMask,
    intersect_effective_state_masks,
    propagate_backward_support,
    propagate_forward_support,
)
from ad_phmm_align.phmm.wavefront import WavefrontSchedule, build_wavefront_schedule


@dataclass(frozen=True)
class DagDynamicProgrammingInput:
    """Validated inputs shared by forward, backward, and posterior kernels."""

    graph: TensorDag
    parameters: PhmmParameterSet
    schedule: WavefrontSchedule
    forward_support: EffectiveStateMask
    backward_support: EffectiveStateMask
    posterior_support: EffectiveStateMask
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ForwardPassResult:
    """Forward-pass outputs over packed DAG windows."""

    input: DagDynamicProgrammingInput
    log_likelihood: Any
    match_log_probs: Optional[Any] = None
    insert_log_probs: Optional[Any] = None
    delete_log_probs: Optional[Any] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class BackwardPassResult:
    """Backward-pass outputs over packed DAG windows."""

    input: DagDynamicProgrammingInput
    log_likelihood: Any
    match_log_probs: Optional[Any] = None
    insert_log_probs: Optional[Any] = None
    delete_log_probs: Optional[Any] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ForwardBackwardResult:
    """Paired forward/backward outputs with shared posterior support."""

    forward: ForwardPassResult
    backward: BackwardPassResult
    posterior_support: EffectiveStateMask


@dataclass(frozen=True)
class PosteriorOccupancy:
    """Posterior occupancy summaries derived from forward/backward outputs."""

    posterior_support: EffectiveStateMask
    match_posterior: Optional[Any] = None
    insert_posterior: Optional[Any] = None
    delete_posterior: Optional[Any] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


def prepare_forward_backward_input(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    forward_support: Optional[EffectiveStateMask] = None,
    backward_support: Optional[EffectiveStateMask] = None,
    schedule: Optional[WavefrontSchedule] = None,
) -> DagDynamicProgrammingInput:
    """Validate shared inputs and derive default effective-support masks."""

    graph.validate()
    schedule = build_wavefront_schedule(graph) if schedule is None else schedule
    schedule.validate(graph.node_count)
    forward_support = (
        propagate_forward_support(graph) if forward_support is None else forward_support
    )
    backward_support = (
        propagate_backward_support(graph) if backward_support is None else backward_support
    )
    posterior_support = intersect_effective_state_masks(forward_support, backward_support)
    return DagDynamicProgrammingInput(
        graph=graph,
        parameters=parameters,
        schedule=schedule,
        forward_support=forward_support,
        backward_support=backward_support,
        posterior_support=posterior_support,
        metadata={
            "graph_id": graph.metadata.graph_id,
            "global_state_count": graph.metadata.global_state_count,
        },
    )


def forward_log_likelihood(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    effective_support: Optional[EffectiveStateMask] = None,
    schedule: Optional[WavefrontSchedule] = None,
) -> ForwardPassResult:
    """Compute DAG-PHMM forward log-likelihood."""

    problem = prepare_forward_backward_input(
        graph,
        parameters,
        forward_support=effective_support,
        schedule=schedule,
    )
    raise NotImplementedError(
        "forward_log_likelihood recurrence is not implemented yet; "
        "use prepare_forward_backward_input for validated graph/support setup"
    )


def backward_log_likelihood(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    effective_support: Optional[EffectiveStateMask] = None,
    schedule: Optional[WavefrontSchedule] = None,
) -> BackwardPassResult:
    """Compute DAG-PHMM backward log-likelihood."""

    problem = prepare_forward_backward_input(
        graph,
        parameters,
        backward_support=effective_support,
        schedule=schedule,
    )
    raise NotImplementedError(
        "backward_log_likelihood recurrence is not implemented yet; "
        "use prepare_forward_backward_input for validated graph/support setup"
    )


def posterior_occupancy(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    forward_result: Optional[ForwardPassResult] = None,
    backward_result: Optional[BackwardPassResult] = None,
) -> PosteriorOccupancy:
    """Compute posterior occupancy summaries from forward/backward passes."""

    if forward_result is None or backward_result is None:
        problem = prepare_forward_backward_input(graph, parameters)
        posterior_support = problem.posterior_support
    else:
        posterior_support = intersect_effective_state_masks(
            forward_result.input.forward_support,
            backward_result.input.backward_support,
        )
    raise NotImplementedError(
        "posterior_occupancy is not implemented yet; "
        "effective posterior support is prepared and validated"
    )
