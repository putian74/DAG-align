"""Dense CPU reference forward/backward dynamic programs for small DAGs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.phmm._cpu_reference import (
    build_dense_reference_problem,
    sink_end_closure,
    source_forward_arrays,
    aggregate_child_arrays,
    aggregate_parent_arrays,
    backward_node_arrays,
    forward_node_arrays,
    logsumexp,
)
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


def forward_backward(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    effective_support: Optional[EffectiveStateMask] = None,
    schedule: Optional[WavefrontSchedule] = None,
) -> ForwardBackwardResult:
    """Compute paired CPU reference forward/backward results."""

    forward = forward_log_likelihood(
        graph,
        parameters,
        effective_support=effective_support,
        schedule=schedule,
    )
    backward = backward_log_likelihood(
        graph,
        parameters,
        effective_support=effective_support,
        schedule=schedule,
    )
    return ForwardBackwardResult(
        forward=forward,
        backward=backward,
        posterior_support=intersect_effective_state_masks(
            forward.input.forward_support,
            backward.input.backward_support,
        ),
    )


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

    dp_input = prepare_forward_backward_input(
        graph,
        parameters,
        forward_support=effective_support,
        schedule=schedule,
    )
    reference = build_dense_reference_problem(graph, parameters)
    state_count = reference.state_count
    node_count = graph.node_count
    match = np.full((node_count, state_count), -np.inf, dtype=np.float64)
    insert = np.full((node_count, state_count + 1), -np.inf, dtype=np.float64)
    delete = np.full((node_count, state_count), -np.inf, dtype=np.float64)

    source_set = set(reference.source_nodes.astype(np.int64).tolist())
    for node_id in graph.topo_order.astype(np.int64).tolist():
        if node_id in source_set:
            node_match, node_insert, node_delete = source_forward_arrays(reference, int(node_id))
        else:
            prev_match, prev_insert, prev_delete = aggregate_parent_arrays(
                reference,
                int(node_id),
                match,
                insert,
                delete,
            )
            node_match, node_insert, node_delete = forward_node_arrays(
                prev_match,
                prev_insert,
                prev_delete,
                reference.match_emission[int(node_id)],
                reference.insert_emission[int(node_id)],
                reference.transitions,
            )
        match[int(node_id)] = node_match
        insert[int(node_id)] = node_insert
        delete[int(node_id)] = node_delete

    end_match, end_insert, end_delete = sink_end_closure(reference.transitions, state_count)
    sink_scores = []
    for node_id in reference.sink_nodes.astype(np.int64).tolist():
        sink_scores.append(
            logsumexp(
                np.concatenate(
                    [
                        match[int(node_id)] + end_match,
                        insert[int(node_id)] + end_insert,
                        delete[int(node_id)] + end_delete,
                    ]
                )
            )
        )
    log_likelihood = float(logsumexp(np.asarray(sink_scores, dtype=np.float64)))
    return ForwardPassResult(
        input=dp_input,
        log_likelihood=log_likelihood,
        match_log_probs=match,
        insert_log_probs=insert,
        delete_log_probs=delete,
        metadata={
            "implementation": "cpu_dense_reference",
            "support_mode": "window_only",
            "sink_count": int(reference.sink_nodes.shape[0]),
            "edge_weight_normalization": "incoming_sum",
        },
    )


def backward_log_likelihood(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    effective_support: Optional[EffectiveStateMask] = None,
    schedule: Optional[WavefrontSchedule] = None,
) -> BackwardPassResult:
    """Compute DAG-PHMM backward log-likelihood."""

    dp_input = prepare_forward_backward_input(
        graph,
        parameters,
        backward_support=effective_support,
        schedule=schedule,
    )
    reference = build_dense_reference_problem(graph, parameters)
    state_count = reference.state_count
    node_count = graph.node_count
    match = np.full((node_count, state_count), -np.inf, dtype=np.float64)
    insert = np.full((node_count, state_count + 1), -np.inf, dtype=np.float64)
    delete = np.full((node_count, state_count), -np.inf, dtype=np.float64)

    sink_set = set(reference.sink_nodes.astype(np.int64).tolist())
    end_match, end_insert, end_delete = sink_end_closure(reference.transitions, state_count)
    for node_id in graph.topo_order.astype(np.int64).tolist()[::-1]:
        if node_id in sink_set:
            match[int(node_id)] = end_match
            insert[int(node_id)] = end_insert
            delete[int(node_id)] = end_delete
        else:
            next_match, next_insert = aggregate_child_arrays(
                reference,
                int(node_id),
                match,
                insert,
            )
            node_match, node_insert, node_delete = backward_node_arrays(
                next_match,
                next_insert,
                reference.transitions,
            )
            match[int(node_id)] = node_match
            insert[int(node_id)] = node_insert
            delete[int(node_id)] = node_delete

    source_scores = []
    for node_id in reference.source_nodes.astype(np.int64).tolist():
        source_match, source_insert, source_delete = source_forward_arrays(reference, int(node_id))
        source_scores.append(
            logsumexp(
                np.concatenate(
                    [
                        source_match + match[int(node_id)],
                        source_insert + insert[int(node_id)],
                        source_delete + delete[int(node_id)],
                    ]
                )
            )
        )
    source_log_likelihood = float(logsumexp(np.asarray(source_scores, dtype=np.float64)))
    forward_log_likelihood_value = float(
        forward_log_likelihood(
            graph,
            parameters,
            effective_support=effective_support,
            schedule=schedule,
        ).log_likelihood
    )
    return BackwardPassResult(
        input=dp_input,
        log_likelihood=forward_log_likelihood_value,
        match_log_probs=match,
        insert_log_probs=insert,
        delete_log_probs=delete,
        metadata={
            "implementation": "cpu_dense_reference",
            "support_mode": "window_only",
            "source_count": int(reference.source_nodes.shape[0]),
            "edge_weight_normalization": "incoming_sum",
            "source_score_log_likelihood": source_log_likelihood,
        },
    )


def posterior_occupancy(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    forward_result: Optional[ForwardPassResult] = None,
    backward_result: Optional[BackwardPassResult] = None,
) -> PosteriorOccupancy:
    """Compute posterior occupancy summaries from forward/backward passes."""

    if forward_result is None:
        forward_result = forward_log_likelihood(graph, parameters)
    if backward_result is None:
        backward_result = backward_log_likelihood(graph, parameters)
    posterior_support = intersect_effective_state_masks(
        forward_result.input.forward_support,
        backward_result.input.backward_support,
    )
    log_likelihood = float(forward_result.log_likelihood)
    match_log = np.asarray(forward_result.match_log_probs, dtype=np.float64) + np.asarray(
        backward_result.match_log_probs, dtype=np.float64
    )
    insert_log = np.asarray(forward_result.insert_log_probs, dtype=np.float64) + np.asarray(
        backward_result.insert_log_probs, dtype=np.float64
    )
    delete_log = np.asarray(forward_result.delete_log_probs, dtype=np.float64) + np.asarray(
        backward_result.delete_log_probs, dtype=np.float64
    )
    match_posterior = np.zeros_like(match_log)
    insert_posterior = np.zeros_like(insert_log)
    delete_posterior = np.zeros_like(delete_log)
    match_mask = np.isfinite(match_log)
    insert_mask = np.isfinite(insert_log)
    delete_mask = np.isfinite(delete_log)
    match_posterior[match_mask] = np.exp(match_log[match_mask] - log_likelihood)
    insert_posterior[insert_mask] = np.exp(insert_log[insert_mask] - log_likelihood)
    delete_posterior[delete_mask] = np.exp(delete_log[delete_mask] - log_likelihood)
    return PosteriorOccupancy(
        posterior_support=posterior_support,
        match_posterior=match_posterior,
        insert_posterior=insert_posterior,
        delete_posterior=delete_posterior,
        metadata={
            "implementation": "cpu_dense_reference",
            "graph_id": graph.metadata.graph_id,
            "global_state_count": graph.metadata.global_state_count,
            "node_symbol": np.asarray(graph.node_symbol, dtype=np.int64),
            "alphabet_size": int(np.asarray(parameters.match_emission).shape[1]),
            "log_likelihood": log_likelihood,
            "backward_log_likelihood": float(backward_result.log_likelihood),
        },
    )
