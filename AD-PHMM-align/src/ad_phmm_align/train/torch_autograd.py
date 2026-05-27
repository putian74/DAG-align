"""Torch-backed dense reference training helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import torch

from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.phmm._decoding_common import prepare_viterbi_input
from ad_phmm_align.phmm.forward_backward import prepare_forward_backward_input
from ad_phmm_align.phmm.parameters import PhmmParameterSet
from ad_phmm_align.phmm.ranges import EffectiveStateMask
from ad_phmm_align.phmm.wavefront import WavefrontSchedule


@dataclass(frozen=True)
class TorchTransitionBundle:
    start_match: torch.Tensor
    start_delete: torch.Tensor
    start_insert: torch.Tensor
    match_to_match: torch.Tensor
    match_to_delete: torch.Tensor
    match_to_insert: torch.Tensor
    match_to_end: torch.Tensor
    delete_to_match: torch.Tensor
    delete_to_delete: torch.Tensor
    delete_to_insert: torch.Tensor
    delete_to_end: torch.Tensor
    insert_to_match: torch.Tensor
    insert_to_delete: torch.Tensor
    insert_to_insert: torch.Tensor
    insert_to_end: torch.Tensor


@dataclass(frozen=True)
class TorchDenseReferenceProblem:
    graph: TensorDag
    transitions: TorchTransitionBundle
    edge_log_weights: torch.Tensor
    incoming_edges: tuple[tuple[int, ...], ...]
    outgoing_edges: tuple[tuple[int, ...], ...]
    source_nodes: tuple[int, ...]
    sink_nodes: tuple[int, ...]
    match_emission: tuple[torch.Tensor, ...]
    insert_emission: tuple[torch.Tensor, ...]
    state_count: int
    alphabet_size: int
    device: torch.device
    dtype: torch.dtype


@dataclass(frozen=True)
class TorchForwardResult:
    log_likelihood: torch.Tensor
    match_log_probs: tuple[torch.Tensor, ...]
    insert_log_probs: tuple[torch.Tensor, ...]
    delete_log_probs: tuple[torch.Tensor, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TorchBackwardResult:
    log_likelihood: torch.Tensor
    match_log_probs: tuple[torch.Tensor, ...]
    insert_log_probs: tuple[torch.Tensor, ...]
    delete_log_probs: tuple[torch.Tensor, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TorchPosteriorOccupancy:
    posterior_support: EffectiveStateMask
    match_posterior: tuple[torch.Tensor, ...]
    insert_posterior: tuple[torch.Tensor, ...]
    delete_posterior: tuple[torch.Tensor, ...]
    node_symbol: torch.Tensor
    alphabet_size: int
    log_likelihood: torch.Tensor
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TorchSoftViterbiResult:
    score: torch.Tensor
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TorchLossSummary:
    loss: torch.Tensor
    loss_components: Mapping[str, float]
    metrics: Mapping[str, float]


def build_dense_reference_problem(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> TorchDenseReferenceProblem:
    transitions = unpack_transition_bundle(parameters, device=device, dtype=dtype)
    state_count = int(parameters.match_emission.shape[0])
    alphabet_size = int(parameters.match_emission.shape[1])
    edge_src = torch.as_tensor(graph.edge_src, dtype=torch.long, device=device)
    edge_dst = torch.as_tensor(graph.edge_dst, dtype=torch.long, device=device)
    edge_weight = torch.as_tensor(graph.edge_weight, dtype=dtype, device=device)
    if torch.any(edge_weight <= 0.0):
        raise ValueError("graph edge weights must be positive for torch reference DP")
    incoming_sum = torch.zeros((graph.node_count,), dtype=dtype, device=device)
    incoming_sum.index_add_(0, edge_dst, edge_weight)
    edge_log_weights = torch.log(edge_weight) - torch.log(incoming_sum[edge_dst])

    incoming_edges = _edges_by_node(graph.edge_dst, graph.edge_count, graph.node_count)
    outgoing_edges = _edges_by_node(graph.edge_src, graph.edge_count, graph.node_count)
    indegree = torch.bincount(edge_dst, minlength=graph.node_count)
    outdegree = torch.bincount(edge_src, minlength=graph.node_count)
    source_nodes = tuple(int(node_id) for node_id in torch.nonzero(indegree == 0, as_tuple=False).flatten())
    sink_nodes = tuple(int(node_id) for node_id in torch.nonzero(outdegree == 0, as_tuple=False).flatten())

    match_emission, insert_emission = build_dense_emission_tables(
        graph,
        parameters,
        state_count=state_count,
        device=device,
        dtype=dtype,
    )
    return TorchDenseReferenceProblem(
        graph=graph,
        transitions=transitions,
        edge_log_weights=edge_log_weights,
        incoming_edges=incoming_edges,
        outgoing_edges=outgoing_edges,
        source_nodes=source_nodes,
        sink_nodes=sink_nodes,
        match_emission=match_emission,
        insert_emission=insert_emission,
        state_count=state_count,
        alphabet_size=alphabet_size,
        device=device,
        dtype=dtype,
    )


def unpack_transition_bundle(
    parameters: PhmmParameterSet,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TorchTransitionBundle:
    tensor = torch.as_tensor(parameters.transition_logits.tensor, dtype=dtype, device=device)
    lookup = {
        name: torch.as_tensor(parameters.transition_logits.require(name), dtype=dtype, device=device)
        for name in tuple(parameters.transition_logits.order)
    }
    match_logits = lookup["_mm"]
    delete_logits = lookup["_md"]
    insert_logits = lookup["_mi"]
    d_match_logits = lookup["_dm"]
    d_delete_logits = lookup["_dd"]
    d_insert_logits = lookup["_di"]
    i_match_logits = lookup["_im"]
    i_delete_logits = lookup["_id"]
    i_insert_logits = lookup["_ii"]
    state_count = int(tensor.shape[0] - 1)
    neg_inf = torch.tensor(float("-inf"), dtype=dtype, device=device)

    start = torch.log_softmax(
        torch.stack([match_logits[0], delete_logits[0], insert_logits[0]]), dim=0
    )
    match_to_match = torch.full((max(0, state_count - 1),), neg_inf, dtype=dtype, device=device)
    match_to_delete = torch.full((max(0, state_count - 1),), neg_inf, dtype=dtype, device=device)
    match_to_insert = torch.full((state_count,), neg_inf, dtype=dtype, device=device)
    for state in range(max(0, state_count - 1)):
        family = torch.log_softmax(
            torch.stack(
                [
                    match_logits[state + 1],
                    delete_logits[state + 1],
                    insert_logits[state + 1],
                ]
            ),
            dim=0,
        )
        match_to_match[state] = family[0]
        match_to_delete[state] = family[1]
        match_to_insert[state] = family[2]
    if state_count > 0:
        match_tail = torch.log_softmax(
            torch.stack([match_logits[state_count], insert_logits[state_count]]), dim=0
        )
        match_to_end = match_tail[0]
        match_to_insert[state_count - 1] = match_tail[1]
    else:
        match_to_end = neg_inf

    insert_to_match = torch.full((state_count,), neg_inf, dtype=dtype, device=device)
    insert_to_delete = torch.full((state_count,), neg_inf, dtype=dtype, device=device)
    insert_to_insert = torch.full((state_count + 1,), neg_inf, dtype=dtype, device=device)
    for state in range(state_count):
        family = torch.log_softmax(
            torch.stack(
                [i_match_logits[state], i_delete_logits[state], i_insert_logits[state]]
            ),
            dim=0,
        )
        insert_to_match[state] = family[0]
        insert_to_delete[state] = family[1]
        insert_to_insert[state] = family[2]
    insert_tail = torch.log_softmax(
        torch.stack([i_match_logits[state_count], i_insert_logits[state_count]]), dim=0
    )
    insert_to_end = insert_tail[0]
    insert_to_insert[state_count] = insert_tail[1]

    delete_to_match = torch.full((max(0, state_count - 1),), neg_inf, dtype=dtype, device=device)
    delete_to_delete = torch.full((max(0, state_count - 1),), neg_inf, dtype=dtype, device=device)
    delete_to_insert = torch.full((state_count,), neg_inf, dtype=dtype, device=device)
    for state in range(max(0, state_count - 1)):
        family = torch.log_softmax(
            torch.stack(
                [
                    d_match_logits[state + 1],
                    d_delete_logits[state + 1],
                    d_insert_logits[state + 1],
                ]
            ),
            dim=0,
        )
        delete_to_match[state] = family[0]
        delete_to_delete[state] = family[1]
        delete_to_insert[state] = family[2]
    if state_count > 0:
        delete_tail = torch.log_softmax(
            torch.stack([d_match_logits[state_count], d_insert_logits[state_count]]), dim=0
        )
        delete_to_end = delete_tail[0]
        delete_to_insert[state_count - 1] = delete_tail[1]
    else:
        delete_to_end = neg_inf

    return TorchTransitionBundle(
        start_match=start[0],
        start_delete=start[1],
        start_insert=start[2],
        match_to_match=match_to_match,
        match_to_delete=match_to_delete,
        match_to_insert=match_to_insert,
        match_to_end=match_to_end,
        delete_to_match=delete_to_match,
        delete_to_delete=delete_to_delete,
        delete_to_insert=delete_to_insert,
        delete_to_end=delete_to_end,
        insert_to_match=insert_to_match,
        insert_to_delete=insert_to_delete,
        insert_to_insert=insert_to_insert,
        insert_to_end=insert_to_end,
    )


def build_dense_emission_tables(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    *,
    state_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    match_logits = torch.as_tensor(parameters.match_emission, dtype=dtype, device=device)
    insert_logits = torch.as_tensor(parameters.insert_emission, dtype=dtype, device=device)
    match_tensor = torch.log_softmax(match_logits, dim=1)
    insert_tensor = torch.log_softmax(insert_logits, dim=1)
    node_symbols = torch.as_tensor(graph.node_symbol, dtype=torch.long, device=device)
    node_window_left = torch.as_tensor(graph.node_window_left, dtype=torch.long, device=device)
    node_window_right = torch.as_tensor(graph.node_window_right, dtype=torch.long, device=device)
    alphabet_size = int(match_tensor.shape[1])
    if torch.any(node_symbols < 0) or torch.any(node_symbols >= alphabet_size):
        raise ValueError("node symbols exceed the PHMM emission alphabet")

    match_emission = []
    insert_emission = []
    neg_inf = float("-inf")
    for node_id in range(graph.node_count):
        symbol = int(node_symbols[node_id].item())
        left = int(node_window_left[node_id].item())
        right = int(node_window_right[node_id].item())
        match_row = torch.full((state_count,), neg_inf, dtype=dtype, device=device)
        insert_row = torch.full((state_count + 1,), neg_inf, dtype=dtype, device=device)
        if right > left:
            match_row[left:right] = match_tensor[left:right, symbol]
            insert_row[left : right + 1] = insert_tensor[left : right + 1, symbol]
        match_emission.append(match_row)
        insert_emission.append(insert_row)
    return tuple(match_emission), tuple(insert_emission)


def forward_log_likelihood(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    *,
    effective_support: Optional[EffectiveStateMask] = None,
    schedule: Optional[WavefrontSchedule] = None,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> TorchForwardResult:
    dp_input = prepare_forward_backward_input(
        graph,
        parameters,
        forward_support=effective_support,
        schedule=schedule,
    )
    reference = build_dense_reference_problem(graph, parameters, device=device, dtype=dtype)
    neg_inf = float("-inf")
    match = [torch.full((reference.state_count,), neg_inf, dtype=dtype, device=device) for _ in range(graph.node_count)]
    insert = [torch.full((reference.state_count + 1,), neg_inf, dtype=dtype, device=device) for _ in range(graph.node_count)]
    delete = [torch.full((reference.state_count,), neg_inf, dtype=dtype, device=device) for _ in range(graph.node_count)]

    source_set = set(reference.source_nodes)
    for node_id in graph.topo_order.astype("int64", copy=False).tolist():
        if node_id in source_set:
            node_match, node_insert, node_delete = source_forward_arrays(reference, int(node_id))
        else:
            prev_match, prev_insert, prev_delete = aggregate_parent_arrays(
                reference, int(node_id), match, insert, delete
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

    end_match, end_insert, end_delete = sink_end_closure(reference.transitions, reference.state_count)
    sink_scores = []
    for node_id in reference.sink_nodes:
        sink_scores.append(
            torch_logsumexp(
                torch.cat(
                    [
                        match[node_id] + end_match,
                        insert[node_id] + end_insert,
                        delete[node_id] + end_delete,
                    ]
                )
            )
        )
    log_likelihood = torch_logsumexp(torch.stack(sink_scores))
    return TorchForwardResult(
        log_likelihood=log_likelihood,
        match_log_probs=tuple(match),
        insert_log_probs=tuple(insert),
        delete_log_probs=tuple(delete),
        metadata={
            "graph_id": graph.metadata.graph_id,
            "global_state_count": graph.metadata.global_state_count,
            "posterior_support": dp_input.posterior_support,
        },
    )


def backward_log_likelihood(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    *,
    effective_support: Optional[EffectiveStateMask] = None,
    schedule: Optional[WavefrontSchedule] = None,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> TorchBackwardResult:
    dp_input = prepare_forward_backward_input(
        graph,
        parameters,
        backward_support=effective_support,
        schedule=schedule,
    )
    reference = build_dense_reference_problem(graph, parameters, device=device, dtype=dtype)
    neg_inf = float("-inf")
    match = [torch.full((reference.state_count,), neg_inf, dtype=dtype, device=device) for _ in range(graph.node_count)]
    insert = [torch.full((reference.state_count + 1,), neg_inf, dtype=dtype, device=device) for _ in range(graph.node_count)]
    delete = [torch.full((reference.state_count,), neg_inf, dtype=dtype, device=device) for _ in range(graph.node_count)]

    sink_set = set(reference.sink_nodes)
    end_match, end_insert, end_delete = sink_end_closure(reference.transitions, reference.state_count)
    for node_id in graph.topo_order.astype("int64", copy=False).tolist()[::-1]:
        if node_id in sink_set:
            match[int(node_id)] = end_match
            insert[int(node_id)] = end_insert
            delete[int(node_id)] = end_delete
        else:
            next_match, next_insert = aggregate_child_arrays(reference, int(node_id), match, insert)
            node_match, node_insert, node_delete = backward_node_arrays(
                next_match,
                next_insert,
                reference.transitions,
            )
            match[int(node_id)] = node_match
            insert[int(node_id)] = node_insert
            delete[int(node_id)] = node_delete

    source_scores = []
    for node_id in reference.source_nodes:
        source_match, source_insert, source_delete = source_forward_arrays(reference, int(node_id))
        source_scores.append(
            torch_logsumexp(
                torch.cat(
                    [
                        source_match + match[node_id],
                        source_insert + insert[node_id],
                        source_delete + delete[node_id],
                    ]
                )
            )
        )
    source_log_likelihood = torch_logsumexp(torch.stack(source_scores))
    return TorchBackwardResult(
        log_likelihood=source_log_likelihood,
        match_log_probs=tuple(match),
        insert_log_probs=tuple(insert),
        delete_log_probs=tuple(delete),
        metadata={
            "graph_id": graph.metadata.graph_id,
            "global_state_count": graph.metadata.global_state_count,
            "posterior_support": dp_input.posterior_support,
        },
    )


def posterior_occupancy(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    *,
    forward_result: TorchForwardResult,
    backward_result: TorchBackwardResult,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> TorchPosteriorOccupancy:
    posterior_support = forward_result.metadata["posterior_support"]
    log_likelihood = forward_result.log_likelihood
    neg_inf = torch.tensor(float("-inf"), dtype=dtype, device=device)
    match_posterior = []
    insert_posterior = []
    delete_posterior = []
    for node_id in range(graph.node_count):
        match_log = forward_result.match_log_probs[node_id] + backward_result.match_log_probs[node_id]
        insert_log = forward_result.insert_log_probs[node_id] + backward_result.insert_log_probs[node_id]
        delete_log = forward_result.delete_log_probs[node_id] + backward_result.delete_log_probs[node_id]
        match_posterior.append(
            torch.exp(torch.where(torch.isfinite(match_log), match_log - log_likelihood, neg_inf))
        )
        insert_posterior.append(
            torch.exp(torch.where(torch.isfinite(insert_log), insert_log - log_likelihood, neg_inf))
        )
        delete_posterior.append(
            torch.exp(torch.where(torch.isfinite(delete_log), delete_log - log_likelihood, neg_inf))
        )
    return TorchPosteriorOccupancy(
        posterior_support=posterior_support,
        match_posterior=tuple(match_posterior),
        insert_posterior=tuple(insert_posterior),
        delete_posterior=tuple(delete_posterior),
        node_symbol=torch.as_tensor(graph.node_symbol, dtype=torch.long, device=device),
        alphabet_size=int(parameters.match_emission.shape[1]),
        log_likelihood=log_likelihood,
        metadata={"graph_id": graph.metadata.graph_id},
    )


def soft_viterbi_score(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    *,
    temperature: float,
    effective_support: Optional[EffectiveStateMask] = None,
    schedule: Optional[WavefrontSchedule] = None,
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> TorchSoftViterbiResult:
    prepare_viterbi_input(
        graph,
        parameters,
        effective_support=effective_support,
        schedule=schedule,
    )
    reference = build_dense_reference_problem(graph, parameters, device=device, dtype=dtype)
    neg_inf = float("-inf")
    match = [torch.full((reference.state_count,), neg_inf, dtype=dtype, device=device) for _ in range(graph.node_count)]
    insert = [torch.full((reference.state_count + 1,), neg_inf, dtype=dtype, device=device) for _ in range(graph.node_count)]
    delete = [torch.full((reference.state_count,), neg_inf, dtype=dtype, device=device) for _ in range(graph.node_count)]

    source_set = set(reference.source_nodes)
    for node_id in graph.topo_order.astype("int64", copy=False).tolist():
        if node_id in source_set:
            node_match, node_insert, node_delete = source_forward_arrays(reference, int(node_id))
        else:
            prev_match, prev_insert, prev_delete = aggregate_parent_arrays(
                reference, int(node_id), match, insert, delete
            )
            node_match = torch.full((reference.state_count,), neg_inf, dtype=dtype, device=device)
            node_insert = torch.full((reference.state_count + 1,), neg_inf, dtype=dtype, device=device)
            for state in range(reference.state_count + 1):
                emit = reference.insert_emission[int(node_id)][state]
                if not torch.isfinite(emit):
                    continue
                candidates = [prev_insert[state] + reference.transitions.insert_to_insert[state]]
                if state > 0:
                    candidates.extend(
                        [
                            prev_match[state - 1] + reference.transitions.match_to_insert[state - 1],
                            prev_delete[state - 1] + reference.transitions.delete_to_insert[state - 1],
                        ]
                    )
                node_insert[state] = emit + smoothmax(candidates, temperature)
            for state in range(reference.state_count):
                emit = reference.match_emission[int(node_id)][state]
                if not torch.isfinite(emit):
                    continue
                candidates = [prev_insert[state] + reference.transitions.insert_to_match[state]]
                if state > 0:
                    candidates.extend(
                        [
                            prev_match[state - 1] + reference.transitions.match_to_match[state - 1],
                            prev_delete[state - 1] + reference.transitions.delete_to_match[state - 1],
                        ]
                    )
                node_match[state] = emit + smoothmax(candidates, temperature)
            node_delete = forward_local_delete_chain(node_match, node_insert, reference.transitions, temperature)
        match[int(node_id)] = node_match
        insert[int(node_id)] = node_insert
        delete[int(node_id)] = node_delete

    end_match, end_insert, end_delete = sink_end_closure(reference.transitions, reference.state_count)
    sink_scores = []
    for node_id in reference.sink_nodes:
        sink_scores.append(
            smoothmax(
                torch.cat(
                    [
                        match[node_id] + end_match,
                        insert[node_id] + end_insert,
                        delete[node_id] + end_delete,
                    ]
                ),
                temperature,
            )
        )
    score = smoothmax(torch.stack(sink_scores), temperature)
    return TorchSoftViterbiResult(score=score, metadata={"graph_id": graph.metadata.graph_id})


def compute_loss(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    reference_parameters: PhmmParameterSet,
    *,
    scheduled_loss_weights: Mapping[str, float],
    temperature: float,
    effective_support: Optional[EffectiveStateMask],
    device: torch.device,
    dtype: torch.dtype = torch.float64,
) -> TorchLossSummary:
    forward = forward_log_likelihood(
        graph,
        parameters,
        effective_support=effective_support,
        device=device,
        dtype=dtype,
    )
    backward = backward_log_likelihood(
        graph,
        parameters,
        effective_support=effective_support,
        device=device,
        dtype=dtype,
    )
    posterior = posterior_occupancy(
        graph,
        parameters,
        forward_result=forward,
        backward_result=backward,
        device=device,
        dtype=dtype,
    )
    soft_viterbi = soft_viterbi_score(
        graph,
        parameters,
        temperature=temperature,
        effective_support=effective_support,
        device=device,
        dtype=dtype,
    )
    reference_torch = _fixed_parameters(reference_parameters, device=device, dtype=dtype)
    n_emit = max(1, graph.node_count)
    nll = -forward.log_likelihood / float(n_emit)
    entropy = soft_alignment_entropy(posterior)
    pairwise = soft_pairwise_score(
        posterior,
        torch_default_pairwise_score_matrix(posterior.alphabet_size, device=device, dtype=dtype),
    )
    soft_viterbi_component = -soft_viterbi.score / float(n_emit)
    transition_anchor = transition_anchor_regularization(parameters, reference_torch)
    emission_anchor = emission_anchor_regularization(parameters, reference_torch)
    emission_smooth = emission_smoothness_regularization(parameters)
    logit_l2 = logit_l2_regularization(parameters)
    active_state_penalty = torch.tensor(
        _active_state_fraction(posterior.posterior_support, int(graph.metadata.global_state_count or 0)),
        dtype=dtype,
        device=device,
    )

    total_loss = (
        float(scheduled_loss_weights["negative_log_likelihood"]) * nll
        + float(scheduled_loss_weights["soft_viterbi"]) * soft_viterbi_component
        + float(scheduled_loss_weights["entropy"]) * entropy
        - float(scheduled_loss_weights["pairwise"]) * pairwise
        + float(scheduled_loss_weights["transition_anchor"]) * transition_anchor
        + float(scheduled_loss_weights["emission_anchor"]) * emission_anchor
        + float(scheduled_loss_weights["emission_smooth"]) * emission_smooth
        + float(scheduled_loss_weights["logit_l2"]) * logit_l2
        + float(scheduled_loss_weights["active_state_penalty"]) * active_state_penalty
    )
    return TorchLossSummary(
        loss=total_loss,
        loss_components={
            "negative_log_likelihood": float(nll.detach().cpu().item()),
            "soft_viterbi": float(soft_viterbi_component.detach().cpu().item()),
            "entropy": float(entropy.detach().cpu().item()),
            "pairwise": float(pairwise.detach().cpu().item()),
            "transition_anchor": float(transition_anchor.detach().cpu().item()),
            "emission_anchor": float(emission_anchor.detach().cpu().item()),
            "emission_smooth": float(emission_smooth.detach().cpu().item()),
            "logit_l2": float(logit_l2.detach().cpu().item()),
            "active_state_penalty": float(active_state_penalty.detach().cpu().item()),
        },
        metrics={
            "log_likelihood": float(forward.log_likelihood.detach().cpu().item()),
            "backward_log_likelihood": float(backward.log_likelihood.detach().cpu().item()),
            "likelihood_gap": float(
                torch.abs(forward.log_likelihood - backward.log_likelihood).detach().cpu().item()
            ),
            "soft_viterbi_score": float(soft_viterbi.score.detach().cpu().item()),
        },
    )


def _fixed_parameters(
    parameters: PhmmParameterSet,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PhmmParameterSet:
    return PhmmParameterSet(
        match_emission=torch.as_tensor(parameters.match_emission, dtype=dtype, device=device),
        insert_emission=torch.as_tensor(parameters.insert_emission, dtype=dtype, device=device),
        transition_logits=parameters.transition_logits.__class__(
            tensor=torch.as_tensor(parameters.transition_logits.tensor, dtype=dtype, device=device),
            order=tuple(parameters.transition_logits.order),
        ),
        metadata=dict(parameters.metadata),
    )


def source_forward_arrays(
    problem: TorchDenseReferenceProblem,
    node_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    transitions = problem.transitions
    state_count = problem.state_count
    device = problem.device
    dtype = problem.dtype
    neg_inf = float("-inf")
    match = torch.full((state_count,), neg_inf, dtype=dtype, device=device)
    insert = torch.full((state_count + 1,), neg_inf, dtype=dtype, device=device)
    delete = torch.full((state_count,), neg_inf, dtype=dtype, device=device)
    if state_count == 0:
        return match, insert, delete

    delete_seed = torch.full((state_count,), neg_inf, dtype=dtype, device=device)
    delete_seed[0] = transitions.start_delete
    for state in range(1, state_count):
        delete_seed[state] = delete_seed[state - 1] + transitions.delete_to_delete[state - 1]

    match_emit = problem.match_emission[node_id]
    insert_emit = problem.insert_emission[node_id]
    if torch.isfinite(match_emit[0]):
        match[0] = transitions.start_match + match_emit[0]
    if torch.isfinite(insert_emit[0]):
        insert[0] = transitions.start_insert + insert_emit[0]
    for state in range(1, state_count):
        if torch.isfinite(match_emit[state]):
            match[state] = (
                delete_seed[state - 1] + transitions.delete_to_match[state - 1] + match_emit[state]
            )
        if torch.isfinite(insert_emit[state]):
            insert[state] = (
                delete_seed[state - 1] + transitions.delete_to_insert[state - 1] + insert_emit[state]
            )
    if torch.isfinite(insert_emit[state_count]):
        insert[state_count] = (
            delete_seed[state_count - 1]
            + transitions.delete_to_insert[state_count - 1]
            + insert_emit[state_count]
        )
    delete = forward_local_delete_chain(match, insert, transitions, temperature=None)
    return match, insert, delete


def forward_node_arrays(
    prev_match: torch.Tensor,
    prev_insert: torch.Tensor,
    prev_delete: torch.Tensor,
    match_emission: torch.Tensor,
    insert_emission: torch.Tensor,
    transitions: TorchTransitionBundle,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_count = int(match_emission.shape[0])
    device = match_emission.device
    dtype = match_emission.dtype
    neg_inf = float("-inf")
    match = torch.full((state_count,), neg_inf, dtype=dtype, device=device)
    insert = torch.full((state_count + 1,), neg_inf, dtype=dtype, device=device)
    for state in range(state_count + 1):
        emit = insert_emission[state]
        if not torch.isfinite(emit):
            continue
        candidates = [prev_insert[state] + transitions.insert_to_insert[state]]
        if state > 0:
            candidates.extend(
                [
                    prev_match[state - 1] + transitions.match_to_insert[state - 1],
                    prev_delete[state - 1] + transitions.delete_to_insert[state - 1],
                ]
            )
        insert[state] = emit + torch_logsumexp(torch.stack(candidates))
    for state in range(state_count):
        emit = match_emission[state]
        if not torch.isfinite(emit):
            continue
        candidates = [prev_insert[state] + transitions.insert_to_match[state]]
        if state > 0:
            candidates.extend(
                [
                    prev_match[state - 1] + transitions.match_to_match[state - 1],
                    prev_delete[state - 1] + transitions.delete_to_match[state - 1],
                ]
            )
        match[state] = emit + torch_logsumexp(torch.stack(candidates))
    delete = forward_local_delete_chain(match, insert, transitions, temperature=None)
    return match, insert, delete


def forward_local_delete_chain(
    match: torch.Tensor,
    insert: torch.Tensor,
    transitions: TorchTransitionBundle,
    temperature: Optional[float],
) -> torch.Tensor:
    state_count = int(match.shape[0])
    delete = torch.full_like(match, float("-inf"))
    for state in range(state_count):
        candidates = [insert[state] + transitions.insert_to_delete[state]]
        if state > 0:
            candidates.extend(
                [
                    match[state - 1] + transitions.match_to_delete[state - 1],
                    delete[state - 1] + transitions.delete_to_delete[state - 1],
                ]
            )
        reducer = smoothmax if temperature is not None else torch_logsumexp
        delete[state] = reducer(torch.stack(candidates), temperature) if temperature is not None else reducer(torch.stack(candidates))
    return delete


def sink_end_closure(
    transitions: TorchTransitionBundle,
    state_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = transitions.start_match.device
    dtype = transitions.start_match.dtype
    neg_inf = float("-inf")
    match = torch.full((state_count,), neg_inf, dtype=dtype, device=device)
    insert = torch.full((state_count + 1,), neg_inf, dtype=dtype, device=device)
    delete = torch.full((state_count,), neg_inf, dtype=dtype, device=device)
    if state_count == 0:
        return match, insert, delete
    delete[state_count - 1] = transitions.delete_to_end
    for state in range(state_count - 2, -1, -1):
        delete[state] = delete[state + 1] + transitions.delete_to_delete[state]
    match[state_count - 1] = transitions.match_to_end
    for state in range(state_count - 1):
        match[state] = delete[state + 1] + transitions.match_to_delete[state]
    insert[state_count] = transitions.insert_to_end
    for state in range(state_count):
        insert[state] = delete[state] + transitions.insert_to_delete[state]
    return match, insert, delete


def backward_node_arrays(
    next_match: torch.Tensor,
    next_insert: torch.Tensor,
    transitions: TorchTransitionBundle,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    state_count = int(next_match.shape[0])
    device = next_match.device
    dtype = next_match.dtype
    delete = torch.full((state_count,), float("-inf"), dtype=dtype, device=device)
    insert = torch.full((state_count + 1,), float("-inf"), dtype=dtype, device=device)
    match = torch.full((state_count,), float("-inf"), dtype=dtype, device=device)
    for state in range(state_count - 1, -1, -1):
        candidates = [next_insert[state + 1] + transitions.delete_to_insert[state]]
        if state < state_count - 1:
            candidates.extend(
                [
                    next_match[state + 1] + transitions.delete_to_match[state],
                    delete[state + 1] + transitions.delete_to_delete[state],
                ]
            )
        delete[state] = torch_logsumexp(torch.stack(candidates))
    insert[state_count] = transitions.insert_to_end
    for state in range(state_count - 1, -1, -1):
        insert[state] = torch_logsumexp(
            torch.stack(
                [
                    next_insert[state] + transitions.insert_to_insert[state],
                    next_match[state] + transitions.insert_to_match[state],
                    delete[state] + transitions.insert_to_delete[state],
                ]
            )
        )
    if state_count > 0:
        match[state_count - 1] = torch_logsumexp(
            torch.stack(
                [
                    transitions.match_to_end,
                    next_insert[state_count] + transitions.match_to_insert[state_count - 1],
                ]
            )
        )
        for state in range(state_count - 2, -1, -1):
            match[state] = torch_logsumexp(
                torch.stack(
                    [
                        next_match[state + 1] + transitions.match_to_match[state],
                        next_insert[state + 1] + transitions.match_to_insert[state],
                        delete[state + 1] + transitions.match_to_delete[state],
                    ]
                )
            )
    return match, insert, delete


def aggregate_parent_arrays(
    problem: TorchDenseReferenceProblem,
    node_id: int,
    match: list[torch.Tensor],
    insert: list[torch.Tensor],
    delete: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not problem.incoming_edges[node_id]:
        return (
            torch.full_like(match[0], float("-inf")),
            torch.full_like(insert[0], float("-inf")),
            torch.full_like(delete[0], float("-inf")),
        )
    prev_match = torch_logsumexp(
        torch.stack(
            [
                match[int(problem.graph.edge_src[edge_id])] + problem.edge_log_weights[edge_id]
                for edge_id in problem.incoming_edges[node_id]
            ]
        ),
        dim=0,
    )
    prev_insert = torch_logsumexp(
        torch.stack(
            [
                insert[int(problem.graph.edge_src[edge_id])] + problem.edge_log_weights[edge_id]
                for edge_id in problem.incoming_edges[node_id]
            ]
        ),
        dim=0,
    )
    prev_delete = torch_logsumexp(
        torch.stack(
            [
                delete[int(problem.graph.edge_src[edge_id])] + problem.edge_log_weights[edge_id]
                for edge_id in problem.incoming_edges[node_id]
            ]
        ),
        dim=0,
    )
    return prev_match, prev_insert, prev_delete


def aggregate_child_arrays(
    problem: TorchDenseReferenceProblem,
    node_id: int,
    beta_match: list[torch.Tensor],
    beta_insert: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not problem.outgoing_edges[node_id]:
        return (
            torch.full_like(beta_match[0], float("-inf")),
            torch.full_like(beta_insert[0], float("-inf")),
        )
    next_match = torch_logsumexp(
        torch.stack(
            [
                beta_match[int(problem.graph.edge_dst[edge_id])]
                + problem.match_emission[int(problem.graph.edge_dst[edge_id])]
                + problem.edge_log_weights[edge_id]
                for edge_id in problem.outgoing_edges[node_id]
            ]
        ),
        dim=0,
    )
    next_insert = torch_logsumexp(
        torch.stack(
            [
                beta_insert[int(problem.graph.edge_dst[edge_id])]
                + problem.insert_emission[int(problem.graph.edge_dst[edge_id])]
                + problem.edge_log_weights[edge_id]
                for edge_id in problem.outgoing_edges[node_id]
            ]
        ),
        dim=0,
    )
    return next_match, next_insert


def soft_alignment_entropy(
    posterior: TorchPosteriorOccupancy,
    *,
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    counts = soft_column_counts(posterior)
    if counts.numel() == 0:
        return counts.new_zeros(())
    column_mass = counts.sum(dim=1)
    active = column_mass > eps
    if not torch.any(active):
        return counts.new_zeros(())
    probs = counts[active] / torch.clamp(column_mass[active].unsqueeze(1), min=eps)
    entropy = -(probs * torch.log(torch.clamp(probs, min=eps))).sum(dim=1)
    value = entropy.mean()
    if normalize:
        value = value / torch.log(torch.tensor(float(posterior.alphabet_size), dtype=value.dtype, device=value.device))
    return value


def soft_pairwise_score(
    posterior: TorchPosteriorOccupancy,
    score_matrix: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    counts = soft_column_counts(posterior)
    diagonal = torch.diagonal(score_matrix)
    numerator = counts.new_zeros(())
    denominator = counts.new_zeros(())
    for column_counts in counts:
        mass = column_counts.sum()
        if float(mass.detach().cpu().item()) <= eps:
            continue
        diag_score = (
            0.5 * column_counts * torch.clamp(column_counts - 1.0, min=0.0) * diagonal
        ).sum()
        outer = torch.outer(column_counts, column_counts)
        off_score = torch.triu(outer * score_matrix, diagonal=1).sum()
        numerator = numerator + diag_score + off_score
        denominator = denominator + torch.clamp(mass * (mass - 1.0), min=0.0)
    if float(denominator.detach().cpu().item()) <= eps:
        return counts.new_zeros(())
    return 2.0 * numerator / denominator


def soft_column_counts(posterior: TorchPosteriorOccupancy) -> torch.Tensor:
    if len(posterior.match_posterior) == 0:
        return torch.zeros((0, posterior.alphabet_size), dtype=posterior.log_likelihood.dtype, device=posterior.log_likelihood.device)
    state_count = int(posterior.match_posterior[0].shape[0])
    counts = torch.zeros(
        (state_count, posterior.alphabet_size),
        dtype=posterior.match_posterior[0].dtype,
        device=posterior.match_posterior[0].device,
    )
    for node_id, symbol in enumerate(posterior.node_symbol.tolist()):
        counts[:, int(symbol)] = counts[:, int(symbol)] + posterior.match_posterior[node_id]
    return counts


def transition_anchor_regularization(
    parameters: PhmmParameterSet,
    reference_parameters: PhmmParameterSet,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    current = unpack_transition_bundle(
        parameters,
        device=parameters.transition_logits.tensor.device,
        dtype=parameters.transition_logits.tensor.dtype,
    )
    reference = unpack_transition_bundle(
        reference_parameters,
        device=parameters.transition_logits.tensor.device,
        dtype=parameters.transition_logits.tensor.dtype,
    )
    families = [
        (
            torch.exp(torch.stack([reference.start_match, reference.start_delete, reference.start_insert])),
            torch.exp(torch.stack([current.start_match, current.start_delete, current.start_insert])),
        )
    ]
    for state in range(max(0, int(reference.match_to_insert.shape[0]) - 1)):
        families.append(
            (
                torch.exp(torch.stack([reference.match_to_match[state], reference.match_to_delete[state], reference.match_to_insert[state]])),
                torch.exp(torch.stack([current.match_to_match[state], current.match_to_delete[state], current.match_to_insert[state]])),
            )
        )
        families.append(
            (
                torch.exp(torch.stack([reference.insert_to_match[state], reference.insert_to_delete[state], reference.insert_to_insert[state]])),
                torch.exp(torch.stack([current.insert_to_match[state], current.insert_to_delete[state], current.insert_to_insert[state]])),
            )
        )
        families.append(
            (
                torch.exp(torch.stack([reference.delete_to_match[state], reference.delete_to_delete[state], reference.delete_to_insert[state]])),
                torch.exp(torch.stack([current.delete_to_match[state], current.delete_to_delete[state], current.delete_to_insert[state]])),
            )
        )
    if int(reference.match_to_insert.shape[0]) > 0:
        last = int(reference.match_to_insert.shape[0]) - 1
        families.append(
            (
                torch.exp(torch.stack([reference.match_to_end, reference.match_to_insert[last]])),
                torch.exp(torch.stack([current.match_to_end, current.match_to_insert[last]])),
            )
        )
        families.append(
            (
                torch.exp(torch.stack([reference.insert_to_end, reference.insert_to_insert[last + 1]])),
                torch.exp(torch.stack([current.insert_to_end, current.insert_to_insert[last + 1]])),
            )
        )
        families.append(
            (
                torch.exp(torch.stack([reference.delete_to_end, reference.delete_to_insert[last]])),
                torch.exp(torch.stack([current.delete_to_end, current.delete_to_insert[last]])),
            )
        )
    return torch.stack([_kl_divergence(ref, cur, eps=eps) for ref, cur in families]).mean()


def emission_anchor_regularization(
    parameters: PhmmParameterSet,
    reference_parameters: PhmmParameterSet,
    *,
    insert_background_weight: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    match_probs = torch.softmax(parameters.match_emission, dim=1)
    reference_match = torch.softmax(reference_parameters.match_emission, dim=1)
    insert_probs = torch.softmax(parameters.insert_emission, dim=1)
    background = torch.full(
        (insert_probs.shape[1],),
        1.0 / float(insert_probs.shape[1]),
        dtype=insert_probs.dtype,
        device=insert_probs.device,
    )
    match_term = torch.stack(
        [_kl_divergence(reference_match[state], match_probs[state], eps=eps) for state in range(match_probs.shape[0])]
    ).mean()
    insert_term = torch.stack(
        [_kl_divergence(background, insert_probs[state], eps=eps) for state in range(insert_probs.shape[0])]
    ).mean()
    return match_term + float(insert_background_weight) * insert_term


def emission_smoothness_regularization(
    parameters: PhmmParameterSet,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    match_probs = torch.softmax(parameters.match_emission, dim=1)
    if int(match_probs.shape[0]) < 2:
        return match_probs.new_zeros(())
    penalties = []
    for state in range(int(match_probs.shape[0]) - 1):
        left = match_probs[state]
        right = match_probs[state + 1]
        midpoint = 0.5 * (left + right)
        penalties.append(
            0.5 * _kl_divergence(left, midpoint, eps=eps)
            + 0.5 * _kl_divergence(right, midpoint, eps=eps)
        )
    return torch.stack(penalties).mean()


def logit_l2_regularization(parameters: PhmmParameterSet) -> torch.Tensor:
    transition = parameters.transition_logits.tensor
    match = parameters.match_emission
    insert = parameters.insert_emission
    return torch.mean(transition**2) + torch.mean(match**2) + torch.mean(insert**2)


def torch_default_pairwise_score_matrix(
    alphabet_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    matrix = torch.full((alphabet_size, alphabet_size), -1.0, dtype=dtype, device=device)
    matrix.fill_diagonal_(1.0)
    return matrix


def smoothmax(values: torch.Tensor | list[torch.Tensor], temperature: float) -> torch.Tensor:
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if isinstance(values, list):
        values = torch.stack(values)
    return temperature * torch.logsumexp(values / temperature, dim=0)


def torch_logsumexp(values: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    if dim is None:
        return torch.logsumexp(values.reshape(-1), dim=0)
    return torch.logsumexp(values, dim=dim)


def _kl_divergence(reference: torch.Tensor, target: torch.Tensor, *, eps: float) -> torch.Tensor:
    mask = reference > eps
    if not torch.any(mask):
        return reference.new_zeros(())
    ref = reference[mask]
    tgt = torch.clamp(target[mask], min=eps)
    return torch.sum(ref * (torch.log(ref) - torch.log(tgt)))


def _active_state_fraction(active_support: EffectiveStateMask, global_state_count: int) -> float:
    if global_state_count <= 0:
        return 0.0
    active = active_support.active_global_state_ids()
    if active.size == 0:
        return 0.0
    return float(len(set(int(value) for value in active.tolist())) / float(global_state_count))


def _edges_by_node(index_array, edge_count: int, node_count: int) -> tuple[tuple[int, ...], ...]:
    buckets = [[] for _ in range(node_count)]
    for edge_id in range(edge_count):
        buckets[int(index_array[edge_id])].append(int(edge_id))
    return tuple(tuple(bucket) for bucket in buckets)
