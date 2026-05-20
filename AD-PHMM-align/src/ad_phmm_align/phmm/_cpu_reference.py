"""Dense NumPy reference helpers for small-graph CPU PHMM dynamic programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.phmm.parameters import PhmmParameterSet

CHANNEL_MATCH = 0
CHANNEL_DELETE = 1
CHANNEL_INSERT = 2
CHANNEL_NAMES = {
    CHANNEL_MATCH: "M",
    CHANNEL_DELETE: "D",
    CHANNEL_INSERT: "I",
}
START_SENTINEL_EDGE = -2
INTERNAL_EDGE = -1


@dataclass(frozen=True)
class TransitionBundle:
    """Normalized log-transition probabilities unpacked from packed logits."""

    start_match: float
    start_delete: float
    start_insert: float
    match_to_match: np.ndarray
    match_to_delete: np.ndarray
    match_to_insert: np.ndarray
    match_to_end: float
    delete_to_match: np.ndarray
    delete_to_delete: np.ndarray
    delete_to_insert: np.ndarray
    delete_to_end: float
    insert_to_match: np.ndarray
    insert_to_delete: np.ndarray
    insert_to_insert: np.ndarray
    insert_to_end: float


@dataclass(frozen=True)
class DenseReferenceProblem:
    """Dense reference tensors derived from one graph/parameter pair."""

    graph: TensorDag
    transitions: TransitionBundle
    edge_log_weights: np.ndarray
    incoming_edges: tuple[tuple[int, ...], ...]
    outgoing_edges: tuple[tuple[int, ...], ...]
    source_nodes: np.ndarray
    sink_nodes: np.ndarray
    match_emission: np.ndarray
    insert_emission: np.ndarray
    state_count: int
    alphabet_size: int


def build_dense_reference_problem(
    graph: TensorDag,
    parameters: PhmmParameterSet,
) -> DenseReferenceProblem:
    """Build dense NumPy inputs for CPU reference dynamic programs."""

    transitions = unpack_transition_bundle(parameters)
    state_count = int(np.asarray(parameters.match_emission).shape[0])
    alphabet_size = int(np.asarray(parameters.match_emission).shape[1])
    incoming_edges = _edges_by_node(graph.edge_dst, graph.edge_count, graph.node_count)
    outgoing_edges = _edges_by_node(graph.edge_src, graph.edge_count, graph.node_count)
    edge_log_weights = normalized_incoming_edge_log_weights(graph)
    source_nodes = np.flatnonzero(np.bincount(graph.edge_dst.astype(np.int64), minlength=graph.node_count) == 0)
    sink_nodes = np.flatnonzero(np.bincount(graph.edge_src.astype(np.int64), minlength=graph.node_count) == 0)
    match_emission, insert_emission = build_dense_emission_tables(
        graph,
        parameters,
        state_count=state_count,
    )
    return DenseReferenceProblem(
        graph=graph,
        transitions=transitions,
        edge_log_weights=edge_log_weights,
        incoming_edges=incoming_edges,
        outgoing_edges=outgoing_edges,
        source_nodes=source_nodes.astype(np.int64, copy=False),
        sink_nodes=sink_nodes.astype(np.int64, copy=False),
        match_emission=match_emission,
        insert_emission=insert_emission,
        state_count=state_count,
        alphabet_size=alphabet_size,
    )


def unpack_transition_bundle(parameters: PhmmParameterSet) -> TransitionBundle:
    """Normalize packed transition logits into per-family log probabilities."""

    tensor = np.asarray(parameters.transition_logits.tensor, dtype=np.float64)
    order = tuple(parameters.transition_logits.order)
    lookup = {
        name: np.asarray(parameters.transition_logits.require(name), dtype=np.float64)
        for name in order
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

    start = log_softmax(np.array([match_logits[0], delete_logits[0], insert_logits[0]], dtype=np.float64))
    match_to_match = np.full((max(0, state_count - 1),), -np.inf, dtype=np.float64)
    match_to_delete = np.full((max(0, state_count - 1),), -np.inf, dtype=np.float64)
    match_to_insert = np.full((state_count,), -np.inf, dtype=np.float64)
    if state_count > 1:
        for state in range(state_count - 1):
            family = log_softmax(
                np.array(
                    [
                        match_logits[state + 1],
                        delete_logits[state + 1],
                        insert_logits[state + 1],
                    ],
                    dtype=np.float64,
                )
            )
            match_to_match[state] = family[0]
            match_to_delete[state] = family[1]
            match_to_insert[state] = family[2]
    if state_count > 0:
        match_tail = log_softmax(
            np.array([match_logits[state_count], insert_logits[state_count]], dtype=np.float64)
        )
        match_to_end = float(match_tail[0])
        match_to_insert[state_count - 1] = float(match_tail[1])
    else:
        match_to_end = -np.inf

    insert_to_match = np.full((state_count,), -np.inf, dtype=np.float64)
    insert_to_delete = np.full((state_count,), -np.inf, dtype=np.float64)
    insert_to_insert = np.full((state_count + 1,), -np.inf, dtype=np.float64)
    for state in range(state_count):
        family = log_softmax(
            np.array(
                [i_match_logits[state], i_delete_logits[state], i_insert_logits[state]],
                dtype=np.float64,
            )
        )
        insert_to_match[state] = family[0]
        insert_to_delete[state] = family[1]
        insert_to_insert[state] = family[2]
    if state_count >= 0:
        insert_tail = log_softmax(
            np.array([i_match_logits[state_count], i_insert_logits[state_count]], dtype=np.float64)
        )
        insert_to_end = float(insert_tail[0])
        insert_to_insert[state_count] = float(insert_tail[1])

    delete_to_match = np.full((max(0, state_count - 1),), -np.inf, dtype=np.float64)
    delete_to_delete = np.full((max(0, state_count - 1),), -np.inf, dtype=np.float64)
    delete_to_insert = np.full((state_count,), -np.inf, dtype=np.float64)
    if state_count > 1:
        for state in range(state_count - 1):
            family = log_softmax(
                np.array(
                    [
                        d_match_logits[state + 1],
                        d_delete_logits[state + 1],
                        d_insert_logits[state + 1],
                    ],
                    dtype=np.float64,
                )
            )
            delete_to_match[state] = family[0]
            delete_to_delete[state] = family[1]
            delete_to_insert[state] = family[2]
    if state_count > 0:
        delete_tail = log_softmax(
            np.array([d_match_logits[state_count], d_insert_logits[state_count]], dtype=np.float64)
        )
        delete_to_end = float(delete_tail[0])
        delete_to_insert[state_count - 1] = float(delete_tail[1])
    else:
        delete_to_end = -np.inf

    return TransitionBundle(
        start_match=float(start[0]),
        start_delete=float(start[1]),
        start_insert=float(start[2]),
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
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast graph symbols against PHMM emissions on the global state axis."""

    match_tensor = np.asarray(parameters.match_emission, dtype=np.float64)
    insert_tensor = np.asarray(parameters.insert_emission, dtype=np.float64)
    node_symbols = graph.node_symbol.astype(np.int64, copy=False)
    alphabet_size = int(match_tensor.shape[1])
    if np.any(node_symbols < 0) or np.any(node_symbols >= alphabet_size):
        raise ValueError("node symbols exceed the PHMM emission alphabet")

    match_emission = np.full((graph.node_count, state_count), -np.inf, dtype=np.float64)
    insert_emission = np.full((graph.node_count, state_count + 1), -np.inf, dtype=np.float64)
    for node_id in range(graph.node_count):
        symbol = int(node_symbols[node_id])
        left = int(graph.node_window_left[node_id])
        right = int(graph.node_window_right[node_id])
        if right <= left:
            continue
        match_emission[node_id, left:right] = match_tensor[left:right, symbol]
        insert_emission[node_id, left : right + 1] = insert_tensor[left : right + 1, symbol]
    return match_emission, insert_emission


def normalized_incoming_edge_log_weights(graph: TensorDag) -> np.ndarray:
    """Normalize edge weights by destination-node incoming totals in log space."""

    edge_weight = np.asarray(graph.edge_weight, dtype=np.float64)
    if np.any(edge_weight <= 0.0):
        raise ValueError("graph edge weights must be positive for CPU reference DP")
    incoming_sum = np.zeros((graph.node_count,), dtype=np.float64)
    for edge_id in range(graph.edge_count):
        incoming_sum[int(graph.edge_dst[edge_id])] += edge_weight[edge_id]
    edge_log_weights = np.full((graph.edge_count,), -np.inf, dtype=np.float64)
    for edge_id in range(graph.edge_count):
        destination = int(graph.edge_dst[edge_id])
        edge_log_weights[edge_id] = np.log(edge_weight[edge_id]) - np.log(incoming_sum[destination])
    return edge_log_weights


def source_forward_arrays(
    problem: DenseReferenceProblem,
    node_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source-node forward arrays seeded from start probabilities."""

    state_count = problem.state_count
    transitions = problem.transitions
    match = np.full((state_count,), -np.inf, dtype=np.float64)
    insert = np.full((state_count + 1,), -np.inf, dtype=np.float64)
    delete = np.full((state_count,), -np.inf, dtype=np.float64)
    if state_count == 0:
        return match, insert, delete

    delete_seed = np.full((state_count,), -np.inf, dtype=np.float64)
    delete_seed[0] = transitions.start_delete
    for state in range(1, state_count):
        delete_seed[state] = delete_seed[state - 1] + transitions.delete_to_delete[state - 1]

    match_emit = problem.match_emission[node_id]
    insert_emit = problem.insert_emission[node_id]
    if np.isfinite(match_emit[0]):
        match[0] = transitions.start_match + match_emit[0]
    if np.isfinite(insert_emit[0]):
        insert[0] = transitions.start_insert + insert_emit[0]
    for state in range(1, state_count):
        if np.isfinite(match_emit[state]):
            match[state] = (
                delete_seed[state - 1] + transitions.delete_to_match[state - 1] + match_emit[state]
            )
        if np.isfinite(insert_emit[state]):
            insert[state] = (
                delete_seed[state - 1] + transitions.delete_to_insert[state - 1] + insert_emit[state]
            )
    if np.isfinite(insert_emit[state_count]):
        insert[state_count] = (
            delete_seed[state_count - 1]
            + transitions.delete_to_insert[state_count - 1]
            + insert_emit[state_count]
        )
    delete = forward_local_delete_chain(match, insert, transitions)
    return match, insert, delete


def forward_node_arrays(
    prev_match: np.ndarray,
    prev_insert: np.ndarray,
    prev_delete: np.ndarray,
    match_emission: np.ndarray,
    insert_emission: np.ndarray,
    transitions: TransitionBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one forward update on the dense global-state axis."""

    state_count = int(match_emission.shape[0])
    match = np.full((state_count,), -np.inf, dtype=np.float64)
    insert = np.full((state_count + 1,), -np.inf, dtype=np.float64)
    delete = np.full((state_count,), -np.inf, dtype=np.float64)
    for state in range(state_count + 1):
        if not np.isfinite(insert_emission[state]):
            continue
        candidates = [prev_insert[state] + transitions.insert_to_insert[state]]
        if state > 0:
            candidates.append(prev_match[state - 1] + transitions.match_to_insert[state - 1])
            candidates.append(prev_delete[state - 1] + transitions.delete_to_insert[state - 1])
        insert[state] = insert_emission[state] + logsumexp(candidates)
    for state in range(state_count):
        if not np.isfinite(match_emission[state]):
            continue
        candidates = [prev_insert[state] + transitions.insert_to_match[state]]
        if state > 0:
            candidates.append(prev_match[state - 1] + transitions.match_to_match[state - 1])
            candidates.append(prev_delete[state - 1] + transitions.delete_to_match[state - 1])
        match[state] = match_emission[state] + logsumexp(candidates)
    delete = forward_local_delete_chain(match, insert, transitions)
    return match, insert, delete


def forward_local_delete_chain(
    match: np.ndarray,
    insert: np.ndarray,
    transitions: TransitionBundle,
) -> np.ndarray:
    """Propagate silent delete transitions after one emitted node."""

    state_count = int(match.shape[0])
    delete = np.full((state_count,), -np.inf, dtype=np.float64)
    for state in range(state_count):
        candidates = [insert[state] + transitions.insert_to_delete[state]]
        if state > 0:
            candidates.append(match[state - 1] + transitions.match_to_delete[state - 1])
            candidates.append(delete[state - 1] + transitions.delete_to_delete[state - 1])
        delete[state] = logsumexp(candidates)
    return delete


def sink_end_closure(transitions: TransitionBundle, state_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return silent sink-to-end closures for backward and final scoring."""

    match = np.full((state_count,), -np.inf, dtype=np.float64)
    insert = np.full((state_count + 1,), -np.inf, dtype=np.float64)
    delete = np.full((state_count,), -np.inf, dtype=np.float64)
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
    next_match: np.ndarray,
    next_insert: np.ndarray,
    transitions: TransitionBundle,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one backward update on the dense global-state axis."""

    state_count = int(next_match.shape[0])
    match = np.full((state_count,), -np.inf, dtype=np.float64)
    insert = np.full((state_count + 1,), -np.inf, dtype=np.float64)
    delete = np.full((state_count,), -np.inf, dtype=np.float64)
    for state in range(state_count - 1, -1, -1):
        candidates = [next_insert[state + 1] + transitions.delete_to_insert[state]]
        if state < state_count - 1:
            candidates.append(next_match[state + 1] + transitions.delete_to_match[state])
            candidates.append(delete[state + 1] + transitions.delete_to_delete[state])
        delete[state] = logsumexp(candidates)
    insert[state_count] = transitions.insert_to_end
    for state in range(state_count - 1, -1, -1):
        insert[state] = logsumexp(
            [
                next_insert[state] + transitions.insert_to_insert[state],
                next_match[state] + transitions.insert_to_match[state],
                delete[state] + transitions.insert_to_delete[state],
            ]
        )
    match[state_count - 1] = logsumexp(
        [
            transitions.match_to_end,
            next_insert[state_count] + transitions.match_to_insert[state_count - 1],
        ]
    )
    for state in range(state_count - 2, -1, -1):
        match[state] = logsumexp(
            [
                next_match[state + 1] + transitions.match_to_match[state],
                next_insert[state + 1] + transitions.match_to_insert[state],
                delete[state + 1] + transitions.match_to_delete[state],
            ]
        )
    return match, insert, delete


def aggregate_parent_arrays(
    problem: DenseReferenceProblem,
    node_id: int,
    match: np.ndarray,
    insert: np.ndarray,
    delete: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate predecessor-node outputs into one dense parent state summary."""

    state_count = problem.state_count
    prev_match = np.full((state_count,), -np.inf, dtype=np.float64)
    prev_insert = np.full((state_count + 1,), -np.inf, dtype=np.float64)
    prev_delete = np.full((state_count,), -np.inf, dtype=np.float64)
    for edge_id in problem.incoming_edges[node_id]:
        parent = int(problem.graph.edge_src[edge_id])
        weight = problem.edge_log_weights[edge_id]
        prev_match = np.logaddexp(prev_match, match[parent] + weight)
        prev_insert = np.logaddexp(prev_insert, insert[parent] + weight)
        prev_delete = np.logaddexp(prev_delete, delete[parent] + weight)
    return prev_match, prev_insert, prev_delete


def aggregate_child_arrays(
    problem: DenseReferenceProblem,
    node_id: int,
    beta_match: np.ndarray,
    beta_insert: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate successor-node backward arrays plus child emissions."""

    state_count = problem.state_count
    next_match = np.full((state_count,), -np.inf, dtype=np.float64)
    next_insert = np.full((state_count + 1,), -np.inf, dtype=np.float64)
    for edge_id in problem.outgoing_edges[node_id]:
        child = int(problem.graph.edge_dst[edge_id])
        weight = problem.edge_log_weights[edge_id]
        next_match = np.logaddexp(
            next_match,
            beta_match[child] + problem.match_emission[child] + weight,
        )
        next_insert = np.logaddexp(
            next_insert,
            beta_insert[child] + problem.insert_emission[child] + weight,
        )
    return next_match, next_insert


def logsumexp(values: Iterable[float] | np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Stable log-sum-exp over one array or iterable of scalars."""

    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    if array.size == 0:
        return float(-np.inf)
    if axis is None:
        finite = np.isfinite(array)
        if not np.any(finite):
            return float(-np.inf)
        max_value = np.max(array[finite])
        return float(max_value + np.log(np.sum(np.exp(array[finite] - max_value))))
    max_value = np.max(array, axis=axis, keepdims=True)
    finite = np.isfinite(max_value)
    shifted = np.where(finite, array - max_value, -np.inf)
    summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    out = np.where(finite, max_value + np.log(summed), -np.inf)
    squeezed = np.squeeze(out, axis=axis)
    return squeezed.astype(np.float64, copy=False)


def smoothmax(values: Iterable[float] | np.ndarray, temperature: float) -> float:
    """Temperature-smoothed max reduction."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    finite = np.isfinite(array)
    if not np.any(finite):
        return float(-np.inf)
    scaled = array[finite] / temperature
    max_value = float(np.max(scaled))
    return float(temperature * (max_value + np.log(np.sum(np.exp(scaled - max_value)))))


def pack_state_index(node_id: int, channel: int, state: int, node_count: int, state_count: int) -> int:
    """Encode one dense node/channel/state cell as a stable flat integer."""

    if channel == CHANNEL_MATCH:
        return node_id * state_count + state
    if channel == CHANNEL_DELETE:
        return node_count * state_count + node_id * state_count + state
    return 2 * node_count * state_count + node_id * (state_count + 1) + state


def log_softmax(values: np.ndarray) -> np.ndarray:
    """Normalize one transition family of arbitrary logit values."""

    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.full_like(values, -np.inf, dtype=np.float64)
    max_value = float(np.max(values[finite]))
    shifted = values[finite] - max_value
    normalizer = max_value + np.log(np.sum(np.exp(shifted)))
    out = np.full_like(values, -np.inf, dtype=np.float64)
    out[finite] = values[finite] - normalizer
    return out


def _edges_by_node(edge_nodes: np.ndarray, edge_count: int, node_count: int) -> tuple[tuple[int, ...], ...]:
    grouped = [[] for _ in range(node_count)]
    for edge_id in range(edge_count):
        grouped[int(edge_nodes[edge_id])].append(int(edge_id))
    return tuple(tuple(items) for items in grouped)
