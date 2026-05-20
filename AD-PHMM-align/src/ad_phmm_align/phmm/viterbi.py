"""Dense CPU reference hard-Viterbi decoding."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.phmm._decoding_common import ViterbiInput, prepare_viterbi_input
from ad_phmm_align.phmm._cpu_reference import (
    CHANNEL_DELETE,
    CHANNEL_INSERT,
    CHANNEL_MATCH,
    CHANNEL_NAMES,
    INTERNAL_EDGE,
    START_SENTINEL_EDGE,
    build_dense_reference_problem,
    pack_state_index,
    sink_end_closure,
    source_forward_arrays,
)
from ad_phmm_align.phmm.parameters import PhmmParameterSet
from ad_phmm_align.phmm.ranges import EffectiveStateMask
from ad_phmm_align.phmm.wavefront import WavefrontSchedule

_ASSIGNMENT_CHANNEL_NAME = {
    CHANNEL_MATCH: "match",
    CHANNEL_DELETE: "delete",
    CHANNEL_INSERT: "insert",
}


@dataclass(frozen=True)
class ViterbiBackpointerTable:
    """Sparse backpointer storage for reachable packed cells only."""

    packed_state_index: Any
    predecessor_edge_id: Any
    predecessor_channel: Any
    predecessor_packed_state_index: Any


@dataclass(frozen=True)
class ViterbiNodeAssignment:
    """One graph node's decoded hidden-state assignment for a sequence subset."""

    node_id: int
    channel: str
    state_index: int
    sequence_ids: Optional[Any] = None
    insertion_step: int = 0
    insertion_slot: int = 0


@dataclass(frozen=True)
class ViterbiDecodeResult:
    """Decoded path/state output for evaluation."""

    score: float
    states: Any
    node_ids: Optional[Any] = None
    global_state_ids: Optional[Any] = None
    node_assignments: Optional[Any] = None
    backpointers: Optional[ViterbiBackpointerTable] = None
    effective_support: Optional[EffectiveStateMask] = None
    metadata: Optional[Mapping[str, Any]] = None


def _node_sequence_ids(graph: TensorDag, node_id: int) -> Optional[np.ndarray]:
    if graph.extra is None:
        return None
    if (
        "node_source_offset" not in graph.extra
        or "node_source_len" not in graph.extra
        or "source_sequence_id" not in graph.extra
    ):
        return None
    node_source_offset = np.asarray(graph.extra["node_source_offset"], dtype=np.int64)
    node_source_len = np.asarray(graph.extra["node_source_len"], dtype=np.int64)
    source_sequence_id = np.asarray(graph.extra["source_sequence_id"], dtype=np.int64)
    start = int(node_source_offset[int(node_id)])
    length = int(node_source_len[int(node_id)])
    return np.unique(source_sequence_id[start : start + length]).astype(np.int64, copy=False)


def _sequence_intersection(
    left: Optional[np.ndarray],
    right: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if left is None:
        return right
    if right is None:
        return left
    return np.intersect1d(left, right, assume_unique=False)


def _merge_sequence_sets(
    current: Optional[np.ndarray],
    incoming: Optional[np.ndarray],
) -> tuple[Optional[np.ndarray], bool]:
    if current is None and incoming is None:
        return None, False
    if current is None:
        merged = np.unique(np.asarray(incoming, dtype=np.int64))
        return merged, True
    if incoming is None:
        return current, False
    merged = np.union1d(current, incoming)
    changed = merged.shape[0] != current.shape[0] or not np.array_equal(merged, current)
    return merged.astype(np.int64, copy=False), changed


def _backpointer_for_cell(
    node_id: int,
    channel: int,
    state: int,
    back_edge_m: np.ndarray,
    back_edge_i: np.ndarray,
    back_edge_d: np.ndarray,
    back_channel_m: np.ndarray,
    back_channel_i: np.ndarray,
    back_channel_d: np.ndarray,
    back_state_m: np.ndarray,
    back_state_i: np.ndarray,
    back_state_d: np.ndarray,
) -> tuple[int, int, int]:
    if channel == CHANNEL_MATCH:
        return (
            int(back_edge_m[node_id, state]),
            int(back_channel_m[node_id, state]),
            int(back_state_m[node_id, state]),
        )
    if channel == CHANNEL_DELETE:
        return (
            int(back_edge_d[node_id, state]),
            int(back_channel_d[node_id, state]),
            int(back_state_d[node_id, state]),
        )
    return (
        int(back_edge_i[node_id, state]),
        int(back_channel_i[node_id, state]),
        int(back_state_i[node_id, state]),
    )


def _collect_node_assignments(
    graph: TensorDag,
    best_terminal: tuple[int, int, int],
    back_edge_m: np.ndarray,
    back_edge_i: np.ndarray,
    back_edge_d: np.ndarray,
    back_channel_m: np.ndarray,
    back_channel_i: np.ndarray,
    back_channel_d: np.ndarray,
    back_state_m: np.ndarray,
    back_state_i: np.ndarray,
    back_state_d: np.ndarray,
) -> tuple[ViterbiNodeAssignment, ...]:
    terminal_node, terminal_channel, terminal_state = best_terminal
    if terminal_node < 0 or terminal_state < 0:
        return ()

    worklist: deque[tuple[int, int, int]] = deque()
    cell_sequences: dict[tuple[int, int, int], Optional[np.ndarray]] = {}
    cell_insert_steps: dict[tuple[int, int, int], int] = {}

    terminal_cell = (int(terminal_node), int(terminal_channel), int(terminal_state))
    cell_sequences[terminal_cell] = _node_sequence_ids(graph, int(terminal_node))
    cell_insert_steps[terminal_cell] = 1 if int(terminal_channel) == CHANNEL_INSERT else 0
    worklist.append(terminal_cell)

    while worklist:
        node_id, channel, state = worklist.popleft()
        edge_id, predecessor_channel, predecessor_state = _backpointer_for_cell(
            node_id,
            channel,
            state,
            back_edge_m,
            back_edge_i,
            back_edge_d,
            back_channel_m,
            back_channel_i,
            back_channel_d,
            back_state_m,
            back_state_i,
            back_state_d,
        )
        if edge_id == START_SENTINEL_EDGE or predecessor_channel < 0 or predecessor_state < 0:
            continue
        predecessor_node = (
            int(node_id) if edge_id == INTERNAL_EDGE else int(graph.edge_src[int(edge_id)])
        )
        propagated_sequences = _sequence_intersection(
            cell_sequences[(node_id, channel, state)],
            _node_sequence_ids(graph, predecessor_node),
        )
        if propagated_sequences is not None and propagated_sequences.shape[0] == 0:
            continue
        predecessor_cell = (
            predecessor_node,
            int(predecessor_channel),
            int(predecessor_state),
        )
        merged_sequences, sequence_changed = _merge_sequence_sets(
            cell_sequences.get(predecessor_cell),
            propagated_sequences,
        )
        predecessor_step = (
            int(cell_insert_steps[(node_id, channel, state)]) + 1
            if int(predecessor_channel) == CHANNEL_INSERT
            else 0
        )
        step_changed = predecessor_step > cell_insert_steps.get(predecessor_cell, -1)
        cell_sequences[predecessor_cell] = merged_sequences
        cell_insert_steps[predecessor_cell] = max(
            predecessor_step,
            cell_insert_steps.get(predecessor_cell, 0),
        )
        if sequence_changed or step_changed or predecessor_cell not in worklist:
            worklist.append(predecessor_cell)

    max_insert_step_by_state: dict[int, int] = {}
    for (node_id, channel, state), step in cell_insert_steps.items():
        if channel != CHANNEL_INSERT:
            continue
        max_insert_step_by_state[state] = max(max_insert_step_by_state.get(state, 0), step)

    assignments = []
    for node_id, channel, state in sorted(cell_sequences, key=lambda key: (key[0], key[2], key[1])):
        sequence_ids = cell_sequences[(node_id, channel, state)]
        insertion_step = cell_insert_steps[(node_id, channel, state)]
        insertion_slot = 0
        if channel == CHANNEL_INSERT:
            insertion_slot = max_insert_step_by_state[int(state)] - int(insertion_step)
        assignments.append(
            ViterbiNodeAssignment(
                node_id=int(node_id),
                channel=_ASSIGNMENT_CHANNEL_NAME[int(channel)],
                state_index=int(state),
                sequence_ids=None
                if sequence_ids is None
                else np.asarray(sequence_ids, dtype=np.int64),
                insertion_step=int(insertion_step),
                insertion_slot=int(insertion_slot),
            )
        )
    return tuple(assignments)


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
    reference = build_dense_reference_problem(graph, parameters)
    state_count = reference.state_count
    node_count = graph.node_count
    match = np.full((node_count, state_count), -np.inf, dtype=np.float64)
    insert = np.full((node_count, state_count + 1), -np.inf, dtype=np.float64)
    delete = np.full((node_count, state_count), -np.inf, dtype=np.float64)
    back_edge_m = np.full((node_count, state_count), START_SENTINEL_EDGE, dtype=np.int64)
    back_edge_i = np.full((node_count, state_count + 1), START_SENTINEL_EDGE, dtype=np.int64)
    back_edge_d = np.full((node_count, state_count), START_SENTINEL_EDGE, dtype=np.int64)
    back_channel_m = np.full((node_count, state_count), -1, dtype=np.int8)
    back_channel_i = np.full((node_count, state_count + 1), -1, dtype=np.int8)
    back_channel_d = np.full((node_count, state_count), -1, dtype=np.int8)
    back_state_m = np.full((node_count, state_count), -1, dtype=np.int64)
    back_state_i = np.full((node_count, state_count + 1), -1, dtype=np.int64)
    back_state_d = np.full((node_count, state_count), -1, dtype=np.int64)

    source_set = set(reference.source_nodes.astype(np.int64).tolist())
    for node_id in graph.topo_order.astype(np.int64).tolist():
        if node_id in source_set:
            node_match, node_insert, node_delete = source_forward_arrays(reference, int(node_id))
            match[int(node_id)] = node_match
            insert[int(node_id)] = node_insert
            delete[int(node_id)] = node_delete
            for state in range(state_count):
                if np.isfinite(node_match[state]):
                    back_edge_m[int(node_id), state] = START_SENTINEL_EDGE
                if np.isfinite(node_delete[state]):
                    candidates = [(insert[int(node_id), state] + reference.transitions.insert_to_delete[state], CHANNEL_INSERT, state)]
                    if state > 0:
                        candidates.extend(
                            [
                                (match[int(node_id), state - 1] + reference.transitions.match_to_delete[state - 1], CHANNEL_MATCH, state - 1),
                                (delete[int(node_id), state - 1] + reference.transitions.delete_to_delete[state - 1], CHANNEL_DELETE, state - 1),
                            ]
                        )
                    best = int(np.argmax([score for score, _, _ in candidates]))
                    back_edge_d[int(node_id), state] = INTERNAL_EDGE
                    back_channel_d[int(node_id), state] = int(candidates[best][1])
                    back_state_d[int(node_id), state] = int(candidates[best][2])
            for state in range(state_count + 1):
                if np.isfinite(node_insert[state]):
                    back_edge_i[int(node_id), state] = START_SENTINEL_EDGE
        else:
            prev_match = np.full((state_count,), -np.inf, dtype=np.float64)
            prev_insert = np.full((state_count + 1,), -np.inf, dtype=np.float64)
            prev_delete = np.full((state_count,), -np.inf, dtype=np.float64)
            prev_edge_match = np.full((state_count,), -1, dtype=np.int64)
            prev_edge_insert = np.full((state_count + 1,), -1, dtype=np.int64)
            prev_edge_delete = np.full((state_count,), -1, dtype=np.int64)
            for edge_id in reference.incoming_edges[int(node_id)]:
                parent = int(graph.edge_src[edge_id])
                weight = reference.edge_log_weights[edge_id]
                candidate_match = match[parent] + weight
                candidate_insert = insert[parent] + weight
                candidate_delete = delete[parent] + weight
                mask = candidate_match > prev_match
                prev_match = np.where(mask, candidate_match, prev_match)
                prev_edge_match = np.where(mask, edge_id, prev_edge_match)
                mask = candidate_insert > prev_insert
                prev_insert = np.where(mask, candidate_insert, prev_insert)
                prev_edge_insert = np.where(mask, edge_id, prev_edge_insert)
                mask = candidate_delete > prev_delete
                prev_delete = np.where(mask, candidate_delete, prev_delete)
                prev_edge_delete = np.where(mask, edge_id, prev_edge_delete)
            for state in range(state_count + 1):
                emit = reference.insert_emission[int(node_id), state]
                if not np.isfinite(emit):
                    continue
                candidates = [(prev_insert[state] + reference.transitions.insert_to_insert[state], prev_edge_insert[state], CHANNEL_INSERT, state)]
                if state > 0:
                    candidates.extend(
                        [
                            (prev_match[state - 1] + reference.transitions.match_to_insert[state - 1], prev_edge_match[state - 1], CHANNEL_MATCH, state - 1),
                            (prev_delete[state - 1] + reference.transitions.delete_to_insert[state - 1], prev_edge_delete[state - 1], CHANNEL_DELETE, state - 1),
                        ]
                    )
                best = int(np.argmax([score for score, *_ in candidates]))
                score, edge_id, channel, predecessor_state = candidates[best]
                insert[int(node_id), state] = score + emit
                back_edge_i[int(node_id), state] = int(edge_id)
                back_channel_i[int(node_id), state] = int(channel)
                back_state_i[int(node_id), state] = int(predecessor_state)
            for state in range(state_count):
                emit = reference.match_emission[int(node_id), state]
                if not np.isfinite(emit):
                    continue
                candidates = [(prev_insert[state] + reference.transitions.insert_to_match[state], prev_edge_insert[state], CHANNEL_INSERT, state)]
                if state > 0:
                    candidates.extend(
                        [
                            (prev_match[state - 1] + reference.transitions.match_to_match[state - 1], prev_edge_match[state - 1], CHANNEL_MATCH, state - 1),
                            (prev_delete[state - 1] + reference.transitions.delete_to_match[state - 1], prev_edge_delete[state - 1], CHANNEL_DELETE, state - 1),
                        ]
                    )
                best = int(np.argmax([score for score, *_ in candidates]))
                score, edge_id, channel, predecessor_state = candidates[best]
                match[int(node_id), state] = score + emit
                back_edge_m[int(node_id), state] = int(edge_id)
                back_channel_m[int(node_id), state] = int(channel)
                back_state_m[int(node_id), state] = int(predecessor_state)
            for state in range(state_count):
                candidates = [(insert[int(node_id), state] + reference.transitions.insert_to_delete[state], INTERNAL_EDGE, CHANNEL_INSERT, state)]
                if state > 0:
                    candidates.extend(
                        [
                            (match[int(node_id), state - 1] + reference.transitions.match_to_delete[state - 1], INTERNAL_EDGE, CHANNEL_MATCH, state - 1),
                            (delete[int(node_id), state - 1] + reference.transitions.delete_to_delete[state - 1], INTERNAL_EDGE, CHANNEL_DELETE, state - 1),
                        ]
                    )
                best = int(np.argmax([score for score, *_ in candidates]))
                score, edge_id, channel, predecessor_state = candidates[best]
                delete[int(node_id), state] = score
                back_edge_d[int(node_id), state] = int(edge_id)
                back_channel_d[int(node_id), state] = int(channel)
                back_state_d[int(node_id), state] = int(predecessor_state)

    end_match, end_insert, end_delete = sink_end_closure(reference.transitions, state_count)
    best_score = -np.inf
    best_terminal = (-1, CHANNEL_MATCH, -1)
    for node_id in reference.sink_nodes.astype(np.int64).tolist():
        for state in range(state_count):
            score = match[int(node_id), state] + end_match[state]
            if score > best_score:
                best_score = float(score)
                best_terminal = (int(node_id), CHANNEL_MATCH, state)
            score = delete[int(node_id), state] + end_delete[state]
            if score > best_score:
                best_score = float(score)
                best_terminal = (int(node_id), CHANNEL_DELETE, state)
        for state in range(state_count + 1):
            score = insert[int(node_id), state] + end_insert[state]
            if score > best_score:
                best_score = float(score)
                best_terminal = (int(node_id), CHANNEL_INSERT, state)

    node_path: list[int] = []
    state_path: list[tuple[str, int]] = []
    global_state_ids: list[int] = []
    node_id, channel, state = best_terminal
    while node_id >= 0 and state >= 0:
        node_path.append(int(node_id))
        state_path.append((CHANNEL_NAMES[int(channel)], int(state)))
        global_state_ids.append(int(state))
        if channel == CHANNEL_MATCH:
            edge_id = int(back_edge_m[node_id, state])
            next_channel = int(back_channel_m[node_id, state])
            next_state = int(back_state_m[node_id, state])
        elif channel == CHANNEL_DELETE:
            edge_id = int(back_edge_d[node_id, state])
            next_channel = int(back_channel_d[node_id, state])
            next_state = int(back_state_d[node_id, state])
        else:
            edge_id = int(back_edge_i[node_id, state])
            next_channel = int(back_channel_i[node_id, state])
            next_state = int(back_state_i[node_id, state])
        if edge_id == START_SENTINEL_EDGE:
            break
        if edge_id >= 0:
            node_id = int(graph.edge_src[edge_id])
        channel = next_channel
        state = next_state
    node_path.reverse()
    state_path.reverse()
    global_state_ids.reverse()
    node_assignments = _collect_node_assignments(
        graph,
        best_terminal,
        back_edge_m,
        back_edge_i,
        back_edge_d,
        back_channel_m,
        back_channel_i,
        back_channel_d,
        back_state_m,
        back_state_i,
        back_state_d,
    )

    current_index = []
    predecessor_edge = []
    predecessor_channel = []
    predecessor_index = []
    for node_id in range(node_count):
        for state in range(state_count):
            if np.isfinite(match[node_id, state]):
                current_index.append(pack_state_index(node_id, CHANNEL_MATCH, state, node_count, state_count))
                predecessor_edge.append(int(back_edge_m[node_id, state]))
                predecessor_channel.append(int(back_channel_m[node_id, state]))
                predecessor_index.append(
                    -1
                    if back_state_m[node_id, state] < 0
                    else pack_state_index(
                        node_id if back_edge_m[node_id, state] == INTERNAL_EDGE else int(graph.edge_src[back_edge_m[node_id, state]]),
                        int(back_channel_m[node_id, state]),
                        int(back_state_m[node_id, state]),
                        node_count,
                        state_count,
                    )
                )
            if np.isfinite(delete[node_id, state]):
                current_index.append(pack_state_index(node_id, CHANNEL_DELETE, state, node_count, state_count))
                predecessor_edge.append(int(back_edge_d[node_id, state]))
                predecessor_channel.append(int(back_channel_d[node_id, state]))
                predecessor_index.append(
                    -1
                    if back_state_d[node_id, state] < 0
                    else pack_state_index(
                        node_id if back_edge_d[node_id, state] == INTERNAL_EDGE else int(graph.edge_src[back_edge_d[node_id, state]]),
                        int(back_channel_d[node_id, state]),
                        int(back_state_d[node_id, state]),
                        node_count,
                        state_count,
                    )
                )
        for state in range(state_count + 1):
            if np.isfinite(insert[node_id, state]):
                current_index.append(pack_state_index(node_id, CHANNEL_INSERT, state, node_count, state_count))
                predecessor_edge.append(int(back_edge_i[node_id, state]))
                predecessor_channel.append(int(back_channel_i[node_id, state]))
                predecessor_index.append(
                    -1
                    if back_state_i[node_id, state] < 0
                    else pack_state_index(
                        node_id if back_edge_i[node_id, state] == INTERNAL_EDGE else int(graph.edge_src[back_edge_i[node_id, state]]),
                        int(back_channel_i[node_id, state]),
                        int(back_state_i[node_id, state]),
                        node_count,
                        state_count,
                    )
                )
    return ViterbiDecodeResult(
        score=float(best_score),
        states=tuple(state_path),
        node_ids=np.asarray(node_path, dtype=np.int64),
        global_state_ids=np.asarray(global_state_ids, dtype=np.int64),
        node_assignments=node_assignments,
        backpointers=ViterbiBackpointerTable(
            packed_state_index=np.asarray(current_index, dtype=np.int64),
            predecessor_edge_id=np.asarray(predecessor_edge, dtype=np.int64),
            predecessor_channel=np.asarray(predecessor_channel, dtype=np.int8),
            predecessor_packed_state_index=np.asarray(predecessor_index, dtype=np.int64),
        ),
        effective_support=problem.effective_support,
        metadata={
            "implementation": "cpu_dense_reference",
            "graph_id": graph.metadata.graph_id,
            "state_path_length": len(state_path),
            "node_assignment_count": len(node_assignments),
            "node_assignments": node_assignments,
        },
    )
