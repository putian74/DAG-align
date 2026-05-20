"""Decode hard state assignments into ``ValSparseMSA`` alignment artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from ad_phmm_align.graph.tensor_dag import TensorDag

from ._msa_representation import load_msa_representation_runtime

_MATCH_CHANNEL = "match"
_INSERT_CHANNEL = "insert"
_COLUMN_CHANNEL_ORDER = {_INSERT_CHANNEL: 0, _MATCH_CHANNEL: 1}
_INSERT_CHANNEL_ALIASES = {"I", "i", "insert"}


@dataclass(frozen=True)
class AlignmentColumnKey:
    """Typed identifier for one decoded alignment column."""

    channel: str
    state_index: int
    slot: int = 0


@dataclass(frozen=True)
class NodeColumnAssignment:
    """Assign one graph node to one decoded alignment column."""

    node_id: int
    column: AlignmentColumnKey
    sequence_ids: Optional[Any] = None
    original_channel: Optional[str] = None
    insertion_step: int = 0


@dataclass(frozen=True)
class DecodedAlignment:
    """Decoded hard alignment plus typed column metadata."""

    msa: Any
    column_keys: tuple[AlignmentColumnKey, ...]
    sequence_ids: Any
    sequence_names: tuple[str, ...]
    metadata: Mapping[str, object] = field(default_factory=dict)


def _require_graph_extra_array(graph: TensorDag, name: str) -> np.ndarray:
    if graph.extra is None or name not in graph.extra:
        raise ValueError(f"graph.extra is missing required provenance array: {name}")
    return np.asarray(graph.extra[name])


def _sequence_table(graph: TensorDag) -> tuple[np.ndarray, tuple[str, ...]]:
    if graph.extra is None:
        raise ValueError("graph.extra is required to decode a hard alignment")
    if "sequence_id" in graph.extra:
        sequence_ids = np.asarray(graph.extra["sequence_id"], dtype=np.uint64)
    elif "source_sequence_id" in graph.extra:
        source_sequence_id = np.asarray(graph.extra["source_sequence_id"], dtype=np.uint64)
        sequence_ids = np.unique(source_sequence_id)
    else:
        raise ValueError("graph provenance does not expose sequence_id or source_sequence_id")
    sequence_names_raw = graph.extra.get("sequence_names")
    if sequence_names_raw is None:
        sequence_names = tuple(f"seq{int(sequence_id)}" for sequence_id in sequence_ids.tolist())
    else:
        sequence_names = tuple(str(name) for name in sequence_names_raw)
        if len(sequence_names) != int(sequence_ids.shape[0]):
            raise ValueError("sequence_names length must match sequence_id length")
    return sequence_ids, sequence_names


def _symbol_value_map(graph: TensorDag) -> dict[int, int]:
    runtime = load_msa_representation_runtime()
    iupac_to_digital = dict(runtime.module.IUPAC_TO_DIGITAL)
    symbol_encoding = {}
    if graph.extra is not None:
        symbol_encoding = dict(graph.extra.get("symbol_encoding", {}))
    if not symbol_encoding and graph.metadata.alphabet is not None:
        symbol_encoding = {
            str(symbol): index for index, symbol in enumerate(graph.metadata.alphabet)
        }
    if not symbol_encoding:
        raise ValueError("graph symbol encoding metadata is required for decode")
    return {
        int(value): int(iupac_to_digital[str(symbol).upper()])
        for symbol, value in symbol_encoding.items()
    }


def _node_sequence_ids(graph: TensorDag, node_id: int) -> np.ndarray:
    node_source_offset = _require_graph_extra_array(graph, "node_source_offset")
    node_source_len = _require_graph_extra_array(graph, "node_source_len")
    source_sequence_id = _require_graph_extra_array(graph, "source_sequence_id")
    start = int(node_source_offset[int(node_id)])
    length = int(node_source_len[int(node_id)])
    return source_sequence_id[start : start + length].astype(np.uint64, copy=False)


def _reference_symbol_vector(
    graph: TensorDag, column_keys: Sequence[AlignmentColumnKey], symbol_value_map: Mapping[int, int]
) -> Optional[np.ndarray]:
    if graph.extra is None or "ref_sequence_symbols" not in graph.extra:
        return None
    raw = np.asarray(graph.extra["ref_sequence_symbols"], dtype=np.int64)
    ref = np.zeros((len(column_keys),), dtype=np.uint8)
    for index, column in enumerate(column_keys):
        if column.channel == _INSERT_CHANNEL:
            ref[index] = 0
            continue
        if column.state_index < 0 or column.state_index >= int(raw.shape[0]):
            raise ValueError("match column exceeds ref_sequence_symbols length")
        ref[index] = np.uint8(symbol_value_map[int(raw[column.state_index])])
    return ref


def _normalize_node_assignments(
    viterbi_result: Any,
    node_assignments: Optional[Iterable[NodeColumnAssignment]],
) -> tuple[NodeColumnAssignment, ...]:
    if node_assignments is not None:
        return tuple(node_assignments)
    assignments = None
    if getattr(viterbi_result, "metadata", None) is not None:
        assignments = viterbi_result.metadata.get("node_assignments")
    if assignments is None:
        raise NotImplementedError(
            "decode_alignment requires per-node hidden-state assignments; "
            "the current CPU Viterbi baseline only exposes one terminal traceback"
        )
    normalized = []
    for assignment in assignments:
        if isinstance(assignment, NodeColumnAssignment):
            normalized.append(assignment)
            continue
        if isinstance(assignment, Mapping):
            data = assignment
        else:
            data = {
                "node_id": getattr(assignment, "node_id"),
                "channel": getattr(assignment, "channel", None),
                "state_index": getattr(assignment, "state_index", None),
                "sequence_ids": getattr(assignment, "sequence_ids", None),
                "insertion_slot": getattr(assignment, "insertion_slot", 0),
                "insertion_step": getattr(assignment, "insertion_step", 0),
            }
        original_channel = str(data.get("channel", _MATCH_CHANNEL))
        column_channel = (
            _INSERT_CHANNEL
            if original_channel in _INSERT_CHANNEL_ALIASES
            else _MATCH_CHANNEL
        )
        normalized.append(
            NodeColumnAssignment(
                node_id=int(data["node_id"]),
                column=AlignmentColumnKey(
                    channel=column_channel,
                    state_index=int(data["state_index"]),
                    slot=int(data.get("insertion_slot", 0)),
                ),
                sequence_ids=data.get("sequence_ids"),
                original_channel=original_channel,
                insertion_step=int(data.get("insertion_step", 0)),
            )
        )
    return tuple(normalized)


def decode_alignment(
    graph: TensorDag,
    viterbi_result: Any,
    node_assignments: Optional[Iterable[NodeColumnAssignment]] = None,
) -> DecodedAlignment:
    """Convert decoded node/state assignments into a ``ValSparseMSA`` artifact."""

    runtime = load_msa_representation_runtime()
    sequence_ids, sequence_names = _sequence_table(graph)
    row_index_by_sequence_id = {
        int(sequence_id): row_index
        for row_index, sequence_id in enumerate(sequence_ids.astype(np.int64).tolist())
    }
    symbol_value_map = _symbol_value_map(graph)
    normalized_assignments = _normalize_node_assignments(viterbi_result, node_assignments)
    if not normalized_assignments:
        raise ValueError("decode_alignment requires at least one node assignment")

    column_keys = tuple(
        sorted(
            {assignment.column for assignment in normalized_assignments},
            key=lambda column: (
                int(column.state_index),
                _COLUMN_CHANNEL_ORDER[column.channel],
                int(column.slot),
            ),
        )
    )
    column_index_by_key = {column: index for index, column in enumerate(column_keys)}
    row_values_by_column = [dict() for _ in column_keys]

    for assignment in normalized_assignments:
        node_id = int(assignment.node_id)
        row_ids = (
            _node_sequence_ids(graph, node_id)
            if assignment.sequence_ids is None
            else np.asarray(assignment.sequence_ids, dtype=np.uint64)
        )
        try:
            symbol_value = symbol_value_map[int(np.asarray(graph.node_symbol)[node_id])]
        except KeyError as exc:
            raise ValueError(f"graph node_symbol[{node_id}] is not covered by symbol_encoding") from exc
        column_index = column_index_by_key[assignment.column]
        row_values = row_values_by_column[column_index]
        for sequence_id in row_ids.astype(np.int64).tolist():
            row_index = row_index_by_sequence_id.get(int(sequence_id))
            if row_index is None:
                raise ValueError(f"sequence_id {sequence_id} is not present in the sequence table")
            existing = row_values.get(row_index)
            if existing is not None and existing != symbol_value:
                raise ValueError(
                    "conflicting decoded symbols map to the same sequence/column cell"
                )
            row_values[row_index] = symbol_value

    reference_vector = _reference_symbol_vector(graph, column_keys, symbol_value_map)
    if reference_vector is None:
        reference_vector = np.zeros((len(column_keys),), dtype=np.uint8)
        for column_index, row_values in enumerate(row_values_by_column):
            counts = {}
            gap_count = len(sequence_ids) - len(row_values)
            if gap_count > 0:
                counts[0] = gap_count
            for value in row_values.values():
                counts[value] = counts.get(value, 0) + 1
            best_count = max(counts.values())
            reference_vector[column_index] = np.uint8(
                min(value for value, count in counts.items() if count == best_count)
            )

    val_positions = []
    for column_index, row_values in enumerate(row_values_by_column):
        value_rows = {}
        for row_index, value in row_values.items():
            if int(value) == int(reference_vector[column_index]):
                continue
            value_rows.setdefault(int(value), []).append(int(row_index))
        explicit_gap_rows = [
            row_index
            for row_index in range(len(sequence_ids))
            if row_index not in row_values and int(reference_vector[column_index]) != 0
        ]
        if explicit_gap_rows:
            value_rows[0] = explicit_gap_rows
        slice_list = [
            [value, np.asarray(sorted(rows), dtype=np.int32)]
            for value, rows in value_rows.items()
        ]
        slice_list.sort(key=lambda item: item[1][0] if len(item[1]) else len(sequence_ids))
        val_positions.append(slice_list)

    msa = runtime.module.ValSparseMSA(
        {
            "nrow": int(sequence_ids.shape[0]),
            "ncol": len(column_keys),
            "alphabet_size": int(runtime.module.ALPHABET_SIZE),
            "name_list": list(sequence_names),
            "ref_vector": reference_vector.astype(np.uint8, copy=False),
            "axis": 1,
            "graph_id": graph.metadata.graph_id,
            "sequence_ids": sequence_ids.astype(np.uint64, copy=False),
            "column_channel": [column.channel for column in column_keys],
            "column_state": np.asarray(
                [column.state_index for column in column_keys], dtype=np.int64
            ),
            "column_slot": np.asarray(
                [column.slot for column in column_keys], dtype=np.int64
            ),
        },
        val_positions=val_positions,
    )
    return DecodedAlignment(
        msa=msa,
        column_keys=column_keys,
        sequence_ids=sequence_ids.astype(np.uint64, copy=False),
        sequence_names=sequence_names,
        metadata={
            "graph_id": graph.metadata.graph_id,
            "decoded_node_count": len(normalized_assignments),
            "representation": "ValSparseMSA",
        },
    )
