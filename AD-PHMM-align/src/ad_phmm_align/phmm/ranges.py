"""Effective packed-state support propagation for DAG dynamic programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from ad_phmm_align.exceptions import ArtifactValidationError
from ad_phmm_align.graph.coordinates import EdgeWindowOverlaps, PackedStateWindows
from ad_phmm_align.graph.tensor_dag import TensorDag


@dataclass(frozen=True)
class EffectiveStateMask:
    """Exact packed active-state mask plus coarse half-open node spans."""

    windows: PackedStateWindows
    packed_mask: np.ndarray
    span_left: np.ndarray
    span_right: np.ndarray
    metadata: Mapping[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate packed-mask and span consistency against the source windows."""

        node_count = int(self.windows.length.shape[0])
        if self.span_left.shape != (node_count,) or self.span_right.shape != (node_count,):
            raise ArtifactValidationError("effective span arrays must match node count")
        if self.packed_mask.dtype != np.bool_:
            raise ArtifactValidationError("effective packed mask must use boolean dtype")
        total_packed_states = _packed_state_count(self.windows)
        if self.packed_mask.shape != (total_packed_states,):
            raise ArtifactValidationError(
                "effective packed mask length must equal packed state count"
            )
        for node_id in range(node_count):
            node_mask = self.node_mask(node_id)
            window_left = int(self.windows.left[node_id])
            window_right = int(self.windows.right[node_id])
            span_left = int(self.span_left[node_id])
            span_right = int(self.span_right[node_id])
            if span_left < window_left or span_right > window_right:
                raise ArtifactValidationError("effective span exceeds legal node window")
            if span_right < span_left:
                raise ArtifactValidationError("effective span right bound is before left")
            if node_mask.any():
                active_local = np.flatnonzero(node_mask)
                expected_left = window_left + int(active_local[0])
                expected_right = window_left + int(active_local[-1]) + 1
                if span_left != expected_left or span_right != expected_right:
                    raise ArtifactValidationError(
                        "effective span does not match active packed mask support"
                    )
            elif span_left != span_right or span_left != window_left:
                raise ArtifactValidationError(
                    "empty effective support must collapse to the node window left bound"
                )

    def node_mask(self, node_id: int) -> np.ndarray:
        """Return the local packed-state mask for one node window."""

        start = int(self.windows.offset[node_id])
        stop = start + int(self.windows.length[node_id])
        return self.packed_mask[start:stop]

    @property
    def active_count(self) -> int:
        """Number of active packed states across the whole graph or subgraph."""

        return int(np.count_nonzero(self.packed_mask))

    def active_global_state_ids(self) -> np.ndarray:
        """Return sorted unique global state IDs touched by the packed support."""

        active_states: list[np.ndarray] = []
        for node_id in range(int(self.windows.length.shape[0])):
            local_mask = self.node_mask(node_id)
            if not local_mask.any():
                continue
            local_ids = np.flatnonzero(local_mask).astype(np.int64, copy=False)
            active_states.append(local_ids + int(self.windows.left[node_id]))
        if not active_states:
            return np.zeros((0,), dtype=np.int64)
        return np.unique(np.concatenate(active_states))


def empty_effective_state_mask(
    windows: PackedStateWindows, metadata: Optional[Mapping[str, object]] = None
) -> EffectiveStateMask:
    """Create an empty effective support mask over the packed node windows."""

    packed_mask = np.zeros((_packed_state_count(windows),), dtype=np.bool_)
    span_left = windows.left.astype(np.int64, copy=True)
    span_right = windows.left.astype(np.int64, copy=True)
    mask = EffectiveStateMask(
        windows=windows,
        packed_mask=packed_mask,
        span_left=span_left,
        span_right=span_right,
        metadata=dict(metadata or {}),
    )
    mask.validate()
    return mask


def full_effective_state_mask(
    windows: PackedStateWindows, metadata: Optional[Mapping[str, object]] = None
) -> EffectiveStateMask:
    """Create a support mask that activates every legal packed state."""

    packed_mask = np.ones((_packed_state_count(windows),), dtype=np.bool_)
    mask = EffectiveStateMask(
        windows=windows,
        packed_mask=packed_mask,
        span_left=windows.left.astype(np.int64, copy=True),
        span_right=windows.right.astype(np.int64, copy=True),
        metadata=dict(metadata or {}),
    )
    mask.validate()
    return mask


def intersect_effective_state_masks(
    left: EffectiveStateMask, right: EffectiveStateMask
) -> EffectiveStateMask:
    """Intersect two masks defined on the same packed window layout."""

    _require_same_windows(left.windows, right.windows)
    return _mask_from_packed(
        left.windows,
        np.logical_and(left.packed_mask, right.packed_mask),
        metadata={
            "operation": "intersection",
            "left": dict(left.metadata),
            "right": dict(right.metadata),
        },
    )


def propagate_forward_support(
    graph: TensorDag,
    seed_mask: Optional[EffectiveStateMask] = None,
) -> EffectiveStateMask:
    """Propagate exact forward-reachable support through branch/merge DAG structure."""

    windows, overlaps = _require_windows_and_overlaps(graph)
    out_edges = _edges_by_node(graph.edge_src, graph.node_count)
    overlap_lookup = _overlap_lookup(overlaps, graph.edge_count)
    indegree = np.bincount(graph.edge_dst.astype(np.int64), minlength=graph.node_count)
    if seed_mask is None:
        seed_mask = empty_effective_state_mask(
            windows,
            metadata={"seed": "graph_sources", "direction": "forward"},
        )
        packed_mask = seed_mask.packed_mask.copy()
        for node_id in np.flatnonzero(indegree == 0):
            _set_full_node_window(packed_mask, windows, int(node_id))
        seed_mask = _mask_from_packed(windows, packed_mask, metadata=seed_mask.metadata)
    else:
        _require_same_windows(windows, seed_mask.windows)

    packed_mask = seed_mask.packed_mask.copy()
    for node_id in graph.topo_order.astype(np.int64).tolist():
        if not packed_mask.any():
            break
        node_mask = _node_view(packed_mask, windows, int(node_id))
        if not node_mask.any():
            continue
        for edge_id in out_edges[int(node_id)]:
            overlap_index = overlap_lookup.get(int(edge_id))
            if overlap_index is None:
                continue
            overlap_len = int(overlaps.length[overlap_index])
            if overlap_len == 0:
                continue
            src_offset = int(overlaps.src_offset[overlap_index])
            dst_offset = int(overlaps.dst_offset[overlap_index])
            dst_node = int(graph.edge_dst[edge_id])
            active = node_mask[src_offset : src_offset + overlap_len]
            if not active.any():
                continue
            dst_mask = _node_view(packed_mask, windows, dst_node)
            dst_mask[dst_offset : dst_offset + overlap_len] |= active
    return _mask_from_packed(
        windows,
        packed_mask,
        metadata={"direction": "forward", "seed": dict(seed_mask.metadata)},
    )


def propagate_backward_support(
    graph: TensorDag,
    seed_mask: Optional[EffectiveStateMask] = None,
) -> EffectiveStateMask:
    """Propagate exact backward-reachable support through reverse DAG structure."""

    windows, overlaps = _require_windows_and_overlaps(graph)
    in_edges = _edges_by_node(graph.edge_dst, graph.node_count)
    overlap_lookup = _overlap_lookup(overlaps, graph.edge_count)
    outdegree = np.bincount(graph.edge_src.astype(np.int64), minlength=graph.node_count)
    if seed_mask is None:
        seed_mask = empty_effective_state_mask(
            windows,
            metadata={"seed": "graph_sinks", "direction": "backward"},
        )
        packed_mask = seed_mask.packed_mask.copy()
        for node_id in np.flatnonzero(outdegree == 0):
            _set_full_node_window(packed_mask, windows, int(node_id))
        seed_mask = _mask_from_packed(windows, packed_mask, metadata=seed_mask.metadata)
    else:
        _require_same_windows(windows, seed_mask.windows)

    packed_mask = seed_mask.packed_mask.copy()
    for node_id in graph.topo_order.astype(np.int64).tolist()[::-1]:
        node_mask = _node_view(packed_mask, windows, int(node_id))
        if not node_mask.any():
            continue
        for edge_id in in_edges[int(node_id)]:
            overlap_index = overlap_lookup.get(int(edge_id))
            if overlap_index is None:
                continue
            overlap_len = int(overlaps.length[overlap_index])
            if overlap_len == 0:
                continue
            src_offset = int(overlaps.src_offset[overlap_index])
            dst_offset = int(overlaps.dst_offset[overlap_index])
            src_node = int(graph.edge_src[edge_id])
            active = node_mask[dst_offset : dst_offset + overlap_len]
            if not active.any():
                continue
            src_mask = _node_view(packed_mask, windows, src_node)
            src_mask[src_offset : src_offset + overlap_len] |= active
    return _mask_from_packed(
        windows,
        packed_mask,
        metadata={"direction": "backward", "seed": dict(seed_mask.metadata)},
    )


def _require_windows_and_overlaps(
    graph: TensorDag,
) -> tuple[PackedStateWindows, EdgeWindowOverlaps]:
    if graph.state_windows is None or graph.edge_overlaps is None:
        raise ArtifactValidationError(
            "graph must provide state_windows and edge_overlaps for effective support propagation"
        )
    return graph.state_windows, graph.edge_overlaps


def _edges_by_node(edge_nodes: np.ndarray, node_count: int) -> list[list[int]]:
    grouped: list[list[int]] = [[] for _ in range(int(node_count))]
    for edge_id, node_id in enumerate(edge_nodes.astype(np.int64).tolist()):
        grouped[int(node_id)].append(int(edge_id))
    return grouped


def _overlap_lookup(
    overlaps: EdgeWindowOverlaps, edge_count: int
) -> dict[int, int]:
    lookup: dict[int, int] = {}
    for overlap_index, edge_id in enumerate(overlaps.edge_ids.astype(np.int64).tolist()):
        if edge_id < 0 or edge_id >= edge_count:
            raise ArtifactValidationError("edge overlap id exceeds edge_count")
        if edge_id in lookup:
            raise ArtifactValidationError("edge overlap ids must be unique")
        lookup[int(edge_id)] = int(overlap_index)
    return lookup


def _packed_state_count(windows: PackedStateWindows) -> int:
    if windows.length.size == 0:
        return 0
    return int(windows.offset[-1]) + int(windows.length[-1])


def _node_view(
    packed_mask: np.ndarray, windows: PackedStateWindows, node_id: int
) -> np.ndarray:
    start = int(windows.offset[node_id])
    stop = start + int(windows.length[node_id])
    return packed_mask[start:stop]


def _set_full_node_window(
    packed_mask: np.ndarray, windows: PackedStateWindows, node_id: int
) -> None:
    node_mask = _node_view(packed_mask, windows, node_id)
    node_mask[:] = True


def _require_same_windows(left: PackedStateWindows, right: PackedStateWindows) -> None:
    if not (
        np.array_equal(left.left, right.left)
        and np.array_equal(left.right, right.right)
        and np.array_equal(left.offset, right.offset)
        and np.array_equal(left.length, right.length)
    ):
        raise ArtifactValidationError("effective state masks must share the same windows")


def _mask_from_packed(
    windows: PackedStateWindows,
    packed_mask: np.ndarray,
    metadata: Optional[Mapping[str, object]] = None,
) -> EffectiveStateMask:
    span_left = windows.left.astype(np.int64, copy=True)
    span_right = windows.left.astype(np.int64, copy=True)
    for node_id in range(int(windows.length.shape[0])):
        node_mask = _node_view(packed_mask, windows, node_id)
        if not node_mask.any():
            continue
        active_local = np.flatnonzero(node_mask).astype(np.int64, copy=False)
        span_left[node_id] = int(windows.left[node_id]) + int(active_local[0])
        span_right[node_id] = int(windows.left[node_id]) + int(active_local[-1]) + 1
    mask = EffectiveStateMask(
        windows=windows,
        packed_mask=np.asarray(packed_mask, dtype=np.bool_),
        span_left=span_left,
        span_right=span_right,
        metadata=dict(metadata or {}),
    )
    mask.validate()
    return mask
