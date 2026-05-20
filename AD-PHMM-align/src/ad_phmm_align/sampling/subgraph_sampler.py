"""Sequence-batch induced subgraph sampling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ad_phmm_align.exceptions import ArtifactValidationError
from ad_phmm_align.graph.coordinates import EdgeWindowOverlaps, PackedStateWindows
from ad_phmm_align.graph.subgraph import SubgraphBatch
from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.io.schema import GraphMetadata
from ad_phmm_align.sampling.state_masks import StateMaskSpec


@dataclass(frozen=True)
class SampledSubgraph:
    """Subgraph batch together with the sampled TensorDag view."""

    subgraph: SubgraphBatch
    graph: TensorDag


class SubgraphSampler:
    """Sequence-batch induced subgraph sampler with optional state-range clipping."""

    def __init__(
        self,
        graph: TensorDag,
        *,
        sequence_batch_size: Optional[int] = None,
        state_mask_spec: Optional[StateMaskSpec] = None,
        max_nodes: Optional[int] = None,
        max_edges: Optional[int] = None,
        seed: int = 1,
    ) -> None:
        self.graph = graph
        self.sequence_batch_size = sequence_batch_size
        self.state_mask_spec = state_mask_spec
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.seed = seed

    def sample(
        self,
        *,
        step_index: int = 0,
        sequence_ids: Optional[np.ndarray] = None,
    ) -> SubgraphBatch:
        """Sample one subgraph batch."""

        selected_sequence_ids = (
            self._sample_sequence_ids(step_index)
            if sequence_ids is None
            else np.asarray(sequence_ids, dtype=np.int64)
        )
        if selected_sequence_ids.ndim != 1:
            raise ValueError("sequence_ids must be a 1D array")
        selected_sequence_ids = np.unique(selected_sequence_ids.astype(np.int64, copy=False))
        node_ids = self._node_ids_for_sequences(selected_sequence_ids)
        if node_ids.size == 0:
            raise ArtifactValidationError("sampled sequence batch produced an empty subgraph")
        edge_ids = self._edge_ids_for_nodes(node_ids)
        self._enforce_size_limits(node_ids, edge_ids)

        node_window_left, node_window_right, active_global_state_ids = self._clip_node_windows(
            node_ids
        )
        state_windows = self._build_state_windows(node_window_left, node_window_right)
        edge_overlaps = self._build_edge_overlaps(node_ids, edge_ids, node_window_left, node_window_right)
        node_coordinate_left = self.graph.node_coordinate_left[node_ids].astype(np.int64, copy=False)
        node_coordinate_right = self.graph.node_coordinate_right[node_ids].astype(np.int64, copy=False)
        topo_positions = {int(node_id): idx for idx, node_id in enumerate(node_ids.tolist())}
        topo_order = np.asarray(
            [topo_positions[int(node_id)] for node_id in self.graph.topo_order if int(node_id) in topo_positions],
            dtype=np.int64,
        )
        return SubgraphBatch(
            batch_id=f"{self.graph.metadata.graph_id}:sequence_batch:{step_index}",
            node_ids=node_ids.astype(np.int64, copy=False),
            edge_ids=edge_ids.astype(np.int64, copy=False),
            node_coordinate_left=node_coordinate_left,
            node_coordinate_right=node_coordinate_right,
            node_window_left=node_window_left,
            node_window_right=node_window_right,
            state_windows=state_windows,
            edge_overlaps=edge_overlaps,
            global_state_ids=active_global_state_ids,
            local_to_global_state=active_global_state_ids,
            node_local_index=np.arange(node_ids.shape[0], dtype=np.int64),
            edge_local_index=np.arange(edge_ids.shape[0], dtype=np.int64),
            sequence_ids=selected_sequence_ids.astype(np.int64, copy=False),
            sequence_weight=np.ones((selected_sequence_ids.shape[0],), dtype=np.float64),
            metadata={
                "sampling_strategy": "sequence_batch",
                "step_index": int(step_index),
                "topo_order": topo_order,
            },
        )

    def materialize(self, subgraph: SubgraphBatch) -> TensorDag:
        """Materialize a TensorDag view for a sampled subgraph batch."""

        node_ids = np.asarray(subgraph.node_ids, dtype=np.int64)
        edge_ids = np.asarray(subgraph.edge_ids, dtype=np.int64)
        node_map = {int(node_id): idx for idx, node_id in enumerate(node_ids.tolist())}
        edge_src = np.asarray(
            [node_map[int(self.graph.edge_src[edge_id])] for edge_id in edge_ids.tolist()],
            dtype=np.int64,
        )
        edge_dst = np.asarray(
            [node_map[int(self.graph.edge_dst[edge_id])] for edge_id in edge_ids.tolist()],
            dtype=np.int64,
        )
        topo_order = np.asarray(subgraph.metadata.get("topo_order"), dtype=np.int64)
        if topo_order.size != node_ids.shape[0]:
            topo_order = np.arange(node_ids.shape[0], dtype=np.int64)

        extra = dict(self.graph.extra or {})
        sampled_extra = {
            key: value
            for key, value in extra.items()
            if key
            not in {
                "sequence_id",
                "sequence_names",
                "source_sequence_id",
                "source_position",
                "node_source_offset",
                "node_source_len",
            }
        }
        sampled_extra.update(self._sampled_provenance(node_ids, subgraph.sequence_ids))
        sampled_extra["sequence_count"] = (
            0 if subgraph.sequence_ids is None else int(np.asarray(subgraph.sequence_ids).shape[0])
        )
        sampled_graph = TensorDag(
            metadata=GraphMetadata(
                graph_id=subgraph.batch_id,
                format_name=self.graph.metadata.format_name,
                format_version=self.graph.metadata.format_version,
                source_path=self.graph.metadata.source_path,
                global_state_count=self.graph.metadata.global_state_count,
                alphabet=self.graph.metadata.alphabet,
                state_interval_semantics=self.graph.metadata.state_interval_semantics,
                extra=dict(self.graph.metadata.extra),
            ),
            node_symbol=self.graph.node_symbol[node_ids],
            node_weight=self.graph.node_weight[node_ids],
            edge_src=edge_src,
            edge_dst=edge_dst,
            edge_weight=self.graph.edge_weight[edge_ids],
            topo_order=topo_order,
            node_coordinate_left=np.asarray(subgraph.node_window_left, dtype=np.int64),
            node_coordinate_right=np.asarray(subgraph.node_window_right, dtype=np.int64),
            node_window_left=np.asarray(subgraph.node_window_left, dtype=np.int64),
            node_window_right=np.asarray(subgraph.node_window_right, dtype=np.int64),
            state_windows=subgraph.state_windows,
            edge_overlaps=subgraph.edge_overlaps,
            node_flags=None
            if self.graph.node_flags is None
            else self.graph.node_flags[node_ids],
            csr_indptr=None,
            csr_indices=None,
            csc_indptr=None,
            csc_indices=None,
            extra=sampled_extra,
        )
        sampled_graph.validate()
        return sampled_graph

    def sample_graph(
        self,
        *,
        step_index: int = 0,
        sequence_ids: Optional[np.ndarray] = None,
    ) -> SampledSubgraph:
        """Sample one batch and materialize the corresponding TensorDag view."""

        subgraph = self.sample(step_index=step_index, sequence_ids=sequence_ids)
        return SampledSubgraph(subgraph=subgraph, graph=self.materialize(subgraph))

    def _sample_sequence_ids(self, step_index: int) -> np.ndarray:
        sequence_ids = self._sequence_id_table()
        if self.sequence_batch_size is None or self.sequence_batch_size >= sequence_ids.shape[0]:
            return sequence_ids.astype(np.int64, copy=False)
        if self.sequence_batch_size <= 0:
            raise ValueError("sequence_batch_size must be positive")
        rng = np.random.default_rng(self.seed + int(step_index))
        sampled = rng.choice(
            sequence_ids.astype(np.int64, copy=False),
            size=int(self.sequence_batch_size),
            replace=False,
        )
        return np.sort(sampled.astype(np.int64, copy=False))

    def _sequence_id_table(self) -> np.ndarray:
        if self.graph.extra is None or "sequence_id" not in self.graph.extra:
            raise ArtifactValidationError(
                "sequence-batch sampling requires graph.extra['sequence_id']"
            )
        return np.asarray(self.graph.extra["sequence_id"], dtype=np.int64)

    def _node_ids_for_sequences(self, sequence_ids: np.ndarray) -> np.ndarray:
        if self.graph.extra is None:
            raise ArtifactValidationError(
                "sequence-batch sampling requires source provenance arrays in graph.extra"
            )
        node_source_offset = np.asarray(self.graph.extra.get("node_source_offset"), dtype=np.int64)
        node_source_len = np.asarray(self.graph.extra.get("node_source_len"), dtype=np.int64)
        source_sequence_id = np.asarray(self.graph.extra.get("source_sequence_id"), dtype=np.int64)
        if (
            node_source_offset.ndim != 1
            or node_source_len.ndim != 1
            or source_sequence_id.ndim != 1
            or node_source_offset.shape[0] != self.graph.node_count
            or node_source_len.shape[0] != self.graph.node_count
        ):
            raise ArtifactValidationError(
                "sequence-batch sampling requires node_source_offset/node_source_len/source_sequence_id"
            )
        selected_nodes = []
        for node_id in range(self.graph.node_count):
            start = int(node_source_offset[node_id])
            stop = start + int(node_source_len[node_id])
            if stop <= start:
                continue
            if np.any(np.isin(source_sequence_id[start:stop], sequence_ids)):
                selected_nodes.append(node_id)
        return np.asarray(selected_nodes, dtype=np.int64)

    def _edge_ids_for_nodes(self, node_ids: np.ndarray) -> np.ndarray:
        selected = set(node_ids.astype(np.int64).tolist())
        edge_ids = [
            edge_id
            for edge_id, (src, dst) in enumerate(
                zip(
                    self.graph.edge_src.astype(np.int64).tolist(),
                    self.graph.edge_dst.astype(np.int64).tolist(),
                )
            )
            if src in selected and dst in selected
        ]
        return np.asarray(edge_ids, dtype=np.int64)

    def _clip_node_windows(self, node_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        node_window_left = self.graph.node_window_left[node_ids].astype(np.int64, copy=True)
        node_window_right = self.graph.node_window_right[node_ids].astype(np.int64, copy=True)
        if self.state_mask_spec is None:
            active_global_state_ids = self._union_window_states(node_window_left, node_window_right)
            return node_window_left, node_window_right, active_global_state_ids

        active_global_state_ids = self.state_mask_spec.active_global_state_ids()
        if active_global_state_ids.shape[0] == 0:
            raise ArtifactValidationError("state mask spec produced no active global states")
        for index in range(node_ids.shape[0]):
            left = int(node_window_left[index])
            right = int(node_window_right[index])
            local_active = active_global_state_ids[
                (active_global_state_ids >= left) & (active_global_state_ids < right)
            ]
            if local_active.shape[0] == 0:
                node_window_right[index] = left
                continue
            node_window_left[index] = int(local_active[0])
            node_window_right[index] = int(local_active[-1]) + 1
        return node_window_left, node_window_right, active_global_state_ids

    @staticmethod
    def _union_window_states(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        active = [
            np.arange(int(start), int(stop), dtype=np.int64)
            for start, stop in zip(left.tolist(), right.tolist())
            if int(stop) > int(start)
        ]
        if not active:
            return np.zeros((0,), dtype=np.int64)
        return np.unique(np.concatenate(active))

    @staticmethod
    def _build_state_windows(
        node_window_left: np.ndarray,
        node_window_right: np.ndarray,
    ) -> PackedStateWindows:
        length = node_window_right - node_window_left
        offset = np.zeros((node_window_left.shape[0],), dtype=np.int64)
        if offset.shape[0] > 1:
            offset[1:] = np.cumsum(length[:-1], dtype=np.int64)
        return PackedStateWindows(
            left=node_window_left.astype(np.int64, copy=False),
            right=node_window_right.astype(np.int64, copy=False),
            offset=offset,
            length=length.astype(np.int64, copy=False),
        )

    def _build_edge_overlaps(
        self,
        node_ids: np.ndarray,
        edge_ids: np.ndarray,
        node_window_left: np.ndarray,
        node_window_right: np.ndarray,
    ) -> EdgeWindowOverlaps:
        local_node_index = {int(node_id): idx for idx, node_id in enumerate(node_ids.tolist())}
        src_offsets = []
        dst_offsets = []
        overlap_lengths = []
        local_edge_ids = []
        for local_edge_id, global_edge_id in enumerate(edge_ids.tolist()):
            src_global = int(self.graph.edge_src[int(global_edge_id)])
            dst_global = int(self.graph.edge_dst[int(global_edge_id)])
            src_local = local_node_index[src_global]
            dst_local = local_node_index[dst_global]
            overlap_left = max(int(node_window_left[src_local]), int(node_window_left[dst_local]))
            overlap_right = min(int(node_window_right[src_local]), int(node_window_right[dst_local]))
            overlap_length = max(0, overlap_right - overlap_left)
            src_offsets.append(overlap_left - int(node_window_left[src_local]))
            dst_offsets.append(overlap_left - int(node_window_left[dst_local]))
            overlap_lengths.append(overlap_length)
            local_edge_ids.append(local_edge_id)
        return EdgeWindowOverlaps(
            edge_ids=np.asarray(local_edge_ids, dtype=np.int64),
            src_offset=np.asarray(src_offsets, dtype=np.int64),
            dst_offset=np.asarray(dst_offsets, dtype=np.int64),
            length=np.asarray(overlap_lengths, dtype=np.int64),
        )

    def _sampled_provenance(
        self,
        node_ids: np.ndarray,
        selected_sequence_ids: Optional[np.ndarray],
    ) -> dict[str, object]:
        if self.graph.extra is None:
            return {}
        extra = self.graph.extra
        sequence_ids = self._sequence_id_table()
        sequence_mask = (
            np.ones((sequence_ids.shape[0],), dtype=np.bool_)
            if selected_sequence_ids is None
            else np.isin(sequence_ids, np.asarray(selected_sequence_ids, dtype=np.int64))
        )
        sampled_sequence_ids = sequence_ids[sequence_mask].astype(np.uint64, copy=False)
        sampled_sequence_names = None
        if "sequence_names" in extra:
            names = list(extra["sequence_names"])
            sampled_sequence_names = [str(names[index]) for index in np.flatnonzero(sequence_mask)]

        node_source_offset = np.asarray(extra.get("node_source_offset"), dtype=np.int64)
        node_source_len = np.asarray(extra.get("node_source_len"), dtype=np.int64)
        source_sequence_id = np.asarray(extra.get("source_sequence_id"), dtype=np.int64)
        source_position = None
        if "source_position" in extra:
            source_position = np.asarray(extra.get("source_position"), dtype=np.int64)
        sampled_source_sequence_id = []
        sampled_source_position = []
        sampled_node_source_offset = []
        sampled_node_source_len = []
        current_offset = 0
        for node_id in node_ids.tolist():
            start = int(node_source_offset[int(node_id)])
            stop = start + int(node_source_len[int(node_id)])
            local_sequence_id = source_sequence_id[start:stop]
            keep = (
                np.ones((local_sequence_id.shape[0],), dtype=np.bool_)
                if selected_sequence_ids is None
                else np.isin(local_sequence_id, np.asarray(selected_sequence_ids, dtype=np.int64))
            )
            filtered_sequence_id = local_sequence_id[keep]
            sampled_node_source_offset.append(current_offset)
            sampled_node_source_len.append(int(filtered_sequence_id.shape[0]))
            sampled_source_sequence_id.extend(filtered_sequence_id.astype(np.uint64).tolist())
            if source_position is not None:
                sampled_source_position.extend(
                    source_position[start:stop][keep].astype(np.uint64).tolist()
                )
            current_offset += int(filtered_sequence_id.shape[0])
        sampled = {
            "sequence_id": sampled_sequence_ids,
            "node_source_offset": np.asarray(sampled_node_source_offset, dtype=np.uint64),
            "node_source_len": np.asarray(sampled_node_source_len, dtype=np.uint64),
            "source_sequence_id": np.asarray(sampled_source_sequence_id, dtype=np.uint64),
        }
        if sampled_sequence_names is not None:
            sampled["sequence_names"] = sampled_sequence_names
        if source_position is not None:
            sampled["source_position"] = np.asarray(sampled_source_position, dtype=np.uint64)
        return sampled

    def _enforce_size_limits(self, node_ids: np.ndarray, edge_ids: np.ndarray) -> None:
        if self.max_nodes is not None and node_ids.shape[0] > int(self.max_nodes):
            raise ArtifactValidationError(
                f"sampled subgraph has {node_ids.shape[0]} nodes, exceeding max_nodes={self.max_nodes}"
            )
        if self.max_edges is not None and edge_ids.shape[0] > int(self.max_edges):
            raise ArtifactValidationError(
                f"sampled subgraph has {edge_ids.shape[0]} edges, exceeding max_edges={self.max_edges}"
            )
