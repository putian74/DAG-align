"""Canonical tensor-backed DAG representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from ad_phmm_align.exceptions import ArtifactValidationError
from ad_phmm_align.graph.coordinates import (
    EdgeWindowOverlaps,
    PackedStateWindows,
    validate_state_intervals,
)
from ad_phmm_align.io.schema import GraphMetadata


@dataclass(frozen=True)
class TensorDag:
    """Format-neutral graph representation consumed by AD-PHMM-align."""

    metadata: GraphMetadata
    node_symbol: np.ndarray
    node_weight: np.ndarray
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_weight: np.ndarray
    topo_order: np.ndarray
    node_coordinate_left: np.ndarray
    node_coordinate_right: np.ndarray
    node_window_left: np.ndarray
    node_window_right: np.ndarray
    state_windows: Optional[PackedStateWindows] = None
    edge_overlaps: Optional[EdgeWindowOverlaps] = None
    node_flags: Optional[np.ndarray] = None
    csr_indptr: Optional[np.ndarray] = None
    csr_indices: Optional[np.ndarray] = None
    csc_indptr: Optional[np.ndarray] = None
    csc_indices: Optional[np.ndarray] = None
    extra: Optional[Mapping[str, Any]] = None

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""

        return int(self.node_symbol.shape[0])

    @property
    def edge_count(self) -> int:
        """Number of directed weighted edges in the graph."""

        return int(self.edge_src.shape[0])

    def validate(self) -> None:
        """Validate structural invariants needed by training code."""

        n = self.node_count
        if self.node_weight.shape[0] != n:
            raise ArtifactValidationError("node_weight length does not match nodes")
        if (
            self.node_coordinate_left.shape[0] != n
            or self.node_coordinate_right.shape[0] != n
            or self.node_window_left.shape[0] != n
            or self.node_window_right.shape[0] != n
        ):
            raise ArtifactValidationError("coordinate/window interval arrays must match nodes")
        if not (
            self.edge_src.shape == self.edge_dst.shape == self.edge_weight.shape
        ):
            raise ArtifactValidationError("edge arrays must have identical shape")
        if np.any(self.edge_src < 0) or np.any(self.edge_dst < 0):
            raise ArtifactValidationError("edge endpoints must be non-negative")
        if self.edge_count and (
            np.max(self.edge_src) >= n or np.max(self.edge_dst) >= n
        ):
            raise ArtifactValidationError("edge endpoint exceeds node_count")
        if self.topo_order.shape[0] != n:
            raise ArtifactValidationError("topological order must include every node")
        if set(map(int, self.topo_order.tolist())) != set(range(n)):
            raise ArtifactValidationError("topological order is not a node permutation")
        topo_rank = np.empty(n, dtype=np.int64)
        topo_rank[self.topo_order] = np.arange(n)
        if self.edge_count and np.any(
            topo_rank[self.edge_src] >= topo_rank[self.edge_dst]
        ):
            raise ArtifactValidationError("topological order violates graph edges")
        if self.metadata.global_state_count is not None:
            validate_state_intervals(
                self.node_coordinate_left,
                self.node_coordinate_right,
                self.metadata.global_state_count,
            )
            validate_state_intervals(
                self.node_window_left,
                self.node_window_right,
                self.metadata.global_state_count,
            )
        elif np.any(self.node_coordinate_right < self.node_coordinate_left) or np.any(
            self.node_window_right < self.node_window_left
        ):
            raise ArtifactValidationError("state interval right bound is before left")
        if np.any(self.node_window_left > self.node_coordinate_left):
            raise ArtifactValidationError(
                "node_window_left must not exclude the raw coordinate left bound"
            )
        if np.any(self.node_window_right < self.node_coordinate_right):
            raise ArtifactValidationError(
                "node_window_right must not exclude the raw coordinate right bound"
            )
        if self.state_windows is not None:
            if self.metadata.global_state_count is None:
                raise ArtifactValidationError(
                    "global_state_count is required for packed windows"
                )
            self.state_windows.validate(self.metadata.global_state_count)
            if not np.array_equal(self.node_window_left, self.state_windows.left):
                raise ArtifactValidationError(
                    "node_window_left must match packed state-window left bounds"
                )
            if not np.array_equal(self.node_window_right, self.state_windows.right):
                raise ArtifactValidationError(
                    "node_window_right must match packed state-window right bounds"
                )
        if self.edge_overlaps is not None:
            self.edge_overlaps.validate(self.edge_count)
            if self.state_windows is not None:
                self.edge_overlaps.validate_against_windows(
                    self.edge_src, self.edge_dst, self.state_windows
                )
