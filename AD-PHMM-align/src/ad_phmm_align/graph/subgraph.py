"""Subgraph batch interfaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np

from ad_phmm_align.graph.coordinates import EdgeWindowOverlaps, PackedStateWindows


@dataclass(frozen=True)
class SubgraphBatch:
    """Sampled subgraph with projections to global PHMM state IDs."""

    batch_id: str
    node_ids: np.ndarray
    edge_ids: np.ndarray
    node_state_left: np.ndarray
    node_state_right: np.ndarray
    state_windows: Optional[PackedStateWindows] = None
    edge_overlaps: Optional[EdgeWindowOverlaps] = None
    global_state_ids: Optional[np.ndarray] = None
    local_to_global_state: Optional[np.ndarray] = None
    node_local_index: Optional[np.ndarray] = None
    edge_local_index: Optional[np.ndarray] = None
    sequence_ids: Optional[np.ndarray] = None
    sequence_weight: Optional[np.ndarray] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        """Number of nodes in the sampled subgraph."""

        return int(self.node_ids.shape[0])

    @property
    def edge_count(self) -> int:
        """Number of edges in the sampled subgraph."""

        return int(self.edge_ids.shape[0])
