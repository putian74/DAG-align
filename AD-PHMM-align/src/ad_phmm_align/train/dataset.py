"""Dataset interfaces for TensorDag and subgraph batches."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ad_phmm_align.graph.subgraph import SubgraphBatch
from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.io.schema import TensorGraphManifest
from ad_phmm_align.train.profiling import ProfilingResult


@dataclass(frozen=True)
class TensorGraphArtifact:
    """Loaded tensor graph artifact and manifest metadata."""

    root: Path
    manifest: TensorGraphManifest
    graph: TensorDag
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedTrainingBatch:
    """Tensor-like data passed from preparation/sampling into training."""

    subgraph: SubgraphBatch
    node_symbol: Any
    node_weight: Any
    edge_src: Any
    edge_dst: Any
    edge_weight: Any
    node_state_left: Any
    node_state_right: Any
    node_flags: Optional[Any] = None
    topo_order: Optional[Any] = None
    csr_indptr: Optional[Any] = None
    csr_indices: Optional[Any] = None
    csc_indptr: Optional[Any] = None
    csc_indices: Optional[Any] = None
    state_window_offset: Optional[Any] = None
    state_window_length: Optional[Any] = None
    edge_overlap_src_offset: Optional[Any] = None
    edge_overlap_dst_offset: Optional[Any] = None
    edge_overlap_length: Optional[Any] = None
    active_global_states: Optional[Any] = None
    sequence_ids: Optional[Any] = None
    loss_scale: float = 1.0
    profiling: Optional[ProfilingResult] = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def batch_id(self) -> str:
        """Stable identifier for logging, checkpointing, and profiling."""

        return self.subgraph.batch_id


class TensorDagDataset:
    """Dataset wrapper around one or more TensorDag artifacts."""

    def __init__(self, graph: TensorDag) -> None:
        self.graph = graph

    def __len__(self) -> int:
        return 1

    def node_indices(self) -> np.ndarray:
        """Return graph node indices in canonical order."""

        return np.arange(self.graph.node_count)
