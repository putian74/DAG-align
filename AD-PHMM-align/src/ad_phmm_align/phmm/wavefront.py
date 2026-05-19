"""Dependency-safe wavefront schedules for DAG dynamic programs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ad_phmm_align.exceptions import ArtifactValidationError
from ad_phmm_align.graph.tensor_dag import TensorDag


@dataclass(frozen=True)
class WavefrontSchedule:
    """Topological level schedule used for CPU batching and CUDA planning."""

    level_ptr: np.ndarray
    level_nodes: np.ndarray
    node_level: np.ndarray

    def validate(self, node_count: int) -> None:
        """Validate shape and permutation properties of the wavefront schedule."""

        if self.level_ptr.ndim != 1 or self.level_nodes.ndim != 1 or self.node_level.ndim != 1:
            raise ArtifactValidationError("wavefront arrays must be one-dimensional")
        if self.node_level.shape[0] != node_count:
            raise ArtifactValidationError("node_level length must equal node_count")
        if self.level_ptr.shape[0] == 0 or int(self.level_ptr[0]) != 0:
            raise ArtifactValidationError("level_ptr must start at 0")
        if int(self.level_ptr[-1]) != int(self.level_nodes.shape[0]):
            raise ArtifactValidationError("level_ptr last value must equal level_nodes size")
        if set(self.level_nodes.astype(np.int64).tolist()) != set(range(node_count)):
            raise ArtifactValidationError("wavefront level_nodes must be a node permutation")


def build_wavefront_schedule(graph: TensorDag) -> WavefrontSchedule:
    """Build a dependency-safe topological level schedule from a TensorDag."""

    graph.validate()
    node_count = graph.node_count
    indegree = np.bincount(graph.edge_dst.astype(np.int64), minlength=node_count)
    children: list[list[int]] = [[] for _ in range(node_count)]
    for src, dst in zip(graph.edge_src.astype(np.int64), graph.edge_dst.astype(np.int64)):
        children[int(src)].append(int(dst))

    ready = [int(node) for node in np.flatnonzero(indegree == 0)]
    level_ptr = [0]
    level_nodes: list[int] = []
    node_level = np.zeros((node_count,), dtype=np.int64)
    current_level = 0

    while ready:
        next_ready: list[int] = []
        for node in ready:
            node_level[node] = current_level
            level_nodes.append(node)
        for node in ready:
            for child in children[node]:
                indegree[child] -= 1
                if indegree[child] == 0:
                    next_ready.append(child)
        level_ptr.append(len(level_nodes))
        ready = next_ready
        current_level += 1

    schedule = WavefrontSchedule(
        level_ptr=np.asarray(level_ptr, dtype=np.int64),
        level_nodes=np.asarray(level_nodes, dtype=np.int64),
        node_level=node_level,
    )
    schedule.validate(node_count)
    return schedule
