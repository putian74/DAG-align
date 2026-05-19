"""Subgraph sampling interfaces."""

from __future__ import annotations

from ad_phmm_align.graph.subgraph import SubgraphBatch
from ad_phmm_align.graph.tensor_dag import TensorDag


class SubgraphSampler:
    """Base class for sequence-batch induced subgraph samplers."""

    def __init__(self, graph: TensorDag) -> None:
        self.graph = graph

    def sample(self) -> SubgraphBatch:
        """Sample one subgraph batch."""

        raise NotImplementedError("Subgraph sampling is not implemented yet")

