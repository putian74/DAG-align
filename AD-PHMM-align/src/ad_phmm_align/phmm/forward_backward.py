"""Differentiable forward/backward dynamic programs."""

from __future__ import annotations

from typing import Any

from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.phmm.parameters import PhmmParameterSet


def forward_log_likelihood(graph: TensorDag, parameters: PhmmParameterSet) -> Any:
    """Compute DAG-PHMM forward log-likelihood."""

    raise NotImplementedError("forward_log_likelihood is not implemented yet")


def posterior_occupancy(graph: TensorDag, parameters: PhmmParameterSet) -> Any:
    """Compute posterior occupancy summaries from forward/backward passes."""

    raise NotImplementedError("posterior_occupancy is not implemented yet")

