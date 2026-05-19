"""Soft-Viterbi / log-sum-exp relaxation interfaces."""

from __future__ import annotations

from typing import Any

from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.phmm.parameters import PhmmParameterSet


def soft_viterbi_score(
    graph: TensorDag, parameters: PhmmParameterSet, temperature: float = 1.0
) -> Any:
    """Compute a temperature-smoothed Viterbi score."""

    raise NotImplementedError("soft_viterbi_score is not implemented yet")

