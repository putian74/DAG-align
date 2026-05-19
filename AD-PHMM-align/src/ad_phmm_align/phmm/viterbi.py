"""Hard Viterbi decoding interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.phmm.parameters import PhmmParameterSet


@dataclass(frozen=True)
class ViterbiDecodeResult:
    """Decoded path/state output for evaluation."""

    score: float
    states: Any
    metadata: Optional[Mapping[str, Any]] = None


def viterbi_decode(
    graph: TensorDag, parameters: PhmmParameterSet
) -> ViterbiDecodeResult:
    """Decode hard Viterbi states."""

    raise NotImplementedError("viterbi_decode is not implemented yet")
