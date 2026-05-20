"""Explicit soft/differentiable inference path interfaces."""

from .forward_backward import (
    BackwardPassResult,
    DagDynamicProgrammingInput,
    ForwardBackwardResult,
    ForwardPassResult,
    PosteriorOccupancy,
    backward_log_likelihood,
    forward_backward,
    forward_log_likelihood,
    posterior_occupancy,
    prepare_forward_backward_input,
)
from .soft_viterbi import SoftViterbiResult, soft_viterbi_score

__all__ = [
    "BackwardPassResult",
    "DagDynamicProgrammingInput",
    "ForwardBackwardResult",
    "ForwardPassResult",
    "PosteriorOccupancy",
    "SoftViterbiResult",
    "backward_log_likelihood",
    "forward_backward",
    "forward_log_likelihood",
    "posterior_occupancy",
    "prepare_forward_backward_input",
    "soft_viterbi_score",
]
