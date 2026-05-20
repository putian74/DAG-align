"""PHMM parameter and dynamic-programming interfaces."""

from ._decoding_common import ViterbiInput, prepare_viterbi_input
from .hard_path import (
    ViterbiBackpointerTable,
    ViterbiDecodeResult,
    ViterbiNodeAssignment,
    viterbi_decode,
)
from .initialization import load_initial_artifact, load_initial_parameters
from .parameters import PhmmParameterSet, TransitionLogitView
from .ranges import (
    EffectiveStateMask,
    empty_effective_state_mask,
    full_effective_state_mask,
    intersect_effective_state_masks,
    propagate_backward_support,
    propagate_forward_support,
)
from .soft_path import (
    BackwardPassResult,
    DagDynamicProgrammingInput,
    ForwardBackwardResult,
    ForwardPassResult,
    PosteriorOccupancy,
    SoftViterbiResult,
    backward_log_likelihood,
    forward_backward,
    forward_log_likelihood,
    posterior_occupancy,
    prepare_forward_backward_input,
    soft_viterbi_score,
)
from .wavefront import WavefrontSchedule, build_wavefront_schedule

__all__ = [
    "BackwardPassResult",
    "DagDynamicProgrammingInput",
    "EffectiveStateMask",
    "ForwardBackwardResult",
    "ForwardPassResult",
    "PhmmParameterSet",
    "PosteriorOccupancy",
    "SoftViterbiResult",
    "TransitionLogitView",
    "ViterbiBackpointerTable",
    "ViterbiDecodeResult",
    "ViterbiInput",
    "ViterbiNodeAssignment",
    "backward_log_likelihood",
    "WavefrontSchedule",
    "build_wavefront_schedule",
    "empty_effective_state_mask",
    "forward_backward",
    "forward_log_likelihood",
    "full_effective_state_mask",
    "intersect_effective_state_masks",
    "load_initial_artifact",
    "load_initial_parameters",
    "posterior_occupancy",
    "prepare_forward_backward_input",
    "prepare_viterbi_input",
    "propagate_backward_support",
    "propagate_forward_support",
    "soft_viterbi_score",
    "viterbi_decode",
]
