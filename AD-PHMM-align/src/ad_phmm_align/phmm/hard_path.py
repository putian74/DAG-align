"""Explicit hard/max-product decoding path interfaces."""

from ._decoding_common import ViterbiInput, prepare_viterbi_input
from .viterbi import (
    ViterbiBackpointerTable,
    ViterbiDecodeResult,
    ViterbiNodeAssignment,
    viterbi_decode,
)

__all__ = [
    "ViterbiBackpointerTable",
    "ViterbiDecodeResult",
    "ViterbiInput",
    "ViterbiNodeAssignment",
    "prepare_viterbi_input",
    "viterbi_decode",
]
