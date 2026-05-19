"""PHMM parameter containers and construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from ad_phmm_align.io.schema import InitialPhmmParameters


@dataclass(frozen=True)
class PhmmParameterSet:
    """PyTorch-ready PHMM parameter tensors.

    Values are typed as Any to avoid importing torch in the package scaffold.
    Concrete implementations should store torch.Tensor or torch.nn.Parameter
    values here.
    """

    match_emission: Any
    insert_emission: Any
    transition_logits: Mapping[str, Any]
    metadata: Mapping[str, Any]

    @classmethod
    def from_initial_parameters(
        cls, initial: InitialPhmmParameters
    ) -> "PhmmParameterSet":
        """Create PHMM parameters from a Rust-produced initialization artifact."""

        tensors = initial.tensors
        _validate_initial_shapes(initial)
        return cls(
            match_emission=tensors["match_emission"],
            insert_emission=tensors["insert_emission"],
            transition_logits=tensors["transition_logits"],
            metadata=initial.as_metadata_dict(),
        )


def _shape(tensor: Any) -> Optional[tuple]:
    value = getattr(tensor, "shape", None)
    if value is None:
        return None
    return tuple(int(dim) for dim in value)


def _validate_initial_shapes(initial: InitialPhmmParameters) -> None:
    global_state_count = initial.graph.global_state_count
    alphabet_size = initial.metadata.get("alphabet_size")
    match_shape = _shape(initial.require_tensor("match_emission"))
    insert_shape = _shape(initial.require_tensor("insert_emission"))

    if global_state_count is not None:
        if match_shape is not None and match_shape[0] != global_state_count:
            raise ValueError("match_emission first dimension must equal global_state_count")
        if insert_shape is not None and insert_shape[0] != global_state_count + 1:
            raise ValueError("insert_emission first dimension must equal global_state_count + 1")
    if alphabet_size is not None:
        alphabet_size = int(alphabet_size)
        if match_shape is not None and len(match_shape) > 1 and match_shape[1] != alphabet_size:
            raise ValueError("match_emission alphabet dimension does not match metadata")
        if insert_shape is not None and len(insert_shape) > 1 and insert_shape[1] != alphabet_size:
            raise ValueError("insert_emission alphabet dimension does not match metadata")
