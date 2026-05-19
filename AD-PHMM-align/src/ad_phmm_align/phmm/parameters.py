"""PHMM parameter containers and construction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from ad_phmm_align.io.schema import InitialPhmmParameters

_DEFAULT_TRANSITION_ORDER = (
    "_mm",
    "_md",
    "_mi",
    "_dm",
    "_dd",
    "_di",
    "_im",
    "_id",
    "_ii",
)


@dataclass(frozen=True)
class TransitionLogitView:
    """Named access to the packed transition-logit tensor."""

    tensor: Any
    order: Sequence[str]

    def require(self, name: str) -> Any:
        """Return one named transition column from the packed transition tensor."""

        try:
            index = tuple(self.order).index(name)
        except ValueError as exc:
            raise KeyError(name) from exc
        return self.tensor[..., index]

    def as_mapping(self) -> Mapping[str, Any]:
        """Expose transition columns through a stable name -> tensor mapping."""

        return {name: self.require(name) for name in self.order}


@dataclass(frozen=True)
class PhmmParameterSet:
    """PyTorch-ready PHMM parameter tensors.

    Values are typed as Any to avoid importing torch in the package scaffold.
    Concrete implementations should store torch.Tensor or torch.nn.Parameter
    values here.
    """

    match_emission: Any
    insert_emission: Any
    transition_logits: TransitionLogitView
    metadata: Mapping[str, Any]

    @classmethod
    def from_initial_parameters(
        cls, initial: InitialPhmmParameters
    ) -> "PhmmParameterSet":
        """Create PHMM parameters from a Rust-produced initialization artifact."""

        tensors = initial.tensors
        _validate_initial_shapes(initial)
        transition_order = tuple(
            str(name)
            for name in initial.metadata.get("transition_order", _DEFAULT_TRANSITION_ORDER)
        )
        return cls(
            match_emission=tensors["match_emission"],
            insert_emission=tensors["insert_emission"],
            transition_logits=TransitionLogitView(
                tensor=tensors["transition_logits"],
                order=transition_order,
            ),
            metadata=initial.as_metadata_dict(),
        )


def _shape(tensor: Any) -> Optional[tuple[int, ...]]:
    value = getattr(tensor, "shape", None)
    if value is None:
        return None
    return tuple(int(dim) for dim in value)


def _validate_initial_shapes(initial: InitialPhmmParameters) -> None:
    global_state_count = initial.graph.global_state_count
    alphabet_size = initial.metadata.get("alphabet_size")
    transition_order = tuple(
        str(name)
        for name in initial.metadata.get("transition_order", _DEFAULT_TRANSITION_ORDER)
    )
    match_shape = _shape(initial.require_tensor("match_emission"))
    insert_shape = _shape(initial.require_tensor("insert_emission"))
    transition_shape = _shape(initial.require_tensor("transition_logits"))

    if global_state_count is not None:
        if match_shape is not None and match_shape[0] != global_state_count:
            raise ValueError("match_emission first dimension must equal global_state_count")
        if insert_shape is not None and insert_shape[0] != global_state_count + 1:
            raise ValueError(
                "insert_emission first dimension must equal global_state_count + 1"
            )
        if transition_shape is not None and transition_shape[0] != global_state_count + 1:
            raise ValueError(
                "transition_logits first dimension must equal global_state_count + 1"
            )
    if alphabet_size is not None:
        alphabet_size = int(alphabet_size)
        if match_shape is not None and len(match_shape) > 1 and match_shape[1] != alphabet_size:
            raise ValueError("match_emission alphabet dimension does not match metadata")
        if insert_shape is not None and len(insert_shape) > 1 and insert_shape[1] != alphabet_size:
            raise ValueError("insert_emission alphabet dimension does not match metadata")
    if transition_shape is not None:
        if len(transition_shape) != 2:
            raise ValueError("transition_logits must be a rank-2 tensor")
        if transition_shape[1] != len(transition_order):
            raise ValueError("transition_logits width does not match transition_order")
