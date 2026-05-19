"""Typed schemas shared by graph and initialization artifact loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence


class InitializationTrack(str, Enum):
    """Initialization tracks produced by DAG-rust/Rust preprocessing."""

    LEGACY_CURRENT = "legacy_current"
    REFERENCE_MSA = "reference_msa"


class StateIntervalSemantics(str, Enum):
    """Supported PHMM state-interval conventions."""

    HALF_OPEN = "half_open"


class SourceFormat(str, Enum):
    """Graph artifact source families supported by the manifest contract."""

    DAG_ALIGN_LEGACY = "dag_align_legacy"
    DAG_RUST = "dag_rust"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class ArraySpec:
    """Manifest entry for one typed artifact array."""

    name: str
    path: Path
    dtype: str
    shape: Sequence[int]
    required: bool = True
    description: Optional[str] = None


@dataclass(frozen=True)
class GraphMetadata:
    """Metadata that identifies a graph artifact and its coordinate system."""

    graph_id: str
    format_name: str
    format_version: str
    source_path: Optional[Path] = None
    global_state_count: Optional[int] = None
    alphabet: Optional[Sequence[str]] = None
    state_interval_semantics: StateIntervalSemantics = (
        StateIntervalSemantics.HALF_OPEN
    )
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TensorGraphManifest:
    """Typed manifest contract for a Pre-AD-prep tensor graph artifact."""

    metadata: GraphMetadata
    node_count: int
    edge_count: int
    sequence_count: int
    arrays: Mapping[str, ArraySpec]
    source_format: SourceFormat
    symbol_encoding: Mapping[str, int] = field(default_factory=dict)
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitialPhmmParameters:
    """Common PHMM initialization artifact consumed by PyTorch code."""

    track: InitializationTrack
    graph: GraphMetadata
    tensors: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def require_tensor(self, name: str) -> Any:
        """Return a named tensor-like object or raise KeyError."""

        return self.tensors[name]

    def as_metadata_dict(self) -> MutableMapping[str, Any]:
        """Return serializable metadata for logging and checkpoints."""

        data: MutableMapping[str, Any] = {
            "track": self.track.value,
            "graph_id": self.graph.graph_id,
            "graph_format": self.graph.format_name,
            "graph_format_version": self.graph.format_version,
            "global_state_count": self.graph.global_state_count,
            "alphabet": self.graph.alphabet,
            "state_interval_semantics": self.graph.state_interval_semantics.value,
        }
        data.update(dict(self.metadata))
        return data


@dataclass(frozen=True)
class InitialPhmmManifest:
    """Typed initialization manifest before loading tensor arrays."""

    track: InitializationTrack
    global_state_count: int
    alphabet_size: int
    transition_order: Sequence[str]
    match_emission: ArraySpec
    insert_emission: ArraySpec
    transition_logits: ArraySpec
    metadata: Mapping[str, Any] = field(default_factory=dict)
