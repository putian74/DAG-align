"""Input/output adapters and artifact schemas."""

from __future__ import annotations

from .schema import (
    ArraySpec,
    GraphMetadata,
    InitialPhmmManifest,
    InitialPhmmParameters,
    InitializationTrack,
    SourceFormat,
    StateIntervalSemantics,
    TensorGraphManifest,
)

__all__ = [
    "ArraySpec",
    "GraphMetadata",
    "InitialPhmmArtifactLoader",
    "InitialPhmmManifest",
    "InitialPhmmParameters",
    "InitializationTrack",
    "LoadedInitialPhmmArtifact",
    "SourceFormat",
    "StateIntervalSemantics",
    "TensorGraphArtifactLoader",
    "TensorGraphManifest",
    "required_array_specs",
]


def __getattr__(name: str):
    if name in {
        "InitialPhmmArtifactLoader",
        "LoadedInitialPhmmArtifact",
        "TensorGraphArtifactLoader",
        "required_array_specs",
    }:
        from .artifact_loader import (
            InitialPhmmArtifactLoader,
            LoadedInitialPhmmArtifact,
            TensorGraphArtifactLoader,
            required_array_specs,
        )

        return {
            "InitialPhmmArtifactLoader": InitialPhmmArtifactLoader,
            "LoadedInitialPhmmArtifact": LoadedInitialPhmmArtifact,
            "TensorGraphArtifactLoader": TensorGraphArtifactLoader,
            "required_array_specs": required_array_specs,
        }[name]
    raise AttributeError(name)
