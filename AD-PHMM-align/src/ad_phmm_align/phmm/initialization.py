"""Initialization artifact loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from ad_phmm_align.io.artifact_loader import (
    InitialPhmmArtifactLoader,
    LoadedInitialPhmmArtifact,
)
from ad_phmm_align.io.schema import InitialPhmmParameters, InitializationTrack


def load_initial_artifact(
    path: Union[Path, str],
    track: Optional[InitializationTrack] = None,
) -> LoadedInitialPhmmArtifact:
    """Load an initialization manifest together with its tensor arrays."""

    return InitialPhmmArtifactLoader(path, track=track).load()


def load_initial_parameters(
    path: Union[Path, str],
    track: Optional[InitializationTrack] = None,
) -> InitialPhmmParameters:
    """Load a Rust-produced common PHMM initialization artifact."""

    return load_initial_artifact(path, track=track).parameters
