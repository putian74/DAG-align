"""Initialization artifact loading and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ad_phmm_align.exceptions import UnsupportedArtifactError
from ad_phmm_align.io.schema import InitialPhmmParameters


def load_initial_parameters(path: Union[Path, str]) -> InitialPhmmParameters:
    """Load a Rust-produced InitialPhmmParameters artifact."""

    raise UnsupportedArtifactError(
        f"Initial parameter artifact loading is not implemented yet: {path}"
    )
