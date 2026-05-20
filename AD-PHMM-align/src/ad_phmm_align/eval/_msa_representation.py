"""Runtime import helpers for the external ``msa_representation`` package."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MsaRepresentationRuntime:
    """Resolved external MSA representation modules."""

    module: Any
    msa_utils: Any


def _candidate_sys_paths() -> tuple[Path, ...]:
    candidates: list[Path] = []
    for key in (
        "AD_PHMM_ALIGN_MSA_REPRESENTATION_PATH",
        "MSA_REPRESENTATION_PATH",
    ):
        value = os.getenv(key)
        if not value:
            continue
        path = Path(value).expanduser()
        if path.name == "msa_representation":
            candidates.append(path.parent)
        else:
            candidates.append(path)
    return tuple(candidates)


def load_msa_representation_runtime() -> MsaRepresentationRuntime:
    """Import the external ``msa_representation`` package with env-path fallback."""

    try:
        module = importlib.import_module("msa_representation")
        msa_utils = importlib.import_module("msa_representation.msa_utils")
        return MsaRepresentationRuntime(module=module, msa_utils=msa_utils)
    except ImportError as original_error:
        for candidate in _candidate_sys_paths():
            candidate_text = str(candidate)
            if candidate_text not in sys.path:
                sys.path.insert(0, candidate_text)
            try:
                module = importlib.import_module("msa_representation")
                msa_utils = importlib.import_module("msa_representation.msa_utils")
                return MsaRepresentationRuntime(module=module, msa_utils=msa_utils)
            except ImportError:
                continue
        raise ImportError(
            "could not import msa_representation; install it, add it to PYTHONPATH, "
            "or set AD_PHMM_ALIGN_MSA_REPRESENTATION_PATH/MSA_REPRESENTATION_PATH"
        ) from original_error
