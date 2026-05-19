"""Loader for DAG-rust tensor-friendly graph exports."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ad_phmm_align.exceptions import UnsupportedArtifactError
from ad_phmm_align.graph.tensor_dag import TensorDag


class DagRustExportLoader:
    """Load DAG-rust exports into the internal TensorDag schema."""

    def __init__(self, export_dir: Union[Path, str]) -> None:
        self.export_dir = Path(export_dir)

    def load(self) -> TensorDag:
        """Load a DAG-rust export directory."""

        raise UnsupportedArtifactError(
            "DAG-rust export loading is not implemented yet."
        )
