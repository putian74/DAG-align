"""Loader for DAG-rust tensor-friendly graph exports."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ad_phmm_align.exceptions import ArtifactValidationError
from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.io.artifact_loader import TensorGraphArtifactLoader
from ad_phmm_align.io.schema import SourceFormat


class DagRustExportLoader:
    """Load DAG-rust exports into the internal TensorDag schema."""

    def __init__(self, export_dir: Union[Path, str]) -> None:
        self.export_dir = Path(export_dir)

    def load(self) -> TensorDag:
        """Load a DAG-rust export directory."""

        artifact = TensorGraphArtifactLoader(self.export_dir).load_artifact()
        if artifact.manifest.source_format is not SourceFormat.DAG_RUST:
            raise ArtifactValidationError("artifact source_format is not dag_rust")
        return artifact.graph
