"""Compatibility loader for current DAG-align graph artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ad_phmm_align.exceptions import ArtifactValidationError
from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.io.artifact_loader import TensorGraphArtifactLoader
from ad_phmm_align.io.schema import SourceFormat


class LegacyDagAlignLoader:
    """Load Pre-AD-prep artifacts derived from current DAG-align graphs.

    This adapter intentionally consumes the typed `tensor_graph.v1` export rather
    than legacy pickle/object arrays directly.
    """

    def __init__(self, graph_dir: Union[Path, str]) -> None:
        self.graph_dir = Path(graph_dir)

    def load(self) -> TensorDag:
        """Load a DAG-align-derived tensor artifact into the internal TensorDag schema."""

        artifact = TensorGraphArtifactLoader(self.graph_dir).load_artifact()
        if artifact.manifest.source_format is not SourceFormat.DAG_ALIGN_LEGACY:
            raise ArtifactValidationError(
                "artifact source_format is not dag_align_legacy"
            )
        return artifact.graph
