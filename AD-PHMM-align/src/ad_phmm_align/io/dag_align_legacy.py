"""Compatibility loader for current DAG-align graph artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Union

from ad_phmm_align.exceptions import UnsupportedArtifactError
from ad_phmm_align.graph.tensor_dag import TensorDag


class LegacyDagAlignLoader:
    """Load current DAG-align artifacts into the internal TensorDag schema.

    This adapter is intentionally isolated so pickle/object-array compatibility
    does not leak into model or training code.
    """

    def __init__(self, graph_dir: Union[Path, str]) -> None:
        self.graph_dir = Path(graph_dir)

    def load(self) -> TensorDag:
        """Load a legacy graph directory.

        The concrete conversion will be implemented after fixture selection.
        """

        raise UnsupportedArtifactError(
            "Legacy DAG-align graph loading is not implemented yet."
        )
