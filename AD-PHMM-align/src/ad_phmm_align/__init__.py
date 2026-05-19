"""AD-PHMM-align public API."""

from ._version import __version__
from .exceptions import AdPhmmAlignError, ArtifactValidationError
from .graph import EdgeWindowOverlaps, PackedStateWindows, SubgraphBatch, TensorDag
from .io.schema import (
    GraphMetadata,
    InitialPhmmParameters,
    InitializationTrack,
    SourceFormat,
    StateIntervalSemantics,
)

__all__ = [
    "__version__",
    "AdPhmmAlignError",
    "ArtifactValidationError",
    "EdgeWindowOverlaps",
    "GraphMetadata",
    "InitialPhmmParameters",
    "InitializationTrack",
    "PackedStateWindows",
    "SourceFormat",
    "StateIntervalSemantics",
    "SubgraphBatch",
    "TensorDag",
]
