"""Shared exception types."""


class AdPhmmAlignError(Exception):
    """Base class for AD-PHMM-align errors."""


class ArtifactValidationError(AdPhmmAlignError):
    """Raised when a graph or initialization artifact fails validation."""


class UnsupportedArtifactError(AdPhmmAlignError):
    """Raised when an input artifact format is not supported yet."""

