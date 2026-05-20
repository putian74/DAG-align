"""Likelihood-based losses."""

from __future__ import annotations

from typing import Any


def negative_log_likelihood(log_likelihood: Any, normalizer: float = 1.0) -> Any:
    """Return a negative log-likelihood loss."""

    if normalizer <= 0.0:
        raise ValueError("normalizer must be positive")
    return -log_likelihood / normalizer
