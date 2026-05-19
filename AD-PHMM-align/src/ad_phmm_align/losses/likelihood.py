"""Likelihood-based losses."""

from __future__ import annotations

from typing import Any


def negative_log_likelihood(log_likelihood: Any) -> Any:
    """Return a negative log-likelihood loss."""

    return -log_likelihood

