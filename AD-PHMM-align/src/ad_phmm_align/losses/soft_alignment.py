"""Differentiable soft alignment statistics and losses."""

from __future__ import annotations

from typing import Any

import numpy as np


def soft_column_counts(posterior: Any) -> tuple[np.ndarray, int]:
    """Aggregate expected per-symbol counts from soft match occupancies."""

    if not hasattr(posterior, "match_posterior"):
        raise TypeError("soft alignment losses expect a PosteriorOccupancy-like object")
    match_posterior = np.asarray(posterior.match_posterior, dtype=np.float64)
    metadata = getattr(posterior, "metadata", {})
    node_symbol = np.asarray(metadata.get("node_symbol"), dtype=np.int64)
    alphabet_size = int(metadata.get("alphabet_size", 0))
    if node_symbol.ndim != 1 or node_symbol.shape[0] != match_posterior.shape[0]:
        raise ValueError("posterior metadata must include node_symbol for every node")
    if alphabet_size <= 0:
        raise ValueError("posterior metadata must include a positive alphabet_size")
    counts = np.zeros((match_posterior.shape[1], alphabet_size), dtype=np.float64)
    for node_id, symbol in enumerate(node_symbol.tolist()):
        counts[:, int(symbol)] += match_posterior[node_id]
    return counts, alphabet_size


def soft_alignment_entropy(
    posterior: Any,
    normalize: bool = True,
    eps: float = 1e-12,
) -> Any:
    """Compute mean soft match-column entropy from posterior occupancies."""

    counts, alphabet_size = soft_column_counts(posterior)
    column_mass = np.sum(counts, axis=1)
    active = column_mass > eps
    if not np.any(active):
        return 0.0
    probs = counts[active] / np.maximum(column_mass[active, None], eps)
    entropy = -np.sum(probs * np.log(np.maximum(probs, eps)), axis=1)
    value = float(np.mean(entropy))
    if normalize:
        value /= float(np.log(alphabet_size))
    return value


def soft_pairwise_score(occupancy: Any, score_matrix: Any, eps: float = 1e-12) -> Any:
    """Compute a differentiable scaled sum-of-pairs approximation."""

    counts, alphabet_size = soft_column_counts(occupancy)
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    if score_matrix.shape != (alphabet_size, alphabet_size):
        raise ValueError("score_matrix shape must match the posterior alphabet size")

    diagonal = np.diag(score_matrix)
    numerator = 0.0
    denominator = 0.0
    for column_counts in counts:
        mass = float(np.sum(column_counts))
        if mass <= eps:
            continue
        diag_score = float(
            np.sum(
                0.5
                * column_counts
                * np.maximum(column_counts - 1.0, 0.0)
                * diagonal
            )
        )
        off_score = 0.0
        for left in range(alphabet_size):
            for right in range(left + 1, alphabet_size):
                off_score += float(
                    column_counts[left] * column_counts[right] * score_matrix[left, right]
                )
        numerator += diag_score + off_score
        denominator += max(mass * (mass - 1.0), 0.0)
    if denominator <= eps:
        return 0.0
    return float(2.0 * numerator / denominator)
