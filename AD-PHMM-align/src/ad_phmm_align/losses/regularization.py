"""Regularization losses used by the CPU reference trainer."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ad_phmm_align.phmm._cpu_reference import log_softmax, unpack_transition_bundle
from ad_phmm_align.phmm.parameters import PhmmParameterSet


def _softmax_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    out = np.empty_like(values, dtype=np.float64)
    for row in range(values.shape[0]):
        out[row] = np.exp(log_softmax(values[row]))
    return out


def _kl_divergence(reference: np.ndarray, target: np.ndarray, eps: float = 1e-12) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = reference > eps
    if not np.any(mask):
        return 0.0
    return float(np.sum(reference[mask] * (np.log(reference[mask]) - np.log(np.maximum(target[mask], eps)))))


def transition_anchor_regularization(
    parameters: PhmmParameterSet,
    reference_parameters: PhmmParameterSet,
) -> float:
    """Mean KL divergence from initialized transition families to current ones."""

    current = unpack_transition_bundle(parameters)
    reference = unpack_transition_bundle(reference_parameters)
    families = [
        (np.exp([reference.start_match, reference.start_delete, reference.start_insert]), np.exp([current.start_match, current.start_delete, current.start_insert])),
    ]
    for state in range(reference.match_to_insert.shape[0] - 1):
        families.append(
            (
                np.exp([reference.match_to_match[state], reference.match_to_delete[state], reference.match_to_insert[state]]),
                np.exp([current.match_to_match[state], current.match_to_delete[state], current.match_to_insert[state]]),
            )
        )
        families.append(
            (
                np.exp([reference.insert_to_match[state], reference.insert_to_delete[state], reference.insert_to_insert[state]]),
                np.exp([current.insert_to_match[state], current.insert_to_delete[state], current.insert_to_insert[state]]),
            )
        )
        families.append(
            (
                np.exp([reference.delete_to_match[state], reference.delete_to_delete[state], reference.delete_to_insert[state]]),
                np.exp([current.delete_to_match[state], current.delete_to_delete[state], current.delete_to_insert[state]]),
            )
        )
    if reference.match_to_insert.shape[0] > 0:
        last = reference.match_to_insert.shape[0] - 1
        families.append(
            (
                np.exp([reference.match_to_end, reference.match_to_insert[last]]),
                np.exp([current.match_to_end, current.match_to_insert[last]]),
            )
        )
        families.append(
            (
                np.exp([reference.insert_to_end, reference.insert_to_insert[last + 1]]),
                np.exp([current.insert_to_end, current.insert_to_insert[last + 1]]),
            )
        )
        families.append(
            (
                np.exp([reference.delete_to_end, reference.delete_to_insert[last]]),
                np.exp([current.delete_to_end, current.delete_to_insert[last]]),
            )
        )
    return float(np.mean([_kl_divergence(ref, cur) for ref, cur in families]))


def emission_anchor_regularization(
    parameters: PhmmParameterSet,
    reference_parameters: PhmmParameterSet,
    insert_background_weight: float = 1.0,
    background: Optional[np.ndarray] = None,
) -> float:
    """Mean KL from initialized emissions to current emissions."""

    match_probs = _softmax_rows(np.asarray(parameters.match_emission, dtype=np.float64))
    reference_match = _softmax_rows(np.asarray(reference_parameters.match_emission, dtype=np.float64))
    insert_probs = _softmax_rows(np.asarray(parameters.insert_emission, dtype=np.float64))
    if background is None:
        background = np.full((insert_probs.shape[1],), 1.0 / insert_probs.shape[1], dtype=np.float64)
    match_term = np.mean([
        _kl_divergence(reference_match[state], match_probs[state])
        for state in range(match_probs.shape[0])
    ])
    insert_term = np.mean([
        _kl_divergence(background, insert_probs[state])
        for state in range(insert_probs.shape[0])
    ])
    return float(match_term + insert_background_weight * insert_term)


def emission_smoothness_regularization(parameters: PhmmParameterSet, eps: float = 1e-12) -> float:
    """Mean Jensen-Shannon divergence between neighboring match emissions."""

    match_probs = _softmax_rows(np.asarray(parameters.match_emission, dtype=np.float64))
    if match_probs.shape[0] < 2:
        return 0.0
    penalties = []
    for state in range(match_probs.shape[0] - 1):
        left = match_probs[state]
        right = match_probs[state + 1]
        midpoint = 0.5 * (left + right)
        penalties.append(
            0.5 * _kl_divergence(left, midpoint, eps=eps)
            + 0.5 * _kl_divergence(right, midpoint, eps=eps)
        )
    return float(np.mean(penalties))


def logit_l2_regularization(parameters: PhmmParameterSet) -> float:
    """Small quadratic safety penalty on transition and emission logits."""

    transition = np.asarray(parameters.transition_logits.tensor, dtype=np.float64)
    match = np.asarray(parameters.match_emission, dtype=np.float64)
    insert = np.asarray(parameters.insert_emission, dtype=np.float64)
    return float(np.mean(transition**2) + np.mean(match**2) + np.mean(insert**2))


def active_state_regularization(
    active_global_states: np.ndarray | Any,
    global_state_count: int,
) -> float:
    """Penalty proportional to the active-state fraction of the global profile."""

    if global_state_count <= 0:
        return 0.0
    active = np.asarray(active_global_states, dtype=np.int64)
    if active.size == 0:
        return 0.0
    return float(np.unique(active).shape[0] / float(global_state_count))


def transition_regularization(
    parameters: PhmmParameterSet,
    reference_parameters: Optional[PhmmParameterSet] = None,
) -> Any:
    """Backward-compatible transition regularization helper."""

    if reference_parameters is None:
        return logit_l2_regularization(parameters)
    return transition_anchor_regularization(parameters, reference_parameters)
