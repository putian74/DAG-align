"""Hard alignment metrics computed from ``ValSparseMSA`` column counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np

from ._msa_representation import load_msa_representation_runtime


@dataclass(frozen=True)
class AlignmentMetrics:
    """Summary metrics for decoded alignments."""

    log_likelihood: Optional[float] = None
    entropy: Optional[float] = None
    pairwise_score: Optional[float] = None
    core_entropy: Optional[float] = None
    alignment_length: Optional[int] = None
    core_column_count: Optional[int] = None
    sequence_count: Optional[int] = None


def default_pairwise_matrices(alphabet_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return simple identity/mismatch matrices for hard scaled-SP."""

    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive")
    positive = np.zeros((alphabet_size, alphabet_size), dtype=np.float64)
    negative = np.zeros((alphabet_size, alphabet_size), dtype=np.float64)
    for base in range(1, min(alphabet_size, 5)):
        positive[base, base] = 1.0
    for left in range(1, min(alphabet_size, 5)):
        for right in range(1, min(alphabet_size, 5)):
            if left != right:
                negative[left, right] = -1.0
    return positive, negative


def _column_value_counts(msa: Any, col_idx: int) -> dict[int, float]:
    counts: dict[int, float] = {}
    rare_total = 0
    for value, positions in msa.get_sparse_col(col_idx):
        if isinstance(positions, tuple) and len(positions) == 3:
            count = int(np.sum(np.asarray(positions[2], dtype=np.int64)))
        else:
            count = int(len(positions))
        counts[int(value)] = counts.get(int(value), 0.0) + float(count)
        rare_total += count
    reference_value = int(np.asarray(msa.ref_vector)[col_idx])
    reference_count = int(msa.nrow) - rare_total
    if reference_count > 0:
        counts[reference_value] = counts.get(reference_value, 0.0) + float(reference_count)
    return counts


def _expanded_canonical_counts(raw_counts: Mapping[int, float]) -> dict[int, float]:
    runtime = load_msa_representation_runtime()
    digital_to_iupac = list(runtime.module.DIGITAL_TO_IUPAC)
    degenerate_map = dict(runtime.msa_utils.IUPAC_DEGENERATE_MAP)
    counts = {int(key): float(value) for key, value in raw_counts.items() if float(value) != 0.0}
    if set(counts) == {0}:
        return counts
    expanded = dict(counts)
    for value in sorted(key for key in counts if key > 4):
        char = str(digital_to_iupac[value])
        canonical_bases = tuple(int(base) for base in degenerate_map[char][0])
        distributed = counts[value] / float(len(canonical_bases))
        for canonical in canonical_bases:
            expanded[canonical] = expanded.get(canonical, 0.0) + distributed
        del expanded[value]
    return expanded


def calculate_column_metrics(
    raw_counts: Mapping[int, float],
    length: int,
    positive: np.ndarray,
    negative: np.ndarray,
) -> tuple[float, float, float, float]:
    """Match legacy DAG-align's per-column SP/entropy scoring semantics."""

    counts = _expanded_canonical_counts(raw_counts)
    if set(counts) == {0}:
        return 0.0, 0.0, 0.0, float(length)

    negative_score = 0.0
    positive_score = 0.0
    values = sorted(counts)
    for left_index, left_value in enumerate(values):
        left_count = counts[left_value]
        positive_score += (
            left_count * (left_count - 1.0) / 2.0 * float(positive[left_value][left_value])
        )
        negative_score += (
            left_count * (left_count - 1.0) / 2.0 * float(negative[left_value][left_value])
        )
        for right_value in values[left_index + 1 :]:
            right_count = counts[right_value]
            positive_score += left_count * right_count * float(
                positive[left_value][right_value]
            )
            negative_score += left_count * right_count * float(
                negative[left_value][right_value]
            )

    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        probability = count / float(length)
        entropy += -probability * float(np.log(probability))
    return negative_score, positive_score, entropy, float(counts.get(0, 0.0))


def summarize_alignment_metrics(
    alignment: Any,
    *,
    positive: Optional[np.ndarray] = None,
    negative: Optional[np.ndarray] = None,
    gap_threshold: float = 0.7,
    log_likelihood: Optional[float] = None,
) -> AlignmentMetrics:
    """Compute hard scaled-SP and entropy directly from ``ValSparseMSA`` columns."""

    msa = alignment.msa if hasattr(alignment, "msa") else alignment
    length = int(msa.ncol)
    sequence_count = int(msa.nrow)
    if length == 0 or sequence_count < 2:
        return AlignmentMetrics(
            log_likelihood=log_likelihood,
            entropy=0.0,
            pairwise_score=0.0,
            core_entropy=0.0,
            alignment_length=length,
            core_column_count=0,
            sequence_count=sequence_count,
        )

    alphabet_size = int(getattr(msa, "meta", {}).get("alphabet_size", 16))
    if positive is None or negative is None:
        positive, negative = default_pairwise_matrices(alphabet_size)

    total_entropy = 0.0
    total_negative = 0.0
    total_positive = 0.0
    total_core_entropy = 0.0
    core_columns = 0
    for col_idx in range(length):
        column_counts = _column_value_counts(msa, col_idx)
        negative_score, positive_score, entropy, gap_count = calculate_column_metrics(
            column_counts,
            sequence_count,
            positive,
            negative,
        )
        total_negative += negative_score
        total_positive += positive_score
        total_entropy += entropy
        if (gap_count / float(sequence_count)) < float(gap_threshold):
            core_columns += 1
            total_core_entropy += entropy

    scaled_sp = 2.0 * (total_positive + total_negative) / (
        float(sequence_count) * float(sequence_count - 1) * float(length)
    )
    return AlignmentMetrics(
        log_likelihood=log_likelihood,
        entropy=float(total_entropy),
        pairwise_score=float(scaled_sp),
        core_entropy=0.0 if core_columns == 0 else float(total_core_entropy / core_columns),
        alignment_length=length,
        core_column_count=core_columns,
        sequence_count=sequence_count,
    )
