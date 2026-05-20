"""Dense CPU reference soft-Viterbi / log-sum-exp relaxation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.phmm._decoding_common import ViterbiInput, prepare_viterbi_input
from ad_phmm_align.phmm._cpu_reference import (
    aggregate_child_arrays,
    aggregate_parent_arrays,
    build_dense_reference_problem,
    sink_end_closure,
    smoothmax,
    source_forward_arrays,
)
from ad_phmm_align.phmm.forward_backward import forward_log_likelihood
from ad_phmm_align.phmm.parameters import PhmmParameterSet
from ad_phmm_align.phmm.ranges import EffectiveStateMask
from ad_phmm_align.phmm.wavefront import WavefrontSchedule


@dataclass(frozen=True)
class SoftViterbiResult:
    """Temperature-smoothed Viterbi score over packed DAG windows."""

    input: ViterbiInput
    temperature: float
    score: Any
    metadata: Mapping[str, object] = field(default_factory=dict)


def soft_viterbi_score(
    graph: TensorDag,
    parameters: PhmmParameterSet,
    temperature: float = 1.0,
    effective_support: Optional[EffectiveStateMask] = None,
    schedule: Optional[WavefrontSchedule] = None,
) -> SoftViterbiResult:
    """Compute a temperature-smoothed Viterbi score."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    problem = prepare_viterbi_input(
        graph,
        parameters,
        effective_support=effective_support,
        schedule=schedule,
    )
    reference = build_dense_reference_problem(graph, parameters)
    state_count = reference.state_count
    node_count = graph.node_count
    match = np.full((node_count, state_count), -np.inf, dtype=np.float64)
    insert = np.full((node_count, state_count + 1), -np.inf, dtype=np.float64)
    delete = np.full((node_count, state_count), -np.inf, dtype=np.float64)

    source_set = set(reference.source_nodes.astype(np.int64).tolist())
    for node_id in graph.topo_order.astype(np.int64).tolist():
        if node_id in source_set:
            node_match, node_insert, node_delete = source_forward_arrays(reference, int(node_id))
        else:
            prev_match, prev_insert, prev_delete = aggregate_parent_arrays(
                reference,
                int(node_id),
                match,
                insert,
                delete,
            )
            node_match = np.full((state_count,), -np.inf, dtype=np.float64)
            node_insert = np.full((state_count + 1,), -np.inf, dtype=np.float64)
            for state in range(state_count + 1):
                emit = reference.insert_emission[int(node_id), state]
                if not np.isfinite(emit):
                    continue
                candidates = [prev_insert[state] + reference.transitions.insert_to_insert[state]]
                if state > 0:
                    candidates.extend(
                        [
                            prev_match[state - 1] + reference.transitions.match_to_insert[state - 1],
                            prev_delete[state - 1] + reference.transitions.delete_to_insert[state - 1],
                        ]
                    )
                node_insert[state] = emit + smoothmax(candidates, temperature)
            for state in range(state_count):
                emit = reference.match_emission[int(node_id), state]
                if not np.isfinite(emit):
                    continue
                candidates = [prev_insert[state] + reference.transitions.insert_to_match[state]]
                if state > 0:
                    candidates.extend(
                        [
                            prev_match[state - 1] + reference.transitions.match_to_match[state - 1],
                            prev_delete[state - 1] + reference.transitions.delete_to_match[state - 1],
                        ]
                    )
                node_match[state] = emit + smoothmax(candidates, temperature)
            node_delete = np.full((state_count,), -np.inf, dtype=np.float64)
            for state in range(state_count):
                candidates = [node_insert[state] + reference.transitions.insert_to_delete[state]]
                if state > 0:
                    candidates.extend(
                        [
                            node_match[state - 1] + reference.transitions.match_to_delete[state - 1],
                            node_delete[state - 1] + reference.transitions.delete_to_delete[state - 1],
                        ]
                    )
                node_delete[state] = smoothmax(candidates, temperature)
        match[int(node_id)] = node_match
        insert[int(node_id)] = node_insert
        delete[int(node_id)] = node_delete

    end_match, end_insert, end_delete = sink_end_closure(reference.transitions, state_count)
    sink_scores = []
    for node_id in reference.sink_nodes.astype(np.int64).tolist():
        sink_scores.append(
            smoothmax(
                np.concatenate(
                    [
                        match[int(node_id)] + end_match,
                        insert[int(node_id)] + end_insert,
                        delete[int(node_id)] + end_delete,
                    ]
                ),
                temperature,
            )
        )
    score = float(smoothmax(np.asarray(sink_scores, dtype=np.float64), temperature))
    return SoftViterbiResult(
        input=problem,
        temperature=float(temperature),
        score=score,
        metadata={
            "implementation": "cpu_dense_reference",
            "forward_log_likelihood": float(forward_log_likelihood(graph, parameters).log_likelihood),
        },
    )
