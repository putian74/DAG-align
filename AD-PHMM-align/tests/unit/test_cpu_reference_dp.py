"""CPU reference DP tests on tiny DAG fixtures."""

from __future__ import annotations

import numpy as np


def _tiny_chain_graph():
    from ad_phmm_align.graph import EdgeWindowOverlaps, PackedStateWindows, TensorDag
    from ad_phmm_align.io.schema import GraphMetadata

    graph = TensorDag(
        metadata=GraphMetadata(
            graph_id="tiny-chain",
            format_name="ad_phmm_tensor_graph",
            format_version="1",
            global_state_count=3,
        ),
        node_symbol=np.array([0, 1, 2], dtype=np.uint16),
        node_weight=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        edge_src=np.array([0, 1], dtype=np.int64),
        edge_dst=np.array([1, 2], dtype=np.int64),
        edge_weight=np.array([1.0, 1.0], dtype=np.float32),
        topo_order=np.array([0, 1, 2], dtype=np.int64),
        node_coordinate_left=np.array([0, 0, 0], dtype=np.int64),
        node_coordinate_right=np.array([3, 3, 3], dtype=np.int64),
        node_window_left=np.array([0, 0, 0], dtype=np.int64),
        node_window_right=np.array([3, 3, 3], dtype=np.int64),
        state_windows=PackedStateWindows(
            left=np.array([0, 0, 0], dtype=np.int64),
            right=np.array([3, 3, 3], dtype=np.int64),
            offset=np.array([0, 3, 6], dtype=np.int64),
            length=np.array([3, 3, 3], dtype=np.int64),
        ),
        edge_overlaps=EdgeWindowOverlaps(
            edge_ids=np.array([0, 1], dtype=np.int64),
            src_offset=np.array([0, 0], dtype=np.int64),
            dst_offset=np.array([0, 0], dtype=np.int64),
            length=np.array([3, 3], dtype=np.int64),
        ),
    )
    graph.validate()
    return graph


def _tiny_parameters():
    from ad_phmm_align.phmm import PhmmParameterSet, TransitionLogitView

    return PhmmParameterSet(
        match_emission=np.log(
            np.array(
                [
                    [0.7, 0.1, 0.1, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.1, 0.1, 0.7, 0.1],
                ],
                dtype=np.float64,
            )
        ),
        insert_emission=np.log(np.full((4, 4), 0.25, dtype=np.float64)),
        transition_logits=TransitionLogitView(
            tensor=np.zeros((4, 9), dtype=np.float64),
            order=("_mm", "_md", "_mi", "_dm", "_dd", "_di", "_im", "_id", "_ii"),
        ),
        metadata={"alphabet_size": 4},
    )


def test_cpu_reference_forward_backward_viterbi_and_losses() -> None:
    from ad_phmm_align.losses import posterior_entropy, soft_pairwise_score
    from ad_phmm_align.phmm import (
        backward_log_likelihood,
        forward_log_likelihood,
        posterior_occupancy,
        soft_viterbi_score,
        viterbi_decode,
    )

    graph = _tiny_chain_graph()
    parameters = _tiny_parameters()

    forward = forward_log_likelihood(graph, parameters)
    backward = backward_log_likelihood(graph, parameters)
    posterior = posterior_occupancy(
        graph,
        parameters,
        forward_result=forward,
        backward_result=backward,
    )
    viterbi = viterbi_decode(graph, parameters)
    soft_viterbi = soft_viterbi_score(graph, parameters, temperature=1.0)

    assert np.isfinite(forward.log_likelihood)
    assert np.isfinite(backward.log_likelihood)
    assert abs(float(forward.log_likelihood) - float(backward.log_likelihood)) < 1e-6
    assert posterior.match_posterior.shape == (3, 3)
    assert posterior.insert_posterior.shape == (3, 4)
    assert posterior.delete_posterior.shape == (3, 3)
    assert np.all(posterior.match_posterior >= 0.0)
    assert np.all(posterior.insert_posterior >= 0.0)
    assert np.all(posterior.delete_posterior >= 0.0)
    assert np.isfinite(viterbi.score)
    assert np.isfinite(soft_viterbi.score)
    assert float(viterbi.score) <= float(soft_viterbi.score) + 1e-6
    assert tuple(viterbi.node_ids.tolist()) == (0, 1, 2)
    assert len(viterbi.states) == 3
    assert viterbi.node_assignments is not None
    assert len(viterbi.node_assignments) >= 3
    assert {assignment.node_id for assignment in viterbi.node_assignments} == {0, 1, 2}
    assert all(hasattr(assignment, "channel") for assignment in viterbi.node_assignments)
    assert viterbi.metadata["node_assignment_count"] == len(viterbi.node_assignments)

    score_matrix = np.full((4, 4), -1.0, dtype=np.float64)
    np.fill_diagonal(score_matrix, 1.0)
    entropy = posterior_entropy(posterior)
    pairwise = soft_pairwise_score(posterior, score_matrix)
    assert np.isfinite(entropy)
    assert np.isfinite(pairwise)
    assert entropy >= 0.0
