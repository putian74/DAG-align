"""Tests for sequence-batch subgraph sampling."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _sampling_graph():
    from ad_phmm_align.graph import EdgeWindowOverlaps, PackedStateWindows, TensorDag
    from ad_phmm_align.io.schema import GraphMetadata

    graph = TensorDag(
        metadata=GraphMetadata(
            graph_id="sampling-graph",
            format_name="ad_phmm_tensor_graph",
            format_version="1",
            global_state_count=2,
        ),
        node_symbol=np.array([0, 1, 2, 3], dtype=np.uint16),
        node_weight=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        edge_src=np.array([0, 0, 1, 2], dtype=np.int64),
        edge_dst=np.array([1, 2, 3, 3], dtype=np.int64),
        edge_weight=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        topo_order=np.array([0, 1, 2, 3], dtype=np.int64),
        node_coordinate_left=np.array([0, 0, 0, 0], dtype=np.int64),
        node_coordinate_right=np.array([2, 2, 2, 2], dtype=np.int64),
        node_window_left=np.array([0, 0, 0, 0], dtype=np.int64),
        node_window_right=np.array([2, 2, 2, 2], dtype=np.int64),
        state_windows=PackedStateWindows(
            left=np.array([0, 0, 0, 0], dtype=np.int64),
            right=np.array([2, 2, 2, 2], dtype=np.int64),
            offset=np.array([0, 2, 4, 6], dtype=np.int64),
            length=np.array([2, 2, 2, 2], dtype=np.int64),
        ),
        edge_overlaps=EdgeWindowOverlaps(
            edge_ids=np.array([0, 1, 2, 3], dtype=np.int64),
            src_offset=np.array([0, 0, 0, 0], dtype=np.int64),
            dst_offset=np.array([0, 0, 0, 0], dtype=np.int64),
            length=np.array([2, 2, 2, 2], dtype=np.int64),
        ),
        extra={
            "symbol_encoding": {"A": 0, "T": 1, "C": 2, "G": 3},
            "sequence_id": np.array([0, 1], dtype=np.uint64),
            "sequence_names": ["seq0", "seq1"],
            "node_source_offset": np.array([0, 2, 3, 4], dtype=np.uint64),
            "node_source_len": np.array([2, 1, 1, 2], dtype=np.uint64),
            "source_sequence_id": np.array([0, 1, 0, 1, 0, 1], dtype=np.uint64),
            "source_position": np.array([0, 0, 1, 1, 2, 2], dtype=np.uint64),
            "ref_sequence_symbols": np.array([0, 1], dtype=np.uint16),
            "manifest_arrays": (),
            "source_format": "synthetic",
        },
    )
    graph.validate()
    return graph


def _tiny_parameters():
    from ad_phmm_align.phmm import PhmmParameterSet, TransitionLogitView

    return PhmmParameterSet(
        match_emission=np.log(
            np.array([[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1]], dtype=np.float64)
        ),
        insert_emission=np.log(np.full((3, 4), 0.25, dtype=np.float64)),
        transition_logits=TransitionLogitView(
            tensor=np.zeros((3, 9), dtype=np.float64),
            order=("_mm", "_md", "_mi", "_dm", "_dd", "_di", "_im", "_id", "_ii"),
        ),
        metadata={"alphabet_size": 4},
    )


def test_state_mask_spec_materializes_ranges() -> None:
    from ad_phmm_align.sampling import CandidateStateRange, StateMaskSpec

    spec = StateMaskSpec(
        global_state_count=6,
        mode="fixed",
        candidate_ranges=(
            CandidateStateRange(left=1, right=3),
            CandidateStateRange(left=4, right=6),
        ),
    )

    assert spec.active_global_state_ids().tolist() == [1, 2, 4, 5]


def test_subgraph_sampler_materializes_sequence_induced_graph() -> None:
    from ad_phmm_align.sampling import CandidateStateRange, StateMaskSpec, SubgraphSampler

    graph = _sampling_graph()
    sampler = SubgraphSampler(
        graph,
        sequence_batch_size=1,
        state_mask_spec=StateMaskSpec(
            global_state_count=2,
            mode="fixed",
            candidate_ranges=(CandidateStateRange(left=1, right=2),),
        ),
        seed=7,
    )

    sampled = sampler.sample_graph(sequence_ids=np.array([0], dtype=np.int64))

    assert sampled.subgraph.sequence_ids.tolist() == [0]
    assert sampled.subgraph.node_ids.tolist() == [0, 1, 3]
    assert sampled.subgraph.edge_ids.tolist() == [0, 2]
    assert sampled.subgraph.global_state_ids.tolist() == [1]
    assert sampled.graph.node_count == 3
    assert sampled.graph.edge_count == 2
    assert sampled.graph.node_window_left.tolist() == [1, 1, 1]
    assert sampled.graph.node_window_right.tolist() == [2, 2, 2]
    assert sampled.graph.extra["sequence_id"].tolist() == [0]
    assert sampled.graph.extra["sequence_names"] == ["seq0"]
    assert sampled.graph.extra["source_sequence_id"].tolist() == [0, 0, 0]


def test_trainer_build_step_input_uses_sampled_graph(tmp_path) -> None:
    from ad_phmm_align.io.schema import GraphMetadata, InitialPhmmParameters, InitializationTrack
    from ad_phmm_align.train import (
        SubgraphSamplingConfig,
        TensorGraphArtifact,
        Trainer,
        TrainingConfig,
        TrainingRuntimeArtifacts,
    )

    graph = _sampling_graph()
    runtime = TrainingRuntimeArtifacts(
        graph_artifact=TensorGraphArtifact(
            root=Path(tmp_path) / "graph",
            manifest=None,  # type: ignore[arg-type]
            graph=graph,
        ),
        initial_parameters=InitialPhmmParameters(
            track=InitializationTrack.LEGACY_CURRENT,
            graph=GraphMetadata(
                graph_id=graph.metadata.graph_id,
                format_name=graph.metadata.format_name,
                format_version=graph.metadata.format_version,
                global_state_count=graph.metadata.global_state_count,
                alphabet=graph.metadata.alphabet,
                state_interval_semantics=graph.metadata.state_interval_semantics,
            ),
            tensors={},
            metadata={"alphabet_size": 4, "transition_order": ("_mm",)},
        ),
        parameters=_tiny_parameters(),
    )
    trainer = Trainer(
        TrainingConfig(
            graph_path=Path(tmp_path) / "graph",
            initialization_path=Path(tmp_path) / "init",
            output_dir=Path(tmp_path) / "out",
            sampling=SubgraphSamplingConfig(
                strategy="sequence_batch",
                sequence_batch_size=1,
            ),
        )
    )

    replica = trainer.build_replicas(runtime.parameters)[0]
    step_input = trainer.build_step_input(runtime, replica, step_index=0)
    sampled_graph = step_input.metadata["_graph"]

    assert sampled_graph.node_count < graph.node_count
    assert step_input.batch.metadata["sampling_strategy"] == "sequence_batch"
    assert step_input.batch.sequence_ids is not None
    assert int(np.asarray(step_input.batch.sequence_ids).shape[0]) == 1
