"""Smoke tests for the AD-PHMM-align scaffold."""


def test_public_imports() -> None:
    import ad_phmm_align

    assert ad_phmm_align.__version__
    assert ad_phmm_align.InitializationTrack.LEGACY_CURRENT.value == "legacy_current"
    assert ad_phmm_align.SourceFormat.DAG_ALIGN_LEGACY.value == "dag_align_legacy"
    assert ad_phmm_align.StateIntervalSemantics.HALF_OPEN.value == "half_open"


def test_half_open_state_interval_validation() -> None:
    import numpy as np

    from ad_phmm_align.graph import validate_state_intervals

    validate_state_intervals(
        np.array([0, 2], dtype=np.int64),
        np.array([2, 5], dtype=np.int64),
        global_state_count=5,
    )


def test_packed_window_contract() -> None:
    import numpy as np

    from ad_phmm_align.graph import PackedStateWindows

    windows = PackedStateWindows(
        left=np.array([0, 2], dtype=np.int64),
        right=np.array([2, 5], dtype=np.int64),
        offset=np.array([0, 2], dtype=np.int64),
        length=np.array([2, 3], dtype=np.int64),
    )
    windows.validate(global_state_count=5)


def test_edge_overlap_contract_checks_window_bounds() -> None:
    import numpy as np

    from ad_phmm_align.graph import EdgeWindowOverlaps, PackedStateWindows

    windows = PackedStateWindows(
        left=np.array([0, 2], dtype=np.int64),
        right=np.array([2, 5], dtype=np.int64),
        offset=np.array([0, 2], dtype=np.int64),
        length=np.array([2, 3], dtype=np.int64),
    )
    overlaps = EdgeWindowOverlaps(
        edge_ids=np.array([0], dtype=np.int64),
        src_offset=np.array([0], dtype=np.int64),
        dst_offset=np.array([1], dtype=np.int64),
        length=np.array([2], dtype=np.int64),
    )
    overlaps.validate_against_windows(
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        windows,
    )


def test_training_contract_imports() -> None:
    from ad_phmm_align.train import (
        EscapeConfig,
        HardInferenceSummary,
        InferenceSummary,
        LossWeights,
        PreparedTrainingBatch,
        ProfilingConfig,
        ProfilingResult,
        RegularizationConfig,
        SoftInferenceSummary,
        TrainingStepInput,
        TrainingStepResult,
    )

    assert LossWeights().negative_log_likelihood == 1.0
    assert EscapeConfig().multi_start.enabled is False
    assert ProfilingConfig().enabled is True
    assert ProfilingResult(wall_seconds=0.0).wall_seconds == 0.0
    assert RegularizationConfig().transition_anchor is True
    assert HardInferenceSummary
    assert InferenceSummary
    assert PreparedTrainingBatch
    assert SoftInferenceSummary
    assert TrainingStepInput
    assert TrainingStepResult


def test_soft_and_hard_path_facades_import() -> None:
    from ad_phmm_align.losses import soft_alignment_entropy, soft_pairwise_score
    from ad_phmm_align.phmm.hard_path import prepare_viterbi_input, viterbi_decode
    from ad_phmm_align.phmm.soft_path import (
        forward_log_likelihood,
        posterior_occupancy,
        soft_viterbi_score,
    )

    assert soft_alignment_entropy
    assert soft_pairwise_score
    assert prepare_viterbi_input
    assert viterbi_decode
    assert forward_log_likelihood
    assert posterior_occupancy
    assert soft_viterbi_score


def test_artifact_loader_and_initialization_loader(tmp_path) -> None:
    import json
    from pathlib import Path

    import numpy as np

    from ad_phmm_align.io import InitialPhmmArtifactLoader, TensorGraphArtifactLoader
    from ad_phmm_align.phmm import load_initial_parameters

    root = Path(tmp_path) / "tensor_graph.v1"
    (root / "graph").mkdir(parents=True)
    (root / "coordinates").mkdir(parents=True)
    (root / "source").mkdir(parents=True)
    (root / "reference").mkdir(parents=True)
    (root / "initialization" / "legacy_current").mkdir(parents=True)

    np.save(root / "graph" / "node_symbol.npy", np.array([0, 1], dtype=np.uint16))
    np.save(root / "graph" / "node_weight.npy", np.array([1.0, 1.0], dtype=np.float32))
    np.save(root / "graph" / "edge_src.npy", np.array([0], dtype=np.uint64))
    np.save(root / "graph" / "edge_dst.npy", np.array([1], dtype=np.uint64))
    np.save(root / "graph" / "edge_weight.npy", np.array([1.0], dtype=np.float32))
    np.save(root / "graph" / "topo_order.npy", np.array([0, 1], dtype=np.uint64))
    np.save(
        root / "coordinates" / "node_coordinate_left.npy",
        np.array([1, 2], dtype=np.uint64),
    )
    np.save(
        root / "coordinates" / "node_coordinate_right.npy",
        np.array([1, 2], dtype=np.uint64),
    )
    np.save(root / "coordinates" / "node_window_left.npy", np.array([0, 1], dtype=np.uint64))
    np.save(
        root / "coordinates" / "node_window_right.npy",
        np.array([2, 3], dtype=np.uint64),
    )
    np.save(root / "coordinates" / "node_state_offset.npy", np.array([0, 2], dtype=np.uint64))
    np.save(root / "coordinates" / "node_state_len.npy", np.array([2, 2], dtype=np.uint64))
    np.save(root / "coordinates" / "edge_state_src_offset.npy", np.array([1], dtype=np.uint64))
    np.save(root / "coordinates" / "edge_state_dst_offset.npy", np.array([0], dtype=np.uint64))
    np.save(root / "coordinates" / "edge_state_overlap_len.npy", np.array([1], dtype=np.uint64))
    np.save(root / "source" / "sequence_id.npy", np.array([0, 1], dtype=np.uint64))
    np.save(
        root / "source" / "sequence_name_offset.npy",
        np.array([0, 4, 8], dtype=np.uint64),
    )
    np.save(
        root / "source" / "sequence_name_bytes.npy",
        np.frombuffer(b"seqAseqB", dtype=np.uint8),
    )
    np.save(
        root / "source" / "node_source_offset.npy",
        np.array([0, 1], dtype=np.uint64),
    )
    np.save(root / "source" / "node_source_len.npy", np.array([1, 1], dtype=np.uint64))
    np.save(
        root / "source" / "source_sequence_id.npy",
        np.array([0, 1], dtype=np.uint64),
    )
    np.save(
        root / "source" / "source_position.npy",
        np.array([10, 11], dtype=np.uint64),
    )
    np.save(root / "reference" / "ref_node_ids.npy", np.array([0, 1], dtype=np.int64))
    np.save(
        root / "reference" / "ref_sequence_symbols.npy",
        np.array([0, 1], dtype=np.uint16),
    )

    manifest = {
        "format_name": "ad_phmm_tensor_graph",
        "format_version": 1,
        "source_format": "dag_align_legacy",
        "source_graph_dir": str(root / "legacy"),
        "node_count": 2,
        "edge_count": 1,
        "sequence_count": 2,
        "global_state_count": 3,
        "alphabet": ["A", "T", "C", "G"],
        "symbol_encoding": [["A", 0], ["T", 1], ["C", 2], ["G", 3]],
        "state_interval_semantics": "half_open",
        "legacy_metadata": [["state_windows_present", "true"]],
        "arrays": [
            {"name": "node_symbol", "path": "graph/node_symbol.npy", "dtype": "u16", "shape": [2], "required": True},
            {"name": "node_weight", "path": "graph/node_weight.npy", "dtype": "f32", "shape": [2], "required": True},
            {"name": "edge_src", "path": "graph/edge_src.npy", "dtype": "u64", "shape": [1], "required": True},
            {"name": "edge_dst", "path": "graph/edge_dst.npy", "dtype": "u64", "shape": [1], "required": True},
            {"name": "edge_weight", "path": "graph/edge_weight.npy", "dtype": "f32", "shape": [1], "required": True},
            {"name": "topo_order", "path": "graph/topo_order.npy", "dtype": "u64", "shape": [2], "required": True},
            {"name": "node_coordinate_left", "path": "coordinates/node_coordinate_left.npy", "dtype": "u64", "shape": [2], "required": True},
            {"name": "node_coordinate_right", "path": "coordinates/node_coordinate_right.npy", "dtype": "u64", "shape": [2], "required": True},
            {"name": "node_window_left", "path": "coordinates/node_window_left.npy", "dtype": "u64", "shape": [2], "required": True},
            {"name": "node_window_right", "path": "coordinates/node_window_right.npy", "dtype": "u64", "shape": [2], "required": True},
            {"name": "node_state_offset", "path": "coordinates/node_state_offset.npy", "dtype": "u64", "shape": [2], "required": True},
            {"name": "node_state_len", "path": "coordinates/node_state_len.npy", "dtype": "u64", "shape": [2], "required": True},
            {"name": "edge_state_src_offset", "path": "coordinates/edge_state_src_offset.npy", "dtype": "u64", "shape": [1], "required": True},
            {"name": "edge_state_dst_offset", "path": "coordinates/edge_state_dst_offset.npy", "dtype": "u64", "shape": [1], "required": True},
            {"name": "edge_state_overlap_len", "path": "coordinates/edge_state_overlap_len.npy", "dtype": "u64", "shape": [1], "required": True},
            {"name": "sequence_id", "path": "source/sequence_id.npy", "dtype": "u64", "shape": [2], "required": True},
            {"name": "sequence_name_offset", "path": "source/sequence_name_offset.npy", "dtype": "u64", "shape": [3], "required": True},
            {"name": "sequence_name_bytes", "path": "source/sequence_name_bytes.npy", "dtype": "utf8_bytes", "shape": [8], "required": True},
            {"name": "node_source_offset", "path": "source/node_source_offset.npy", "dtype": "u64", "shape": [2], "required": True},
            {"name": "node_source_len", "path": "source/node_source_len.npy", "dtype": "u64", "shape": [2], "required": True},
            {"name": "source_sequence_id", "path": "source/source_sequence_id.npy", "dtype": "u64", "shape": [2], "required": True},
            {"name": "source_position", "path": "source/source_position.npy", "dtype": "u64", "shape": [2], "required": True},
            {"name": "ref_node_ids", "path": "reference/ref_node_ids.npy", "dtype": "i64", "shape": [2], "required": True},
            {"name": "ref_sequence_symbols", "path": "reference/ref_sequence_symbols.npy", "dtype": "u16", "shape": [2], "required": True},
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    np.save(
        root / "initialization" / "legacy_current" / "match_emission.npy",
        np.log(np.full((3, 4), 0.25, dtype=np.float64)),
    )
    np.save(
        root / "initialization" / "legacy_current" / "insert_emission.npy",
        np.log(np.full((4, 4), 0.25, dtype=np.float64)),
    )
    np.save(
        root / "initialization" / "legacy_current" / "transition_logits.npy",
        np.zeros((4, 9), dtype=np.float64),
    )
    init_manifest = {
        "track": "legacy_current",
        "global_state_count": 3,
        "alphabet_size": 4,
        "transition_order": ["_mm", "_md", "_mi", "_dm", "_dd", "_di", "_im", "_id", "_ii"],
        "match_emission": {
            "name": "match_emission",
            "path": "initialization/legacy_current/match_emission.npy",
            "dtype": "f64",
            "shape": [3, 4],
            "required": True,
        },
        "insert_emission": {
            "name": "insert_emission",
            "path": "initialization/legacy_current/insert_emission.npy",
            "dtype": "f64",
            "shape": [4, 4],
            "required": True,
        },
        "transition_logits": {
            "name": "transition_logits",
            "path": "initialization/legacy_current/transition_logits.npy",
            "dtype": "f64",
            "shape": [4, 9],
            "required": True,
        },
        "metadata": [["alphabet_size", "4"]],
    }
    (root / "initialization" / "legacy_current" / "manifest.json").write_text(
        json.dumps(init_manifest), encoding="utf-8"
    )

    artifact = TensorGraphArtifactLoader(root).load_artifact()
    assert artifact.graph.node_count == 2
    assert artifact.graph.edge_count == 1
    assert artifact.graph.state_windows is not None
    assert artifact.graph.edge_overlaps is not None
    assert artifact.graph.extra is not None
    assert artifact.graph.extra["sequence_names"] == ["seqA", "seqB"]
    assert artifact.graph.extra["sequence_count"] == 2
    assert artifact.graph.extra["source_sequence_id"].tolist() == [0, 1]

    initial = InitialPhmmArtifactLoader(root).load()
    assert initial.manifest.track.value == "legacy_current"
    assert initial.parameters.graph.global_state_count == 3

    loaded = load_initial_parameters(root)
    assert loaded.metadata["transition_order"][0] == "_mm"

    from ad_phmm_align.train import EscapeConfig, MultiStartConfig, Trainer, TrainingConfig

    fit_result = Trainer(
        TrainingConfig(
            graph_path=root,
            initialization_path=root,
            output_dir=root / "out",
            max_steps=1,
            escape=EscapeConfig(
                multi_start=MultiStartConfig(
                    enabled=True,
                    replicas=2,
                    transition_logit_std=0.0,
                    emission_logit_std=0.0,
                    temperature_std=0.0,
                )
            ),
        )
    ).fit()
    assert fit_result.steps_completed == 2
    assert fit_result.final_loss is not None
    assert fit_result.metrics["optimization_ready"] is True
    assert fit_result.metrics["cpu_reference_ready"] is True
    assert fit_result.metrics["hard_decode_status"] == "viterbi_only"
    assert fit_result.checkpoint_path is not None
    assert fit_result.metadata["initialization_track"] == "legacy_current"
    assert fit_result.metadata["replica_ids"] == ("replica-0", "replica-1")


def test_transition_logit_view_and_effective_support() -> None:
    import numpy as np

    from ad_phmm_align.graph import EdgeWindowOverlaps, PackedStateWindows, TensorDag
    from ad_phmm_align.io.schema import GraphMetadata
    from ad_phmm_align.phmm import (
        PhmmParameterSet,
        TransitionLogitView,
        build_wavefront_schedule,
        intersect_effective_state_masks,
        propagate_backward_support,
        propagate_forward_support,
    )

    graph = TensorDag(
        metadata=GraphMetadata(
            graph_id="diamond",
            format_name="ad_phmm_tensor_graph",
            format_version="1",
            global_state_count=4,
        ),
        node_symbol=np.array([0, 1, 2, 3], dtype=np.uint16),
        node_weight=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        edge_src=np.array([0, 0, 1, 2], dtype=np.int64),
        edge_dst=np.array([1, 2, 3, 3], dtype=np.int64),
        edge_weight=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        topo_order=np.array([0, 1, 2, 3], dtype=np.int64),
        node_coordinate_left=np.array([1, 1, 2, 2], dtype=np.int64),
        node_coordinate_right=np.array([1, 2, 2, 2], dtype=np.int64),
        node_window_left=np.array([0, 1, 2, 1], dtype=np.int64),
        node_window_right=np.array([2, 3, 4, 4], dtype=np.int64),
        state_windows=PackedStateWindows(
            left=np.array([0, 1, 2, 1], dtype=np.int64),
            right=np.array([2, 3, 4, 4], dtype=np.int64),
            offset=np.array([0, 2, 4, 6], dtype=np.int64),
            length=np.array([2, 2, 2, 3], dtype=np.int64),
        ),
        edge_overlaps=EdgeWindowOverlaps(
            edge_ids=np.array([0, 1, 2, 3], dtype=np.int64),
            src_offset=np.array([1, 0, 0, 0], dtype=np.int64),
            dst_offset=np.array([0, 0, 0, 1], dtype=np.int64),
            length=np.array([1, 1, 1, 1], dtype=np.int64),
        ),
    )
    graph.validate()

    forward = propagate_forward_support(graph)
    backward = propagate_backward_support(graph)
    posterior = intersect_effective_state_masks(forward, backward)
    assert forward.active_count > 0
    assert backward.active_count > 0
    assert np.array_equal(
        posterior.active_global_state_ids(), np.array([0, 1, 2], dtype=np.int64)
    )

    schedule = build_wavefront_schedule(graph)
    assert schedule.level_ptr.tolist() == [0, 1, 3, 4]
    assert schedule.node_level.tolist() == [0, 1, 1, 2]

    transitions = TransitionLogitView(
        tensor=np.arange(36, dtype=np.float64).reshape(4, 9),
        order=("_mm", "_md", "_mi", "_dm", "_dd", "_di", "_im", "_id", "_ii"),
    )
    params = PhmmParameterSet(
        match_emission=np.zeros((4, 4), dtype=np.float64),
        insert_emission=np.zeros((5, 4), dtype=np.float64),
        transition_logits=transitions,
        metadata={},
    )
    assert params.transition_logits.require("_dm").shape == (4,)
    assert set(params.transition_logits.as_mapping()) == {
        "_mm",
        "_md",
        "_mi",
        "_dm",
        "_dd",
        "_di",
        "_im",
        "_id",
        "_ii",
    }
