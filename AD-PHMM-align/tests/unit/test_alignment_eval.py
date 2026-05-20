"""Tests for hard alignment decode and metrics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from pathlib import Path


def _fake_runtime():
    class FakeValSparseMSA:
        def __init__(self, meta, val_positions=None):
            self.meta = dict(meta)
            self.val_positions = list(val_positions or [])
            self.ref_vector = np.asarray(meta["ref_vector"], dtype=np.uint8)
            self.nrow = int(meta["nrow"])
            self.ncol = int(meta["ncol"])
            self.axis = int(meta["axis"])

        def get_sparse_col(self, col_idx):
            return self.val_positions[col_idx]

    digital_to_iupac = [
        "-",
        "A",
        "T",
        "C",
        "G",
        "R",
        "Y",
        "M",
        "K",
        "S",
        "W",
        "H",
        "B",
        "V",
        "D",
        "N",
    ]
    return SimpleNamespace(
        module=SimpleNamespace(
            ValSparseMSA=FakeValSparseMSA,
            IUPAC_TO_DIGITAL={char: idx for idx, char in enumerate(digital_to_iupac)},
            DIGITAL_TO_IUPAC=digital_to_iupac,
            ALPHABET_SIZE=len(digital_to_iupac),
        ),
        msa_utils=SimpleNamespace(
            IUPAC_DEGENERATE_MAP={
                "-": ([0], [12]),
                "A": ([1], [12]),
                "T": ([2], [12]),
                "C": ([3], [12]),
                "G": ([4], [12]),
                "N": ([1, 2, 3, 4], [3, 3, 3, 3]),
            }
        ),
    )


def _tiny_graph():
    from ad_phmm_align.graph import EdgeWindowOverlaps, PackedStateWindows, TensorDag
    from ad_phmm_align.io.schema import GraphMetadata

    return TensorDag(
        metadata=GraphMetadata(
            graph_id="decode-graph",
            format_name="ad_phmm_tensor_graph",
            format_version="1",
            global_state_count=2,
        ),
        node_symbol=np.array([0, 1, 2], dtype=np.uint16),
        node_weight=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        edge_src=np.array([0, 1], dtype=np.uint64),
        edge_dst=np.array([1, 2], dtype=np.uint64),
        edge_weight=np.array([1.0, 1.0], dtype=np.float32),
        topo_order=np.array([0, 1, 2], dtype=np.uint64),
        node_coordinate_left=np.array([0, 0, 1], dtype=np.uint64),
        node_coordinate_right=np.array([1, 1, 2], dtype=np.uint64),
        node_window_left=np.array([0, 0, 1], dtype=np.uint64),
        node_window_right=np.array([1, 1, 2], dtype=np.uint64),
        state_windows=PackedStateWindows(
            left=np.array([0, 0, 1], dtype=np.int64),
            right=np.array([1, 1, 2], dtype=np.int64),
            offset=np.array([0, 1, 2], dtype=np.int64),
            length=np.array([1, 1, 1], dtype=np.int64),
        ),
        edge_overlaps=EdgeWindowOverlaps(
            edge_ids=np.array([0, 1], dtype=np.int64),
            src_offset=np.array([0, 0], dtype=np.int64),
            dst_offset=np.array([0, 0], dtype=np.int64),
            length=np.array([1, 1], dtype=np.int64),
        ),
        extra={
            "symbol_encoding": {"A": 0, "T": 1, "C": 2, "G": 3},
            "sequence_id": np.array([0, 1, 2], dtype=np.uint64),
            "sequence_names": ["seq0", "seq1", "seq2"],
            "node_source_offset": np.array([0, 2, 3], dtype=np.uint64),
            "node_source_len": np.array([2, 1, 2], dtype=np.uint64),
            "source_sequence_id": np.array([0, 1, 1, 0, 2], dtype=np.uint64),
            "source_position": np.array([10, 10, 11, 12, 12], dtype=np.uint64),
            "ref_sequence_symbols": np.array([0, 2], dtype=np.uint16),
        },
    )


def _viterbi_graph():
    graph = _tiny_graph()
    graph.extra["node_source_offset"] = np.array([0, 3, 6], dtype=np.uint64)
    graph.extra["node_source_len"] = np.array([3, 3, 3], dtype=np.uint64)
    graph.extra["source_sequence_id"] = np.array(
        [0, 1, 2, 0, 1, 2, 0, 1, 2],
        dtype=np.uint64,
    )
    graph.extra["source_position"] = np.array(
        [10, 10, 10, 11, 11, 11, 12, 12, 12],
        dtype=np.uint64,
    )
    graph.extra["ref_sequence_symbols"] = np.array([0, 1, 2], dtype=np.uint16)
    return graph


def _tiny_parameters():
    from ad_phmm_align.phmm import PhmmParameterSet, TransitionLogitView

    return PhmmParameterSet(
        match_emission=np.log(
            np.array(
                [
                    [0.7, 0.1, 0.1, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                ],
                dtype=np.float64,
            )
        ),
        insert_emission=np.log(np.full((3, 4), 0.25, dtype=np.float64)),
        transition_logits=TransitionLogitView(
            tensor=np.zeros((3, 9), dtype=np.float64),
            order=("_mm", "_md", "_mi", "_dm", "_dd", "_di", "_im", "_id", "_ii"),
        ),
        metadata={"alphabet_size": 4},
    )


def test_decode_alignment_builds_valsparse_columns(monkeypatch) -> None:
    from ad_phmm_align.eval import AlignmentColumnKey, NodeColumnAssignment, decode_alignment
    import ad_phmm_align.eval.decode as decode_module

    monkeypatch.setattr(
        decode_module,
        "load_msa_representation_runtime",
        lambda: _fake_runtime(),
    )
    graph = _tiny_graph()
    decoded = decode_alignment(
        graph,
        viterbi_result=object(),
        node_assignments=[
            NodeColumnAssignment(
                node_id=0,
                column=AlignmentColumnKey(channel="match", state_index=0),
            ),
            NodeColumnAssignment(
                node_id=1,
                column=AlignmentColumnKey(channel="insert", state_index=1, slot=0),
            ),
            NodeColumnAssignment(
                node_id=2,
                column=AlignmentColumnKey(channel="match", state_index=1),
            ),
        ],
    )

    assert [column.channel for column in decoded.column_keys] == ["match", "insert", "match"]
    assert [column.state_index for column in decoded.column_keys] == [0, 1, 1]
    assert decoded.sequence_names == ("seq0", "seq1", "seq2")
    assert decoded.msa.ref_vector.tolist() == [1, 0, 3]
    assert decoded.msa.val_positions[0][0][0] == 0
    assert decoded.msa.val_positions[0][0][1].tolist() == [2]
    assert decoded.msa.val_positions[1][0][0] == 2
    assert decoded.msa.val_positions[1][0][1].tolist() == [1]
    assert decoded.msa.val_positions[2][0][0] == 0
    assert decoded.msa.val_positions[2][0][1].tolist() == [1]


def test_decode_alignment_uses_viterbi_node_assignments(monkeypatch) -> None:
    from ad_phmm_align.eval import decode_alignment
    from ad_phmm_align.phmm import viterbi_decode
    import ad_phmm_align.eval.decode as decode_module

    monkeypatch.setattr(
        decode_module,
        "load_msa_representation_runtime",
        lambda: _fake_runtime(),
    )
    graph = _viterbi_graph()
    viterbi = viterbi_decode(graph, _tiny_parameters())

    decoded = decode_alignment(graph, viterbi)

    assert viterbi.node_assignments is not None
    assert len(viterbi.node_assignments) >= 3
    assert decoded.metadata["decoded_node_count"] == len(viterbi.node_assignments)
    assert decoded.msa.nrow == 3
    assert decoded.msa.ncol >= 1


def test_trainer_run_inference_paths_reports_soft_and_hard(monkeypatch, tmp_path) -> None:
    from ad_phmm_align.train import Trainer, TrainingConfig
    import ad_phmm_align.eval.decode as decode_module
    import ad_phmm_align.eval.alignment_metrics as metrics_module

    runtime = _fake_runtime()
    monkeypatch.setattr(
        decode_module,
        "load_msa_representation_runtime",
        lambda: runtime,
    )
    monkeypatch.setattr(
        metrics_module,
        "load_msa_representation_runtime",
        lambda: runtime,
    )
    trainer = Trainer(
        TrainingConfig(
            graph_path=Path(tmp_path) / "graph",
            initialization_path=Path(tmp_path) / "init",
            output_dir=Path(tmp_path) / "out",
        )
    )

    summary = trainer.run_inference_paths(_viterbi_graph(), _tiny_parameters(), temperature=1.0)

    assert summary.soft.metrics["log_likelihood"] == pytest.approx(
        float(summary.soft.forward_result.log_likelihood)
    )
    assert summary.hard.decode_status == "decoded_alignment"
    assert summary.hard.alignment_metrics is not None
    assert summary.hard.alignment_metrics.alignment_length >= 1
    assert summary.hard.metrics["hard_alignment_length"] >= 1


def test_summarize_alignment_metrics_matches_column_counts(monkeypatch) -> None:
    from ad_phmm_align.eval import summarize_alignment_metrics
    import ad_phmm_align.eval.alignment_metrics as metrics_module
    import ad_phmm_align.eval.decode as decode_module

    runtime = _fake_runtime()
    monkeypatch.setattr(
        decode_module,
        "load_msa_representation_runtime",
        lambda: runtime,
    )
    monkeypatch.setattr(
        metrics_module,
        "load_msa_representation_runtime",
        lambda: runtime,
    )
    graph = _tiny_graph()
    decoded = decode_module.decode_alignment(
        graph,
        viterbi_result=object(),
        node_assignments=[
            decode_module.NodeColumnAssignment(
                node_id=0,
                column=decode_module.AlignmentColumnKey(channel="match", state_index=0),
            ),
            decode_module.NodeColumnAssignment(
                node_id=1,
                column=decode_module.AlignmentColumnKey(
                    channel="insert",
                    state_index=1,
                    slot=0,
                ),
            ),
            decode_module.NodeColumnAssignment(
                node_id=2,
                column=decode_module.AlignmentColumnKey(channel="match", state_index=1),
            ),
        ],
    )

    metrics = summarize_alignment_metrics(decoded, log_likelihood=-5.0)
    assert metrics.log_likelihood == -5.0
    assert metrics.alignment_length == 3
    assert metrics.sequence_count == 3
    assert metrics.core_column_count == 3
    expected_entropy = -(2.0 / 3.0) * np.log(2.0 / 3.0) - (1.0 / 3.0) * np.log(1.0 / 3.0)
    assert metrics.pairwise_score == pytest.approx(2.0 / 9.0)
    assert metrics.entropy == pytest.approx(3.0 * expected_entropy)
    assert metrics.core_entropy == pytest.approx(expected_entropy)
