"""Smoke tests for the CLI entry points."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _write_minimal_artifact(root: Path) -> Path:
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
        np.array([0, 1], dtype=np.uint64),
    )
    np.save(
        root / "coordinates" / "node_coordinate_right.npy",
        np.array([1, 2], dtype=np.uint64),
    )
    np.save(root / "coordinates" / "node_window_left.npy", np.array([0, 0], dtype=np.uint64))
    np.save(root / "coordinates" / "node_window_right.npy", np.array([2, 2], dtype=np.uint64))
    np.save(root / "coordinates" / "node_state_offset.npy", np.array([0, 2], dtype=np.uint64))
    np.save(root / "coordinates" / "node_state_len.npy", np.array([2, 2], dtype=np.uint64))
    np.save(
        root / "coordinates" / "edge_state_src_offset.npy",
        np.array([0], dtype=np.uint64),
    )
    np.save(
        root / "coordinates" / "edge_state_dst_offset.npy",
        np.array([0], dtype=np.uint64),
    )
    np.save(
        root / "coordinates" / "edge_state_overlap_len.npy",
        np.array([2], dtype=np.uint64),
    )
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
        "source_format": "synthetic",
        "node_count": 2,
        "edge_count": 1,
        "sequence_count": 2,
        "global_state_count": 2,
        "alphabet": ["A", "T", "C", "G"],
        "symbol_encoding": [["A", 0], ["T", 1], ["C", 2], ["G", 3]],
        "state_interval_semantics": "half_open",
        "legacy_metadata": [["smoke", "true"]],
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
            {"name": "ref_node_ids", "path": "reference/ref_node_ids.npy", "dtype": "i64", "shape": [2], "required": False},
            {"name": "ref_sequence_symbols", "path": "reference/ref_sequence_symbols.npy", "dtype": "u16", "shape": [2], "required": False},
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    np.save(
        root / "initialization" / "legacy_current" / "match_emission.npy",
        np.log(np.full((2, 4), 0.25, dtype=np.float64)),
    )
    np.save(
        root / "initialization" / "legacy_current" / "insert_emission.npy",
        np.log(np.full((3, 4), 0.25, dtype=np.float64)),
    )
    np.save(
        root / "initialization" / "legacy_current" / "transition_logits.npy",
        np.zeros((3, 9), dtype=np.float64),
    )
    init_manifest = {
        "track": "legacy_current",
        "global_state_count": 2,
        "alphabet_size": 4,
        "transition_order": ["_mm", "_md", "_mi", "_dm", "_dd", "_di", "_im", "_id", "_ii"],
        "match_emission": {
            "name": "match_emission",
            "path": "initialization/legacy_current/match_emission.npy",
            "dtype": "f64",
            "shape": [2, 4],
            "required": True,
        },
        "insert_emission": {
            "name": "insert_emission",
            "path": "initialization/legacy_current/insert_emission.npy",
            "dtype": "f64",
            "shape": [3, 4],
            "required": True,
        },
        "transition_logits": {
            "name": "transition_logits",
            "path": "initialization/legacy_current/transition_logits.npy",
            "dtype": "f64",
            "shape": [3, 9],
            "required": True,
        },
        "metadata": [["alphabet_size", "4"]],
    }
    (root / "initialization" / "legacy_current" / "manifest.json").write_text(
        json.dumps(init_manifest), encoding="utf-8"
    )
    return root


def test_cli_validate_train_and_decode_smoke(tmp_path) -> None:
    from ad_phmm_align.cli.main import main

    root = _write_minimal_artifact(Path(tmp_path) / "tensor_graph.v1")
    train_output = Path(tmp_path) / "train.json"
    decode_output = Path(tmp_path) / "decode.json"

    assert (
        main(
            [
                "validate-artifacts",
                "--graph",
                str(root),
                "--initialization",
                str(root),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "train",
                "--graph",
                str(root),
                "--initialization",
                str(root),
                "--output-dir",
                str(Path(tmp_path) / "out"),
                "--max-steps",
                "0",
                "--output",
                str(train_output),
            ]
        )
        == 0
    )
    train_payload = json.loads(train_output.read_text(encoding="utf-8"))
    assert train_payload["steps_completed"] == 0
    assert train_payload["metrics"]["cpu_reference_ready"] is True

    assert (
        main(
            [
                "decode",
                "--graph",
                str(root),
                "--initialization",
                str(root),
                "--output",
                str(decode_output),
            ]
        )
        == 0
    )
    decode_payload = json.loads(decode_output.read_text(encoding="utf-8"))
    assert decode_payload["node_assignment_count"] >= 0
    assert decode_payload["decode_status"] in {"decoded_alignment", "viterbi_only"}
