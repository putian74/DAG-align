#!/usr/bin/env python3
"""Transitional DAG-align legacy bridge.

This script is intentionally isolated under Pre-AD-prep because current
DAG-align graph saves use pickle/object arrays. It converts the first graph
slice into typed NumPy arrays plus JSON metadata for the Rust adapter.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


SYMBOL_ENCODING = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "U": 3,
    "N": 4,
    "-": 5,
}

PHMM_BASE_ORDER = ["A", "T", "C", "G"]
TRANSITION_ORDER = ["_mm", "_md", "_mi", "_dm", "_dd", "_di", "_im", "_id", "_ii"]


def _load_npz_array(npz: Any, name: str) -> np.ndarray:
    if name not in npz:
        raise ValueError(f"data.npz is missing required array: {name}")
    return np.asarray(npz[name])


def _encode_fragment(fragment: Any) -> int:
    text = str(fragment)
    if not text:
        return SYMBOL_ENCODING["N"]
    return SYMBOL_ENCODING.get(text[-1].upper(), SYMBOL_ENCODING["N"])


def _edge_arrays(edge_weight_dict: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[int, int, float]] = []
    for row in edge_weight_dict.tolist():
        if len(row) != 3:
            raise ValueError(f"edgeWeightDict row must have 3 values, got {row!r}")
        src, dst, weight = row
        rows.append((int(src), int(dst), float(weight)))
    rows.sort(key=lambda item: (item[0], item[1], item[2]))
    edge_src = np.asarray([row[0] for row in rows], dtype=np.uint64)
    edge_dst = np.asarray([row[1] for row in rows], dtype=np.uint64)
    edge_weight = np.asarray([row[2] for row in rows], dtype=np.float32)
    return edge_src, edge_dst, edge_weight


def _topological_order(
    node_count: int, edge_src: np.ndarray, edge_dst: np.ndarray
) -> np.ndarray:
    indegree = np.zeros(node_count, dtype=np.int64)
    children: list[list[int]] = [[] for _ in range(node_count)]
    for src, dst in zip(edge_src.tolist(), edge_dst.tolist()):
        if src < 0 or src >= node_count or dst < 0 or dst >= node_count:
            raise ValueError(
                f"edge endpoint out of bounds for node_count={node_count}: {src}->{dst}"
            )
        indegree[dst] += 1
        children[src].append(dst)

    ready = [node for node in range(node_count) if indegree[node] == 0]
    order: list[int] = []
    cursor = 0
    while cursor < len(ready):
        node = ready[cursor]
        cursor += 1
        order.append(node)
        for child in sorted(children[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != node_count:
        raise ValueError("legacy graph is cyclic or has inconsistent edge endpoints")
    return np.asarray(order, dtype=np.uint64)


def _csr(
    node_count: int,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = np.zeros(node_count, dtype=np.uint64)
    for src in edge_src.tolist():
        counts[src] += 1
    indptr = np.zeros(node_count + 1, dtype=np.uint64)
    indptr[1:] = np.cumsum(counts)
    cursor = indptr[:-1].copy()
    indices = np.zeros(edge_dst.shape[0], dtype=np.uint64)
    edge_id = np.zeros(edge_dst.shape[0], dtype=np.uint64)
    for eid, (src, dst) in enumerate(zip(edge_src.tolist(), edge_dst.tolist())):
        pos = int(cursor[src])
        indices[pos] = dst
        edge_id[pos] = eid
        cursor[src] += 1
    return indptr, indices, edge_id


def _csc(
    node_count: int,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return _csr(node_count, edge_dst, edge_src)


def _array_spec(name: str, rel_path: str, array: np.ndarray) -> dict[str, Any]:
    return {
        "name": name,
        "path": rel_path,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "required": True,
    }


def _optional_array(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def _first_matching_file(pattern: str) -> Path | None:
    matches = sorted(Path(path) for path in glob.glob(pattern))
    return matches[0] if matches else None


def _coerce_ref_node(value: Any) -> int:
    if value is None:
        return -1
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "x":
            return -1
        return int(text)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return int(value)


def _normalize_insert_ranges(insert_ranges: Any) -> list[list[int]]:
    normalized: list[list[int]] = []
    for item in np.asarray(insert_ranges, dtype=object).tolist():
        if isinstance(item, np.ndarray):
            item = item.tolist()
        if not isinstance(item, (list, tuple)):
            continue
        values = [int(value) for value in item]
        if len(values) >= 2:
            normalized.append([values[0], values[1]])
    return normalized


def _encode_reference_sequence(ref_seq: str) -> np.ndarray:
    return np.asarray([SYMBOL_ENCODING.get(base.upper(), SYMBOL_ENCODING["N"]) for base in ref_seq], dtype=np.uint16)


def _write_reference_artifacts(
    graph_dir: Path,
    output_dir: Path,
    diagnostics_out: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    reference_npz = _first_matching_file(str(graph_dir / "thr_*.npz"))
    if reference_npz is None:
        return [], None

    with np.load(reference_npz, allow_pickle=True) as ref_data:
        ref_seq = str(ref_data["ref_seq"])
        ref_node_ids = np.asarray(
            [_coerce_ref_node(value) for value in np.asarray(ref_data["ref_node_list"], dtype=object).tolist()],
            dtype=np.int64,
        )
        em_prob_matrix = np.asarray(ref_data["emProbMatrix"], dtype=np.float64)
        insert_ranges = _normalize_insert_ranges(ref_data["insert_range"])

    reference_out = output_dir / "reference"
    reference_out.mkdir(parents=True, exist_ok=True)
    ref_sequence_symbols = _encode_reference_sequence(ref_seq)
    np.save(reference_out / "ref_node_ids.npy", ref_node_ids)
    np.save(reference_out / "ref_sequence_symbols.npy", ref_sequence_symbols)

    array_specs = [
        _array_spec("ref_node_ids", "reference/ref_node_ids.npy", ref_node_ids),
        _array_spec("ref_sequence_symbols", "reference/ref_sequence_symbols.npy", ref_sequence_symbols),
    ]
    if insert_ranges:
        insert_region_left = np.asarray([item[0] for item in insert_ranges], dtype=np.int64)
        insert_region_right = np.asarray([item[1] for item in insert_ranges], dtype=np.int64)
        np.save(reference_out / "insert_region_left.npy", insert_region_left)
        np.save(reference_out / "insert_region_right.npy", insert_region_right)
        array_specs.extend(
            [
                _array_spec("insert_region_left", "reference/insert_region_left.npy", insert_region_left),
                _array_spec("insert_region_right", "reference/insert_region_right.npy", insert_region_right),
            ]
        )

    reference_core = {
        "source_path": str(reference_npz),
        "ref_seq": ref_seq,
        "ref_node_ids": ref_node_ids.tolist(),
        "insert_ranges": insert_ranges,
        "em_prob_matrix_shape": list(em_prob_matrix.shape),
        "alphabet_order": PHMM_BASE_ORDER,
    }
    (diagnostics_out / "reference_core.json").write_text(
        json.dumps(reference_core, indent=2),
        encoding="utf-8",
    )
    return array_specs, reference_core


def _stack_transition_arrays(parameter_dict: dict[str, Any]) -> np.ndarray:
    return np.stack(
        [np.asarray(parameter_dict[name], dtype=np.float64) for name in TRANSITION_ORDER],
        axis=1,
    )


def _write_init_track(
    output_dir: Path,
    track_name: str,
    match_emission: np.ndarray,
    insert_emission: np.ndarray,
    transition_logits: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    track_dir = output_dir / "initialization" / track_name
    track_dir.mkdir(parents=True, exist_ok=True)
    np.save(track_dir / "match_emission.npy", np.asarray(match_emission, dtype=np.float64))
    np.save(track_dir / "insert_emission.npy", np.asarray(insert_emission, dtype=np.float64))
    np.save(track_dir / "transition_logits.npy", np.asarray(transition_logits, dtype=np.float64))
    def init_array_spec(name: str, rel_path: str, shape: tuple[int, ...]) -> dict[str, Any]:
        return {
            "name": name,
            "path": rel_path,
            "dtype": "f64",
            "shape": list(shape),
            "required": True,
        }

    manifest = {
        "track": track_name,
        "global_state_count": int(match_emission.shape[0]),
        "alphabet_size": int(match_emission.shape[1]),
        "transition_order": TRANSITION_ORDER,
        "match_emission": init_array_spec(
            "match_emission",
            f"initialization/{track_name}/match_emission.npy",
            np.asarray(match_emission, dtype=np.float64).shape,
        ),
        "insert_emission": init_array_spec(
            "insert_emission",
            f"initialization/{track_name}/insert_emission.npy",
            np.asarray(insert_emission, dtype=np.float64).shape,
        ),
        "transition_logits": init_array_spec(
            "transition_logits",
            f"initialization/{track_name}/transition_logits.npy",
            np.asarray(transition_logits, dtype=np.float64).shape,
        ),
        "metadata": [[key, str(value)] for key, value in metadata.items()],
    }
    (track_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _build_reference_bootstrap_parameters(
    ref_seq: str,
    em_prob_matrix: np.ndarray,
    insert_ranges: list[list[int]],
) -> dict[str, np.ndarray]:
    me = np.log(0.5)
    md = -2.0
    mi = -5.0
    ii = np.log(0.5)
    dm = np.log(0.5)
    pi_mid = np.asarray([1.0, 1.0, 1.0], dtype=np.float64)

    n_positions = len(ref_seq) + 1
    mm_base = np.log(1.0 - np.exp(md) - np.exp(mi))
    im_base = np.log(1.0 - np.exp(ii))
    dd_base = np.log(1.0 - np.exp(dm))
    high_mi = np.logaddexp(mi, np.log(0.1))

    _mi = np.full(n_positions, mi, dtype=np.float64)
    _md = np.full(n_positions, md, dtype=np.float64)
    _mm = np.full(n_positions, mm_base, dtype=np.float64)
    for left, right in insert_ranges:
        start = max(0, left)
        stop = min(n_positions, right)
        if stop > start:
            _mi[start:stop] = high_mi

    pi_sum = float(np.sum(pi_mid))
    _mi[0] = np.log(pi_mid[1] / pi_sum)
    _md[0] = np.log(pi_mid[2] / pi_sum)
    _mm[0] = np.log(pi_mid[0] / pi_sum)
    _mm[-1] = me
    _mi[-1] = np.log(1.0 - np.exp(me))

    _ii = np.full(n_positions, ii, dtype=np.float64)
    _im = np.full(n_positions, im_base, dtype=np.float64)
    _id = np.full(n_positions, -np.inf, dtype=np.float64)
    _dm = np.full(n_positions, dm, dtype=np.float64)
    _dd = np.full(n_positions, dd_base, dtype=np.float64)
    _di = np.full(n_positions, -np.inf, dtype=np.float64)
    _dm[0] = -np.inf
    _dd[0] = -np.inf
    _dd[-1] = -np.inf
    _dm[-1] = 0.0

    match_emission = np.log(np.asarray(em_prob_matrix, dtype=np.float64).T + 1e-16)
    insert_emission = np.full((match_emission.shape[0] + 1, match_emission.shape[1]), np.log(0.25), dtype=np.float64)
    return {
        "_mm": _mm,
        "_md": _md,
        "_mi": _mi,
        "_dm": _dm,
        "_dd": _dd,
        "_di": _di,
        "_im": _im,
        "_id": _id,
        "_ii": _ii,
        "match_emission": match_emission,
        "insert_emission": insert_emission,
    }


def _write_initialization_artifacts(
    graph_dir: Path,
    output_dir: Path,
    diagnostics_out: Path,
    reference_core: dict[str, Any] | None,
) -> list[str]:
    exported_tracks: list[str] = []
    init_file = _first_matching_file(str(graph_dir / "ini" / "init_*.npy"))
    if init_file is not None:
        parameter_dict = np.load(init_file, allow_pickle=True).item()
        manifest = _write_init_track(
            output_dir=output_dir,
            track_name="legacy_current",
            match_emission=np.asarray(parameter_dict["match_emission"], dtype=np.float64),
            insert_emission=np.asarray(parameter_dict["insert_emission"], dtype=np.float64),
            transition_logits=_stack_transition_arrays(parameter_dict),
            metadata={"source_path": str(init_file), "parameter_space": "legacy_log_probability"},
        )
        (diagnostics_out / "init_legacy_current.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        exported_tracks.append("legacy_current")

    if reference_core is not None:
        reference_npz = Path(reference_core["source_path"])
        with np.load(reference_npz, allow_pickle=True) as ref_data:
            em_prob_matrix = np.asarray(ref_data["emProbMatrix"], dtype=np.float64)
        bootstrap = _build_reference_bootstrap_parameters(
            ref_seq=reference_core["ref_seq"],
            em_prob_matrix=em_prob_matrix,
            insert_ranges=reference_core["insert_ranges"],
        )
        manifest = _write_init_track(
            output_dir=output_dir,
            track_name="reference_msa",
            match_emission=bootstrap["match_emission"],
            insert_emission=bootstrap["insert_emission"],
            transition_logits=_stack_transition_arrays(bootstrap),
            metadata={
                "source_path": str(reference_npz),
                "parameter_space": "legacy_log_probability",
                "bootstrap_source": "reference_artifact",
            },
        )
        (diagnostics_out / "init_reference_msa.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        exported_tracks.append("reference_msa")

    return exported_tracks


def _sequence_tables(v_id: np.ndarray | None) -> dict[str, tuple[str, np.ndarray]]:
    if v_id is None:
        return {}

    names: list[bytes] = []
    for row in np.asarray(v_id, dtype=object).tolist():
        if isinstance(row, (list, tuple)) and row:
            names.append(str(row[0]).encode("utf-8"))
        else:
            names.append(str(row).encode("utf-8"))

    offsets = np.zeros(len(names) + 1, dtype=np.uint64)
    payload = bytearray()
    for index, name in enumerate(names):
        payload.extend(name)
        offsets[index + 1] = len(payload)

    return {
        "sequence_id": ("source/sequence_id.npy", np.arange(len(names), dtype=np.uint64)),
        "sequence_name_offset": ("source/sequence_name_offset.npy", offsets),
        "sequence_name_bytes": (
            "source/sequence_name_bytes.npy",
            np.frombuffer(bytes(payload), dtype=np.uint8),
        ),
    }


def _flatten_osm_sources(osm: np.ndarray, node_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(osm) != node_count:
        raise ValueError("osm.npy length must equal node_count")
    offsets = np.zeros(node_count, dtype=np.uint64)
    lengths = np.zeros(node_count, dtype=np.uint64)
    flat: list[int] = []
    for node_id, record in enumerate(np.asarray(osm, dtype=object).tolist()):
        offsets[node_id] = len(flat)
        values = np.asarray(record, dtype=np.uint64).reshape(-1)
        lengths[node_id] = values.size
        flat.extend(int(value) for value in values.tolist())
    return (
        np.asarray(flat, dtype=np.uint64),
        offsets,
        lengths,
    )


def _flatten_onm_sources(
    onm: np.ndarray,
    onm_index: np.ndarray,
    node_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(onm_index) != node_count:
        raise ValueError("onm_index.npy length must equal node_count")
    flat = np.asarray([int(value) for value in np.asarray(onm, dtype=object).tolist()], dtype=np.uint64)
    offsets = np.asarray(onm_index[:, 0], dtype=np.uint64)
    lengths = np.asarray(onm_index[:, 1] - onm_index[:, 0], dtype=np.uint64)
    return flat, offsets, lengths


def _try_load_graph_bits(graph_dir: Path) -> dict[str, int] | None:
    graph_pkl = graph_dir / "graph.pkl"
    if not graph_pkl.exists():
        return None
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        with graph_pkl.open("rb") as handle:
            graph = pickle.load(handle)
    except Exception:
        return None

    fields = {}
    for name in ("firstBitofOSM", "allBitofOSM", "firstBitofONM", "allBitofONM"):
        value = getattr(graph, name, None)
        if isinstance(value, (int, np.integer)):
            fields[name] = int(value)
    return fields or None


def _decode_packed_records(
    packed: np.ndarray,
    first_bit: int,
    all_bit: int,
) -> tuple[np.ndarray, np.ndarray]:
    shift = all_bit - first_bit
    mask = (1 << shift) - 1
    sequence_ids = np.right_shift(packed, shift).astype(np.uint64, copy=False)
    positions = np.bitwise_and(packed, mask).astype(np.uint64, copy=False)
    return sequence_ids, positions


def convert(graph_dir: Path, output_dir: Path) -> None:
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    data_path = graph_dir / "data.npz"
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    graph_out = output_dir / "graph"
    source_out = output_dir / "source"
    reference_out = output_dir / "reference"
    diagnostics_out = output_dir / "diagnostics"
    graph_out.mkdir(parents=True, exist_ok=True)
    source_out.mkdir(parents=True, exist_ok=True)
    reference_out.mkdir(parents=True, exist_ok=True)
    diagnostics_out.mkdir(parents=True, exist_ok=True)

    with np.load(data_path, allow_pickle=True) as data:
        edge_weight_dict = _load_npz_array(data, "edgeWeightDict")
        fragments = _load_npz_array(data, "fragments")
        weights = _load_npz_array(data, "weights")
        start_nodes = _load_npz_array(data, "startNodeSet").astype(np.uint64)
        end_nodes = _load_npz_array(data, "endNodeSet").astype(np.uint64)

    node_count = int(weights.shape[0])
    edge_src, edge_dst, edge_weight = _edge_arrays(edge_weight_dict)
    edge_count = int(edge_src.shape[0])
    node_symbol = np.asarray([_encode_fragment(value) for value in fragments], dtype=np.uint16)
    node_weight = weights.astype(np.float32, copy=False)
    node_flags = np.zeros(node_count, dtype=np.uint32)
    for node in start_nodes.tolist():
        if 0 <= int(node) < node_count:
            node_flags[int(node)] |= 1
    for node in end_nodes.tolist():
        if 0 <= int(node) < node_count:
            node_flags[int(node)] |= 2

    topo_order = _topological_order(node_count, edge_src, edge_dst)
    csr_indptr, csr_indices, csr_edge_id = _csr(node_count, edge_src, edge_dst)
    csc_indptr, csc_indices, csc_edge_id = _csc(node_count, edge_src, edge_dst)

    arrays = {
        "node_symbol": ("graph/node_symbol.npy", node_symbol),
        "node_weight": ("graph/node_weight.npy", node_weight),
        "node_flags": ("graph/node_flags.npy", node_flags),
        "edge_src": ("graph/edge_src.npy", edge_src),
        "edge_dst": ("graph/edge_dst.npy", edge_dst),
        "edge_weight": ("graph/edge_weight.npy", edge_weight),
        "csr_indptr": ("graph/csr_indptr.npy", csr_indptr),
        "csr_indices": ("graph/csr_indices.npy", csr_indices),
        "csr_edge_id": ("graph/csr_edge_id.npy", csr_edge_id),
        "csc_indptr": ("graph/csc_indptr.npy", csc_indptr),
        "csc_indices": ("graph/csc_indices.npy", csc_indices),
        "csc_edge_id": ("graph/csc_edge_id.npy", csc_edge_id),
        "topo_order": ("graph/topo_order.npy", topo_order),
    }
    for rel_path, array in arrays.values():
        np.save(output_dir / rel_path, array)

    sequence_count = 0
    v_id_path = graph_dir / "v_id.npy"
    v_id = _optional_array(v_id_path)
    if v_id is not None:
        sequence_count = int(v_id.shape[0])

    source_arrays: dict[str, tuple[str, np.ndarray]] = {}
    source_diagnostics: list[dict[str, Any]] = []
    graph_bits = _try_load_graph_bits(graph_dir)

    source_arrays.update(_sequence_tables(v_id))

    osm = _optional_array(graph_dir / "osm.npy")
    onm = _optional_array(graph_dir / "onm.npy")
    onm_index = _optional_array(graph_dir / "onm_index.npy")
    source_decode_status = "missing"
    source_packed = None
    if osm is not None:
        source_packed, node_source_offset, node_source_len = _flatten_osm_sources(
            osm, node_count
        )
        source_decode_status = "raw_osm_packed"
    elif onm is not None and onm_index is not None:
        source_packed, node_source_offset, node_source_len = _flatten_onm_sources(
            onm, np.asarray(onm_index), node_count
        )
        source_decode_status = "raw_onm_traceability"
    else:
        node_source_offset = np.zeros(node_count, dtype=np.uint64)
        node_source_len = np.zeros(node_count, dtype=np.uint64)
        source_diagnostics.append(
            {
                "severity": "warning",
                "code": "source_records_absent",
                "message": "legacy graph did not provide osm/onm source provenance arrays",
            }
        )

    source_arrays["node_source_offset"] = (
        "source/node_source_offset.npy",
        node_source_offset,
    )
    source_arrays["node_source_len"] = ("source/node_source_len.npy", node_source_len)
    if source_packed is not None:
        source_arrays["source_packed"] = ("source/source_packed.npy", source_packed)
        if source_decode_status == "raw_osm_packed" and graph_bits is not None:
            if "firstBitofOSM" in graph_bits and "allBitofOSM" in graph_bits:
                decoded_sequence_id, decoded_position = _decode_packed_records(
                    source_packed,
                    graph_bits["firstBitofOSM"],
                    graph_bits["allBitofOSM"],
                )
                source_arrays["source_sequence_id"] = (
                    "source/source_sequence_id.npy",
                    decoded_sequence_id,
                )
                source_arrays["source_position"] = (
                    "source/source_position.npy",
                    decoded_position,
                )
                source_decode_status = "decoded_osm"
        elif source_decode_status == "raw_osm_packed":
            source_diagnostics.append(
                {
                    "severity": "warning",
                    "code": "source_decode_unavailable",
                    "message": "graph.pkl could not be decoded for OSM bit widths; raw packed source records were preserved",
                }
            )

    reference_specs, reference_core = _write_reference_artifacts(graph_dir, output_dir, diagnostics_out)
    init_tracks = _write_initialization_artifacts(graph_dir, output_dir, diagnostics_out, reference_core)

    manifest = {
        "format_name": "ad_phmm_tensor_graph",
        "format_version": 1,
        "source_format": "dag_align_legacy",
        "source_graph_dir": str(graph_dir),
        "node_count": node_count,
        "edge_count": edge_count,
        "sequence_count": sequence_count,
        "alphabet": ["A", "C", "G", "T", "N", "-"],
        "symbol_encoding": SYMBOL_ENCODING,
        "state_interval_semantics": "half_open",
        "arrays": [
            _array_spec(name, rel_path, array) for name, (rel_path, array) in arrays.items()
        ]
        + [
            _array_spec(name, rel_path, array)
            for name, (rel_path, array) in source_arrays.items()
        ]
        + reference_specs,
        "legacy": {
            "edge_sort": "src_dst_weight",
            "fragment_symbol_policy": "last_character",
            "state_windows_present": False,
            "source_decode_status": source_decode_status,
            "initialization_tracks": init_tracks,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for rel_path, array in source_arrays.values():
        np.save(output_dir / rel_path, array)

    graph_core = {
        "node_symbol": node_symbol.tolist(),
        "node_weight": node_weight.tolist(),
        "node_flags": node_flags.tolist(),
        "edge_src": edge_src.tolist(),
        "edge_dst": edge_dst.tolist(),
        "edge_weight": edge_weight.tolist(),
        "topo_order": topo_order.tolist(),
        "source_format": manifest["source_format"],
        "source_graph_dir": manifest["source_graph_dir"],
        "sequence_count": sequence_count,
        "alphabet": manifest["alphabet"],
        "symbol_encoding": manifest["symbol_encoding"],
        "state_interval_semantics": manifest["state_interval_semantics"],
        "arrays": manifest["arrays"],
        "legacy": manifest["legacy"],
        "diagnostics": [],
        "profiling": {
            "wall_seconds": time.perf_counter() - start_wall,
            "cpu_seconds": time.process_time() - start_cpu,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    }
    (diagnostics_out / "graph_core.json").write_text(
        json.dumps(graph_core, indent=2), encoding="utf-8"
    )
    (diagnostics_out / "source_core.json").write_text(
        json.dumps(
            {
                "sequence_count": sequence_count,
                "source_decode_status": source_decode_status,
                "source_record_count": int(source_packed.shape[0]) if source_packed is not None else 0,
                "diagnostics": source_diagnostics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    convert(args.graph_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
