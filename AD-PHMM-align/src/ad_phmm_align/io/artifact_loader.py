"""Typed loaders for Pre-AD-prep tensor graph and initialization artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

import numpy as np

from ad_phmm_align.exceptions import ArtifactValidationError
from ad_phmm_align.graph.coordinates import EdgeWindowOverlaps, PackedStateWindows
from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.io.schema import (
    ArraySpec,
    GraphMetadata,
    InitialPhmmManifest,
    InitialPhmmParameters,
    InitializationTrack,
    SourceFormat,
    StateIntervalSemantics,
    TensorGraphManifest,
)

if TYPE_CHECKING:
    from ad_phmm_align.train.dataset import TensorGraphArtifact

_DTYPE_BY_NAME = {
    "u8": np.dtype(np.uint8),
    "u16": np.dtype(np.uint16),
    "u32": np.dtype(np.uint32),
    "u64": np.dtype(np.uint64),
    "i64": np.dtype(np.int64),
    "f32": np.dtype(np.float32),
    "f64": np.dtype(np.float64),
    "utf8_bytes": np.dtype(np.uint8),
}

_REQUIRED_GRAPH_ARRAYS = (
    "node_symbol",
    "node_weight",
    "edge_src",
    "edge_dst",
    "edge_weight",
    "topo_order",
    "node_coordinate_left",
    "node_coordinate_right",
    "node_window_left",
    "node_window_right",
)

_OPTIONAL_GRAPH_EXTRA_ARRAYS = (
    "sequence_id",
    "sequence_name_offset",
    "sequence_name_bytes",
    "source_packed",
    "source_sequence_id",
    "source_position",
    "node_source_offset",
    "node_source_len",
    "ref_node_ids",
    "ref_sequence_symbols",
    "insert_region_left",
    "insert_region_right",
)


@dataclass(frozen=True)
class LoadedInitialPhmmArtifact:
    """Initialization manifest together with loaded tensor arrays."""

    manifest: InitialPhmmManifest
    parameters: InitialPhmmParameters


def _as_path(value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value)


def _parse_array_spec(payload: Mapping[str, Any]) -> ArraySpec:
    return ArraySpec(
        name=str(payload["name"]),
        path=Path(payload["path"]),
        dtype=str(payload["dtype"]),
        shape=tuple(int(dim) for dim in payload["shape"]),
        required=bool(payload.get("required", True)),
        description=payload.get("description"),
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _expected_dtype(dtype_name: str) -> np.dtype:
    try:
        return _DTYPE_BY_NAME[dtype_name]
    except KeyError as exc:
        raise ArtifactValidationError(f"unsupported manifest dtype: {dtype_name}") from exc


def _load_array(root: Path, spec: ArraySpec) -> np.ndarray:
    path = root / spec.path
    if not path.exists():
        if spec.required:
            raise ArtifactValidationError(f"required array is missing: {path}")
        raise FileNotFoundError(path)
    array = np.load(path, allow_pickle=False)
    expected_dtype = _expected_dtype(spec.dtype)
    if array.dtype != expected_dtype:
        raise ArtifactValidationError(
            f"array {spec.name} has dtype {array.dtype}, expected {expected_dtype}"
        )
    if tuple(int(dim) for dim in array.shape) != tuple(int(dim) for dim in spec.shape):
        raise ArtifactValidationError(
            f"array {spec.name} has shape {tuple(array.shape)}, expected {tuple(spec.shape)}"
        )
    return array


def _load_optional_array(root: Path, spec: Optional[ArraySpec]) -> Optional[np.ndarray]:
    if spec is None:
        return None
    try:
        return _load_array(root, spec)
    except FileNotFoundError:
        return None


def _decode_utf8_table(offsets: np.ndarray, payload: np.ndarray) -> list[str]:
    byte_buffer = payload.tobytes()
    names: list[str] = []
    for index in range(int(offsets.shape[0]) - 1):
        start = int(offsets[index])
        end = int(offsets[index + 1])
        if start < 0 or end < start or end > len(byte_buffer):
            raise ArtifactValidationError("sequence_name offsets exceed sequence_name_bytes")
        names.append(byte_buffer[start:end].decode("utf-8"))
    return names


class TensorGraphArtifactLoader:
    """Load typed `tensor_graph.v1` artifacts produced by Pre-AD-prep."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def load_manifest(self) -> TensorGraphManifest:
        payload = _load_json(self.manifest_path)
        arrays = {
            spec.name: spec
            for spec in (_parse_array_spec(item) for item in payload.get("arrays", []))
        }
        metadata = GraphMetadata(
            graph_id=self.root.name,
            format_name=str(payload["format_name"]),
            format_version=str(payload["format_version"]),
            source_path=_as_path(payload.get("source_graph_dir")),
            global_state_count=payload.get("global_state_count"),
            alphabet=[str(symbol) for symbol in payload.get("alphabet", [])],
            state_interval_semantics=StateIntervalSemantics(
                payload.get("state_interval_semantics", StateIntervalSemantics.HALF_OPEN.value)
            ),
            extra={"legacy_metadata": dict(payload.get("legacy_metadata", []))},
        )
        return TensorGraphManifest(
            metadata=metadata,
            node_count=int(payload["node_count"]),
            edge_count=int(payload["edge_count"]),
            sequence_count=int(payload.get("sequence_count", 0)),
            arrays=arrays,
            source_format=SourceFormat(payload["source_format"]),
            symbol_encoding={str(key): int(value) for key, value in payload.get("symbol_encoding", [])},
            extra={"legacy_metadata": dict(payload.get("legacy_metadata", []))},
        )

    def load_graph(self) -> TensorDag:
        manifest = self.load_manifest()
        missing = [name for name in _REQUIRED_GRAPH_ARRAYS if name not in manifest.arrays]
        if missing:
            raise ArtifactValidationError(
                f"tensor graph manifest is missing required arrays: {', '.join(missing)}"
            )

        node_coordinate_left = _load_array(self.root, manifest.arrays["node_coordinate_left"])
        node_coordinate_right = _load_array(
            self.root, manifest.arrays["node_coordinate_right"]
        )
        node_window_left = _load_array(self.root, manifest.arrays["node_window_left"])
        node_window_right = _load_array(self.root, manifest.arrays["node_window_right"])
        windows = None
        if "node_state_offset" in manifest.arrays and "node_state_len" in manifest.arrays:
            node_state_offset = _load_array(self.root, manifest.arrays["node_state_offset"])
            node_state_len = _load_array(self.root, manifest.arrays["node_state_len"])
            if not np.array_equal(node_state_len, node_window_right - node_window_left):
                raise ArtifactValidationError(
                    "packed state-window lengths do not match node_window_right - node_window_left; "
                    "export explicit packed window bounds or keep node_window_left/right aligned "
                    "with the packed state-window contract"
                )
            windows = PackedStateWindows(
                left=node_window_left,
                right=node_window_right,
                offset=node_state_offset,
                length=node_state_len,
            )

        overlaps = None
        if (
            "edge_state_src_offset" in manifest.arrays
            and "edge_state_dst_offset" in manifest.arrays
            and "edge_state_overlap_len" in manifest.arrays
        ):
            overlaps = EdgeWindowOverlaps(
                edge_ids=np.arange(manifest.edge_count, dtype=np.int64),
                src_offset=_load_array(self.root, manifest.arrays["edge_state_src_offset"]),
                dst_offset=_load_array(self.root, manifest.arrays["edge_state_dst_offset"]),
                length=_load_array(self.root, manifest.arrays["edge_state_overlap_len"]),
            )

        extra_arrays = {
            name: array
            for name in _OPTIONAL_GRAPH_EXTRA_ARRAYS
            if (array := _load_optional_array(self.root, manifest.arrays.get(name))) is not None
        }
        if (
            "sequence_name_offset" in extra_arrays
            and "sequence_name_bytes" in extra_arrays
        ):
            extra_arrays["sequence_names"] = _decode_utf8_table(
                extra_arrays["sequence_name_offset"],
                extra_arrays["sequence_name_bytes"],
            )
        extra_arrays["sequence_count"] = int(manifest.sequence_count)

        graph = TensorDag(
            metadata=manifest.metadata,
            node_symbol=_load_array(self.root, manifest.arrays["node_symbol"]),
            node_weight=_load_array(self.root, manifest.arrays["node_weight"]),
            edge_src=_load_array(self.root, manifest.arrays["edge_src"]),
            edge_dst=_load_array(self.root, manifest.arrays["edge_dst"]),
            edge_weight=_load_array(self.root, manifest.arrays["edge_weight"]),
            topo_order=_load_array(self.root, manifest.arrays["topo_order"]),
            node_coordinate_left=node_coordinate_left,
            node_coordinate_right=node_coordinate_right,
            node_window_left=node_window_left,
            node_window_right=node_window_right,
            state_windows=windows,
            edge_overlaps=overlaps,
            node_flags=_load_optional_array(self.root, manifest.arrays.get("node_flags")),
            csr_indptr=_load_optional_array(self.root, manifest.arrays.get("csr_indptr")),
            csr_indices=_load_optional_array(self.root, manifest.arrays.get("csr_indices")),
            csc_indptr=_load_optional_array(self.root, manifest.arrays.get("csc_indptr")),
            csc_indices=_load_optional_array(self.root, manifest.arrays.get("csc_indices")),
            extra={
                "manifest_arrays": tuple(manifest.arrays.keys()),
                "symbol_encoding": dict(manifest.symbol_encoding),
                "source_format": manifest.source_format.value,
                **extra_arrays,
            },
        )
        graph.validate()
        return graph

    def load_artifact(self) -> "TensorGraphArtifact":
        from ad_phmm_align.train.dataset import TensorGraphArtifact

        manifest = self.load_manifest()
        graph = self.load_graph()
        return TensorGraphArtifact(root=self.root, manifest=manifest, graph=graph)


class InitialPhmmArtifactLoader:
    """Load common PHMM initialization manifests and tensor arrays."""

    def __init__(
        self,
        path: Path | str,
        track: Optional[InitializationTrack] = None,
    ) -> None:
        self.path = Path(path)
        self.track = track

    def resolve_manifest_path(self) -> Path:
        if self.path.is_file():
            return self.path
        if self.path.name in {track.value for track in InitializationTrack} and (
            self.path / "manifest.json"
        ).exists():
            return self.path / "manifest.json"
        if self.track is not None:
            candidate = self.path / "initialization" / self.track.value / "manifest.json"
            if candidate.exists():
                return candidate
        candidate = (
            self.path / "initialization" / InitializationTrack.LEGACY_CURRENT.value / "manifest.json"
        )
        if candidate.exists():
            return candidate
        if (self.path / "manifest.json").exists():
            return self.path / "manifest.json"
        raise ArtifactValidationError(
            f"could not resolve initialization manifest from {self.path}"
        )

    def _graph_root_from_manifest(self, manifest_path: Path) -> Path:
        if manifest_path.parent.name in {track.value for track in InitializationTrack}:
            return manifest_path.parent.parent.parent
        return manifest_path.parent

    def load_manifest(self) -> InitialPhmmManifest:
        manifest_path = self.resolve_manifest_path()
        payload = _load_json(manifest_path)
        return InitialPhmmManifest(
            track=InitializationTrack(payload["track"]),
            global_state_count=int(payload["global_state_count"]),
            alphabet_size=int(payload["alphabet_size"]),
            transition_order=tuple(str(name) for name in payload["transition_order"]),
            match_emission=_parse_array_spec(payload["match_emission"]),
            insert_emission=_parse_array_spec(payload["insert_emission"]),
            transition_logits=_parse_array_spec(payload["transition_logits"]),
            metadata=dict(payload.get("metadata", [])),
        )

    def load(self) -> LoadedInitialPhmmArtifact:
        manifest_path = self.resolve_manifest_path()
        manifest = self.load_manifest()
        graph_root = self._graph_root_from_manifest(manifest_path)
        if (graph_root / "manifest.json").exists():
            graph_metadata = TensorGraphArtifactLoader(graph_root).load_manifest().metadata
        else:
            graph_metadata = GraphMetadata(
                graph_id=graph_root.name,
                format_name="ad_phmm_tensor_graph",
                format_version="1",
                global_state_count=manifest.global_state_count,
                state_interval_semantics=StateIntervalSemantics.HALF_OPEN,
            )
        tensors = {
            "match_emission": _load_array(graph_root, manifest.match_emission),
            "insert_emission": _load_array(graph_root, manifest.insert_emission),
            "transition_logits": _load_array(graph_root, manifest.transition_logits),
        }
        parameters = InitialPhmmParameters(
            track=manifest.track,
            graph=GraphMetadata(
                graph_id=graph_metadata.graph_id,
                format_name=graph_metadata.format_name,
                format_version=graph_metadata.format_version,
                source_path=graph_metadata.source_path,
                global_state_count=manifest.global_state_count,
                alphabet=graph_metadata.alphabet,
                state_interval_semantics=graph_metadata.state_interval_semantics,
                extra=dict(graph_metadata.extra),
            ),
            tensors=tensors,
            metadata={
                **dict(manifest.metadata),
                "alphabet_size": manifest.alphabet_size,
                "transition_order": tuple(manifest.transition_order),
            },
        )
        return LoadedInitialPhmmArtifact(manifest=manifest, parameters=parameters)

    def load_parameters(self) -> InitialPhmmParameters:
        return self.load().parameters


def required_array_specs(
    manifest: TensorGraphManifest, names: Sequence[str]
) -> list[ArraySpec]:
    """Return a checked list of required manifest array specs."""

    specs: list[ArraySpec] = []
    for name in names:
        spec = manifest.arrays.get(name)
        if spec is None:
            raise ArtifactValidationError(f"required manifest array is missing: {name}")
        specs.append(spec)
    return specs
