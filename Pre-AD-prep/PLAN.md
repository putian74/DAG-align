# Pre-AD-prep implementation plan

## Purpose

Pre-AD-prep is the preprocessing bridge between graph construction and PyTorch training.

It consumes current DAG-align graph artifacts or future DAG-rust graph exports, then produces AD-ready typed artifacts for AD-PHMM-align. It owns CPU-side work that does not require PyTorch: graph conversion, typed tensor graph cache production, global PHMM state windows, edge-window overlaps, topological scheduling metadata, reference-path MSA preprocessing, initialization artifacts, state-sampling ranges, and subgraph projections.

## Project boundaries

| Project | Responsibility |
| --- | --- |
| `DAG-rust/` | Graph construction, merging, post-processing, core graph persistence/export |
| `Pre-AD-prep/` | AD-ready preprocessing and typed artifact generation |
| `AD-PHMM-align/` | PyTorch model, differentiable DP, losses, SGD, decoding, evaluation |

AD-PHMM-align should not parse legacy DAG-align pickle/object arrays directly in mature workflows. It should consume Pre-AD-prep outputs.

## Output artifact contract

Pre-AD-prep should produce a stable directory format:

```text
tensor_graph.v1/
  manifest.json
  graph/
    node_symbol.npy
    node_weight.npy
    node_flags.npy
    edge_src.npy
    edge_dst.npy
    edge_weight.npy
    csr_indptr.npy
    csr_indices.npy
    csr_edge_id.npy
    csc_indptr.npy
    csc_indices.npy
    csc_edge_id.npy
    topo_order.npy
    topo_level_ptr.npy
    topo_level_nodes.npy
  coordinates/
    node_state_left.npy
    node_state_right.npy
    node_state_offset.npy
    node_state_len.npy
    edge_state_src_offset.npy
    edge_state_dst_offset.npy
    edge_state_overlap_len.npy
  source/
    sequence_id.npy
    sequence_name_offset.npy
    sequence_name_bytes.npy
    source_record_offset.npy
    source_sequence_id.npy
    source_position.npy
    node_source_offset.npy
    node_source_len.npy
  reference/
    ref_node_ids.npy
    ref_sequence_symbols.npy
    insert_region_left.npy
    insert_region_right.npy
  initialization/
    legacy_current/
    reference_msa/
  subgraphs/
  diagnostics/
```

The format should avoid Python object arrays. All arrays should have explicit dtype, shape, semantics, and required/optional status in `manifest.json`.

All PHMM state intervals use half-open `[left, right)` semantics so `node_state_len` is exactly `right - left` and downstream NumPy/PyTorch slicing is unambiguous.

## Implementation phases

1. **Scaffold Pre-AD-prep**
   - Create Rust-first project layout with CLI, library modules, tests, fixtures, and benches.
   - Add temporary compatibility space for a Python legacy converter only if strictly needed to read current pickle/object artifacts.

2. **Legacy DAG-align converter**
   - Read current graph artifacts: `graph.pkl`, `data.npz`, `osm.npy`, `onm.npy`, `onm_index.npy`, `v_id.npy`, optional `thr_*.npz`, optional `ini/*.npy`.
   - Convert edges, nodes, symbols, weights, flags, sequence IDs, and source/provenance records into typed arrays.
   - Build deterministic CSR/CSC adjacency and topological order.

3. **Tensor graph manifest and validation**
   - Define `manifest.json` schema.
   - Validate dtypes, shapes, node/edge counts, CSR/CSC consistency, acyclicity, topological order, and source ranges.
   - Add small golden fixtures from current DAG-align outputs.

4. **Global PHMM coordinates**
   - Compute or import global reference/backbone coordinates.
   - Export `node_state_left/right` and reference path arrays.
   - Track ambiguous mappings from legacy graph data.

5. **Banded state-window export**
   - Build packed ragged state-window layout with `node_state_offset/len`.
   - Precompute edge-window overlaps for GPU banded DP.
   - Export topological levels or coarse paths for dependency-safe GPU scheduling.

6. **Reference-path MSA preprocessing**
   - Extract source paths.
   - Align each path to the global reference path with banded/interval-constrained DP.
   - Parallelize across paths with deterministic reducers.
   - Accumulate sufficient statistics without materializing dense MSA when possible.

7. **Initialization artifacts**
   - Export `legacy_current` initialization in a common schema.
   - Export `reference_msa` initialization in the same schema.
   - Include metadata for graph/reference version, smoothing, priors, parameter shapes, provenance, and track label.

8. **State-sampling ranges and projections**
   - Derive candidate state-position ranges from MSA support, entropy, gap rate, insertions, deletions, and uncertainty.
   - Export MSA-to-global and global-to-subgraph projection arrays.

9. **DAG-rust input path**
   - Add support for DAG-rust exported graphs.
   - Ensure current DAG-align conversion and DAG-rust input produce the same `tensor_graph.v1` contract.

10. **Performance and diagnostics**
    - Benchmark conversion throughput, preprocessing throughput, memory use, window-width distribution, edge-overlap density, and artifact load time.
    - Emit diagnostic summaries for graph quality, coordinate uncertainty, initialization tracks, and sampling ranges.

## Validation invariants

- Graph is acyclic.
- Edge endpoints are valid.
- CSR/CSC arrays match edge arrays.
- Topological order respects all edges.
- Source/provenance ranges are valid.
- State windows are valid global coordinate intervals.
- Edge-window overlap metadata equals hand-computed interval intersections.
- Topological level batches respect dependencies.
- Initialization tracks produce compatible parameter shapes and metadata.
- Subgraph projections preserve global PHMM state identity.

## Near-term first slice

Start with the legacy DAG-align converter and `tensor_graph.v1` manifest/validation. This creates the artifact contract that both Rust preprocessing and AD-PHMM-align can iterate against.

The initial implemented converter slice uses a transitional Python/Numpy bridge only for reading current DAG-align pickle/object `.npz/.npy` artifacts. The Rust adapter owns the public API, input validation, converter orchestration, typed graph return value, and downstream validation. The bridge output is typed `.npy` arrays plus JSON metadata, so AD-PHMM-align does not need to consume Python object arrays.

The Rust adapter and AD-PHMM-align manifest schema must stay synchronized for source format, alphabet, symbol encoding, array identity, half-open state interval semantics, diagnostics, and profiling metadata.

## Detailed module implementation plans

### `error`

Purpose: provide a small, explicit error surface for every preprocessing stage.

Implementation plan:

- Keep `PreAdPrepError` as the crate-wide error enum with variants for validation, unsupported features, filesystem I/O, and JSON/manifest parsing.
- Add contextual constructors only when they reduce repeated string formatting in call sites.
- Avoid broad catch-all success-shaped fallbacks: unsupported initialization, missing source provenance, missing state windows, and ambiguous coordinates should return explicit errors or diagnostics depending on whether the operation can still produce a partial artifact.

Detailed blueprint:

- Public API to keep:
  - `type Result<T> = std::result::Result<T, PreAdPrepError>;`
  - `enum PreAdPrepError { Validation, Unsupported, Io, Json, ... }`
- Next additions:
  - `Manifest(String)` once manifest serialization moves fully into Rust;
  - `ExternalTool { tool: String, detail: String }` if the Python bridge remains for multiple phases;
  - helper constructors for `missing_required(path)`, `invalid_array(name)`, and `contract_mismatch(module, detail)` if repeated sites become noisy.
- Implementation order:
  1. keep current enum minimal;
  2. add dedicated variants only when multiple call sites need structured branching;
  3. ensure CLI maps validation errors to user-facing messages and nonzero exit codes.
- Test targets:
  - conversions from `std::io::Error` and `serde_json::Error`;
  - string formatting for representative validation failures;
  - explicit error propagation from the legacy bridge and future manifest loader.

Acceptance criteria:

- Public APIs return `Result<T>` consistently.
- Error messages include the module operation and the relevant path, array name, node ID, edge ID, or state interval.

### `validate`

Purpose: accumulate warnings and errors while still allowing partial artifact diagnostics.

Implementation plan:

- Keep `Validate::validate() -> Result<ValidationReport>` for structural checks.
- Use `ValidationReport::into_result()` only at operation boundaries where training/export cannot continue.
- Add reusable validators for:
  - array shape and dtype declarations against manifest specs;
  - node/edge counts;
  - topological order;
  - CSR/CSC consistency;
  - source/provenance offsets;
  - half-open state windows;
  - edge-window overlaps;
  - initialization tensor shapes.

Detailed blueprint:

- Public API to implement:
  - `trait Validate { fn validate(&self) -> Result<ValidationReport>; }`
  - `ValidationReport::{push,error,warning,has_errors,into_result}`
  - reusable free functions:
    - `validate_manifest_arrays(...)`
    - `validate_required_arrays_present(...)`
    - `validate_training_ready_artifact(...)`
    - `validate_reference_arrays(...)`
    - `validate_initialization_manifest(...)`
- Module split once size grows:
  - `validate/graph.rs`
  - `validate/manifest.rs`
  - `validate/source.rs`
  - `validate/init.rs`
- Implementation order:
  1. add manifest/array validators;
  2. add source/provenance validators;
  3. add coordinate/window validators;
  4. add initialization validators;
  5. expose one training-ready validation entrypoint used by CLI and tests.
- Test targets:
  - multiple warnings plus one fatal error in a single report;
  - manifest shape mismatch;
  - missing required array;
  - invalid source offsets;
  - invalid overlap bounds.

Acceptance criteria:

- Validation can report multiple issues in one pass.
- Fatal errors are not hidden as warnings.
- Validation codes are stable enough to be asserted in tests and surfaced in diagnostics.

### `graph`

Purpose: hold the canonical typed DAG representation independent of legacy DAG-align or DAG-rust input.

Implementation plan:

- Keep `TensorGraph` as the in-memory graph core with node arrays, edge arrays, CSR/CSC, topological order, optional levels, optional state windows, and optional edge overlaps.
- Implement graph construction helpers:
  - deterministic edge sorting by `(src, dst, weight, original_edge_id)`;
  - CSR and CSC builders that preserve edge IDs;
  - Kahn topological sort with cycle detection;
  - topological level construction for dependency-safe parallel scheduling.
- Add source/provenance attachment by composition rather than overloading `TensorGraph`; a future `source` module should own typed source records and node-source ranges.
- Keep node symbols encoded as compact integer IDs. Full fragments can be exported only as optional diagnostics or source metadata.

Detailed blueprint:

- Public API to implement or keep:
  - `TensorGraph`
  - `AdjacencyCsr`
  - `TopologicalLevels`
  - `NodeFlags`
  - helpers:
    - `TensorGraph::new(...)`
    - `TensorGraph::node_count()`
    - `TensorGraph::edge_count()`
    - `TensorGraph::validate_with_global_states(...)`
    - `build_topological_order(...)`
    - `build_adjacency(...)`
    - `build_topological_levels(...)`
- Planned internal data flow:
  1. legacy/DAG-rust adapter produces raw node/edge arrays;
  2. graph helpers normalize edge order and derive CSR/CSC/topo;
  3. coordinate module augments graph with windows/overlaps;
  4. export module writes graph arrays and manifest entries.
- Invariants to preserve:
  - edge order is deterministic and stable across runs;
  - CSR/CSC refer to the same edge IDs as `edge_src/edge_dst`;
  - node flags encode start/end/reference independently;
  - topological levels are optional but, when present, partition nodes in topological order.
- Test targets:
  - cycle detection;
  - duplicate edge order stability;
  - CSR/CSC reconstruction of children/parents;
  - topological level correctness on branched DAGs.

Acceptance criteria:

- `TensorGraph::validate_with_global_states()` checks graph structure, adjacency, optional windows, and optional overlaps.
- CSR/CSC arrays round-trip against edge arrays in tests.
- Topological levels respect all DAG dependencies.

### `coordinates`

Purpose: define global PHMM coordinate contracts and packed layouts used by GPU-friendly banded DP.

Implementation plan:

- Use half-open `StateInterval { left, right }` everywhere.
- Keep `PackedStateWindows` as the source of node-local packed state ranges:
  - `intervals[node] = [left, right)`;
  - `offsets[node]` indexes a flat state buffer;
  - `lengths[node] == right - left`.
- Extend `EdgeWindowOverlaps` to support construction from graph edges plus source/destination windows.
- Add projection types:
  - reference/MSA column to global state;
  - global state to subgraph-local state;
  - node-local window index to global state.
- Add ambiguity annotations for legacy-derived coordinate mappings, especially repeats and missing anchors.

Detailed blueprint:

- Public API to implement or expand:
  - `StateInterval`
  - `PackedStateWindows`
  - `EdgeWindowOverlap`
  - `EdgeWindowOverlaps`
  - `StateProjection`
  - future additions:
    - `ReferencePath`
    - `GlobalCoordinateMap`
    - `SamplingProposal`
    - `AmbiguousMapping`
- Planned sub-splits:
  - `coordinates/intervals.rs`
  - `coordinates/global.rs`
  - `coordinates/windows.rs`
  - `coordinates/projections.rs`
- Implementation order:
  1. finish interval + overlap validation;
  2. implement global coordinate propagation;
  3. implement packed windows from intervals;
  4. implement global-to-subgraph projections;
  5. implement MSA/state-sampling proposals.
- Data flow:
  - global graph + reference anchors -> node global intervals;
  - node intervals -> packed windows;
  - packed windows + edges -> overlap rows;
  - windows + subgraph selection -> local/global projections.
- Test targets:
  - half-open interval edge cases;
  - overlap construction and zero-overlap behavior;
  - repeat/ambiguous anchor detection;
  - projection identity preservation.

Acceptance criteria:

- Empty intervals are allowed only when explicitly useful and documented.
- Edge overlap construction matches hand-computed interval intersections.
- Every state proposal and subgraph batch preserves global state identity.

### `export`

Purpose: define and write the stable `tensor_graph.v1` artifact contract consumed by AD-PHMM-align.

Implementation plan:

- Add manifest serialization/deserialization for `TensorGraphManifest`.
- Keep manifest fields synchronized with AD-PHMM-align:
  - `format_name`;
  - `format_version`;
  - `source_format`;
  - `source_graph_dir`;
  - `node_count`;
  - `edge_count`;
  - `sequence_count`;
  - `global_state_count`;
  - `alphabet`;
  - `symbol_encoding`;
  - `state_interval_semantics`;
  - `arrays`;
  - source/legacy metadata;
  - diagnostics/profiling metadata where appropriate.
- Implement typed array writers/readers. The first writer may delegate `.npy` writing to the transitional bridge, but Rust should own manifest validation and eventually own `.npy` writing.
- Add an artifact writer that creates directory structure atomically enough to avoid half-written training inputs.

Detailed blueprint:

- Public API to implement or expand:
  - `DataType`
  - `SourceFormat`
  - `StateIntervalSemantics`
  - `ArraySpec`
  - `TensorGraphManifest`
  - `TensorGraphArtifact`
  - future writer/reader functions:
    - `write_manifest(...)`
    - `read_manifest(...)`
    - `write_graph_arrays(...)`
    - `write_source_arrays(...)`
    - `write_coordinate_arrays(...)`
    - `write_diagnostics(...)`
- File layout responsibilities:
  - `graph/`: graph topology and node/edge arrays;
  - `source/`: sequence/source provenance;
  - `coordinates/`: global intervals, packed windows, overlaps;
  - `reference/`: reference path arrays;
  - `initialization/`: common init-track outputs;
  - `subgraphs/`: future projection/materialized subgraph caches;
  - `diagnostics/`: summaries and profiling.
- Implementation order:
  1. Rust manifest serialization/deserialization;
  2. manifest-file-to-array consistency checks;
  3. training-ready validation level;
  4. atomic-ish directory writing;
  5. eventual Rust `.npy` writing replacement for the bridge.
- Test targets:
  - manifest round-trip;
  - missing/extra array spec handling;
  - dtype string parity with AD-PHMM-align;
  - partial artifact detection.

Acceptance criteria:

- Manifest array specs match actual files, dtypes, and shapes.
- Required arrays are present before training-ready validation passes.
- AD-PHMM-align can load the manifest without schema translation hacks.

### `legacy`

Purpose: convert current DAG-align graph directories into typed `tensor_graph.v1` artifacts.

Implementation plan:

- Keep `LegacyDagAlignInput` as the path contract for current DAG-align artifacts.
- Current implemented slice reads `data.npz` through the isolated Python/Numpy bridge and returns typed graph arrays.
- Next legacy work:
  - flatten `osm.npy` or `onm.npy`/`onm_index.npy` into source/provenance arrays;
  - parse `v_id.npy` into sequence ID/name byte tables;
  - read optional `thr_*.npz` reference artifacts into `reference/`;
  - read optional `ini/*.npy` artifacts only for `legacy_current` initialization export;
  - record missing optional inputs as diagnostics, not silent absence.
- Keep `LegacyConversionOptions` meaningful:
  - `allow_python_object_bridge` gates transitional Python object-array loading;
  - `include_initialization` controls optional initialization artifact export;
  - `require_state_windows` should fail until coordinate preprocessing is available.

Detailed blueprint:

- Public API to keep:
  - `LegacyDagAlignInput`
  - `LegacyConversionOptions`
  - `LegacyConversionOutput`
  - `trait LegacyAdapter`
  - `LegacyDagAlignAdapter`
- Step-by-step implementation:
  1. preflight required files (`graph.pkl`, `data.npz`);
  2. inspect optional files and record diagnostics;
  3. run transitional Python bridge only when requested;
  4. parse typed JSON sidecar back into Rust structures;
  5. build validated `TensorGraph`;
  6. build validated manifest;
  7. attach diagnostics/profiling;
  8. later add source/reference/init export.
- Optional-input handling plan:
  - `v_id.npy`: sequence table;
  - `osm.npy`: build-graph source records;
  - `onm.npy`/`onm_index.npy`: traceability/original-node mappings;
  - `Traceability_path.npy`: merge-mode traceability;
  - `thr_*.npz`: reference anchors and insert regions;
  - `ini/*.npy`: current initialization baseline.
- Test targets:
  - build-graph directory with minimal files;
  - merge-graph directory with traceability-only provenance;
  - missing required file failure;
  - optional file absence warning;
  - malformed object-array bridge output.

Acceptance criteria:

- Missing required files fail before bridge execution.
- Missing optional files emit diagnostics.
- Output graph arrays are typed and object-free.
- Converter output passes graph, manifest, and source/provenance validation.

### `source` module to add

Purpose: own sequence/source provenance needed for likelihood scaling, source-path extraction, initialization, and evaluation.

Implementation plan:

- Add `src/source.rs` with:
  - `SequenceTable`: sequence IDs, name offsets, name bytes;
  - `SourceRecordTable`: source sequence IDs, source positions, optional packed raw records;
  - `NodeSourceRanges`: node-source offsets and lengths;
  - validation for all offset/length ranges.
- Implement legacy flattening:
  - build graphs: decode `osm.npy` when bit widths are known, otherwise preserve raw packed records plus diagnostics;
  - merge graphs: use `Traceability_path.npy`, `onm.npy`, and `onm_index.npy` where available;
  - always preserve enough metadata to reproduce or audit decoding assumptions.
- Export under `source/` with manifest entries.

Detailed blueprint:

- File to add: `src/source.rs` (later splittable into `source/sequence.rs`, `source/records.rs`, `source/decode.rs`).
- Public API to implement:
  - `SequenceTable`
  - `SourceRecord`
  - `SourceRecordTable`
  - `NodeSourceRanges`
  - `SourceDecodeStatus`
  - `flatten_legacy_build_sources(...)`
  - `flatten_legacy_merge_sources(...)`
  - `validate_source_tables(...)`
- Data flow:
  1. load legacy source arrays;
  2. decode to `(sequence_id, source_position)` when bit packing is understood;
  3. preserve packed raw fallback when decode is incomplete;
  4. build flat tables plus node ranges;
  5. export typed arrays and diagnostics.
- Test targets:
  - exact match between decoded record counts and node weights when possible;
  - fallback path when decode is ambiguous;
  - sequence-name byte table round-trip.

Acceptance criteria:

- Every node can report zero or more supporting source records.
- Source ranges are valid and match node weights when legacy data makes that check possible.
- Sequence names are represented as offsets plus UTF-8 bytes, not Python objects.

### `coordinates/global` work to add inside `coordinates` or a submodule

Purpose: assign globally consistent PHMM state coordinates before SGD subgraph training.

Implementation plan:

- Accept a full/global graph and reference/backbone source:
  - existing `thr_*.npz` reference path if present;
  - graph-derived reference path if available;
  - later explicit DAG-rust coordinates.
- Implement `calculateStateRange`-style propagation in Rust with half-open output.
- Export:
  - `node_state_left.npy`;
  - `node_state_right.npy`;
  - `ref_node_ids.npy`;
  - `ref_sequence_symbols.npy`;
  - ambiguous/missing coordinate diagnostics.
- Add repeated-fragment fixtures that force ambiguous legacy mappings.

Detailed blueprint:

- File to add: `src/coordinates/global.rs` or `src/global_coordinates.rs`.
- Public API to implement:
  - `GlobalCoordinateConfig`
  - `GlobalCoordinateOutput`
  - `build_global_coordinates(...)`
  - `project_reference_to_nodes(...)`
  - `propagate_state_ranges(...)`
- Planned inputs:
  - `TensorGraph`
  - optional `ReferencePath`
  - optional legacy `thr_*.npz` arrays
  - optional source/provenance support
- Planned outputs:
  - node interval table
  - reference node IDs
  - reference sequence symbols
  - ambiguity diagnostics
  - later: anchor confidence scores
- Test targets:
  - linear path;
  - branched DAG with clear reference;
  - repeated fragment ambiguity;
  - missing anchor segments.

Acceptance criteria:

- Global state count is stable for all subgraphs derived from the same full graph.
- Repeats and missing anchors are diagnosed.
- Subgraph projections never invent local state IDs disconnected from global IDs.

### `coordinates/windows` work to add

Purpose: convert node global intervals into packed GPU-friendly windows and edge overlaps.

Implementation plan:

- Build `PackedStateWindows` from node intervals and configurable padding.
- Build `EdgeWindowOverlaps` from source/destination interval intersections.
- Export:
  - `node_state_offset.npy`;
  - `node_state_len.npy`;
  - `edge_state_src_offset.npy`;
  - `edge_state_dst_offset.npy`;
  - `edge_state_overlap_len.npy`;
  - optional edge IDs for overlap rows if order differs from edge order.
- Add window-width diagnostics: min/mean/max, percentile distribution, and edges with zero overlap.

Detailed blueprint:

- File to add: `src/coordinates/windows.rs`.
- Public API to implement:
  - `WindowBuildConfig { padding_left, padding_right, clamp_to_global, ... }`
  - `WindowBuildOutput`
  - `build_packed_windows(...)`
  - `build_edge_window_overlaps(...)`
  - `summarize_window_diagnostics(...)`
- Execution order:
  1. read validated global node intervals;
  2. apply padding/clamping;
  3. compute packed offsets/lengths;
  4. intersect edge endpoint windows;
  5. export arrays and diagnostics.
- Test targets:
  - no-padding exact window;
  - padded then clamped window;
  - zero-overlap edge;
  - dense branch overlap fan-out.

Acceptance criteria:

- Packed lengths equal `right - left`.
- Every overlap fits both endpoint windows.
- Window diagnostics are available before AD-PHMM-align training starts.

### `init`

Purpose: emit common PHMM initialization artifacts for all initialization tracks.

Implementation plan:

- Keep `InitializationTrack::{LegacyCurrent, ReferenceMsa}`.
- Define one common output schema:
  - match emission counts/logits;
  - insert emission counts/logits;
  - transition counts/logits;
  - per-position support;
  - smoothing/prior metadata;
  - graph/reference version metadata;
  - `global_state_count`;
  - `alphabet_size`.
- Implement `legacy_current` as a baseline track that mirrors current DAG-align initialization where possible.
- Implement `reference_msa` as the preferred track from Rust reference-path MSA sufficient statistics.
- Keep initialization generation deterministic and independently testable from graph conversion.

Detailed blueprint:

- Public API to implement:
  - `InitialPhmmManifest`
  - `InitializationBundle`
  - `build_legacy_current_initialization(...)`
  - `build_reference_msa_initialization(...)`
  - `validate_initialization_shapes(...)`
- Planned file split:
  - `init/common.rs`
  - `init/legacy_current.rs`
  - `init/reference_msa.rs`
  - `init/export.rs`
- Tensor contract to preserve:
  - `match_emission`
  - `insert_emission`
  - `transition_logits`
  - metadata with `global_state_count`, `alphabet_size`, smoothing/prior values, support summaries
- Test targets:
  - shape parity across both tracks;
  - metadata completeness;
  - deterministic output for fixed inputs.

Acceptance criteria:

- Both tracks export the same tensor names and compatible shapes.
- AD-PHMM-align can construct `InitialPhmmParameters` without track-specific code.
- Initialization metadata records source graph, reference, smoothing, priors, and diagnostics.

### `reference_msa` module to add

Purpose: generate the graph-derived reference-MSA initialization and sampling proposals.

Implementation plan:

- Add `src/reference_msa.rs` or `src/init/reference_msa.rs` once modules are split.
- Input:
  - typed graph;
  - source/provenance records;
  - global reference path;
  - node/global coordinate intervals.
- Extract source paths as node IDs plus symbols and intervals.
- Align paths independently to the reference path using a banded or interval-constrained DP.
- Parallelization:
  - start with path-level CPU parallelism;
  - batch paths by reference interval and length;
  - reduce partial sufficient statistics deterministically.
- Export:
  - sufficient statistics;
  - optional sparse/compressed MSA diagnostics;
  - candidate state sampling ranges with scores and reasons.

Detailed blueprint:

- File to add: `src/reference_msa.rs` or nested under `init/`.
- Public API to implement:
  - `ReferenceMsaConfig`
  - `ReferencePathAlignment`
  - `ReferenceMsaStatistics`
  - `SamplingRangeProposal`
  - `run_reference_msa_preprocessing(...)`
- Algorithm staging:
  1. extract typed source paths from provenance;
  2. bucket paths by length/reference span;
  3. align each path against the reference interval;
  4. reduce counts deterministically;
  5. derive init tensors and sampling proposals.
- Test targets:
  - tiny hand-labeled path set;
  - deterministic merge independent of worker ordering;
  - insertion-heavy region proposal generation.

Acceptance criteria:

- Tiny fixtures reproduce hand-computed MSA counts.
- Parallel reductions are deterministic.
- Sampling proposals are global-coordinate intervals.

### `diagnostics`

Purpose: provide conversion, coordinate, initialization, and performance diagnostics from the beginning.

Implementation plan:

- Keep `DiagnosticReport` and `ProfilingSummary`.
- Add structured summaries for:
  - graph conversion;
  - source/provenance decoding;
  - coordinate ambiguity;
  - window-width and overlap-density distribution;
  - initialization counts/support/entropy;
  - runtime and peak memory.
- Export diagnostics as JSON files under `diagnostics/`.

Detailed blueprint:

- Public API to implement or extend:
  - `DiagnosticSeverity`
  - `Diagnostic`
  - `DiagnosticReport`
  - `ProfilingSummary`
  - `ConversionDiagnostics`
  - helper writers:
    - `write_diagnostic_report(...)`
    - `write_profiling_summary(...)`
    - `write_graph_summary(...)`
- Diagnostics files to standardize:
  - `diagnostics/graph_core.json`
  - `diagnostics/source_decode.json`
  - `diagnostics/coordinates.json`
  - `diagnostics/windows.json`
  - `diagnostics/init_legacy_current.json`
  - `diagnostics/init_reference_msa.json`
- Test targets:
  - missing-array diagnostics;
  - profiling present on successful conversion;
  - warning-only vs error diagnostics.

Acceptance criteria:

- Every partial export explains which training-critical arrays are missing.
- Profiling data exists for conversion and later preprocessing stages.
- Diagnostics can be consumed by both humans and automated tests.

### `main` / CLI

Purpose: expose stable command boundaries for conversion, validation, and diagnostics.

Implementation plan:

- Replace placeholder parsing with subcommands:
  - `convert-legacy --graph-dir --output-dir [--allow-python-object-bridge]`;
  - `validate --artifact-dir --level graph|training-ready`;
  - `build-coordinates --artifact-dir --reference-source ...`;
  - `build-windows --artifact-dir --padding ...`;
  - `init-legacy-current --artifact-dir ...`;
  - `init-reference-msa --artifact-dir ...`;
  - `diagnose --artifact-dir`.
- Keep CLI as a thin wrapper over library APIs so tests can call the same implementation.

Detailed blueprint:

- CLI rollout order:
  1. `convert-legacy`
  2. `validate`
  3. `diagnose`
  4. `build-coordinates`
  5. `build-windows`
  6. `init-legacy-current`
  7. `init-reference-msa`
- Parsing/dispatch plan:
  - keep std-only parsing at first;
  - map each subcommand to a library function with one config struct;
  - print concise status lines and write detailed diagnostics to files.
- Test targets:
  - successful `--version`;
  - subcommand dispatch;
  - nonzero exit on validation failure;
  - artifact path propagation from CLI to library.

Acceptance criteria:

- CLI exits nonzero on validation errors.
- Library and CLI behavior match for the same inputs.
- Commands print concise summaries and write detailed diagnostics to files.

### `tests`, `fixtures`, and `benches`

Purpose: make artifact contracts stable before heavy optimization.

Implementation plan:

- Add fixtures:
  - tiny linear graph;
  - branch graph with insertion/deletion-like paths;
  - repeated fragment graph with ambiguous coordinates;
  - tiny legacy DAG-align graph;
  - small source/provenance example;
  - hand-computed state-window and edge-overlap example.
- Add tests by layer:
  - unit tests for interval/window/overlap math;
  - graph validation tests;
  - manifest round-trip tests;
  - legacy converter tests;
  - source/provenance flattening tests;
  - coordinate propagation tests;
  - initialization shape/statistic tests.
- Add benchmarks later for:
  - legacy conversion;
  - source flattening;
  - coordinate propagation;
  - edge-overlap construction;
  - reference-MSA path alignment.

Detailed blueprint:

- Directory growth plan:
  - `tests/unit/` once test volume exceeds one file;
  - `tests/integration/` for CLI/artifact round-trips;
  - `fixtures/legacy/`, `fixtures/graph/`, `fixtures/coordinates/`, `fixtures/init/`;
  - `benches/` once correctness stabilizes.
- Fixture creation strategy:
  - generate tiny arrays programmatically where possible;
  - keep only the smallest reviewable golden artifacts in-repo;
  - avoid large binary fixtures until formats stabilize.
- Benchmark entrypoints to add later:
  - `bench_legacy_convert`
  - `bench_source_flatten`
  - `bench_build_global_coordinates`
  - `bench_build_windows`
  - `bench_reference_msa`

Acceptance criteria:

- Each module has at least one direct test before being used by downstream modules.
- Golden artifact outputs are small enough to review and stable across runs.

## Dependency order for implementation

1. Complete manifest serialization and training-ready validation.
2. Extend legacy conversion to source/provenance and optional reference artifacts.
3. Implement global coordinate construction.
4. Implement packed state windows and edge overlaps.
5. Implement source-path extraction.
6. Implement `legacy_current` initialization export.
7. Implement reference-path MSA sufficient-statistics preprocessing.
8. Export MSA-derived candidate state ranges and projections.
9. Add DAG-rust input adapter once DAG-rust exports stabilize.
10. Optimize and benchmark only after correctness fixtures are stable.

Module dependency map:

- `error` and `validate` are foundational and should remain dependency-light.
- `graph` depends on `error` + `validate`.
- `coordinates` depends on `graph` + `error` + `validate`.
- `export` depends on `graph`, `coordinates`, `diagnostics`, and later `source` + `init`.
- `legacy` depends on `graph`, `export`, `diagnostics`, `error`, and later `source` + `reference`.
- `source` depends on `error` + `validate`, and feeds `legacy`, `coordinates`, `reference_msa`, and `init`.
- `init` depends on `coordinates`, `source`, `diagnostics`, and optionally `reference_msa`.
- `reference_msa` depends on `graph`, `source`, `coordinates`, `diagnostics`, and `init` common contracts.
- CLI depends on all stable library entrypoints but should not contain business logic.

## Immediate next implementation targets

1. Wire `coordinates/global` into legacy reference inputs and export `node_state_left/right`, `ref_node_ids`, and `ref_sequence_symbols`.
2. Wire `coordinates/windows` into artifact export for `node_state_offset/len` and edge-overlap arrays.
3. Extend `legacy` to load optional `thr_*.npz` reference artifacts and optional `ini/*.npy` initialization inputs.
4. Start `init/legacy_current` on top of the validated graph/source/coordinate artifacts.
