# Pre-AD-prep progress

## Current status

- Directory created: `Pre-AD-prep/`.
- Project role defined: preprocessing bridge between DAG construction and AD-PHMM-align.
- Initial plan created in `PLAN.md`.
- Rust-first crate scaffold created with typed public contracts.
- Detailed module-level implementation plans added to `PLAN.md`.
- Legacy conversion now exports training-ready coordinate/window arrays, reference artifacts, initialization manifests, diagnostics JSON, and working CLI commands.

## Active phase

**Next:** use `legacy_current` initialization as the baseline path for end-to-end PHMM training, then add reference-path MSA preprocessing later as a comparison-track initializer.

## Completed

- Created placeholder directory with `.gitkeep`.
- Defined the planned `tensor_graph.v1` artifact contract.
- Assigned preprocessing ownership here instead of DAG-rust or AD-PHMM-align.
- Added Rust crate layout with graph, coordinate, export/manifest, initialization, validation, diagnostics, and legacy adapter modules.
- Standardized PHMM state intervals as half-open `[left, right)`.
- Implemented the first legacy DAG-align converter slice:
  - Rust adapter validates inputs and drives an isolated transitional Python/Numpy object-array bridge.
  - Converts `data.npz` graph arrays into typed `tensor_graph.v1` graph `.npy` files and `manifest.json`.
  - Builds deterministic edge ordering, CSR/CSC arrays, topological order, node flags, and Rust `TensorGraph` output.
  - Adds public API tests for converter call signatures and a tiny legacy graph conversion.
- Completed a consistency audit pass:
  - Tightened half-open interval validation.
  - Added adjacency index/edge-id validation.
  - Mirrored manifest source metadata and symbol encoding through the Rust adapter.
  - Routed converter profiling summaries into diagnostics.
  - Made the currently unsupported initialization export explicit as a diagnostic.
- Added detailed module implementation plans for `error`, `validate`, `graph`, `coordinates`, `export`, `legacy`, planned `source`, planned coordinate/window submodules, `init`, planned `reference_msa`, `diagnostics`, CLI, tests, fixtures, and benchmarks.
- Implemented the `export` + `validate` module slice:
  - Rust manifest write/read round-trip for `manifest.json`.
  - Artifact-level graph-core and training-ready validation.
  - `.npy` header checks for dtype/shape consistency against manifest specs.
- Implemented the `source` module slice:
  - Added typed sequence/source provenance contracts in Rust.
  - Extended the legacy bridge to export `source/sequence_id.npy`, `sequence_name_offset.npy`, `sequence_name_bytes.npy`, `node_source_offset.npy`, `node_source_len.npy`, `source_packed.npy`, and decoded source arrays when OSM bit widths are available.
  - Added validation/tests for source table shapes and legacy source export.
- Implemented foundational `coordinates/global` and `coordinates/windows` builders:
  - Added Rust `build_global_coordinates(...)` with DAG-align-style state-range propagation.
  - Added Rust `build_packed_windows(...)` and `build_edge_window_overlaps(...)`.
  - Added unit tests covering reference-path intervals and overlap construction.
- Completed the first end-to-end preprocessing baseline:
  - Extended the Python bridge to export optional legacy reference artifacts from `thr_*.npz`.
  - Extended the bridge to export normalized `legacy_current` and bootstrap `reference_msa` initialization tracks under `initialization/*/manifest.json`.
  - Added Rust `.npy` writers for generated coordinate and initialization-adjacent arrays.
  - Wired Rust legacy conversion to derive or import a reference path, export training-ready coordinate/window arrays and `edge_state_*`, attach `global_state_count`, and validate at `TrainingReady` when coordinates are present.
  - Added JSON read/write support for conversion diagnostics and a working CLI with `convert-legacy`, `validate`, and `diagnose`.
  - Added integration coverage for training-ready conversion plus initialization manifest export.
  - Established `legacy_current` as the first training baseline, with `reference_msa` kept as a later comparative initialization path rather than a blocker for baseline PHMM training.
- Split raw coordinate spans from padded runtime windows explicitly:
  - `node_coordinate_left/right` now carry propagated legal PHMM spans.
  - `node_window_left/right` now carry the padded DP windows paired with `node_state_offset/len` and `edge_state_*`.
  - Validation now treats the raw coordinate span and static window as separate contracts instead of overloading one interval pair.

## Pending phases

1. Complete baseline PHMM training using the current `legacy_current` initialization path.
2. Add full reference-path MSA preprocessing beyond the current bootstrap `reference_msa` track for later initialization comparisons.
3. Add state-sampling ranges and global/subgraph projection exports.
4. Add DAG-rust input support.
5. Add richer performance diagnostics and benchmarks.

## Open decisions

- Whether the first legacy converter uses a temporary Python compatibility script before moving to Rust-first loading.
- When to replace the remaining legacy bridge reads with fully Rust-native loaders/writers.
- How much of the bootstrap `reference_msa` path should remain once full MSA preprocessing lands.
