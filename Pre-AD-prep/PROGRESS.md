# Pre-AD-prep progress

## Current status

- Directory created: `Pre-AD-prep/`.
- Project role defined: preprocessing bridge between DAG construction and AD-PHMM-align.
- Initial plan created in `PLAN.md`.
- Rust-first crate scaffold created with typed public contracts.
- Detailed module-level implementation plans added to `PLAN.md`.

## Active phase

**Next:** connect global PHMM coordinates and packed state-window export into legacy artifact generation.

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

## Pending phases

1. Connect the new global-coordinate builder to legacy reference artifacts and export `node_state_left/right` plus reference arrays.
2. Connect packed-window/edge-overlap builders to artifact export under `coordinates/`.
3. Extend legacy conversion to optional reference artifacts (`thr_*.npz`) and initialization inputs (`ini/*.npy`).
4. Add reference-path MSA preprocessing.
5. Add initialization artifacts for `legacy_current` and `reference_msa`.
6. Add state-sampling ranges and projections.
7. Add DAG-rust input support.
8. Add performance diagnostics and benchmarks.

## Open decisions

- Whether the first legacy converter uses a temporary Python compatibility script before moving to Rust-first loading.
- Exact manifest schema syntax and dtype naming.
- How much legacy initialization compatibility is needed in the first implementation pass.
