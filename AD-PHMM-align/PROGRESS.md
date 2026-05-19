# AD-PHMM-align progress

## Current status

- Python/PyTorch project scaffold created.
- Package name: `ad_phmm_align`.
- CLI entry point: `ad-phmm-align`.
- Smoke test passes with:

```bash
cd AD-PHMM-align && PYTHONPATH=src python3 -m pytest -q
```

## Project role

AD-PHMM-align consumes typed artifacts produced by Pre-AD-prep and focuses on:

- loading and validating graph/preprocessing artifacts;
- instantiating PyTorch PHMM parameters;
- differentiable forward/backward;
- soft-Viterbi and hard Viterbi;
- losses, SGD, checkpoints, decoding, and evaluation.

It should not own legacy DAG-align pickle/object conversion or CPU-heavy preprocessing in mature workflows.

## Completed

- Created package scaffold under `AD-PHMM-align/`.
- Added public schemas for graph metadata and initialization tracks.
- Added placeholder interfaces for graph loading, PHMM parameters, DP, Viterbi, losses, sampling, training, and evaluation.
- Added smoke test for imports.
- Added project-local `PLAN.md` beside this progress file.
- Standardized PHMM state intervals as half-open `[left, right)`.
- Added public contracts for packed state windows, edge-window overlaps, richer subgraph batches, prepared training batches, training step results, and CPU/memory profiling.
- Completed a consistency audit against Pre-AD-prep:
  - Aligned manifest source format, alphabet, array identity, and half-open interval schemas.
  - Added edge-overlap cross-checks against packed node windows.
  - Added training artifact precondition checks for required state windows and edge overlaps.
  - Extended prepared training batches with graph scheduling/adjacency fields needed by DP kernels.
- Implemented the first loader/range scaffold pass:
  - Added typed `tensor_graph.v1` and initialization artifact loaders with dtype/shape validation.
  - Added named transition-logit views so PHMM code consumes one packed tensor through stable transition names.
  - Added effective packed-state masks, forward/backward support propagation, and wavefront scheduling scaffolds.
  - Added typed forward/backward, Viterbi, and soft-Viterbi preparer interfaces plus trainer runtime-artifact loading.
  - Tightened consistency checks for packed-window offsets, overlap edge IDs, transition tensor rank/width, and graph/init compatibility.
  - Added unit coverage for loader wiring, transition views, effective-support propagation, and wavefront schedules.
  - Adopted the explicit coordinate/window split from Pre-AD-prep:
    - `node_coordinate_left/right` carry raw propagated legal spans.
    - `node_window_left/right` carry the static padded DP windows that must match packed-window arrays.
    - `TensorDag`, loaders, and public batch contracts now distinguish those two interval layers directly.

## Active phase

**Next:** implement the CPU reference forward/backward and hard/soft Viterbi recurrences on top of the now-typed loader, effective-support, and wavefront scaffolds.

## Pending phases

1. Implement CPU reference forward/backward for branching/merging DAGs.
2. Implement soft-Viterbi and hard Viterbi over the same packed-window layout.
3. Lift the reference kernels to a PyTorch wavefront backend.
4. Profile and optimize CUDA-oriented merge-reduction and overlap-transfer hotspots.
5. Implement losses and metrics.
6. Implement subgraph SGD with global state projections.
7. Compare `reference_msa` initialization after baseline training is stable.
8. Add DAG-rust/Pre-AD-prep migration tests and benchmarks.

## Alternating iteration workflow

Pre-AD-prep and AD-PHMM-align should evolve together:

1. Pre-AD-prep exports a minimal typed artifact.
2. AD-PHMM-align loads and validates it.
3. AD-PHMM-align exposes training/runtime needs.
4. Pre-AD-prep updates artifact layout or diagnostics.
5. Repeat until DP/training performance and correctness are stable.
