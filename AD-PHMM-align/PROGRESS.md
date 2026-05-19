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

## Active phase

**Next:** implement artifact loaders for the `Pre-AD-prep/tensor_graph.v1` manifest and arrays using the tightened contracts.

## Pending phases

1. Implement artifact loaders for `tensor_graph.v1`.
2. Validate graph arrays, state windows, edge overlaps, and initialization artifacts.
3. Instantiate PHMM tensors from `InitialPhmmParameters`.
4. Implement CPU reference banded DP for tiny tests.
5. Implement PyTorch packed-window forward/backward.
6. Implement soft-Viterbi and hard Viterbi over the same packed-window layout.
7. Implement losses and metrics.
8. Implement subgraph SGD with global state projections.
9. Add DAG-rust/Pre-AD-prep migration tests and benchmarks.

## Alternating iteration workflow

Pre-AD-prep and AD-PHMM-align should evolve together:

1. Pre-AD-prep exports a minimal typed artifact.
2. AD-PHMM-align loads and validates it.
3. AD-PHMM-align exposes training/runtime needs.
4. Pre-AD-prep updates artifact layout or diagnostics.
5. Repeat until DP/training performance and correctness are stable.
