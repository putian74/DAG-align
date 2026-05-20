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
- Implemented the first CPU dense-reference training core for tiny DAGs:
  - Added NumPy forward/backward recurrences over the global PHMM state axis with DAG branch fan-out and merge reductions through normalized incoming-edge weights.
  - Added posterior occupancy summaries from the CPU reference forward/backward path.
  - Added hard Viterbi decoding with backpointer capture and a soft-Viterbi score baseline.
  - Added CPU-reference likelihood, entropy, pairwise-score, and regularization helpers.
  - Upgraded the trainer scaffold so a baseline step now computes real CPU-reference losses/metrics instead of only dry-run validation metadata.
  - Added unit coverage for forward/backward likelihood parity, posterior shapes, Viterbi/soft-Viterbi behavior, and trainer execution on tiny synthetic artifacts.
- Split soft and hard inference into explicit path modules and threaded both through trainer-side inference summaries.
- Added a baseline sequence-batch subgraph sampler:
  - materializes provenance-aware sampled `TensorDag` views;
  - supports optional global-state range clipping through `StateMaskSpec`;
  - lets trainer step execution run on sampled subgraphs instead of only full-graph batches.

## Active phase

**Next:** complete and validate the full inference baseline end to end: preprocessing-produced graph/init artifacts -> sampled/full `TensorDag` batches -> CPU forward/backward/posterior -> hard/soft Viterbi -> hard/soft entropy and scaled-SP -> trainer reporting. Only after that baseline is correct should the same recurrences be lifted to the PyTorch wavefront backend.

## Pending phases

1. Finish the complete CPU inference/objective path, especially robust `ValSparseMSA`-based hard decode/metrics and exported sampling/projection integration from Pre-AD-prep.
2. Lift the reference kernels to a PyTorch wavefront backend.
3. Profile and optimize CUDA-oriented merge-reduction and overlap-transfer hotspots.
4. Extend baseline sequence-batch sampling into full subgraph SGD with exported state projections.
5. Compare `reference_msa` initialization after baseline training is stable.
6. Add DAG-rust/Pre-AD-prep migration tests and benchmarks.

## Alternating iteration workflow

Pre-AD-prep and AD-PHMM-align should evolve together:

1. Pre-AD-prep exports a minimal typed artifact.
2. AD-PHMM-align loads and validates it.
3. AD-PHMM-align exposes training/runtime needs.
4. Pre-AD-prep updates artifact layout or diagnostics.
5. Repeat until DP/training performance and correctness are stable.
