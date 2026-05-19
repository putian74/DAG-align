# AD-PHMM-align plan

## Purpose

AD-PHMM-align is the Python/PyTorch project for differentiable PHMM training and alignment on DAGs. It consumes typed artifacts from `Pre-AD-prep` and keeps graph conversion, reference-MSA construction, global coordinate construction, and CPU-heavy preprocessing outside the training loop.

## Project boundary

- `Pre-AD-prep` produces `tensor_graph.v1` artifacts, initialization tracks, packed state windows, edge-window overlaps, global/subgraph projections, and diagnostics.
- `AD-PHMM-align` loads those artifacts, validates contracts, instantiates trainable PHMM tensors, runs differentiable forward/backward and Viterbi-style programs, trains with SGD, records profiling data, decodes alignments, and reports metrics.
- Mature AD-PHMM-align workflows should not parse legacy DAG-align pickle/object arrays directly.

## Coordinate convention

All PHMM state intervals are half-open: `[left, right)`.

This convention matches NumPy/PyTorch slicing, makes packed-window lengths exactly `right - left`, and avoids inclusive-bound off-by-one ambiguity. `Pre-AD-prep` and AD-PHMM-align must use the same convention in manifests, node windows, edge overlaps, subgraph projections, and sampling ranges.

## Core artifact contracts

AD-PHMM-align should treat the following as first-class public contracts:

- tensor graph metadata and manifest information;
- node arrays, edge arrays, CSR/CSC adjacency, topological order, and optional topological levels;
- half-open node state windows;
- packed window offsets and lengths;
- edge-window overlap offsets and lengths;
- source-format, alphabet, symbol-encoding, and array-identity manifest fields shared with `Pre-AD-prep`;
- sequence/source provenance needed for likelihood scaling and alignment-quality losses;
- initialization tracks: `legacy_current` and `reference_msa`;
- global-to-subgraph state projections and active-state masks.

Training must refuse to run when required window/projection arrays are absent, while validation and diagnostics may still inspect partial artifacts.

## Data-preparation module design

Initial public interfaces should support:

- loading a typed graph artifact into a `TensorDag`;
- validating half-open intervals, packed windows, and edge-window overlaps;
- representing sampled subgraphs without losing global state IDs;
- constructing prepared training batches with tensor-like fields but without importing PyTorch at package import time;
- carrying batch metadata, sequence counts, likelihood scaling weights, and optional profiling information.

## Training module design

Initial public interfaces should support:

- `TrainingConfig` with optimizer, loss weights, subgraph sampling, device, reproducibility, and profiling options;
- `TrainingStepInput` as the boundary between data preparation and PHMM training;
- `TrainingStepResult` with total loss, component losses, metrics, step index, batch ID, and profiling results;
- `FitResult` for high-level training summaries and checkpoint metadata;
- trainer placeholders that make future implementation expectations explicit.

## Profiling from the start

CPU time and memory profiling should be available before heavy optimization begins. Early training and data-preparation steps should be able to record wall time, process CPU time, peak resident memory, and optional device-memory metrics. These results should flow through training-step outputs and progress diagnostics so later optimizations have a baseline.

## Implementation phases

1. Keep the scaffold importable with NumPy-only core dependencies and optional PyTorch training extras.
2. Finalize typed artifact, half-open interval, packed-window, edge-overlap, and subgraph batch contracts.
3. Implement artifact loaders for `tensor_graph.v1`.
4. Validate initialization artifacts and instantiate PHMM parameters from shared initialization tracks.
5. Implement CPU reference banded DP on tiny graphs.
6. Implement PyTorch packed-window forward/backward and soft/hard Viterbi.
7. Add losses, metrics, subgraph SGD, checkpointing, and profiling reports.
8. Iterate with `Pre-AD-prep` whenever training needs additional exported arrays, diagnostics, or layout changes.
