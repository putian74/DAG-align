# AD-PHMM-align

AD-PHMM-align is the Python/PyTorch training project for automatic-differentiation PHMM alignment on DAGs.

The project boundary is intentional:

- **DAG-rust / Rust preprocessing** owns graph construction, reference path selection, path-to-reference initialization, PHMM initialization artifact export, state sampling ranges, and global/subgraph projections.
- **AD-PHMM-align** loads typed graph and initialization artifacts, instantiates PHMM parameters, runs differentiable dynamic programs, executes the current baseline trainer, decodes alignments, and evaluates metrics.

Initial development supports current DAG-align graph artifacts through a compatibility adapter and typed cache. As DAG-rust matures, AD-PHMM-align should consume DAG-rust exports without changing the PyTorch training core.

## Status

The scaffold now has working baseline CLI entrypoints:

- `validate-artifacts` loads and checks graph/init artifact compatibility.
- `train` now runs a torch-backed autograd optimizer loop on the current dense reference objective, emits a JSON summary, and writes a final checkpoint.
- `decode` runs hard Viterbi and emits decoded-alignment metrics/summary when runtime dependencies are available.

The current remaining training gaps are higher-performance GPU kernels, richer checkpoint/resume workflow, and broader persisted alignment outputs beyond the current decode summary.
