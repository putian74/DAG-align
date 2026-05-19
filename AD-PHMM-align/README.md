# AD-PHMM-align

AD-PHMM-align is the Python/PyTorch training project for automatic-differentiation PHMM alignment on DAGs.

The project boundary is intentional:

- **DAG-rust / Rust preprocessing** owns graph construction, reference path selection, path-to-reference initialization, PHMM initialization artifact export, state sampling ranges, and global/subgraph projections.
- **AD-PHMM-align** loads typed graph and initialization artifacts, instantiates PyTorch PHMM parameters, runs differentiable dynamic programs, trains with SGD, decodes alignments, and evaluates metrics.

Initial development supports current DAG-align graph artifacts through a compatibility adapter and typed cache. As DAG-rust matures, AD-PHMM-align should consume DAG-rust exports without changing the PyTorch training core.

## Status

This scaffold defines modules, public interfaces, and placeholders. Algorithm implementations will be filled in phase by phase.

