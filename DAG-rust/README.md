# DAG-rust

`DAG-rust` is an idiomatic Rust overhaul of the FTO-DAG graph construction and
merging core. It is intentionally isolated from the existing Python package.

Initial scope:

- graph-only DAG construction and merging;
- generic DNA/RNA/protein/custom-chain alphabets;
- memory-conscious bit-packed representations;
- deterministic build and merge flow with future similarity-first ordering;
- graph/reference/export metadata for downstream AD-PHMM workflows.

The current scaffold defines module boundaries, public placeholder types, and
smoke tests. Algorithm implementations will be added module by module after
interface review.

Development checks:

```bash
cargo fmt --check
cargo test
cargo clippy --all-targets --all-features
```

