//! DAG-rust: graph construction and export core for FTO-DAG workflows.
//!
//! # Implementation rules
//!
//! 1. Do not modify the existing Python package. This crate is isolated under
//!    `DAG-rust/`.
//! 2. This is an idiomatic Rust overhaul, not a line-by-line Python translation.
//! 3. Memory efficiency is a first-class design goal. Bit-packed and tiered
//!    representations must be considered before large data structures are
//!    finalized.
//! 4. Downstream AD-PHMM export requires globally stable reference/state
//!    coordinates.
//! 5. Construction and merging are deterministic by default, with optional
//!    similarity-first ordering policies added after validation.

pub mod algorithms;
pub mod foundations;
pub mod graph_model;
pub mod interfaces;
pub mod persistence;
pub mod prelude;
pub mod sequence_model;
