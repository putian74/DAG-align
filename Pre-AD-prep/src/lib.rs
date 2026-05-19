//! Rust-first preprocessing contracts for AD-PHMM-align.
//!
//! The crate owns graph conversion, coordinate/window construction,
//! initialization artifacts, validation, diagnostics, and typed exports.

pub mod coordinates;
pub mod diagnostics;
pub mod error;
pub mod export;
pub mod graph;
pub mod init;
pub mod legacy;
pub mod source;
pub mod validate;

pub use coordinates::{
    CoordinateMode, EdgeWindowOverlap, EdgeWindowOverlaps, GlobalCoordinateOutput,
    PackedStateWindows, ReferencePath, StateInterval, StateProjection, WindowBuildConfig,
    build_edge_window_overlaps, build_global_coordinates, build_packed_windows,
};
pub use diagnostics::{ConversionDiagnostics, Diagnostic, DiagnosticReport, ProfilingSummary};
pub use error::{PreAdPrepError, Result};
pub use export::{
    ArraySpec, DataType, SourceFormat, StateIntervalSemantics, TensorGraphArtifact,
    TensorGraphManifest, write_npy_1d, write_npy_2d,
};
pub use graph::{AdjacencyCsr, NodeFlags, TensorGraph, TopologicalLevels};
pub use init::{
    InitialPhmmManifest, InitializationBundle, InitializationTrack, TRANSITION_ORDER,
    write_initialization_track,
};
pub use legacy::{
    LegacyAdapter, LegacyConversionOptions, LegacyDagAlignAdapter, LegacyDagAlignInput,
};
pub use source::{
    NodeSourceRanges, SequenceTable, SourceDecodeStatus, SourceRecordTable, SourceTables,
};
pub use validate::{
    ArtifactValidationLevel, Validate, ValidationIssue, ValidationReport, ValidationSeverity,
    validate_tensor_graph_artifact,
};
