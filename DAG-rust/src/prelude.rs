//! Common public imports for users of `dag_rust`.

pub use crate::algorithms::build::{
    block_plan_from_anchor_path, node_kind_for_path_position, select_bounded_reuse_candidate,
    select_monotone_anchor_path, select_monotone_anchor_path_with_graph, AnchorBlock,
    AnchorCandidate, AnchorCandidateSet, AnchorDecision, AnchorPath, AnchorRejectReason, BlockPlan,
    BuildConfig, BuildOrderingPolicy, BuildReport, BuildSequenceResult, BuildSequenceSummary,
    BuildTimingBreakdown, CoordinateInterval, FtoDagBuilder, IntegrationDecision, PathRange,
    ProfiledSequenceBuildOutcome, ProfiledSequenceBuildSummaryOutcome, RejectedSequence,
    RejectionPolicy, SequenceBuildOutcome, SequenceBuildSummaryOutcome, SimilarityThreshold,
    TopologyUpdateCounters, TopologyUpdateStrategy, UnanchoredBlock,
};
pub use crate::algorithms::merge::{
    merge_graphs, merge_trace_path_graphs_as_count_only, AnchorMap, MergeConfig, MergeDecision,
    MergeOrderingPolicy, MergeOrientation, MergePlan, MergeStats, NodeRemap, RejectedMerge,
};
pub use crate::algorithms::ordering::{
    GraphSketch, OrderingPolicy, PairingPlan, SequenceSketch, SimilarityBucket, SimilaritySketch,
    SketchConfig,
};
pub use crate::algorithms::postprocess::{
    secondary_merge_graph, PostprocessStats, ResolutionConfig, SecondaryMergeConfig,
};
pub use crate::algorithms::scheduler::{
    BuildScheduler, ChunkPlan, MemoryBudget, MergeRoundPlan, ProgressEvent, ProgressSink,
};
pub use crate::foundations::bit_encoding::{BitWidth, NodeFlags, PackedPair, PackedWindow};
pub use crate::foundations::error::{DagError, Result};
pub use crate::foundations::id::{
    ChunkId, GlobalStateId, GraphId, NodeId, ProvenancePosition, RoundId, SequenceId,
    TopologicalCoordinate, Weight,
};
pub use crate::graph_model::graph::{
    EdgeIndexStrategy, EdgeKey, EdgeUpdate, EndpointIndex, FragmentIndex, FtoDag, GraphStats, Node,
    NodeKind, StoredFragmentKey, WeightedEdge,
};
pub use crate::graph_model::provenance::{
    PackedProvenanceRecord, ProvenanceRange, ProvenanceRecord, ProvenanceStorageStrategy,
    ProvenanceTable,
};
pub use crate::graph_model::reference::{
    ExportProfile, InsertionRegion, ReferencePath, ReferencePathStrategy, ReferencePathWeighting,
    StateInterval, SubgraphProjection, TensorExportLayout,
};
pub use crate::graph_model::topology::{
    Children, DagTopology, Parents, ReverseTopologicalOrder, TopologicalOrder, TraversalDirection,
    WeightedChildren, WeightedParents,
};
pub use crate::graph_model::validate::{GraphValidationError, ValidateGraph, ValidationReport};
pub use crate::persistence::storage::{
    GraphFormatVersion, GraphStorage, NativeGraphStorage, StorageConfig,
};
pub use crate::sequence_model::alphabet::{
    Alphabet, AlphabetKind, AmbiguityPolicy, BuiltinAlphabet, CustomAlphabet, NormalizationPolicy,
    SymbolId,
};
pub use crate::sequence_model::fragment::{
    DefaultFragmentEncoder, FragmentEncoder, FragmentKey, FragmentOccurrence, FragmentOccurrences,
    FragmentWindows, PathPositionKind,
};
pub use crate::sequence_model::sequence::{
    EncodedSequence, SequenceInput, SequenceRecord, VecSequenceInput,
};
