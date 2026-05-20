//! Pairwise FTO-DAG merge interfaces.

use crate::algorithms::build::{
    BuildConfig, FtoDagBuilder, SimilarityThreshold, TopologyUpdateStrategy,
};
use crate::foundations::error::{DagError, Result};
use crate::foundations::id::{GraphId, NodeId};
use crate::graph_model::graph::FtoDag;
use crate::graph_model::provenance::ProvenanceStorageStrategy;
use crate::sequence_model::fragment::{FragmentOccurrence, PathPositionKind};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MergeOrderingPolicy {
    DeterministicBinary,
    SketchBucketedSimilarity,
    UserProvided,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MergeConfig {
    pub ordering_policy: MergeOrderingPolicy,
    pub min_initial_anchor_ratio: Option<SimilarityThreshold>,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            ordering_policy: MergeOrderingPolicy::DeterministicBinary,
            min_initial_anchor_ratio: None,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MergeOrientation {
    LeftIntoRight,
    RightIntoLeft,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AnchorMap {
    pub pairs: Vec<(NodeId, NodeId)>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NodeRemap {
    pub pairs: Vec<(NodeId, NodeId)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MergePlan {
    pub orientation: MergeOrientation,
    pub anchors: AnchorMap,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct MergeStats {
    pub matched_nodes: usize,
    pub added_nodes: usize,
    pub added_edges: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct RejectedMerge {
    pub base_graph_id: GraphId,
    pub add_graph_id: GraphId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MergeDecision {
    Merge(MergePlan),
    Reject(RejectedMerge),
}

pub fn merge_graphs(left: FtoDag, right: FtoDag, config: MergeConfig) -> Result<FtoDag> {
    if left.fragment_len() != right.fragment_len() {
        return Err(crate::foundations::error::DagError::InvalidStorage(
            format!(
                "cannot merge graphs with fragment lengths {} and {}",
                left.fragment_len(),
                right.fragment_len()
            ),
        ));
    }
    if left.provenance_storage_strategy() != right.provenance_storage_strategy() {
        return Err(crate::foundations::error::DagError::InvalidStorage(
            format!(
                "cannot merge graphs with different provenance storage {:?} and {:?}",
                left.provenance_storage_strategy(),
                right.provenance_storage_strategy()
            ),
        ));
    }
    match left.provenance_storage_strategy() {
        ProvenanceStorageStrategy::TracePaths => replay_trace_path_graphs(left, right, config),
        ProvenanceStorageStrategy::CountOnly => merge_count_only_graphs(left, right),
        ProvenanceStorageStrategy::FullRecords | ProvenanceStorageStrategy::Packed32 => {
            Err(DagError::UnsupportedOperation(
                "merge_graphs currently supports TracePaths and CountOnly provenance storage",
            ))
        }
    }
}

pub fn merge_trace_path_graphs_as_count_only(
    left: FtoDag,
    right: FtoDag,
    config: MergeConfig,
) -> Result<FtoDag> {
    if left.provenance_storage_strategy() != ProvenanceStorageStrategy::TracePaths
        || right.provenance_storage_strategy() != ProvenanceStorageStrategy::TracePaths
    {
        return Err(DagError::InvalidStorage(
            "merge_trace_path_graphs_as_count_only requires TracePaths inputs".to_string(),
        ));
    }
    merge_graphs(left, right, config)?.to_count_only()
}

fn replay_trace_path_graphs(base: FtoDag, add: FtoDag, config: MergeConfig) -> Result<FtoDag> {
    let base_sequence_offset = base.sequence_trace_path_count()?;

    let mut build_config = BuildConfig::new(base.fragment_len());
    build_config.min_initial_match_ratio = config.min_initial_anchor_ratio;
    build_config.provenance_storage_strategy = base.provenance_storage_strategy();
    build_config.edge_index_strategy = base.edge_index_strategy();
    build_config.topology_update_strategy = TopologyUpdateStrategy::IncrementalForwardRelaxation;
    let mut builder = FtoDagBuilder::from_graph(base, build_config)?;

    let mut trace_paths = add.sequence_trace_paths()?;
    while let Some((sequence_offset, trace_path)) = trace_paths.next_path()? {
        if trace_path.is_empty() {
            continue;
        }
        let sequence_id = crate::foundations::id::SequenceId::try_from(
            base_sequence_offset.saturating_add(sequence_offset),
        )?;
        let occurrences = occurrences_from_trace_path(&add, trace_path)?;
        builder.add_sequence_from_occurrences(sequence_id, occurrences)?;
    }

    builder.finalize_graph()
}

fn merge_count_only_graphs(_left: FtoDag, _right: FtoDag) -> Result<FtoDag> {
    Err(DagError::UnsupportedOperation(
        "native CountOnly graph merge is a stub until skeleton/corridor merge sketches are implemented",
    ))
}

fn occurrences_from_trace_path(
    graph: &FtoDag,
    trace_path: &[NodeId],
) -> Result<Vec<FragmentOccurrence>> {
    let mut occurrences = Vec::with_capacity(trace_path.len());
    for (position, node_id) in trace_path.iter().copied().enumerate() {
        let node = graph.node(node_id)?;
        occurrences.push(FragmentOccurrence {
            position,
            kind: path_position_kind(position, trace_path.len()),
            key: node.fragment.to_fragment_key(),
        });
    }
    Ok(occurrences)
}

fn path_position_kind(position: usize, window_count: usize) -> PathPositionKind {
    if window_count == 1 {
        PathPositionKind::Singleton
    } else if position == 0 {
        PathPositionKind::Start
    } else if position + 1 == window_count {
        PathPositionKind::End
    } else {
        PathPositionKind::Internal
    }
}
