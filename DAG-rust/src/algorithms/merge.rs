//! Pairwise FTO-DAG merge interfaces.

use crate::algorithms::build::{
    BuildConfig, FtoDagBuilder, SimilarityThreshold, TopologyUpdateStrategy,
};
use crate::foundations::error::Result;
use crate::foundations::id::{GraphId, NodeId};
use crate::graph_model::graph::FtoDag;
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

pub fn merge_graphs(base: FtoDag, add: FtoDag, config: MergeConfig) -> Result<FtoDag> {
    if base.fragment_len() != add.fragment_len() {
        return Err(crate::foundations::error::DagError::InvalidStorage(
            format!(
                "cannot merge graphs with fragment lengths {} and {}",
                base.fragment_len(),
                add.fragment_len()
            ),
        ));
    }
    let base_sequence_offset = base.sequence_trace_paths()?.len();
    let add_trace_paths = add.sequence_trace_paths()?;

    let mut build_config = BuildConfig::new(base.fragment_len());
    build_config.min_initial_match_ratio = config.min_initial_anchor_ratio;
    build_config.provenance_storage_strategy = base.provenance_storage_strategy();
    build_config.edge_index_strategy = base.edge_index_strategy();
    build_config.topology_update_strategy = TopologyUpdateStrategy::IncrementalForwardRelaxation;
    let mut builder = FtoDagBuilder::from_graph(base, build_config)?;

    for (sequence_offset, trace_path) in add_trace_paths.iter().enumerate() {
        if trace_path.is_empty() {
            continue;
        }
        let sequence_id = crate::foundations::id::SequenceId::try_from(
            base_sequence_offset.saturating_add(sequence_offset),
        )?;
        let occurrences = occurrences_from_trace_path(&add, trace_path)?;
        builder.add_sequence_from_occurrences(sequence_id, occurrences)?;
    }

    Ok(builder.into_graph())
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
            key: node.fragment.clone(),
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
