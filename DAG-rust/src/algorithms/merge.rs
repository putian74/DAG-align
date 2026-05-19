//! Pairwise FTO-DAG merge interfaces.

use crate::algorithms::build::SimilarityThreshold;
use crate::foundations::error::Result;
use crate::foundations::id::{GraphId, NodeId};
use crate::graph_model::graph::FtoDag;

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

pub fn merge_graphs(base: FtoDag, _add: FtoDag, _config: MergeConfig) -> Result<FtoDag> {
    Ok(base)
}
