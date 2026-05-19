//! Reference/backbone path and global PHMM-coordinate metadata.

use crate::foundations::id::{GlobalStateId, NodeId};

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum ReferencePathStrategy {
    #[default]
    MaxWeightPath,
    StemPath,
    LongestCoordinatePath,
    ExternalPath,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReferencePath {
    pub strategy: ReferencePathStrategy,
    pub nodes: Vec<NodeId>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct StateInterval {
    pub start: GlobalStateId,
    pub end: GlobalStateId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InsertionRegion {
    pub left: GlobalStateId,
    pub right: GlobalStateId,
    pub support: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SubgraphProjection {
    pub local_to_global_nodes: Vec<(NodeId, NodeId)>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ExportProfile {
    GraphOnly,
    GraphWithReference,
    AdPhmmTraining,
    CoordinateWindow,
    Diagnostics,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum TensorExportLayout {
    ArrayDirectory,
    SingleContainer,
}
