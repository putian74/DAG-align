//! Reference/backbone path and global PHMM-coordinate metadata.

use crate::foundations::error::{DagError, Result};
use crate::foundations::id::{GlobalStateId, NodeId};
use crate::graph_model::graph::{EdgeKey, FtoDag};
use crate::graph_model::topology::DagTopology;
use std::collections::HashSet;

const SECONDARY_PATH_PRIMARY_WEIGHT_NUMERATOR: u128 = 4;
const SECONDARY_PATH_PRIMARY_WEIGHT_DENOMINATOR: u128 = 5;

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum ReferencePathStrategy {
    #[default]
    MaxWeightPath,
    StemPath,
    LongestCoordinatePath,
    ExternalPath,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum ReferencePathWeighting {
    #[default]
    EdgeWeightThenNodeWeight,
    WeightedHybrid {
        edge_weight_multiplier: u64,
        node_weight_multiplier: u64,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ReferencePath {
    pub strategy: ReferencePathStrategy,
    pub weighting: ReferencePathWeighting,
    pub nodes: Vec<NodeId>,
}

impl ReferencePath {
    pub fn max_weight(graph: &FtoDag, topology: &DagTopology) -> Result<Self> {
        Self::max_weight_with_weighting(
            graph,
            topology,
            ReferencePathWeighting::EdgeWeightThenNodeWeight,
        )
    }

    pub fn max_weight_with_weighting(
        graph: &FtoDag,
        topology: &DagTopology,
        weighting: ReferencePathWeighting,
    ) -> Result<Self> {
        Ok(Self {
            strategy: ReferencePathStrategy::MaxWeightPath,
            weighting,
            nodes: max_weight_path_nodes(graph, topology, weighting)?,
        })
    }

    pub fn max_weight_pair(graph: &FtoDag, topology: &DagTopology) -> Result<Vec<Self>> {
        Self::max_weight_pair_with_weighting(
            graph,
            topology,
            ReferencePathWeighting::EdgeWeightThenNodeWeight,
        )
    }

    pub fn max_weight_pair_with_order(graph: &FtoDag, order: &[NodeId]) -> Result<Vec<Self>> {
        Self::max_weight_pair_with_weighting_and_order(
            graph,
            order,
            ReferencePathWeighting::EdgeWeightThenNodeWeight,
        )
    }

    pub fn max_weight_pair_with_weighting(
        graph: &FtoDag,
        topology: &DagTopology,
        weighting: ReferencePathWeighting,
    ) -> Result<Vec<Self>> {
        Self::max_weight_pair_with_weighting_and_order(graph, topology.topological_order(), weighting)
    }

    fn max_weight_pair_with_weighting_and_order(
        graph: &FtoDag,
        order: &[NodeId],
        weighting: ReferencePathWeighting,
    ) -> Result<Vec<Self>> {
        let primary = Self {
            strategy: ReferencePathStrategy::MaxWeightPath,
            weighting,
            nodes: max_weight_path_nodes_with_order(graph, order, weighting)?,
        };
        let excluded_nodes: HashSet<NodeId> = primary
            .nodes
            .iter()
            .copied()
            .skip(1)
            .take(primary.nodes.len().saturating_sub(2))
            .collect();
        if excluded_nodes.is_empty() {
            return Ok(vec![primary]);
        }
        if !should_compute_secondary_path(graph, &primary.nodes)? {
            return Ok(vec![primary]);
        }
        let secondary_nodes =
            max_weight_path_nodes_with_order_and_exclusions(graph, order, weighting, &excluded_nodes)?;
        if secondary_nodes.len() < 2 || secondary_nodes == primary.nodes {
            return Ok(vec![primary]);
        }
        Ok(vec![
            primary,
            Self {
                strategy: ReferencePathStrategy::MaxWeightPath,
                weighting,
                nodes: secondary_nodes,
            },
        ])
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct PathScore {
    weighted_sum: u128,
    edge_sum: u128,
    node_sum: u128,
}

impl ReferencePathWeighting {
    fn seed(self, node_weight: u64) -> PathScore {
        let node_sum = u128::from(node_weight);
        PathScore {
            weighted_sum: self.hybrid_increment(0, node_weight),
            edge_sum: 0,
            node_sum,
        }
    }

    fn extend(self, parent: PathScore, edge_weight: u64, node_weight: u64) -> PathScore {
        PathScore {
            weighted_sum: parent.weighted_sum + self.hybrid_increment(edge_weight, node_weight),
            edge_sum: parent.edge_sum + u128::from(edge_weight),
            node_sum: parent.node_sum + u128::from(node_weight),
        }
    }

    fn prefers(self, candidate: PathScore, current: PathScore) -> bool {
        match self {
            Self::EdgeWeightThenNodeWeight => {
                candidate.edge_sum > current.edge_sum
                    || (candidate.edge_sum == current.edge_sum
                        && candidate.node_sum > current.node_sum)
            }
            Self::WeightedHybrid { .. } => {
                candidate.weighted_sum > current.weighted_sum
                    || (candidate.weighted_sum == current.weighted_sum
                        && (candidate.edge_sum > current.edge_sum
                            || (candidate.edge_sum == current.edge_sum
                                && candidate.node_sum > current.node_sum)))
            }
        }
    }

    fn hybrid_increment(self, edge_weight: u64, node_weight: u64) -> u128 {
        match self {
            Self::EdgeWeightThenNodeWeight => 0,
            Self::WeightedHybrid {
                edge_weight_multiplier,
                node_weight_multiplier,
            } => {
                u128::from(edge_weight_multiplier) * u128::from(edge_weight)
                    + u128::from(node_weight_multiplier) * u128::from(node_weight)
            }
        }
    }
}

fn max_weight_path_nodes(
    graph: &FtoDag,
    topology: &DagTopology,
    weighting: ReferencePathWeighting,
) -> Result<Vec<NodeId>> {
    max_weight_path_nodes_with_order(graph, topology.topological_order(), weighting)
}

fn max_weight_path_nodes_with_order(
    graph: &FtoDag,
    order: &[NodeId],
    weighting: ReferencePathWeighting,
) -> Result<Vec<NodeId>> {
    max_weight_path_nodes_with_order_and_exclusions(graph, order, weighting, &HashSet::new())
}

fn max_weight_path_nodes_with_order_and_exclusions(
    graph: &FtoDag,
    order: &[NodeId],
    weighting: ReferencePathWeighting,
    excluded_nodes: &HashSet<NodeId>,
) -> Result<Vec<NodeId>> {
    let mut scores = vec![PathScore::default(); graph.node_count()];
    let mut previous = vec![None; graph.node_count()];
    let mut reachable = vec![false; graph.node_count()];

    for node_id in order.iter().copied() {
        if excluded_nodes.contains(&node_id) {
            continue;
        }
        let node_weight = graph.node(node_id)?.weight.raw();
        let mut best_score = PathScore::default();
        let mut best_parent = None;
        let parents = graph.parents(node_id)?;
        let mut found_parent = false;
        for parent in parents.iter().copied() {
            if excluded_nodes.contains(&parent) || !reachable[parent.to_usize()] {
                continue;
            }
            let edge_weight = graph
                .edge_weight(EdgeKey {
                    parent,
                    child: node_id,
                })
                .ok_or(DagError::InvalidEdge {
                    parent: parent.to_usize(),
                    child: node_id.to_usize(),
                })?
                .raw();
            let candidate_score =
                weighting.extend(scores[parent.to_usize()], edge_weight, node_weight);
            if !found_parent
                || weighting.prefers(candidate_score, best_score)
                || (candidate_score == best_score
                    && best_parent.is_none_or(|current| parent < current))
            {
                best_score = candidate_score;
                best_parent = Some(parent);
                found_parent = true;
            }
        }
        if found_parent {
            reachable[node_id.to_usize()] = true;
        } else if parents.is_empty() {
            best_score = weighting.seed(node_weight);
            reachable[node_id.to_usize()] = true;
        } else {
            continue;
        }
        scores[node_id.to_usize()] = best_score;
        previous[node_id.to_usize()] = best_parent;
    }

    let mut best_end = None;
    for node_id in graph.endpoints().structural_sinks().iter().copied() {
        if !reachable[node_id.to_usize()] {
            continue;
        }
        let score = scores[node_id.to_usize()];
        if best_end.is_none_or(|current: NodeId| {
            weighting.prefers(score, scores[current.to_usize()])
                || (score == scores[current.to_usize()] && node_id < current)
        }) {
            best_end = Some(node_id);
        }
    }

    let mut best_any = None;
    if best_end.is_none() {
        for node_id in order.iter().copied() {
            if !reachable[node_id.to_usize()] {
                continue;
            }
            let score = scores[node_id.to_usize()];
            if best_any.is_none_or(|current: NodeId| {
                weighting.prefers(score, scores[current.to_usize()])
                    || (score == scores[current.to_usize()] && node_id < current)
            }) {
                best_any = Some(node_id);
            }
        }
    }

    let mut cursor = best_end.or(best_any);
    let mut path = Vec::new();
    while let Some(node_id) = cursor {
        path.push(node_id);
        cursor = previous[node_id.to_usize()];
    }
    path.reverse();
    Ok(path)
}

fn should_compute_secondary_path(graph: &FtoDag, primary_nodes: &[NodeId]) -> Result<bool> {
    let total_weight = graph
        .nodes()
        .iter()
        .fold(0_u128, |sum, node| sum + u128::from(node.weight.raw()));
    if total_weight == 0 {
        return Ok(false);
    }
    let primary_weight = primary_nodes.iter().try_fold(0_u128, |sum, node_id| {
        Ok::<_, DagError>(sum + u128::from(graph.node(*node_id)?.weight.raw()))
    })?;
    Ok(
        primary_weight * SECONDARY_PATH_PRIMARY_WEIGHT_DENOMINATOR
            < total_weight * SECONDARY_PATH_PRIMARY_WEIGHT_NUMERATOR,
    )
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
    CoordinateWindow,
    Diagnostics,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum TensorExportLayout {
    ArrayDirectory,
    SingleContainer,
}

#[cfg(test)]
mod tests {
    use super::{ReferencePath, ReferencePathWeighting};
    use crate::foundations::id::Weight;
    use crate::graph_model::graph::{FtoDag, NodeKind};
    use crate::graph_model::topology::DagTopology;
    use crate::sequence_model::alphabet::SymbolId;
    use crate::sequence_model::fragment::FragmentKey;

    fn key(raw: u16) -> FragmentKey {
        FragmentKey::symbols(vec![SymbolId::new(raw)])
    }

    fn build_branch_graph(
        left_node_weight: u64,
        right_node_weight: u64,
        left_edge_weight: u64,
        right_edge_weight: u64,
    ) -> (FtoDag, [crate::foundations::id::NodeId; 4]) {
        let mut graph = FtoDag::with_provenance_storage(
            1,
            crate::graph_model::provenance::ProvenanceStorageStrategy::CountOnly,
        );
        let start = graph.add_node(key(0), NodeKind::Start).unwrap();
        let left = graph.add_node(key(1), NodeKind::Internal).unwrap();
        let right = graph.add_node(key(2), NodeKind::Internal).unwrap();
        let end = graph.add_node(key(3), NodeKind::End).unwrap();

        graph.add_provenance_count(start, 1).unwrap();
        graph.add_provenance_count(left, left_node_weight).unwrap();
        graph
            .add_provenance_count(right, right_node_weight)
            .unwrap();
        graph.add_provenance_count(end, 1).unwrap();

        graph
            .add_or_increment_edge(start, left, Weight::new(left_edge_weight))
            .unwrap();
        graph
            .add_or_increment_edge(left, end, Weight::new(left_edge_weight))
            .unwrap();
        graph
            .add_or_increment_edge(start, right, Weight::new(right_edge_weight))
            .unwrap();
        graph
            .add_or_increment_edge(right, end, Weight::new(right_edge_weight))
            .unwrap();

        (graph, [start, left, right, end])
    }

    #[test]
    fn max_weight_path_prefers_edge_weight_before_node_weight() {
        let (graph, [start, left, right, end]) = build_branch_graph(1, 10, 10, 3);
        let topology = DagTopology::try_from_graph(&graph).unwrap();

        let path = ReferencePath::max_weight(&graph, &topology).unwrap();

        assert_eq!(
            path.weighting,
            ReferencePathWeighting::EdgeWeightThenNodeWeight
        );
        assert_eq!(path.nodes, vec![start, left, end]);
        assert_ne!(path.nodes, vec![start, right, end]);
    }

    #[test]
    fn max_weight_path_breaks_edge_ties_with_node_weight() {
        let (graph, [start, _, right, end]) = build_branch_graph(2, 6, 5, 5);
        let topology = DagTopology::try_from_graph(&graph).unwrap();

        let path = ReferencePath::max_weight(&graph, &topology).unwrap();

        assert_eq!(path.nodes, vec![start, right, end]);
    }

    #[test]
    fn weighted_hybrid_can_override_edge_first_choice() {
        let (graph, [start, _, right, end]) = build_branch_graph(1, 10, 10, 3);
        let topology = DagTopology::try_from_graph(&graph).unwrap();

        let path = ReferencePath::max_weight_with_weighting(
            &graph,
            &topology,
            ReferencePathWeighting::WeightedHybrid {
                edge_weight_multiplier: 1,
                node_weight_multiplier: 10,
            },
        )
        .unwrap();

        assert_eq!(path.nodes, vec![start, right, end]);
    }

    #[test]
    fn max_weight_path_breaks_full_ties_by_smallest_node_id() {
        let (graph, [start, left, _, end]) = build_branch_graph(4, 4, 5, 5);
        let topology = DagTopology::try_from_graph(&graph).unwrap();

        let path = ReferencePath::max_weight(&graph, &topology).unwrap();

        assert_eq!(path.nodes, vec![start, left, end]);
    }

    #[test]
    fn max_weight_pair_returns_primary_then_secondary_branch() {
        let (graph, [start, left, right, end]) = build_branch_graph(3, 2, 9, 7);
        let topology = DagTopology::try_from_graph(&graph).unwrap();

        let paths = ReferencePath::max_weight_pair(&graph, &topology).unwrap();

        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0].nodes, vec![start, left, end]);
        assert_eq!(paths[1].nodes, vec![start, right, end]);
    }

    #[test]
    fn max_weight_pair_omits_secondary_when_no_alternative_path_exists() {
        let mut graph = FtoDag::with_provenance_storage(
            1,
            crate::graph_model::provenance::ProvenanceStorageStrategy::CountOnly,
        );
        let start = graph.add_node(key(0), NodeKind::Start).unwrap();
        let mid = graph.add_node(key(1), NodeKind::Internal).unwrap();
        let end = graph.add_node(key(2), NodeKind::End).unwrap();
        graph.add_provenance_count(start, 1).unwrap();
        graph.add_provenance_count(mid, 5).unwrap();
        graph.add_provenance_count(end, 1).unwrap();
        graph
            .add_or_increment_edge(start, mid, Weight::new(4))
            .unwrap();
        graph
            .add_or_increment_edge(mid, end, Weight::new(4))
            .unwrap();

        let topology = DagTopology::try_from_graph(&graph).unwrap();
        let paths = ReferencePath::max_weight_pair(&graph, &topology).unwrap();

        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].nodes, vec![start, mid, end]);
    }

    #[test]
    fn max_weight_pair_skips_secondary_when_primary_dominates_total_weight() {
        let (graph, [start, left, _, end]) = build_branch_graph(98, 1, 8, 7);
        let topology = DagTopology::try_from_graph(&graph).unwrap();

        let paths = ReferencePath::max_weight_pair(&graph, &topology).unwrap();

        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0].nodes, vec![start, left, end]);
    }
}
