//! Resolution adjustment, atomization, and secondary merge interfaces.

use crate::foundations::error::{DagError, Result};
use crate::foundations::id::{NodeId, ProvenancePosition, SequenceId, TopologicalCoordinate};
use crate::graph_model::graph::{FtoDag, NodeKind, StoredFragmentKey};
use crate::graph_model::provenance::{ProvenanceRecord, ProvenanceStorageStrategy};
use crate::graph_model::topology::{DagTopology, GraphCoordinateSnapshot};
use smallvec::SmallVec;
use std::collections::HashMap;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ResolutionConfig {
    pub target_fragment_len: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SecondaryMergeConfig {
    pub use_forward_coordinates: bool,
    pub use_reverse_coordinates: bool,
}

impl Default for SecondaryMergeConfig {
    fn default() -> Self {
        Self {
            use_forward_coordinates: true,
            use_reverse_coordinates: true,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct PostprocessStats {
    pub removed_nodes: usize,
    pub merged_nodes: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
struct SecondaryCoordinateKey {
    forward_coordinate: Option<u64>,
    reverse_coordinate: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct LegacySecondaryMergeKey {
    fragment: StoredFragmentKey,
    kind: NodeKind,
    forward_coordinate: Option<u64>,
    reverse_coordinate: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SecondaryMergeCoordinateSeed {
    pub order: Vec<NodeId>,
    pub forward_coordinates: Vec<TopologicalCoordinate>,
    pub reverse_coordinates: Vec<TopologicalCoordinate>,
}

impl SecondaryMergeCoordinateSeed {
    pub(crate) fn from_snapshot(snapshot: &GraphCoordinateSnapshot) -> Self {
        Self {
            order: snapshot.topological_order().to_vec(),
            forward_coordinates: snapshot.forward_coordinates().to_vec(),
            reverse_coordinates: snapshot.reverse_coordinates().to_vec(),
        }
    }
}

pub fn secondary_merge_graph(graph: FtoDag, config: SecondaryMergeConfig) -> Result<FtoDag> {
    let (graph, _) = secondary_merge_graph_with_stats_and_seed(graph, config, None)?;
    Ok(graph)
}

pub fn secondary_merge_graph_with_stats(
    graph: FtoDag,
    config: SecondaryMergeConfig,
) -> Result<(FtoDag, PostprocessStats)> {
    secondary_merge_graph_with_stats_and_seed(graph, config, None)
}

pub(crate) fn secondary_merge_graph_with_coordinate_seed(
    graph: FtoDag,
    config: SecondaryMergeConfig,
    coordinate_seed: SecondaryMergeCoordinateSeed,
) -> Result<FtoDag> {
    let (graph, _) =
        secondary_merge_graph_with_stats_and_seed(graph, config, Some(coordinate_seed))?;
    Ok(graph)
}

fn secondary_merge_graph_with_stats_and_seed(
    mut graph: FtoDag,
    config: SecondaryMergeConfig,
    coordinate_seed: Option<SecondaryMergeCoordinateSeed>,
) -> Result<(FtoDag, PostprocessStats)> {
    if graph.provenance_storage_strategy() != ProvenanceStorageStrategy::CountOnly {
        return secondary_merge_graph_legacy(graph, config);
    }
    let pass_configs = secondary_merge_pass_configs(config);
    if pass_configs.is_empty() {
        graph.compact_storage()?;
        return Ok((graph, PostprocessStats::default()));
    }

    let mut total_stats = PostprocessStats::default();
    let mut reusable_seed = coordinate_seed;
    let mut reusable_snapshot: Option<GraphCoordinateSnapshot> = None;
    for pass_config in pass_configs.iter() {
        let (order, forward_coordinates, reverse_coordinates) =
            if let Some(seed) = reusable_seed.as_ref() {
                (
                    seed.order.as_slice(),
                    pass_config
                        .use_forward_coordinates
                        .then_some(seed.forward_coordinates.as_slice()),
                    pass_config
                        .use_reverse_coordinates
                        .then_some(seed.reverse_coordinates.as_slice()),
                )
            } else {
                if reusable_snapshot.is_none() {
                    reusable_snapshot = Some(GraphCoordinateSnapshot::from_graph(&graph)?);
                }
                let coordinates = reusable_snapshot
                    .as_ref()
                    .expect("snapshot should exist after construction");
                (
                    coordinates.topological_order(),
                    pass_config
                        .use_forward_coordinates
                        .then_some(coordinates.forward_coordinates()),
                    pass_config
                        .use_reverse_coordinates
                        .then_some(coordinates.reverse_coordinates()),
                )
            };
        let Some(remap) =
            build_secondary_merge_remap(&graph, order, forward_coordinates, reverse_coordinates)?
        else {
            continue;
        };
        let merged_nodes = remap
            .iter()
            .enumerate()
            .filter(|(index, representative)| representative.to_usize() != *index)
            .count();
        let removed_nodes = merged_nodes;
        graph = rebuild_graph_with_remap(&graph, order, &remap)?;
        total_stats.merged_nodes += merged_nodes;
        total_stats.removed_nodes += removed_nodes;
        reusable_seed = None;
        reusable_snapshot = None;
    }
    graph.compact_storage()?;
    Ok((graph, total_stats))
}

fn secondary_merge_graph_legacy(
    mut graph: FtoDag,
    config: SecondaryMergeConfig,
) -> Result<(FtoDag, PostprocessStats)> {
    if !config.use_forward_coordinates && !config.use_reverse_coordinates {
        graph.compact_storage()?;
        return Ok((graph, PostprocessStats::default()));
    }
    if graph.provenance_storage_strategy() == ProvenanceStorageStrategy::CountOnly {
        graph.compact_storage()?;
        return Ok((graph, PostprocessStats::default()));
    }

    let mut total_stats = PostprocessStats::default();
    loop {
        let topology = DagTopology::try_from_graph(&graph)?;
        let Some(remap) = build_legacy_secondary_merge_remap(&graph, &topology, config)? else {
            graph.compact_storage()?;
            return Ok((graph, total_stats));
        };
        let merged_nodes = remap
            .iter()
            .enumerate()
            .filter(|(index, representative)| representative.to_usize() != *index)
            .count();
        let removed_nodes = merged_nodes;
        graph = rebuild_graph_with_remap(&graph, topology.topological_order(), &remap)?;
        total_stats.merged_nodes += merged_nodes;
        total_stats.removed_nodes += removed_nodes;
    }
}

fn build_legacy_secondary_merge_remap(
    graph: &FtoDag,
    topology: &DagTopology,
    config: SecondaryMergeConfig,
) -> Result<Option<Vec<NodeId>>> {
    let mut remap = vec![NodeId::new(0); graph.node_count()];
    let mut representatives = HashMap::<LegacySecondaryMergeKey, NodeId>::new();

    for node_id in topology.topological_order().iter().copied() {
        let node = graph.node(node_id)?;
        let key = LegacySecondaryMergeKey {
            fragment: node.fragment.clone(),
            kind: node.kind,
            forward_coordinate: if config.use_forward_coordinates {
                Some(u64::from(topology.forward_coordinate(node_id)?.raw()))
            } else {
                None
            },
            reverse_coordinate: if config.use_reverse_coordinates {
                Some(u64::from(topology.reverse_coordinate(node_id)?.raw()))
            } else {
                None
            },
        };
        let representative = representatives.entry(key).or_insert(node_id);
        remap[node_id.to_usize()] = *representative;
    }

    let merged_any = remap
        .iter()
        .enumerate()
        .any(|(index, representative)| representative.to_usize() != index);
    Ok(merged_any.then_some(remap))
}

fn secondary_merge_pass_configs(config: SecondaryMergeConfig) -> Vec<SecondaryMergeConfig> {
    match (
        config.use_forward_coordinates,
        config.use_reverse_coordinates,
    ) {
        (false, false) => Vec::new(),
        (true, false) => vec![SecondaryMergeConfig {
            use_forward_coordinates: true,
            use_reverse_coordinates: false,
        }],
        (false, true) => vec![SecondaryMergeConfig {
            use_forward_coordinates: false,
            use_reverse_coordinates: true,
        }],
        (true, true) => vec![
            SecondaryMergeConfig {
                use_forward_coordinates: true,
                use_reverse_coordinates: false,
            },
            SecondaryMergeConfig {
                use_forward_coordinates: false,
                use_reverse_coordinates: true,
            },
        ],
    }
}

fn build_secondary_merge_remap(
    graph: &FtoDag,
    order: &[NodeId],
    forward_coordinates: Option<&[TopologicalCoordinate]>,
    reverse_coordinates: Option<&[TopologicalCoordinate]>,
) -> Result<Option<Vec<NodeId>>> {
    validate_secondary_merge_coordinates(graph, forward_coordinates, reverse_coordinates)?;

    let mut remap = (0..graph.node_count())
        .map(NodeId::try_from)
        .collect::<Result<Vec<_>>>()?;
    let mut order_rank = vec![usize::MAX; graph.node_count()];
    for (rank, node_id) in order.iter().copied().enumerate() {
        order_rank[node_id.to_usize()] = rank;
    }

    let mut groups = HashMap::<SecondaryCoordinateKey, SmallVec<[NodeId; 2]>>::new();
    let mut merged_any = false;
    graph.fragment_index().for_each_bucket(|_kind, node_ids| {
        if node_ids.len() < 2 {
            return;
        }

        groups.clear();
        for node_id in node_ids.iter().copied() {
            let key = secondary_coordinate_key(node_id, forward_coordinates, reverse_coordinates);
            groups.entry(key).or_default().push(node_id);
        }

        for grouped_node_ids in groups.values() {
            if grouped_node_ids.len() < 2 {
                continue;
            }
            let mut representative = grouped_node_ids[0];
            let mut representative_rank = order_rank[representative.to_usize()];
            for node_id in grouped_node_ids.iter().copied().skip(1) {
                let rank = order_rank[node_id.to_usize()];
                if rank < representative_rank {
                    representative = node_id;
                    representative_rank = rank;
                }
            }
            for node_id in grouped_node_ids.iter().copied() {
                remap[node_id.to_usize()] = representative;
            }
            merged_any = true;
        }
    });

    Ok(merged_any.then_some(remap))
}

fn validate_secondary_merge_coordinates(
    graph: &FtoDag,
    forward_coordinates: Option<&[TopologicalCoordinate]>,
    reverse_coordinates: Option<&[TopologicalCoordinate]>,
) -> Result<()> {
    for coordinates in [forward_coordinates, reverse_coordinates]
        .into_iter()
        .flatten()
    {
        if coordinates.len() != graph.node_count() {
            return Err(DagError::InvalidStorage(format!(
                "secondary merge coordinate vector length {} did not match node count {}",
                coordinates.len(),
                graph.node_count()
            )));
        }
    }
    Ok(())
}

fn secondary_coordinate_key(
    node_id: NodeId,
    forward_coordinates: Option<&[TopologicalCoordinate]>,
    reverse_coordinates: Option<&[TopologicalCoordinate]>,
) -> SecondaryCoordinateKey {
    SecondaryCoordinateKey {
        forward_coordinate: coordinate_value(forward_coordinates, node_id),
        reverse_coordinate: coordinate_value(reverse_coordinates, node_id),
    }
}

fn coordinate_value(coordinates: Option<&[TopologicalCoordinate]>, node_id: NodeId) -> Option<u64> {
    coordinates.map(|coordinates| u64::from(coordinates[node_id.to_usize()].raw()))
}

fn rebuild_graph_with_remap(graph: &FtoDag, order: &[NodeId], remap: &[NodeId]) -> Result<FtoDag> {
    if graph.provenance_storage_strategy() == ProvenanceStorageStrategy::CountOnly {
        return rebuild_count_only_graph_with_remap(graph, order, remap);
    }

    let mut rebuilt = FtoDag::with_provenance_and_edge_storage(
        graph.fragment_len(),
        graph.provenance_storage_strategy(),
        graph.edge_index_strategy(),
    );
    let mut representative_new_ids = vec![None; graph.node_count()];
    let mut new_remap = vec![NodeId::new(0); graph.node_count()];

    for node_id in order.iter().copied() {
        let representative = remap[node_id.to_usize()];
        let new_id = if let Some(existing) = representative_new_ids[representative.to_usize()] {
            existing
        } else {
            let node = graph.node(representative)?;
            let created = rebuilt.add_node(node.fragment.clone(), node.kind)?;
            representative_new_ids[representative.to_usize()] = Some(created);
            created
        };
        new_remap[node_id.to_usize()] = new_id;
    }

    for edge in graph.edges() {
        let parent = new_remap[edge.key.parent.to_usize()];
        let child = new_remap[edge.key.child.to_usize()];
        if parent == child {
            continue;
        }
        rebuilt.add_or_increment_edge(parent, child, edge.weight)?;
    }

    match graph.provenance_storage_strategy() {
        ProvenanceStorageStrategy::FullRecords | ProvenanceStorageStrategy::Packed32 => {
            for node in graph.nodes() {
                let target = new_remap[node.id.to_usize()];
                for record in graph.provenance_records(node.id)? {
                    rebuilt.add_provenance_record(target, record)?;
                }
            }
        }
        ProvenanceStorageStrategy::TracePaths => {
            let mut paths = graph.sequence_trace_paths()?;
            while let Some((sequence_index, path)) = paths.next_path()? {
                let sequence_id = SequenceId::try_from(sequence_index)?;
                for (position, node_id) in path.iter().copied().enumerate() {
                    rebuilt.add_provenance_record(
                        new_remap[node_id.to_usize()],
                        ProvenanceRecord {
                            sequence_id,
                            position: ProvenancePosition::new(position as u64),
                        },
                    )?;
                }
            }
        }
        ProvenanceStorageStrategy::CountOnly => {
            for node in graph.nodes() {
                rebuilt.add_provenance_count(new_remap[node.id.to_usize()], node.weight.raw())?;
            }
        }
    }

    Ok(rebuilt)
}

fn rebuild_count_only_graph_with_remap(
    graph: &FtoDag,
    order: &[NodeId],
    remap: &[NodeId],
) -> Result<FtoDag> {
    let mut rebuilt = FtoDag::with_provenance_and_edge_storage(
        graph.fragment_len(),
        graph.provenance_storage_strategy(),
        graph.edge_index_strategy(),
    );
    let mut representative_new_ids = vec![None; graph.node_count()];
    let mut new_remap = vec![NodeId::new(0); graph.node_count()];

    for node_id in order.iter().copied() {
        let representative = remap[node_id.to_usize()];
        let new_id = if let Some(existing) = representative_new_ids[representative.to_usize()] {
            existing
        } else {
            let node = graph.node(representative)?;
            let created = rebuilt.add_node(node.fragment.clone(), node.kind)?;
            representative_new_ids[representative.to_usize()] = Some(created);
            created
        };
        new_remap[node_id.to_usize()] = new_id;
    }

    for edge in graph.edges() {
        let parent = new_remap[edge.key.parent.to_usize()];
        let child = new_remap[edge.key.child.to_usize()];
        if parent == child {
            continue;
        }
        rebuilt.add_or_increment_edge(parent, child, edge.weight)?;
    }

    let mut node_weights = vec![0_u64; rebuilt.node_count()];
    for node in graph.nodes() {
        node_weights[new_remap[node.id.to_usize()].to_usize()] += node.weight.raw();
    }
    for (index, weight) in node_weights.into_iter().enumerate() {
        if weight == 0 {
            continue;
        }
        rebuilt.add_provenance_count(NodeId::try_from(index)?, weight)?;
    }

    Ok(rebuilt)
}
