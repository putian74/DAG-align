//! Resolution adjustment, atomization, and secondary merge interfaces.

use crate::foundations::error::Result;
use crate::foundations::id::{NodeId, ProvenancePosition, SequenceId};
use crate::graph_model::graph::{FtoDag, NodeKind};
use crate::graph_model::provenance::{ProvenanceRecord, ProvenanceStorageStrategy};
use crate::graph_model::topology::DagTopology;
use crate::sequence_model::fragment::FragmentKey;
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

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct SecondaryMergeKey {
    fragment: FragmentKey,
    kind: NodeKind,
    forward_coordinate: Option<u64>,
    reverse_coordinate: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum SecondaryMergeBucketKey {
    Shared(SecondaryMergeKey),
}

pub fn secondary_merge_graph(graph: FtoDag, config: SecondaryMergeConfig) -> Result<FtoDag> {
    let (graph, _) = secondary_merge_graph_with_stats(graph, config)?;
    Ok(graph)
}

pub fn secondary_merge_graph_with_stats(
    mut graph: FtoDag,
    config: SecondaryMergeConfig,
) -> Result<(FtoDag, PostprocessStats)> {
    if !config.use_forward_coordinates && !config.use_reverse_coordinates {
        graph.compact_storage()?;
        return Ok((graph, PostprocessStats::default()));
    }
    if matches!(
        graph.provenance_storage_strategy(),
        ProvenanceStorageStrategy::CountOnly
    ) {
        graph.compact_storage()?;
        return Ok((graph, PostprocessStats::default()));
    }

    let mut total_stats = PostprocessStats::default();
    loop {
        let topology = DagTopology::try_from_graph(&graph)?;
        let Some(remap) = build_secondary_merge_remap(&graph, &topology, config)? else {
            graph.compact_storage()?;
            return Ok((graph, total_stats));
        };
        let merged_nodes = remap
            .iter()
            .enumerate()
            .filter(|(index, representative)| representative.to_usize() != *index)
            .count();
        let removed_nodes = merged_nodes;
        graph = rebuild_graph_with_remap(&graph, &topology, &remap)?;
        total_stats.merged_nodes += merged_nodes;
        total_stats.removed_nodes += removed_nodes;
    }
}

fn build_secondary_merge_remap(
    graph: &FtoDag,
    topology: &DagTopology,
    config: SecondaryMergeConfig,
) -> Result<Option<Vec<NodeId>>> {
    let keys = build_secondary_merge_keys(graph, topology, config)?;
    let mut remap = vec![NodeId::new(0); graph.node_count()];

    let mut representatives = HashMap::new();
    for node_id in topology.topological_order().iter().copied() {
        let bucket_key = SecondaryMergeBucketKey::Shared(keys[node_id.to_usize()].clone());
        let representative = representatives.entry(bucket_key).or_insert(node_id);
        remap[node_id.to_usize()] = *representative;
    }

    let merged_any = remap
        .iter()
        .enumerate()
        .any(|(index, representative)| representative.to_usize() != index);
    Ok(merged_any.then_some(remap))
}

fn build_secondary_merge_keys(
    graph: &FtoDag,
    topology: &DagTopology,
    config: SecondaryMergeConfig,
) -> Result<Vec<SecondaryMergeKey>> {
    let mut keys = vec![None; graph.node_count()];
    for node_id in topology.topological_order().iter().copied() {
        let node = graph.node(node_id)?;
        keys[node_id.to_usize()] = Some(SecondaryMergeKey {
            fragment: node.fragment.to_fragment_key(),
            kind: node.kind,
            forward_coordinate: config
                .use_forward_coordinates
                .then(|| {
                    topology
                        .forward_coordinate(node_id)
                        .map(|coordinate| u64::from(coordinate.raw()))
                })
                .transpose()?,
            reverse_coordinate: config
                .use_reverse_coordinates
                .then(|| {
                    topology
                        .reverse_coordinate(node_id)
                        .map(|coordinate| u64::from(coordinate.raw()))
                })
                .transpose()?,
        });
    }
    Ok(keys
        .into_iter()
        .map(|key| key.expect("topological order covers every node"))
        .collect())
}

fn rebuild_graph_with_remap(
    graph: &FtoDag,
    topology: &DagTopology,
    remap: &[NodeId],
) -> Result<FtoDag> {
    let mut rebuilt = FtoDag::with_provenance_and_edge_storage(
        graph.fragment_len(),
        graph.provenance_storage_strategy(),
        graph.edge_index_strategy(),
    );
    let mut representative_new_ids = HashMap::new();
    let mut new_remap = vec![NodeId::new(0); graph.node_count()];

    for node_id in topology.topological_order().iter().copied() {
        let representative = remap[node_id.to_usize()];
        let new_id = if let Some(existing) = representative_new_ids.get(&representative).copied() {
            existing
        } else {
            let node = graph.node(representative)?;
            let created = rebuilt.add_node(node.fragment.clone(), node.kind)?;
            representative_new_ids.insert(representative, created);
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
