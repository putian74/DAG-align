//! Core weighted FTO-DAG data structures.

use crate::foundations::bit_encoding::NodeFlags;
use crate::foundations::error::{DagError, Result};
use crate::foundations::id::{NodeId, SequenceId, Weight};
use crate::graph_model::provenance::{
    PackedProvenanceRecord, ProvenanceRange, ProvenanceRecord, ProvenanceStorageStrategy,
    ProvenanceTable,
};
use crate::sequence_model::fragment::FragmentKey;
use std::collections::HashMap;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum NodeKind {
    Start,
    Internal,
    End,
    Singleton,
}

impl NodeKind {
    pub fn flags(self) -> NodeFlags {
        match self {
            Self::Start => NodeFlags::from_bits(NodeFlags::START),
            Self::Internal => NodeFlags::empty(),
            Self::End => NodeFlags::from_bits(NodeFlags::END),
            Self::Singleton => NodeFlags::from_bits(NodeFlags::START | NodeFlags::END),
        }
    }

    pub const fn is_sequence_start(self) -> bool {
        matches!(self, Self::Start | Self::Singleton)
    }

    pub const fn is_sequence_end(self) -> bool {
        matches!(self, Self::End | Self::Singleton)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Node {
    pub id: NodeId,
    pub fragment: FragmentKey,
    pub kind: NodeKind,
    pub weight: Weight,
    pub flags: NodeFlags,
    pub provenance_range: ProvenanceRange,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct EdgeKey {
    pub parent: NodeId,
    pub child: NodeId,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct WeightedEdge {
    pub key: EdgeKey,
    pub weight: Weight,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct EdgeUpdate {
    pub key: EdgeKey,
    pub weight: Weight,
    pub inserted: bool,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum EdgeIndexStrategy {
    #[default]
    GlobalHash,
    LowDegreeHybrid,
}

const HYBRID_EDGE_INLINE_LIMIT: usize = 8;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct InlineEdgeEntry {
    child: NodeId,
    edge_index: usize,
}

#[derive(Clone, Debug)]
enum EdgeIndexStorage {
    Global(HashMap<EdgeKey, usize>),
    Hybrid {
        inline: Vec<Vec<InlineEdgeEntry>>,
        overflow: HashMap<EdgeKey, usize>,
    },
}

impl EdgeIndexStorage {
    fn new(strategy: EdgeIndexStrategy) -> Self {
        match strategy {
            EdgeIndexStrategy::GlobalHash => Self::Global(HashMap::new()),
            EdgeIndexStrategy::LowDegreeHybrid => Self::Hybrid {
                inline: Vec::new(),
                overflow: HashMap::new(),
            },
        }
    }

    fn strategy(&self) -> EdgeIndexStrategy {
        match self {
            Self::Global(_) => EdgeIndexStrategy::GlobalHash,
            Self::Hybrid { .. } => EdgeIndexStrategy::LowDegreeHybrid,
        }
    }

    fn push_node(&mut self) {
        if let Self::Hybrid { inline, .. } = self {
            inline.push(Vec::new());
        }
    }

    fn get(&self, key: EdgeKey) -> Option<usize> {
        match self {
            Self::Global(index) => index.get(&key).copied(),
            Self::Hybrid { inline, overflow } => inline
                .get(key.parent.to_usize())
                .and_then(|entries| {
                    entries
                        .iter()
                        .find(|entry| entry.child == key.child)
                        .map(|entry| entry.edge_index)
                })
                .or_else(|| overflow.get(&key).copied()),
        }
    }

    fn insert(&mut self, key: EdgeKey, edge_index: usize) {
        match self {
            Self::Global(index) => {
                index.insert(key, edge_index);
            }
            Self::Hybrid { inline, overflow } => {
                let Some(entries) = inline.get_mut(key.parent.to_usize()) else {
                    overflow.insert(key, edge_index);
                    return;
                };
                if entries.len() < HYBRID_EDGE_INLINE_LIMIT {
                    entries.push(InlineEdgeEntry {
                        child: key.child,
                        edge_index,
                    });
                } else {
                    overflow.insert(key, edge_index);
                }
            }
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
struct PackedInlineIndexKey {
    kind: NodeKind,
    bits_per_symbol: u8,
    len: u16,
    value: u128,
}

impl PackedInlineIndexKey {
    const fn new(kind: NodeKind, bits_per_symbol: u8, len: u16, value: u128) -> Self {
        Self {
            kind,
            bits_per_symbol,
            len,
            value,
        }
    }

    fn from_fragment(fragment: &FragmentKey, kind: NodeKind) -> Option<Self> {
        match fragment {
            FragmentKey::PackedInline {
                bits_per_symbol,
                len,
                value,
            } => Some(Self::new(kind, *bits_per_symbol, *len, *value)),
            FragmentKey::PackedWords { .. } | FragmentKey::Symbols(_) => None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct FragmentIndex {
    entries: HashMap<NodeKind, HashMap<FragmentKey, Vec<NodeId>>>,
    packed_inline_entries: HashMap<PackedInlineIndexKey, Vec<NodeId>>,
}

impl FragmentIndex {
    pub fn insert(&mut self, fragment: &FragmentKey, kind: NodeKind, node_id: NodeId) {
        if let Some(key) = PackedInlineIndexKey::from_fragment(fragment, kind) {
            self.packed_inline_entries
                .entry(key)
                .or_default()
                .push(node_id);
        } else {
            self.entries
                .entry(kind)
                .or_default()
                .entry(fragment.clone())
                .or_default()
                .push(node_id);
        }
    }

    pub fn nodes_for(&self, fragment: &FragmentKey, kind: NodeKind) -> &[NodeId] {
        if let Some(key) = PackedInlineIndexKey::from_fragment(fragment, kind) {
            return self
                .packed_inline_entries
                .get(&key)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
        }
        self.entries
            .get(&kind)
            .and_then(|entries| entries.get(fragment))
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn contains(&self, fragment: &FragmentKey, kind: NodeKind, node_id: NodeId) -> bool {
        self.nodes_for(fragment, kind).contains(&node_id)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.packed_inline_entries.clear();
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EndpointIndex {
    sequence_starts: Vec<NodeId>,
    sequence_ends: Vec<NodeId>,
    structural_roots: Vec<NodeId>,
    structural_sinks: Vec<NodeId>,
    structural_root_positions: Vec<u32>,
    structural_sink_positions: Vec<u32>,
}

impl EndpointIndex {
    const MISSING_POSITION: u32 = u32::MAX;

    pub fn sequence_starts(&self) -> &[NodeId] {
        &self.sequence_starts
    }

    pub fn sequence_ends(&self) -> &[NodeId] {
        &self.sequence_ends
    }

    pub fn structural_roots(&self) -> &[NodeId] {
        &self.structural_roots
    }

    pub fn structural_sinks(&self) -> &[NodeId] {
        &self.structural_sinks
    }

    fn record_node_kind(&mut self, node_id: NodeId, kind: NodeKind) {
        if kind.is_sequence_start() {
            self.sequence_starts.push(node_id);
        }
        if kind.is_sequence_end() {
            self.sequence_ends.push(node_id);
        }
        let root_position =
            u32::try_from(self.structural_roots.len()).expect("endpoint count exceeds u32");
        let sink_position =
            u32::try_from(self.structural_sinks.len()).expect("endpoint count exceeds u32");
        Self::ensure_position_len(&mut self.structural_root_positions, node_id);
        Self::ensure_position_len(&mut self.structural_sink_positions, node_id);
        self.structural_root_positions[node_id.to_usize()] = root_position;
        self.structural_sink_positions[node_id.to_usize()] = sink_position;
        self.structural_roots.push(node_id);
        self.structural_sinks.push(node_id);
    }

    fn record_edge_insertion(&mut self, parent: NodeId, child: NodeId) {
        Self::remove_endpoint(
            &mut self.structural_sinks,
            &mut self.structural_sink_positions,
            parent,
        );
        Self::remove_endpoint(
            &mut self.structural_roots,
            &mut self.structural_root_positions,
            child,
        );
    }

    fn ensure_position_len(positions: &mut Vec<u32>, node_id: NodeId) {
        let required_len = node_id.to_usize() + 1;
        if positions.len() < required_len {
            positions.resize(required_len, Self::MISSING_POSITION);
        }
    }

    fn remove_endpoint(endpoints: &mut Vec<NodeId>, positions: &mut [u32], node_id: NodeId) {
        let node_index = node_id.to_usize();
        let Some(position) = positions.get_mut(node_index) else {
            return;
        };
        if *position == Self::MISSING_POSITION {
            return;
        }
        let position_index = *position as usize;
        *position = Self::MISSING_POSITION;
        let moved = endpoints.swap_remove(position_index);
        debug_assert_eq!(moved, node_id);
        if let Some(replacement) = endpoints.get(position_index) {
            positions[replacement.to_usize()] =
                u32::try_from(position_index).expect("endpoint count exceeds u32");
        }
    }
}

#[derive(Clone, Debug)]
pub struct FtoDag {
    fragment_len: usize,
    nodes: Vec<Node>,
    edges: Vec<WeightedEdge>,
    edge_index: EdgeIndexStorage,
    parents: Vec<Vec<NodeId>>,
    children: Vec<Vec<NodeId>>,
    provenance_table: ProvenanceTable,
    node_provenance: NodeProvenanceStorage,
    sequence_trace_paths: Vec<Vec<NodeId>>,
    node_last_sequences: Vec<Option<SequenceId>>,
    fragment_index: FragmentIndex,
    endpoint_index: EndpointIndex,
}

#[derive(Clone, Debug)]
pub(crate) enum NodeProvenanceStorage {
    Full(Vec<Vec<ProvenanceRecord>>),
    Packed32(Vec<Vec<PackedProvenanceRecord>>),
    TracePaths(Vec<u64>),
    CountOnly(Vec<u64>),
}

#[derive(Clone, Debug)]
pub(crate) enum ProvenanceSnapshot {
    Full(Vec<Vec<ProvenanceRecord>>),
    Packed32(Vec<Vec<PackedProvenanceRecord>>),
    TracePaths {
        node_counts: Vec<u64>,
        sequence_trace_paths: Vec<Vec<NodeId>>,
    },
    CountOnly(Vec<u64>),
}

impl ProvenanceSnapshot {
    pub(crate) fn strategy(&self) -> ProvenanceStorageStrategy {
        match self {
            Self::Full(_) => ProvenanceStorageStrategy::FullRecords,
            Self::Packed32(_) => ProvenanceStorageStrategy::Packed32,
            Self::TracePaths { .. } => ProvenanceStorageStrategy::TracePaths,
            Self::CountOnly(_) => ProvenanceStorageStrategy::CountOnly,
        }
    }

    fn node_count(&self) -> usize {
        match self {
            Self::Full(records) => records.len(),
            Self::Packed32(records) => records.len(),
            Self::TracePaths { node_counts, .. } => node_counts.len(),
            Self::CountOnly(counts) => counts.len(),
        }
    }

    fn sequence_trace_paths(&self) -> Vec<Vec<NodeId>> {
        match self {
            Self::TracePaths {
                sequence_trace_paths,
                ..
            } => sequence_trace_paths.clone(),
            Self::Full(_) | Self::Packed32(_) | Self::CountOnly(_) => Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FtoDagSnapshot {
    pub fragment_len: usize,
    pub nodes: Vec<Node>,
    pub edges: Vec<WeightedEdge>,
    pub edge_index_strategy: EdgeIndexStrategy,
    pub provenance: ProvenanceSnapshot,
    pub node_last_sequences: Vec<Option<SequenceId>>,
}

impl NodeProvenanceStorage {
    fn new(strategy: ProvenanceStorageStrategy) -> Self {
        match strategy {
            ProvenanceStorageStrategy::FullRecords => Self::Full(Vec::new()),
            ProvenanceStorageStrategy::Packed32 => Self::Packed32(Vec::new()),
            ProvenanceStorageStrategy::TracePaths => Self::TracePaths(Vec::new()),
            ProvenanceStorageStrategy::CountOnly => Self::CountOnly(Vec::new()),
        }
    }

    fn strategy(&self) -> ProvenanceStorageStrategy {
        match self {
            Self::Full(_) => ProvenanceStorageStrategy::FullRecords,
            Self::Packed32(_) => ProvenanceStorageStrategy::Packed32,
            Self::TracePaths(_) => ProvenanceStorageStrategy::TracePaths,
            Self::CountOnly(_) => ProvenanceStorageStrategy::CountOnly,
        }
    }

    fn push_node(&mut self) {
        match self {
            Self::Full(records) => records.push(Vec::new()),
            Self::Packed32(records) => records.push(Vec::new()),
            Self::TracePaths(counts) => counts.push(0),
            Self::CountOnly(counts) => counts.push(0),
        }
    }

    fn ensure_node_exists(&self, node_id: NodeId) -> Result<()> {
        let exists = match self {
            Self::Full(records) => node_id.to_usize() < records.len(),
            Self::Packed32(records) => node_id.to_usize() < records.len(),
            Self::TracePaths(counts) => node_id.to_usize() < counts.len(),
            Self::CountOnly(counts) => node_id.to_usize() < counts.len(),
        };
        if exists {
            Ok(())
        } else {
            Err(DagError::MissingNode {
                node: node_id.to_usize(),
            })
        }
    }

    fn add_record(&mut self, node_id: NodeId, record: ProvenanceRecord) -> Result<()> {
        let node_index = node_id.to_usize();
        match self {
            Self::Full(records) => records
                .get_mut(node_index)
                .ok_or(DagError::MissingNode { node: node_index })?
                .push(record),
            Self::Packed32(records) => records
                .get_mut(node_index)
                .ok_or(DagError::MissingNode { node: node_index })?
                .push(PackedProvenanceRecord::try_from_record(record)?),
            Self::TracePaths(counts) => {
                let count = counts
                    .get_mut(node_index)
                    .ok_or(DagError::MissingNode { node: node_index })?;
                *count += 1;
            }
            Self::CountOnly(counts) => {
                let count = counts
                    .get_mut(node_index)
                    .ok_or(DagError::MissingNode { node: node_index })?;
                *count += 1;
            }
        }
        Ok(())
    }

    fn add_count(&mut self, node_id: NodeId, count: u64) -> Result<()> {
        let node_index = node_id.to_usize();
        match self {
            Self::CountOnly(counts) => {
                let slot = counts
                    .get_mut(node_index)
                    .ok_or(DagError::MissingNode { node: node_index })?;
                *slot += count;
                Ok(())
            }
            Self::Full(_) | Self::Packed32(_) | Self::TracePaths(_) => {
                Err(DagError::UnsupportedOperation(
                    "bulk provenance count transfer is only supported with CountOnly provenance storage",
                ))
            }
        }
    }

    fn record_count(&self, node_id: NodeId) -> Result<usize> {
        let node_index = node_id.to_usize();
        match self {
            Self::Full(records) => records
                .get(node_index)
                .map(Vec::len)
                .ok_or(DagError::MissingNode { node: node_index }),
            Self::Packed32(records) => records
                .get(node_index)
                .map(Vec::len)
                .ok_or(DagError::MissingNode { node: node_index }),
            Self::TracePaths(counts) => counts
                .get(node_index)
                .map(|count| *count as usize)
                .ok_or(DagError::MissingNode { node: node_index }),
            Self::CountOnly(counts) => counts
                .get(node_index)
                .map(|count| *count as usize)
                .ok_or(DagError::MissingNode { node: node_index }),
        }
    }

    fn records(&self, node_id: NodeId) -> Result<Vec<ProvenanceRecord>> {
        let node_index = node_id.to_usize();
        match self {
            Self::Full(records) => records
                .get(node_index)
                .cloned()
                .ok_or(DagError::MissingNode { node: node_index }),
            Self::Packed32(records) => records
                .get(node_index)
                .map(|records| {
                    records
                        .iter()
                        .copied()
                        .map(PackedProvenanceRecord::unpack)
                        .collect()
                })
                .ok_or(DagError::MissingNode { node: node_index }),
            Self::TracePaths(_) => Err(DagError::UnsupportedOperation(
                "node provenance records are not retained with TracePaths provenance storage",
            )),
            Self::CountOnly(_) => Err(DagError::UnsupportedOperation(
                "node provenance records are not retained with CountOnly provenance storage",
            )),
        }
    }

    fn can_accept_sequence(&self, node_id: NodeId, sequence_id: SequenceId) -> Result<bool> {
        let node_index = node_id.to_usize();
        match self {
            Self::Full(records) => Ok(records
                .get(node_index)
                .ok_or(DagError::MissingNode { node: node_index })?
                .iter()
                .all(|record| record.sequence_id != sequence_id)),
            Self::Packed32(records) => Ok(records
                .get(node_index)
                .ok_or(DagError::MissingNode { node: node_index })?
                .iter()
                .all(|record| record.sequence_id() != sequence_id)),
            Self::TracePaths(_) => Err(DagError::UnsupportedOperation(
                "out-of-order duplicate provenance checks require retained node provenance records",
            )),
            Self::CountOnly(_) => Err(DagError::UnsupportedOperation(
                "out-of-order duplicate provenance checks require retained node provenance records",
            )),
        }
    }

    fn retains_records(&self) -> bool {
        matches!(self, Self::Full(_) | Self::Packed32(_))
    }

    fn retains_trace_paths(&self) -> bool {
        matches!(self, Self::TracePaths(_))
    }

    fn snapshot(&self, sequence_trace_paths: &[Vec<NodeId>]) -> ProvenanceSnapshot {
        match self {
            Self::Full(records) => ProvenanceSnapshot::Full(records.clone()),
            Self::Packed32(records) => ProvenanceSnapshot::Packed32(records.clone()),
            Self::TracePaths(node_counts) => ProvenanceSnapshot::TracePaths {
                node_counts: node_counts.clone(),
                sequence_trace_paths: sequence_trace_paths.to_vec(),
            },
            Self::CountOnly(counts) => ProvenanceSnapshot::CountOnly(counts.clone()),
        }
    }

    fn from_snapshot(snapshot: &ProvenanceSnapshot) -> Self {
        match snapshot {
            ProvenanceSnapshot::Full(records) => Self::Full(records.clone()),
            ProvenanceSnapshot::Packed32(records) => Self::Packed32(records.clone()),
            ProvenanceSnapshot::TracePaths { node_counts, .. } => {
                Self::TracePaths(node_counts.clone())
            }
            ProvenanceSnapshot::CountOnly(counts) => Self::CountOnly(counts.clone()),
        }
    }
}

impl FtoDag {
    pub fn new(fragment_len: usize) -> Self {
        Self::with_provenance_storage(fragment_len, ProvenanceStorageStrategy::FullRecords)
    }

    pub fn with_provenance_storage(
        fragment_len: usize,
        provenance_storage: ProvenanceStorageStrategy,
    ) -> Self {
        Self::with_provenance_and_edge_storage(
            fragment_len,
            provenance_storage,
            EdgeIndexStrategy::GlobalHash,
        )
    }

    pub fn with_provenance_and_edge_storage(
        fragment_len: usize,
        provenance_storage: ProvenanceStorageStrategy,
        edge_index_strategy: EdgeIndexStrategy,
    ) -> Self {
        Self {
            fragment_len,
            nodes: Vec::new(),
            edges: Vec::new(),
            edge_index: EdgeIndexStorage::new(edge_index_strategy),
            parents: Vec::new(),
            children: Vec::new(),
            provenance_table: ProvenanceTable::new(),
            node_provenance: NodeProvenanceStorage::new(provenance_storage),
            sequence_trace_paths: Vec::new(),
            node_last_sequences: Vec::new(),
            fragment_index: FragmentIndex::default(),
            endpoint_index: EndpointIndex::default(),
        }
    }

    pub fn fragment_len(&self) -> usize {
        self.fragment_len
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn edges(&self) -> &[WeightedEdge] {
        &self.edges
    }

    pub fn node(&self, node_id: NodeId) -> Result<&Node> {
        self.nodes
            .get(node_id.to_usize())
            .ok_or(DagError::MissingNode {
                node: node_id.to_usize(),
            })
    }

    pub fn edge_weight(&self, key: EdgeKey) -> Option<Weight> {
        self.edge_index
            .get(key)
            .map(|edge_index| self.edges[edge_index].weight)
    }

    pub fn parents(&self, node_id: NodeId) -> Result<&[NodeId]> {
        self.parents
            .get(node_id.to_usize())
            .map(Vec::as_slice)
            .ok_or(DagError::MissingNode {
                node: node_id.to_usize(),
            })
    }

    pub fn children(&self, node_id: NodeId) -> Result<&[NodeId]> {
        self.children
            .get(node_id.to_usize())
            .map(Vec::as_slice)
            .ok_or(DagError::MissingNode {
                node: node_id.to_usize(),
            })
    }

    pub fn provenance_table(&self) -> &ProvenanceTable {
        &self.provenance_table
    }

    pub fn provenance_storage_strategy(&self) -> ProvenanceStorageStrategy {
        self.node_provenance.strategy()
    }

    pub fn edge_index_strategy(&self) -> EdgeIndexStrategy {
        self.edge_index.strategy()
    }

    pub fn to_count_only(&self) -> Result<Self> {
        let mut snapshot = self.snapshot();
        snapshot.provenance = ProvenanceSnapshot::CountOnly(
            self.nodes.iter().map(|node| node.weight.raw()).collect(),
        );
        Self::from_snapshot(snapshot)
    }

    pub fn provenance_records(&self, node_id: NodeId) -> Result<Vec<ProvenanceRecord>> {
        self.node_provenance.records(node_id)
    }

    pub fn provenance_record_count(&self, node_id: NodeId) -> Result<usize> {
        self.node_provenance.record_count(node_id)
    }

    pub fn retains_provenance_records(&self) -> bool {
        self.node_provenance.retains_records()
    }

    pub fn retains_sequence_trace_paths(&self) -> bool {
        self.node_provenance.retains_trace_paths()
    }

    pub fn sequence_trace_path(&self, sequence_id: SequenceId) -> Result<&[NodeId]> {
        if !self.retains_sequence_trace_paths() {
            return Err(DagError::UnsupportedOperation(
                "sequence trace paths are only retained with TracePaths provenance storage",
            ));
        }
        self.sequence_trace_paths
            .get(sequence_id.to_usize())
            .map(Vec::as_slice)
            .ok_or(DagError::InvalidRange {
                start: sequence_id.to_usize(),
                end: sequence_id.to_usize().saturating_add(1),
                len: self.sequence_trace_paths.len(),
            })
    }

    pub(crate) fn sequence_trace_paths(&self) -> Result<&[Vec<NodeId>]> {
        if !self.retains_sequence_trace_paths() {
            return Err(DagError::UnsupportedOperation(
                "sequence trace paths are only retained with TracePaths provenance storage",
            ));
        }
        Ok(&self.sequence_trace_paths)
    }

    pub fn can_node_accept_sequence(
        &self,
        node_id: NodeId,
        sequence_id: SequenceId,
    ) -> Result<bool> {
        let node_index = node_id.to_usize();
        self.node_provenance.ensure_node_exists(node_id)?;
        match self.node_last_sequences.get(node_index).copied().flatten() {
            Some(last_sequence_id) if last_sequence_id < sequence_id => Ok(true),
            Some(last_sequence_id) if last_sequence_id == sequence_id => Ok(false),
            _ => self
                .node_provenance
                .can_accept_sequence(node_id, sequence_id),
        }
    }

    pub fn fragment_index(&self) -> &FragmentIndex {
        &self.fragment_index
    }

    pub fn endpoints(&self) -> &EndpointIndex {
        &self.endpoint_index
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    pub fn add_node(&mut self, fragment: FragmentKey, kind: NodeKind) -> Result<NodeId> {
        let id = NodeId::try_from(self.nodes.len())?;
        let node = Node {
            id,
            fragment: fragment.clone(),
            kind,
            weight: Weight::new(0),
            flags: kind.flags(),
            provenance_range: ProvenanceRange::default(),
        };
        self.fragment_index.insert(&fragment, kind, id);
        self.endpoint_index.record_node_kind(id, kind);
        self.nodes.push(node);
        self.parents.push(Vec::new());
        self.children.push(Vec::new());
        self.edge_index.push_node();
        self.node_provenance.push_node();
        self.node_last_sequences.push(None);
        Ok(id)
    }

    pub fn add_provenance_record(
        &mut self,
        node_id: NodeId,
        record: ProvenanceRecord,
    ) -> Result<()> {
        let node_index = node_id.to_usize();
        self.node_provenance.add_record(node_id, record)?;
        self.record_sequence_trace_path(node_id, record)?;
        let last_sequence = &mut self.node_last_sequences[node_index];
        if last_sequence.is_none_or(|last| last < record.sequence_id) {
            *last_sequence = Some(record.sequence_id);
        }
        let node = &mut self.nodes[node_index];
        node.provenance_range =
            ProvenanceRange::new(0, self.node_provenance.record_count(node_id)? as u64);
        node.weight = Weight::new(node.weight.raw() + 1);
        Ok(())
    }

    pub fn add_provenance_count(&mut self, node_id: NodeId, count: u64) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let node_index = node_id.to_usize();
        self.node_provenance.add_count(node_id, count)?;
        let node = self
            .nodes
            .get_mut(node_index)
            .ok_or(DagError::MissingNode { node: node_index })?;
        node.provenance_range =
            ProvenanceRange::new(0, self.node_provenance.record_count(node_id)? as u64);
        node.weight = Weight::new(node.weight.raw() + count);
        Ok(())
    }

    fn record_sequence_trace_path(
        &mut self,
        node_id: NodeId,
        record: ProvenanceRecord,
    ) -> Result<()> {
        if !self.retains_sequence_trace_paths() {
            return Ok(());
        }
        let sequence_index = record.sequence_id.to_usize();
        if self.sequence_trace_paths.len() <= sequence_index {
            self.sequence_trace_paths
                .resize_with(sequence_index.saturating_add(1), Vec::new);
        }
        let position =
            usize::try_from(record.position.raw()).map_err(|_| DagError::ValueDoesNotFit {
                value: record.position.raw() as u128,
                bits: usize::BITS as u8,
            })?;
        let path = &mut self.sequence_trace_paths[sequence_index];
        if position != path.len() {
            return Err(DagError::InvalidRange {
                start: position,
                end: position.saturating_add(1),
                len: path.len(),
            });
        }
        path.push(node_id);
        Ok(())
    }

    pub fn add_or_increment_edge(
        &mut self,
        parent: NodeId,
        child: NodeId,
        increment: Weight,
    ) -> Result<EdgeUpdate> {
        if parent.to_usize() >= self.nodes.len() || child.to_usize() >= self.nodes.len() {
            return Err(DagError::InvalidEdge {
                parent: parent.to_usize(),
                child: child.to_usize(),
            });
        }
        let key = EdgeKey { parent, child };
        let (weight, inserted) = if let Some(edge_index) = self.edge_index.get(key) {
            let edge = &mut self.edges[edge_index];
            edge.weight = Weight::new(edge.weight.raw() + increment.raw());
            (edge.weight, false)
        } else {
            let edge_index = self.edges.len();
            self.edges.push(WeightedEdge {
                key,
                weight: increment,
            });
            self.edge_index.insert(key, edge_index);
            self.children[parent.to_usize()].push(child);
            self.parents[child.to_usize()].push(parent);
            (increment, true)
        };
        if inserted {
            self.endpoint_index.record_edge_insertion(parent, child);
        }
        Ok(EdgeUpdate {
            key,
            weight,
            inserted,
        })
    }

    pub fn rebuild_fragment_index(&mut self) {
        self.fragment_index.clear();
        for node in &self.nodes {
            self.fragment_index
                .insert(&node.fragment, node.kind, node.id);
        }
    }

    pub fn stats(&self) -> GraphStats {
        GraphStats {
            node_count: self.node_count(),
            edge_count: self.edge_count(),
            fragment_len: self.fragment_len,
        }
    }

    pub(crate) fn snapshot(&self) -> FtoDagSnapshot {
        FtoDagSnapshot {
            fragment_len: self.fragment_len,
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            edge_index_strategy: self.edge_index.strategy(),
            provenance: self.node_provenance.snapshot(&self.sequence_trace_paths),
            node_last_sequences: self.node_last_sequences.clone(),
        }
    }

    pub(crate) fn from_snapshot(snapshot: FtoDagSnapshot) -> Result<Self> {
        if snapshot.nodes.len() != snapshot.provenance.node_count() {
            return Err(DagError::InvalidStorage(format!(
                "node/provenance length mismatch: {} nodes vs {} provenance slots",
                snapshot.nodes.len(),
                snapshot.provenance.node_count()
            )));
        }
        if snapshot.nodes.len() != snapshot.node_last_sequences.len() {
            return Err(DagError::InvalidStorage(format!(
                "node/node-last-sequence length mismatch: {} nodes vs {} markers",
                snapshot.nodes.len(),
                snapshot.node_last_sequences.len()
            )));
        }

        let mut graph = Self {
            fragment_len: snapshot.fragment_len,
            nodes: snapshot.nodes,
            edges: snapshot.edges,
            edge_index: EdgeIndexStorage::new(snapshot.edge_index_strategy),
            parents: vec![Vec::new(); snapshot.provenance.node_count()],
            children: vec![Vec::new(); snapshot.provenance.node_count()],
            provenance_table: ProvenanceTable::new(),
            node_provenance: NodeProvenanceStorage::from_snapshot(&snapshot.provenance),
            sequence_trace_paths: snapshot.provenance.sequence_trace_paths(),
            node_last_sequences: snapshot.node_last_sequences,
            fragment_index: FragmentIndex::default(),
            endpoint_index: EndpointIndex::default(),
        };

        for (expected, node) in graph.nodes.iter().enumerate() {
            if node.id.to_usize() != expected {
                return Err(DagError::InvalidStorage(format!(
                    "node id mismatch at index {expected}: found {}",
                    node.id.to_usize()
                )));
            }
            graph
                .fragment_index
                .insert(&node.fragment, node.kind, node.id);
            graph.endpoint_index.record_node_kind(node.id, node.kind);
        }

        for (edge_index, edge) in graph.edges.iter().enumerate() {
            let parent = edge.key.parent.to_usize();
            let child = edge.key.child.to_usize();
            if parent >= graph.nodes.len() || child >= graph.nodes.len() {
                return Err(DagError::InvalidEdge { parent, child });
            }
            graph.edge_index.insert(edge.key, edge_index);
            graph.children[parent].push(edge.key.child);
            graph.parents[child].push(edge.key.parent);
            graph
                .endpoint_index
                .record_edge_insertion(edge.key.parent, edge.key.child);
        }

        Ok(graph)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub fragment_len: usize,
}
