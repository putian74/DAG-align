//! Core weighted FTO-DAG data structures.

use crate::foundations::bit_encoding::NodeFlags;
use crate::foundations::error::{DagError, Result};
use crate::foundations::id::{NodeId, ProvenancePosition, SequenceId, Weight};
use crate::graph_model::provenance::{
    PackedProvenanceRecord, ProvenanceRange, ProvenanceRecord, ProvenanceStorageStrategy,
    ProvenanceTable,
};
use crate::sequence_model::fragment::FragmentKey;
use std::collections::HashMap;
use std::fs::{File, OpenOptions, remove_file};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

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
const TRACE_PATH_NODE_BYTES: u64 = std::mem::size_of::<u32>() as u64;
const TRACE_PATH_FLUSH_NODE_COUNT: usize = 16_384;
const MISSING_SEQUENCE_MARKER: u32 = u32::MAX;
static NEXT_TRACE_PATH_FILE_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct InlineEdgeEntry {
    child: NodeId,
    edge_index: usize,
}

#[derive(Clone, Debug)]
enum EdgeIndexStorage {
    Global(HashMap<EdgeKey, usize>),
    HybridMutable {
        inline: Vec<Vec<InlineEdgeEntry>>,
        overflow: HashMap<EdgeKey, usize>,
    },
    HybridPacked {
        inline_offsets: Vec<u32>,
        inline_entries: Vec<InlineEdgeEntry>,
        overflow: HashMap<EdgeKey, usize>,
    },
}

impl EdgeIndexStorage {
    fn new(strategy: EdgeIndexStrategy) -> Self {
        match strategy {
            EdgeIndexStrategy::GlobalHash => Self::Global(HashMap::new()),
            EdgeIndexStrategy::LowDegreeHybrid => Self::HybridMutable {
                inline: Vec::new(),
                overflow: HashMap::new(),
            },
        }
    }

    fn strategy(&self) -> EdgeIndexStrategy {
        match self {
            Self::Global(_) => EdgeIndexStrategy::GlobalHash,
            Self::HybridMutable { .. } | Self::HybridPacked { .. } => {
                EdgeIndexStrategy::LowDegreeHybrid
            }
        }
    }

    fn push_node(&mut self) {
        if matches!(self, Self::HybridPacked { .. }) {
            self.unpack_hybrid();
        }
        if let Self::HybridMutable { inline, .. } = self {
            inline.push(Vec::new());
        }
    }

    fn get(&self, key: EdgeKey) -> Option<usize> {
        match self {
            Self::Global(index) => index.get(&key).copied(),
            Self::HybridMutable { inline, overflow } => inline
                .get(key.parent.to_usize())
                .and_then(|entries| {
                    entries
                        .iter()
                        .find(|entry| entry.child == key.child)
                        .map(|entry| entry.edge_index)
                })
                .or_else(|| overflow.get(&key).copied()),
            Self::HybridPacked {
                inline_offsets,
                inline_entries,
                overflow,
            } => packed_range(inline_offsets, key.parent)
                .and_then(|(start, end)| inline_entries.get(start..end))
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
        if matches!(self, Self::HybridPacked { .. }) {
            self.unpack_hybrid();
        }
        match self {
            Self::Global(index) => {
                index.insert(key, edge_index);
            }
            Self::HybridMutable { inline, overflow } => {
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
            Self::HybridPacked { .. } => {
                unreachable!("packed hybrid storage is unpacked before insertion")
            }
        }
    }

    fn compact(&mut self) -> Result<()> {
        let replacement = match self {
            Self::Global(_) | Self::HybridPacked { .. } => None,
            Self::HybridMutable { inline, overflow } => {
                let total_entries = inline.iter().map(Vec::len).sum::<usize>();
                let mut inline_offsets = Vec::with_capacity(inline.len() + 1);
                inline_offsets.push(0);
                let mut running_total = 0usize;
                for entries in inline.iter() {
                    running_total = running_total.checked_add(entries.len()).ok_or(
                        DagError::ValueDoesNotFit {
                            value: total_entries as u128,
                            bits: usize::BITS as u8,
                        },
                    )?;
                    inline_offsets.push(u32::try_from(running_total).map_err(|_| {
                        DagError::ValueDoesNotFit {
                            value: running_total as u128,
                            bits: 32,
                        }
                    })?);
                }
                let mut inline_entries = Vec::with_capacity(total_entries);
                for entries in inline.iter() {
                    inline_entries.extend(entries.iter().copied());
                }
                let overflow = std::mem::take(overflow);
                Some(Self::HybridPacked {
                    inline_offsets,
                    inline_entries,
                    overflow,
                })
            }
        };
        if let Some(storage) = replacement {
            *self = storage;
        }
        Ok(())
    }

    fn unpack_hybrid(&mut self) {
        let packed = match std::mem::replace(self, Self::Global(HashMap::new())) {
            Self::HybridPacked {
                inline_offsets,
                inline_entries,
                overflow,
            } => Some((inline_offsets, inline_entries, overflow)),
            other => {
                *self = other;
                None
            }
        };
        let Some((inline_offsets, inline_entries, overflow)) = packed else {
            return;
        };
        let node_count = inline_offsets.len().saturating_sub(1);
        let mut inline = Vec::with_capacity(node_count);
        for node_index in 0..node_count {
            let start = inline_offsets[node_index] as usize;
            let end = inline_offsets[node_index + 1] as usize;
            inline.push(inline_entries[start..end].to_vec());
        }
        *self = Self::HybridMutable { inline, overflow };
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PackedAdjacency {
    offsets: Vec<u32>,
    nodes: Vec<NodeId>,
}

impl PackedAdjacency {
    fn from_lists(lists: &[Vec<NodeId>]) -> Result<Self> {
        let total_neighbors = lists.iter().map(Vec::len).sum::<usize>();
        let mut offsets = Vec::with_capacity(lists.len() + 1);
        offsets.push(0);
        let mut running_total = 0usize;
        for neighbors in lists {
            running_total =
                running_total
                    .checked_add(neighbors.len())
                    .ok_or(DagError::ValueDoesNotFit {
                        value: total_neighbors as u128,
                        bits: usize::BITS as u8,
                    })?;
            offsets.push(
                u32::try_from(running_total).map_err(|_| DagError::ValueDoesNotFit {
                    value: running_total as u128,
                    bits: 32,
                })?,
            );
        }
        let mut nodes = Vec::with_capacity(total_neighbors);
        for neighbors in lists {
            nodes.extend(neighbors.iter().copied());
        }
        Ok(Self { offsets, nodes })
    }

    fn from_edges(node_count: usize, edges: &[WeightedEdge], use_children: bool) -> Result<Self> {
        let mut counts = vec![0u32; node_count];
        for edge in edges {
            let index = if use_children {
                edge.key.parent.to_usize()
            } else {
                edge.key.child.to_usize()
            };
            let count = counts.get_mut(index).ok_or(DagError::InvalidEdge {
                parent: edge.key.parent.to_usize(),
                child: edge.key.child.to_usize(),
            })?;
            *count = count.checked_add(1).ok_or(DagError::ValueDoesNotFit {
                value: u128::from(*count) + 1,
                bits: 32,
            })?;
        }
        let mut offsets = Vec::with_capacity(node_count + 1);
        offsets.push(0);
        let mut running_total = 0u32;
        for count in &counts {
            running_total = running_total
                .checked_add(*count)
                .ok_or(DagError::ValueDoesNotFit {
                    value: u128::from(running_total) + u128::from(*count),
                    bits: 32,
                })?;
            offsets.push(running_total);
        }
        let mut positions = offsets[..node_count].to_vec();
        let mut nodes = vec![NodeId::new(0); edges.len()];
        for edge in edges {
            let owner = if use_children {
                edge.key.parent.to_usize()
            } else {
                edge.key.child.to_usize()
            };
            let neighbor = if use_children {
                edge.key.child
            } else {
                edge.key.parent
            };
            let position = positions[owner] as usize;
            nodes[position] = neighbor;
            positions[owner] += 1;
        }
        Ok(Self { offsets, nodes })
    }

    fn neighbors(&self, node_id: NodeId) -> Result<&[NodeId]> {
        let (start, end) = packed_range(&self.offsets, node_id).ok_or(DagError::MissingNode {
            node: node_id.to_usize(),
        })?;
        self.nodes.get(start..end).ok_or(DagError::InvalidRange {
            start,
            end,
            len: self.nodes.len(),
        })
    }

    fn into_lists(self) -> Vec<Vec<NodeId>> {
        let node_count = self.offsets.len().saturating_sub(1);
        let mut lists = Vec::with_capacity(node_count);
        for node_index in 0..node_count {
            let start = self.offsets[node_index] as usize;
            let end = self.offsets[node_index + 1] as usize;
            lists.push(self.nodes[start..end].to_vec());
        }
        lists
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum AdjacencyLists {
    Ragged(Vec<Vec<NodeId>>),
    Packed(PackedAdjacency),
}

impl AdjacencyLists {
    fn new() -> Self {
        Self::Ragged(Vec::new())
    }

    fn from_edges(node_count: usize, edges: &[WeightedEdge], use_children: bool) -> Result<Self> {
        Ok(Self::Packed(PackedAdjacency::from_edges(
            node_count,
            edges,
            use_children,
        )?))
    }

    fn neighbors(&self, node_id: NodeId) -> Result<&[NodeId]> {
        match self {
            Self::Ragged(lists) => {
                lists
                    .get(node_id.to_usize())
                    .map(Vec::as_slice)
                    .ok_or(DagError::MissingNode {
                        node: node_id.to_usize(),
                    })
            }
            Self::Packed(packed) => packed.neighbors(node_id),
        }
    }

    fn push_node(&mut self) {
        self.ensure_ragged().push(Vec::new());
    }

    fn push_neighbor(&mut self, owner: NodeId, neighbor: NodeId) -> Result<()> {
        self.ensure_ragged()
            .get_mut(owner.to_usize())
            .ok_or(DagError::MissingNode {
                node: owner.to_usize(),
            })?
            .push(neighbor);
        Ok(())
    }

    fn compact(&mut self) -> Result<()> {
        let replacement = match self {
            Self::Ragged(lists) => Some(Self::Packed(PackedAdjacency::from_lists(lists)?)),
            Self::Packed(_) => None,
        };
        if let Some(storage) = replacement {
            *self = storage;
        }
        Ok(())
    }

    fn ensure_ragged(&mut self) -> &mut Vec<Vec<NodeId>> {
        if matches!(self, Self::Packed(_)) {
            let packed = match std::mem::replace(self, Self::Ragged(Vec::new())) {
                Self::Packed(packed) => packed,
                Self::Ragged(lists) => {
                    *self = Self::Ragged(lists);
                    unreachable!("packed adjacency was replaced with ragged storage")
                }
            };
            *self = Self::Ragged(packed.into_lists());
        }
        match self {
            Self::Ragged(lists) => lists,
            Self::Packed(_) => unreachable!("packed adjacency was unpacked into ragged storage"),
        }
    }
}

fn packed_range(offsets: &[u32], node_id: NodeId) -> Option<(usize, usize)> {
    let start = *offsets.get(node_id.to_usize())? as usize;
    let end = *offsets.get(node_id.to_usize() + 1)? as usize;
    Some((start, end))
}

fn sequence_id_from_marker(marker: u32) -> Option<SequenceId> {
    (marker != MISSING_SEQUENCE_MARKER).then_some(SequenceId::new(marker))
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
    parents: AdjacencyLists,
    children: AdjacencyLists,
    provenance_table: ProvenanceTable,
    node_provenance: NodeProvenanceStorage,
    sequence_trace_paths: SequenceTraceStore,
    node_last_sequences: Vec<u32>,
    fragment_index: FragmentIndex,
    endpoint_index: EndpointIndex,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SequenceTraceStore {
    offsets: Vec<u64>,
    backing: SequenceTraceBacking,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum SequenceTraceBacking {
    InMemory(Vec<NodeId>),
    DeferredExternal,
    External(Arc<ExternalTraceFile>),
}

#[derive(Debug)]
struct ExternalTraceFile {
    path: PathBuf,
    state: Mutex<ExternalTraceState>,
}

#[derive(Debug)]
struct ExternalTraceState {
    file: File,
    pending_nodes: Vec<NodeId>,
}

impl PartialEq for ExternalTraceFile {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path
    }
}

impl Eq for ExternalTraceFile {}

impl Drop for ExternalTraceFile {
    fn drop(&mut self) {
        let _ = remove_file(&self.path);
    }
}

impl Default for SequenceTraceStore {
    fn default() -> Self {
        Self {
            offsets: vec![0],
            backing: SequenceTraceBacking::InMemory(Vec::new()),
        }
    }
}

impl SequenceTraceStore {
    fn deferred_external() -> Self {
        Self {
            offsets: vec![0],
            backing: SequenceTraceBacking::DeferredExternal,
        }
    }

    fn from_parts(offsets: Vec<u64>, nodes: Vec<NodeId>, externalize: bool) -> Result<Self> {
        if offsets.is_empty() {
            return Err(DagError::InvalidStorage(
                "trace-path offsets must include an initial zero".to_string(),
            ));
        }
        if offsets[0] != 0 {
            return Err(DagError::InvalidStorage(format!(
                "trace-path offsets must start at 0, found {}",
                offsets[0]
            )));
        }
        let expected_end = u64::try_from(nodes.len()).map_err(|_| DagError::ValueDoesNotFit {
            value: nodes.len() as u128,
            bits: 64,
        })?;
        for window in offsets.windows(2) {
            if window[0] > window[1] {
                return Err(DagError::InvalidStorage(
                    "trace-path offsets must be monotone".to_string(),
                ));
            }
        }
        if offsets[offsets.len() - 1] != expected_end {
            return Err(DagError::InvalidStorage(format!(
                "trace-path offsets end {} does not match stored node count {expected_end}",
                offsets[offsets.len() - 1]
            )));
        }
        let backing = if externalize {
            let file = ExternalTraceFile::create_temp()?;
            file.write_all_nodes(&nodes)?;
            SequenceTraceBacking::External(Arc::new(file))
        } else {
            SequenceTraceBacking::InMemory(nodes)
        };
        Ok(Self { offsets, backing })
    }

    fn path_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn offsets(&self) -> &[u64] {
        &self.offsets
    }

    fn path(&self, sequence_id: SequenceId) -> Result<Vec<NodeId>> {
        let mut buffer = Vec::new();
        self.read_path_into(sequence_id, &mut buffer)?;
        Ok(buffer)
    }

    fn path_bounds(&self, sequence_index: usize) -> Option<(usize, usize)> {
        let start = *self.offsets.get(sequence_index)?;
        let end = *self.offsets.get(sequence_index + 1)?;
        let start = usize::try_from(start).ok()?;
        let end = usize::try_from(end).ok()?;
        Some((start, end))
    }

    fn read_path_into(&self, sequence_id: SequenceId, buffer: &mut Vec<NodeId>) -> Result<()> {
        self.read_path_into_index(sequence_id.to_usize(), buffer)
    }

    fn read_path_into_index(&self, sequence_index: usize, buffer: &mut Vec<NodeId>) -> Result<()> {
        let Some((start, end)) = self.path_bounds(sequence_index) else {
            return Err(DagError::InvalidRange {
                start: sequence_index,
                end: sequence_index.saturating_add(1),
                len: self.path_count(),
            });
        };
        self.read_nodes(start, end, buffer)
    }

    fn read_nodes(&self, start: usize, end: usize, buffer: &mut Vec<NodeId>) -> Result<()> {
        match &self.backing {
            SequenceTraceBacking::InMemory(nodes) => {
                let path = nodes.get(start..end).ok_or(DagError::InvalidRange {
                    start,
                    end,
                    len: nodes.len(),
                })?;
                buffer.clear();
                buffer.extend_from_slice(path);
                Ok(())
            }
            SequenceTraceBacking::DeferredExternal => {
                if start == end {
                    buffer.clear();
                    Ok(())
                } else {
                    Err(DagError::InvalidStorage(
                        "trace-path external backing is missing stored nodes".to_string(),
                    ))
                }
            }
            SequenceTraceBacking::External(file) => file.read_nodes(start, end, buffer),
        }
    }

    fn snapshot_nodes(&self) -> Result<Vec<NodeId>> {
        let Some(total) = self.offsets.last().copied() else {
            return Err(DagError::InvalidStorage(
                "trace-path offsets must include an initial zero".to_string(),
            ));
        };
        let total = usize::try_from(total).map_err(|_| DagError::ValueDoesNotFit {
            value: total as u128,
            bits: usize::BITS as u8,
        })?;
        let mut nodes = Vec::with_capacity(total);
        self.read_nodes(0, total, &mut nodes)?;
        Ok(nodes)
    }

    fn reader(&self) -> SequenceTracePathReader<'_> {
        SequenceTracePathReader {
            store: self,
            next_index: 0,
            buffer: Vec::new(),
        }
    }

    fn append_node(
        &mut self,
        sequence_id: SequenceId,
        position: ProvenancePosition,
        node_id: NodeId,
    ) -> Result<()> {
        let sequence_index = sequence_id.to_usize();
        let current_len = self.offsets.last().copied().unwrap_or(0);
        while self.path_count() < sequence_index + 1 {
            self.offsets.push(current_len);
        }
        if sequence_index + 1 != self.path_count() {
            return Err(DagError::InvalidRange {
                start: sequence_index,
                end: sequence_index.saturating_add(1),
                len: self.path_count(),
            });
        }
        let expected_position = self
            .path_bounds(sequence_index)
            .map(|(start, end)| end - start)
            .ok_or(DagError::InvalidRange {
                start: sequence_index,
                end: sequence_index.saturating_add(1),
                len: self.path_count(),
            })?;
        let position = usize::try_from(position.raw()).map_err(|_| DagError::ValueDoesNotFit {
            value: position.raw() as u128,
            bits: usize::BITS as u8,
        })?;
        if position != expected_position {
            return Err(DagError::InvalidRange {
                start: position,
                end: position.saturating_add(1),
                len: expected_position,
            });
        }
        match &mut self.backing {
            SequenceTraceBacking::InMemory(nodes) => {
                nodes.push(node_id);
                self.offsets[sequence_index + 1] = nodes.len() as u64;
            }
            SequenceTraceBacking::DeferredExternal | SequenceTraceBacking::External(_) => {
                let store = self.ensure_unique_external_backing()?;
                store.append_node(node_id)?;
                self.offsets[sequence_index + 1] += 1;
            }
        }
        Ok(())
    }

    fn ensure_unique_external_backing(&mut self) -> Result<&ExternalTraceFile> {
        let logical_len = self.offsets.last().copied().unwrap_or(0);
        let logical_len = usize::try_from(logical_len).map_err(|_| DagError::ValueDoesNotFit {
            value: logical_len as u128,
            bits: usize::BITS as u8,
        })?;
        let replacement = match &self.backing {
            SequenceTraceBacking::DeferredExternal => Some(SequenceTraceBacking::External(
                Arc::new(ExternalTraceFile::create_temp()?),
            )),
            SequenceTraceBacking::External(file) if Arc::strong_count(file) > 1 => Some(
                SequenceTraceBacking::External(Arc::new(file.clone_prefix(logical_len)?)),
            ),
            SequenceTraceBacking::InMemory(_) | SequenceTraceBacking::External(_) => None,
        };
        if let Some(backing) = replacement {
            self.backing = backing;
        }
        match &self.backing {
            SequenceTraceBacking::External(file) => Ok(file.as_ref()),
            SequenceTraceBacking::InMemory(_) | SequenceTraceBacking::DeferredExternal => {
                Err(DagError::InvalidStorage(
                    "trace-path storage is not backed by an external store".to_string(),
                ))
            }
        }
    }
}

impl ExternalTraceFile {
    fn create_temp() -> Result<Self> {
        let directory = std::env::temp_dir();
        for _ in 0..1024 {
            let suffix = NEXT_TRACE_PATH_FILE_ID.fetch_add(1, Ordering::Relaxed);
            let path = directory.join(format!(
                "dag-rust-tracepaths-{}-{suffix}.bin",
                std::process::id()
            ));
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(file) => {
                    return Ok(Self {
                        path,
                        state: Mutex::new(ExternalTraceState {
                            file,
                            pending_nodes: Vec::with_capacity(TRACE_PATH_FLUSH_NODE_COUNT),
                        }),
                    });
                }
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) => {
                    return Err(DagError::Io(format!("create {}: {err}", path.display())));
                }
            }
        }
        Err(DagError::Io(
            "failed to allocate a unique trace-path temporary file".to_string(),
        ))
    }

    fn append_node(&self, node_id: NodeId) -> Result<()> {
        let mut state = self.lock_state()?;
        state.pending_nodes.push(node_id);
        if state.pending_nodes.len() >= TRACE_PATH_FLUSH_NODE_COUNT {
            self.flush_locked(&mut state)?;
        }
        Ok(())
    }

    fn write_all_nodes(&self, nodes: &[NodeId]) -> Result<()> {
        let mut state = self.lock_state()?;
        state.pending_nodes.clear();
        state
            .file
            .set_len(0)
            .map_err(|err| DagError::Io(format!("truncate {}: {err}", self.path.display())))?;
        state
            .file
            .seek(SeekFrom::Start(0))
            .map_err(|err| DagError::Io(format!("seek {}: {err}", self.path.display())))?;
        self.write_nodes_locked(&mut state.file, nodes)
    }

    fn read_nodes(&self, start: usize, end: usize, buffer: &mut Vec<NodeId>) -> Result<()> {
        let node_count = end.saturating_sub(start);
        buffer.clear();
        if node_count == 0 {
            return Ok(());
        }
        let start_byte = trace_path_byte_offset(start)?;
        let byte_len = trace_path_byte_offset(node_count)?;
        let byte_len = usize::try_from(byte_len).map_err(|_| DagError::ValueDoesNotFit {
            value: byte_len as u128,
            bits: usize::BITS as u8,
        })?;
        let mut raw = vec![0_u8; byte_len];
        let mut state = self.lock_state()?;
        self.flush_locked(&mut state)?;
        state
            .file
            .seek(SeekFrom::Start(start_byte))
            .map_err(|err| DagError::Io(format!("seek {}: {err}", self.path.display())))?;
        state
            .file
            .read_exact(&mut raw)
            .map_err(|err| DagError::Io(format!("read {}: {err}", self.path.display())))?;
        buffer.reserve(node_count);
        for chunk in raw.chunks_exact(TRACE_PATH_NODE_BYTES as usize) {
            buffer.push(NodeId::new(u32::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
            ])));
        }
        Ok(())
    }

    fn clone_prefix(&self, node_count: usize) -> Result<Self> {
        let mut nodes = Vec::with_capacity(node_count);
        self.read_nodes(0, node_count, &mut nodes)?;
        let cloned = Self::create_temp()?;
        cloned.write_all_nodes(&nodes)?;
        Ok(cloned)
    }

    fn flush_locked(&self, state: &mut ExternalTraceState) -> Result<()> {
        if state.pending_nodes.is_empty() {
            return Ok(());
        }
        state
            .file
            .seek(SeekFrom::End(0))
            .map_err(|err| DagError::Io(format!("seek {}: {err}", self.path.display())))?;
        let pending_nodes = std::mem::take(&mut state.pending_nodes);
        let result = self.write_nodes_locked(&mut state.file, &pending_nodes);
        match result {
            Ok(()) => {
                state.pending_nodes = pending_nodes;
                state.pending_nodes.clear();
                Ok(())
            }
            Err(err) => {
                state.pending_nodes = pending_nodes;
                Err(err)
            }
        }
    }

    fn write_nodes_locked(&self, file: &mut File, nodes: &[NodeId]) -> Result<()> {
        let byte_len = nodes
            .len()
            .checked_mul(TRACE_PATH_NODE_BYTES as usize)
            .ok_or(DagError::ValueDoesNotFit {
                value: nodes.len() as u128 * u128::from(TRACE_PATH_NODE_BYTES),
                bits: usize::BITS as u8,
            })?;
        let mut raw = Vec::with_capacity(byte_len);
        for node_id in nodes {
            raw.extend_from_slice(&node_id.raw().to_le_bytes());
        }
        file.write_all(&raw)
            .map_err(|err| DagError::Io(format!("write {}: {err}", self.path.display())))
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, ExternalTraceState>> {
        self.state.lock().map_err(|_| {
            DagError::Io(format!(
                "lock {}: trace-path temporary file handle was poisoned",
                self.path.display()
            ))
        })
    }
}

fn trace_path_byte_offset(node_count: usize) -> Result<u64> {
    let node_count = u64::try_from(node_count).map_err(|_| DagError::ValueDoesNotFit {
        value: node_count as u128,
        bits: 64,
    })?;
    node_count
        .checked_mul(TRACE_PATH_NODE_BYTES)
        .ok_or(DagError::ValueDoesNotFit {
            value: u128::from(node_count) * u128::from(TRACE_PATH_NODE_BYTES),
            bits: 64,
        })
}

pub(crate) struct SequenceTracePathReader<'a> {
    store: &'a SequenceTraceStore,
    next_index: usize,
    buffer: Vec<NodeId>,
}

impl<'a> SequenceTracePathReader<'a> {
    pub(crate) fn next_path(&mut self) -> Result<Option<(usize, &[NodeId])>> {
        if self.next_index >= self.store.path_count() {
            return Ok(None);
        }
        let sequence_index = self.next_index;
        self.store
            .read_path_into_index(sequence_index, &mut self.buffer)?;
        self.next_index += 1;
        Ok(Some((sequence_index, self.buffer.as_slice())))
    }
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
        sequence_trace_offsets: Vec<u64>,
        sequence_trace_nodes: Vec<NodeId>,
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
}

#[derive(Clone, Debug)]
pub(crate) struct FtoDagSnapshot {
    pub fragment_len: usize,
    pub nodes: Vec<Node>,
    pub edges: Vec<WeightedEdge>,
    pub edge_index_strategy: EdgeIndexStrategy,
    pub provenance: ProvenanceSnapshot,
    pub node_last_sequences: Vec<u32>,
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

    fn snapshot(&self, sequence_trace_paths: &SequenceTraceStore) -> Result<ProvenanceSnapshot> {
        Ok(match self {
            Self::Full(records) => ProvenanceSnapshot::Full(records.clone()),
            Self::Packed32(records) => ProvenanceSnapshot::Packed32(records.clone()),
            Self::TracePaths(node_counts) => ProvenanceSnapshot::TracePaths {
                node_counts: node_counts.clone(),
                sequence_trace_offsets: sequence_trace_paths.offsets().to_vec(),
                sequence_trace_nodes: sequence_trace_paths.snapshot_nodes()?,
            },
            Self::CountOnly(counts) => ProvenanceSnapshot::CountOnly(counts.clone()),
        })
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
            parents: AdjacencyLists::new(),
            children: AdjacencyLists::new(),
            provenance_table: ProvenanceTable::new(),
            node_provenance: NodeProvenanceStorage::new(provenance_storage),
            sequence_trace_paths: if provenance_storage == ProvenanceStorageStrategy::TracePaths {
                SequenceTraceStore::deferred_external()
            } else {
                SequenceTraceStore::default()
            },
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
        self.parents.neighbors(node_id)
    }

    pub fn children(&self, node_id: NodeId) -> Result<&[NodeId]> {
        self.children.neighbors(node_id)
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
        let mut snapshot = self.snapshot()?;
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

    pub fn sequence_trace_path(&self, sequence_id: SequenceId) -> Result<Vec<NodeId>> {
        if !self.retains_sequence_trace_paths() {
            return Err(DagError::UnsupportedOperation(
                "sequence trace paths are only retained with TracePaths provenance storage",
            ));
        }
        self.sequence_trace_paths.path(sequence_id)
    }

    pub(crate) fn sequence_trace_path_count(&self) -> Result<usize> {
        if !self.retains_sequence_trace_paths() {
            return Err(DagError::UnsupportedOperation(
                "sequence trace paths are only retained with TracePaths provenance storage",
            ));
        }
        Ok(self.sequence_trace_paths.path_count())
    }

    pub(crate) fn sequence_trace_paths(&self) -> Result<SequenceTracePathReader<'_>> {
        if !self.retains_sequence_trace_paths() {
            return Err(DagError::UnsupportedOperation(
                "sequence trace paths are only retained with TracePaths provenance storage",
            ));
        }
        Ok(self.sequence_trace_paths.reader())
    }

    pub fn can_node_accept_sequence(
        &self,
        node_id: NodeId,
        sequence_id: SequenceId,
    ) -> Result<bool> {
        let node_index = node_id.to_usize();
        self.node_provenance.ensure_node_exists(node_id)?;
        match self
            .node_last_sequences
            .get(node_index)
            .copied()
            .and_then(sequence_id_from_marker)
        {
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
        self.parents.push_node();
        self.children.push_node();
        self.edge_index.push_node();
        self.node_provenance.push_node();
        self.node_last_sequences.push(MISSING_SEQUENCE_MARKER);
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
        if sequence_id_from_marker(*last_sequence).is_none_or(|last| last < record.sequence_id) {
            *last_sequence = record.sequence_id.raw();
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
        self.sequence_trace_paths
            .append_node(record.sequence_id, record.position, node_id)
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
            self.children.push_neighbor(parent, child)?;
            self.parents.push_neighbor(child, parent)?;
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

    pub(crate) fn compact_storage(&mut self) -> Result<()> {
        self.parents.compact()?;
        self.children.compact()?;
        self.edge_index.compact()?;
        Ok(())
    }

    pub(crate) fn snapshot(&self) -> Result<FtoDagSnapshot> {
        Ok(FtoDagSnapshot {
            fragment_len: self.fragment_len,
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            edge_index_strategy: self.edge_index.strategy(),
            provenance: self.node_provenance.snapshot(&self.sequence_trace_paths)?,
            node_last_sequences: self.node_last_sequences.clone(),
        })
    }

    pub(crate) fn from_snapshot(snapshot: FtoDagSnapshot) -> Result<Self> {
        let FtoDagSnapshot {
            fragment_len,
            nodes,
            edges,
            edge_index_strategy,
            provenance,
            node_last_sequences,
        } = snapshot;

        if nodes.len() != provenance.node_count() {
            return Err(DagError::InvalidStorage(format!(
                "node/provenance length mismatch: {} nodes vs {} provenance slots",
                nodes.len(),
                provenance.node_count()
            )));
        }
        if nodes.len() != node_last_sequences.len() {
            return Err(DagError::InvalidStorage(format!(
                "node/node-last-sequence length mismatch: {} nodes vs {} markers",
                nodes.len(),
                node_last_sequences.len()
            )));
        }
        let node_count = provenance.node_count();
        let node_provenance = NodeProvenanceStorage::from_snapshot(&provenance);
        let sequence_trace_paths = match provenance {
            ProvenanceSnapshot::TracePaths {
                sequence_trace_offsets,
                sequence_trace_nodes,
                ..
            } => {
                SequenceTraceStore::from_parts(sequence_trace_offsets, sequence_trace_nodes, true)?
            }
            ProvenanceSnapshot::Full(_)
            | ProvenanceSnapshot::Packed32(_)
            | ProvenanceSnapshot::CountOnly(_) => SequenceTraceStore::default(),
        };
        let parents = AdjacencyLists::from_edges(node_count, &edges, false)?;
        let children = AdjacencyLists::from_edges(node_count, &edges, true)?;

        let mut graph = Self {
            fragment_len,
            nodes,
            edges,
            edge_index: EdgeIndexStorage::new(edge_index_strategy),
            parents,
            children,
            provenance_table: ProvenanceTable::new(),
            node_provenance,
            sequence_trace_paths,
            node_last_sequences,
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
            graph
                .endpoint_index
                .record_edge_insertion(edge.key.parent, edge.key.child);
        }

        graph.compact_storage()?;

        Ok(graph)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub fragment_len: usize,
}
