//! Native graph storage and export profile interfaces.

use crate::foundations::error::{DagError, Result};
use crate::foundations::id::{NodeId, ProvenancePosition, SequenceId, Weight};
use crate::graph_model::graph::{
    EdgeIndexStrategy, EdgeKey, FtoDag, FtoDagSnapshot, Node, NodeKind, ProvenanceSnapshot,
    WeightedEdge,
};
use crate::graph_model::provenance::{
    PackedProvenanceRecord, ProvenanceRange, ProvenanceRecord, ProvenanceStorageStrategy,
};
use crate::sequence_model::alphabet::SymbolId;
use crate::sequence_model::fragment::FragmentKey;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

const STORAGE_MAGIC: [u8; 8] = *b"DAGRUST\0";

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct GraphFormatVersion {
    pub major: u16,
    pub minor: u16,
}

impl GraphFormatVersion {
    pub const CURRENT: Self = Self { major: 0, minor: 1 };
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct StorageConfig {
    pub version: GraphFormatVersion,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            version: GraphFormatVersion::CURRENT,
        }
    }
}

pub trait GraphStorage {
    fn save_graph(&self, graph: &FtoDag, path: &Path, config: StorageConfig) -> Result<()>;
    fn load_graph(&self, path: &Path) -> Result<FtoDag>;
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct NativeGraphStorage;

impl GraphStorage for NativeGraphStorage {
    fn save_graph(&self, graph: &FtoDag, path: &Path, config: StorageConfig) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|err| DagError::Io(format!("create {}: {err}", parent.display())))?;
        }
        let file = File::create(path)
            .map_err(|err| DagError::Io(format!("create {}: {err}", path.display())))?;
        let mut writer = BufWriter::new(file);
        let snapshot = graph.snapshot();

        writer
            .write_all(&STORAGE_MAGIC)
            .map_err(|err| DagError::Io(format!("write {}: {err}", path.display())))?;
        write_u32(&mut writer, packed_version(config.version), path)?;
        write_u64(&mut writer, snapshot.fragment_len as u64, path)?;
        write_u8(
            &mut writer,
            encode_edge_index_strategy(snapshot.edge_index_strategy),
            path,
        )?;
        write_u8(
            &mut writer,
            encode_provenance_strategy(snapshot.provenance.strategy()),
            path,
        )?;
        write_u64(&mut writer, snapshot.nodes.len() as u64, path)?;
        write_u64(&mut writer, snapshot.edges.len() as u64, path)?;

        for node in &snapshot.nodes {
            write_node(&mut writer, node, path)?;
        }
        for edge in &snapshot.edges {
            write_edge(&mut writer, edge, path)?;
        }
        write_optional_sequence_ids(&mut writer, &snapshot.node_last_sequences, path)?;
        write_provenance_snapshot(&mut writer, &snapshot.provenance, path)?;

        writer
            .flush()
            .map_err(|err| DagError::Io(format!("flush {}: {err}", path.display())))
    }

    fn load_graph(&self, path: &Path) -> Result<FtoDag> {
        let file = File::open(path)
            .map_err(|err| DagError::Io(format!("open {}: {err}", path.display())))?;
        let mut reader = BufReader::new(file);
        let mut magic = [0_u8; STORAGE_MAGIC.len()];
        reader
            .read_exact(&mut magic)
            .map_err(|err| DagError::Io(format!("read {}: {err}", path.display())))?;
        if magic != STORAGE_MAGIC {
            return Err(DagError::InvalidStorage(format!(
                "{} does not contain a DAG-rust graph",
                path.display()
            )));
        }

        let found_version = read_u32(&mut reader, path)?;
        let expected_version = packed_version(GraphFormatVersion::CURRENT);
        if found_version != expected_version {
            return Err(DagError::StorageVersionMismatch {
                expected: expected_version,
                found: found_version,
            });
        }

        let fragment_len = read_usize(&mut reader, path)?;
        let edge_index_strategy = decode_edge_index_strategy(read_u8(&mut reader, path)?)?;
        let provenance_strategy = decode_provenance_strategy(read_u8(&mut reader, path)?)?;
        let node_count = read_usize(&mut reader, path)?;
        let edge_count = read_usize(&mut reader, path)?;

        let mut nodes = Vec::with_capacity(node_count);
        for _ in 0..node_count {
            nodes.push(read_node(&mut reader, path)?);
        }
        let mut edges = Vec::with_capacity(edge_count);
        for _ in 0..edge_count {
            edges.push(read_edge(&mut reader, path)?);
        }
        let node_last_sequences = read_optional_sequence_ids(&mut reader, node_count, path)?;
        let provenance =
            read_provenance_snapshot(&mut reader, provenance_strategy, node_count, path)?;

        FtoDag::from_snapshot(FtoDagSnapshot {
            fragment_len,
            nodes,
            edges,
            edge_index_strategy,
            provenance,
            node_last_sequences,
        })
    }
}

fn packed_version(version: GraphFormatVersion) -> u32 {
    (u32::from(version.major) << 16) | u32::from(version.minor)
}

fn write_node(writer: &mut impl Write, node: &Node, path: &Path) -> Result<()> {
    write_u32(writer, node.id.raw(), path)?;
    write_fragment_key(writer, &node.fragment, path)?;
    write_u8(writer, encode_node_kind(node.kind), path)?;
    write_u64(writer, node.weight.raw(), path)?;
    write_u16(writer, node.flags.bits(), path)?;
    write_u64(writer, node.provenance_range.start, path)?;
    write_u64(writer, node.provenance_range.len, path)
}

fn read_node(reader: &mut impl Read, path: &Path) -> Result<Node> {
    Ok(Node {
        id: NodeId::new(read_u32(reader, path)?),
        fragment: read_fragment_key(reader, path)?,
        kind: decode_node_kind(read_u8(reader, path)?)?,
        weight: Weight::new(read_u64(reader, path)?),
        flags: crate::foundations::bit_encoding::NodeFlags::from_bits(read_u16(reader, path)?),
        provenance_range: ProvenanceRange::new(read_u64(reader, path)?, read_u64(reader, path)?),
    })
}

fn write_edge(writer: &mut impl Write, edge: &WeightedEdge, path: &Path) -> Result<()> {
    write_u32(writer, edge.key.parent.raw(), path)?;
    write_u32(writer, edge.key.child.raw(), path)?;
    write_u64(writer, edge.weight.raw(), path)
}

fn read_edge(reader: &mut impl Read, path: &Path) -> Result<WeightedEdge> {
    Ok(WeightedEdge {
        key: EdgeKey {
            parent: NodeId::new(read_u32(reader, path)?),
            child: NodeId::new(read_u32(reader, path)?),
        },
        weight: Weight::new(read_u64(reader, path)?),
    })
}

fn write_provenance_snapshot(
    writer: &mut impl Write,
    provenance: &ProvenanceSnapshot,
    path: &Path,
) -> Result<()> {
    match provenance {
        ProvenanceSnapshot::Full(records) => write_record_lists(writer, records, path),
        ProvenanceSnapshot::Packed32(records) => {
            write_u64(writer, records.len() as u64, path)?;
            for node_records in records {
                write_u64(writer, node_records.len() as u64, path)?;
                for record in node_records {
                    write_provenance_record(writer, record.unpack(), path)?;
                }
            }
            Ok(())
        }
        ProvenanceSnapshot::TracePaths {
            node_counts,
            sequence_trace_paths,
        } => {
            write_u64(writer, node_counts.len() as u64, path)?;
            for count in node_counts {
                write_u64(writer, *count, path)?;
            }
            write_u64(writer, sequence_trace_paths.len() as u64, path)?;
            for trace_path in sequence_trace_paths {
                write_u64(writer, trace_path.len() as u64, path)?;
                for node_id in trace_path {
                    write_u32(writer, node_id.raw(), path)?;
                }
            }
            Ok(())
        }
        ProvenanceSnapshot::CountOnly(counts) => {
            write_u64(writer, counts.len() as u64, path)?;
            for count in counts {
                write_u64(writer, *count, path)?;
            }
            Ok(())
        }
    }
}

fn read_provenance_snapshot(
    reader: &mut impl Read,
    strategy: ProvenanceStorageStrategy,
    node_count: usize,
    path: &Path,
) -> Result<ProvenanceSnapshot> {
    match strategy {
        ProvenanceStorageStrategy::FullRecords => Ok(ProvenanceSnapshot::Full(read_record_lists(
            reader, node_count, path,
        )?)),
        ProvenanceStorageStrategy::Packed32 => {
            let records = read_record_lists(reader, node_count, path)?
                .into_iter()
                .map(|node_records| {
                    node_records
                        .into_iter()
                        .map(PackedProvenanceRecord::try_from_record)
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(ProvenanceSnapshot::Packed32(records))
        }
        ProvenanceStorageStrategy::TracePaths => {
            let stored_node_count = read_usize(reader, path)?;
            if stored_node_count != node_count {
                return Err(DagError::InvalidStorage(format!(
                    "trace-path node-count mismatch: expected {node_count}, found {stored_node_count}"
                )));
            }
            let mut node_counts = Vec::with_capacity(node_count);
            for _ in 0..node_count {
                node_counts.push(read_u64(reader, path)?);
            }
            let path_count = read_usize(reader, path)?;
            let mut sequence_trace_paths = Vec::with_capacity(path_count);
            for _ in 0..path_count {
                let len = read_usize(reader, path)?;
                let mut trace_path = Vec::with_capacity(len);
                for _ in 0..len {
                    trace_path.push(NodeId::new(read_u32(reader, path)?));
                }
                sequence_trace_paths.push(trace_path);
            }
            Ok(ProvenanceSnapshot::TracePaths {
                node_counts,
                sequence_trace_paths,
            })
        }
        ProvenanceStorageStrategy::CountOnly => {
            let stored_node_count = read_usize(reader, path)?;
            if stored_node_count != node_count {
                return Err(DagError::InvalidStorage(format!(
                    "count-only node-count mismatch: expected {node_count}, found {stored_node_count}"
                )));
            }
            let mut counts = Vec::with_capacity(node_count);
            for _ in 0..node_count {
                counts.push(read_u64(reader, path)?);
            }
            Ok(ProvenanceSnapshot::CountOnly(counts))
        }
    }
}

fn write_record_lists(
    writer: &mut impl Write,
    records: &[Vec<ProvenanceRecord>],
    path: &Path,
) -> Result<()> {
    write_u64(writer, records.len() as u64, path)?;
    for node_records in records {
        write_u64(writer, node_records.len() as u64, path)?;
        for record in node_records {
            write_provenance_record(writer, *record, path)?;
        }
    }
    Ok(())
}

fn read_record_lists(
    reader: &mut impl Read,
    node_count: usize,
    path: &Path,
) -> Result<Vec<Vec<ProvenanceRecord>>> {
    let stored_node_count = read_usize(reader, path)?;
    if stored_node_count != node_count {
        return Err(DagError::InvalidStorage(format!(
            "record-list node-count mismatch: expected {node_count}, found {stored_node_count}"
        )));
    }
    let mut records = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        let record_count = read_usize(reader, path)?;
        let mut node_records = Vec::with_capacity(record_count);
        for _ in 0..record_count {
            node_records.push(read_provenance_record(reader, path)?);
        }
        records.push(node_records);
    }
    Ok(records)
}

fn write_provenance_record(
    writer: &mut impl Write,
    record: ProvenanceRecord,
    path: &Path,
) -> Result<()> {
    write_u32(writer, record.sequence_id.raw(), path)?;
    write_u64(writer, record.position.raw(), path)
}

fn read_provenance_record(reader: &mut impl Read, path: &Path) -> Result<ProvenanceRecord> {
    Ok(ProvenanceRecord {
        sequence_id: SequenceId::new(read_u32(reader, path)?),
        position: ProvenancePosition::new(read_u64(reader, path)?),
    })
}

fn write_fragment_key(writer: &mut impl Write, key: &FragmentKey, path: &Path) -> Result<()> {
    match key {
        FragmentKey::PackedInline {
            bits_per_symbol,
            len,
            value,
        } => {
            write_u8(writer, 0, path)?;
            write_u8(writer, *bits_per_symbol, path)?;
            write_u16(writer, *len, path)?;
            write_u128(writer, *value, path)
        }
        FragmentKey::PackedWords {
            bits_per_symbol,
            len,
            words,
        } => {
            write_u8(writer, 1, path)?;
            write_u8(writer, *bits_per_symbol, path)?;
            write_u32(writer, *len, path)?;
            write_u64(writer, words.len() as u64, path)?;
            for word in words {
                write_u64(writer, *word, path)?;
            }
            Ok(())
        }
        FragmentKey::Symbols(symbols) => {
            write_u8(writer, 2, path)?;
            write_u64(writer, symbols.len() as u64, path)?;
            for symbol in symbols {
                write_u16(writer, symbol.raw(), path)?;
            }
            Ok(())
        }
    }
}

fn read_fragment_key(reader: &mut impl Read, path: &Path) -> Result<FragmentKey> {
    match read_u8(reader, path)? {
        0 => Ok(FragmentKey::PackedInline {
            bits_per_symbol: read_u8(reader, path)?,
            len: read_u16(reader, path)?,
            value: read_u128(reader, path)?,
        }),
        1 => {
            let bits_per_symbol = read_u8(reader, path)?;
            let len = read_u32(reader, path)?;
            let word_count = read_usize(reader, path)?;
            let mut words = Vec::with_capacity(word_count);
            for _ in 0..word_count {
                words.push(read_u64(reader, path)?);
            }
            Ok(FragmentKey::PackedWords {
                bits_per_symbol,
                len,
                words,
            })
        }
        2 => {
            let symbol_count = read_usize(reader, path)?;
            let mut symbols = Vec::with_capacity(symbol_count);
            for _ in 0..symbol_count {
                symbols.push(SymbolId::new(read_u16(reader, path)?));
            }
            Ok(FragmentKey::Symbols(symbols))
        }
        tag => Err(DagError::InvalidStorage(format!(
            "unknown fragment-key tag {tag} in {}",
            path.display()
        ))),
    }
}

fn write_optional_sequence_ids(
    writer: &mut impl Write,
    values: &[Option<SequenceId>],
    path: &Path,
) -> Result<()> {
    write_u64(writer, values.len() as u64, path)?;
    for value in values {
        match value {
            Some(sequence_id) => {
                write_u8(writer, 1, path)?;
                write_u32(writer, sequence_id.raw(), path)?;
            }
            None => write_u8(writer, 0, path)?,
        }
    }
    Ok(())
}

fn read_optional_sequence_ids(
    reader: &mut impl Read,
    expected_len: usize,
    path: &Path,
) -> Result<Vec<Option<SequenceId>>> {
    let stored_len = read_usize(reader, path)?;
    if stored_len != expected_len {
        return Err(DagError::InvalidStorage(format!(
            "node-last-sequence length mismatch: expected {expected_len}, found {stored_len}"
        )));
    }
    let mut values = Vec::with_capacity(expected_len);
    for _ in 0..expected_len {
        match read_u8(reader, path)? {
            0 => values.push(None),
            1 => values.push(Some(SequenceId::new(read_u32(reader, path)?))),
            tag => {
                return Err(DagError::InvalidStorage(format!(
                    "unknown optional-sequence-id tag {tag} in {}",
                    path.display()
                )));
            }
        }
    }
    Ok(values)
}

fn encode_node_kind(kind: NodeKind) -> u8 {
    match kind {
        NodeKind::Start => 0,
        NodeKind::Internal => 1,
        NodeKind::End => 2,
        NodeKind::Singleton => 3,
    }
}

fn decode_node_kind(tag: u8) -> Result<NodeKind> {
    match tag {
        0 => Ok(NodeKind::Start),
        1 => Ok(NodeKind::Internal),
        2 => Ok(NodeKind::End),
        3 => Ok(NodeKind::Singleton),
        _ => Err(DagError::InvalidStorage(format!(
            "unknown node-kind tag {tag}"
        ))),
    }
}

fn encode_provenance_strategy(strategy: ProvenanceStorageStrategy) -> u8 {
    match strategy {
        ProvenanceStorageStrategy::FullRecords => 0,
        ProvenanceStorageStrategy::Packed32 => 1,
        ProvenanceStorageStrategy::TracePaths => 2,
        ProvenanceStorageStrategy::CountOnly => 3,
    }
}

fn decode_provenance_strategy(tag: u8) -> Result<ProvenanceStorageStrategy> {
    match tag {
        0 => Ok(ProvenanceStorageStrategy::FullRecords),
        1 => Ok(ProvenanceStorageStrategy::Packed32),
        2 => Ok(ProvenanceStorageStrategy::TracePaths),
        3 => Ok(ProvenanceStorageStrategy::CountOnly),
        _ => Err(DagError::InvalidStorage(format!(
            "unknown provenance-strategy tag {tag}"
        ))),
    }
}

fn encode_edge_index_strategy(strategy: EdgeIndexStrategy) -> u8 {
    match strategy {
        EdgeIndexStrategy::GlobalHash => 0,
        EdgeIndexStrategy::LowDegreeHybrid => 1,
    }
}

fn decode_edge_index_strategy(tag: u8) -> Result<EdgeIndexStrategy> {
    match tag {
        0 => Ok(EdgeIndexStrategy::GlobalHash),
        1 => Ok(EdgeIndexStrategy::LowDegreeHybrid),
        _ => Err(DagError::InvalidStorage(format!(
            "unknown edge-index-strategy tag {tag}"
        ))),
    }
}

fn write_u8(writer: &mut impl Write, value: u8, path: &Path) -> Result<()> {
    writer
        .write_all(&[value])
        .map_err(|err| DagError::Io(format!("write {}: {err}", path.display())))
}

fn write_u16(writer: &mut impl Write, value: u16, path: &Path) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|err| DagError::Io(format!("write {}: {err}", path.display())))
}

fn write_u32(writer: &mut impl Write, value: u32, path: &Path) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|err| DagError::Io(format!("write {}: {err}", path.display())))
}

fn write_u64(writer: &mut impl Write, value: u64, path: &Path) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|err| DagError::Io(format!("write {}: {err}", path.display())))
}

fn write_u128(writer: &mut impl Write, value: u128, path: &Path) -> Result<()> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|err| DagError::Io(format!("write {}: {err}", path.display())))
}

fn read_u8(reader: &mut impl Read, path: &Path) -> Result<u8> {
    let mut buffer = [0_u8; 1];
    reader
        .read_exact(&mut buffer)
        .map_err(|err| DagError::Io(format!("read {}: {err}", path.display())))?;
    Ok(buffer[0])
}

fn read_u16(reader: &mut impl Read, path: &Path) -> Result<u16> {
    let mut buffer = [0_u8; 2];
    reader
        .read_exact(&mut buffer)
        .map_err(|err| DagError::Io(format!("read {}: {err}", path.display())))?;
    Ok(u16::from_le_bytes(buffer))
}

fn read_u32(reader: &mut impl Read, path: &Path) -> Result<u32> {
    let mut buffer = [0_u8; 4];
    reader
        .read_exact(&mut buffer)
        .map_err(|err| DagError::Io(format!("read {}: {err}", path.display())))?;
    Ok(u32::from_le_bytes(buffer))
}

fn read_u64(reader: &mut impl Read, path: &Path) -> Result<u64> {
    let mut buffer = [0_u8; 8];
    reader
        .read_exact(&mut buffer)
        .map_err(|err| DagError::Io(format!("read {}: {err}", path.display())))?;
    Ok(u64::from_le_bytes(buffer))
}

fn read_u128(reader: &mut impl Read, path: &Path) -> Result<u128> {
    let mut buffer = [0_u8; 16];
    reader
        .read_exact(&mut buffer)
        .map_err(|err| DagError::Io(format!("read {}: {err}", path.display())))?;
    Ok(u128::from_le_bytes(buffer))
}

fn read_usize(reader: &mut impl Read, path: &Path) -> Result<usize> {
    usize::try_from(read_u64(reader, path)?).map_err(|_| {
        DagError::InvalidStorage(format!("value in {} does not fit usize", path.display()))
    })
}
