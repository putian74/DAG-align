//! Sequence and source-provenance contracts for preprocessing artifacts.

use crate::error::{PreAdPrepError, Result};
use crate::validate::{Validate, ValidationReport};

/// How source records were decoded from legacy graph artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SourceDecodeStatus {
    DecodedOsm,
    RawOsmPacked,
    RawOnmTraceability,
    #[default]
    Missing,
}

/// Table of sequence identifiers and UTF-8 names.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SequenceTable {
    pub sequence_ids: Vec<u64>,
    pub sequence_name_offsets: Vec<usize>,
    pub sequence_name_bytes: Vec<u8>,
}

impl SequenceTable {
    pub fn sequence_count(&self) -> usize {
        self.sequence_ids.len()
    }
}

impl Validate for SequenceTable {
    fn validate(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();
        if self.sequence_name_offsets.len() != self.sequence_ids.len() + 1 {
            report.error(
                "sequence_name_offsets_shape",
                "sequence_name_offsets length must equal sequence_count + 1",
            );
        }
        if let Some(last) = self.sequence_name_offsets.last().copied()
            && last != self.sequence_name_bytes.len()
        {
            report.error(
                "sequence_name_offsets_terminal",
                "last sequence_name_offset must equal name byte length",
            );
        }
        for window in self.sequence_name_offsets.windows(2) {
            if window[1] < window[0] {
                report.error(
                    "sequence_name_offsets_order",
                    "sequence name offsets must be monotonic",
                );
            }
        }
        Ok(report)
    }
}

/// Flat source-record payloads plus optional decoded sequence IDs/positions.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceRecordTable {
    pub packed_records: Vec<u64>,
    pub source_sequence_id: Option<Vec<u64>>,
    pub source_position: Option<Vec<u64>>,
    pub decode_status: SourceDecodeStatus,
}

impl SourceRecordTable {
    pub fn record_count(&self) -> usize {
        self.packed_records.len()
    }
}

impl Validate for SourceRecordTable {
    fn validate(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();
        if let Some(sequence_ids) = &self.source_sequence_id
            && sequence_ids.len() != self.packed_records.len()
        {
            report.error(
                "source_sequence_id_shape",
                "decoded source_sequence_id length must match packed_records",
            );
        }
        if let Some(positions) = &self.source_position
            && positions.len() != self.packed_records.len()
        {
            report.error(
                "source_position_shape",
                "decoded source_position length must match packed_records",
            );
        }
        Ok(report)
    }
}

/// Per-node ranges into the flat source-record table.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NodeSourceRanges {
    pub node_source_offset: Vec<usize>,
    pub node_source_len: Vec<usize>,
}

impl NodeSourceRanges {
    pub fn validate_for_record_count(&self, record_count: usize) -> Result<()> {
        if self.node_source_offset.len() != self.node_source_len.len() {
            return Err(PreAdPrepError::Validation(
                "node source offset/length arrays must match".into(),
            ));
        }
        for (offset, length) in self
            .node_source_offset
            .iter()
            .copied()
            .zip(self.node_source_len.iter().copied())
        {
            if offset + length > record_count {
                return Err(PreAdPrepError::Validation(format!(
                    "node source range [{offset}, {}) exceeds record_count {record_count}",
                    offset + length
                )));
            }
        }
        Ok(())
    }
}

impl Validate for NodeSourceRanges {
    fn validate(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();
        if self.node_source_offset.len() != self.node_source_len.len() {
            report.error(
                "node_source_ranges_shape",
                "node_source_offset length must match node_source_len",
            );
        }
        Ok(report)
    }
}

/// Full source/provenance bundle attached to one tensor graph artifact.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceTables {
    pub sequences: Option<SequenceTable>,
    pub records: SourceRecordTable,
    pub node_ranges: NodeSourceRanges,
}

impl Validate for SourceTables {
    fn validate(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();
        if let Some(sequences) = &self.sequences {
            report.issues.extend(sequences.validate()?.issues);
        }
        report.issues.extend(self.records.validate()?.issues);
        report.issues.extend(self.node_ranges.validate()?.issues);
        if let Err(error) = self
            .node_ranges
            .validate_for_record_count(self.records.record_count())
        {
            report.error("node_source_record_bounds", error.to_string());
        }
        Ok(report)
    }
}
