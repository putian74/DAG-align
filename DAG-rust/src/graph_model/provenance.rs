//! Provenance traceability records and compact provenance tables.

use crate::foundations::error::{DagError, Result};
use crate::foundations::id::{ProvenancePosition, SequenceId};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ProvenanceRecord {
    pub sequence_id: SequenceId,
    pub position: ProvenancePosition,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum ProvenanceStorageStrategy {
    #[default]
    FullRecords,
    Packed32,
    TracePaths,
    CountOnly,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct PackedProvenanceRecord {
    raw: u64,
}

impl PackedProvenanceRecord {
    pub fn try_from_record(record: ProvenanceRecord) -> Result<Self> {
        let position =
            u32::try_from(record.position.raw()).map_err(|_| DagError::ValueDoesNotFit {
                value: record.position.raw() as u128,
                bits: 32,
            })?;
        Ok(Self {
            raw: (u64::from(record.sequence_id.raw()) << 32) | u64::from(position),
        })
    }

    pub fn unpack(self) -> ProvenanceRecord {
        ProvenanceRecord {
            sequence_id: SequenceId::new((self.raw >> 32) as u32),
            position: ProvenancePosition::new(self.raw & u64::from(u32::MAX)),
        }
    }

    pub fn sequence_id(self) -> SequenceId {
        SequenceId::new((self.raw >> 32) as u32)
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct ProvenanceRange {
    pub start: u64,
    pub len: u64,
}

impl ProvenanceRange {
    pub const fn new(start: u64, len: u64) -> Self {
        Self { start, len }
    }

    pub const fn is_empty(self) -> bool {
        self.len == 0
    }

    pub const fn end(self) -> u64 {
        self.start + self.len
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProvenanceTable {
    records: Vec<ProvenanceRecord>,
}

impl ProvenanceTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, record: ProvenanceRecord) {
        self.records.push(record);
    }

    pub fn append(&mut self, record: ProvenanceRecord) -> ProvenanceRange {
        let start = self.records.len() as u64;
        self.records.push(record);
        ProvenanceRange::new(start, 1)
    }

    pub fn append_many<I>(&mut self, records: I) -> ProvenanceRange
    where
        I: IntoIterator<Item = ProvenanceRecord>,
    {
        let start = self.records.len() as u64;
        self.records.extend(records);
        ProvenanceRange::new(start, self.records.len() as u64 - start)
    }

    pub fn records(&self) -> &[ProvenanceRecord] {
        &self.records
    }

    pub fn records_for(&self, range: ProvenanceRange) -> Result<&[ProvenanceRecord]> {
        let start = range.start as usize;
        let end = range.end() as usize;
        if start > end || end > self.records.len() {
            return Err(DagError::InvalidRange {
                start,
                end,
                len: self.records.len(),
            });
        }
        Ok(&self.records[start..end])
    }
}
