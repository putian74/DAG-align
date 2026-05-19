//! Source traceability records and compact source tables.

use crate::foundations::error::{DagError, Result};
use crate::foundations::id::{SequenceId, SourcePosition};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct SourceRecord {
    pub sequence_id: SequenceId,
    pub position: SourcePosition,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum SourceStorageStrategy {
    #[default]
    FullRecords,
    Packed32,
    TracePaths,
    CountOnly,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct PackedSourceRecord {
    raw: u64,
}

impl PackedSourceRecord {
    pub fn try_from_record(record: SourceRecord) -> Result<Self> {
        let position =
            u32::try_from(record.position.raw()).map_err(|_| DagError::ValueDoesNotFit {
                value: record.position.raw() as u128,
                bits: 32,
            })?;
        Ok(Self {
            raw: (u64::from(record.sequence_id.raw()) << 32) | u64::from(position),
        })
    }

    pub fn unpack(self) -> SourceRecord {
        SourceRecord {
            sequence_id: SequenceId::new((self.raw >> 32) as u32),
            position: SourcePosition::new(self.raw & u64::from(u32::MAX)),
        }
    }

    pub fn sequence_id(self) -> SequenceId {
        SequenceId::new((self.raw >> 32) as u32)
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct SourceRange {
    pub start: u64,
    pub len: u64,
}

impl SourceRange {
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
pub struct SourceTable {
    records: Vec<SourceRecord>,
}

impl SourceTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, record: SourceRecord) {
        self.records.push(record);
    }

    pub fn append(&mut self, record: SourceRecord) -> SourceRange {
        let start = self.records.len() as u64;
        self.records.push(record);
        SourceRange::new(start, 1)
    }

    pub fn append_many<I>(&mut self, records: I) -> SourceRange
    where
        I: IntoIterator<Item = SourceRecord>,
    {
        let start = self.records.len() as u64;
        self.records.extend(records);
        SourceRange::new(start, self.records.len() as u64 - start)
    }

    pub fn records(&self) -> &[SourceRecord] {
        &self.records
    }

    pub fn records_for(&self, range: SourceRange) -> Result<&[SourceRecord]> {
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
