//! Sequence records and streaming inputs.

use crate::foundations::error::Result;
use crate::sequence_model::alphabet::{Alphabet, SymbolId};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequenceRecord {
    pub id: String,
    pub symbols: String,
}

impl SequenceRecord {
    pub fn new(id: impl Into<String>, symbols: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            symbols: symbols.into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncodedSequence {
    id: String,
    symbols: Vec<SymbolId>,
}

impl EncodedSequence {
    pub fn encode(record: SequenceRecord, alphabet: &dyn Alphabet) -> Result<Self> {
        let symbols = record
            .symbols
            .chars()
            .map(|symbol| alphabet.encode(symbol))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            id: record.id,
            symbols,
        })
    }

    pub fn from_symbols(id: impl Into<String>, symbols: Vec<SymbolId>) -> Self {
        Self {
            id: id.into(),
            symbols,
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn symbols(&self) -> &[SymbolId] {
        &self.symbols
    }

    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, SymbolId> {
        self.symbols.iter()
    }
}

pub trait SequenceInput {
    fn next_record(&mut self) -> Result<Option<SequenceRecord>>;
}

#[derive(Clone, Debug, Default)]
pub struct VecSequenceInput {
    records: Vec<SequenceRecord>,
    cursor: usize,
}

impl VecSequenceInput {
    pub fn new(records: Vec<SequenceRecord>) -> Self {
        Self { records, cursor: 0 }
    }
}

impl SequenceInput for VecSequenceInput {
    fn next_record(&mut self) -> Result<Option<SequenceRecord>> {
        if self.cursor >= self.records.len() {
            return Ok(None);
        }
        let record = self.records[self.cursor].clone();
        self.cursor += 1;
        Ok(Some(record))
    }
}
