//! Alphabet definitions and explicit normalization policies.

use crate::foundations::error::{DagError, Result};
use std::collections::HashMap;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct SymbolId(u16);

impl SymbolId {
    pub const fn new(raw: u16) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u16 {
        self.0
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum AlphabetKind {
    Dna,
    Rna,
    Protein,
    Custom,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum NormalizationPolicy {
    Strict,
    CaseFold,
    RnaToDna,
    DnaToRna,
    Custom,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum AmbiguityPolicy {
    ExactSymbol,
    Reject,
    EquivalenceClass,
}

pub trait Alphabet {
    fn kind(&self) -> AlphabetKind;
    fn normalization(&self) -> NormalizationPolicy;
    fn ambiguity_policy(&self) -> AmbiguityPolicy;
    fn symbol_count(&self) -> usize;
    fn bits_per_symbol(&self) -> u8 {
        bits_needed(self.symbol_count())
    }
    fn encode(&self, symbol: char) -> Result<SymbolId>;
    fn decode(&self, symbol: SymbolId) -> Option<char>;
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BuiltinAlphabet {
    kind: AlphabetKind,
    normalization: NormalizationPolicy,
    ambiguity_policy: AmbiguityPolicy,
}

impl BuiltinAlphabet {
    pub const fn new(
        kind: AlphabetKind,
        normalization: NormalizationPolicy,
        ambiguity_policy: AmbiguityPolicy,
    ) -> Self {
        Self {
            kind,
            normalization,
            ambiguity_policy,
        }
    }

    pub const fn dna_exact() -> Self {
        Self::new(
            AlphabetKind::Dna,
            NormalizationPolicy::CaseFold,
            AmbiguityPolicy::ExactSymbol,
        )
    }

    pub const fn dna_canonical() -> Self {
        Self::new(
            AlphabetKind::Dna,
            NormalizationPolicy::CaseFold,
            AmbiguityPolicy::Reject,
        )
    }

    pub const fn rna_exact(normalization: NormalizationPolicy) -> Self {
        Self::new(
            AlphabetKind::Rna,
            normalization,
            AmbiguityPolicy::ExactSymbol,
        )
    }

    pub const fn protein_exact() -> Self {
        Self::new(
            AlphabetKind::Protein,
            NormalizationPolicy::CaseFold,
            AmbiguityPolicy::ExactSymbol,
        )
    }

    pub fn normalize_char(&self, symbol: char) -> char {
        let symbol = match self.normalization {
            NormalizationPolicy::CaseFold
            | NormalizationPolicy::RnaToDna
            | NormalizationPolicy::DnaToRna => symbol.to_ascii_uppercase(),
            NormalizationPolicy::Strict | NormalizationPolicy::Custom => symbol,
        };
        match self.normalization {
            NormalizationPolicy::RnaToDna if symbol == 'U' => 'T',
            NormalizationPolicy::DnaToRna if symbol == 'T' => 'U',
            _ => symbol,
        }
    }
}

impl Alphabet for BuiltinAlphabet {
    fn kind(&self) -> AlphabetKind {
        self.kind
    }

    fn normalization(&self) -> NormalizationPolicy {
        self.normalization
    }

    fn ambiguity_policy(&self) -> AmbiguityPolicy {
        self.ambiguity_policy
    }

    fn symbol_count(&self) -> usize {
        match self.kind {
            AlphabetKind::Dna => dna_symbols(self.ambiguity_policy).len(),
            AlphabetKind::Rna => rna_symbols(self.ambiguity_policy).len(),
            AlphabetKind::Protein => protein_symbols(self.ambiguity_policy).len(),
            AlphabetKind::Custom => 0,
        }
    }

    fn encode(&self, symbol: char) -> Result<SymbolId> {
        let normalized = self.normalize_char(symbol);
        let code = match self.kind {
            AlphabetKind::Dna => dna_code(normalized, self.ambiguity_policy),
            AlphabetKind::Rna => rna_code(normalized, self.ambiguity_policy),
            AlphabetKind::Protein => protein_code(normalized, self.ambiguity_policy),
            AlphabetKind::Custom => None,
        };
        code.map(SymbolId::new)
            .ok_or_else(|| DagError::InvalidSymbol {
                symbol: symbol.to_string(),
            })
    }

    fn decode(&self, symbol: SymbolId) -> Option<char> {
        match self.kind {
            AlphabetKind::Dna => dna_symbols(self.ambiguity_policy)
                .chars()
                .nth(symbol.raw() as usize),
            AlphabetKind::Rna => rna_symbols(self.ambiguity_policy)
                .chars()
                .nth(symbol.raw() as usize),
            AlphabetKind::Protein => protein_symbols(self.ambiguity_policy)
                .chars()
                .nth(symbol.raw() as usize),
            AlphabetKind::Custom => None,
        }
    }
}

fn dna_code(symbol: char, ambiguity_policy: AmbiguityPolicy) -> Option<u16> {
    dna_symbols(ambiguity_policy)
        .chars()
        .position(|candidate| candidate == symbol)
        .map(|x| x as u16)
}

fn rna_code(symbol: char, ambiguity_policy: AmbiguityPolicy) -> Option<u16> {
    rna_symbols(ambiguity_policy)
        .chars()
        .position(|candidate| candidate == symbol)
        .map(|x| x as u16)
}

fn protein_code(symbol: char, ambiguity_policy: AmbiguityPolicy) -> Option<u16> {
    protein_symbols(ambiguity_policy)
        .chars()
        .position(|candidate| candidate == symbol)
        .map(|x| x as u16)
}

fn dna_symbols(ambiguity_policy: AmbiguityPolicy) -> &'static str {
    match ambiguity_policy {
        AmbiguityPolicy::Reject => "ACGT",
        AmbiguityPolicy::ExactSymbol | AmbiguityPolicy::EquivalenceClass => "ACGTRYSWKMBDHVN",
    }
}

fn rna_symbols(ambiguity_policy: AmbiguityPolicy) -> &'static str {
    match ambiguity_policy {
        AmbiguityPolicy::Reject => "ACGU",
        AmbiguityPolicy::ExactSymbol | AmbiguityPolicy::EquivalenceClass => "ACGURYSWKMBDHVN",
    }
}

fn protein_symbols(ambiguity_policy: AmbiguityPolicy) -> &'static str {
    match ambiguity_policy {
        AmbiguityPolicy::Reject => "ACDEFGHIKLMNPQRSTVWY",
        AmbiguityPolicy::ExactSymbol | AmbiguityPolicy::EquivalenceClass => {
            "ACDEFGHIKLMNPQRSTVWYBJZXUO*"
        }
    }
}

fn bits_needed(symbol_count: usize) -> u8 {
    let count = symbol_count.max(1);
    (usize::BITS as u8 - (count - 1).leading_zeros() as u8).max(1)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CustomAlphabet {
    normalization: NormalizationPolicy,
    ambiguity_policy: AmbiguityPolicy,
    symbols: Vec<char>,
    ids: HashMap<char, SymbolId>,
}

impl CustomAlphabet {
    pub fn new(symbols: impl IntoIterator<Item = char>) -> Result<Self> {
        Self::with_policy(
            symbols,
            NormalizationPolicy::Strict,
            AmbiguityPolicy::ExactSymbol,
        )
    }

    pub fn with_policy(
        symbols: impl IntoIterator<Item = char>,
        normalization: NormalizationPolicy,
        ambiguity_policy: AmbiguityPolicy,
    ) -> Result<Self> {
        let mut ordered = Vec::new();
        let mut ids = HashMap::new();
        for symbol in symbols {
            let normalized = normalize_custom_char(symbol, normalization);
            if ids.contains_key(&normalized) {
                return Err(DagError::InvalidSymbol {
                    symbol: normalized.to_string(),
                });
            }
            let id = SymbolId::new(u16::try_from(ordered.len()).map_err(|_| {
                DagError::InvalidSymbol {
                    symbol: symbol.to_string(),
                }
            })?);
            ordered.push(normalized);
            ids.insert(normalized, id);
        }
        if ordered.is_empty() {
            return Err(DagError::InvalidSymbol {
                symbol: "empty custom alphabet".to_string(),
            });
        }
        Ok(Self {
            normalization,
            ambiguity_policy,
            symbols: ordered,
            ids,
        })
    }

    pub fn symbols(&self) -> &[char] {
        &self.symbols
    }
}

impl Alphabet for CustomAlphabet {
    fn kind(&self) -> AlphabetKind {
        AlphabetKind::Custom
    }

    fn normalization(&self) -> NormalizationPolicy {
        self.normalization
    }

    fn ambiguity_policy(&self) -> AmbiguityPolicy {
        self.ambiguity_policy
    }

    fn symbol_count(&self) -> usize {
        self.symbols.len()
    }

    fn encode(&self, symbol: char) -> Result<SymbolId> {
        let normalized = normalize_custom_char(symbol, self.normalization);
        self.ids
            .get(&normalized)
            .copied()
            .ok_or_else(|| DagError::InvalidSymbol {
                symbol: symbol.to_string(),
            })
    }

    fn decode(&self, symbol: SymbolId) -> Option<char> {
        self.symbols.get(symbol.raw() as usize).copied()
    }
}

fn normalize_custom_char(symbol: char, normalization: NormalizationPolicy) -> char {
    match normalization {
        NormalizationPolicy::CaseFold => symbol.to_ascii_uppercase(),
        _ => symbol,
    }
}
