//! Fragment keys and sliding-window iteration.

use crate::foundations::bit_encoding::{BitWidth, PackedWindow};
use crate::foundations::error::{DagError, Result};
use crate::graph_model::graph::StoredFragmentKey;
use crate::sequence_model::alphabet::SymbolId;
use crate::sequence_model::sequence::EncodedSequence;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum FragmentKey {
    PackedInline {
        bits_per_symbol: u8,
        len: u16,
        value: u128,
    },
    PackedWords {
        bits_per_symbol: u8,
        len: u32,
        words: Vec<u64>,
    },
    Symbols(Vec<SymbolId>),
}

impl FragmentKey {
    pub fn symbols(symbols: Vec<SymbolId>) -> Self {
        Self::Symbols(symbols)
    }

    pub const fn packed_inline(bits_per_symbol: u8, len: u16, value: u128) -> Self {
        Self::PackedInline {
            bits_per_symbol,
            len,
            value,
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::PackedInline { len, .. } => usize::from(*len),
            Self::PackedWords { len, .. } => *len as usize,
            Self::Symbols(symbols) => symbols.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bits_per_symbol(&self) -> Option<u8> {
        match self {
            Self::PackedInline {
                bits_per_symbol, ..
            }
            | Self::PackedWords {
                bits_per_symbol, ..
            } => Some(*bits_per_symbol),
            Self::Symbols(_) => None,
        }
    }
}

pub trait FragmentEncoder {
    fn encode_window(&self, window: &[SymbolId]) -> Result<FragmentKey>;

    fn encode_occurrences(
        &self,
        sequence: &EncodedSequence,
        fragment_len: usize,
    ) -> Result<Vec<FragmentOccurrence>>
    where
        Self: Sized,
    {
        FragmentOccurrences::new(sequence, fragment_len, self)?.collect()
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DefaultFragmentEncoder {
    bits_per_symbol: Option<BitWidth>,
}

impl DefaultFragmentEncoder {
    pub const fn general() -> Self {
        Self {
            bits_per_symbol: None,
        }
    }

    pub const fn packed(bits_per_symbol: BitWidth) -> Self {
        Self {
            bits_per_symbol: Some(bits_per_symbol),
        }
    }
}

impl FragmentEncoder for DefaultFragmentEncoder {
    fn encode_window(&self, window: &[SymbolId]) -> Result<FragmentKey> {
        if let Some(bits_per_symbol) = self.bits_per_symbol {
            if window.len() > PackedWindow::inline_capacity(bits_per_symbol) {
                return Ok(FragmentKey::Symbols(window.to_vec()));
            }
            let packed = PackedWindow::from_symbols(
                window.iter().copied().map(SymbolId::raw),
                bits_per_symbol,
            )?;
            Ok(FragmentKey::PackedInline {
                bits_per_symbol: packed.bits_per_symbol().bits(),
                len: packed.len(),
                value: packed.value(),
            })
        } else {
            Ok(FragmentKey::Symbols(window.to_vec()))
        }
    }

    fn encode_occurrences(
        &self,
        sequence: &EncodedSequence,
        fragment_len: usize,
    ) -> Result<Vec<FragmentOccurrence>> {
        let Some(bits_per_symbol) = self.bits_per_symbol else {
            return FragmentOccurrences::new(sequence, fragment_len, self)?.collect();
        };
        if fragment_len == 0 || fragment_len > sequence.len() {
            return Err(DagError::InvalidFragmentLength {
                fragment_len,
                sequence_len: sequence.len(),
            });
        }
        if fragment_len > PackedWindow::inline_capacity(bits_per_symbol) {
            return FragmentOccurrences::new(sequence, fragment_len, self)?.collect();
        }

        let window_count = sequence.len() - fragment_len + 1;
        let bits = bits_per_symbol.bits();
        let symbol_mask = bits_per_symbol.mask_u128();
        let retained_bits = usize::from(bits) * (fragment_len - 1);
        let retained_mask = if retained_bits == 128 {
            u128::MAX
        } else if retained_bits == 0 {
            0
        } else {
            (1_u128 << retained_bits) - 1
        };
        let symbols = sequence.symbols();
        let mut value = 0_u128;
        for symbol in &symbols[..fragment_len] {
            let symbol = u128::from(symbol.raw());
            if symbol > symbol_mask {
                return Err(DagError::ValueDoesNotFit {
                    value: symbol,
                    bits,
                });
            }
            value = (value << bits) | symbol;
        }

        let mut occurrences = Vec::with_capacity(window_count);
        for position in 0..window_count {
            if position > 0 {
                let symbol = u128::from(symbols[position + fragment_len - 1].raw());
                if symbol > symbol_mask {
                    return Err(DagError::ValueDoesNotFit {
                        value: symbol,
                        bits,
                    });
                }
                value = ((value & retained_mask) << bits) | symbol;
            }
            occurrences.push(FragmentOccurrence {
                position,
                kind: path_position_kind(position, window_count),
                key: StoredFragmentKey::from(FragmentKey::PackedInline {
                    bits_per_symbol: bits,
                    len: fragment_len as u16,
                    value,
                }),
            });
        }
        Ok(occurrences)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct FragmentOccurrence {
    pub position: usize,
    pub kind: PathPositionKind,
    pub key: StoredFragmentKey,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PathPositionKind {
    Start,
    Internal,
    End,
    Singleton,
}

pub struct FragmentWindows<'a, E> {
    sequence: &'a EncodedSequence,
    fragment_len: usize,
    position: usize,
    encoder: &'a E,
}

impl<'a, E: FragmentEncoder> FragmentWindows<'a, E> {
    pub fn new(sequence: &'a EncodedSequence, fragment_len: usize, encoder: &'a E) -> Result<Self> {
        if fragment_len == 0 || fragment_len > sequence.len() {
            return Err(DagError::InvalidFragmentLength {
                fragment_len,
                sequence_len: sequence.len(),
            });
        }
        Ok(Self {
            sequence,
            fragment_len,
            position: 0,
            encoder,
        })
    }
}

impl<E: FragmentEncoder> Iterator for FragmentWindows<'_, E> {
    type Item = Result<(usize, FragmentKey)>;

    fn next(&mut self) -> Option<Self::Item> {
        let end = self.position + self.fragment_len;
        if end > self.sequence.len() {
            return None;
        }
        let start = self.position;
        self.position += 1;
        Some(
            self.encoder
                .encode_window(&self.sequence.symbols()[start..end])
                .map(|key| (start, key)),
        )
    }
}

pub struct FragmentOccurrences<'a, E> {
    windows: FragmentWindows<'a, E>,
    window_count: usize,
}

impl<'a, E: FragmentEncoder> FragmentOccurrences<'a, E> {
    pub fn new(sequence: &'a EncodedSequence, fragment_len: usize, encoder: &'a E) -> Result<Self> {
        let window_count = sequence
            .len()
            .checked_sub(fragment_len)
            .map_or(0, |x| x + 1);
        Ok(Self {
            windows: FragmentWindows::new(sequence, fragment_len, encoder)?,
            window_count,
        })
    }
}

impl<E: FragmentEncoder> Iterator for FragmentOccurrences<'_, E> {
    type Item = Result<FragmentOccurrence>;

    fn next(&mut self) -> Option<Self::Item> {
        self.windows.next().map(|result| {
            result.map(|(position, key)| FragmentOccurrence {
                position,
                kind: path_position_kind(position, self.window_count),
                key: StoredFragmentKey::from(key),
            })
        })
    }
}

fn path_position_kind(position: usize, window_count: usize) -> PathPositionKind {
    if window_count == 1 {
        PathPositionKind::Singleton
    } else if position == 0 {
        PathPositionKind::Start
    } else if position + 1 == window_count {
        PathPositionKind::End
    } else {
        PathPositionKind::Internal
    }
}
