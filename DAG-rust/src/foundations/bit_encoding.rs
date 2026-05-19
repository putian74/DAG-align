//! Low-level bit-packing helpers shared by storage, graph, source, and fragment code.

use crate::foundations::error::{DagError, Result};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BitWidth(u8);

impl BitWidth {
    pub fn new(bits: u8) -> Result<Self> {
        if (1..=128).contains(&bits) {
            Ok(Self(bits))
        } else {
            Err(DagError::InvalidBitWidth { bits })
        }
    }

    pub const fn bits(self) -> u8 {
        self.0
    }

    pub fn mask_u128(self) -> u128 {
        if self.0 == 128 {
            u128::MAX
        } else {
            (1_u128 << self.0) - 1
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct PackedPair {
    raw: u128,
    low_bits: BitWidth,
}

impl PackedPair {
    pub fn pack(high: u128, low: u128, low_bits: BitWidth) -> Result<Self> {
        if low > low_bits.mask_u128() {
            return Err(DagError::ValueDoesNotFit {
                value: low,
                bits: low_bits.bits(),
            });
        }
        Ok(Self {
            raw: (high << low_bits.bits()) | low,
            low_bits,
        })
    }

    pub const fn raw(self) -> u128 {
        self.raw
    }

    pub fn high(self) -> u128 {
        self.raw >> self.low_bits.bits()
    }

    pub fn low(self) -> u128 {
        self.raw & self.low_bits.mask_u128()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct PackedWindow {
    value: u128,
    bits_per_symbol: BitWidth,
    len: u16,
}

impl PackedWindow {
    pub fn inline_capacity(bits_per_symbol: BitWidth) -> usize {
        128 / usize::from(bits_per_symbol.bits())
    }

    pub fn from_symbols<I>(symbols: I, bits_per_symbol: BitWidth) -> Result<Self>
    where
        I: IntoIterator<Item = u16>,
    {
        let mut value = 0_u128;
        let mut len = 0_u16;
        let mask = bits_per_symbol.mask_u128();
        for symbol in symbols {
            if usize::from(len) >= Self::inline_capacity(bits_per_symbol) {
                return Err(DagError::ValueDoesNotFit {
                    value: usize::from(len) as u128 + 1,
                    bits: 128,
                });
            }
            let symbol = u128::from(symbol);
            if symbol > mask {
                return Err(DagError::ValueDoesNotFit {
                    value: symbol,
                    bits: bits_per_symbol.bits(),
                });
            }
            value = (value << bits_per_symbol.bits()) | symbol;
            len += 1;
        }
        Ok(Self {
            value,
            bits_per_symbol,
            len,
        })
    }

    pub const fn value(&self) -> u128 {
        self.value
    }

    pub const fn bits_per_symbol(&self) -> BitWidth {
        self.bits_per_symbol
    }

    pub const fn len(&self) -> u16 {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct NodeFlags {
    bits: u16,
}

impl NodeFlags {
    pub const START: u16 = 0b0000_0001;
    pub const END: u16 = 0b0000_0010;
    pub const REMOVED: u16 = 0b0000_0100;
    pub const DIRTY_COORDINATES: u16 = 0b0000_1000;
    pub const REFERENCE: u16 = 0b0001_0000;
    pub const HAS_FRAGMENT_OFFSET: u16 = 0b0010_0000;

    pub const fn empty() -> Self {
        Self { bits: 0 }
    }

    pub const fn from_bits(bits: u16) -> Self {
        Self { bits }
    }

    pub const fn bits(self) -> u16 {
        self.bits
    }

    pub const fn contains(self, flag: u16) -> bool {
        (self.bits & flag) == flag
    }

    pub fn insert(&mut self, flag: u16) {
        self.bits |= flag;
    }
}
