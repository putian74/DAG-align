//! Shared error types.

use std::fmt::{Display, Formatter};

pub type Result<T> = std::result::Result<T, DagError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DagError {
    InvalidBitWidth {
        bits: u8,
    },
    ValueDoesNotFit {
        value: u128,
        bits: u8,
    },
    InvalidThreshold {
        basis_points: u32,
    },
    IdOverflow {
        type_name: &'static str,
        value: usize,
    },
    InvalidSymbol {
        symbol: String,
    },
    InvalidFragmentLength {
        fragment_len: usize,
        sequence_len: usize,
    },
    InvalidRange {
        start: usize,
        end: usize,
        len: usize,
    },
    MissingNode {
        node: usize,
    },
    InvalidEdge {
        parent: usize,
        child: usize,
    },
    DuplicateSequenceProvenance {
        node: usize,
        sequence: u32,
    },
    CycleDetected,
    StorageVersionMismatch {
        expected: u32,
        found: u32,
    },
    Io(String),
    InvalidStorage(String),
    UnsupportedOperation(&'static str),
}

impl Display for DagError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidBitWidth { bits } => write!(f, "invalid bit width: {bits}"),
            Self::ValueDoesNotFit { value, bits } => {
                write!(f, "value {value} does not fit in {bits} bits")
            }
            Self::InvalidThreshold { basis_points } => {
                write!(
                    f,
                    "invalid similarity threshold: {basis_points} basis points"
                )
            }
            Self::IdOverflow { type_name, value } => {
                write!(f, "{type_name} cannot represent value {value}")
            }
            Self::InvalidSymbol { symbol } => write!(f, "invalid symbol: {symbol}"),
            Self::InvalidFragmentLength {
                fragment_len,
                sequence_len,
            } => write!(
                f,
                "fragment length {fragment_len} is invalid for sequence length {sequence_len}"
            ),
            Self::InvalidRange { start, end, len } => {
                write!(f, "invalid range {start}..{end} for length {len}")
            }
            Self::MissingNode { node } => write!(f, "missing node {node}"),
            Self::InvalidEdge { parent, child } => {
                write!(f, "invalid edge ({parent}, {child})")
            }
            Self::DuplicateSequenceProvenance { node, sequence } => {
                write!(f, "node {node} already contains sequence {sequence}")
            }
            Self::CycleDetected => write!(f, "cycle detected"),
            Self::StorageVersionMismatch { expected, found } => {
                write!(
                    f,
                    "storage version mismatch: expected {expected}, found {found}"
                )
            }
            Self::Io(message) => write!(f, "i/o error: {message}"),
            Self::InvalidStorage(message) => write!(f, "invalid storage: {message}"),
            Self::UnsupportedOperation(message) => write!(f, "unsupported operation: {message}"),
        }
    }
}

impl std::error::Error for DagError {}
