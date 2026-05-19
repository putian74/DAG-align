//! Typed IDs and scale-aware numeric wrappers.

use crate::foundations::error::{DagError, Result};

macro_rules! typed_numeric {
    ($name:ident, $raw:ty) => {
        #[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name($raw);

        impl $name {
            pub const fn new(raw: $raw) -> Self {
                Self(raw)
            }

            pub const fn raw(self) -> $raw {
                self.0
            }

            pub fn to_usize(self) -> usize {
                self.0 as usize
            }
        }

        impl From<$raw> for $name {
            fn from(value: $raw) -> Self {
                Self(value)
            }
        }

        impl TryFrom<usize> for $name {
            type Error = DagError;

            fn try_from(value: usize) -> Result<Self> {
                let raw = <$raw>::try_from(value).map_err(|_| DagError::IdOverflow {
                    type_name: stringify!($name),
                    value,
                })?;
                Ok(Self(raw))
            }
        }
    };
}

typed_numeric!(NodeId, u32);
typed_numeric!(SequenceId, u32);
typed_numeric!(GraphId, u32);
typed_numeric!(ChunkId, u32);
typed_numeric!(RoundId, u32);
typed_numeric!(TopologicalCoordinate, u32);
typed_numeric!(Weight, u64);
typed_numeric!(ProvenancePosition, u64);
typed_numeric!(GlobalStateId, u32);
