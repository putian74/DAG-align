//! Chunked build and deterministic merge scheduling interfaces.

use crate::foundations::id::{ChunkId, RoundId};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChunkPlan {
    pub chunks: Vec<ChunkId>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MergeRoundPlan {
    pub round: RoundId,
    pub pairs: Vec<(ChunkId, ChunkId)>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MemoryBudget {
    pub max_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProgressEvent {
    Started(&'static str),
    Advanced { completed: usize, total: usize },
    Finished(&'static str),
}

pub trait ProgressSink {
    fn on_progress(&mut self, event: ProgressEvent);
}

#[derive(Clone, Debug, Default)]
pub struct BuildScheduler;
