//! Deterministic and similarity-first ordering policies.

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum OrderingPolicy {
    InputOrder,
    ChunkLocalSimilarity,
    SketchBucketedSimilarity,
    UserProvided,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SketchConfig {
    pub k: usize,
    pub sketch_size: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SequenceSketch {
    pub hashes: Vec<u64>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct GraphSketch {
    pub hashes: Vec<u64>,
}

pub trait SimilaritySketch {
    fn hashes(&self) -> &[u64];
}

impl SimilaritySketch for SequenceSketch {
    fn hashes(&self) -> &[u64] {
        &self.hashes
    }
}

impl SimilaritySketch for GraphSketch {
    fn hashes(&self) -> &[u64] {
        &self.hashes
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct SimilarityBucket {
    pub key: u64,
    pub item_indices: Vec<usize>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct PairingPlan {
    pub pairs: Vec<(usize, usize)>,
}
