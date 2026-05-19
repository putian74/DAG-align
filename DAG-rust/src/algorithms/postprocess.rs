//! Resolution adjustment, atomization, and secondary merge interfaces.

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ResolutionConfig {
    pub target_fragment_len: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SecondaryMergeConfig {
    pub use_forward_coordinates: bool,
    pub use_reverse_coordinates: bool,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct PostprocessStats {
    pub removed_nodes: usize,
    pub merged_nodes: usize,
}
