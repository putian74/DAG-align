//! Native graph storage and export profile interfaces.

use crate::foundations::error::Result;
use crate::graph_model::graph::FtoDag;
use std::path::Path;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct GraphFormatVersion {
    pub major: u16,
    pub minor: u16,
}

impl GraphFormatVersion {
    pub const CURRENT: Self = Self { major: 0, minor: 1 };
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct StorageConfig {
    pub version: GraphFormatVersion,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            version: GraphFormatVersion::CURRENT,
        }
    }
}

pub trait GraphStorage {
    fn save_graph(&self, graph: &FtoDag, path: &Path, config: StorageConfig) -> Result<()>;
    fn load_graph(&self, path: &Path) -> Result<FtoDag>;
}
