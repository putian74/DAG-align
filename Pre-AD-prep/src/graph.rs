//! Typed DAG graph contracts consumed by preprocessing and export.

use crate::coordinates::{EdgeWindowOverlaps, PackedStateWindows};
use crate::error::{PreAdPrepError, Result};
use crate::validate::{Validate, ValidationReport};

pub type NodeId = usize;
pub type EdgeId = usize;
pub type SymbolId = u16;

/// Bit flags assigned to graph nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NodeFlags(pub u32);

impl NodeFlags {
    pub const START: u32 = 1 << 0;
    pub const END: u32 = 1 << 1;
    pub const REFERENCE: u32 = 1 << 2;

    pub fn contains(self, flag: u32) -> bool {
        self.0 & flag != 0
    }
}

/// CSR/CSC adjacency representation with edge IDs preserved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdjacencyCsr {
    pub indptr: Vec<usize>,
    pub indices: Vec<NodeId>,
    pub edge_ids: Vec<EdgeId>,
}

impl AdjacencyCsr {
    pub fn validate(&self, node_count: usize, edge_count: usize) -> Result<()> {
        if self.indptr.len() != node_count + 1 {
            return Err(PreAdPrepError::Validation(
                "adjacency indptr length must equal node_count + 1".into(),
            ));
        }
        if self.indices.len() != self.edge_ids.len() {
            return Err(PreAdPrepError::Validation(
                "adjacency indices and edge_ids lengths differ".into(),
            ));
        }
        if self.indices.len() != edge_count {
            return Err(PreAdPrepError::Validation(
                "adjacency edge count does not match graph edge count".into(),
            ));
        }
        if self.indptr.windows(2).any(|window| window[1] < window[0]) {
            return Err(PreAdPrepError::Validation(
                "adjacency indptr must be monotonic".into(),
            ));
        }
        if self.indptr.last().copied() != Some(edge_count) {
            return Err(PreAdPrepError::Validation(
                "adjacency indptr last value must equal edge count".into(),
            ));
        }
        if self.indices.iter().any(|&node| node >= node_count) {
            return Err(PreAdPrepError::Validation(
                "adjacency index exceeds node count".into(),
            ));
        }
        if self.edge_ids.iter().any(|&edge_id| edge_id >= edge_count) {
            return Err(PreAdPrepError::Validation(
                "adjacency edge id exceeds edge count".into(),
            ));
        }
        Ok(())
    }
}

/// Optional topological level batches for dependency-safe parallel execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologicalLevels {
    pub level_ptr: Vec<usize>,
    pub level_nodes: Vec<NodeId>,
}

/// Canonical typed DAG representation produced by all input adapters.
#[derive(Debug, Clone)]
pub struct TensorGraph {
    pub node_symbol: Vec<SymbolId>,
    pub node_weight: Vec<f32>,
    pub node_flags: Vec<NodeFlags>,
    pub edge_src: Vec<NodeId>,
    pub edge_dst: Vec<NodeId>,
    pub edge_weight: Vec<f32>,
    pub topo_order: Vec<NodeId>,
    pub csr: Option<AdjacencyCsr>,
    pub csc: Option<AdjacencyCsr>,
    pub topo_levels: Option<TopologicalLevels>,
    pub state_windows: Option<PackedStateWindows>,
    pub edge_overlaps: Option<EdgeWindowOverlaps>,
}

impl TensorGraph {
    pub fn new(
        node_symbol: Vec<SymbolId>,
        node_weight: Vec<f32>,
        edge_src: Vec<NodeId>,
        edge_dst: Vec<NodeId>,
        edge_weight: Vec<f32>,
        topo_order: Vec<NodeId>,
    ) -> Self {
        let node_flags = vec![NodeFlags::default(); node_symbol.len()];
        Self {
            node_symbol,
            node_weight,
            node_flags,
            edge_src,
            edge_dst,
            edge_weight,
            topo_order,
            csr: None,
            csc: None,
            topo_levels: None,
            state_windows: None,
            edge_overlaps: None,
        }
    }

    pub fn node_count(&self) -> usize {
        self.node_symbol.len()
    }

    pub fn edge_count(&self) -> usize {
        self.edge_src.len()
    }

    pub fn validate_with_global_states(&self, global_state_count: Option<usize>) -> Result<()> {
        self.validate()?.into_result()?;
        if let Some(global_state_count) = global_state_count {
            if let Some(windows) = &self.state_windows {
                windows.validate(global_state_count)?;
            }
        }
        if let Some(overlaps) = &self.edge_overlaps {
            overlaps.validate(self.edge_count())?;
            if let Some(windows) = &self.state_windows {
                overlaps.validate_against_windows(&self.edge_src, &self.edge_dst, windows)?;
            }
        }
        Ok(())
    }
}

impl Validate for TensorGraph {
    fn validate(&self) -> Result<ValidationReport> {
        let mut report = ValidationReport::new();
        let node_count = self.node_count();
        let edge_count = self.edge_count();

        if self.node_weight.len() != node_count {
            report.error(
                "node_weight_shape",
                "node_weight length must match node count",
            );
        }
        if self.node_flags.len() != node_count {
            report.error(
                "node_flags_shape",
                "node_flags length must match node count",
            );
        }
        if self.edge_dst.len() != edge_count || self.edge_weight.len() != edge_count {
            report.error("edge_shape", "edge arrays must have identical lengths");
        }
        if self.topo_order.len() != node_count {
            report.error(
                "topo_order_shape",
                "topological order must include every node",
            );
        }

        let mut seen = vec![false; node_count];
        for &node in &self.topo_order {
            if node >= node_count {
                report.error("topo_order_bounds", "topological order has invalid node id");
            } else if seen[node] {
                report.error(
                    "topo_order_duplicate",
                    "topological order contains duplicate node",
                );
            } else {
                seen[node] = true;
            }
        }
        if seen.iter().any(|seen| !seen) {
            report.error(
                "topo_order_missing",
                "topological order is missing at least one node",
            );
        }

        let mut topo_rank = vec![0usize; node_count];
        for (rank, &node) in self.topo_order.iter().enumerate() {
            if node < node_count {
                topo_rank[node] = rank;
            }
        }
        for edge_id in 0..edge_count {
            let src = self.edge_src[edge_id];
            let dst = self.edge_dst[edge_id];
            if src >= node_count || dst >= node_count {
                report.error("edge_endpoint_bounds", "edge endpoint exceeds node count");
                continue;
            }
            if topo_rank[src] >= topo_rank[dst] {
                report.error("topo_order_edge", "topological order violates an edge");
            }
        }

        if let Some(csr) = &self.csr {
            if let Err(error) = csr.validate(node_count, edge_count) {
                report.error("csr_invalid", error.to_string());
            }
        }
        if let Some(csc) = &self.csc {
            if let Err(error) = csc.validate(node_count, edge_count) {
                report.error("csc_invalid", error.to_string());
            }
        }

        Ok(report)
    }
}
