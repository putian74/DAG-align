//! Global PHMM coordinate, packed-window, and projection contracts.

use crate::error::{PreAdPrepError, Result};
use crate::graph::TensorGraph;

/// Half-open PHMM state interval `[left, right)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateInterval {
    pub left: usize,
    pub right: usize,
}

impl StateInterval {
    pub fn new(left: usize, right: usize) -> Self {
        Self { left, right }
    }

    pub fn len(&self) -> usize {
        self.right.saturating_sub(self.left)
    }

    pub fn is_empty(&self) -> bool {
        self.left == self.right
    }

    pub fn validate(&self, global_state_count: usize) -> Result<()> {
        if self.left > global_state_count {
            return Err(PreAdPrepError::Validation(format!(
                "state interval left {} exceeds global state count {}",
                self.left, global_state_count
            )));
        }
        if self.right < self.left {
            return Err(PreAdPrepError::Validation(format!(
                "state interval right {} is before left {}",
                self.right, self.left
            )));
        }
        if self.right > global_state_count {
            return Err(PreAdPrepError::Validation(format!(
                "state interval right {} exceeds global state count {}",
                self.right, global_state_count
            )));
        }
        Ok(())
    }

    pub fn intersection(&self, other: &Self) -> Option<Self> {
        let left = self.left.max(other.left);
        let right = self.right.min(other.right);
        (left < right).then_some(Self { left, right })
    }
}

/// Packed ragged node windows matching AD-PHMM-align's `[left, right)` contract.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedStateWindows {
    pub intervals: Vec<StateInterval>,
    pub offsets: Vec<usize>,
    pub lengths: Vec<usize>,
}

impl PackedStateWindows {
    pub fn from_intervals(intervals: Vec<StateInterval>) -> Self {
        let mut next_offset = 0usize;
        let mut offsets = Vec::with_capacity(intervals.len());
        let mut lengths = Vec::with_capacity(intervals.len());
        for interval in &intervals {
            offsets.push(next_offset);
            let len = interval.len();
            lengths.push(len);
            next_offset += len;
        }
        Self {
            intervals,
            offsets,
            lengths,
        }
    }

    pub fn validate(&self, global_state_count: usize) -> Result<()> {
        if self.intervals.len() != self.offsets.len() || self.intervals.len() != self.lengths.len()
        {
            return Err(PreAdPrepError::Validation(
                "packed state-window arrays have inconsistent lengths".into(),
            ));
        }
        for (index, interval) in self.intervals.iter().enumerate() {
            interval.validate(global_state_count)?;
            if self.lengths[index] != interval.len() {
                return Err(PreAdPrepError::Validation(format!(
                    "packed state-window length mismatch at node {index}"
                )));
            }
        }
        Ok(())
    }
}

/// Configuration for turning global intervals into padded packed windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WindowBuildConfig {
    pub left_padding: usize,
    pub right_padding: usize,
}

/// Overlap between one graph edge's source and target state windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EdgeWindowOverlap {
    pub edge_id: usize,
    pub src_offset: usize,
    pub dst_offset: usize,
    pub len: usize,
}

/// Packed edge-window overlap metadata used by banded DP kernels.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct EdgeWindowOverlaps {
    pub overlaps: Vec<EdgeWindowOverlap>,
}

impl EdgeWindowOverlaps {
    pub fn validate(&self, edge_count: usize) -> Result<()> {
        for overlap in &self.overlaps {
            if overlap.edge_id >= edge_count {
                return Err(PreAdPrepError::Validation(format!(
                    "edge-window overlap references edge {} but edge_count is {}",
                    overlap.edge_id, edge_count
                )));
            }
        }
        Ok(())
    }

    pub fn validate_against_windows(
        &self,
        edge_src: &[usize],
        edge_dst: &[usize],
        windows: &PackedStateWindows,
    ) -> Result<()> {
        self.validate(edge_src.len())?;
        if edge_src.len() != edge_dst.len() {
            return Err(PreAdPrepError::Validation(
                "edge source and destination arrays have inconsistent lengths".into(),
            ));
        }
        for overlap in &self.overlaps {
            let src = edge_src[overlap.edge_id];
            let dst = edge_dst[overlap.edge_id];
            let Some(&src_len) = windows.lengths.get(src) else {
                return Err(PreAdPrepError::Validation(format!(
                    "edge-window overlap source node {src} has no state window"
                )));
            };
            let Some(&dst_len) = windows.lengths.get(dst) else {
                return Err(PreAdPrepError::Validation(format!(
                    "edge-window overlap destination node {dst} has no state window"
                )));
            };
            if overlap.src_offset + overlap.len > src_len {
                return Err(PreAdPrepError::Validation(format!(
                    "edge-window overlap exceeds source window for edge {}",
                    overlap.edge_id
                )));
            }
            if overlap.dst_offset + overlap.len > dst_len {
                return Err(PreAdPrepError::Validation(format!(
                    "edge-window overlap exceeds destination window for edge {}",
                    overlap.edge_id
                )));
            }
        }
        Ok(())
    }
}

/// Mapping between local subgraph state indices and global PHMM state IDs.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StateProjection {
    pub local_to_global_state: Vec<usize>,
    pub active_global_states: Vec<usize>,
}

impl StateProjection {
    pub fn validate(&self, global_state_count: usize) -> Result<()> {
        for state_id in self
            .local_to_global_state
            .iter()
            .chain(self.active_global_states.iter())
        {
            if *state_id >= global_state_count {
                return Err(PreAdPrepError::Validation(format!(
                    "projected state id {state_id} exceeds global state count {global_state_count}"
                )));
            }
        }
        Ok(())
    }
}

/// State-range propagation mode copied from current DAG-align behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinateMode {
    Build,
    Hmm,
}

/// Ordered reference path used to anchor global PHMM coordinates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferencePath {
    pub node_ids: Vec<Option<usize>>,
}

impl ReferencePath {
    pub fn global_state_count(&self) -> usize {
        self.node_ids.len()
    }
}

/// Result of propagating global PHMM coordinates onto graph nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GlobalCoordinateOutput {
    pub global_state_count: usize,
    pub node_intervals: Vec<StateInterval>,
    pub reference_path: ReferencePath,
}

impl GlobalCoordinateOutput {
    pub fn validate(&self) -> Result<()> {
        for interval in &self.node_intervals {
            interval.validate(self.global_state_count)?;
        }
        Ok(())
    }
}

/// Build global node intervals from a graph and ordered reference path.
pub fn build_global_coordinates(
    graph: &TensorGraph,
    reference_path: ReferencePath,
    mode: CoordinateMode,
) -> Result<GlobalCoordinateOutput> {
    let global_state_count = reference_path.global_state_count();
    let node_count = graph.node_count();
    let mut intervals = vec![StateInterval::new(0, global_state_count); node_count];
    for (index, node_id) in reference_path.node_ids.iter().enumerate() {
        if let Some(node_id) = node_id {
            if *node_id >= node_count {
                return Err(PreAdPrepError::Validation(format!(
                    "reference node {} exceeds node_count {}",
                    node_id, node_count
                )));
            }
            intervals[*node_id] = StateInterval::new(index, index);
        }
    }

    let (parents, children) = build_adjacency_lists(graph);
    for &node in &graph.topo_order {
        let mut left = intervals[node].left;
        for &parent in &parents[node] {
            left = left.max(intervals[parent].left);
        }
        intervals[node].left = left;
    }
    for &node in graph.topo_order.iter().rev() {
        let mut right = intervals[node].right;
        for &child in &children[node] {
            right = right.min(intervals[child].right);
        }
        intervals[node].right = right;
    }

    if mode == CoordinateMode::Hmm {
        let mut expanded = intervals.clone();
        for node in 0..node_count {
            let mut right = intervals[node].right;
            for &child in &children[node] {
                right = right.max(intervals[child].right);
            }
            expanded[node].right = right;
        }
        intervals = expanded;
    }

    let output = GlobalCoordinateOutput {
        global_state_count,
        node_intervals: intervals,
        reference_path,
    };
    output.validate()?;
    Ok(output)
}

fn build_adjacency_lists(graph: &TensorGraph) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let node_count = graph.node_count();
    let mut parents = vec![Vec::new(); node_count];
    let mut children = vec![Vec::new(); node_count];
    for (&src, &dst) in graph.edge_src.iter().zip(graph.edge_dst.iter()) {
        if src < node_count && dst < node_count {
            children[src].push(dst);
            parents[dst].push(src);
        }
    }
    (parents, children)
}

/// Build padded packed state windows from global node intervals.
pub fn build_packed_windows(
    coordinates: &GlobalCoordinateOutput,
    config: WindowBuildConfig,
) -> Result<PackedStateWindows> {
    let intervals = coordinates
        .node_intervals
        .iter()
        .map(|interval| {
            let left = interval.left.saturating_sub(config.left_padding);
            let right = (interval.right + config.right_padding).min(coordinates.global_state_count);
            StateInterval::new(left, right)
        })
        .collect();
    let windows = PackedStateWindows::from_intervals(intervals);
    windows.validate(coordinates.global_state_count)?;
    Ok(windows)
}

/// Build packed edge overlaps from graph edges and node windows.
pub fn build_edge_window_overlaps(
    graph: &TensorGraph,
    windows: &PackedStateWindows,
) -> Result<EdgeWindowOverlaps> {
    if windows.intervals.len() != graph.node_count() {
        return Err(PreAdPrepError::Validation(
            "packed windows must exist for every graph node".into(),
        ));
    }
    let mut overlaps = Vec::new();
    for (edge_id, (&src, &dst)) in graph.edge_src.iter().zip(graph.edge_dst.iter()).enumerate() {
        let src_interval = windows.intervals.get(src).ok_or_else(|| {
            PreAdPrepError::Validation(format!("missing source window for node {src}"))
        })?;
        let dst_interval = windows.intervals.get(dst).ok_or_else(|| {
            PreAdPrepError::Validation(format!("missing destination window for node {dst}"))
        })?;
        if let Some(intersection) = src_interval.intersection(dst_interval) {
            overlaps.push(EdgeWindowOverlap {
                edge_id,
                src_offset: intersection.left - src_interval.left,
                dst_offset: intersection.left - dst_interval.left,
                len: intersection.len(),
            });
        }
    }
    let overlaps = EdgeWindowOverlaps { overlaps };
    overlaps.validate_against_windows(&graph.edge_src, &graph.edge_dst, windows)?;
    Ok(overlaps)
}
