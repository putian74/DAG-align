//! Graph validation traits and reports.

use crate::foundations::error::DagError;
use crate::graph_model::graph::FtoDag;
use crate::graph_model::topology::DagTopology;
use std::collections::HashSet;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphValidationError {
    InvalidNodeId { expected: usize, found: usize },
    InvalidEdgeEndpoint { source: usize, target: usize },
    CycleDetected,
    MismatchedSourceWeight { node: usize },
    DuplicateSequenceSource { node: usize, sequence: usize },
    MissingFragmentIndexEntry { node: usize },
    NodeKindInconsistency { node: usize },
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ValidationReport {
    pub errors: Vec<GraphValidationError>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }
}

pub trait ValidateGraph {
    fn validate(&self) -> ValidationReport;
}

impl ValidateGraph for FtoDag {
    fn validate(&self) -> ValidationReport {
        let mut report = ValidationReport::default();
        for (expected, node) in self.nodes().iter().enumerate() {
            if node.id.to_usize() != expected {
                report.errors.push(GraphValidationError::InvalidNodeId {
                    expected,
                    found: node.id.to_usize(),
                });
            }
            if node.flags != node.kind.flags() {
                report
                    .errors
                    .push(GraphValidationError::NodeKindInconsistency { node: expected });
            }
            if !self
                .fragment_index()
                .contains(&node.fragment, node.kind, node.id)
            {
                report
                    .errors
                    .push(GraphValidationError::MissingFragmentIndexEntry { node: expected });
            }
            match self.source_record_count(node.id) {
                Ok(count) => {
                    if count as u64 != node.weight.raw() {
                        report
                            .errors
                            .push(GraphValidationError::MismatchedSourceWeight { node: expected });
                    }
                    if self.retains_source_records() {
                        match self.source_records(node.id) {
                            Ok(records) => {
                                let mut sequences = HashSet::new();
                                for record in records {
                                    if !sequences.insert(record.sequence_id) {
                                        report.errors.push(
                                            GraphValidationError::DuplicateSequenceSource {
                                                node: expected,
                                                sequence: record.sequence_id.to_usize(),
                                            },
                                        );
                                    }
                                }
                            }
                            Err(_) => {
                                report
                                    .errors
                                    .push(GraphValidationError::MismatchedSourceWeight {
                                        node: expected,
                                    })
                            }
                        }
                    }
                }
                Err(_) => report
                    .errors
                    .push(GraphValidationError::MismatchedSourceWeight { node: expected }),
            }
        }
        for edge in self.edges() {
            if edge.key.source.to_usize() >= self.node_count()
                || edge.key.target.to_usize() >= self.node_count()
            {
                report
                    .errors
                    .push(GraphValidationError::InvalidEdgeEndpoint {
                        source: edge.key.source.to_usize(),
                        target: edge.key.target.to_usize(),
                    });
            }
        }
        let mut has_parent = vec![false; self.node_count()];
        let mut has_child = vec![false; self.node_count()];
        for edge in self.edges() {
            if edge.key.source.to_usize() < self.node_count()
                && edge.key.target.to_usize() < self.node_count()
            {
                has_child[edge.key.source.to_usize()] = true;
                has_parent[edge.key.target.to_usize()] = true;
            }
        }
        for node in self.endpoints().structural_roots() {
            if has_parent[node.to_usize()] {
                report
                    .errors
                    .push(GraphValidationError::NodeKindInconsistency {
                        node: node.to_usize(),
                    });
            }
        }
        for node in self.endpoints().structural_sinks() {
            if has_child[node.to_usize()] {
                report
                    .errors
                    .push(GraphValidationError::NodeKindInconsistency {
                        node: node.to_usize(),
                    });
            }
        }
        if matches!(
            DagTopology::try_from_graph(self),
            Err(DagError::CycleDetected)
        ) {
            report.errors.push(GraphValidationError::CycleDetected);
        }
        report
    }
}
