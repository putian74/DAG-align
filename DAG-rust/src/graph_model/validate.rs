//! Graph validation traits and reports.

use crate::foundations::error::DagError;
use crate::graph_model::graph::FtoDag;
use crate::graph_model::topology::DagTopology;
use std::collections::HashSet;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GraphValidationError {
    InvalidNodeId { expected: usize, found: usize },
    InvalidEdgeEndpoint { parent: usize, child: usize },
    CycleDetected,
    MismatchedProvenanceWeight { node: usize },
    DuplicateSequenceProvenance { node: usize, sequence: usize },
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
            match self.provenance_record_count(node.id) {
                Ok(count) => {
                    if count as u64 != node.weight.raw() {
                        report
                            .errors
                            .push(GraphValidationError::MismatchedProvenanceWeight {
                                node: expected,
                            });
                    }
                    if self.retains_provenance_records() {
                        match self.provenance_records(node.id) {
                            Ok(records) => {
                                let mut sequences = HashSet::new();
                                for record in records {
                                    if !sequences.insert(record.sequence_id) {
                                        report.errors.push(
                                            GraphValidationError::DuplicateSequenceProvenance {
                                                node: expected,
                                                sequence: record.sequence_id.to_usize(),
                                            },
                                        );
                                    }
                                }
                            }
                            Err(_) => report.errors.push(
                                GraphValidationError::MismatchedProvenanceWeight { node: expected },
                            ),
                        }
                    }
                }
                Err(_) => report
                    .errors
                    .push(GraphValidationError::MismatchedProvenanceWeight { node: expected }),
            }
        }
        for edge in self.edges() {
            if edge.key.parent.to_usize() >= self.node_count()
                || edge.key.child.to_usize() >= self.node_count()
            {
                report
                    .errors
                    .push(GraphValidationError::InvalidEdgeEndpoint {
                        parent: edge.key.parent.to_usize(),
                        child: edge.key.child.to_usize(),
                    });
            }
        }
        let mut has_parent = vec![false; self.node_count()];
        let mut has_child = vec![false; self.node_count()];
        for edge in self.edges() {
            if edge.key.parent.to_usize() < self.node_count()
                && edge.key.child.to_usize() < self.node_count()
            {
                has_child[edge.key.parent.to_usize()] = true;
                has_parent[edge.key.child.to_usize()] = true;
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
