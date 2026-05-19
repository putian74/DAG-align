//! Topology views, traversal iterators, and coordinate algorithms.

use crate::foundations::error::{DagError, Result};
use crate::foundations::id::{NodeId, TopologicalCoordinate};
use crate::graph_model::graph::FtoDag;
use std::collections::VecDeque;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum TraversalDirection {
    Forward,
    Reverse,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DagTopology {
    node_count: usize,
    parents: Vec<Vec<NodeId>>,
    children: Vec<Vec<NodeId>>,
    topological_order: Vec<NodeId>,
    forward_coordinates: Vec<TopologicalCoordinate>,
    reverse_coordinates: Vec<TopologicalCoordinate>,
}

impl DagTopology {
    pub fn from_graph(graph: &FtoDag) -> Result<Self> {
        Self::try_from_graph(graph)
    }

    pub fn try_from_graph(graph: &FtoDag) -> Result<Self> {
        let node_count = graph.node_count();
        let mut parents = vec![Vec::new(); node_count];
        let mut children = vec![Vec::new(); node_count];
        for edge in graph.edges() {
            let parent = edge.key.parent.to_usize();
            let child = edge.key.child.to_usize();
            if parent >= node_count || child >= node_count {
                return Err(DagError::InvalidEdge { parent, child });
            }
            children[parent].push(edge.key.child);
            parents[child].push(edge.key.parent);
        }

        let topological_order = topological_sort(&parents, &children)?;
        let forward_coordinates = forward_coordinates(&topological_order, &parents);
        let reverse_coordinates = reverse_coordinates(&topological_order, &children);

        Ok(Self {
            node_count,
            parents,
            children,
            topological_order,
            forward_coordinates,
            reverse_coordinates,
        })
    }

    pub fn empty() -> Self {
        Self {
            node_count: 0,
            parents: Vec::new(),
            children: Vec::new(),
            topological_order: Vec::new(),
            forward_coordinates: Vec::new(),
            reverse_coordinates: Vec::new(),
        }
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn parents(&self, node: NodeId) -> Result<Parents<'_>> {
        let parents = self
            .parents
            .get(node.to_usize())
            .ok_or(DagError::MissingNode {
                node: node.to_usize(),
            })?;
        Ok(Parents(parents.iter()))
    }

    pub fn children(&self, node: NodeId) -> Result<Children<'_>> {
        let children = self
            .children
            .get(node.to_usize())
            .ok_or(DagError::MissingNode {
                node: node.to_usize(),
            })?;
        Ok(Children(children.iter()))
    }

    pub fn topological_order(&self) -> &[NodeId] {
        &self.topological_order
    }

    pub fn reverse_topological_order(&self) -> ReverseTopologicalOrder<'_> {
        ReverseTopologicalOrder::new(&self.topological_order)
    }

    pub fn forward_coordinate(&self, node: NodeId) -> Result<TopologicalCoordinate> {
        self.forward_coordinates
            .get(node.to_usize())
            .copied()
            .ok_or(DagError::MissingNode {
                node: node.to_usize(),
            })
    }

    pub fn reverse_coordinate(&self, node: NodeId) -> Result<TopologicalCoordinate> {
        self.reverse_coordinates
            .get(node.to_usize())
            .copied()
            .ok_or(DagError::MissingNode {
                node: node.to_usize(),
            })
    }
}

fn topological_sort(parents: &[Vec<NodeId>], children: &[Vec<NodeId>]) -> Result<Vec<NodeId>> {
    let mut indegree = parents.iter().map(Vec::len).collect::<Vec<_>>();
    let mut queue = indegree
        .iter()
        .enumerate()
        .filter(|(_, degree)| **degree == 0)
        .map(|(node, _)| NodeId::try_from(node).expect("node count exceeds NodeId capacity"))
        .collect::<VecDeque<_>>();
    let mut order = Vec::with_capacity(parents.len());
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for child in &children[node.to_usize()] {
            let degree = &mut indegree[child.to_usize()];
            *degree -= 1;
            if *degree == 0 {
                queue.push_back(*child);
            }
        }
    }
    if order.len() == parents.len() {
        Ok(order)
    } else {
        Err(DagError::CycleDetected)
    }
}

fn forward_coordinates(order: &[NodeId], parents: &[Vec<NodeId>]) -> Vec<TopologicalCoordinate> {
    let mut coordinates = vec![TopologicalCoordinate::new(0); parents.len()];
    for node in order {
        let coordinate = parents[node.to_usize()]
            .iter()
            .map(|parent| coordinates[parent.to_usize()].raw())
            .max()
            .unwrap_or(0)
            + 1;
        coordinates[node.to_usize()] = TopologicalCoordinate::new(coordinate);
    }
    coordinates
}

fn reverse_coordinates(order: &[NodeId], children: &[Vec<NodeId>]) -> Vec<TopologicalCoordinate> {
    let mut coordinates = vec![TopologicalCoordinate::new(0); children.len()];
    for node in order.iter().rev() {
        let coordinate = children[node.to_usize()]
            .iter()
            .map(|child| coordinates[child.to_usize()].raw())
            .max()
            .unwrap_or(0)
            + 1;
        coordinates[node.to_usize()] = TopologicalCoordinate::new(coordinate);
    }
    coordinates
}

pub struct TopologicalOrder<'a> {
    nodes: &'a [NodeId],
    cursor: usize,
}

impl<'a> TopologicalOrder<'a> {
    pub fn new(nodes: &'a [NodeId]) -> Self {
        Self { nodes, cursor: 0 }
    }
}

impl Iterator for TopologicalOrder<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        let node = self.nodes.get(self.cursor).copied();
        self.cursor += usize::from(node.is_some());
        node
    }
}

pub struct ReverseTopologicalOrder<'a> {
    nodes: &'a [NodeId],
    cursor: usize,
}

impl<'a> ReverseTopologicalOrder<'a> {
    pub fn new(nodes: &'a [NodeId]) -> Self {
        Self {
            nodes,
            cursor: nodes.len(),
        }
    }
}

impl Iterator for ReverseTopologicalOrder<'_> {
    type Item = NodeId;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor == 0 {
            return None;
        }
        self.cursor -= 1;
        self.nodes.get(self.cursor).copied()
    }
}

pub struct Parents<'a>(pub std::slice::Iter<'a, NodeId>);
pub struct Children<'a>(pub std::slice::Iter<'a, NodeId>);
pub struct WeightedParents<'a, T>(pub std::slice::Iter<'a, T>);
pub struct WeightedChildren<'a, T>(pub std::slice::Iter<'a, T>);
