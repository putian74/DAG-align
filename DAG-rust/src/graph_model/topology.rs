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
    parents: PackedAdjacency,
    children: PackedAdjacency,
    topological_order: Vec<NodeId>,
    forward_coordinates: Vec<TopologicalCoordinate>,
    reverse_coordinates: Vec<TopologicalCoordinate>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GraphCoordinateSnapshot {
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
        let parents = PackedAdjacency::from_graph_edges(graph, false)?;
        let children = PackedAdjacency::from_graph_edges(graph, true)?;

        let topological_order = topological_sort(node_count, &parents, &children)?;
        let forward_coordinates = forward_coordinates_from_adjacency(&topological_order, &parents);
        let reverse_coordinates = reverse_coordinates_from_adjacency(&topological_order, &children);

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
            parents: PackedAdjacency::empty(),
            children: PackedAdjacency::empty(),
            topological_order: Vec::new(),
            forward_coordinates: Vec::new(),
            reverse_coordinates: Vec::new(),
        }
    }

    pub fn node_count(&self) -> usize {
        self.node_count
    }

    pub fn parents(&self, node: NodeId) -> Result<Parents<'_>> {
        Ok(Parents(self.parents.neighbors(node)?.iter()))
    }

    pub fn children(&self, node: NodeId) -> Result<Children<'_>> {
        Ok(Children(self.children.neighbors(node)?.iter()))
    }

    pub fn topological_order(&self) -> &[NodeId] {
        &self.topological_order
    }

    pub fn forward_coordinates(&self) -> &[TopologicalCoordinate] {
        &self.forward_coordinates
    }

    pub fn reverse_coordinates(&self) -> &[TopologicalCoordinate] {
        &self.reverse_coordinates
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

impl GraphCoordinateSnapshot {
    pub fn from_graph(graph: &FtoDag) -> Result<Self> {
        let topological_order = topological_order_from_graph(graph)?;
        let forward_coordinates = forward_coordinates_from_graph(graph, &topological_order)?;
        let reverse_coordinates = reverse_coordinates_from_graph(graph, &topological_order)?;
        Ok(Self {
            topological_order,
            forward_coordinates,
            reverse_coordinates,
        })
    }

    pub fn topological_order(&self) -> &[NodeId] {
        &self.topological_order
    }

    pub fn forward_coordinates(&self) -> &[TopologicalCoordinate] {
        &self.forward_coordinates
    }

    pub fn reverse_coordinates(&self) -> &[TopologicalCoordinate] {
        &self.reverse_coordinates
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

    pub fn into_topological_order(self) -> Vec<NodeId> {
        self.topological_order
    }
}

pub fn topological_order_from_graph(graph: &FtoDag) -> Result<Vec<NodeId>> {
    let mut indegree = Vec::with_capacity(graph.node_count());
    let mut queue = VecDeque::new();
    for index in 0..graph.node_count() {
        let node = NodeId::try_from(index)?;
        let degree = graph.parents(node)?.len();
        indegree.push(degree);
        if degree == 0 {
            queue.push_back(node);
        }
    }

    let mut order = Vec::with_capacity(graph.node_count());
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for child in graph.children(node)? {
            let degree = &mut indegree[child.to_usize()];
            *degree -= 1;
            if *degree == 0 {
                queue.push_back(*child);
            }
        }
    }

    if order.len() == graph.node_count() {
        Ok(order)
    } else {
        Err(DagError::CycleDetected)
    }
}

fn topological_sort(
    node_count: usize,
    parents: &PackedAdjacency,
    children: &PackedAdjacency,
) -> Result<Vec<NodeId>> {
    let mut indegree = Vec::with_capacity(node_count);
    for node_index in 0..node_count {
        let node_id = NodeId::try_from(node_index).expect("node count exceeds NodeId capacity");
        indegree.push(parents.neighbors(node_id)?.len());
    }
    let mut queue = indegree
        .iter()
        .enumerate()
        .filter(|(_, degree)| **degree == 0)
        .map(|(node, _)| NodeId::try_from(node).expect("node count exceeds NodeId capacity"))
        .collect::<VecDeque<_>>();
    let mut order = Vec::with_capacity(node_count);
    while let Some(node) = queue.pop_front() {
        order.push(node);
        for child in children.neighbors(node)? {
            let degree = &mut indegree[child.to_usize()];
            *degree -= 1;
            if *degree == 0 {
                queue.push_back(*child);
            }
        }
    }
    if order.len() == node_count {
        Ok(order)
    } else {
        Err(DagError::CycleDetected)
    }
}

fn forward_coordinates_from_adjacency(
    order: &[NodeId],
    parents: &PackedAdjacency,
) -> Vec<TopologicalCoordinate> {
    let mut coordinates = vec![TopologicalCoordinate::new(0); parents.node_count()];
    for node in order {
        let coordinate = parents
            .neighbors(*node)
            .expect("topology adjacency covers every node")
            .iter()
            .map(|parent| coordinates[parent.to_usize()].raw())
            .max()
            .unwrap_or(0)
            + 1;
        coordinates[node.to_usize()] = TopologicalCoordinate::new(coordinate);
    }
    coordinates
}

fn reverse_coordinates_from_adjacency(
    order: &[NodeId],
    children: &PackedAdjacency,
) -> Vec<TopologicalCoordinate> {
    let mut coordinates = vec![TopologicalCoordinate::new(0); children.node_count()];
    for node in order.iter().rev() {
        let coordinate = children
            .neighbors(*node)
            .expect("topology adjacency covers every node")
            .iter()
            .map(|child| coordinates[child.to_usize()].raw())
            .max()
            .unwrap_or(0)
            + 1;
        coordinates[node.to_usize()] = TopologicalCoordinate::new(coordinate);
    }
    coordinates
}

fn forward_coordinates_from_graph(
    graph: &FtoDag,
    order: &[NodeId],
) -> Result<Vec<TopologicalCoordinate>> {
    let mut coordinates = vec![TopologicalCoordinate::new(0); graph.node_count()];
    for node in order.iter().copied() {
        let coordinate = graph
            .parents(node)?
            .iter()
            .map(|parent| coordinates[parent.to_usize()].raw())
            .max()
            .unwrap_or(0)
            + 1;
        coordinates[node.to_usize()] = TopologicalCoordinate::new(coordinate);
    }
    Ok(coordinates)
}

fn reverse_coordinates_from_graph(
    graph: &FtoDag,
    order: &[NodeId],
) -> Result<Vec<TopologicalCoordinate>> {
    let mut coordinates = vec![TopologicalCoordinate::new(0); graph.node_count()];
    for node in order.iter().rev().copied() {
        let coordinate = graph
            .children(node)?
            .iter()
            .map(|child| coordinates[child.to_usize()].raw())
            .max()
            .unwrap_or(0)
            + 1;
        coordinates[node.to_usize()] = TopologicalCoordinate::new(coordinate);
    }
    Ok(coordinates)
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

#[derive(Clone, Debug, Eq, PartialEq)]
struct PackedAdjacency {
    offsets: Vec<u32>,
    nodes: Vec<NodeId>,
}

impl PackedAdjacency {
    fn empty() -> Self {
        Self {
            offsets: vec![0],
            nodes: Vec::new(),
        }
    }

    fn from_graph_edges(graph: &FtoDag, use_children: bool) -> Result<Self> {
        let node_count = graph.node_count();
        let mut counts = vec![0u32; node_count];
        for edge in graph.edges() {
            let owner = if use_children {
                edge.key.parent.to_usize()
            } else {
                edge.key.child.to_usize()
            };
            if owner >= node_count {
                return Err(DagError::InvalidEdge {
                    parent: edge.key.parent.to_usize(),
                    child: edge.key.child.to_usize(),
                });
            }
            counts[owner] = counts[owner]
                .checked_add(1)
                .ok_or(DagError::ValueDoesNotFit {
                    value: u128::from(counts[owner]) + 1,
                    bits: 32,
                })?;
        }
        let mut offsets = Vec::with_capacity(node_count + 1);
        offsets.push(0);
        let mut running_total = 0u32;
        for count in &counts {
            running_total = running_total
                .checked_add(*count)
                .ok_or(DagError::ValueDoesNotFit {
                    value: u128::from(running_total) + u128::from(*count),
                    bits: 32,
                })?;
            offsets.push(running_total);
        }
        let mut positions = offsets[..node_count].to_vec();
        let mut nodes = vec![NodeId::new(0); graph.edges().len()];
        for edge in graph.edges() {
            let owner = if use_children {
                edge.key.parent.to_usize()
            } else {
                edge.key.child.to_usize()
            };
            let neighbor = if use_children {
                edge.key.child
            } else {
                edge.key.parent
            };
            let position = positions[owner] as usize;
            nodes[position] = neighbor;
            positions[owner] += 1;
        }
        Ok(Self { offsets, nodes })
    }

    fn node_count(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    fn neighbors(&self, node: NodeId) -> Result<&[NodeId]> {
        let start = *self
            .offsets
            .get(node.to_usize())
            .ok_or(DagError::MissingNode {
                node: node.to_usize(),
            })? as usize;
        let end = *self
            .offsets
            .get(node.to_usize() + 1)
            .ok_or(DagError::MissingNode {
                node: node.to_usize(),
            })? as usize;
        self.nodes.get(start..end).ok_or(DagError::InvalidRange {
            start,
            end,
            len: self.nodes.len(),
        })
    }
}
