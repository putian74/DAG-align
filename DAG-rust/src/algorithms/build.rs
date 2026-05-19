//! Incremental sequence-to-FTO-DAG construction interfaces.

use crate::foundations::error::DagError;
use crate::foundations::error::Result;
use crate::foundations::id::{
    GraphId, NodeId, ProvenancePosition, SequenceId, TopologicalCoordinate, Weight,
};
use crate::graph_model::graph::{EdgeIndexStrategy, EdgeKey, FtoDag, NodeKind};
use crate::graph_model::provenance::{ProvenanceRecord, ProvenanceStorageStrategy};
use crate::graph_model::topology::DagTopology;
use crate::sequence_model::alphabet::Alphabet;
use crate::sequence_model::fragment::{
    FragmentEncoder, FragmentKey, FragmentOccurrence, PathPositionKind,
};
use crate::sequence_model::sequence::{EncodedSequence, SequenceInput};
use std::collections::{HashSet, VecDeque};
use std::time::{Duration, Instant};

const GREEDY_ANCHOR_SET_LIMIT: usize = 2_048;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BuildOrderingPolicy {
    InputOrder,
    ChunkLocalSimilarity,
    SketchBucketedSimilarity,
    UserProvided,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BuildConfig {
    pub fragment_len: usize,
    pub ordering_policy: BuildOrderingPolicy,
    pub graph_id: GraphId,
    pub min_initial_match_ratio: Option<SimilarityThreshold>,
    pub rejection_policy: RejectionPolicy,
    pub topology_update_strategy: TopologyUpdateStrategy,
    pub provenance_storage_strategy: ProvenanceStorageStrategy,
    pub edge_index_strategy: EdgeIndexStrategy,
    pub collect_topology_counters: bool,
    pub topology_rebuild_queue_threshold: Option<usize>,
}

impl BuildConfig {
    pub const fn new(fragment_len: usize) -> Self {
        Self {
            fragment_len,
            ordering_policy: BuildOrderingPolicy::InputOrder,
            graph_id: GraphId::new(0),
            min_initial_match_ratio: None,
            rejection_policy: RejectionPolicy::RecordAndSkip,
            topology_update_strategy: TopologyUpdateStrategy::FullRebuild,
            provenance_storage_strategy: ProvenanceStorageStrategy::FullRecords,
            edge_index_strategy: EdgeIndexStrategy::GlobalHash,
            collect_topology_counters: false,
            topology_rebuild_queue_threshold: None,
        }
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub enum TopologyUpdateStrategy {
    #[default]
    FullRebuild,
    IncrementalAffectedRegion,
    IncrementalForwardOnly,
    IncrementalForwardRescan,
    IncrementalForwardRelaxation,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct SimilarityThreshold {
    basis_points: u16,
}

impl SimilarityThreshold {
    pub const BASIS_POINTS_PER_ONE: u16 = 10_000;

    pub fn from_basis_points(basis_points: u32) -> Result<Self> {
        if basis_points <= u32::from(Self::BASIS_POINTS_PER_ONE) {
            Ok(Self {
                basis_points: basis_points as u16,
            })
        } else {
            Err(DagError::InvalidThreshold { basis_points })
        }
    }

    pub const fn all() -> Self {
        Self {
            basis_points: Self::BASIS_POINTS_PER_ONE,
        }
    }

    pub const fn none() -> Self {
        Self { basis_points: 0 }
    }

    pub const fn basis_points(self) -> u16 {
        self.basis_points
    }

    pub fn meets(self, mergeable: usize, total: usize) -> bool {
        if total == 0 {
            return false;
        }
        mergeable.saturating_mul(usize::from(Self::BASIS_POINTS_PER_ONE))
            >= total.saturating_mul(usize::from(self.basis_points))
    }

    fn required_matches(self, total: usize) -> usize {
        if total == 0 {
            return 1;
        }
        let numerator = total.saturating_mul(usize::from(self.basis_points));
        numerator.div_ceil(usize::from(Self::BASIS_POINTS_PER_ONE))
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum RejectionPolicy {
    RecordAndSkip,
    ForceIntegrate,
    DeferForRetry,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnchorCandidate {
    pub node: NodeId,
    pub kind: NodeKind,
    pub coordinate: Option<TopologicalCoordinate>,
    pub reverse_coordinate: Option<TopologicalCoordinate>,
    pub weight: Weight,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnchorCandidateSet {
    pub position: usize,
    pub kind: NodeKind,
    pub fragment: FragmentKey,
    pub candidates: Vec<AnchorCandidate>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SelectedAnchor {
    pub node: NodeId,
    pub coordinate: TopologicalCoordinate,
    pub weight: Weight,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AnchorDecision {
    Matched(NodeId),
    Unmatched(AnchorRejectReason),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum AnchorRejectReason {
    NoCandidate,
    TieForMaxWeight,
    NodeKindMismatch,
    WouldCreateCycle,
    MissingTopology,
    NotSelected,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AnchorPath {
    pub decisions: Vec<AnchorDecision>,
    pub selected_anchors: Vec<Option<SelectedAnchor>>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct PathRange {
    pub start: usize,
    pub end: usize,
}

impl PathRange {
    pub const fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    pub fn try_new(start: usize, end: usize) -> Result<Self> {
        if start <= end {
            Ok(Self { start, end })
        } else {
            Err(crate::foundations::error::DagError::InvalidRange {
                start,
                end,
                len: end,
            })
        }
    }

    pub const fn len(self) -> usize {
        self.end - self.start
    }

    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct CoordinateInterval {
    pub left: Option<TopologicalCoordinate>,
    pub right: Option<TopologicalCoordinate>,
}

impl CoordinateInterval {
    pub const fn unbounded() -> Self {
        Self {
            left: None,
            right: None,
        }
    }

    pub fn contains_open(self, coordinate: TopologicalCoordinate) -> bool {
        self.left.is_none_or(|left| left < coordinate)
            && self.right.is_none_or(|right| coordinate < right)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnchorBlock {
    pub path_range: PathRange,
    pub coordinate_start: TopologicalCoordinate,
    pub coordinate_end: TopologicalCoordinate,
    pub score: Weight,
    pub anchors: Vec<NodeId>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnanchoredBlock {
    pub path_range: PathRange,
    pub coordinate_interval: CoordinateInterval,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BlockPlan {
    pub accepted_anchors: Vec<AnchorBlock>,
    pub rejected_anchors: Vec<AnchorBlock>,
    pub unanchored_blocks: Vec<UnanchoredBlock>,
    pub reused_nodes: Vec<NodeId>,
    pub new_nodes: Vec<NodeId>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BuildSequenceResult {
    pub sequence_id: SequenceId,
    pub node_path: Vec<NodeId>,
    pub inserted_edges: usize,
    pub provenance_records_added: usize,
    pub block_plan: BlockPlan,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct RejectedSequence {
    pub sequence_id: SequenceId,
    pub subgraph_id: GraphId,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum IntegrationDecision {
    Proceed,
    Reject(RejectedSequence),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SequenceBuildOutcome {
    Integrated(BuildSequenceResult),
    Rejected(RejectedSequence),
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BuildReport {
    pub integrated_sequences: Vec<SequenceId>,
    pub rejected_sequences: Vec<RejectedSequence>,
    pub attempted_sequences: usize,
    pub total_nodes_created: usize,
    pub total_edges_inserted: usize,
    pub total_provenance_records_added: usize,
    pub topology_counters: TopologyUpdateCounters,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct TopologyUpdateCounters {
    pub full_rebuilds: usize,
    pub full_rebuild_fallbacks: usize,
    pub new_nodes: usize,
    pub inserted_edges: usize,
    pub inserted_edges_to_new_children: usize,
    pub inserted_edges_to_existing_children: usize,
    pub safe_forward_edges: usize,
    pub forward_relax_attempts: usize,
    pub forward_coordinate_updates: usize,
    pub forward_queue_pops: usize,
    pub forward_parent_scans: usize,
    pub reverse_coordinate_updates: usize,
    pub reverse_queue_pops: usize,
    pub reverse_child_scans: usize,
}

impl TopologyUpdateCounters {
    fn add_assign(&mut self, other: Self) {
        self.full_rebuilds += other.full_rebuilds;
        self.full_rebuild_fallbacks += other.full_rebuild_fallbacks;
        self.new_nodes += other.new_nodes;
        self.inserted_edges += other.inserted_edges;
        self.inserted_edges_to_new_children += other.inserted_edges_to_new_children;
        self.inserted_edges_to_existing_children += other.inserted_edges_to_existing_children;
        self.safe_forward_edges += other.safe_forward_edges;
        self.forward_relax_attempts += other.forward_relax_attempts;
        self.forward_coordinate_updates += other.forward_coordinate_updates;
        self.forward_queue_pops += other.forward_queue_pops;
        self.forward_parent_scans += other.forward_parent_scans;
        self.reverse_coordinate_updates += other.reverse_coordinate_updates;
        self.reverse_queue_pops += other.reverse_queue_pops;
        self.reverse_child_scans += other.reverse_child_scans;
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct BuildTimingBreakdown {
    pub initialize: Duration,
    pub initial_match: Duration,
    pub topology: Duration,
    pub fragment_occurrences: Duration,
    pub candidate_collection: Duration,
    pub anchor_selection: Duration,
    pub block_planning: Duration,
    pub mutation: Duration,
    pub report_update: Duration,
    pub total: Duration,
}

impl BuildTimingBreakdown {
    pub fn add_assign(&mut self, other: Self) {
        self.initialize += other.initialize;
        self.initial_match += other.initial_match;
        self.topology += other.topology;
        self.fragment_occurrences += other.fragment_occurrences;
        self.candidate_collection += other.candidate_collection;
        self.anchor_selection += other.anchor_selection;
        self.block_planning += other.block_planning;
        self.mutation += other.mutation;
        self.report_update += other.report_update;
        self.total += other.total;
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfiledSequenceBuildOutcome {
    pub outcome: SequenceBuildOutcome,
    pub timing: BuildTimingBreakdown,
}

#[derive(Clone, Debug)]
pub struct FtoDagBuilder {
    config: BuildConfig,
    graph: FtoDag,
    report: BuildReport,
    topology_cache: Option<TopologyCache>,
}

impl FtoDagBuilder {
    pub fn new(config: BuildConfig) -> Self {
        Self {
            graph: FtoDag::with_provenance_and_edge_storage(
                config.fragment_len,
                config.provenance_storage_strategy,
                config.edge_index_strategy,
            ),
            config,
            report: BuildReport::default(),
            topology_cache: None,
        }
    }

    pub fn from_graph(graph: FtoDag, mut config: BuildConfig) -> Result<Self> {
        if graph.fragment_len() != config.fragment_len {
            return Err(DagError::InvalidStorage(format!(
                "builder fragment length {} does not match graph fragment length {}",
                config.fragment_len,
                graph.fragment_len()
            )));
        }
        config.provenance_storage_strategy = graph.provenance_storage_strategy();
        config.edge_index_strategy = graph.edge_index_strategy();
        Ok(Self {
            config,
            graph,
            report: BuildReport::default(),
            topology_cache: None,
        })
    }

    pub fn config(&self) -> BuildConfig {
        self.config
    }

    pub fn graph(&self) -> &FtoDag {
        &self.graph
    }

    pub fn report(&self) -> &BuildReport {
        &self.report
    }

    pub fn into_graph(self) -> FtoDag {
        self.graph
    }

    pub fn initialize_from_encoded<E: FragmentEncoder>(
        &mut self,
        sequence_id: SequenceId,
        sequence: &EncodedSequence,
        encoder: &E,
    ) -> Result<BuildSequenceResult> {
        let occurrences = encoder.encode_occurrences(sequence, self.config.fragment_len)?;
        self.initialize_from_occurrences(sequence_id, occurrences)
    }

    fn initialize_from_occurrences(
        &mut self,
        sequence_id: SequenceId,
        occurrences: Vec<FragmentOccurrence>,
    ) -> Result<BuildSequenceResult> {
        let mut previous = None;
        let mut node_path = Vec::new();
        let mut inserted_edges = 0;
        for occurrence in occurrences {
            let node_id = self.add_occurrence_node(sequence_id, occurrence)?;
            if let Some(previous) = previous {
                let update = self
                    .graph
                    .add_or_increment_edge(previous, node_id, Weight::new(1))?;
                inserted_edges += usize::from(update.inserted);
            }
            previous = Some(node_id);
            node_path.push(node_id);
        }
        let provenance_records_added = node_path.len();
        Ok(BuildSequenceResult {
            sequence_id,
            block_plan: BlockPlan {
                new_nodes: node_path.clone(),
                ..BlockPlan::default()
            },
            node_path,
            inserted_edges,
            provenance_records_added,
        })
    }

    pub fn add_sequence_from_encoded<E: FragmentEncoder>(
        &mut self,
        sequence_id: SequenceId,
        sequence: &EncodedSequence,
        encoder: &E,
    ) -> Result<SequenceBuildOutcome> {
        Ok(self
            .add_sequence_from_encoded_profiled(sequence_id, sequence, encoder)?
            .outcome)
    }

    pub fn add_sequence_from_occurrences(
        &mut self,
        sequence_id: SequenceId,
        occurrences: Vec<FragmentOccurrence>,
    ) -> Result<SequenceBuildOutcome> {
        Ok(self
            .add_sequence_from_occurrences_profiled(sequence_id, occurrences)?
            .outcome)
    }

    pub fn add_sequence_from_encoded_profiled<E: FragmentEncoder>(
        &mut self,
        sequence_id: SequenceId,
        sequence: &EncodedSequence,
        encoder: &E,
    ) -> Result<ProfiledSequenceBuildOutcome> {
        let total_start = Instant::now();
        let mut timing = BuildTimingBreakdown::default();
        let start = Instant::now();
        let occurrences = encoder.encode_occurrences(sequence, self.config.fragment_len)?;
        timing.fragment_occurrences += start.elapsed();
        self.add_sequence_from_occurrences_profiled_with_timing(
            sequence_id,
            occurrences,
            total_start,
            timing,
        )
    }

    pub fn add_sequence_from_occurrences_profiled(
        &mut self,
        sequence_id: SequenceId,
        occurrences: Vec<FragmentOccurrence>,
    ) -> Result<ProfiledSequenceBuildOutcome> {
        self.add_sequence_from_occurrences_profiled_with_timing(
            sequence_id,
            occurrences,
            Instant::now(),
            BuildTimingBreakdown::default(),
        )
    }

    fn add_sequence_from_occurrences_profiled_with_timing(
        &mut self,
        sequence_id: SequenceId,
        occurrences: Vec<FragmentOccurrence>,
        total_start: Instant,
        mut timing: BuildTimingBreakdown,
    ) -> Result<ProfiledSequenceBuildOutcome> {
        self.report.attempted_sequences += 1;
        if self.graph.node_count() == 0 {
            let start = Instant::now();
            let result = self.initialize_from_occurrences(sequence_id, occurrences)?;
            timing.initialize += start.elapsed();
            if self.config.topology_update_strategy.is_incremental() {
                let start = Instant::now();
                self.rebuild_topology_cache()?;
                timing.topology += start.elapsed();
            }
            let start = Instant::now();
            self.record_integrated_sequence(&result);
            timing.report_update += start.elapsed();
            timing.total = total_start.elapsed();
            return Ok(ProfiledSequenceBuildOutcome {
                outcome: SequenceBuildOutcome::Integrated(result),
                timing,
            });
        }

        let start = Instant::now();
        if let IntegrationDecision::Reject(rejected) =
            self.initial_match_decision_for_occurrences(sequence_id, &occurrences)
        {
            timing.initial_match += start.elapsed();
            let start = Instant::now();
            self.report.rejected_sequences.push(rejected);
            timing.report_update += start.elapsed();
            timing.total = total_start.elapsed();
            return Ok(ProfiledSequenceBuildOutcome {
                outcome: SequenceBuildOutcome::Rejected(rejected),
                timing,
            });
        }
        timing.initial_match += start.elapsed();

        let start = Instant::now();
        let topology = match self.config.topology_update_strategy {
            TopologyUpdateStrategy::FullRebuild => Some(DagTopology::try_from_graph(&self.graph)?),
            TopologyUpdateStrategy::IncrementalAffectedRegion
            | TopologyUpdateStrategy::IncrementalForwardOnly
            | TopologyUpdateStrategy::IncrementalForwardRescan
            | TopologyUpdateStrategy::IncrementalForwardRelaxation => {
                self.ensure_topology_cache()?;
                None
            }
        };
        timing.topology += start.elapsed();
        let (candidate_sets, anchor_path) = if topology.is_none()
            && occurrences.len() > GREEDY_ANCHOR_SET_LIMIT
        {
            let cache = self
                .topology_cache
                .as_ref()
                .expect("incremental topology cache is initialized");
            let start = Instant::now();
            let selected =
                self.collect_greedy_anchor_path_with_cache(sequence_id, &occurrences, cache)?;
            timing.candidate_collection += start.elapsed();
            selected
        } else {
            let start = Instant::now();
            let candidate_sets = if let Some(topology) = &topology {
                self.collect_anchor_candidates_for_sequence(
                    Some(sequence_id),
                    &occurrences,
                    Some(topology as &dyn CoordinateProvider),
                )?
            } else {
                let cache = self
                    .topology_cache
                    .as_ref()
                    .expect("incremental topology cache is initialized");
                self.collect_anchor_candidates_with_cache(Some(sequence_id), &occurrences, cache)?
            };
            timing.candidate_collection += start.elapsed();
            let start = Instant::now();
            let anchor_path = self.select_monotone_anchor_path(&candidate_sets);
            timing.anchor_selection += start.elapsed();
            (candidate_sets, anchor_path)
        };
        let start = Instant::now();
        let mut block_plan = block_plan_from_anchor_path(&anchor_path, &candidate_sets);
        timing.block_planning += start.elapsed();
        let next_anchors = next_matched_anchors_after(&anchor_path, &candidate_sets);
        let mut node_path = Vec::with_capacity(occurrences.len());
        let mut previous = None;
        let mut last_existing_coordinate = None;
        let mut used_existing_node_marks = vec![false; self.graph.node_count()];
        let mut inserted_edges = 0;
        let mut inserted_edge_keys = Vec::new();
        let mut provenance_records_added = 0;

        let start = Instant::now();
        for (index, (occurrence, decision)) in occurrences
            .into_iter()
            .zip(anchor_path.decisions.iter())
            .enumerate()
        {
            let node_id = match decision {
                AnchorDecision::Matched(node_id) => {
                    self.add_source_to_existing_node(*node_id, sequence_id, occurrence.position)?;
                    mark_used_node(&mut used_existing_node_marks, *node_id);
                    last_existing_coordinate =
                        selected_coordinate(&anchor_path, &candidate_sets, index, *node_id)
                            .or(last_existing_coordinate);
                    *node_id
                }
                AnchorDecision::Unmatched(_) => {
                    let next_anchor = next_anchors[index];
                    let interval = CoordinateInterval {
                        left: last_existing_coordinate,
                        right: next_anchor.map(|(_, coordinate)| coordinate),
                    };
                    if let Some(reuse) = select_bounded_reuse_candidate_with_marks(
                        &candidate_sets[index],
                        interval,
                        &used_existing_node_marks,
                        &self.graph,
                        previous,
                        next_anchor.map(|(node, _)| node),
                    ) {
                        let node_id = reuse.node;
                        self.add_source_to_existing_node(
                            node_id,
                            sequence_id,
                            occurrence.position,
                        )?;
                        mark_used_node(&mut used_existing_node_marks, node_id);
                        last_existing_coordinate = reuse.coordinate.or(last_existing_coordinate);
                        block_plan.reused_nodes.push(node_id);
                        node_id
                    } else {
                        let node_id = self.add_occurrence_node(sequence_id, occurrence)?;
                        block_plan.new_nodes.push(node_id);
                        node_id
                    }
                }
            };
            provenance_records_added += 1;
            if let Some(previous) = previous {
                let update = self
                    .graph
                    .add_or_increment_edge(previous, node_id, Weight::new(1))?;
                inserted_edges += usize::from(update.inserted);
                if update.inserted {
                    inserted_edge_keys.push(update.key);
                }
            }
            previous = Some(node_id);
            node_path.push(node_id);
        }
        timing.mutation += start.elapsed();

        if self.config.topology_update_strategy.is_incremental() {
            let start = Instant::now();
            self.apply_topology_delta(&inserted_edge_keys)?;
            timing.topology += start.elapsed();
        }

        let result = BuildSequenceResult {
            sequence_id,
            node_path,
            inserted_edges,
            provenance_records_added,
            block_plan,
        };
        let start = Instant::now();
        self.record_integrated_sequence(&result);
        timing.report_update += start.elapsed();
        timing.total = total_start.elapsed();
        Ok(ProfiledSequenceBuildOutcome {
            outcome: SequenceBuildOutcome::Integrated(result),
            timing,
        })
    }

    pub fn initial_match_decision<E: FragmentEncoder>(
        &self,
        sequence_id: SequenceId,
        sequence: &EncodedSequence,
        encoder: &E,
    ) -> Result<IntegrationDecision> {
        let occurrences = encoder.encode_occurrences(sequence, self.config.fragment_len)?;
        Ok(self.initial_match_decision_for_occurrences(sequence_id, &occurrences))
    }

    fn initial_match_decision_for_occurrences(
        &self,
        sequence_id: SequenceId,
        occurrences: &[FragmentOccurrence],
    ) -> IntegrationDecision {
        if self.graph.node_count() == 0
            || self.config.rejection_policy == RejectionPolicy::ForceIntegrate
        {
            return IntegrationDecision::Proceed;
        }

        let Some(threshold) = self.config.min_initial_match_ratio else {
            return IntegrationDecision::Proceed;
        };

        let required_matches = threshold.required_matches(occurrences.len());
        if required_matches == 0 {
            return IntegrationDecision::Proceed;
        }

        let mut mergeable = 0_usize;
        for (index, occurrence) in occurrences.iter().enumerate() {
            let node_kind = node_kind_for_path_position(occurrence.kind);
            if !self
                .graph
                .fragment_index()
                .nodes_for(&occurrence.key, node_kind)
                .is_empty()
            {
                mergeable += 1;
                if mergeable >= required_matches {
                    return IntegrationDecision::Proceed;
                }
            }
            let remaining = occurrences.len() - index - 1;
            if mergeable + remaining < required_matches {
                break;
            }
        }

        if mergeable >= required_matches {
            IntegrationDecision::Proceed
        } else {
            IntegrationDecision::Reject(RejectedSequence {
                sequence_id,
                subgraph_id: self.config.graph_id,
            })
        }
    }

    pub fn collect_anchor_candidates<E: FragmentEncoder>(
        &self,
        sequence: &EncodedSequence,
        encoder: &E,
        topology: Option<&DagTopology>,
    ) -> Result<Vec<AnchorCandidateSet>> {
        let occurrences = encoder.encode_occurrences(sequence, self.config.fragment_len)?;
        self.collect_anchor_candidates_for_sequence(
            None,
            &occurrences,
            topology.map(|topology| topology as &dyn CoordinateProvider),
        )
    }

    fn collect_anchor_candidates_for_sequence(
        &self,
        sequence_id: Option<SequenceId>,
        occurrences: &[FragmentOccurrence],
        coordinates: Option<&dyn CoordinateProvider>,
    ) -> Result<Vec<AnchorCandidateSet>> {
        let mut sets = Vec::with_capacity(occurrences.len());
        for occurrence in occurrences {
            let kind = node_kind_for_path_position(occurrence.kind);
            let nodes = self.graph.fragment_index().nodes_for(&occurrence.key, kind);
            let mut candidates = Vec::with_capacity(nodes.len());
            for node_id in nodes.iter().copied() {
                if sequence_id
                    .is_some_and(|sequence_id| !self.can_node_accept_sequence(node_id, sequence_id))
                {
                    continue;
                }

                let node = self.graph.node(node_id)?;
                let coordinate = coordinates
                    .map(|coordinates| coordinates.forward_coordinate(node_id))
                    .transpose()?;
                let reverse_coordinate = coordinates
                    .map(|coordinates| coordinates.reverse_coordinate(node_id))
                    .transpose()?
                    .flatten();
                candidates.push(AnchorCandidate {
                    node: node_id,
                    kind,
                    coordinate,
                    reverse_coordinate,
                    weight: node.weight,
                });
            }
            sets.push(AnchorCandidateSet {
                position: occurrence.position,
                kind,
                fragment: occurrence.key.clone(),
                candidates,
            });
        }
        Ok(sets)
    }

    fn collect_anchor_candidates_with_cache(
        &self,
        sequence_id: Option<SequenceId>,
        occurrences: &[FragmentOccurrence],
        cache: &TopologyCache,
    ) -> Result<Vec<AnchorCandidateSet>> {
        let mut sets = Vec::with_capacity(occurrences.len());
        for occurrence in occurrences {
            let kind = node_kind_for_path_position(occurrence.kind);
            let nodes = self.graph.fragment_index().nodes_for(&occurrence.key, kind);
            let mut candidates = Vec::with_capacity(nodes.len());
            for node_id in nodes.iter().copied() {
                if sequence_id
                    .is_some_and(|sequence_id| !self.can_node_accept_sequence(node_id, sequence_id))
                {
                    continue;
                }

                let node_index = node_id.to_usize();
                let node = self
                    .graph
                    .nodes()
                    .get(node_index)
                    .ok_or(DagError::MissingNode { node: node_index })?;
                let coordinate = cache
                    .forward_coordinates
                    .get(node_index)
                    .copied()
                    .ok_or(DagError::MissingNode { node: node_index })?;
                let reverse_coordinate = cache
                    .reverse_coordinates
                    .as_ref()
                    .map(|coordinates| {
                        coordinates
                            .get(node_index)
                            .copied()
                            .ok_or(DagError::MissingNode { node: node_index })
                    })
                    .transpose()?;
                candidates.push(AnchorCandidate {
                    node: node_id,
                    kind,
                    coordinate: Some(coordinate),
                    reverse_coordinate,
                    weight: node.weight,
                });
            }
            sets.push(AnchorCandidateSet {
                position: occurrence.position,
                kind,
                fragment: occurrence.key.clone(),
                candidates,
            });
        }
        Ok(sets)
    }

    fn collect_greedy_anchor_path_with_cache(
        &self,
        sequence_id: SequenceId,
        occurrences: &[FragmentOccurrence],
        cache: &TopologyCache,
    ) -> Result<(Vec<AnchorCandidateSet>, AnchorPath)> {
        let mut candidate_sets = Vec::with_capacity(occurrences.len());
        let mut decisions = Vec::with_capacity(occurrences.len());
        let mut selected_anchors = Vec::with_capacity(occurrences.len());
        let mut used_nodes = UsedNodeTracker::new(Some(self.graph.node_count()));
        let mut previous: Option<NodeId> = None;
        let mut last_coordinate: Option<TopologicalCoordinate> = None;

        for occurrence in occurrences {
            let kind = node_kind_for_path_position(occurrence.kind);
            let state = GreedySelectionState {
                previous,
                last_coordinate,
                used_nodes: &used_nodes,
            };
            if let Some(candidate) = self.direct_child_extension_for_occurrence(
                sequence_id,
                occurrence,
                kind,
                state,
                cache,
            )? {
                used_nodes.insert(candidate.node);
                previous = Some(candidate.node);
                last_coordinate = candidate.coordinate;
                decisions.push(AnchorDecision::Matched(candidate.node));
                selected_anchors.push(candidate.coordinate.map(|coordinate| SelectedAnchor {
                    node: candidate.node,
                    coordinate,
                    weight: candidate.weight,
                }));
                candidate_sets.push(AnchorCandidateSet {
                    position: occurrence.position,
                    kind,
                    fragment: occurrence.key.clone(),
                    candidates: Vec::new(),
                });
                continue;
            }

            let candidate_set = self.collect_anchor_candidate_set_with_cache(
                Some(sequence_id),
                occurrence,
                kind,
                cache,
            )?;
            let mut best: Option<(&AnchorCandidate, Weight)> = None;
            let mut has_coordinate = false;
            for candidate in &candidate_set.candidates {
                let Some(coordinate) = candidate.coordinate else {
                    continue;
                };
                has_coordinate = true;
                if last_coordinate.is_some_and(|last| last >= coordinate)
                    || used_nodes.contains(candidate.node)
                {
                    continue;
                }
                let score = greedy_candidate_score(candidate, previous, &|key| {
                    self.graph.edge_weight(key).unwrap_or_default()
                });
                if best.is_none_or(|(best_candidate, best_score)| {
                    better_greedy_candidate(candidate, score, best_candidate, best_score)
                }) {
                    best = Some((candidate, score));
                }
            }

            if let Some((candidate, _)) = best {
                used_nodes.insert(candidate.node);
                previous = Some(candidate.node);
                last_coordinate = candidate.coordinate;
                decisions.push(AnchorDecision::Matched(candidate.node));
                selected_anchors.push(candidate.coordinate.map(|coordinate| SelectedAnchor {
                    node: candidate.node,
                    coordinate,
                    weight: candidate.weight,
                }));
            } else if candidate_set.candidates.is_empty() {
                decisions.push(AnchorDecision::Unmatched(AnchorRejectReason::NoCandidate));
                selected_anchors.push(None);
            } else if has_coordinate {
                decisions.push(AnchorDecision::Unmatched(AnchorRejectReason::NotSelected));
                selected_anchors.push(None);
            } else {
                decisions.push(AnchorDecision::Unmatched(
                    AnchorRejectReason::MissingTopology,
                ));
                selected_anchors.push(None);
            }
            candidate_sets.push(candidate_set);
        }

        Ok((
            candidate_sets,
            AnchorPath {
                decisions,
                selected_anchors,
            },
        ))
    }

    fn collect_anchor_candidate_set_with_cache(
        &self,
        sequence_id: Option<SequenceId>,
        occurrence: &FragmentOccurrence,
        kind: NodeKind,
        cache: &TopologyCache,
    ) -> Result<AnchorCandidateSet> {
        let nodes = self.graph.fragment_index().nodes_for(&occurrence.key, kind);
        let mut candidates = Vec::with_capacity(nodes.len());
        for node_id in nodes.iter().copied() {
            if sequence_id
                .is_some_and(|sequence_id| !self.can_node_accept_sequence(node_id, sequence_id))
            {
                continue;
            }

            candidates.push(self.anchor_candidate_from_cache(node_id, kind, cache)?);
        }
        Ok(AnchorCandidateSet {
            position: occurrence.position,
            kind,
            fragment: occurrence.key.clone(),
            candidates,
        })
    }

    fn direct_child_extension_for_occurrence(
        &self,
        sequence_id: SequenceId,
        occurrence: &FragmentOccurrence,
        kind: NodeKind,
        state: GreedySelectionState<'_>,
        cache: &TopologyCache,
    ) -> Result<Option<AnchorCandidate>> {
        let Some(previous) = state.previous else {
            return Ok(None);
        };
        let Some(last_coordinate) = state.last_coordinate else {
            return Ok(None);
        };
        let next_coordinate = TopologicalCoordinate::new(last_coordinate.raw() + 1);
        let mut best = None;
        for child in self.graph.children(previous)? {
            if state.used_nodes.contains(*child)
                || !self.can_node_accept_sequence(*child, sequence_id)
            {
                continue;
            }
            let node = self.graph.node(*child)?;
            if node.kind != kind || node.fragment != occurrence.key {
                continue;
            }
            let candidate = self.anchor_candidate_from_cache(*child, kind, cache)?;
            if candidate.coordinate != Some(next_coordinate) {
                continue;
            }
            let score = self
                .graph
                .edge_weight(EdgeKey {
                    parent: previous,
                    child: candidate.node,
                })
                .unwrap_or_default();
            if score == Weight::new(0) {
                continue;
            }
            if best.as_ref().is_none_or(|(best_candidate, best_score)| {
                better_greedy_candidate(&candidate, score, best_candidate, *best_score)
            }) {
                best = Some((candidate, score));
            }
        }
        Ok(best.map(|(candidate, _)| candidate))
    }

    fn anchor_candidate_from_cache(
        &self,
        node_id: NodeId,
        kind: NodeKind,
        cache: &TopologyCache,
    ) -> Result<AnchorCandidate> {
        let node_index = node_id.to_usize();
        let node = self
            .graph
            .nodes()
            .get(node_index)
            .ok_or(DagError::MissingNode { node: node_index })?;
        let coordinate = cache
            .forward_coordinates
            .get(node_index)
            .copied()
            .ok_or(DagError::MissingNode { node: node_index })?;
        let reverse_coordinate = cache
            .reverse_coordinates
            .as_ref()
            .map(|coordinates| {
                coordinates
                    .get(node_index)
                    .copied()
                    .ok_or(DagError::MissingNode { node: node_index })
            })
            .transpose()?;
        Ok(AnchorCandidate {
            node: node_id,
            kind,
            coordinate: Some(coordinate),
            reverse_coordinate,
            weight: node.weight,
        })
    }

    pub fn select_monotone_anchor_path(&self, candidate_sets: &[AnchorCandidateSet]) -> AnchorPath {
        select_monotone_anchor_path_with_graph(candidate_sets, &self.graph)
    }

    pub fn build_from_input<E: FragmentEncoder>(
        &mut self,
        input: &mut dyn SequenceInput,
        alphabet: &dyn Alphabet,
        encoder: &E,
    ) -> Result<BuildReport> {
        self.build_from_input_starting_at(input, alphabet, encoder, SequenceId::new(0))
    }

    pub fn build_from_input_starting_at<E: FragmentEncoder>(
        &mut self,
        input: &mut dyn SequenceInput,
        alphabet: &dyn Alphabet,
        encoder: &E,
        first_sequence_id: SequenceId,
    ) -> Result<BuildReport> {
        let mut offset = 0_usize;
        while let Some(record) = input.next_record()? {
            let sequence_id =
                SequenceId::try_from(first_sequence_id.to_usize().saturating_add(offset))?;
            let encoded = EncodedSequence::encode(record, alphabet)?;
            self.add_sequence_from_encoded(sequence_id, &encoded, encoder)?;
            offset += 1;
        }
        Ok(self.report.clone())
    }

    fn add_occurrence_node(
        &mut self,
        sequence_id: SequenceId,
        occurrence: FragmentOccurrence,
    ) -> Result<NodeId> {
        let position = occurrence.position;
        let node_id = self
            .graph
            .add_node(occurrence.key, node_kind_for_path_position(occurrence.kind))?;
        self.graph.add_provenance_record(
            node_id,
            ProvenanceRecord {
                sequence_id,
                position: ProvenancePosition::new(position as u64),
            },
        )?;
        Ok(node_id)
    }

    fn add_source_to_existing_node(
        &mut self,
        node_id: NodeId,
        sequence_id: SequenceId,
        position: usize,
    ) -> Result<()> {
        if !self.can_node_accept_sequence(node_id, sequence_id) {
            return Err(DagError::DuplicateSequenceProvenance {
                node: node_id.to_usize(),
                sequence: sequence_id.raw(),
            });
        }
        self.graph.add_provenance_record(
            node_id,
            ProvenanceRecord {
                sequence_id,
                position: ProvenancePosition::new(position as u64),
            },
        )
    }

    fn can_node_accept_sequence(&self, node_id: NodeId, sequence_id: SequenceId) -> bool {
        self.graph
            .can_node_accept_sequence(node_id, sequence_id)
            .unwrap_or(false)
    }

    fn record_integrated_sequence(&mut self, result: &BuildSequenceResult) {
        self.report.integrated_sequences.push(result.sequence_id);
        self.report.total_nodes_created += result.block_plan.new_nodes.len();
        self.report.total_edges_inserted += result.inserted_edges;
        self.report.total_provenance_records_added += result.provenance_records_added;
    }

    fn ensure_topology_cache(&mut self) -> Result<()> {
        let needs_rebuild = self
            .topology_cache
            .as_ref()
            .is_none_or(|cache| cache.node_count() != self.graph.node_count());
        if needs_rebuild {
            self.rebuild_topology_cache()?;
        }
        Ok(())
    }

    fn rebuild_topology_cache(&mut self) -> Result<()> {
        self.topology_cache = Some(TopologyCache::from_graph(
            &self.graph,
            self.config
                .topology_update_strategy
                .maintains_reverse_coordinates(),
        )?);
        if self.config.collect_topology_counters {
            self.report.topology_counters.full_rebuilds += 1;
        }
        Ok(())
    }

    fn apply_topology_delta(&mut self, inserted_edges: &[EdgeKey]) -> Result<()> {
        if self.topology_cache.is_none() {
            self.rebuild_topology_cache()?;
        }
        if self.should_rebuild_topology_for_delta(inserted_edges) {
            if self.config.collect_topology_counters {
                self.report.topology_counters.inserted_edges += inserted_edges.len();
                self.report.topology_counters.full_rebuild_fallbacks += 1;
            }
            self.rebuild_topology_cache()?;
            return Ok(());
        }
        let cache = self
            .topology_cache
            .as_mut()
            .expect("topology cache is initialized");
        let counters = cache.apply_delta(
            &self.graph,
            inserted_edges,
            self.config.topology_update_strategy,
            self.config.collect_topology_counters,
            self.config.topology_rebuild_queue_threshold,
        )?;
        if self.config.collect_topology_counters {
            self.report.topology_counters.add_assign(counters);
        }
        Ok(())
    }

    fn should_rebuild_topology_for_delta(&self, inserted_edges: &[EdgeKey]) -> bool {
        if self.config.topology_update_strategy
            != TopologyUpdateStrategy::IncrementalForwardRelaxation
        {
            return false;
        }
        let Some(cache) = &self.topology_cache else {
            return false;
        };
        let previous_node_count = cache.node_count();
        let existing_child_edges = inserted_edges
            .iter()
            .filter(|edge| edge.child.to_usize() < previous_node_count)
            .count();
        existing_child_edges > 1_024
            && existing_child_edges.saturating_mul(4) > self.graph.node_count()
    }
}

impl TopologyUpdateStrategy {
    const fn is_incremental(self) -> bool {
        matches!(
            self,
            Self::IncrementalAffectedRegion
                | Self::IncrementalForwardOnly
                | Self::IncrementalForwardRescan
                | Self::IncrementalForwardRelaxation
        )
    }

    const fn maintains_reverse_coordinates(self) -> bool {
        matches!(self, Self::FullRebuild | Self::IncrementalAffectedRegion)
    }

    const fn uses_forward_relaxation(self) -> bool {
        matches!(self, Self::IncrementalForwardRelaxation)
    }
}

trait CoordinateProvider {
    fn forward_coordinate(&self, node: NodeId) -> Result<TopologicalCoordinate>;
    fn reverse_coordinate(&self, node: NodeId) -> Result<Option<TopologicalCoordinate>>;
}

impl CoordinateProvider for DagTopology {
    fn forward_coordinate(&self, node: NodeId) -> Result<TopologicalCoordinate> {
        DagTopology::forward_coordinate(self, node)
    }

    fn reverse_coordinate(&self, node: NodeId) -> Result<Option<TopologicalCoordinate>> {
        DagTopology::reverse_coordinate(self, node).map(Some)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TopologyCache {
    forward_coordinates: Vec<TopologicalCoordinate>,
    reverse_coordinates: Option<Vec<TopologicalCoordinate>>,
}

impl TopologyCache {
    fn from_graph(graph: &FtoDag, include_reverse: bool) -> Result<Self> {
        let topology = DagTopology::try_from_graph(graph)?;
        let mut forward_coordinates = Vec::with_capacity(graph.node_count());
        let mut reverse_coordinates =
            include_reverse.then(|| Vec::with_capacity(graph.node_count()));
        for index in 0..graph.node_count() {
            let node = NodeId::try_from(index)?;
            forward_coordinates.push(topology.forward_coordinate(node)?);
            if let Some(reverse_coordinates) = &mut reverse_coordinates {
                reverse_coordinates.push(topology.reverse_coordinate(node)?);
            }
        }
        Ok(Self {
            forward_coordinates,
            reverse_coordinates,
        })
    }

    fn node_count(&self) -> usize {
        self.forward_coordinates.len()
    }

    fn apply_delta(
        &mut self,
        graph: &FtoDag,
        inserted_edges: &[EdgeKey],
        strategy: TopologyUpdateStrategy,
        collect_counters: bool,
        rebuild_queue_threshold: Option<usize>,
    ) -> Result<TopologyUpdateCounters> {
        if self.node_count() > graph.node_count() {
            return Err(DagError::InvalidRange {
                start: graph.node_count(),
                end: self.node_count(),
                len: graph.node_count(),
            });
        }

        let previous_node_count = self.node_count();
        let mut counters = TopologyUpdateCounters::default();
        if collect_counters {
            counters.new_nodes = graph.node_count() - previous_node_count;
            counters.inserted_edges = inserted_edges.len();
            counters.inserted_edges_to_new_children = inserted_edges
                .iter()
                .filter(|edge| edge.child.to_usize() >= previous_node_count)
                .count();
            counters.inserted_edges_to_existing_children = inserted_edges
                .iter()
                .filter(|edge| edge.child.to_usize() < previous_node_count)
                .count();
        }
        for index in previous_node_count..graph.node_count() {
            let node = NodeId::try_from(index)?;
            let parents = graph.parents(node)?;
            if collect_counters {
                counters.forward_parent_scans += parents.len();
            }
            let forward = parents
                .iter()
                .map(|parent| {
                    self.forward_coordinates
                        .get(parent.to_usize())
                        .map(|coordinate| coordinate.raw() + 1)
                        .unwrap_or(1)
                })
                .max()
                .unwrap_or(1);
            self.forward_coordinates
                .push(TopologicalCoordinate::new(forward));
            if let Some(reverse_coordinates) = &mut self.reverse_coordinates {
                let children = graph.children(node)?;
                if collect_counters {
                    counters.reverse_child_scans += children.len();
                }
                let reverse = children
                    .iter()
                    .map(|child| {
                        reverse_coordinates
                            .get(child.to_usize())
                            .map(|coordinate| coordinate.raw() + 1)
                            .unwrap_or(1)
                    })
                    .max()
                    .unwrap_or(1);
                reverse_coordinates.push(TopologicalCoordinate::new(reverse));
            }
        }
        let propagation_completed = if strategy.uses_forward_relaxation() {
            self.propagate_forward_by_relaxation(
                graph,
                inserted_edges,
                previous_node_count,
                &mut counters,
                collect_counters,
                rebuild_queue_threshold,
            )?
        } else {
            self.propagate_forward_by_parent_scan(
                graph,
                inserted_edges,
                previous_node_count,
                &mut counters,
                collect_counters,
                rebuild_queue_threshold,
            )?
        };
        if !propagation_completed {
            *self = Self::from_graph(graph, strategy.maintains_reverse_coordinates())?;
            if collect_counters {
                counters.full_rebuilds += 1;
                counters.full_rebuild_fallbacks += 1;
            }
            return Ok(counters);
        }
        if strategy.maintains_reverse_coordinates() {
            let reverse_completed = self.propagate_reverse(
                graph,
                inserted_edges,
                &mut counters,
                collect_counters,
                rebuild_queue_threshold,
            )?;
            if !reverse_completed {
                *self = Self::from_graph(graph, true)?;
                if collect_counters {
                    counters.full_rebuilds += 1;
                    counters.full_rebuild_fallbacks += 1;
                }
            }
        }
        Ok(counters)
    }

    fn propagate_forward_by_parent_scan(
        &mut self,
        graph: &FtoDag,
        inserted_edges: &[EdgeKey],
        previous_node_count: usize,
        counters: &mut TopologyUpdateCounters,
        collect_counters: bool,
        rebuild_queue_threshold: Option<usize>,
    ) -> Result<bool> {
        let mut queue = VecDeque::new();
        let mut queue_pops = 0_usize;
        for edge in inserted_edges {
            if edge.child.to_usize() >= previous_node_count {
                continue;
            }
            if self.refresh_forward_node(graph, edge.child, counters, collect_counters)? {
                queue.push_back(edge.child);
            }
        }
        while let Some(node) = queue.pop_front() {
            queue_pops += 1;
            if rebuild_queue_threshold.is_some_and(|threshold| queue_pops > threshold) {
                return Ok(false);
            }
            if collect_counters {
                counters.forward_queue_pops += 1;
            }
            for child in graph.children(node)? {
                if self.refresh_forward_node(graph, *child, counters, collect_counters)? {
                    queue.push_back(*child);
                }
            }
        }
        Ok(true)
    }

    fn propagate_forward_by_relaxation(
        &mut self,
        graph: &FtoDag,
        inserted_edges: &[EdgeKey],
        previous_node_count: usize,
        counters: &mut TopologyUpdateCounters,
        collect_counters: bool,
        rebuild_queue_threshold: Option<usize>,
    ) -> Result<bool> {
        let mut queue = VecDeque::new();
        let mut queued = vec![false; graph.node_count()];
        let mut queue_pops = 0_usize;
        for edge in inserted_edges {
            if edge.child.to_usize() >= previous_node_count {
                continue;
            }
            self.relax_forward_edge(
                edge.parent,
                edge.child,
                counters,
                collect_counters,
                &mut queue,
                &mut queued,
            )?;
        }
        while let Some(node) = queue.pop_front() {
            queue_pops += 1;
            if rebuild_queue_threshold.is_some_and(|threshold| queue_pops > threshold) {
                return Ok(false);
            }
            queued[node.to_usize()] = false;
            if collect_counters {
                counters.forward_queue_pops += 1;
            }
            for child in graph.children(node)? {
                self.relax_forward_edge(
                    node,
                    *child,
                    counters,
                    collect_counters,
                    &mut queue,
                    &mut queued,
                )?;
            }
        }
        Ok(true)
    }

    fn propagate_reverse(
        &mut self,
        graph: &FtoDag,
        inserted_edges: &[EdgeKey],
        counters: &mut TopologyUpdateCounters,
        collect_counters: bool,
        rebuild_queue_threshold: Option<usize>,
    ) -> Result<bool> {
        if self.reverse_coordinates.is_none() {
            return Ok(true);
        }
        let mut queue = VecDeque::new();
        let mut queue_pops = 0_usize;
        for edge in inserted_edges {
            if self.refresh_reverse_node(graph, edge.parent, counters, collect_counters)? {
                queue.push_back(edge.parent);
            }
        }
        while let Some(node) = queue.pop_front() {
            queue_pops += 1;
            if rebuild_queue_threshold.is_some_and(|threshold| queue_pops > threshold) {
                return Ok(false);
            }
            if collect_counters {
                counters.reverse_queue_pops += 1;
            }
            for parent in graph.parents(node)? {
                if self.refresh_reverse_node(graph, *parent, counters, collect_counters)? {
                    queue.push_back(*parent);
                }
            }
        }
        Ok(true)
    }

    fn relax_forward_edge(
        &mut self,
        parent: NodeId,
        child: NodeId,
        counters: &mut TopologyUpdateCounters,
        collect_counters: bool,
        queue: &mut VecDeque<NodeId>,
        queued: &mut [bool],
    ) -> Result<bool> {
        if collect_counters {
            counters.forward_relax_attempts += 1;
        }
        let parent_coordinate = self
            .forward_coordinates
            .get(parent.to_usize())
            .copied()
            .ok_or(DagError::InvalidEdge {
                parent: parent.to_usize(),
                child: child.to_usize(),
            })?;
        let child_coordinate =
            self.forward_coordinates
                .get_mut(child.to_usize())
                .ok_or(DagError::InvalidEdge {
                    parent: parent.to_usize(),
                    child: child.to_usize(),
                })?;
        let candidate = parent_coordinate.raw() + 1;
        if candidate > child_coordinate.raw() {
            *child_coordinate = TopologicalCoordinate::new(candidate);
            if collect_counters {
                counters.forward_coordinate_updates += 1;
            }
            let child_index = child.to_usize();
            if !queued[child_index] {
                queued[child_index] = true;
                queue.push_back(child);
            }
            Ok(true)
        } else {
            if collect_counters {
                counters.safe_forward_edges += 1;
            }
            Ok(false)
        }
    }

    fn refresh_forward_node(
        &mut self,
        graph: &FtoDag,
        node: NodeId,
        counters: &mut TopologyUpdateCounters,
        collect_counters: bool,
    ) -> Result<bool> {
        let parents = graph.parents(node)?;
        if collect_counters {
            counters.forward_parent_scans += parents.len();
        }
        let coordinate = parents
            .iter()
            .map(|parent| self.forward_coordinates[parent.to_usize()].raw() + 1)
            .max()
            .unwrap_or(1);
        let current = &mut self.forward_coordinates[node.to_usize()];
        if coordinate > current.raw() {
            *current = TopologicalCoordinate::new(coordinate);
            if collect_counters {
                counters.forward_coordinate_updates += 1;
            }
            Ok(true)
        } else {
            if collect_counters {
                counters.safe_forward_edges += 1;
            }
            Ok(false)
        }
    }

    fn refresh_reverse_node(
        &mut self,
        graph: &FtoDag,
        node: NodeId,
        counters: &mut TopologyUpdateCounters,
        collect_counters: bool,
    ) -> Result<bool> {
        let Some(reverse_coordinates) = &mut self.reverse_coordinates else {
            return Ok(false);
        };
        let children = graph.children(node)?;
        if collect_counters {
            counters.reverse_child_scans += children.len();
        }
        let coordinate = children
            .iter()
            .map(|child| reverse_coordinates[child.to_usize()].raw() + 1)
            .max()
            .unwrap_or(1);
        let current = &mut reverse_coordinates[node.to_usize()];
        if coordinate > current.raw() {
            *current = TopologicalCoordinate::new(coordinate);
            if collect_counters {
                counters.reverse_coordinate_updates += 1;
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

impl CoordinateProvider for TopologyCache {
    fn forward_coordinate(&self, node: NodeId) -> Result<TopologicalCoordinate> {
        self.forward_coordinates
            .get(node.to_usize())
            .copied()
            .ok_or(DagError::MissingNode {
                node: node.to_usize(),
            })
    }

    fn reverse_coordinate(&self, node: NodeId) -> Result<Option<TopologicalCoordinate>> {
        self.reverse_coordinates
            .as_ref()
            .map(|coordinates| {
                coordinates
                    .get(node.to_usize())
                    .copied()
                    .ok_or(DagError::MissingNode {
                        node: node.to_usize(),
                    })
            })
            .transpose()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence_model::alphabet::SymbolId;

    fn key(raw: u16) -> FragmentKey {
        FragmentKey::symbols(vec![SymbolId::new(raw)])
    }

    fn add_edge(graph: &mut FtoDag, parent: NodeId, child: NodeId) -> EdgeKey {
        let update = graph
            .add_or_increment_edge(parent, child, Weight::new(1))
            .expect("edge insertion succeeds");
        assert!(update.inserted);
        update.key
    }

    fn assert_forward_cache_matches_full(graph: &FtoDag, cache: &TopologyCache) {
        let topology = DagTopology::try_from_graph(graph).expect("test graph is acyclic");
        for index in 0..graph.node_count() {
            let node = NodeId::try_from(index).unwrap();
            assert_eq!(
                cache.forward_coordinate(node).unwrap(),
                topology.forward_coordinate(node).unwrap(),
                "forward coordinate differs at node {index}"
            );
        }
    }

    #[test]
    fn forward_relaxation_updates_old_targets_and_descendants() {
        let mut graph = FtoDag::new(1);
        let start = graph.add_node(key(0), NodeKind::Start).unwrap();
        let old_target = graph.add_node(key(1), NodeKind::Internal).unwrap();
        let old_child = graph.add_node(key(2), NodeKind::End).unwrap();
        add_edge(&mut graph, start, old_target);
        add_edge(&mut graph, old_target, old_child);
        let mut cache = TopologyCache::from_graph(&graph, false).unwrap();

        let new_left = graph.add_node(key(3), NodeKind::Internal).unwrap();
        let new_right = graph.add_node(key(4), NodeKind::Internal).unwrap();
        let inserted_edges = vec![
            add_edge(&mut graph, start, new_left),
            add_edge(&mut graph, new_left, new_right),
            add_edge(&mut graph, new_right, old_target),
        ];

        let counters = cache
            .apply_delta(
                &graph,
                &inserted_edges,
                TopologyUpdateStrategy::IncrementalForwardRelaxation,
                true,
                None,
            )
            .unwrap();

        assert_eq!(counters.inserted_edges_to_existing_children, 1);
        assert!(counters.forward_relax_attempts > 0);
        assert!(counters.forward_coordinate_updates >= 2);
        assert_eq!(counters.forward_parent_scans, 2);
        assert_forward_cache_matches_full(&graph, &cache);
    }

    #[test]
    fn forward_rescan_matches_relaxation_coordinates() {
        let mut graph = FtoDag::new(1);
        let start = graph.add_node(key(0), NodeKind::Start).unwrap();
        let old_target = graph.add_node(key(1), NodeKind::Internal).unwrap();
        let old_child = graph.add_node(key(2), NodeKind::End).unwrap();
        add_edge(&mut graph, start, old_target);
        add_edge(&mut graph, old_target, old_child);
        let mut relaxation_cache = TopologyCache::from_graph(&graph, false).unwrap();
        let mut rescan_cache = relaxation_cache.clone();

        let new_left = graph.add_node(key(3), NodeKind::Internal).unwrap();
        let new_right = graph.add_node(key(4), NodeKind::Internal).unwrap();
        let inserted_edges = vec![
            add_edge(&mut graph, start, new_left),
            add_edge(&mut graph, new_left, new_right),
            add_edge(&mut graph, new_right, old_target),
        ];

        relaxation_cache
            .apply_delta(
                &graph,
                &inserted_edges,
                TopologyUpdateStrategy::IncrementalForwardRelaxation,
                true,
                None,
            )
            .unwrap();
        rescan_cache
            .apply_delta(
                &graph,
                &inserted_edges,
                TopologyUpdateStrategy::IncrementalForwardRescan,
                true,
                None,
            )
            .unwrap();

        assert_eq!(
            relaxation_cache.forward_coordinates,
            rescan_cache.forward_coordinates
        );
        assert_forward_cache_matches_full(&graph, &relaxation_cache);
    }

    #[test]
    fn broad_forward_delta_can_fallback_to_full_rebuild() {
        let mut graph = FtoDag::new(1);
        let start = graph.add_node(key(0), NodeKind::Start).unwrap();
        let old_target = graph.add_node(key(1), NodeKind::Internal).unwrap();
        let old_child = graph.add_node(key(2), NodeKind::End).unwrap();
        add_edge(&mut graph, start, old_target);
        add_edge(&mut graph, old_target, old_child);
        let mut cache = TopologyCache::from_graph(&graph, false).unwrap();

        let new_left = graph.add_node(key(3), NodeKind::Internal).unwrap();
        let new_right = graph.add_node(key(4), NodeKind::Internal).unwrap();
        let inserted_edges = vec![
            add_edge(&mut graph, start, new_left),
            add_edge(&mut graph, new_left, new_right),
            add_edge(&mut graph, new_right, old_target),
        ];

        let counters = cache
            .apply_delta(
                &graph,
                &inserted_edges,
                TopologyUpdateStrategy::IncrementalForwardRescan,
                true,
                Some(1),
            )
            .unwrap();

        assert_eq!(counters.full_rebuild_fallbacks, 1);
        assert_forward_cache_matches_full(&graph, &cache);
    }
}

pub const fn node_kind_for_path_position(kind: PathPositionKind) -> NodeKind {
    match kind {
        PathPositionKind::Start => NodeKind::Start,
        PathPositionKind::Internal => NodeKind::Internal,
        PathPositionKind::End => NodeKind::End,
        PathPositionKind::Singleton => NodeKind::Singleton,
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct AnchorDpCell {
    len: usize,
    continuity: Weight,
    weight: Weight,
    prev: Option<(usize, usize)>,
}

struct GreedySelectionState<'a> {
    previous: Option<NodeId>,
    last_coordinate: Option<TopologicalCoordinate>,
    used_nodes: &'a UsedNodeTracker,
}

pub fn select_monotone_anchor_path(candidate_sets: &[AnchorCandidateSet]) -> AnchorPath {
    select_monotone_anchor_path_with_edge_score(candidate_sets, |_| Weight::new(0), None)
}

pub fn select_monotone_anchor_path_with_graph(
    candidate_sets: &[AnchorCandidateSet],
    graph: &FtoDag,
) -> AnchorPath {
    if candidate_sets.len() > GREEDY_ANCHOR_SET_LIMIT {
        return select_greedy_monotone_anchor_path_with_graph(candidate_sets, graph);
    }
    select_monotone_anchor_path_with_edge_score(
        candidate_sets,
        |key| graph.edge_weight(key).unwrap_or_default(),
        Some(graph.node_count()),
    )
}

fn select_monotone_anchor_path_with_edge_score(
    candidate_sets: &[AnchorCandidateSet],
    edge_score: impl Fn(EdgeKey) -> Weight,
    node_count: Option<usize>,
) -> AnchorPath {
    if candidate_sets.len() > GREEDY_ANCHOR_SET_LIMIT {
        return select_greedy_monotone_anchor_path(candidate_sets, edge_score, node_count);
    }

    let mut dp: Vec<Vec<Option<AnchorDpCell>>> = candidate_sets
        .iter()
        .map(|set| vec![None; set.candidates.len()])
        .collect();

    let mut best: Option<(usize, usize, AnchorDpCell)> = None;
    for (set_index, set) in candidate_sets.iter().enumerate() {
        for (candidate_index, candidate) in set.candidates.iter().enumerate() {
            let Some(coordinate) = candidate.coordinate else {
                continue;
            };
            let mut cell = AnchorDpCell {
                len: 1,
                continuity: Weight::new(0),
                weight: candidate.weight,
                prev: None,
            };

            for prev_set_index in 0..set_index {
                for (prev_candidate_index, prev_candidate) in
                    candidate_sets[prev_set_index].candidates.iter().enumerate()
                {
                    let Some(prev_coordinate) = prev_candidate.coordinate else {
                        continue;
                    };
                    if prev_coordinate >= coordinate || prev_candidate.node == candidate.node {
                        continue;
                    }
                    let Some(prev_cell) = dp[prev_set_index][prev_candidate_index] else {
                        continue;
                    };
                    let next = AnchorDpCell {
                        len: prev_cell.len + 1,
                        continuity: Weight::new(
                            prev_cell.continuity.raw()
                                + edge_score(EdgeKey {
                                    parent: prev_candidate.node,
                                    child: candidate.node,
                                })
                                .raw(),
                        ),
                        weight: Weight::new(prev_cell.weight.raw() + candidate.weight.raw()),
                        prev: Some((prev_set_index, prev_candidate_index)),
                    };
                    if better_cell(next, cell, candidate.node, candidate.node) {
                        cell = next;
                    }
                }
            }
            dp[set_index][candidate_index] = Some(cell);
            match best {
                Some((best_set_index, best_candidate_index, best_cell)) => {
                    let best_node =
                        candidate_sets[best_set_index].candidates[best_candidate_index].node;
                    if better_cell(cell, best_cell, candidate.node, best_node) {
                        best = Some((set_index, candidate_index, cell));
                    }
                }
                None => best = Some((set_index, candidate_index, cell)),
            }
        }
    }

    let mut decisions = candidate_sets
        .iter()
        .map(|set| {
            if set.candidates.is_empty() {
                AnchorDecision::Unmatched(AnchorRejectReason::NoCandidate)
            } else if set
                .candidates
                .iter()
                .any(|candidate| candidate.coordinate.is_some())
            {
                AnchorDecision::Unmatched(AnchorRejectReason::NotSelected)
            } else {
                AnchorDecision::Unmatched(AnchorRejectReason::MissingTopology)
            }
        })
        .collect::<Vec<_>>();
    let mut selected_anchors = vec![None; candidate_sets.len()];

    if let Some((mut set_index, mut candidate_index, _)) = best {
        loop {
            let candidate = &candidate_sets[set_index].candidates[candidate_index];
            decisions[set_index] = AnchorDecision::Matched(candidate.node);
            if let Some(coordinate) = candidate.coordinate {
                selected_anchors[set_index] = Some(SelectedAnchor {
                    node: candidate.node,
                    coordinate,
                    weight: candidate.weight,
                });
            }
            let Some(cell) = dp[set_index][candidate_index] else {
                break;
            };
            if let Some((prev_set_index, prev_candidate_index)) = cell.prev {
                set_index = prev_set_index;
                candidate_index = prev_candidate_index;
            } else {
                break;
            }
        }
    }

    AnchorPath {
        decisions,
        selected_anchors,
    }
}

fn select_greedy_monotone_anchor_path(
    candidate_sets: &[AnchorCandidateSet],
    edge_score: impl Fn(EdgeKey) -> Weight,
    node_count: Option<usize>,
) -> AnchorPath {
    let mut decisions = Vec::with_capacity(candidate_sets.len());
    let mut selected_anchors = Vec::with_capacity(candidate_sets.len());
    let mut used_nodes = UsedNodeTracker::new(node_count);
    let mut previous: Option<NodeId> = None;
    let mut last_coordinate: Option<TopologicalCoordinate> = None;

    for set in candidate_sets {
        if let Some(candidate) =
            direct_extension_candidate(set, previous, last_coordinate, &used_nodes, &edge_score)
        {
            used_nodes.insert(candidate.node);
            previous = Some(candidate.node);
            last_coordinate = candidate.coordinate;
            decisions.push(AnchorDecision::Matched(candidate.node));
            selected_anchors.push(candidate.coordinate.map(|coordinate| SelectedAnchor {
                node: candidate.node,
                coordinate,
                weight: candidate.weight,
            }));
            continue;
        }

        let mut best: Option<(&AnchorCandidate, Weight)> = None;
        let mut has_coordinate = false;
        for candidate in &set.candidates {
            let Some(coordinate) = candidate.coordinate else {
                continue;
            };
            has_coordinate = true;
            if last_coordinate.is_some_and(|last| last >= coordinate)
                || used_nodes.contains(candidate.node)
            {
                continue;
            }
            let score = greedy_candidate_score(candidate, previous, &edge_score);
            if best.is_none_or(|(best_candidate, best_score)| {
                better_greedy_candidate(candidate, score, best_candidate, best_score)
            }) {
                best = Some((candidate, score));
            }
        }

        if let Some((candidate, _)) = best {
            used_nodes.insert(candidate.node);
            previous = Some(candidate.node);
            last_coordinate = candidate.coordinate;
            decisions.push(AnchorDecision::Matched(candidate.node));
            selected_anchors.push(candidate.coordinate.map(|coordinate| SelectedAnchor {
                node: candidate.node,
                coordinate,
                weight: candidate.weight,
            }));
        } else if set.candidates.is_empty() {
            decisions.push(AnchorDecision::Unmatched(AnchorRejectReason::NoCandidate));
            selected_anchors.push(None);
        } else if has_coordinate {
            decisions.push(AnchorDecision::Unmatched(AnchorRejectReason::NotSelected));
            selected_anchors.push(None);
        } else {
            decisions.push(AnchorDecision::Unmatched(
                AnchorRejectReason::MissingTopology,
            ));
            selected_anchors.push(None);
        }
    }

    AnchorPath {
        decisions,
        selected_anchors,
    }
}

fn select_greedy_monotone_anchor_path_with_graph(
    candidate_sets: &[AnchorCandidateSet],
    graph: &FtoDag,
) -> AnchorPath {
    let mut decisions = Vec::with_capacity(candidate_sets.len());
    let mut selected_anchors = Vec::with_capacity(candidate_sets.len());
    let mut used_nodes = UsedNodeTracker::new(Some(graph.node_count()));
    let mut previous: Option<NodeId> = None;
    let mut last_coordinate: Option<TopologicalCoordinate> = None;

    for set in candidate_sets {
        if let Some(candidate) =
            direct_child_extension_candidate(set, previous, last_coordinate, &used_nodes, graph)
        {
            used_nodes.insert(candidate.node);
            previous = Some(candidate.node);
            last_coordinate = candidate.coordinate;
            decisions.push(AnchorDecision::Matched(candidate.node));
            selected_anchors.push(candidate.coordinate.map(|coordinate| SelectedAnchor {
                node: candidate.node,
                coordinate,
                weight: candidate.weight,
            }));
            continue;
        }

        let mut best: Option<(&AnchorCandidate, Weight)> = None;
        let mut has_coordinate = false;
        for candidate in &set.candidates {
            let Some(coordinate) = candidate.coordinate else {
                continue;
            };
            has_coordinate = true;
            if last_coordinate.is_some_and(|last| last >= coordinate)
                || used_nodes.contains(candidate.node)
            {
                continue;
            }
            let score = greedy_candidate_score(candidate, previous, &|key| {
                graph.edge_weight(key).unwrap_or_default()
            });
            if best.is_none_or(|(best_candidate, best_score)| {
                better_greedy_candidate(candidate, score, best_candidate, best_score)
            }) {
                best = Some((candidate, score));
            }
        }

        if let Some((candidate, _)) = best {
            used_nodes.insert(candidate.node);
            previous = Some(candidate.node);
            last_coordinate = candidate.coordinate;
            decisions.push(AnchorDecision::Matched(candidate.node));
            selected_anchors.push(candidate.coordinate.map(|coordinate| SelectedAnchor {
                node: candidate.node,
                coordinate,
                weight: candidate.weight,
            }));
        } else if set.candidates.is_empty() {
            decisions.push(AnchorDecision::Unmatched(AnchorRejectReason::NoCandidate));
            selected_anchors.push(None);
        } else if has_coordinate {
            decisions.push(AnchorDecision::Unmatched(AnchorRejectReason::NotSelected));
            selected_anchors.push(None);
        } else {
            decisions.push(AnchorDecision::Unmatched(
                AnchorRejectReason::MissingTopology,
            ));
            selected_anchors.push(None);
        }
    }

    AnchorPath {
        decisions,
        selected_anchors,
    }
}

fn direct_child_extension_candidate<'a>(
    set: &'a AnchorCandidateSet,
    previous: Option<NodeId>,
    last_coordinate: Option<TopologicalCoordinate>,
    used_nodes: &UsedNodeTracker,
    graph: &FtoDag,
) -> Option<&'a AnchorCandidate> {
    let previous = previous?;
    let next_coordinate = TopologicalCoordinate::new(last_coordinate?.raw() + 1);
    let mut best: Option<(&AnchorCandidate, Weight)> = None;
    let children = graph.children(previous).ok()?;
    for child in children {
        if used_nodes.contains(*child) {
            continue;
        }
        let Some(candidate) = set.candidates.iter().find(|candidate| {
            candidate.node == *child && candidate.coordinate == Some(next_coordinate)
        }) else {
            continue;
        };
        let score = graph
            .edge_weight(EdgeKey {
                parent: previous,
                child: candidate.node,
            })
            .unwrap_or_default();
        if score == Weight::new(0) {
            continue;
        }
        if best.is_none_or(|(best_candidate, best_score)| {
            better_greedy_candidate(candidate, score, best_candidate, best_score)
        }) {
            best = Some((candidate, score));
        }
    }
    best.map(|(candidate, _)| candidate)
}

fn direct_extension_candidate<'a>(
    set: &'a AnchorCandidateSet,
    previous: Option<NodeId>,
    last_coordinate: Option<TopologicalCoordinate>,
    used_nodes: &UsedNodeTracker,
    edge_score: &impl Fn(EdgeKey) -> Weight,
) -> Option<&'a AnchorCandidate> {
    let previous = previous?;
    let next_coordinate = TopologicalCoordinate::new(last_coordinate?.raw() + 1);
    let mut best: Option<(&AnchorCandidate, Weight)> = None;
    for candidate in &set.candidates {
        if candidate.coordinate != Some(next_coordinate) || used_nodes.contains(candidate.node) {
            continue;
        }
        let score = edge_score(EdgeKey {
            parent: previous,
            child: candidate.node,
        });
        if score == Weight::new(0) {
            continue;
        }
        if best.is_none_or(|(best_candidate, best_score)| {
            better_greedy_candidate(candidate, score, best_candidate, best_score)
        }) {
            best = Some((candidate, score));
        }
    }
    best.map(|(candidate, _)| candidate)
}

fn greedy_candidate_score(
    candidate: &AnchorCandidate,
    previous: Option<NodeId>,
    edge_score: &impl Fn(EdgeKey) -> Weight,
) -> Weight {
    previous
        .map(|parent| {
            edge_score(EdgeKey {
                parent,
                child: candidate.node,
            })
        })
        .unwrap_or_default()
}

fn better_greedy_candidate(
    candidate: &AnchorCandidate,
    score: Weight,
    current: &AnchorCandidate,
    current_score: Weight,
) -> bool {
    score > current_score
        || (score == current_score && candidate.weight > current.weight)
        || (score == current_score
            && candidate.weight == current.weight
            && candidate.node < current.node)
}

enum UsedNodeTracker {
    Marks(Vec<bool>),
    Set(HashSet<NodeId>),
}

impl UsedNodeTracker {
    fn new(node_count: Option<usize>) -> Self {
        node_count
            .map(|node_count| Self::Marks(vec![false; node_count]))
            .unwrap_or_else(|| Self::Set(HashSet::new()))
    }

    fn contains(&self, node: NodeId) -> bool {
        match self {
            Self::Marks(marks) => marks.get(node.to_usize()).copied().unwrap_or(false),
            Self::Set(nodes) => nodes.contains(&node),
        }
    }

    fn insert(&mut self, node: NodeId) {
        match self {
            Self::Marks(marks) => {
                if let Some(mark) = marks.get_mut(node.to_usize()) {
                    *mark = true;
                }
            }
            Self::Set(nodes) => {
                nodes.insert(node);
            }
        }
    }
}

pub fn select_bounded_reuse_candidate<'a>(
    candidate_set: &'a AnchorCandidateSet,
    interval: CoordinateInterval,
    used_nodes: &[NodeId],
    graph: &FtoDag,
    previous_node: Option<NodeId>,
    next_node: Option<NodeId>,
) -> Option<&'a AnchorCandidate> {
    let mut best = None;
    for candidate in &candidate_set.candidates {
        if !candidate.coordinate.is_some_and(|coordinate| {
            interval.contains_open(coordinate)
                && !used_nodes.contains(&candidate.node)
                && next_node != Some(candidate.node)
        }) {
            continue;
        }
        let score = reuse_score(candidate, graph, previous_node, next_node);
        if best.is_none_or(|(best_candidate, best_score)| {
            better_greedy_candidate(candidate, score, best_candidate, best_score)
        }) {
            best = Some((candidate, score));
        }
    }
    best.map(|(candidate, _)| candidate)
}

fn select_bounded_reuse_candidate_with_marks<'a>(
    candidate_set: &'a AnchorCandidateSet,
    interval: CoordinateInterval,
    used_node_marks: &[bool],
    graph: &FtoDag,
    previous_node: Option<NodeId>,
    next_node: Option<NodeId>,
) -> Option<&'a AnchorCandidate> {
    let mut best = None;
    for candidate in &candidate_set.candidates {
        if !candidate.coordinate.is_some_and(|coordinate| {
            interval.contains_open(coordinate)
                && !used_node_marks
                    .get(candidate.node.to_usize())
                    .copied()
                    .unwrap_or(false)
                && next_node != Some(candidate.node)
        }) {
            continue;
        }
        let score = reuse_score(candidate, graph, previous_node, next_node);
        if best.is_none_or(|(best_candidate, best_score)| {
            better_greedy_candidate(candidate, score, best_candidate, best_score)
        }) {
            best = Some((candidate, score));
        }
    }
    best.map(|(candidate, _)| candidate)
}

fn mark_used_node(used_node_marks: &mut [bool], node: NodeId) {
    if let Some(mark) = used_node_marks.get_mut(node.to_usize()) {
        *mark = true;
    }
}

pub fn block_plan_from_anchor_path(
    anchor_path: &AnchorPath,
    candidate_sets: &[AnchorCandidateSet],
) -> BlockPlan {
    let mut accepted_anchors = Vec::new();
    let mut current_block: Option<AnchorBlock> = None;
    let mut last_anchor: Option<(usize, TopologicalCoordinate)> = None;
    let mut unanchored_blocks = Vec::new();
    let mut unmatched_start: Option<usize> = None;

    for (index, decision) in anchor_path.decisions.iter().enumerate() {
        match decision {
            AnchorDecision::Matched(node_id) => {
                if let Some(start) = unmatched_start.take() {
                    unanchored_blocks.push(UnanchoredBlock {
                        path_range: PathRange::new(start, index),
                        coordinate_interval: CoordinateInterval {
                            left: last_anchor.map(|(_, coordinate)| coordinate),
                            right: selected_coordinate(
                                anchor_path,
                                candidate_sets,
                                index,
                                *node_id,
                            ),
                        },
                    });
                }

                let Some(selected) = selected_anchor(anchor_path, candidate_sets, index, *node_id)
                else {
                    continue;
                };
                let coordinate = selected.coordinate;
                let extends_current = last_anchor.is_some_and(|(last_index, last_coordinate)| {
                    last_index + 1 == index && last_coordinate.raw() + 1 == coordinate.raw()
                });
                if extends_current {
                    if let Some(block) = &mut current_block {
                        block.path_range.end = index + 1;
                        block.coordinate_end = coordinate;
                        block.score = Weight::new(block.score.raw() + selected.weight.raw());
                        block.anchors.push(*node_id);
                    }
                } else {
                    if let Some(block) = current_block.take() {
                        accepted_anchors.push(block);
                    }
                    current_block = Some(AnchorBlock {
                        path_range: PathRange::new(index, index + 1),
                        coordinate_start: coordinate,
                        coordinate_end: coordinate,
                        score: selected.weight,
                        anchors: vec![*node_id],
                    });
                }
                last_anchor = Some((index, coordinate));
            }
            AnchorDecision::Unmatched(_) => {
                if unmatched_start.is_none() {
                    unmatched_start = Some(index);
                }
            }
        }
    }

    if let Some(block) = current_block {
        accepted_anchors.push(block);
    }
    if let Some(start) = unmatched_start {
        unanchored_blocks.push(UnanchoredBlock {
            path_range: PathRange::new(start, anchor_path.decisions.len()),
            coordinate_interval: CoordinateInterval {
                left: last_anchor.map(|(_, coordinate)| coordinate),
                right: None,
            },
        });
    }

    BlockPlan {
        accepted_anchors,
        rejected_anchors: Vec::new(),
        unanchored_blocks,
        reused_nodes: Vec::new(),
        new_nodes: Vec::new(),
    }
}

fn next_matched_anchors_after(
    anchor_path: &AnchorPath,
    candidate_sets: &[AnchorCandidateSet],
) -> Vec<Option<(NodeId, TopologicalCoordinate)>> {
    let mut next = vec![None; anchor_path.decisions.len()];
    let mut current = None;
    for index in (0..anchor_path.decisions.len()).rev() {
        next[index] = current;
        if let AnchorDecision::Matched(node_id) = anchor_path.decisions[index] {
            current = selected_coordinate(anchor_path, candidate_sets, index, node_id)
                .map(|coordinate| (node_id, coordinate));
        }
    }
    next
}

fn reuse_score(
    candidate: &AnchorCandidate,
    graph: &FtoDag,
    previous_node: Option<NodeId>,
    next_node: Option<NodeId>,
) -> Weight {
    let previous_score = previous_node
        .and_then(|previous| {
            graph.edge_weight(EdgeKey {
                parent: previous,
                child: candidate.node,
            })
        })
        .unwrap_or_default();
    let next_score = next_node
        .and_then(|next| {
            graph.edge_weight(EdgeKey {
                parent: candidate.node,
                child: next,
            })
        })
        .unwrap_or_default();
    Weight::new(previous_score.raw() + next_score.raw())
}

fn matched_candidate(
    candidate_sets: &[AnchorCandidateSet],
    index: usize,
    node_id: NodeId,
) -> Option<&AnchorCandidate> {
    candidate_sets
        .get(index)?
        .candidates
        .iter()
        .find(|candidate| candidate.node == node_id)
}

fn selected_anchor(
    anchor_path: &AnchorPath,
    candidate_sets: &[AnchorCandidateSet],
    index: usize,
    node_id: NodeId,
) -> Option<SelectedAnchor> {
    anchor_path
        .selected_anchors
        .get(index)
        .copied()
        .flatten()
        .filter(|selected| selected.node == node_id)
        .or_else(|| {
            matched_candidate(candidate_sets, index, node_id).and_then(|candidate| {
                candidate.coordinate.map(|coordinate| SelectedAnchor {
                    node: candidate.node,
                    coordinate,
                    weight: candidate.weight,
                })
            })
        })
}

fn selected_coordinate(
    anchor_path: &AnchorPath,
    candidate_sets: &[AnchorCandidateSet],
    index: usize,
    node_id: NodeId,
) -> Option<TopologicalCoordinate> {
    selected_anchor(anchor_path, candidate_sets, index, node_id).map(|selected| selected.coordinate)
}

fn better_cell(
    candidate: AnchorDpCell,
    current: AnchorDpCell,
    candidate_node: NodeId,
    current_node: NodeId,
) -> bool {
    candidate.len > current.len
        || (candidate.len == current.len && candidate.continuity > current.continuity)
        || (candidate.len == current.len
            && candidate.continuity == current.continuity
            && candidate.weight > current.weight)
        || (candidate.len == current.len
            && candidate.continuity == current.continuity
            && candidate.weight == current.weight
            && candidate_node < current_node)
}
