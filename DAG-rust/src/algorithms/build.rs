//! Incremental sequence-to-FTO-DAG construction interfaces.

use crate::algorithms::postprocess::{SecondaryMergeConfig, secondary_merge_graph};
use crate::foundations::error::DagError;
use crate::foundations::error::Result;
use crate::foundations::id::{
    GraphId, NodeId, ProvenancePosition, SequenceId, TopologicalCoordinate, Weight,
};
use crate::graph_model::graph::{EdgeIndexStrategy, EdgeKey, FtoDag, NodeKind};
use crate::graph_model::provenance::{ProvenanceRecord, ProvenanceStorageStrategy};
use crate::graph_model::topology::DagTopology;
use crate::sequence_model::alphabet::Alphabet;
use crate::sequence_model::fragment::{FragmentEncoder, FragmentOccurrence, PathPositionKind};
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
    pub candidates: Vec<AnchorCandidate>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SelectedAnchor {
    pub node: NodeId,
    pub coordinate: TopologicalCoordinate,
    pub weight: Weight,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct MatchedAnchorEntry {
    index: usize,
    selected: SelectedAnchor,
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
    matched_anchors: Vec<MatchedAnchorEntry>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct SparseMatchedPath {
    matched_anchors: Vec<MatchedAnchorEntry>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct MutationCandidate {
    node: NodeId,
    coordinate: TopologicalCoordinate,
    weight: Weight,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct MutationCandidateStorage {
    offsets: Vec<usize>,
    candidates: Vec<MutationCandidate>,
}

impl MutationCandidateStorage {
    fn with_position_capacity(position_count: usize) -> Self {
        let mut offsets = Vec::with_capacity(position_count + 1);
        offsets.push(0);
        Self {
            offsets,
            candidates: Vec::new(),
        }
    }

    fn push_position<I>(&mut self, candidates: I)
    where
        I: IntoIterator<Item = MutationCandidate>,
    {
        self.candidates.extend(candidates);
        self.offsets.push(self.candidates.len());
    }

    fn candidates_for(&self, position: usize) -> &[MutationCandidate] {
        let start = self.offsets[position];
        let end = self.offsets[position + 1];
        &self.candidates[start..end]
    }
}

pub(crate) trait CandidateLike {
    fn node(&self) -> NodeId;
    fn coordinate(&self) -> Option<TopologicalCoordinate>;
    fn weight(&self) -> Weight;
}

pub(crate) trait CandidateSetLike<C: CandidateLike> {
    fn candidates(&self) -> &[C];
}

impl CandidateLike for AnchorCandidate {
    fn node(&self) -> NodeId {
        self.node
    }

    fn coordinate(&self) -> Option<TopologicalCoordinate> {
        self.coordinate
    }

    fn weight(&self) -> Weight {
        self.weight
    }
}

impl CandidateSetLike<AnchorCandidate> for AnchorCandidateSet {
    fn candidates(&self) -> &[AnchorCandidate] {
        &self.candidates
    }
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

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct BuildSequenceSummary {
    pub sequence_id: SequenceId,
    pub inserted_edges: usize,
    pub provenance_records_added: usize,
    pub new_nodes_created: usize,
    pub reused_nodes: usize,
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SequenceBuildSummaryOutcome {
    Integrated(BuildSequenceSummary),
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfiledSequenceBuildSummaryOutcome {
    pub outcome: SequenceBuildSummaryOutcome,
    pub timing: BuildTimingBreakdown,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum BuildDetailMode {
    Diagnostics,
    Summary,
}

impl BuildDetailMode {
    const fn retains_node_path(self) -> bool {
        matches!(self, Self::Diagnostics)
    }

    const fn retains_block_plan(self) -> bool {
        matches!(self, Self::Diagnostics)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct IntegratedSequenceBuildData {
    sequence_id: SequenceId,
    node_path: Option<Vec<NodeId>>,
    inserted_edges: usize,
    provenance_records_added: usize,
    block_plan: Option<BlockPlan>,
    new_nodes_created: usize,
    reused_nodes: usize,
}

impl IntegratedSequenceBuildData {
    fn into_result(self) -> BuildSequenceResult {
        BuildSequenceResult {
            sequence_id: self.sequence_id,
            node_path: self
                .node_path
                .expect("diagnostic build data retains node path"),
            inserted_edges: self.inserted_edges,
            provenance_records_added: self.provenance_records_added,
            block_plan: self
                .block_plan
                .expect("diagnostic build data retains block plan"),
        }
    }

    fn into_summary(self) -> BuildSequenceSummary {
        BuildSequenceSummary {
            sequence_id: self.sequence_id,
            inserted_edges: self.inserted_edges,
            provenance_records_added: self.provenance_records_added,
            new_nodes_created: self.new_nodes_created,
            reused_nodes: self.reused_nodes,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum SequenceBuildDataOutcome {
    Integrated(IntegratedSequenceBuildData),
    Rejected(RejectedSequence),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProfiledSequenceBuildDataOutcome {
    outcome: SequenceBuildDataOutcome,
    timing: BuildTimingBreakdown,
}

#[derive(Clone, Debug)]
pub struct FtoDagBuilder {
    config: BuildConfig,
    graph: FtoDag,
    report: BuildReport,
    topology_cache: Option<TopologyCache>,
    existing_node_marks: ExistingNodeMarks,
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
            existing_node_marks: ExistingNodeMarks::default(),
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
            existing_node_marks: ExistingNodeMarks::default(),
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

    pub fn finalize_graph(self) -> Result<FtoDag> {
        secondary_merge_graph(self.graph, SecondaryMergeConfig::default())
    }

    pub fn initialize_from_encoded<E: FragmentEncoder>(
        &mut self,
        sequence_id: SequenceId,
        sequence: &EncodedSequence,
        encoder: &E,
    ) -> Result<BuildSequenceResult> {
        let occurrences = encoder.encode_occurrences(sequence, self.config.fragment_len)?;
        Ok(self
            .initialize_from_occurrences_with_detail(
                sequence_id,
                occurrences,
                BuildDetailMode::Diagnostics,
            )?
            .into_result())
    }

    fn initialize_from_occurrences_with_detail(
        &mut self,
        sequence_id: SequenceId,
        occurrences: Vec<FragmentOccurrence>,
        detail_mode: BuildDetailMode,
    ) -> Result<IntegratedSequenceBuildData> {
        let mut previous = None;
        let mut node_path = detail_mode.retains_node_path().then(Vec::new);
        let mut inserted_edges = 0;
        let new_nodes_created = occurrences.len();
        for occurrence in occurrences {
            let node_id = self.add_occurrence_node(sequence_id, occurrence)?;
            if let Some(previous) = previous {
                let update = self
                    .graph
                    .add_or_increment_edge(previous, node_id, Weight::new(1))?;
                inserted_edges += usize::from(update.inserted);
            }
            previous = Some(node_id);
            if let Some(path) = node_path.as_mut() {
                path.push(node_id);
            }
        }
        let block_plan = if detail_mode.retains_block_plan() {
            Some(BlockPlan {
                new_nodes: node_path
                    .clone()
                    .expect("diagnostic build data retains node path"),
                ..BlockPlan::default()
            })
        } else {
            None
        };
        Ok(IntegratedSequenceBuildData {
            sequence_id,
            node_path,
            inserted_edges,
            provenance_records_added: new_nodes_created,
            block_plan,
            new_nodes_created,
            reused_nodes: 0,
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

    pub fn add_sequence_from_encoded_summary<E: FragmentEncoder>(
        &mut self,
        sequence_id: SequenceId,
        sequence: &EncodedSequence,
        encoder: &E,
    ) -> Result<SequenceBuildSummaryOutcome> {
        Ok(self
            .add_sequence_from_encoded_profiled_summary(sequence_id, sequence, encoder)?
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

    pub fn add_sequence_from_occurrences_summary(
        &mut self,
        sequence_id: SequenceId,
        occurrences: Vec<FragmentOccurrence>,
    ) -> Result<SequenceBuildSummaryOutcome> {
        Ok(self
            .add_sequence_from_occurrences_profiled_summary(sequence_id, occurrences)?
            .outcome)
    }

    pub(crate) fn add_exact_trace_path_summary(
        &mut self,
        sequence_id: SequenceId,
        source_graph: &FtoDag,
        trace_path: &[NodeId],
    ) -> Result<Option<Vec<NodeId>>> {
        let mapped_path = if let Some(mapped_path) =
            self.resolve_unique_exact_trace_path(sequence_id, source_graph, trace_path)?
        {
            mapped_path
        } else if let Some(mapped_path) =
            self.resolve_greedy_exact_trace_path(sequence_id, source_graph, trace_path)?
        {
            mapped_path
        } else {
            return Ok(None);
        };

        self.integrate_mapped_trace_path_summary(sequence_id, &mapped_path)?;
        Ok(Some(mapped_path))
    }

    pub(crate) fn integrate_mapped_trace_path_summary(
        &mut self,
        sequence_id: SequenceId,
        mapped_path: &[NodeId],
    ) -> Result<()> {
        self.report.attempted_sequences += 1;
        let mut previous = None;
        let mut inserted_edges = 0;
        for node_id in mapped_path.iter().copied() {
            if !self.can_node_accept_sequence(node_id, sequence_id) {
                return Err(DagError::DuplicateSequenceProvenance {
                    node: node_id.to_usize(),
                    sequence: sequence_id.raw(),
                });
            }
            if let Some(previous) = previous {
                let update = self
                    .graph
                    .add_or_increment_edge(previous, node_id, Weight::new(1))?;
                inserted_edges += usize::from(update.inserted);
            }
            previous = Some(node_id);
        }
        self.graph
            .add_trace_path_provenance(sequence_id, mapped_path)?;

        let result = IntegratedSequenceBuildData {
            sequence_id,
            node_path: None,
            inserted_edges,
            provenance_records_added: mapped_path.len(),
            block_plan: None,
            new_nodes_created: 0,
            reused_nodes: mapped_path.len(),
        };
        self.record_integrated_sequence_data(&result);
        Ok(())
    }

    pub fn add_sequence_from_encoded_profiled<E: FragmentEncoder>(
        &mut self,
        sequence_id: SequenceId,
        sequence: &EncodedSequence,
        encoder: &E,
    ) -> Result<ProfiledSequenceBuildOutcome> {
        let profiled = self.add_sequence_from_encoded_profiled_with_detail(
            sequence_id,
            sequence,
            encoder,
            BuildDetailMode::Diagnostics,
        )?;
        Ok(ProfiledSequenceBuildOutcome {
            outcome: match profiled.outcome {
                SequenceBuildDataOutcome::Integrated(result) => {
                    SequenceBuildOutcome::Integrated(result.into_result())
                }
                SequenceBuildDataOutcome::Rejected(rejected) => {
                    SequenceBuildOutcome::Rejected(rejected)
                }
            },
            timing: profiled.timing,
        })
    }

    pub fn add_sequence_from_encoded_profiled_summary<E: FragmentEncoder>(
        &mut self,
        sequence_id: SequenceId,
        sequence: &EncodedSequence,
        encoder: &E,
    ) -> Result<ProfiledSequenceBuildSummaryOutcome> {
        let profiled = self.add_sequence_from_encoded_profiled_with_detail(
            sequence_id,
            sequence,
            encoder,
            BuildDetailMode::Summary,
        )?;
        Ok(ProfiledSequenceBuildSummaryOutcome {
            outcome: match profiled.outcome {
                SequenceBuildDataOutcome::Integrated(result) => {
                    SequenceBuildSummaryOutcome::Integrated(result.into_summary())
                }
                SequenceBuildDataOutcome::Rejected(rejected) => {
                    SequenceBuildSummaryOutcome::Rejected(rejected)
                }
            },
            timing: profiled.timing,
        })
    }

    fn add_sequence_from_encoded_profiled_with_detail<E: FragmentEncoder>(
        &mut self,
        sequence_id: SequenceId,
        sequence: &EncodedSequence,
        encoder: &E,
        detail_mode: BuildDetailMode,
    ) -> Result<ProfiledSequenceBuildDataOutcome> {
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
            detail_mode,
        )
    }

    pub fn add_sequence_from_occurrences_profiled(
        &mut self,
        sequence_id: SequenceId,
        occurrences: Vec<FragmentOccurrence>,
    ) -> Result<ProfiledSequenceBuildOutcome> {
        let profiled = self.add_sequence_from_occurrences_profiled_with_timing(
            sequence_id,
            occurrences,
            Instant::now(),
            BuildTimingBreakdown::default(),
            BuildDetailMode::Diagnostics,
        )?;
        Ok(ProfiledSequenceBuildOutcome {
            outcome: match profiled.outcome {
                SequenceBuildDataOutcome::Integrated(result) => {
                    SequenceBuildOutcome::Integrated(result.into_result())
                }
                SequenceBuildDataOutcome::Rejected(rejected) => {
                    SequenceBuildOutcome::Rejected(rejected)
                }
            },
            timing: profiled.timing,
        })
    }

    pub fn add_sequence_from_occurrences_profiled_summary(
        &mut self,
        sequence_id: SequenceId,
        occurrences: Vec<FragmentOccurrence>,
    ) -> Result<ProfiledSequenceBuildSummaryOutcome> {
        let profiled = self.add_sequence_from_occurrences_profiled_with_timing(
            sequence_id,
            occurrences,
            Instant::now(),
            BuildTimingBreakdown::default(),
            BuildDetailMode::Summary,
        )?;
        Ok(ProfiledSequenceBuildSummaryOutcome {
            outcome: match profiled.outcome {
                SequenceBuildDataOutcome::Integrated(result) => {
                    SequenceBuildSummaryOutcome::Integrated(result.into_summary())
                }
                SequenceBuildDataOutcome::Rejected(rejected) => {
                    SequenceBuildSummaryOutcome::Rejected(rejected)
                }
            },
            timing: profiled.timing,
        })
    }

    fn add_sequence_from_occurrences_profiled_with_timing(
        &mut self,
        sequence_id: SequenceId,
        occurrences: Vec<FragmentOccurrence>,
        total_start: Instant,
        mut timing: BuildTimingBreakdown,
        detail_mode: BuildDetailMode,
    ) -> Result<ProfiledSequenceBuildDataOutcome> {
        self.report.attempted_sequences += 1;
        if self.graph.node_count() == 0 {
            let start = Instant::now();
            let result = self.initialize_from_occurrences_with_detail(
                sequence_id,
                occurrences,
                detail_mode,
            )?;
            timing.initialize += start.elapsed();
            if self.config.topology_update_strategy.is_incremental() {
                let start = Instant::now();
                self.rebuild_topology_cache()?;
                timing.topology += start.elapsed();
            }
            let start = Instant::now();
            self.record_integrated_sequence_data(&result);
            timing.report_update += start.elapsed();
            timing.total = total_start.elapsed();
            return Ok(ProfiledSequenceBuildDataOutcome {
                outcome: SequenceBuildDataOutcome::Integrated(result),
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
            return Ok(ProfiledSequenceBuildDataOutcome {
                outcome: SequenceBuildDataOutcome::Rejected(rejected),
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
        if topology.is_none()
            && occurrences.len() > GREEDY_ANCHOR_SET_LIMIT
            && !detail_mode.retains_node_path()
            && !detail_mode.retains_block_plan()
        {
            let cache = self
                .topology_cache
                .as_ref()
                .expect("incremental topology cache is initialized");
            let start = Instant::now();
            let (candidate_sets, sparse_path) =
                self.collect_greedy_mutation_plan_with_cache(sequence_id, &occurrences, cache)?;
            timing.candidate_collection += start.elapsed();

            let mut matched_anchor_cursor = 0usize;
            let mut previous = None;
            let mut last_existing_coordinate = None;
            self.existing_node_marks
                .begin_sequence(self.graph.node_count());
            let mut inserted_edges = 0;
            let mut inserted_edge_keys = Vec::new();
            let mut provenance_records_added = 0;
            let mut new_nodes_created = 0;
            let mut reused_nodes = 0;

            let start = Instant::now();
            for (index, occurrence) in occurrences.into_iter().enumerate() {
                let candidate_set = candidate_sets.candidates_for(index);
                while matched_anchor_cursor < sparse_path.matched_anchors.len()
                    && sparse_path.matched_anchors[matched_anchor_cursor].index < index
                {
                    matched_anchor_cursor += 1;
                }
                let current_selected = sparse_path
                    .matched_anchors
                    .get(matched_anchor_cursor)
                    .filter(|entry| entry.index == index)
                    .map(|entry| entry.selected);
                let next_anchor = sparse_path
                    .matched_anchors
                    .get(matched_anchor_cursor + usize::from(current_selected.is_some()))
                    .map(|entry| (entry.selected.node, entry.selected.coordinate));
                let node_id = if let Some(selected) = current_selected {
                    self.add_source_to_existing_node(
                        selected.node,
                        sequence_id,
                        occurrence.position,
                    )?;
                    self.existing_node_marks.mark(selected.node);
                    last_existing_coordinate = Some(selected.coordinate);
                    selected.node
                } else {
                    let interval = CoordinateInterval {
                        left: last_existing_coordinate,
                        right: next_anchor.map(|(_, coordinate)| coordinate),
                    };
                    if let Some(reuse) = select_bounded_reuse_candidate_with_marks(
                        candidate_set,
                        interval,
                        &self.existing_node_marks,
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
                        self.existing_node_marks.mark(node_id);
                        last_existing_coordinate = Some(reuse.coordinate);
                        reused_nodes += 1;
                        node_id
                    } else {
                        let node_id = self.add_occurrence_node(sequence_id, occurrence)?;
                        new_nodes_created += 1;
                        node_id
                    }
                };
                provenance_records_added += 1;
                if let Some(previous) = previous {
                    let update =
                        self.graph
                            .add_or_increment_edge(previous, node_id, Weight::new(1))?;
                    inserted_edges += usize::from(update.inserted);
                    if update.inserted {
                        inserted_edge_keys.push(update.key);
                    }
                }
                previous = Some(node_id);
            }
            timing.mutation += start.elapsed();

            if self.config.topology_update_strategy.is_incremental() {
                let start = Instant::now();
                self.apply_topology_delta(&inserted_edge_keys)?;
                timing.topology += start.elapsed();
            }

            let result = IntegratedSequenceBuildData {
                sequence_id,
                node_path: None,
                inserted_edges,
                provenance_records_added,
                block_plan: None,
                new_nodes_created,
                reused_nodes,
            };
            let start = Instant::now();
            self.record_integrated_sequence_data(&result);
            timing.report_update += start.elapsed();
            timing.total = total_start.elapsed();
            return Ok(ProfiledSequenceBuildDataOutcome {
                outcome: SequenceBuildDataOutcome::Integrated(result),
                timing,
            });
        }
        let (candidate_sets, compact_mutation_sets, anchor_path) = if topology.is_none()
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
            (None, Some(selected.0), selected.1)
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
            (Some(candidate_sets), None, anchor_path)
        };
        let mut block_plan = if detail_mode.retains_block_plan() {
            let start = Instant::now();
            let block_plan = block_plan_from_anchor_path(&anchor_path);
            timing.block_planning += start.elapsed();
            Some(block_plan)
        } else {
            None
        };
        let candidate_sets = if let Some(candidate_sets) = candidate_sets {
            compact_candidate_sets_for_mutation(candidate_sets, &anchor_path)
        } else {
            compact_mutation_sets.expect("greedy candidate path provides compact mutation sets")
        };
        let AnchorPath {
            decisions,
            matched_anchors,
        } = anchor_path;
        let mut matched_anchor_cursor = 0usize;
        let mut node_path = detail_mode
            .retains_node_path()
            .then(|| Vec::with_capacity(occurrences.len()));
        let mut previous = None;
        let mut last_existing_coordinate = None;
        self.existing_node_marks
            .begin_sequence(self.graph.node_count());
        let mut inserted_edges = 0;
        let mut inserted_edge_keys = Vec::new();
        let mut provenance_records_added = 0;
        let mut new_nodes_created = 0;
        let mut reused_nodes = 0;

        let start = Instant::now();
        for (index, (occurrence, decision)) in
            occurrences.into_iter().zip(decisions.iter()).enumerate()
        {
            let candidate_set = candidate_sets.candidates_for(index);
            while matched_anchor_cursor < matched_anchors.len()
                && matched_anchors[matched_anchor_cursor].index < index
            {
                matched_anchor_cursor += 1;
            }
            let current_selected = matched_anchors
                .get(matched_anchor_cursor)
                .filter(|entry| entry.index == index)
                .map(|entry| entry.selected);
            let next_anchor = matched_anchors
                .get(matched_anchor_cursor + usize::from(current_selected.is_some()))
                .map(|entry| (entry.selected.node, entry.selected.coordinate));
            let node_id = match decision {
                AnchorDecision::Matched(node_id) => {
                    self.add_source_to_existing_node(*node_id, sequence_id, occurrence.position)?;
                    self.existing_node_marks.mark(*node_id);
                    last_existing_coordinate = current_selected
                        .map(|selected| selected.coordinate)
                        .or(last_existing_coordinate);
                    *node_id
                }
                AnchorDecision::Unmatched(_) => {
                    let interval = CoordinateInterval {
                        left: last_existing_coordinate,
                        right: next_anchor.map(|(_, coordinate)| coordinate),
                    };
                    if let Some(reuse) = select_bounded_reuse_candidate_with_marks(
                        candidate_set,
                        interval,
                        &self.existing_node_marks,
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
                        self.existing_node_marks.mark(node_id);
                        last_existing_coordinate = Some(reuse.coordinate);
                        reused_nodes += 1;
                        if let Some(plan) = block_plan.as_mut() {
                            plan.reused_nodes.push(node_id);
                        }
                        node_id
                    } else {
                        let node_id = self.add_occurrence_node(sequence_id, occurrence)?;
                        new_nodes_created += 1;
                        if let Some(plan) = block_plan.as_mut() {
                            plan.new_nodes.push(node_id);
                        }
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
            if let Some(path) = node_path.as_mut() {
                path.push(node_id);
            }
        }
        timing.mutation += start.elapsed();

        if self.config.topology_update_strategy.is_incremental() {
            let start = Instant::now();
            self.apply_topology_delta(&inserted_edge_keys)?;
            timing.topology += start.elapsed();
        }

        let result = IntegratedSequenceBuildData {
            sequence_id,
            node_path,
            inserted_edges,
            provenance_records_added,
            block_plan,
            new_nodes_created,
            reused_nodes,
        };
        let start = Instant::now();
        self.record_integrated_sequence_data(&result);
        timing.report_update += start.elapsed();
        timing.total = total_start.elapsed();
        Ok(ProfiledSequenceBuildDataOutcome {
            outcome: SequenceBuildDataOutcome::Integrated(result),
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
                .nodes_for_stored(&occurrence.key, node_kind)
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
            let nodes = self
                .graph
                .fragment_index()
                .nodes_for_stored(&occurrence.key, kind);
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
            let nodes = self
                .graph
                .fragment_index()
                .nodes_for_stored(&occurrence.key, kind);
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
    ) -> Result<(MutationCandidateStorage, AnchorPath)> {
        let mut candidate_sets =
            MutationCandidateStorage::with_position_capacity(occurrences.len());
        let mut decisions = Vec::with_capacity(occurrences.len());
        let mut matched_anchors = Vec::new();
        let mut used_nodes =
            UsedNodeTracker::new_for_path(Some(self.graph.node_count()), occurrences.len());
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
                if let Some(coordinate) = candidate.coordinate {
                    matched_anchors.push(MatchedAnchorEntry {
                        index: occurrence.position,
                        selected: SelectedAnchor {
                            node: candidate.node,
                            coordinate,
                            weight: candidate.weight,
                        },
                    });
                }
                candidate_sets.push_position(std::iter::empty::<MutationCandidate>());
                continue;
            }

            let mut best: Option<(MutationCandidate, Weight)> = None;
            let mut has_candidate = false;
            let mut has_coordinate = false;
            let nodes = self
                .graph
                .fragment_index()
                .nodes_for_stored(&occurrence.key, kind);
            let mut mutation_candidates = Vec::with_capacity(nodes.len());
            for node_id in nodes.iter().copied() {
                if !self.can_node_accept_sequence(node_id, sequence_id) {
                    continue;
                }
                let candidate = self.anchor_candidate_from_cache(node_id, kind, cache)?;
                has_candidate = true;
                let Some(coordinate) = candidate.coordinate else {
                    continue;
                };
                has_coordinate = true;
                if last_coordinate.is_some_and(|last| last >= coordinate)
                    || used_nodes.contains(candidate.node)
                {
                    continue;
                }
                let score = greedy_candidate_score(&candidate, previous, &|key| {
                    self.graph.edge_weight(key).unwrap_or_default()
                });
                if best.is_none_or(|(best_candidate, best_score)| {
                    score > best_score
                        || (score == best_score && candidate.weight > best_candidate.weight)
                        || (score == best_score
                            && candidate.weight == best_candidate.weight
                            && candidate.node < best_candidate.node)
                }) {
                    best = Some((
                        MutationCandidate {
                            node: candidate.node,
                            coordinate,
                            weight: candidate.weight,
                        },
                        score,
                    ));
                }
                mutation_candidates.push(MutationCandidate {
                    node: candidate.node,
                    coordinate,
                    weight: candidate.weight,
                });
            }

            if let Some((candidate, _)) = best {
                used_nodes.insert(candidate.node);
                previous = Some(candidate.node);
                last_coordinate = Some(candidate.coordinate);
                decisions.push(AnchorDecision::Matched(candidate.node));
                matched_anchors.push(MatchedAnchorEntry {
                    index: occurrence.position,
                    selected: SelectedAnchor {
                        node: candidate.node,
                        coordinate: candidate.coordinate,
                        weight: candidate.weight,
                    },
                });
            } else if !has_candidate {
                decisions.push(AnchorDecision::Unmatched(AnchorRejectReason::NoCandidate));
            } else if has_coordinate {
                decisions.push(AnchorDecision::Unmatched(AnchorRejectReason::NotSelected));
            } else {
                decisions.push(AnchorDecision::Unmatched(
                    AnchorRejectReason::MissingTopology,
                ));
            }
            candidate_sets.push_position(mutation_candidates);
        }

        Ok((
            candidate_sets,
            AnchorPath {
                decisions,
                matched_anchors,
            },
        ))
    }

    fn collect_greedy_mutation_plan_with_cache(
        &self,
        sequence_id: SequenceId,
        occurrences: &[FragmentOccurrence],
        cache: &TopologyCache,
    ) -> Result<(MutationCandidateStorage, SparseMatchedPath)> {
        let mut candidate_sets =
            MutationCandidateStorage::with_position_capacity(occurrences.len());
        let mut matched_anchors = Vec::new();
        let mut used_nodes =
            UsedNodeTracker::new_for_path(Some(self.graph.node_count()), occurrences.len());
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
                if let Some(coordinate) = candidate.coordinate {
                    matched_anchors.push(MatchedAnchorEntry {
                        index: occurrence.position,
                        selected: SelectedAnchor {
                            node: candidate.node,
                            coordinate,
                            weight: candidate.weight,
                        },
                    });
                }
                candidate_sets.push_position(std::iter::empty::<MutationCandidate>());
                continue;
            }

            let mut best: Option<(MutationCandidate, Weight)> = None;
            let nodes = self
                .graph
                .fragment_index()
                .nodes_for_stored(&occurrence.key, kind);
            let mut mutation_candidates = Vec::with_capacity(nodes.len());
            for node_id in nodes.iter().copied() {
                if !self.can_node_accept_sequence(node_id, sequence_id) {
                    continue;
                }
                let candidate = self.anchor_candidate_from_cache(node_id, kind, cache)?;
                let Some(coordinate) = candidate.coordinate else {
                    continue;
                };
                if last_coordinate.is_some_and(|last| last >= coordinate)
                    || used_nodes.contains(candidate.node)
                {
                    continue;
                }
                let score = greedy_candidate_score(&candidate, previous, &|key| {
                    self.graph.edge_weight(key).unwrap_or_default()
                });
                if best.is_none_or(|(best_candidate, best_score)| {
                    score > best_score
                        || (score == best_score && candidate.weight > best_candidate.weight)
                        || (score == best_score
                            && candidate.weight == best_candidate.weight
                            && candidate.node < best_candidate.node)
                }) {
                    best = Some((
                        MutationCandidate {
                            node: candidate.node,
                            coordinate,
                            weight: candidate.weight,
                        },
                        score,
                    ));
                }
                mutation_candidates.push(MutationCandidate {
                    node: candidate.node,
                    coordinate,
                    weight: candidate.weight,
                });
            }

            if let Some((candidate, _)) = best {
                used_nodes.insert(candidate.node);
                previous = Some(candidate.node);
                last_coordinate = Some(candidate.coordinate);
                matched_anchors.push(MatchedAnchorEntry {
                    index: occurrence.position,
                    selected: SelectedAnchor {
                        node: candidate.node,
                        coordinate: candidate.coordinate,
                        weight: candidate.weight,
                    },
                });
            }
            candidate_sets.push_position(mutation_candidates);
        }

        Ok((candidate_sets, SparseMatchedPath { matched_anchors }))
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

    fn direct_child_extension_for_stored_fragment(
        &self,
        sequence_id: SequenceId,
        kind: NodeKind,
        fragment: &crate::graph_model::graph::StoredFragmentKey,
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
        for child in self.graph.children(previous)?.iter().copied() {
            if state.used_nodes.contains(child)
                || !self.can_node_accept_sequence(child, sequence_id)
            {
                continue;
            }
            let node = self.graph.node(child)?;
            if node.kind != kind || node.fragment != *fragment {
                continue;
            }
            let candidate = self.anchor_candidate_from_cache(child, kind, cache)?;
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

    fn resolve_unique_exact_trace_path(
        &self,
        sequence_id: SequenceId,
        source_graph: &FtoDag,
        trace_path: &[NodeId],
    ) -> Result<Option<Vec<NodeId>>> {
        let Some((&first_source, rest)) = trace_path.split_first() else {
            return Ok(Some(Vec::new()));
        };

        let first_source_node = source_graph.node(first_source)?;
        let mut mapped = None;
        for node_id in self
            .graph
            .fragment_index()
            .nodes_for_stored(&first_source_node.fragment, first_source_node.kind)
            .iter()
            .copied()
        {
            if !self.can_node_accept_sequence(node_id, sequence_id) {
                continue;
            }
            if mapped.is_some() {
                return Ok(None);
            }
            mapped = Some(node_id);
        }

        let Some(mut current) = mapped else {
            return Ok(None);
        };
        let mut mapped_path = Vec::with_capacity(trace_path.len());
        mapped_path.push(current);

        for source_node_id in rest.iter().copied() {
            let source_node = source_graph.node(source_node_id)?;
            let mut next_match = None;
            for child in self.graph.children(current)?.iter().copied() {
                if !self.can_node_accept_sequence(child, sequence_id) {
                    continue;
                }
                let child_node = self.graph.node(child)?;
                if child_node.kind != source_node.kind
                    || child_node.fragment != source_node.fragment
                {
                    continue;
                }
                if next_match.is_some() {
                    return Ok(None);
                }
                next_match = Some(child);
            }
            let Some(next) = next_match else {
                return Ok(None);
            };
            mapped_path.push(next);
            current = next;
        }

        Ok(Some(mapped_path))
    }

    fn resolve_greedy_exact_trace_path(
        &mut self,
        sequence_id: SequenceId,
        source_graph: &FtoDag,
        trace_path: &[NodeId],
    ) -> Result<Option<Vec<NodeId>>> {
        if trace_path.len() <= GREEDY_ANCHOR_SET_LIMIT {
            return Ok(None);
        }
        let Some((&first_source, rest)) = trace_path.split_first() else {
            return Ok(Some(Vec::new()));
        };

        self.ensure_topology_cache()?;
        let cache = self
            .topology_cache
            .as_ref()
            .expect("incremental topology cache is initialized");
        let first_source_node = source_graph.node(first_source)?;
        let edge_score = |key| self.graph.edge_weight(key).unwrap_or_default();
        let mut best: Option<(AnchorCandidate, Weight)> = None;
        for node_id in self
            .graph
            .fragment_index()
            .nodes_for_stored(&first_source_node.fragment, first_source_node.kind)
            .iter()
            .copied()
        {
            if !self.can_node_accept_sequence(node_id, sequence_id) {
                continue;
            }
            let candidate =
                self.anchor_candidate_from_cache(node_id, first_source_node.kind, cache)?;
            let score = greedy_candidate_score(&candidate, None, &edge_score);
            if best.as_ref().is_none_or(|(best_candidate, best_score)| {
                better_greedy_candidate(&candidate, score, best_candidate, *best_score)
            }) {
                best = Some((candidate, score));
            }
        }

        let Some((first_candidate, _)) = best else {
            return Ok(None);
        };

        let mut used_nodes =
            UsedNodeTracker::new_for_path(Some(self.graph.node_count()), trace_path.len());
        used_nodes.insert(first_candidate.node);
        let mut mapped_path = Vec::with_capacity(trace_path.len());
        mapped_path.push(first_candidate.node);
        let mut previous = Some(first_candidate.node);
        let mut last_coordinate = first_candidate.coordinate;

        for source_node_id in rest.iter().copied() {
            let source_node = source_graph.node(source_node_id)?;
            let state = GreedySelectionState {
                previous,
                last_coordinate,
                used_nodes: &used_nodes,
            };
            let Some(candidate) = self.direct_child_extension_for_stored_fragment(
                sequence_id,
                source_node.kind,
                &source_node.fragment,
                state,
                cache,
            )?
            else {
                return Ok(None);
            };
            used_nodes.insert(candidate.node);
            previous = Some(candidate.node);
            last_coordinate = candidate.coordinate;
            mapped_path.push(candidate.node);
        }

        Ok(Some(mapped_path))
    }

    fn record_integrated_sequence_data(&mut self, result: &IntegratedSequenceBuildData) {
        self.report.integrated_sequences.push(result.sequence_id);
        self.report.total_nodes_created += result.new_nodes_created;
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
        let order = topological_order_from_graph(graph)?;
        let mut forward_coordinates = vec![TopologicalCoordinate::new(0); graph.node_count()];
        for node in &order {
            let coordinate = graph
                .parents(*node)?
                .iter()
                .map(|parent| forward_coordinates[parent.to_usize()].raw())
                .max()
                .unwrap_or(0)
                + 1;
            forward_coordinates[node.to_usize()] = TopologicalCoordinate::new(coordinate);
        }
        let reverse_coordinates = if include_reverse {
            let mut reverse_coordinates = vec![TopologicalCoordinate::new(0); graph.node_count()];
            for node in order.iter().rev() {
                let coordinate = graph
                    .children(*node)?
                    .iter()
                    .map(|child| reverse_coordinates[child.to_usize()].raw())
                    .max()
                    .unwrap_or(0)
                    + 1;
                reverse_coordinates[node.to_usize()] = TopologicalCoordinate::new(coordinate);
            }
            Some(reverse_coordinates)
        } else {
            None
        };
        Ok(Self {
            forward_coordinates,
            reverse_coordinates,
        })
    }
}

fn topological_order_from_graph(graph: &FtoDag) -> Result<Vec<NodeId>> {
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

impl TopologyCache {
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
    use crate::foundations::id::ProvenancePosition;
    use crate::graph_model::provenance::ProvenanceRecord;
    use crate::graph_model::provenance::ProvenanceStorageStrategy;
    use crate::sequence_model::alphabet::SymbolId;
    use crate::sequence_model::fragment::FragmentKey;

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

    fn add_trace_path(graph: &mut FtoDag, sequence_id: u32, path: &[NodeId]) {
        for (position, node_id) in path.iter().copied().enumerate() {
            graph
                .add_provenance_record(
                    node_id,
                    ProvenanceRecord {
                        sequence_id: SequenceId::new(sequence_id),
                        position: ProvenancePosition::new(position as u64),
                    },
                )
                .expect("trace-path provenance insertion succeeds");
        }
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
    fn topology_cache_from_graph_matches_full_with_reverse_coordinates() {
        let mut graph = FtoDag::new(1);
        let start = graph.add_node(key(0), NodeKind::Start).unwrap();
        let left = graph.add_node(key(1), NodeKind::Internal).unwrap();
        let right = graph.add_node(key(2), NodeKind::Internal).unwrap();
        let end = graph.add_node(key(3), NodeKind::End).unwrap();
        add_edge(&mut graph, start, left);
        add_edge(&mut graph, start, right);
        add_edge(&mut graph, left, end);
        add_edge(&mut graph, right, end);

        let cache = TopologyCache::from_graph(&graph, true).unwrap();
        let topology = DagTopology::try_from_graph(&graph).unwrap();
        for index in 0..graph.node_count() {
            let node = NodeId::try_from(index).unwrap();
            assert_eq!(
                cache.forward_coordinate(node).unwrap(),
                topology.forward_coordinate(node).unwrap()
            );
            assert_eq!(
                cache.reverse_coordinate(node).unwrap(),
                Some(topology.reverse_coordinate(node).unwrap())
            );
        }
    }

    #[test]
    fn existing_node_marks_reset_between_sequences() {
        let mut marks = ExistingNodeMarks::default();
        marks.begin_sequence(4);
        marks.mark(NodeId::new(2));
        assert!(marks.contains(NodeId::new(2)));

        marks.begin_sequence(4);
        assert!(!marks.contains(NodeId::new(2)));
        marks.mark(NodeId::new(1));
        assert!(marks.contains(NodeId::new(1)));
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

    #[test]
    fn exact_trace_path_summary_reuses_unique_existing_path() {
        let mut base = FtoDag::with_provenance_storage(1, ProvenanceStorageStrategy::TracePaths);
        let start = base.add_node(key(0), NodeKind::Start).unwrap();
        let middle = base.add_node(key(1), NodeKind::Internal).unwrap();
        let end = base.add_node(key(2), NodeKind::End).unwrap();
        add_edge(&mut base, start, middle);
        add_edge(&mut base, middle, end);
        add_trace_path(&mut base, 0, &[start, middle, end]);

        let mut add = FtoDag::with_provenance_storage(1, ProvenanceStorageStrategy::TracePaths);
        let add_start = add.add_node(key(0), NodeKind::Start).unwrap();
        let add_middle = add.add_node(key(1), NodeKind::Internal).unwrap();
        let add_end = add.add_node(key(2), NodeKind::End).unwrap();
        add_edge(&mut add, add_start, add_middle);
        add_edge(&mut add, add_middle, add_end);
        add_trace_path(&mut add, 0, &[add_start, add_middle, add_end]);

        let mut config = BuildConfig::new(1);
        config.provenance_storage_strategy = ProvenanceStorageStrategy::TracePaths;
        let mut builder = FtoDagBuilder::from_graph(base, config).unwrap();
        let trace_path = add.sequence_trace_path(SequenceId::new(0)).unwrap();

        assert_eq!(
            builder
                .add_exact_trace_path_summary(SequenceId::new(1), &add, &trace_path)
                .unwrap(),
            Some(vec![start, middle, end])
        );
        assert_eq!(builder.report().attempted_sequences, 1);
        assert_eq!(
            builder.report().integrated_sequences,
            vec![SequenceId::new(1)]
        );
        assert_eq!(builder.report().total_nodes_created, 0);
        assert_eq!(builder.report().total_edges_inserted, 0);
        assert_eq!(builder.report().total_provenance_records_added, 3);
        assert_eq!(builder.graph().provenance_record_count(start).unwrap(), 2);
        assert_eq!(builder.graph().provenance_record_count(middle).unwrap(), 2);
        assert_eq!(builder.graph().provenance_record_count(end).unwrap(), 2);
        assert_eq!(
            builder.graph().edge_weight(EdgeKey {
                parent: start,
                child: middle,
            }),
            Some(Weight::new(2))
        );
        assert_eq!(
            builder.graph().edge_weight(EdgeKey {
                parent: middle,
                child: end,
            }),
            Some(Weight::new(2))
        );
        assert_eq!(
            builder
                .graph()
                .sequence_trace_path(SequenceId::new(1))
                .unwrap(),
            vec![start, middle, end]
        );
    }

    #[test]
    fn exact_trace_path_summary_skips_ambiguous_existing_paths() {
        let mut base = FtoDag::with_provenance_storage(1, ProvenanceStorageStrategy::TracePaths);
        let left_start = base.add_node(key(0), NodeKind::Start).unwrap();
        let left_mid = base.add_node(key(1), NodeKind::Internal).unwrap();
        let left_end = base.add_node(key(2), NodeKind::End).unwrap();
        add_edge(&mut base, left_start, left_mid);
        add_edge(&mut base, left_mid, left_end);
        add_trace_path(&mut base, 0, &[left_start, left_mid, left_end]);

        let right_start = base.add_node(key(0), NodeKind::Start).unwrap();
        let right_mid = base.add_node(key(1), NodeKind::Internal).unwrap();
        let right_end = base.add_node(key(2), NodeKind::End).unwrap();
        add_edge(&mut base, right_start, right_mid);
        add_edge(&mut base, right_mid, right_end);
        add_trace_path(&mut base, 1, &[right_start, right_mid, right_end]);

        let mut add = FtoDag::with_provenance_storage(1, ProvenanceStorageStrategy::TracePaths);
        let add_start = add.add_node(key(0), NodeKind::Start).unwrap();
        let add_mid = add.add_node(key(1), NodeKind::Internal).unwrap();
        let add_end = add.add_node(key(2), NodeKind::End).unwrap();
        add_edge(&mut add, add_start, add_mid);
        add_edge(&mut add, add_mid, add_end);
        add_trace_path(&mut add, 0, &[add_start, add_mid, add_end]);

        let mut config = BuildConfig::new(1);
        config.provenance_storage_strategy = ProvenanceStorageStrategy::TracePaths;
        let mut builder = FtoDagBuilder::from_graph(base, config).unwrap();
        let trace_path = add.sequence_trace_path(SequenceId::new(0)).unwrap();

        assert_eq!(
            builder
                .add_exact_trace_path_summary(SequenceId::new(2), &add, &trace_path)
                .unwrap(),
            None
        );
        assert_eq!(builder.report().attempted_sequences, 0);
        assert!(builder.report().integrated_sequences.is_empty());
        assert!(
            builder
                .graph()
                .sequence_trace_path(SequenceId::new(2))
                .is_err()
        );
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
    select_monotone_anchor_path_with_graph_view(candidate_sets, graph)
}

pub(crate) fn select_monotone_anchor_path_with_graph_view<C, S>(
    candidate_sets: &[S],
    graph: &FtoDag,
) -> AnchorPath
where
    C: CandidateLike,
    S: CandidateSetLike<C>,
{
    if candidate_sets.len() > GREEDY_ANCHOR_SET_LIMIT {
        return select_greedy_monotone_anchor_path_with_graph(candidate_sets, graph);
    }
    select_monotone_anchor_path_with_edge_score(
        candidate_sets,
        |key| graph.edge_weight(key).unwrap_or_default(),
        Some(graph.node_count()),
    )
}

fn select_monotone_anchor_path_with_edge_score<C, S>(
    candidate_sets: &[S],
    edge_score: impl Fn(EdgeKey) -> Weight,
    node_count: Option<usize>,
) -> AnchorPath
where
    C: CandidateLike,
    S: CandidateSetLike<C>,
{
    if candidate_sets.len() > GREEDY_ANCHOR_SET_LIMIT {
        return select_greedy_monotone_anchor_path(candidate_sets, edge_score, node_count);
    }

    let mut dp: Vec<Vec<Option<AnchorDpCell>>> = candidate_sets
        .iter()
        .map(|set| vec![None; set.candidates().len()])
        .collect();

    let mut best: Option<(usize, usize, AnchorDpCell)> = None;
    for (set_index, set) in candidate_sets.iter().enumerate() {
        for (candidate_index, candidate) in set.candidates().iter().enumerate() {
            let Some(coordinate) = candidate.coordinate() else {
                continue;
            };
            let mut cell = AnchorDpCell {
                len: 1,
                continuity: Weight::new(0),
                weight: candidate.weight(),
                prev: None,
            };

            for prev_set_index in 0..set_index {
                for (prev_candidate_index, prev_candidate) in candidate_sets[prev_set_index]
                    .candidates()
                    .iter()
                    .enumerate()
                {
                    let Some(prev_coordinate) = prev_candidate.coordinate() else {
                        continue;
                    };
                    if prev_coordinate >= coordinate || prev_candidate.node() == candidate.node() {
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
                                    parent: prev_candidate.node(),
                                    child: candidate.node(),
                                })
                                .raw(),
                        ),
                        weight: Weight::new(prev_cell.weight.raw() + candidate.weight().raw()),
                        prev: Some((prev_set_index, prev_candidate_index)),
                    };
                    if better_cell(next, cell, candidate.node(), candidate.node()) {
                        cell = next;
                    }
                }
            }
            dp[set_index][candidate_index] = Some(cell);
            match best {
                Some((best_set_index, best_candidate_index, best_cell)) => {
                    let best_node =
                        candidate_sets[best_set_index].candidates()[best_candidate_index].node();
                    if better_cell(cell, best_cell, candidate.node(), best_node) {
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
            if set.candidates().is_empty() {
                AnchorDecision::Unmatched(AnchorRejectReason::NoCandidate)
            } else if set
                .candidates()
                .iter()
                .any(|candidate| candidate.coordinate().is_some())
            {
                AnchorDecision::Unmatched(AnchorRejectReason::NotSelected)
            } else {
                AnchorDecision::Unmatched(AnchorRejectReason::MissingTopology)
            }
        })
        .collect::<Vec<_>>();
    let mut matched_anchors = Vec::new();

    if let Some((mut set_index, mut candidate_index, _)) = best {
        loop {
            let candidate = &candidate_sets[set_index].candidates()[candidate_index];
            decisions[set_index] = AnchorDecision::Matched(candidate.node());
            if let Some(coordinate) = candidate.coordinate() {
                matched_anchors.push(MatchedAnchorEntry {
                    index: set_index,
                    selected: SelectedAnchor {
                        node: candidate.node(),
                        coordinate,
                        weight: candidate.weight(),
                    },
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
    matched_anchors.reverse();

    AnchorPath {
        decisions,
        matched_anchors,
    }
}

fn select_greedy_monotone_anchor_path<C, S>(
    candidate_sets: &[S],
    edge_score: impl Fn(EdgeKey) -> Weight,
    node_count: Option<usize>,
) -> AnchorPath
where
    C: CandidateLike,
    S: CandidateSetLike<C>,
{
    let mut decisions = Vec::with_capacity(candidate_sets.len());
    let mut matched_anchors = Vec::new();
    let mut used_nodes = UsedNodeTracker::new_for_path(node_count, candidate_sets.len());
    let mut previous: Option<NodeId> = None;
    let mut last_coordinate: Option<TopologicalCoordinate> = None;

    for set in candidate_sets {
        if let Some(candidate) =
            direct_extension_candidate(set, previous, last_coordinate, &used_nodes, &edge_score)
        {
            used_nodes.insert(candidate.node());
            previous = Some(candidate.node());
            last_coordinate = candidate.coordinate();
            decisions.push(AnchorDecision::Matched(candidate.node()));
            if let Some(coordinate) = candidate.coordinate() {
                matched_anchors.push(MatchedAnchorEntry {
                    index: decisions.len() - 1,
                    selected: SelectedAnchor {
                        node: candidate.node(),
                        coordinate,
                        weight: candidate.weight(),
                    },
                });
            }
            continue;
        }

        let mut best: Option<(&C, Weight)> = None;
        let mut has_coordinate = false;
        for candidate in set.candidates() {
            let Some(coordinate) = candidate.coordinate() else {
                continue;
            };
            has_coordinate = true;
            if last_coordinate.is_some_and(|last| last >= coordinate)
                || used_nodes.contains(candidate.node())
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
            used_nodes.insert(candidate.node());
            previous = Some(candidate.node());
            last_coordinate = candidate.coordinate();
            decisions.push(AnchorDecision::Matched(candidate.node()));
            if let Some(coordinate) = candidate.coordinate() {
                matched_anchors.push(MatchedAnchorEntry {
                    index: decisions.len() - 1,
                    selected: SelectedAnchor {
                        node: candidate.node(),
                        coordinate,
                        weight: candidate.weight(),
                    },
                });
            }
        } else if set.candidates().is_empty() {
            decisions.push(AnchorDecision::Unmatched(AnchorRejectReason::NoCandidate));
        } else if has_coordinate {
            decisions.push(AnchorDecision::Unmatched(AnchorRejectReason::NotSelected));
        } else {
            decisions.push(AnchorDecision::Unmatched(
                AnchorRejectReason::MissingTopology,
            ));
        }
    }

    AnchorPath {
        decisions,
        matched_anchors,
    }
}

fn select_greedy_monotone_anchor_path_with_graph<C, S>(
    candidate_sets: &[S],
    graph: &FtoDag,
) -> AnchorPath
where
    C: CandidateLike,
    S: CandidateSetLike<C>,
{
    let mut decisions = Vec::with_capacity(candidate_sets.len());
    let mut matched_anchors = Vec::new();
    let mut used_nodes =
        UsedNodeTracker::new_for_path(Some(graph.node_count()), candidate_sets.len());
    let mut previous: Option<NodeId> = None;
    let mut last_coordinate: Option<TopologicalCoordinate> = None;

    for set in candidate_sets {
        if let Some(candidate) =
            direct_child_extension_candidate(set, previous, last_coordinate, &used_nodes, graph)
        {
            used_nodes.insert(candidate.node());
            previous = Some(candidate.node());
            last_coordinate = candidate.coordinate();
            decisions.push(AnchorDecision::Matched(candidate.node()));
            if let Some(coordinate) = candidate.coordinate() {
                matched_anchors.push(MatchedAnchorEntry {
                    index: decisions.len() - 1,
                    selected: SelectedAnchor {
                        node: candidate.node(),
                        coordinate,
                        weight: candidate.weight(),
                    },
                });
            }
            continue;
        }

        let mut best: Option<(&C, Weight)> = None;
        let mut has_coordinate = false;
        for candidate in set.candidates() {
            let Some(coordinate) = candidate.coordinate() else {
                continue;
            };
            has_coordinate = true;
            if last_coordinate.is_some_and(|last| last >= coordinate)
                || used_nodes.contains(candidate.node())
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
            used_nodes.insert(candidate.node());
            previous = Some(candidate.node());
            last_coordinate = candidate.coordinate();
            decisions.push(AnchorDecision::Matched(candidate.node()));
            if let Some(coordinate) = candidate.coordinate() {
                matched_anchors.push(MatchedAnchorEntry {
                    index: decisions.len() - 1,
                    selected: SelectedAnchor {
                        node: candidate.node(),
                        coordinate,
                        weight: candidate.weight(),
                    },
                });
            }
        } else if set.candidates().is_empty() {
            decisions.push(AnchorDecision::Unmatched(AnchorRejectReason::NoCandidate));
        } else if has_coordinate {
            decisions.push(AnchorDecision::Unmatched(AnchorRejectReason::NotSelected));
        } else {
            decisions.push(AnchorDecision::Unmatched(
                AnchorRejectReason::MissingTopology,
            ));
        }
    }

    AnchorPath {
        decisions,
        matched_anchors,
    }
}

fn direct_child_extension_candidate<'a, C, S>(
    set: &'a S,
    previous: Option<NodeId>,
    last_coordinate: Option<TopologicalCoordinate>,
    used_nodes: &UsedNodeTracker,
    graph: &FtoDag,
) -> Option<&'a C>
where
    C: CandidateLike,
    S: CandidateSetLike<C>,
{
    let previous = previous?;
    let next_coordinate = TopologicalCoordinate::new(last_coordinate?.raw() + 1);
    let mut best: Option<(&C, Weight)> = None;
    let children = graph.children(previous).ok()?;
    for child in children {
        if used_nodes.contains(*child) {
            continue;
        }
        let Some(candidate) = set.candidates().iter().find(|candidate| {
            candidate.node() == *child && candidate.coordinate() == Some(next_coordinate)
        }) else {
            continue;
        };
        let score = graph
            .edge_weight(EdgeKey {
                parent: previous,
                child: candidate.node(),
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

fn direct_extension_candidate<'a, C, S>(
    set: &'a S,
    previous: Option<NodeId>,
    last_coordinate: Option<TopologicalCoordinate>,
    used_nodes: &UsedNodeTracker,
    edge_score: &impl Fn(EdgeKey) -> Weight,
) -> Option<&'a C>
where
    C: CandidateLike,
    S: CandidateSetLike<C>,
{
    let previous = previous?;
    let next_coordinate = TopologicalCoordinate::new(last_coordinate?.raw() + 1);
    let mut best: Option<(&C, Weight)> = None;
    for candidate in set.candidates() {
        if candidate.coordinate() != Some(next_coordinate) || used_nodes.contains(candidate.node())
        {
            continue;
        }
        let score = edge_score(EdgeKey {
            parent: previous,
            child: candidate.node(),
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

fn greedy_candidate_score<C: CandidateLike>(
    candidate: &C,
    previous: Option<NodeId>,
    edge_score: &impl Fn(EdgeKey) -> Weight,
) -> Weight {
    previous
        .map(|parent| {
            edge_score(EdgeKey {
                parent,
                child: candidate.node(),
            })
        })
        .unwrap_or_default()
}

fn better_greedy_candidate<C: CandidateLike>(
    candidate: &C,
    score: Weight,
    current: &C,
    current_score: Weight,
) -> bool {
    score > current_score
        || (score == current_score && candidate.weight() > current.weight())
        || (score == current_score
            && candidate.weight() == current.weight()
            && candidate.node() < current.node())
}

enum UsedNodeTracker {
    Marks(Vec<bool>),
    Set(HashSet<NodeId>),
}

impl UsedNodeTracker {
    fn new_for_path(node_count: Option<usize>, expected_used: usize) -> Self {
        const DENSE_MARK_LIMIT_FACTOR: usize = 64;
        match node_count {
            Some(node_count)
                if node_count
                    <= expected_used
                        .saturating_mul(DENSE_MARK_LIMIT_FACTOR)
                        .max(4096) =>
            {
                Self::Marks(vec![false; node_count])
            }
            Some(_) | None => Self::Set(HashSet::with_capacity(expected_used.saturating_mul(2))),
        }
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

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ExistingNodeMarks {
    marks: Vec<u64>,
    touched: Vec<NodeId>,
}

impl ExistingNodeMarks {
    fn begin_sequence(&mut self, node_count: usize) {
        for node in self.touched.drain(..) {
            let word_index = node.to_usize() / u64::BITS as usize;
            let bit_index = node.to_usize() % u64::BITS as usize;
            if let Some(word) = self.marks.get_mut(word_index) {
                *word &= !(1_u64 << bit_index);
            }
        }
        let word_count = node_count.div_ceil(u64::BITS as usize);
        if self.marks.len() < word_count {
            self.marks.resize(word_count, 0);
        }
    }

    fn contains(&self, node: NodeId) -> bool {
        let word_index = node.to_usize() / u64::BITS as usize;
        let bit_index = node.to_usize() % u64::BITS as usize;
        self.marks
            .get(word_index)
            .is_some_and(|word| (*word & (1_u64 << bit_index)) != 0)
    }

    fn mark(&mut self, node: NodeId) {
        let word_index = node.to_usize() / u64::BITS as usize;
        let bit_index = node.to_usize() % u64::BITS as usize;
        if let Some(word) = self.marks.get_mut(word_index) {
            let mask = 1_u64 << bit_index;
            if (*word & mask) == 0 {
                *word |= mask;
                self.touched.push(node);
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
        let score = reuse_score_for_node(candidate.node, graph, previous_node, next_node);
        if best.is_none_or(|(best_candidate, best_score)| {
            better_greedy_candidate(candidate, score, best_candidate, best_score)
        }) {
            best = Some((candidate, score));
        }
    }
    best.map(|(candidate, _)| candidate)
}

fn select_bounded_reuse_candidate_with_marks<'a>(
    candidate_set: &'a [MutationCandidate],
    interval: CoordinateInterval,
    used_node_marks: &ExistingNodeMarks,
    graph: &FtoDag,
    previous_node: Option<NodeId>,
    next_node: Option<NodeId>,
) -> Option<&'a MutationCandidate> {
    let mut best: Option<(&MutationCandidate, Weight)> = None;
    for candidate in candidate_set {
        if !interval.contains_open(candidate.coordinate)
            || used_node_marks.contains(candidate.node)
            || next_node == Some(candidate.node)
        {
            continue;
        }
        let score = reuse_score_for_node(candidate.node, graph, previous_node, next_node);
        if best.is_none_or(|(best_candidate, best_score)| {
            score > best_score
                || (score == best_score && candidate.weight > best_candidate.weight)
                || (score == best_score
                    && candidate.weight == best_candidate.weight
                    && candidate.node < best_candidate.node)
        }) {
            best = Some((candidate, score));
        }
    }
    best.map(|(candidate, _)| candidate)
}

pub fn block_plan_from_anchor_path(anchor_path: &AnchorPath) -> BlockPlan {
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
                            right: selected_coordinate(anchor_path, index, *node_id),
                        },
                    });
                }

                let Some(selected) = selected_anchor(anchor_path, index, *node_id) else {
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

fn compact_candidate_sets_for_mutation(
    candidate_sets: Vec<AnchorCandidateSet>,
    anchor_path: &AnchorPath,
) -> MutationCandidateStorage {
    let mut storage = MutationCandidateStorage::with_position_capacity(candidate_sets.len());
    for (candidate_set, decision) in candidate_sets.into_iter().zip(anchor_path.decisions.iter()) {
        match decision {
            AnchorDecision::Matched(_) => {
                storage.push_position(std::iter::empty::<MutationCandidate>())
            }
            AnchorDecision::Unmatched(_) => {
                storage.push_position(candidate_set.candidates.into_iter().filter_map(
                    |candidate| {
                        candidate.coordinate.map(|coordinate| MutationCandidate {
                            node: candidate.node,
                            coordinate,
                            weight: candidate.weight,
                        })
                    },
                ))
            }
        }
    }
    storage
}

fn reuse_score_for_node(
    node: NodeId,
    graph: &FtoDag,
    previous_node: Option<NodeId>,
    next_node: Option<NodeId>,
) -> Weight {
    let previous_score = previous_node
        .and_then(|previous| {
            graph.edge_weight(EdgeKey {
                parent: previous,
                child: node,
            })
        })
        .unwrap_or_default();
    let next_score = next_node
        .and_then(|next| {
            graph.edge_weight(EdgeKey {
                parent: node,
                child: next,
            })
        })
        .unwrap_or_default();
    Weight::new(previous_score.raw() + next_score.raw())
}

fn selected_anchor(
    anchor_path: &AnchorPath,
    index: usize,
    node_id: NodeId,
) -> Option<SelectedAnchor> {
    anchor_path
        .matched_anchors
        .binary_search_by_key(&index, |entry| entry.index)
        .ok()
        .and_then(|matched_index| anchor_path.matched_anchors.get(matched_index))
        .map(|entry| entry.selected)
        .filter(|selected| selected.node == node_id)
}

fn selected_coordinate(
    anchor_path: &AnchorPath,
    index: usize,
    node_id: NodeId,
) -> Option<TopologicalCoordinate> {
    selected_anchor(anchor_path, index, node_id).map(|selected| selected.coordinate)
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
