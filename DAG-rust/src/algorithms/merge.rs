//! Pairwise FTO-DAG merge interfaces.

use crate::algorithms::build::{
    select_monotone_anchor_path_with_graph_view, AnchorCandidate, AnchorDecision, AnchorPath,
    BuildConfig, BuildTimingBreakdown, CandidateSetLike, FtoDagBuilder, SimilarityThreshold,
    TopologyUpdateStrategy,
};
use crate::algorithms::postprocess::{
    secondary_merge_graph_with_coordinate_seed, SecondaryMergeConfig, SecondaryMergeCoordinateSeed,
};
use crate::foundations::error::{DagError, Result};
use crate::foundations::id::{GraphId, NodeId, SequenceId};
use crate::graph_model::graph::{FtoDag, NodeKind, StoredFragmentKey};
use crate::graph_model::provenance::ProvenanceStorageStrategy;
use crate::graph_model::reference::ReferencePath;
use crate::graph_model::topology::GraphCoordinateSnapshot;
use crate::sequence_model::fragment::{FragmentOccurrence, PathPositionKind};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

const FULL_PATH_MISS_CHECK_LIMIT: usize = 32;
const SAMPLED_MISS_CHECK_COUNT: usize = 12;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum MergeOrderingPolicy {
    DeterministicBinary,
    SketchBucketedSimilarity,
    UserProvided,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct MergeConfig {
    pub ordering_policy: MergeOrderingPolicy,
    pub min_initial_anchor_ratio: Option<SimilarityThreshold>,
}

impl Default for MergeConfig {
    fn default() -> Self {
        Self {
            ordering_policy: MergeOrderingPolicy::DeterministicBinary,
            min_initial_anchor_ratio: None,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MergeOrientation {
    LeftIntoRight,
    RightIntoLeft,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AnchorMap {
    pub pairs: Vec<(NodeId, NodeId)>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct NodeRemap {
    pub pairs: Vec<(NodeId, NodeId)>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MergePlan {
    pub orientation: MergeOrientation,
    pub anchors: AnchorMap,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
pub struct MergeStats {
    pub matched_nodes: usize,
    pub added_nodes: usize,
    pub added_edges: usize,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct RejectedMerge {
    pub base_graph_id: GraphId,
    pub add_graph_id: GraphId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MergeDecision {
    Merge(MergePlan),
    Reject(RejectedMerge),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
struct TracePathFingerprint {
    len: usize,
    hash: u64,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct CachedMergedPath {
    source_sequence_offset: usize,
    merged_sequence_id: SequenceId,
}

#[derive(Clone, Debug, Default)]
struct DuplicatePathBatchStats {
    total_paths: usize,
    unique_paths: usize,
    duplicate_hits: usize,
    exact_attempt_paths: usize,
    exact_unique_hits: usize,
    fallback_unique_paths: usize,
    cheap_miss_rejects: usize,
    sampled_fragment_checks: usize,
    collision_checks: usize,
    collision_buckets: usize,
    max_bucket_size: usize,
}

impl DuplicatePathBatchStats {
    fn print_summary(&self, cache_bucket_count: usize) {
        println!(
            "trace_path_duplicate_batch total_paths={} unique_paths={} duplicate_hits={} exact_attempt_paths={} exact_unique_hits={} fallback_unique_paths={} cheap_miss_rejects={} sampled_fragment_checks={} cache_buckets={} collision_buckets={} collision_checks={} max_bucket_size={}",
            self.total_paths,
            self.unique_paths,
            self.duplicate_hits,
            self.exact_attempt_paths,
            self.exact_unique_hits,
            self.fallback_unique_paths,
            self.cheap_miss_rejects,
            self.sampled_fragment_checks,
            cache_bucket_count,
            self.collision_buckets,
            self.collision_checks,
            self.max_bucket_size,
        );
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq)]
struct MemoryUsage {
    rss_kib: Option<u64>,
    peak_rss_kib: Option<u64>,
}

impl MemoryUsage {
    fn current() -> Self {
        let Ok(status) = fs::read_to_string("/proc/self/status") else {
            return Self::default();
        };
        let mut usage = Self::default();
        for line in status.lines() {
            if let Some(value) = line.strip_prefix("VmRSS:") {
                usage.rss_kib = parse_status_kib(value);
            } else if let Some(value) = line.strip_prefix("VmHWM:") {
                usage.peak_rss_kib = parse_status_kib(value);
            }
        }
        usage
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct PeakMemoryObservation {
    rss_kib: u64,
    peak_rss_kib: u64,
    phase: &'static str,
    sequence_offset: usize,
    path_len: usize,
}

#[derive(Clone, Debug, Default)]
struct MergePhaseProfile {
    total_paths: usize,
    duplicate_hit_paths: usize,
    exact_attempt_paths: usize,
    exact_hit_paths: usize,
    fallback_paths: usize,
    total_path_nodes: usize,
    exact_attempt_nodes: usize,
    fallback_nodes: usize,
    duplicate_lookup: Duration,
    duplicate_integration: Duration,
    exact_attempt: Duration,
    occurrence_materialization: Duration,
    fallback_timing: BuildTimingBreakdown,
    finalize: Duration,
    peak_observation: Option<PeakMemoryObservation>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum CountOnlyBackboneRole {
    Primary,
    Secondary,
}

impl CountOnlyBackboneRole {
    fn from_index(index: usize) -> Self {
        match index {
            0 => Self::Primary,
            1 => Self::Secondary,
            other => panic!("unsupported count-only backbone index {other}"),
        }
    }
}

#[derive(Clone, Debug)]
struct CountOnlyMergePlan {
    orientation: MergeOrientation,
    add_path_role: CountOnlyBackboneRole,
    base_path_role: CountOnlyBackboneRole,
    anchors: AnchorMap,
    score: u64,
    matched_nodes: usize,
}

#[derive(Clone, Debug)]
struct SelectedCountOnlyMergePlan {
    base_graph: FtoDag,
    add_graph: FtoDag,
    add_order: Vec<NodeId>,
    plan: CountOnlyMergePlan,
}

#[derive(Clone, Debug)]
struct PreparedReferencePath {
    role: CountOnlyBackboneRole,
    nodes: Vec<NodeId>,
    anchor_candidates: HashMap<(StoredFragmentKey, NodeKind), Vec<AnchorCandidate>>,
}

struct BorrowedAnchorCandidateSet<'a> {
    candidates: &'a [AnchorCandidate],
}

impl CandidateSetLike<AnchorCandidate> for BorrowedAnchorCandidateSet<'_> {
    fn candidates(&self) -> &[AnchorCandidate] {
        self.candidates
    }
}

struct PreparedCountOnlyGraph {
    coordinates: GraphCoordinateSnapshot,
    reference_paths: Vec<PreparedReferencePath>,
}

impl MergePhaseProfile {
    fn observe_memory(&mut self, phase: &'static str, sequence_offset: usize, path_len: usize) {
        let usage = MemoryUsage::current();
        let rss_kib = usage.rss_kib.unwrap_or(0);
        let peak_rss_kib = usage.peak_rss_kib.unwrap_or(0);
        if self
            .peak_observation
            .as_ref()
            .is_none_or(|current| rss_kib > current.rss_kib)
        {
            self.peak_observation = Some(PeakMemoryObservation {
                rss_kib,
                peak_rss_kib,
                phase,
                sequence_offset,
                path_len,
            });
        }
    }

    fn print_summary(&self) {
        let (peak_rss_mib, peak_hwm_mib, phase, sequence_offset, path_len) =
            if let Some(observation) = &self.peak_observation {
                (
                    kib_to_mib_string(Some(observation.rss_kib)),
                    kib_to_mib_string(Some(observation.peak_rss_kib)),
                    observation.phase,
                    observation.sequence_offset,
                    observation.path_len,
                )
            } else {
                ("n/a".to_string(), "n/a".to_string(), "n/a", 0, 0)
            };
        println!(
            "trace_path_merge_profile total_paths={} duplicate_hit_paths={} exact_attempt_paths={} exact_hit_paths={} fallback_paths={} total_path_nodes={} exact_attempt_nodes={} fallback_nodes={} duplicate_lookup_secs={:.6} duplicate_integration_secs={:.6} exact_attempt_secs={:.6} occurrence_secs={:.6} fallback_total_secs={:.6} fallback_initial_match_secs={:.6} fallback_topology_secs={:.6} fallback_candidate_secs={:.6} fallback_anchor_secs={:.6} fallback_block_plan_secs={:.6} fallback_mutation_secs={:.6} fallback_report_secs={:.6} finalize_secs={:.6} peak_phase={} peak_sequence_offset={} peak_path_len={} peak_current_rss_mib={} peak_hwm_mib={}",
            self.total_paths,
            self.duplicate_hit_paths,
            self.exact_attempt_paths,
            self.exact_hit_paths,
            self.fallback_paths,
            self.total_path_nodes,
            self.exact_attempt_nodes,
            self.fallback_nodes,
            self.duplicate_lookup.as_secs_f64(),
            self.duplicate_integration.as_secs_f64(),
            self.exact_attempt.as_secs_f64(),
            self.occurrence_materialization.as_secs_f64(),
            self.fallback_timing.total.as_secs_f64(),
            self.fallback_timing.initial_match.as_secs_f64(),
            self.fallback_timing.topology.as_secs_f64(),
            self.fallback_timing.candidate_collection.as_secs_f64(),
            self.fallback_timing.anchor_selection.as_secs_f64(),
            self.fallback_timing.block_planning.as_secs_f64(),
            self.fallback_timing.mutation.as_secs_f64(),
            self.fallback_timing.report_update.as_secs_f64(),
            self.finalize.as_secs_f64(),
            phase,
            sequence_offset,
            path_len,
            peak_rss_mib,
            peak_hwm_mib,
        );
    }
}

pub fn merge_graphs(left: FtoDag, right: FtoDag, config: MergeConfig) -> Result<FtoDag> {
    if left.fragment_len() != right.fragment_len() {
        return Err(crate::foundations::error::DagError::InvalidStorage(
            format!(
                "cannot merge graphs with fragment lengths {} and {}",
                left.fragment_len(),
                right.fragment_len()
            ),
        ));
    }
    if left.provenance_storage_strategy() != right.provenance_storage_strategy() {
        return Err(crate::foundations::error::DagError::InvalidStorage(
            format!(
                "cannot merge graphs with different provenance storage {:?} and {:?}",
                left.provenance_storage_strategy(),
                right.provenance_storage_strategy()
            ),
        ));
    }
    match left.provenance_storage_strategy() {
        ProvenanceStorageStrategy::TracePaths => replay_trace_path_graphs(left, right, config),
        ProvenanceStorageStrategy::CountOnly => merge_count_only_graphs(left, right, config),
        ProvenanceStorageStrategy::FullRecords | ProvenanceStorageStrategy::Packed32 => {
            Err(DagError::UnsupportedOperation(
                "merge_graphs currently supports TracePaths and CountOnly provenance storage",
            ))
        }
    }
}

pub fn merge_trace_path_graphs_as_count_only(
    left: FtoDag,
    right: FtoDag,
    config: MergeConfig,
) -> Result<FtoDag> {
    if left.provenance_storage_strategy() != ProvenanceStorageStrategy::TracePaths
        || right.provenance_storage_strategy() != ProvenanceStorageStrategy::TracePaths
    {
        return Err(DagError::InvalidStorage(
            "merge_trace_path_graphs_as_count_only requires TracePaths inputs".to_string(),
        ));
    }
    merge_graphs(left, right, config)?.to_count_only()
}

fn replay_trace_path_graphs(base: FtoDag, add: FtoDag, config: MergeConfig) -> Result<FtoDag> {
    let batch_instrumentation_enabled =
        std::env::var_os("DAG_RUST_TRACE_PATH_DUPLICATE_BATCH_INSTRUMENT").is_some();
    let phase_profile_enabled = std::env::var_os("DAG_RUST_TRACE_PATH_MERGE_PROFILE").is_some();
    let base_sequence_offset = base.sequence_trace_path_count()?;

    let mut build_config = BuildConfig::new(base.fragment_len());
    build_config.min_initial_match_ratio = config.min_initial_anchor_ratio;
    build_config.provenance_storage_strategy = base.provenance_storage_strategy();
    build_config.edge_index_strategy = base.edge_index_strategy();
    build_config.topology_update_strategy = TopologyUpdateStrategy::IncrementalForwardRelaxation;
    let mut builder = FtoDagBuilder::from_graph(base, build_config)?;
    let mut duplicate_cache = HashMap::<TracePathFingerprint, Vec<CachedMergedPath>>::new();
    let mut stats = (batch_instrumentation_enabled || phase_profile_enabled)
        .then(DuplicatePathBatchStats::default);
    let mut phase_profile = phase_profile_enabled.then(MergePhaseProfile::default);

    let mut trace_paths = add.sequence_trace_paths()?;
    while let Some((sequence_offset, trace_path)) = trace_paths.next_path()? {
        if trace_path.is_empty() {
            continue;
        }
        if let Some(stats) = stats.as_mut() {
            stats.total_paths += 1;
        }
        if let Some(profile) = phase_profile.as_mut() {
            profile.total_paths += 1;
            profile.total_path_nodes += trace_path.len();
        }
        let sequence_id =
            SequenceId::try_from(base_sequence_offset.saturating_add(sequence_offset))?;
        let fingerprint = fingerprint_trace_path(trace_path);

        let duplicate_lookup_start = phase_profile_enabled.then(Instant::now);
        if let Some(cached_sequence_id) = lookup_cached_duplicate_sequence(
            &add,
            trace_path,
            fingerprint,
            &duplicate_cache,
            stats.as_mut(),
        )? {
            if let (Some(profile), Some(start)) = (phase_profile.as_mut(), duplicate_lookup_start) {
                profile.duplicate_lookup += start.elapsed();
            }
            let mapped_path = builder.graph().sequence_trace_path(cached_sequence_id)?;
            let duplicate_integration_start = phase_profile_enabled.then(Instant::now);
            builder.integrate_mapped_trace_path_summary(sequence_id, &mapped_path)?;
            if let Some(stats) = stats.as_mut() {
                stats.duplicate_hits += 1;
            }
            if let Some(profile) = phase_profile.as_mut() {
                profile.duplicate_hit_paths += 1;
                if let Some(start) = duplicate_integration_start {
                    profile.duplicate_integration += start.elapsed();
                }
                profile.observe_memory("duplicate_hit", sequence_offset, trace_path.len());
            }
            continue;
        }
        if let (Some(profile), Some(start)) = (phase_profile.as_mut(), duplicate_lookup_start) {
            profile.duplicate_lookup += start.elapsed();
        }

        let try_exact =
            cheap_exact_reuse_possible(builder.graph(), &add, trace_path, stats.as_mut())?;
        if try_exact {
            if let Some(stats) = stats.as_mut() {
                stats.exact_attempt_paths += 1;
            }
            let exact_attempt_start = phase_profile_enabled.then(Instant::now);
            if let Some(profile) = phase_profile.as_mut() {
                profile.exact_attempt_paths += 1;
                profile.exact_attempt_nodes += trace_path.len();
            }
            if let Some(_mapped_path) =
                builder.add_exact_trace_path_summary(sequence_id, &add, trace_path)?
            {
                if let (Some(profile), Some(start)) = (phase_profile.as_mut(), exact_attempt_start)
                {
                    profile.exact_attempt += start.elapsed();
                }
                store_cached_merged_path(
                    &mut duplicate_cache,
                    fingerprint,
                    sequence_offset,
                    sequence_id,
                    stats.as_mut(),
                );
                if let Some(stats) = stats.as_mut() {
                    stats.unique_paths += 1;
                    stats.exact_unique_hits += 1;
                }
                if let Some(profile) = phase_profile.as_mut() {
                    profile.exact_hit_paths += 1;
                    profile.observe_memory("exact_hit", sequence_offset, trace_path.len());
                }
                continue;
            }
            if let (Some(profile), Some(start)) = (phase_profile.as_mut(), exact_attempt_start) {
                profile.exact_attempt += start.elapsed();
            }
        } else if let Some(stats) = stats.as_mut() {
            stats.cheap_miss_rejects += 1;
        }

        let occurrence_start = phase_profile_enabled.then(Instant::now);
        let occurrences = occurrences_from_trace_path(&add, trace_path)?;
        if let (Some(profile), Some(start)) = (phase_profile.as_mut(), occurrence_start) {
            profile.occurrence_materialization += start.elapsed();
            profile.observe_memory("occurrences", sequence_offset, trace_path.len());
        }
        if phase_profile_enabled {
            let profiled =
                builder.add_sequence_from_occurrences_profiled_summary(sequence_id, occurrences)?;
            if let Some(profile) = phase_profile.as_mut() {
                profile.fallback_paths += 1;
                profile.fallback_nodes += trace_path.len();
                profile.fallback_timing.add_assign(profiled.timing);
                profile.observe_memory("fallback", sequence_offset, trace_path.len());
            }
        } else {
            builder.add_sequence_from_occurrences_summary(sequence_id, occurrences)?;
        }
        store_cached_merged_path(
            &mut duplicate_cache,
            fingerprint,
            sequence_offset,
            sequence_id,
            stats.as_mut(),
        );
        if let Some(stats) = stats.as_mut() {
            stats.unique_paths += 1;
            stats.fallback_unique_paths += 1;
        }
    }

    let finalize_start = phase_profile_enabled.then(Instant::now);
    let graph = builder.finalize_graph()?;
    if let (Some(profile), Some(start)) = (phase_profile.as_mut(), finalize_start) {
        profile.finalize += start.elapsed();
        profile.observe_memory("finalize", usize::MAX, graph.node_count());
    }
    if let Some(stats) = stats.as_ref() {
        stats.print_summary(duplicate_cache.len());
    }
    if let Some(profile) = phase_profile.as_ref() {
        profile.print_summary();
    }
    Ok(graph)
}

fn merge_count_only_graphs(left: FtoDag, right: FtoDag, _config: MergeConfig) -> Result<FtoDag> {
    if left.node_count() == 0 {
        return Ok(right);
    }
    if right.node_count() == 0 {
        return Ok(left);
    }
    let selected = select_count_only_merge_plan(left, right)?;
    let merged = apply_count_only_merge_plan(
        selected.base_graph,
        selected.add_graph,
        &selected.add_order,
        &selected.plan,
    )?;
    let merged_coordinates = GraphCoordinateSnapshot::from_graph(&merged)?;
    secondary_merge_graph_with_coordinate_seed(
        merged,
        SecondaryMergeConfig::default(),
        SecondaryMergeCoordinateSeed::from_snapshot(&merged_coordinates),
    )
}

fn select_count_only_merge_plan(left: FtoDag, right: FtoDag) -> Result<SelectedCountOnlyMergePlan> {
    let left_prepared = prepare_count_only_graph(&left)?;
    let right_prepared = prepare_count_only_graph(&right)?;
    let right_into_left = plan_count_only_merge(
        &left,
        &left_prepared,
        &right,
        &right_prepared,
        MergeOrientation::RightIntoLeft,
    )?;
    let left_into_right = plan_count_only_merge(
        &right,
        &right_prepared,
        &left,
        &left_prepared,
        MergeOrientation::LeftIntoRight,
    )?;
    let PreparedCountOnlyGraph {
        coordinates: left_coordinates,
        ..
    } = left_prepared;
    let PreparedCountOnlyGraph {
        coordinates: right_coordinates,
        ..
    } = right_prepared;
    Ok(
        if prefer_count_only_plan(&left_into_right, &right_into_left) {
            SelectedCountOnlyMergePlan {
                base_graph: right,
                add_graph: left,
                add_order: left_coordinates.into_topological_order(),
                plan: left_into_right,
            }
        } else {
            SelectedCountOnlyMergePlan {
                base_graph: left,
                add_graph: right,
                add_order: right_coordinates.into_topological_order(),
                plan: right_into_left,
            }
        },
    )
}

fn prepare_count_only_graph(graph: &FtoDag) -> Result<PreparedCountOnlyGraph> {
    let coordinates = GraphCoordinateSnapshot::from_graph(graph)?;
    let reference_paths = count_only_reference_paths(graph, &coordinates)?;
    Ok(PreparedCountOnlyGraph {
        coordinates,
        reference_paths,
    })
}

fn plan_count_only_merge(
    base_graph: &FtoDag,
    base: &PreparedCountOnlyGraph,
    add_graph: &FtoDag,
    add: &PreparedCountOnlyGraph,
    orientation: MergeOrientation,
) -> Result<CountOnlyMergePlan> {
    let mut best_plan = None;
    for add_path in &add.reference_paths {
        for base_path in eligible_base_paths(&base.reference_paths, add_path.role) {
            let candidate_sets = count_only_candidate_sets(add_graph, &add_path.nodes, base_path)?;
            let anchor_path = select_monotone_anchor_path_with_graph_view(&candidate_sets, base_graph);
            let anchors = anchor_map_from_reference_path(&add_path.nodes, &anchor_path);
            let score = anchor_score(base_graph, add_graph, &anchors)?;
            let candidate = CountOnlyMergePlan {
                orientation,
                add_path_role: add_path.role,
                base_path_role: base_path.role,
                matched_nodes: anchors.pairs.len(),
                anchors,
                score,
            };
            if best_plan
                .as_ref()
                .is_none_or(|current| prefer_count_only_plan(&candidate, current))
            {
                best_plan = Some(candidate);
            }
        }
    }
    best_plan.ok_or(DagError::InvalidStorage(
        "count-only merge requires at least one backbone path pair".to_string(),
    ))
}

fn prefer_count_only_plan(candidate: &CountOnlyMergePlan, current: &CountOnlyMergePlan) -> bool {
    candidate.score > current.score
        || (candidate.score == current.score && candidate.matched_nodes > current.matched_nodes)
        || (candidate.score == current.score
            && candidate.matched_nodes == current.matched_nodes
            && count_only_path_pair_priority(candidate) < count_only_path_pair_priority(current))
        || (candidate.score == current.score
            && candidate.matched_nodes == current.matched_nodes
            && count_only_path_pair_priority(candidate) == count_only_path_pair_priority(current)
            && matches!(candidate.orientation, MergeOrientation::RightIntoLeft)
            && matches!(current.orientation, MergeOrientation::LeftIntoRight))
}

fn count_only_path_pair_priority(plan: &CountOnlyMergePlan) -> u8 {
    match (plan.add_path_role, plan.base_path_role) {
        (CountOnlyBackboneRole::Primary, CountOnlyBackboneRole::Primary) => 0,
        (CountOnlyBackboneRole::Primary, CountOnlyBackboneRole::Secondary) => 1,
        (CountOnlyBackboneRole::Secondary, CountOnlyBackboneRole::Primary) => 2,
        (CountOnlyBackboneRole::Secondary, CountOnlyBackboneRole::Secondary) => 3,
    }
}

fn count_only_reference_paths(
    graph: &FtoDag,
    coordinates: &GraphCoordinateSnapshot,
) -> Result<Vec<PreparedReferencePath>> {
    ReferencePath::max_weight_pair_with_order(graph, coordinates.topological_order())?
        .into_iter()
        .enumerate()
        .map(|(index, path)| {
            prepare_reference_path(
                graph,
                coordinates,
                CountOnlyBackboneRole::from_index(index),
                path.nodes,
            )
        })
        .collect()
}

fn prepare_reference_path(
    graph: &FtoDag,
    coordinates: &GraphCoordinateSnapshot,
    role: CountOnlyBackboneRole,
    nodes: Vec<NodeId>,
) -> Result<PreparedReferencePath> {
    let mut anchor_candidates = HashMap::<(StoredFragmentKey, NodeKind), Vec<AnchorCandidate>>::new();
    for node_id in nodes.iter().copied() {
        let node = graph.node(node_id)?;
        anchor_candidates
            .entry((node.fragment.clone(), node.kind))
            .or_default()
            .push(AnchorCandidate {
                node: node_id,
                kind: node.kind,
                coordinate: Some(coordinates.forward_coordinate(node_id)?),
                reverse_coordinate: Some(coordinates.reverse_coordinate(node_id)?),
                weight: node.weight,
            });
    }
    Ok(PreparedReferencePath {
        role,
        nodes,
        anchor_candidates,
    })
}

fn eligible_base_paths(
    base_paths: &[PreparedReferencePath],
    add_role: CountOnlyBackboneRole,
) -> impl Iterator<Item = &PreparedReferencePath> {
    base_paths.iter().filter(move |base_path| {
        !matches!(
            (add_role, base_path.role),
            (
                CountOnlyBackboneRole::Secondary,
                CountOnlyBackboneRole::Secondary
            )
        )
    })
}

fn count_only_candidate_sets<'a>(
    add: &FtoDag,
    add_path: &[NodeId],
    base_path: &'a PreparedReferencePath,
) -> Result<Vec<BorrowedAnchorCandidateSet<'a>>> {
    let mut sets = Vec::with_capacity(add_path.len());
    for (position, add_node_id) in add_path.iter().copied().enumerate() {
        let add_node = add.node(add_node_id)?;
        let path_key = (add_node.fragment.clone(), add_node.kind);
        let _ = position;
        sets.push(BorrowedAnchorCandidateSet {
            candidates: base_path
                .anchor_candidates
                .get(&path_key)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
        });
    }
    Ok(sets)
}

fn anchor_map_from_reference_path(add_path: &[NodeId], anchor_path: &AnchorPath) -> AnchorMap {
    AnchorMap {
        pairs: anchor_path
            .decisions
            .iter()
            .enumerate()
            .filter_map(|(index, decision)| match decision {
                AnchorDecision::Matched(base_node) => Some((add_path[index], *base_node)),
                AnchorDecision::Unmatched(_) => None,
            })
            .collect(),
    }
}

fn anchor_score(base: &FtoDag, add: &FtoDag, anchors: &AnchorMap) -> Result<u64> {
    anchors
        .pairs
        .iter()
        .try_fold(0_u64, |score, (add_node, base_node)| {
            let add_weight = add.node(*add_node)?.weight.raw();
            let base_weight = base.node(*base_node)?.weight.raw();
            Ok(score + add_weight + base_weight)
        })
}

fn apply_count_only_merge_plan(
    mut base: FtoDag,
    add: FtoDag,
    add_order: &[NodeId],
    plan: &CountOnlyMergePlan,
) -> Result<FtoDag> {
    let mut remap = vec![None; add.node_count()];
    for (add_node, base_node) in &plan.anchors.pairs {
        remap[add_node.to_usize()] = Some(*base_node);
    }

    for add_node in add_order.iter().copied() {
        if remap[add_node.to_usize()].is_some() {
            continue;
        }
        let source = add.node(add_node)?;
        let merged = base.add_node(source.fragment.clone(), source.kind)?;
        remap[add_node.to_usize()] = Some(merged);
    }

    for edge in add.edges() {
        let parent = remap[edge.key.parent.to_usize()].ok_or(DagError::InvalidEdge {
            parent: edge.key.parent.to_usize(),
            child: edge.key.child.to_usize(),
        })?;
        let child = remap[edge.key.child.to_usize()].ok_or(DagError::InvalidEdge {
            parent: edge.key.parent.to_usize(),
            child: edge.key.child.to_usize(),
        })?;
        if parent == child {
            continue;
        }
        base.add_or_increment_edge(parent, child, edge.weight)?;
    }

    let mut node_weights = vec![0_u64; base.node_count()];
    for node in add.nodes() {
        let merged = remap[node.id.to_usize()].ok_or(DagError::MissingNode {
            node: node.id.to_usize(),
        })?;
        node_weights[merged.to_usize()] += node.weight.raw();
    }
    for (index, weight) in node_weights.into_iter().enumerate() {
        if weight == 0 {
            continue;
        }
        base.add_provenance_count(NodeId::try_from(index)?, weight)?;
    }

    Ok(base)
}

fn fingerprint_trace_path(trace_path: &[NodeId]) -> TracePathFingerprint {
    let mut hasher = DefaultHasher::new();
    trace_path.len().hash(&mut hasher);
    for node_id in trace_path {
        node_id.hash(&mut hasher);
    }
    TracePathFingerprint {
        len: trace_path.len(),
        hash: hasher.finish(),
    }
}

fn lookup_cached_duplicate_sequence(
    add: &FtoDag,
    trace_path: &[NodeId],
    fingerprint: TracePathFingerprint,
    duplicate_cache: &HashMap<TracePathFingerprint, Vec<CachedMergedPath>>,
    stats: Option<&mut DuplicatePathBatchStats>,
) -> Result<Option<SequenceId>> {
    let Some(entries) = duplicate_cache.get(&fingerprint) else {
        return Ok(None);
    };

    if let Some(stats) = stats {
        if entries.len() > 1 {
            stats.collision_buckets += 1;
        }
        stats.collision_checks += entries.len();
    }

    for entry in entries {
        let source_sequence_id = SequenceId::try_from(entry.source_sequence_offset)?;
        if add.sequence_trace_path(source_sequence_id)?.as_slice() == trace_path {
            return Ok(Some(entry.merged_sequence_id));
        }
    }
    Ok(None)
}

fn cheap_exact_reuse_possible(
    base_graph: &FtoDag,
    add_graph: &FtoDag,
    trace_path: &[NodeId],
    stats: Option<&mut DuplicatePathBatchStats>,
) -> Result<bool> {
    if trace_path.is_empty() {
        return Ok(true);
    }
    let sample_positions = sampled_miss_check_positions(trace_path.len());
    if let Some(stats) = stats {
        stats.sampled_fragment_checks += sample_positions.len();
    }
    for position in sample_positions {
        let node = add_graph.node(trace_path[position])?;
        if base_graph
            .fragment_index()
            .nodes_for_stored(&node.fragment, node.kind)
            .is_empty()
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn sampled_miss_check_positions(path_len: usize) -> Vec<usize> {
    if path_len <= 1 {
        return vec![0];
    }
    if path_len <= FULL_PATH_MISS_CHECK_LIMIT {
        return (0..path_len).collect();
    }

    let sample_count = SAMPLED_MISS_CHECK_COUNT.min(path_len);
    let last_index = path_len - 1;
    let mut positions = Vec::with_capacity(sample_count + 2);
    positions.push(0);
    for sample_index in 1..sample_count.saturating_sub(1) {
        let numerator = sample_index.saturating_mul(last_index);
        positions.push(numerator / (sample_count - 1));
    }
    positions.push(last_index);
    positions.sort_unstable();
    positions.dedup();
    positions
}

fn parse_status_kib(raw: &str) -> Option<u64> {
    raw.split_whitespace().next()?.parse().ok()
}

fn kib_to_mib_string(value: Option<u64>) -> String {
    value.map_or_else(
        || "n/a".to_string(),
        |kib| format!("{:.1}", kib as f64 / 1024.0),
    )
}

fn store_cached_merged_path(
    duplicate_cache: &mut HashMap<TracePathFingerprint, Vec<CachedMergedPath>>,
    fingerprint: TracePathFingerprint,
    source_sequence_offset: usize,
    merged_sequence_id: SequenceId,
    stats: Option<&mut DuplicatePathBatchStats>,
) {
    let bucket = duplicate_cache.entry(fingerprint).or_default();
    bucket.push(CachedMergedPath {
        source_sequence_offset,
        merged_sequence_id,
    });
    if let Some(stats) = stats {
        stats.max_bucket_size = stats.max_bucket_size.max(bucket.len());
    }
}

fn occurrences_from_trace_path(
    graph: &FtoDag,
    trace_path: &[NodeId],
) -> Result<Vec<FragmentOccurrence>> {
    let mut occurrences = Vec::with_capacity(trace_path.len());
    for (position, node_id) in trace_path.iter().copied().enumerate() {
        let node = graph.node(node_id)?;
        occurrences.push(FragmentOccurrence {
            position,
            kind: path_position_kind(position, trace_path.len()),
            key: node.fragment.clone(),
        });
    }
    Ok(occurrences)
}

fn path_position_kind(position: usize, window_count: usize) -> PathPositionKind {
    if window_count == 1 {
        PathPositionKind::Singleton
    } else if position == 0 {
        PathPositionKind::Start
    } else if position + 1 == window_count {
        PathPositionKind::End
    } else {
        PathPositionKind::Internal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundations::id::Weight;
    use crate::graph_model::graph::NodeKind;
    use crate::sequence_model::alphabet::SymbolId;
    use crate::sequence_model::fragment::FragmentKey;

    fn key(raw: u16) -> FragmentKey {
        FragmentKey::symbols(vec![SymbolId::new(raw)])
    }

    fn add_edge(graph: &mut FtoDag, parent: NodeId, child: NodeId) {
        graph
            .add_or_increment_edge(parent, child, Weight::new(1))
            .unwrap();
    }

    #[test]
    fn sampled_miss_check_positions_cover_full_short_paths() {
        assert_eq!(sampled_miss_check_positions(1), vec![0]);
        assert_eq!(sampled_miss_check_positions(4), vec![0, 1, 2, 3]);
        assert_eq!(
            sampled_miss_check_positions(32),
            (0..32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn cheap_exact_reuse_possible_rejects_missing_sampled_fragment() {
        let mut base = FtoDag::with_provenance_storage(1, ProvenanceStorageStrategy::TracePaths);
        let base_start = base.add_node(key(0), NodeKind::Start).unwrap();
        let base_mid = base.add_node(key(1), NodeKind::Internal).unwrap();
        let base_end = base.add_node(key(2), NodeKind::End).unwrap();
        add_edge(&mut base, base_start, base_mid);
        add_edge(&mut base, base_mid, base_end);

        let mut add = FtoDag::with_provenance_storage(1, ProvenanceStorageStrategy::TracePaths);
        let add_start = add.add_node(key(0), NodeKind::Start).unwrap();
        let add_mid = add.add_node(key(9), NodeKind::Internal).unwrap();
        let add_end = add.add_node(key(2), NodeKind::End).unwrap();
        add_edge(&mut add, add_start, add_mid);
        add_edge(&mut add, add_mid, add_end);

        assert!(
            !cheap_exact_reuse_possible(&base, &add, &[add_start, add_mid, add_end], None).unwrap()
        );
    }
}
