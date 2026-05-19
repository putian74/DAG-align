use dag_rust::prelude::*;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

const SARS_COV2_ARCHIVE: &str = "/disk2/tian/hCoV-19_msa_0211.tar.xz";
const SARS_COV2_SEQUENCE_LIMIT: usize = 128;
const SARS_COV2_SNIPPET_LEN: usize = 2_000;
const SARS_COV2_BUILD_SEQUENCE_LIMIT: usize = 32;
const SARS_COV2_BUILD_SNIPPET_LEN: usize = 1_000;
const SARS_COV2_HEAVY_SEQUENCE_LIMIT: usize = 5_000;

#[test]
#[ignore = "uses local SARS-CoV-2 dataset outside the repository"]
fn local_sars_cov2_archive_subset_encodes_and_scores() {
    if !Path::new(SARS_COV2_ARCHIVE).exists() {
        return;
    }

    let snippets = load_sars_cov2_snippets(SARS_COV2_SEQUENCE_LIMIT, SARS_COV2_SNIPPET_LEN);
    assert!(
        snippets.len() >= SARS_COV2_SEQUENCE_LIMIT / 2,
        "expected a substantial FASTA subset, got {} records",
        snippets.len()
    );

    let alphabet = BuiltinAlphabet::dna_exact();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let encoded = snippets
        .iter()
        .map(|(id, sequence)| {
            EncodedSequence::encode(SequenceRecord::new(id, sequence.clone()), &alphabet)
        })
        .collect::<Result<Vec<_>>>()
        .expect("SARS-CoV-2 subset encodes");

    let mut config = BuildConfig::new(31);
    config.min_initial_match_ratio = Some(SimilarityThreshold::from_basis_points(100).unwrap());
    config.topology_update_strategy = topology_strategy_from_env();
    config.source_storage_strategy = source_storage_strategy_from_env();
    config.edge_index_strategy = edge_index_strategy_from_env();
    config.collect_topology_counters = topology_counters_from_env();
    config.topology_rebuild_queue_threshold = topology_rebuild_queue_threshold_from_env();
    let mut builder = FtoDagBuilder::new(config);
    builder
        .initialize_from_encoded(SequenceId::new(0), &encoded[0], &encoder)
        .expect("first dataset snippet initializes");

    assert!(builder.graph().node_count() > SARS_COV2_SNIPPET_LEN - 64);
    let mut proceeded = 0;
    let mut rejected = 0;
    for (index, sequence) in encoded.iter().enumerate().skip(1) {
        match builder
            .initial_match_decision(SequenceId::new(index as u32), sequence, &encoder)
            .expect("dataset snippet can be scored")
        {
            IntegrationDecision::Proceed => proceeded += 1,
            IntegrationDecision::Reject(rejected_sequence) => {
                rejected += 1;
                assert_eq!(rejected_sequence.sequence_id, SequenceId::new(index as u32));
            }
        }
    }

    assert_eq!(proceeded + rejected, encoded.len() - 1);
    assert!(
        proceeded > 0,
        "expected at least one similar SARS-CoV-2 sequence"
    );
}

#[test]
#[ignore = "uses local SARS-CoV-2 dataset outside the repository"]
fn local_sars_cov2_archive_subset_builds_dag() {
    if !Path::new(SARS_COV2_ARCHIVE).exists() {
        return;
    }

    let snippets =
        load_sars_cov2_snippets(SARS_COV2_BUILD_SEQUENCE_LIMIT, SARS_COV2_BUILD_SNIPPET_LEN);
    assert!(
        snippets.len() >= SARS_COV2_BUILD_SEQUENCE_LIMIT / 2,
        "expected a substantial FASTA build subset, got {} records",
        snippets.len()
    );

    let alphabet = BuiltinAlphabet::dna_exact();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let mut config = BuildConfig::new(31);
    config.min_initial_match_ratio = Some(SimilarityThreshold::from_basis_points(100).unwrap());
    config.source_storage_strategy = source_storage_strategy_from_env();
    config.edge_index_strategy = edge_index_strategy_from_env();
    let mut builder = FtoDagBuilder::new(config);
    let mut source = VecSequenceSource::new(
        snippets
            .iter()
            .map(|(id, sequence)| SequenceRecord::new(id, sequence.clone()))
            .collect(),
    );

    let report = builder
        .build_from_source(&mut source, &alphabet, &encoder)
        .expect("SARS-CoV-2 subset builds into DAG");

    assert_eq!(report.attempted_sequences, snippets.len());
    assert_eq!(
        report.integrated_sequences.len() + report.rejected_sequences.len(),
        snippets.len()
    );
    assert!(
        report.integrated_sequences.len() >= snippets.len() / 2,
        "expected most SARS-CoV-2 snippets to pass the low similarity gate"
    );
    assert!(builder.graph().node_count() > SARS_COV2_BUILD_SNIPPET_LEN - 64);
    assert!(report.total_source_records_added > SARS_COV2_BUILD_SNIPPET_LEN - 31);
    assert!(builder.graph().validate().is_valid());
    DagTopology::try_from_graph(builder.graph()).expect("dataset DAG remains acyclic");
}

#[test]
#[ignore = "heavy local SARS-CoV-2 full-genome test; set DAG_RUST_RUN_HEAVY_DATASET_TESTS=1"]
fn local_sars_cov2_archive_5000_full_genomes_score_similarity_gate() {
    if std::env::var_os("DAG_RUST_RUN_HEAVY_DATASET_TESTS").is_none() {
        return;
    }
    if !Path::new(SARS_COV2_ARCHIVE).exists() {
        return;
    }

    let load_start = Instant::now();
    let records = load_sars_cov2_full_sequences(SARS_COV2_HEAVY_SEQUENCE_LIMIT);
    assert_eq!(
        records.len(),
        SARS_COV2_HEAVY_SEQUENCE_LIMIT,
        "expected {SARS_COV2_HEAVY_SEQUENCE_LIMIT} full SARS-CoV-2 records"
    );

    let alphabet = BuiltinAlphabet::dna_exact();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let mut config = BuildConfig::new(31);
    config.min_initial_match_ratio = Some(SimilarityThreshold::from_basis_points(100).unwrap());
    let mut builder = FtoDagBuilder::new(config);
    let first = EncodedSequence::encode(
        SequenceRecord::new(&records[0].0, records[0].1.clone()),
        &alphabet,
    )
    .expect("first full SARS-CoV-2 record encodes");
    builder
        .initialize_from_encoded(SequenceId::new(0), &first, &encoder)
        .expect("first full SARS-CoV-2 record initializes");

    let score_start = Instant::now();
    let mut proceeded = 0_usize;
    let mut rejected = 0_usize;
    for (index, (id, sequence)) in records.iter().enumerate().skip(1) {
        let encoded = EncodedSequence::encode(SequenceRecord::new(id, sequence.clone()), &alphabet)
            .expect("full SARS-CoV-2 record encodes");
        match builder
            .initial_match_decision(SequenceId::try_from(index).unwrap(), &encoded, &encoder)
            .expect("full SARS-CoV-2 record can be scored")
        {
            IntegrationDecision::Proceed => proceeded += 1,
            IntegrationDecision::Reject(rejected_sequence) => {
                rejected += 1;
                assert_eq!(
                    rejected_sequence.sequence_id,
                    SequenceId::try_from(index).unwrap()
                );
            }
        }
    }

    assert_eq!(proceeded + rejected, SARS_COV2_HEAVY_SEQUENCE_LIMIT - 1);
    assert!(
        proceeded > SARS_COV2_HEAVY_SEQUENCE_LIMIT / 2,
        "expected most full SARS-CoV-2 genomes to pass a low similarity gate"
    );
    assert!(builder.graph().validate().is_valid());
    println!(
        "heavy_sars_cov2_full_score records={} seed_nodes={} proceeded={} rejected={} load_secs={:.2} score_secs={:.2}",
        records.len(),
        builder.graph().node_count(),
        proceeded,
        rejected,
        load_start.elapsed().as_secs_f64(),
        score_start.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore = "very heavy local SARS-CoV-2 full DAG integration; set DAG_RUST_RUN_FULL_INTEGRATION_TESTS=1"]
fn local_sars_cov2_archive_5000_full_genomes_builds_dag() {
    if std::env::var_os("DAG_RUST_RUN_FULL_INTEGRATION_TESTS").is_none() {
        return;
    }
    if !Path::new(SARS_COV2_ARCHIVE).exists() {
        return;
    }

    let record_limit = heavy_integration_record_limit();
    let load_start = Instant::now();
    let records = load_sars_cov2_full_sequences(record_limit);
    assert_eq!(
        records.len(),
        record_limit,
        "expected {record_limit} full SARS-CoV-2 records"
    );

    let alphabet = BuiltinAlphabet::dna_exact();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let mut config = BuildConfig::new(31);
    config.min_initial_match_ratio = Some(SimilarityThreshold::from_basis_points(100).unwrap());
    let mut builder = FtoDagBuilder::new(config);

    let build_start = Instant::now();
    for (index, (id, sequence)) in records.iter().enumerate() {
        let encoded = EncodedSequence::encode(SequenceRecord::new(id, sequence.clone()), &alphabet)
            .expect("full SARS-CoV-2 record encodes");
        builder
            .add_sequence_from_encoded(SequenceId::try_from(index).unwrap(), &encoded, &encoder)
            .expect("full SARS-CoV-2 record integrates or rejects");
        if (index + 1) % 100 == 0 || index + 1 == records.len() {
            let report = builder.report();
            let memory = MemoryUsage::current();
            println!(
                "heavy_sars_cov2_full_build_progress processed={} integrated={} rejected={} nodes={} edges={} sources={} elapsed_secs={:.2} rss_mib={} peak_rss_mib={}",
                index + 1,
                report.integrated_sequences.len(),
                report.rejected_sequences.len(),
                builder.graph().node_count(),
                builder.graph().edge_count(),
                report.total_source_records_added,
                build_start.elapsed().as_secs_f64(),
                memory.rss_mib(),
                memory.peak_rss_mib(),
            );
        }
    }
    let report = builder.report().clone();
    let topology_start = Instant::now();
    DagTopology::try_from_graph(builder.graph()).expect("full-genome DAG remains acyclic");

    assert_eq!(report.attempted_sequences, record_limit);
    assert_eq!(
        report.integrated_sequences.len() + report.rejected_sequences.len(),
        record_limit
    );
    assert!(
        report.integrated_sequences.len() > record_limit / 2,
        "expected most full SARS-CoV-2 genomes to integrate with a low similarity gate"
    );
    assert!(builder.graph().node_count() >= 29_000);
    assert!(builder.graph().edge_count() >= 28_000);
    assert!(
        report.total_source_records_added >= report.integrated_sequences.len() * 25_000,
        "expected full-genome integration to add many source records"
    );
    let memory = MemoryUsage::current();
    println!(
        "heavy_sars_cov2_full_build records={} integrated={} rejected={} nodes={} edges={} sources={} load_secs={:.2} build_secs={:.2} topology_secs={:.2} rss_mib={} peak_rss_mib={}",
        records.len(),
        report.integrated_sequences.len(),
        report.rejected_sequences.len(),
        builder.graph().node_count(),
        builder.graph().edge_count(),
        report.total_source_records_added,
        load_start.elapsed().as_secs_f64(),
        build_start.elapsed().as_secs_f64(),
        topology_start.elapsed().as_secs_f64(),
        memory.rss_mib(),
        memory.peak_rss_mib(),
    );
}

#[test]
#[ignore = "profile local SARS-CoV-2 full DAG integration; set DAG_RUST_RUN_PROFILE_TESTS=1"]
fn local_sars_cov2_archive_1000_full_genomes_profiles_build_phases() {
    if std::env::var_os("DAG_RUST_RUN_PROFILE_TESTS").is_none() {
        return;
    }
    if !Path::new(SARS_COV2_ARCHIVE).exists() {
        return;
    }

    let record_limit = profiling_record_limit();
    let load_start = Instant::now();
    let records = load_sars_cov2_full_sequences(record_limit);
    let load_time = load_start.elapsed();
    assert_eq!(
        records.len(),
        record_limit,
        "expected {record_limit} full SARS-CoV-2 records"
    );

    let alphabet = BuiltinAlphabet::dna_exact();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let mut config = BuildConfig::new(31);
    config.min_initial_match_ratio = Some(SimilarityThreshold::from_basis_points(100).unwrap());
    config.topology_update_strategy = topology_strategy_from_env();
    config.source_storage_strategy = source_storage_strategy_from_env();
    config.edge_index_strategy = edge_index_strategy_from_env();
    config.collect_topology_counters = topology_counters_from_env();
    config.topology_rebuild_queue_threshold = topology_rebuild_queue_threshold_from_env();
    let mut builder = FtoDagBuilder::new(config);
    let mut timings = BuildTimingBreakdown::default();
    let mut encode_time = Duration::default();
    let build_start = Instant::now();

    for (index, (id, sequence)) in records.iter().enumerate() {
        let start = Instant::now();
        let encoded = EncodedSequence::encode(SequenceRecord::new(id, sequence.clone()), &alphabet)
            .expect("full SARS-CoV-2 record encodes");
        encode_time += start.elapsed();

        let profiled = builder
            .add_sequence_from_encoded_profiled(
                SequenceId::try_from(index).unwrap(),
                &encoded,
                &encoder,
            )
            .expect("profiled full SARS-CoV-2 record integrates or rejects");
        timings.add_assign(profiled.timing);

        if (index + 1) % 100 == 0 || index + 1 == records.len() {
            print_profile_progress(
                index + 1,
                &builder,
                encode_time,
                timings,
                build_start.elapsed(),
            );
        }
    }
    let topology_start = Instant::now();
    DagTopology::try_from_graph(builder.graph()).expect("profiled DAG remains acyclic");
    let final_topology_time = topology_start.elapsed();
    let report = builder.report();

    assert_eq!(report.attempted_sequences, record_limit);
    assert_eq!(
        report.integrated_sequences.len() + report.rejected_sequences.len(),
        record_limit
    );
    let memory = MemoryUsage::current();
    println!(
        "profile_sars_cov2_full_build_summary strategy={:?} source_storage={:?} edge_index={:?} records={} integrated={} rejected={} nodes={} edges={} sources={} load_secs={:.3} encode_secs={:.3} initialize_secs={:.3} initial_match_secs={:.3} topology_secs={:.3} occurrences_secs={:.3} candidate_collection_secs={:.3} anchor_selection_secs={:.3} block_planning_secs={:.3} mutation_secs={:.3} report_update_secs={:.3} builder_total_secs={:.3} final_topology_secs={:.3} wall_secs={:.3} topology_full_rebuilds={} topology_fallback_rebuilds={} topology_new_nodes={} topology_inserted_edges={} topology_new_target_edges={} topology_existing_target_edges={} topology_safe_forward_edges={} topology_forward_relax_attempts={} topology_forward_updates={} topology_forward_queue_pops={} topology_forward_parent_scans={} topology_reverse_updates={} topology_reverse_queue_pops={} topology_reverse_child_scans={} rss_mib={} peak_rss_mib={}",
        config.topology_update_strategy,
        config.source_storage_strategy,
        config.edge_index_strategy,
        records.len(),
        report.integrated_sequences.len(),
        report.rejected_sequences.len(),
        builder.graph().node_count(),
        builder.graph().edge_count(),
        report.total_source_records_added,
        seconds(load_time),
        seconds(encode_time),
        seconds(timings.initialize),
        seconds(timings.initial_match),
        seconds(timings.topology),
        seconds(timings.fragment_occurrences),
        seconds(timings.candidate_collection),
        seconds(timings.anchor_selection),
        seconds(timings.block_planning),
        seconds(timings.mutation),
        seconds(timings.report_update),
        seconds(timings.total),
        seconds(final_topology_time),
        seconds(build_start.elapsed()),
        report.topology_counters.full_rebuilds,
        report.topology_counters.full_rebuild_fallbacks,
        report.topology_counters.new_nodes,
        report.topology_counters.inserted_edges,
        report.topology_counters.inserted_edges_to_new_targets,
        report.topology_counters.inserted_edges_to_existing_targets,
        report.topology_counters.safe_forward_edges,
        report.topology_counters.forward_relax_attempts,
        report.topology_counters.forward_coordinate_updates,
        report.topology_counters.forward_queue_pops,
        report.topology_counters.forward_parent_scans,
        report.topology_counters.reverse_coordinate_updates,
        report.topology_counters.reverse_queue_pops,
        report.topology_counters.reverse_child_scans,
        memory.rss_mib(),
        memory.peak_rss_mib(),
    );
    assert!(
        builder.graph().validate().is_valid(),
        "profiled DAG passes full graph validation"
    );
}

#[test]
#[ignore = "compare local SARS-CoV-2 graph structure across source-storage modes"]
fn local_sars_cov2_archive_1000_full_genomes_builds_identical_graphs_across_storage_modes() {
    if std::env::var_os("DAG_RUST_RUN_PROFILE_TESTS").is_none() {
        return;
    }
    if !Path::new(SARS_COV2_ARCHIVE).exists() {
        return;
    }

    let record_limit = profiling_record_limit();
    let records = load_sars_cov2_full_sequences(record_limit);
    assert_eq!(records.len(), record_limit);

    let trace_graph = build_profile_graph(
        &records,
        SourceStorageStrategy::TracePaths,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let packed_graph = build_profile_graph(
        &records,
        SourceStorageStrategy::Packed32,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let count_graph = build_profile_graph(
        &records,
        SourceStorageStrategy::CountOnly,
        EdgeIndexStrategy::LowDegreeHybrid,
    );

    assert_graph_structure_eq(&trace_graph, &packed_graph);
    assert_graph_structure_eq(&trace_graph, &count_graph);
    assert!(trace_graph.validate().is_valid());
    assert!(packed_graph.validate().is_valid());
    assert!(count_graph.validate().is_valid());
}

fn heavy_integration_record_limit() -> usize {
    std::env::var("DAG_RUST_FULL_INTEGRATION_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(SARS_COV2_HEAVY_SEQUENCE_LIMIT)
}

fn profiling_record_limit() -> usize {
    std::env::var("DAG_RUST_PROFILE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(1_000)
}

fn topology_strategy_from_env() -> TopologyUpdateStrategy {
    match std::env::var("DAG_RUST_TOPOLOGY_STRATEGY") {
        Ok(value) if value.eq_ignore_ascii_case("incremental") => {
            TopologyUpdateStrategy::IncrementalAffectedRegion
        }
        Ok(value) if value.eq_ignore_ascii_case("incremental_affected_region") => {
            TopologyUpdateStrategy::IncrementalAffectedRegion
        }
        Ok(value) if value.eq_ignore_ascii_case("forward_only") => {
            TopologyUpdateStrategy::IncrementalForwardOnly
        }
        Ok(value) if value.eq_ignore_ascii_case("incremental_forward_only") => {
            TopologyUpdateStrategy::IncrementalForwardOnly
        }
        Ok(value) if value.eq_ignore_ascii_case("forward_rescan") => {
            TopologyUpdateStrategy::IncrementalForwardRescan
        }
        Ok(value) if value.eq_ignore_ascii_case("incremental_forward_rescan") => {
            TopologyUpdateStrategy::IncrementalForwardRescan
        }
        Ok(value) if value.eq_ignore_ascii_case("forward_relaxation") => {
            TopologyUpdateStrategy::IncrementalForwardRelaxation
        }
        Ok(value) if value.eq_ignore_ascii_case("incremental_forward_relaxation") => {
            TopologyUpdateStrategy::IncrementalForwardRelaxation
        }
        _ => TopologyUpdateStrategy::FullRebuild,
    }
}

fn topology_counters_from_env() -> bool {
    std::env::var("DAG_RUST_TOPOLOGY_COUNTERS")
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

fn topology_rebuild_queue_threshold_from_env() -> Option<usize> {
    match std::env::var("DAG_RUST_TOPOLOGY_REBUILD_QUEUE_THRESHOLD") {
        Ok(value) if value.eq_ignore_ascii_case("off") || value == "0" => None,
        Ok(value) => value.parse::<usize>().ok(),
        Err(_) => None,
    }
}

fn source_storage_strategy_from_env() -> SourceStorageStrategy {
    match std::env::var("DAG_RUST_SOURCE_STORAGE") {
        Ok(value) if value.eq_ignore_ascii_case("packed32") => SourceStorageStrategy::Packed32,
        Ok(value) if value.eq_ignore_ascii_case("packed") => SourceStorageStrategy::Packed32,
        Ok(value) if value.eq_ignore_ascii_case("trace_paths") => SourceStorageStrategy::TracePaths,
        Ok(value) if value.eq_ignore_ascii_case("tracepaths") => SourceStorageStrategy::TracePaths,
        Ok(value) if value.eq_ignore_ascii_case("paths") => SourceStorageStrategy::TracePaths,
        Ok(value) if value.eq_ignore_ascii_case("count_only") => SourceStorageStrategy::CountOnly,
        Ok(value) if value.eq_ignore_ascii_case("countonly") => SourceStorageStrategy::CountOnly,
        Ok(value) if value.eq_ignore_ascii_case("counts") => SourceStorageStrategy::CountOnly,
        _ => SourceStorageStrategy::FullRecords,
    }
}

fn edge_index_strategy_from_env() -> EdgeIndexStrategy {
    match std::env::var("DAG_RUST_EDGE_INDEX") {
        Ok(value) if value.eq_ignore_ascii_case("hybrid") => EdgeIndexStrategy::LowDegreeHybrid,
        Ok(value) if value.eq_ignore_ascii_case("low_degree") => EdgeIndexStrategy::LowDegreeHybrid,
        Ok(value) if value.eq_ignore_ascii_case("low_degree_hybrid") => {
            EdgeIndexStrategy::LowDegreeHybrid
        }
        _ => EdgeIndexStrategy::GlobalHash,
    }
}

fn print_profile_progress(
    processed: usize,
    builder: &FtoDagBuilder,
    encode_time: Duration,
    timings: BuildTimingBreakdown,
    wall_time: Duration,
) {
    let report = builder.report();
    let memory = MemoryUsage::current();
    println!(
        "profile_sars_cov2_full_build_progress processed={} integrated={} rejected={} nodes={} edges={} sources={} encode_secs={:.3} initial_match_secs={:.3} topology_secs={:.3} occurrences_secs={:.3} candidate_collection_secs={:.3} anchor_selection_secs={:.3} mutation_secs={:.3} builder_total_secs={:.3} wall_secs={:.3} topology_safe_forward_edges={} topology_forward_relax_attempts={} topology_forward_updates={} topology_forward_parent_scans={} rss_mib={} peak_rss_mib={}",
        processed,
        report.integrated_sequences.len(),
        report.rejected_sequences.len(),
        builder.graph().node_count(),
        builder.graph().edge_count(),
        report.total_source_records_added,
        seconds(encode_time),
        seconds(timings.initial_match),
        seconds(timings.topology),
        seconds(timings.fragment_occurrences),
        seconds(timings.candidate_collection),
        seconds(timings.anchor_selection),
        seconds(timings.mutation),
        seconds(timings.total),
        seconds(wall_time),
        report.topology_counters.safe_forward_edges,
        report.topology_counters.forward_relax_attempts,
        report.topology_counters.forward_coordinate_updates,
        report.topology_counters.forward_parent_scans,
        memory.rss_mib(),
        memory.peak_rss_mib(),
    );
}

fn build_profile_graph(
    records: &[(String, String)],
    source_storage_strategy: SourceStorageStrategy,
    edge_index_strategy: EdgeIndexStrategy,
) -> FtoDag {
    let alphabet = BuiltinAlphabet::dna_exact();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let mut config = BuildConfig::new(31);
    config.min_initial_match_ratio = Some(SimilarityThreshold::from_basis_points(100).unwrap());
    config.topology_update_strategy = TopologyUpdateStrategy::IncrementalForwardOnly;
    config.source_storage_strategy = source_storage_strategy;
    config.edge_index_strategy = edge_index_strategy;
    let mut builder = FtoDagBuilder::new(config);
    for (index, (id, sequence)) in records.iter().enumerate() {
        let encoded = EncodedSequence::encode(SequenceRecord::new(id, sequence.clone()), &alphabet)
            .expect("full SARS-CoV-2 record encodes");
        builder
            .add_sequence_from_encoded_profiled(
                SequenceId::try_from(index).unwrap(),
                &encoded,
                &encoder,
            )
            .expect("record integrates or rejects");
    }
    builder.into_graph()
}

fn assert_graph_structure_eq(left: &FtoDag, right: &FtoDag) {
    assert_eq!(left.node_count(), right.node_count(), "node count differs");
    assert_eq!(left.edge_count(), right.edge_count(), "edge count differs");
    assert_eq!(left.endpoints(), right.endpoints(), "endpoints differ");
    for (index, (left_node, right_node)) in left.nodes().iter().zip(right.nodes()).enumerate() {
        assert_eq!(left_node.id, right_node.id, "node {index} id differs");
        assert_eq!(
            left_node.fragment, right_node.fragment,
            "node {index} fragment differs"
        );
        assert_eq!(left_node.kind, right_node.kind, "node {index} kind differs");
        assert_eq!(
            left_node.weight, right_node.weight,
            "node {index} weight differs"
        );
        assert_eq!(
            left_node.flags, right_node.flags,
            "node {index} flags differ"
        );
        assert_eq!(
            left.source_record_count(left_node.id).unwrap(),
            right.source_record_count(right_node.id).unwrap(),
            "node {index} source count differs"
        );
    }
    assert_eq!(left.edges(), right.edges(), "weighted edge list differs");
}

fn seconds(duration: Duration) -> f64 {
    duration.as_secs_f64()
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

    fn rss_mib(self) -> String {
        kib_to_mib_string(self.rss_kib)
    }

    fn peak_rss_mib(self) -> String {
        kib_to_mib_string(self.peak_rss_kib)
    }
}

fn parse_status_kib(value: &str) -> Option<u64> {
    value.split_whitespace().next()?.parse().ok()
}

fn kib_to_mib_string(kib: Option<u64>) -> String {
    kib.map(|kib| format!("{:.1}", kib as f64 / 1024.0))
        .unwrap_or_else(|| "NA".to_string())
}

fn load_sars_cov2_snippets(record_limit: usize, snippet_len: usize) -> Vec<(String, String)> {
    let script = format!(
        "tar -xOf {SARS_COV2_ARCHIVE} msa_0211/msa_0211.fasta | awk 'BEGIN{{c=0}} /^>/{{c++; if(c>{record_limit}) exit; print; next}} c>0{{print}}'"
    );
    let output = Command::new("sh")
        .arg("-c")
        .arg(script)
        .output()
        .expect("tar command runs");
    assert!(output.status.success(), "failed to read local archive");

    let fasta = String::from_utf8(output.stdout).expect("archive contains UTF-8 FASTA");
    fasta_snippets(&fasta, snippet_len)
}

fn load_sars_cov2_full_sequences(record_limit: usize) -> Vec<(String, String)> {
    let script = format!(
        "tar -xOf {SARS_COV2_ARCHIVE} msa_0211/msa_0211.fasta | awk 'BEGIN{{c=0}} /^>/{{c++; if(c>{record_limit}) exit; print; next}} c>0{{print}}'"
    );
    let output = Command::new("sh")
        .arg("-c")
        .arg(script)
        .output()
        .expect("tar command runs");
    assert!(output.status.success(), "failed to read local archive");

    let fasta = String::from_utf8(output.stdout).expect("archive contains UTF-8 FASTA");
    fasta_sequences(&fasta)
}

fn fasta_snippets(fasta: &str, max_len: usize) -> Vec<(String, String)> {
    let mut records = Vec::new();
    let mut current_id = None;
    let mut current_sequence = String::new();
    for line in fasta.lines() {
        if let Some(header) = line.strip_prefix('>') {
            if let Some(id) = current_id.take() {
                push_snippet(&mut records, id, &current_sequence, max_len);
                current_sequence.clear();
            }
            current_id = Some(header.to_string());
        } else {
            current_sequence.push_str(line.trim());
        }
    }
    if let Some(id) = current_id {
        push_snippet(&mut records, id, &current_sequence, max_len);
    }
    records
}

fn fasta_sequences(fasta: &str) -> Vec<(String, String)> {
    let mut records = Vec::new();
    let mut current_id = None;
    let mut current_sequence = String::new();
    for line in fasta.lines() {
        if let Some(header) = line.strip_prefix('>') {
            if let Some(id) = current_id.take() {
                push_full_sequence(&mut records, id, &current_sequence);
                current_sequence.clear();
            }
            current_id = Some(header.to_string());
        } else {
            current_sequence.push_str(line.trim());
        }
    }
    if let Some(id) = current_id {
        push_full_sequence(&mut records, id, &current_sequence);
    }
    records
}

fn push_snippet(records: &mut Vec<(String, String)>, id: String, sequence: &str, max_len: usize) {
    let snippet = sequence
        .chars()
        .filter(|symbol| matches!(symbol.to_ascii_uppercase(), 'A' | 'C' | 'G' | 'T' | 'N'))
        .map(|symbol| symbol.to_ascii_uppercase())
        .take(max_len)
        .collect::<String>();
    if snippet.len() >= 100 {
        records.push((id, snippet));
    }
}

fn push_full_sequence(records: &mut Vec<(String, String)>, id: String, sequence: &str) {
    let full_sequence = sequence
        .chars()
        .filter(|symbol| matches!(symbol.to_ascii_uppercase(), 'A' | 'C' | 'G' | 'T' | 'N'))
        .map(|symbol| symbol.to_ascii_uppercase())
        .collect::<String>();
    if full_sequence.len() >= 10_000 {
        records.push((id, full_sequence));
    }
}
