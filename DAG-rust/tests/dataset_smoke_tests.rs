use dag_rust::prelude::*;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::{Duration, Instant};

const SARS_COV2_ARCHIVE: &str = "/disk2/tian/hCoV-19_msa_0211.tar.xz";
const SARS_COV2_SEQUENCE_LIMIT: usize = 128;
const SARS_COV2_SNIPPET_LEN: usize = 2_000;
const SARS_COV2_BUILD_SEQUENCE_LIMIT: usize = 32;
const SARS_COV2_BUILD_SNIPPET_LEN: usize = 1_000;
const SARS_COV2_SHARED_SEQUENCE_LIMIT: usize = 4_000;
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
    config.provenance_storage_strategy = provenance_storage_strategy_from_env();
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
    config.provenance_storage_strategy = provenance_storage_strategy_from_env();
    config.edge_index_strategy = edge_index_strategy_from_env();
    let mut builder = FtoDagBuilder::new(config);
    let mut input = VecSequenceInput::new(
        snippets
            .iter()
            .map(|(id, sequence)| SequenceRecord::new(id, sequence.clone()))
            .collect(),
    );

    let report = builder
        .build_from_input(&mut input, &alphabet, &encoder)
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
    assert!(report.total_provenance_records_added > SARS_COV2_BUILD_SNIPPET_LEN - 31);
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
                report.total_provenance_records_added,
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
        report.total_provenance_records_added >= report.integrated_sequences.len() * 25_000,
        "expected full-genome integration to add many provenance records"
    );
    let memory = MemoryUsage::current();
    println!(
        "heavy_sars_cov2_full_build records={} integrated={} rejected={} nodes={} edges={} sources={} load_secs={:.2} build_secs={:.2} topology_secs={:.2} rss_mib={} peak_rss_mib={}",
        records.len(),
        report.integrated_sequences.len(),
        report.rejected_sequences.len(),
        builder.graph().node_count(),
        builder.graph().edge_count(),
        report.total_provenance_records_added,
        load_start.elapsed().as_secs_f64(),
        build_start.elapsed().as_secs_f64(),
        topology_start.elapsed().as_secs_f64(),
        memory.rss_mib(),
        memory.peak_rss_mib(),
    );
}

#[test]
#[ignore = "builds and saves local SARS-CoV-2 TracePaths graphs from a shared 4000-genome set"]
fn local_sars_cov2_archive_4000_subsets_build_and_save_graphs() {
    if std::env::var_os("DAG_RUST_RUN_SAVED_BUILD_TESTS").is_none() {
        return;
    }
    if !Path::new(SARS_COV2_ARCHIVE).exists() {
        return;
    }

    let output_dir = saved_build_output_dir();
    fs::create_dir_all(&output_dir).expect("saved-build output directory is created");
    let records = load_sars_cov2_full_sequences(SARS_COV2_SHARED_SEQUENCE_LIMIT);
    assert_eq!(
        records.len(),
        SARS_COV2_SHARED_SEQUENCE_LIMIT,
        "expected 4000 full SARS-CoV-2 records"
    );
    write_saved_sequence_manifest(&output_dir, &records);

    let mut summary = fs::File::create(output_dir.join("summary.tsv"))
        .expect("saved-build summary file is created");
    writeln!(
        summary,
        "label\trecords\tintegrated\trejected\tnodes\tedges\tprovenance_records\tcompression_ratio\tload_secs\tbuild_secs\ttopology_secs\tgraph_path"
    )
    .unwrap();

    let storage = NativeGraphStorage;
    for (label, start, end) in saved_build_partitions() {
        let result =
            build_and_save_trace_path_graph(label, &records[start..end], &output_dir, &storage);
        writeln!(
            summary,
            "{label}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.3}\t{:.3}\t{:.3}\t{}",
            result.records,
            result.integrated,
            result.rejected,
            result.nodes,
            result.edges,
            result.provenance_records,
            result.compression_ratio,
            result.load_secs,
            result.build_secs,
            result.topology_secs,
            result.graph_path.display()
        )
        .unwrap();
    }
}

#[test]
#[ignore = "merges saved local SARS-CoV-2 TracePaths graphs and compares against direct-build saved targets"]
fn local_saved_sars_cov2_trace_path_graphs_merge_to_match_saved_results() {
    if std::env::var_os("DAG_RUST_RUN_SAVED_MERGE_TESTS").is_none() {
        return;
    }

    let output_dir = saved_build_output_dir();
    let storage = NativeGraphStorage;
    let graph_1000_a = storage
        .load_graph(&saved_graph_path(&output_dir, "1000_a"))
        .expect("1000_a graph loads");
    let graph_1000_b = storage
        .load_graph(&saved_graph_path(&output_dir, "1000_b"))
        .expect("1000_b graph loads");
    let expected_2000_ab = storage
        .load_graph(&saved_graph_path(&output_dir, "2000_ab"))
        .expect("2000_ab graph loads");
    let graph_2000_cd = storage
        .load_graph(&saved_graph_path(&output_dir, "2000_cd"))
        .expect("2000_cd graph loads");
    let expected_4000_abcd = storage
        .load_graph(&saved_graph_path(&output_dir, "4000_abcd"))
        .expect("4000_abcd graph loads");

    let merge_config = MergeConfig {
        min_initial_anchor_ratio: Some(SimilarityThreshold::from_basis_points(100).unwrap()),
        ..MergeConfig::default()
    };

    let merged_2000_ab = merge_graphs(graph_1000_a, graph_1000_b, merge_config).unwrap();
    assert_graph_structure_eq(&merged_2000_ab, &expected_2000_ab);
    assert!(merged_2000_ab.validate().is_valid());

    let merged_4000_abcd = merge_graphs(merged_2000_ab, graph_2000_cd, merge_config).unwrap();
    assert_graph_structure_eq(&merged_4000_abcd, &expected_4000_abcd);
    assert!(merged_4000_abcd.validate().is_valid());
}

#[test]
#[ignore = "reports saved-graph merge timings against saved scratch-build graphs"]
fn local_saved_sars_cov2_trace_path_merge_comparison_reports() {
    if std::env::var_os("DAG_RUST_RUN_SAVED_MERGE_TESTS").is_none() {
        return;
    }

    let output_dir = saved_build_output_dir();
    let storage = NativeGraphStorage;
    let summary = load_saved_build_summary(&output_dir.join("summary.tsv"));

    report_saved_merge_case(
        &storage,
        &output_dir,
        "1000_a",
        "1000_b",
        "2000_ab",
        &summary,
    );
    report_saved_merge_case(
        &storage,
        &output_dir,
        "1000_c",
        "1000_d",
        "2000_cd",
        &summary,
    );
    report_saved_merge_case(
        &storage,
        &output_dir,
        "2000_ab",
        "2000_cd",
        "4000_abcd",
        &summary,
    );
}

#[test]
#[ignore = "reports optimized hierarchy merges from saved 250/500/1000 TracePaths building blocks"]
fn local_saved_sars_cov2_trace_path_hierarchy_merge_comparison_reports() {
    if std::env::var_os("DAG_RUST_RUN_SAVED_MERGE_TESTS").is_none() {
        return;
    }

    let output_dir = saved_build_output_dir();
    let storage = NativeGraphStorage;
    let summary = load_saved_build_summary(&output_dir.join("summary.tsv"));
    let mut rows = Vec::new();

    rows.extend(report_saved_merge_hierarchy(
        &storage,
        &output_dir,
        "250",
        &[
            "250_a1", "250_a2", "250_a3", "250_a4", "250_b1", "250_b2", "250_b3", "250_b4",
            "250_c1", "250_c2", "250_c3", "250_c4", "250_d1", "250_d2", "250_d3", "250_d4",
        ],
        &[
            vec![
                "500_a12", "500_a34", "500_b12", "500_b34", "500_c12", "500_c34", "500_d12",
                "500_d34",
            ],
            vec!["1000_a", "1000_b", "1000_c", "1000_d"],
            vec!["2000_ab", "2000_cd"],
            vec!["4000_abcd"],
        ],
        &summary,
    ));
    rows.extend(report_saved_merge_hierarchy(
        &storage,
        &output_dir,
        "500",
        &[
            "500_a12", "500_a34", "500_b12", "500_b34", "500_c12", "500_c34", "500_d12", "500_d34",
        ],
        &[
            vec!["1000_a", "1000_b", "1000_c", "1000_d"],
            vec!["2000_ab", "2000_cd"],
            vec!["4000_abcd"],
        ],
        &summary,
    ));
    rows.extend(report_saved_merge_hierarchy(
        &storage,
        &output_dir,
        "1000",
        &["1000_a", "1000_b", "1000_c", "1000_d"],
        &[vec!["2000_ab", "2000_cd"], vec!["4000_abcd"]],
        &summary,
    ));

    write_saved_merge_summary(&output_dir.join("merge_summary.tsv"), &rows);
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
    config.provenance_storage_strategy = provenance_storage_strategy_from_env();
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
        "profile_sars_cov2_full_build_summary strategy={:?} provenance_storage={:?} edge_index={:?} records={} integrated={} rejected={} nodes={} edges={} provenance_records={} load_secs={:.3} encode_secs={:.3} initialize_secs={:.3} initial_match_secs={:.3} topology_secs={:.3} occurrences_secs={:.3} candidate_collection_secs={:.3} anchor_selection_secs={:.3} block_planning_secs={:.3} mutation_secs={:.3} report_update_secs={:.3} builder_total_secs={:.3} final_topology_secs={:.3} wall_secs={:.3} topology_full_rebuilds={} topology_fallback_rebuilds={} topology_new_nodes={} topology_inserted_edges={} topology_new_child_edges={} topology_existing_child_edges={} topology_safe_forward_edges={} topology_forward_relax_attempts={} topology_forward_updates={} topology_forward_queue_pops={} topology_forward_parent_scans={} topology_reverse_updates={} topology_reverse_queue_pops={} topology_reverse_child_scans={} rss_mib={} peak_rss_mib={}",
        config.topology_update_strategy,
        config.provenance_storage_strategy,
        config.edge_index_strategy,
        records.len(),
        report.integrated_sequences.len(),
        report.rejected_sequences.len(),
        builder.graph().node_count(),
        builder.graph().edge_count(),
        report.total_provenance_records_added,
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
        report.topology_counters.inserted_edges_to_new_children,
        report.topology_counters.inserted_edges_to_existing_children,
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
#[ignore = "compare local SARS-CoV-2 graph structure across provenance-storage modes"]
fn local_sars_cov2_archive_1000_full_genomes_builds_identical_graphs_across_provenance_modes() {
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
        ProvenanceStorageStrategy::TracePaths,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let packed_graph = build_profile_graph(
        &records,
        ProvenanceStorageStrategy::Packed32,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let count_graph = build_profile_graph(
        &records,
        ProvenanceStorageStrategy::CountOnly,
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

fn saved_build_output_dir() -> std::path::PathBuf {
    std::env::var("DAG_RUST_SAVED_BUILD_OUTPUT_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("target/local-sars-cov2-shared4000"))
}

fn saved_graph_path(output_dir: &Path, label: &str) -> std::path::PathBuf {
    output_dir.join(format!("{label}.dagrs"))
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

fn provenance_storage_strategy_from_env() -> ProvenanceStorageStrategy {
    let value = std::env::var("DAG_RUST_PROVENANCE_STORAGE")
        .or_else(|_| std::env::var("DAG_RUST_SOURCE_STORAGE"));
    match value {
        Ok(value) if value.eq_ignore_ascii_case("packed32") => ProvenanceStorageStrategy::Packed32,
        Ok(value) if value.eq_ignore_ascii_case("packed") => ProvenanceStorageStrategy::Packed32,
        Ok(value) if value.eq_ignore_ascii_case("trace_paths") => {
            ProvenanceStorageStrategy::TracePaths
        }
        Ok(value) if value.eq_ignore_ascii_case("tracepaths") => {
            ProvenanceStorageStrategy::TracePaths
        }
        Ok(value) if value.eq_ignore_ascii_case("paths") => ProvenanceStorageStrategy::TracePaths,
        Ok(value) if value.eq_ignore_ascii_case("count_only") => {
            ProvenanceStorageStrategy::CountOnly
        }
        Ok(value) if value.eq_ignore_ascii_case("countonly") => {
            ProvenanceStorageStrategy::CountOnly
        }
        Ok(value) if value.eq_ignore_ascii_case("counts") => ProvenanceStorageStrategy::CountOnly,
        _ => ProvenanceStorageStrategy::FullRecords,
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
        "profile_sars_cov2_full_build_progress processed={} integrated={} rejected={} nodes={} edges={} provenance_records={} encode_secs={:.3} initial_match_secs={:.3} topology_secs={:.3} occurrences_secs={:.3} candidate_collection_secs={:.3} anchor_selection_secs={:.3} mutation_secs={:.3} builder_total_secs={:.3} wall_secs={:.3} topology_safe_forward_edges={} topology_forward_relax_attempts={} topology_forward_updates={} topology_forward_parent_scans={} rss_mib={} peak_rss_mib={}",
        processed,
        report.integrated_sequences.len(),
        report.rejected_sequences.len(),
        builder.graph().node_count(),
        builder.graph().edge_count(),
        report.total_provenance_records_added,
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
    provenance_storage_strategy: ProvenanceStorageStrategy,
    edge_index_strategy: EdgeIndexStrategy,
) -> FtoDag {
    let alphabet = BuiltinAlphabet::dna_exact();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let mut config = BuildConfig::new(31);
    config.min_initial_match_ratio = Some(SimilarityThreshold::from_basis_points(100).unwrap());
    config.topology_update_strategy = TopologyUpdateStrategy::IncrementalForwardOnly;
    config.provenance_storage_strategy = provenance_storage_strategy;
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

#[derive(Clone, Debug)]
struct SavedBuildResult {
    graph_path: std::path::PathBuf,
    records: usize,
    integrated: usize,
    rejected: usize,
    nodes: usize,
    edges: usize,
    provenance_records: usize,
    compression_ratio: f64,
    load_secs: f64,
    build_secs: f64,
    topology_secs: f64,
}

#[derive(Clone, Debug)]
struct SavedBuildSummaryRow {
    build_secs: f64,
    compression_ratio: f64,
    nodes: usize,
    edges: usize,
    provenance_records: usize,
}

#[derive(Clone, Debug)]
struct SavedMergeSummaryRow {
    scheme: String,
    stage: usize,
    left_label: String,
    right_label: String,
    expected_label: String,
    merge_secs: f64,
    cumulative_merge_secs: f64,
    scratch_build_secs: f64,
    speedup_vs_scratch: f64,
    nodes: usize,
    edges: usize,
    provenance_records: usize,
    compression_ratio: f64,
}

fn build_and_save_trace_path_graph(
    label: &str,
    records: &[(String, String)],
    output_dir: &Path,
    storage: &impl GraphStorage,
) -> SavedBuildResult {
    let alphabet = BuiltinAlphabet::dna_exact();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let config = optimized_saved_trace_path_build_config();
    let mut builder = FtoDagBuilder::new(config);

    let load_secs = 0.0;
    let build_start = Instant::now();
    for (index, (id, sequence)) in records.iter().enumerate() {
        let encoded = EncodedSequence::encode(SequenceRecord::new(id, sequence.clone()), &alphabet)
            .expect("shared-record sequence encodes");
        builder
            .add_sequence_from_encoded(SequenceId::try_from(index).unwrap(), &encoded, &encoder)
            .expect("shared-record sequence integrates or rejects");
        if (index + 1) % 100 == 0 || index + 1 == records.len() {
            let report = builder.report();
            let memory = MemoryUsage::current();
            println!(
                "saved_trace_path_build_progress label={} processed={} integrated={} rejected={} nodes={} edges={} provenance_records={} elapsed_secs={:.2} rss_mib={} peak_rss_mib={}",
                label,
                index + 1,
                report.integrated_sequences.len(),
                report.rejected_sequences.len(),
                builder.graph().node_count(),
                builder.graph().edge_count(),
                report.total_provenance_records_added,
                build_start.elapsed().as_secs_f64(),
                memory.rss_mib(),
                memory.peak_rss_mib(),
            );
        }
    }
    let report = builder.report().clone();
    let graph = builder.into_graph();
    let topology_start = Instant::now();
    DagTopology::try_from_graph(&graph).expect("saved TracePaths graph remains acyclic");
    let topology_secs = topology_start.elapsed().as_secs_f64();
    assert!(graph.validate().is_valid());

    let graph_path = saved_graph_path(output_dir, label);
    storage
        .save_graph(&graph, &graph_path, StorageConfig::default())
        .expect("graph saves");
    let compression_ratio =
        report.total_provenance_records_added as f64 / graph.node_count() as f64;
    let memory = MemoryUsage::current();
    println!(
        "saved_trace_path_build_summary label={} records={} integrated={} rejected={} nodes={} edges={} provenance_records={} compression_ratio={:.6} build_secs={:.2} topology_secs={:.2} rss_mib={} peak_rss_mib={} graph_path={}",
        label,
        records.len(),
        report.integrated_sequences.len(),
        report.rejected_sequences.len(),
        graph.node_count(),
        graph.edge_count(),
        report.total_provenance_records_added,
        compression_ratio,
        build_start.elapsed().as_secs_f64(),
        topology_secs,
        memory.rss_mib(),
        memory.peak_rss_mib(),
        graph_path.display(),
    );

    SavedBuildResult {
        graph_path,
        records: records.len(),
        integrated: report.integrated_sequences.len(),
        rejected: report.rejected_sequences.len(),
        nodes: graph.node_count(),
        edges: graph.edge_count(),
        provenance_records: report.total_provenance_records_added,
        compression_ratio,
        load_secs,
        build_secs: build_start.elapsed().as_secs_f64(),
        topology_secs,
    }
}

fn saved_build_partitions() -> Vec<(&'static str, usize, usize)> {
    vec![
        ("250_a1", 0, 250),
        ("250_a2", 250, 500),
        ("250_a3", 500, 750),
        ("250_a4", 750, 1_000),
        ("250_b1", 1_000, 1_250),
        ("250_b2", 1_250, 1_500),
        ("250_b3", 1_500, 1_750),
        ("250_b4", 1_750, 2_000),
        ("250_c1", 2_000, 2_250),
        ("250_c2", 2_250, 2_500),
        ("250_c3", 2_500, 2_750),
        ("250_c4", 2_750, 3_000),
        ("250_d1", 3_000, 3_250),
        ("250_d2", 3_250, 3_500),
        ("250_d3", 3_500, 3_750),
        ("250_d4", 3_750, 4_000),
        ("500_a12", 0, 500),
        ("500_a34", 500, 1_000),
        ("500_b12", 1_000, 1_500),
        ("500_b34", 1_500, 2_000),
        ("500_c12", 2_000, 2_500),
        ("500_c34", 2_500, 3_000),
        ("500_d12", 3_000, 3_500),
        ("500_d34", 3_500, 4_000),
        ("1000_a", 0, 1_000),
        ("1000_b", 1_000, 2_000),
        ("1000_c", 2_000, 3_000),
        ("1000_d", 3_000, 4_000),
        ("2000_ab", 0, 2_000),
        ("2000_cd", 2_000, 4_000),
        ("4000_abcd", 0, 4_000),
    ]
}

fn optimized_saved_trace_path_build_config() -> BuildConfig {
    let mut config = BuildConfig::new(31);
    config.min_initial_match_ratio = Some(SimilarityThreshold::from_basis_points(100).unwrap());
    config.topology_update_strategy = TopologyUpdateStrategy::IncrementalForwardRelaxation;
    config.provenance_storage_strategy = ProvenanceStorageStrategy::TracePaths;
    config.edge_index_strategy = EdgeIndexStrategy::LowDegreeHybrid;
    config
}

fn write_saved_sequence_manifest(output_dir: &Path, records: &[(String, String)]) {
    let mut manifest = fs::File::create(output_dir.join("sequence_ids.tsv"))
        .expect("sequence-id manifest is created");
    writeln!(manifest, "index\tsequence_id\tlength").unwrap();
    for (index, (id, sequence)) in records.iter().enumerate() {
        writeln!(manifest, "{index}\t{id}\t{}", sequence.len()).unwrap();
    }
}

fn load_saved_build_summary(
    path: &Path,
) -> std::collections::HashMap<String, SavedBuildSummaryRow> {
    let content = fs::read_to_string(path).expect("saved-build summary is readable");
    let mut rows = std::collections::HashMap::new();
    for line in content.lines().skip(1) {
        if line.trim().is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        assert_eq!(fields.len(), 12, "unexpected summary.tsv column count");
        rows.insert(
            fields[0].to_string(),
            SavedBuildSummaryRow {
                nodes: fields[4].parse().expect("nodes parse"),
                edges: fields[5].parse().expect("edges parse"),
                provenance_records: fields[6].parse().expect("provenance parse"),
                compression_ratio: fields[7].parse().expect("compression ratio parse"),
                build_secs: fields[9].parse().expect("build seconds parse"),
            },
        );
    }
    rows
}

fn total_provenance_records(graph: &FtoDag) -> usize {
    graph
        .nodes()
        .iter()
        .map(|node| node.weight.raw() as usize)
        .sum()
}

fn report_saved_merge_case(
    storage: &impl GraphStorage,
    output_dir: &Path,
    left_label: &str,
    right_label: &str,
    expected_label: &str,
    summary: &std::collections::HashMap<String, SavedBuildSummaryRow>,
) {
    let left = storage
        .load_graph(&saved_graph_path(output_dir, left_label))
        .unwrap_or_else(|err| panic!("failed to load {left_label}: {err}"));
    let right = storage
        .load_graph(&saved_graph_path(output_dir, right_label))
        .unwrap_or_else(|err| panic!("failed to load {right_label}: {err}"));
    let expected = storage
        .load_graph(&saved_graph_path(output_dir, expected_label))
        .unwrap_or_else(|err| panic!("failed to load {expected_label}: {err}"));
    let expected_summary = summary
        .get(expected_label)
        .unwrap_or_else(|| panic!("missing summary row for {expected_label}"));

    let merge_config = optimized_merge_config();
    let merge_start = Instant::now();
    let merged = merge_graphs(left, right, merge_config).unwrap();
    let merge_secs = merge_start.elapsed().as_secs_f64();
    assert_graph_structure_eq(&merged, &expected);
    assert!(merged.validate().is_valid());

    let merged_provenance = total_provenance_records(&merged);
    let merged_ratio = merged_provenance as f64 / merged.node_count() as f64;
    let memory = MemoryUsage::current();
    println!(
        "saved_trace_path_merge_comparison left={} right={} expected={} merge_secs={:.3} scratch_build_secs={:.3} speedup_vs_scratch={:.3} nodes={} edges={} provenance_records={} compression_ratio={:.6} rss_mib={} peak_rss_mib={} expected_nodes={} expected_edges={} expected_provenance_records={} expected_compression_ratio={:.6}",
        left_label,
        right_label,
        expected_label,
        merge_secs,
        expected_summary.build_secs,
        expected_summary.build_secs / merge_secs,
        merged.node_count(),
        merged.edge_count(),
        merged_provenance,
        merged_ratio,
        memory.rss_mib(),
        memory.peak_rss_mib(),
        expected_summary.nodes,
        expected_summary.edges,
        expected_summary.provenance_records,
        expected_summary.compression_ratio,
    );
}

fn optimized_merge_config() -> MergeConfig {
    MergeConfig {
        min_initial_anchor_ratio: Some(SimilarityThreshold::from_basis_points(100).unwrap()),
        ..MergeConfig::default()
    }
}

fn report_saved_merge_hierarchy(
    storage: &impl GraphStorage,
    output_dir: &Path,
    scheme: &str,
    leaf_labels: &[&str],
    stages: &[Vec<&str>],
    summary: &std::collections::HashMap<String, SavedBuildSummaryRow>,
) -> Vec<SavedMergeSummaryRow> {
    let mut current = leaf_labels
        .iter()
        .map(|label| {
            (
                (*label).to_string(),
                storage
                    .load_graph(&saved_graph_path(output_dir, label))
                    .unwrap_or_else(|err| panic!("failed to load {label}: {err}")),
            )
        })
        .collect::<Vec<_>>();
    let mut rows = Vec::new();
    let mut cumulative_merge_secs = 0.0;

    for (stage_index, expected_labels) in stages.iter().enumerate() {
        assert_eq!(
            current.len(),
            expected_labels.len() * 2,
            "scheme {scheme} stage {} width mismatch",
            stage_index + 1
        );
        let mut next = Vec::with_capacity(expected_labels.len());
        let mut current_iter = current.into_iter();
        for expected_label in expected_labels {
            let (left_label, left) = current_iter.next().expect("left graph exists");
            let (right_label, right) = current_iter.next().expect("right graph exists");
            let expected = storage
                .load_graph(&saved_graph_path(output_dir, expected_label))
                .unwrap_or_else(|err| panic!("failed to load {expected_label}: {err}"));
            let expected_summary = summary
                .get(*expected_label)
                .unwrap_or_else(|| panic!("missing summary row for {expected_label}"));

            let merge_start = Instant::now();
            let merged = merge_graphs(left, right, optimized_merge_config()).unwrap();
            let merge_secs = merge_start.elapsed().as_secs_f64();
            cumulative_merge_secs += merge_secs;

            assert_graph_structure_eq(&merged, &expected);
            assert!(merged.validate().is_valid());

            let provenance_records = total_provenance_records(&merged);
            let compression_ratio = provenance_records as f64 / merged.node_count() as f64;
            let speedup_vs_scratch = expected_summary.build_secs / merge_secs;
            let memory = MemoryUsage::current();
            println!(
                "saved_trace_path_hierarchy_merge_comparison scheme={} stage={} left={} right={} expected={} merge_secs={:.3} cumulative_merge_secs={:.3} scratch_build_secs={:.3} speedup_vs_scratch={:.3} nodes={} edges={} provenance_records={} compression_ratio={:.6} rss_mib={} peak_rss_mib={}",
                scheme,
                stage_index + 1,
                left_label,
                right_label,
                expected_label,
                merge_secs,
                cumulative_merge_secs,
                expected_summary.build_secs,
                speedup_vs_scratch,
                merged.node_count(),
                merged.edge_count(),
                provenance_records,
                compression_ratio,
                memory.rss_mib(),
                memory.peak_rss_mib(),
            );
            rows.push(SavedMergeSummaryRow {
                scheme: scheme.to_string(),
                stage: stage_index + 1,
                left_label,
                right_label,
                expected_label: (*expected_label).to_string(),
                merge_secs,
                cumulative_merge_secs,
                scratch_build_secs: expected_summary.build_secs,
                speedup_vs_scratch,
                nodes: merged.node_count(),
                edges: merged.edge_count(),
                provenance_records,
                compression_ratio,
            });
            next.push(((*expected_label).to_string(), merged));
        }
        current = next;
    }

    rows
}

fn write_saved_merge_summary(path: &Path, rows: &[SavedMergeSummaryRow]) {
    let mut summary = fs::File::create(path).expect("merge summary file is created");
    writeln!(
        summary,
        "scheme\tstage\tleft\tright\texpected\tmerge_secs\tcumulative_merge_secs\tscratch_build_secs\tspeedup_vs_scratch\tnodes\tedges\tprovenance_records\tcompression_ratio"
    )
    .unwrap();
    for row in rows {
        writeln!(
            summary,
            "{}\t{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{:.6}",
            row.scheme,
            row.stage,
            row.left_label,
            row.right_label,
            row.expected_label,
            row.merge_secs,
            row.cumulative_merge_secs,
            row.scratch_build_secs,
            row.speedup_vs_scratch,
            row.nodes,
            row.edges,
            row.provenance_records,
            row.compression_ratio,
        )
        .unwrap();
    }
}

fn assert_graph_structure_eq(left: &FtoDag, right: &FtoDag) {
    assert_eq!(left.node_count(), right.node_count(), "node count differs");
    assert_eq!(left.edge_count(), right.edge_count(), "edge count differs");
    assert_endpoint_index_eq(left.endpoints(), right.endpoints());
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
            left.provenance_record_count(left_node.id).unwrap(),
            right.provenance_record_count(right_node.id).unwrap(),
            "node {index} provenance count differs"
        );
    }
    assert_eq!(left.edges(), right.edges(), "weighted edge list differs");
}

fn assert_endpoint_index_eq(left: &EndpointIndex, right: &EndpointIndex) {
    let mut left_sequence_starts = left.sequence_starts().to_vec();
    let mut right_sequence_starts = right.sequence_starts().to_vec();
    let mut left_sequence_ends = left.sequence_ends().to_vec();
    let mut right_sequence_ends = right.sequence_ends().to_vec();
    let mut left_roots = left.structural_roots().to_vec();
    let mut right_roots = right.structural_roots().to_vec();
    let mut left_sinks = left.structural_sinks().to_vec();
    let mut right_sinks = right.structural_sinks().to_vec();
    left_sequence_starts.sort_unstable();
    right_sequence_starts.sort_unstable();
    left_sequence_ends.sort_unstable();
    right_sequence_ends.sort_unstable();
    left_roots.sort_unstable();
    right_roots.sort_unstable();
    left_sinks.sort_unstable();
    right_sinks.sort_unstable();
    assert_eq!(
        left_sequence_starts, right_sequence_starts,
        "sequence starts differ"
    );
    assert_eq!(
        left_sequence_ends, right_sequence_ends,
        "sequence ends differ"
    );
    assert_eq!(left_roots, right_roots, "structural roots differ");
    assert_eq!(left_sinks, right_sinks, "structural sinks differ");
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
