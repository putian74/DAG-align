use dag_rust::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
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
#[ignore = "profiles local SARS-CoV-2 CountOnly graph merge against direct build"]
fn local_sars_cov2_count_only_merge_comparison_reports() {
    if std::env::var_os("DAG_RUST_RUN_COUNT_ONLY_MERGE_TESTS").is_none() {
        return;
    }
    if std::env::var_os("DAG_RUST_ENABLE_COUNT_ONLY_STUBBED_TEST_BODY").is_none() {
        println!("count_only_merge_comparison skipped=native_count_only_merge_stub");
        return;
    }
    if !Path::new(SARS_COV2_ARCHIVE).exists() {
        return;
    }

    let records = load_sars_cov2_full_sequences(2_000);
    assert_eq!(
        records.len(),
        2_000,
        "expected 2000 full SARS-CoV-2 records"
    );

    let left_start = Instant::now();
    let left = build_profile_graph(
        &records[..1_000],
        ProvenanceStorageStrategy::CountOnly,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let left_secs = left_start.elapsed().as_secs_f64();

    let right_start = Instant::now();
    let right = build_profile_graph(
        &records[1_000..],
        ProvenanceStorageStrategy::CountOnly,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let right_secs = right_start.elapsed().as_secs_f64();

    let merge_start = Instant::now();
    let merged = merge_graphs(left, right, optimized_merge_config()).unwrap();
    let merge_secs = merge_start.elapsed().as_secs_f64();
    assert!(merged.validate().is_valid());

    let scratch_start = Instant::now();
    let scratch = build_profile_graph(
        &records,
        ProvenanceStorageStrategy::CountOnly,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let scratch_secs = scratch_start.elapsed().as_secs_f64();
    assert!(scratch.validate().is_valid());

    let merged_provenance = total_provenance_records(&merged);
    let scratch_provenance = total_provenance_records(&scratch);
    assert_eq!(
        merged_provenance, scratch_provenance,
        "CountOnly merge must preserve total provenance count"
    );

    let memory = MemoryUsage::current();
    println!(
        "count_only_merge_comparison left_build_secs={:.3} right_build_secs={:.3} merge_secs={:.3} scratch_build_secs={:.3} speedup_vs_scratch={:.3} merged_nodes={} merged_edges={} merged_provenance_records={} merged_compression_ratio={:.6} scratch_nodes={} scratch_edges={} scratch_provenance_records={} scratch_compression_ratio={:.6} rss_mib={} peak_rss_mib={}",
        left_secs,
        right_secs,
        merge_secs,
        scratch_secs,
        scratch_secs / merge_secs,
        merged.node_count(),
        merged.edge_count(),
        merged_provenance,
        merged_provenance as f64 / merged.node_count() as f64,
        scratch.node_count(),
        scratch.edge_count(),
        scratch_provenance,
        scratch_provenance as f64 / scratch.node_count() as f64,
        memory.rss_mib(),
        memory.peak_rss_mib(),
    );
}

#[test]
#[ignore = "profiles local SARS-CoV-2 TracePaths merge against fresh direct build"]
fn local_sars_cov2_trace_path_merge_comparison_reports_fresh_builds() {
    if std::env::var_os("DAG_RUST_RUN_TRACE_PATH_MERGE_TESTS").is_none() {
        return;
    }
    if !Path::new(SARS_COV2_ARCHIVE).exists() {
        return;
    }

    let records = load_sars_cov2_full_sequences(2_000);
    assert_eq!(
        records.len(),
        2_000,
        "expected 2000 full SARS-CoV-2 records"
    );

    let left_start = Instant::now();
    let left = build_profile_graph(
        &records[..1_000],
        ProvenanceStorageStrategy::TracePaths,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let left_secs = left_start.elapsed().as_secs_f64();

    let right_start = Instant::now();
    let right = build_profile_graph(
        &records[1_000..],
        ProvenanceStorageStrategy::TracePaths,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let right_secs = right_start.elapsed().as_secs_f64();

    let merge_start = Instant::now();
    let merged = merge_graphs(left, right, optimized_merge_config()).unwrap();
    let merge_secs = merge_start.elapsed().as_secs_f64();
    assert!(merged.validate().is_valid());

    let scratch_start = Instant::now();
    let scratch = build_profile_graph(
        &records,
        ProvenanceStorageStrategy::TracePaths,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let scratch_secs = scratch_start.elapsed().as_secs_f64();
    assert_graph_structure_eq(&merged, &scratch);
    assert!(scratch.validate().is_valid());

    let merged_provenance = total_provenance_records(&merged);
    let memory = MemoryUsage::current();
    println!(
        "trace_path_merge_comparison left_build_secs={:.3} right_build_secs={:.3} merge_secs={:.3} scratch_build_secs={:.3} speedup_vs_scratch={:.3} nodes={} edges={} provenance_records={} compression_ratio={:.6} rss_mib={} peak_rss_mib={}",
        left_secs,
        right_secs,
        merge_secs,
        scratch_secs,
        scratch_secs / merge_secs,
        merged.node_count(),
        merged.edge_count(),
        merged_provenance,
        merged_provenance as f64 / merged.node_count() as f64,
        memory.rss_mib(),
        memory.peak_rss_mib(),
    );
}

#[test]
#[ignore = "compares fresh TracePaths and CountOnly subgraphs and merge outputs on the same local dataset"]
fn local_trace_path_and_count_only_merge_difference_reports() {
    if std::env::var_os("DAG_RUST_RUN_MERGE_DIFF_ANALYSIS").is_none() {
        return;
    }
    if std::env::var_os("DAG_RUST_ENABLE_COUNT_ONLY_STUBBED_TEST_BODY").is_none() {
        println!("trace_count_merge_difference skipped=native_count_only_merge_stub");
        return;
    }
    if !Path::new(SARS_COV2_ARCHIVE).exists() {
        return;
    }

    let records = load_sars_cov2_full_sequences(2_000);
    assert_eq!(
        records.len(),
        2_000,
        "expected 2000 full SARS-CoV-2 records"
    );

    let left_trace = build_profile_graph(
        &records[..1_000],
        ProvenanceStorageStrategy::TracePaths,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let right_trace = build_profile_graph(
        &records[1_000..],
        ProvenanceStorageStrategy::TracePaths,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let scratch_trace = build_profile_graph(
        &records,
        ProvenanceStorageStrategy::TracePaths,
        EdgeIndexStrategy::LowDegreeHybrid,
    );

    let left_count = build_profile_graph(
        &records[..1_000],
        ProvenanceStorageStrategy::CountOnly,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let right_count = build_profile_graph(
        &records[1_000..],
        ProvenanceStorageStrategy::CountOnly,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let scratch_count = build_profile_graph(
        &records,
        ProvenanceStorageStrategy::CountOnly,
        EdgeIndexStrategy::LowDegreeHybrid,
    );

    assert_graph_structure_eq(&left_trace, &left_count);
    assert_graph_structure_eq(&right_trace, &right_count);
    assert_graph_structure_eq(&scratch_trace, &scratch_count);

    let native_plan = simulate_count_only_merge_plan(left_count.clone(), right_count.clone());
    let merged_trace = merge_graphs(
        left_trace.clone(),
        right_trace.clone(),
        optimized_merge_config(),
    )
    .unwrap();
    let merged_trace_as_count =
        merge_trace_path_graphs_as_count_only(left_trace, right_trace, optimized_merge_config())
            .unwrap();
    let merged_count_pre_secondary =
        simulate_count_only_merge_without_secondary(left_count.clone(), right_count.clone());
    let merged_count = merge_graphs(left_count, right_count, optimized_merge_config()).unwrap();
    assert_graph_structure_eq(&merged_trace, &scratch_trace);
    assert_graph_structure_eq(&merged_trace_as_count, &scratch_count);
    let (trace_secondary, trace_secondary_stats) =
        dag_rust::algorithms::postprocess::secondary_merge_graph_with_stats(
            merged_trace.clone(),
            SecondaryMergeConfig::default(),
        )
        .unwrap();

    let diff = graph_structure_difference(&merged_count, &merged_trace);
    let detailed = analyze_count_vs_trace_merge_difference(&merged_count, &merged_trace);
    let secondary_risk = analyze_trace_secondary_merge_risk(&merged_trace);
    let trace_branch_loss = analyze_branch_constraint_delta(&merged_trace, &merged_count);
    let count_secondary_branch_loss =
        analyze_branch_constraint_delta(&merged_count_pre_secondary, &merged_count);
    println!(
        "trace_count_native_plan orientation={:?} main_path_len={} main_path_weight={} anchored_nodes={} anchored_weight={} total_add_nodes={} total_add_weight={} anchored_start={} anchored_internal={} anchored_end={} add_roots={} add_sinks={} anchored_roots={} anchored_sinks={} fresh_nodes={}",
        native_plan.plan.orientation,
        native_plan.main_path_len,
        native_plan.main_path_weight,
        native_plan.plan.matched_nodes,
        native_plan.anchored_add_weight,
        native_plan.add_graph.node_count(),
        native_plan.total_add_weight,
        native_plan.anchored_start_count,
        native_plan.anchored_internal_count,
        native_plan.anchored_end_count,
        native_plan.add_graph.endpoints().structural_roots().len(),
        native_plan.add_graph.endpoints().structural_sinks().len(),
        native_plan.anchored_root_count,
        native_plan.anchored_sink_count,
        native_plan
            .add_graph
            .node_count()
            .saturating_sub(native_plan.plan.matched_nodes),
    );
    println!(
        "trace_replay_count_control identical_to_trace={} identical_to_scratch_count={} count_nodes={} count_edges={}",
        graph_structure_difference(&merged_trace_as_count, &merged_trace).is_none(),
        graph_structure_difference(&merged_trace_as_count, &scratch_count).is_none(),
        merged_trace_as_count.node_count(),
        merged_trace_as_count.edge_count(),
    );
    println!(
        "trace_count_merge_difference identical={} count_nodes={} trace_nodes={} count_edges={} trace_edges={} detail={}",
        diff.is_none(),
        merged_count.node_count(),
        merged_trace.node_count(),
        merged_count.edge_count(),
        merged_trace.edge_count(),
        diff.unwrap_or_else(|| "none".to_string()),
    );
    println!(
        "trace_count_merge_phase_summary pre_secondary_nodes={} pre_secondary_edges={} final_count_nodes={} final_count_edges={} pre_secondary_diff={} final_diff={}",
        merged_count_pre_secondary.node_count(),
        merged_count_pre_secondary.edge_count(),
        merged_count.node_count(),
        merged_count.edge_count(),
        graph_structure_difference(&merged_count_pre_secondary, &merged_trace)
            .unwrap_or_else(|| "none".to_string()),
        graph_structure_difference(&merged_count, &merged_trace)
            .unwrap_or_else(|| "none".to_string()),
    );
    println!(
        "trace_count_merge_phase_kinds pre_secondary={} final_count={} trace={}",
        graph_kind_summary(&merged_count_pre_secondary),
        graph_kind_summary(&merged_count),
        graph_kind_summary(&merged_trace),
    );
    println!(
        "trace_count_merge_difference_summary differing_keys={} trace_excess_nodes={} preserved_weight_keys={} multiplicity_histogram={:?}",
        detailed.differing_key_count,
        detailed.trace_excess_nodes,
        detailed.preserved_weight_key_count,
        detailed.trace_multiplicity_histogram,
    );
    println!(
        "trace_count_merge_difference_topological first_mismatch={} fragment_kind_deltas={:?}",
        detailed.first_topological_mismatch, detailed.fragment_kind_deltas,
    );
    println!(
        "trace_count_merge_secondary_match identical={} removed_nodes={} merged_nodes={} risk_duplicate_keys={} risk_duplicate_nodes={} risk_conflicting_keys={} multiplicity_histogram={:?}",
        graph_structure_difference(&trace_secondary, &merged_count).is_none(),
        trace_secondary_stats.removed_nodes,
        trace_secondary_stats.merged_nodes,
        secondary_risk.duplicate_key_count,
        secondary_risk.duplicate_node_count,
        secondary_risk.conflicting_neighborhood_key_count,
        secondary_risk.multiplicity_histogram,
    );
    println!(
        "trace_count_branch_constraint_summary trace_branch_nodes={} count_branch_nodes={} trace_fan_in_nodes={} count_fan_in_nodes={} trace_fan_out_nodes={} count_fan_out_nodes={} trace_excess_branch_pattern_nodes={}",
        trace_branch_loss.left_branch_nodes,
        trace_branch_loss.right_branch_nodes,
        trace_branch_loss.left_fan_in_nodes,
        trace_branch_loss.right_fan_in_nodes,
        trace_branch_loss.left_fan_out_nodes,
        trace_branch_loss.right_fan_out_nodes,
        trace_branch_loss.left_excess_branch_pattern_nodes,
    );
    println!(
        "count_secondary_branch_constraint_summary pre_secondary_branch_nodes={} final_branch_nodes={} pre_secondary_fan_in_nodes={} final_fan_in_nodes={} pre_secondary_fan_out_nodes={} final_fan_out_nodes={} pre_secondary_excess_branch_pattern_nodes={}",
        count_secondary_branch_loss.left_branch_nodes,
        count_secondary_branch_loss.right_branch_nodes,
        count_secondary_branch_loss.left_fan_in_nodes,
        count_secondary_branch_loss.right_fan_in_nodes,
        count_secondary_branch_loss.left_fan_out_nodes,
        count_secondary_branch_loss.right_fan_out_nodes,
        count_secondary_branch_loss.left_excess_branch_pattern_nodes,
    );
    for example in detailed.examples {
        println!("trace_count_merge_difference_example {example}");
    }
    for example in secondary_risk.examples {
        println!("trace_count_secondary_risk_example {example}");
    }
    for example in trace_branch_loss.examples {
        println!("trace_count_branch_constraint_example {example}");
    }
    for example in count_secondary_branch_loss.examples {
        println!("count_secondary_branch_constraint_example {example}");
    }
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
    config.topology_update_strategy = TopologyUpdateStrategy::IncrementalForwardRelaxation;
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
    builder.finalize_graph().expect("profile graph finalizes")
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
    let graph = builder.finalize_graph().expect("saved graph finalizes");
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

fn graph_structure_difference(left: &FtoDag, right: &FtoDag) -> Option<String> {
    if left.node_count() != right.node_count() {
        return Some(format!(
            "node count differs: left={} right={}",
            left.node_count(),
            right.node_count()
        ));
    }
    if left.edge_count() != right.edge_count() {
        return Some(format!(
            "edge count differs: left={} right={}",
            left.edge_count(),
            right.edge_count()
        ));
    }

    let endpoint_diff = endpoint_index_difference(left.endpoints(), right.endpoints());
    if endpoint_diff != "none" {
        return Some(endpoint_diff);
    }

    for (index, (left_node, right_node)) in left.nodes().iter().zip(right.nodes()).enumerate() {
        if left_node.id != right_node.id {
            return Some(format!(
                "node {index} id differs: left={:?} right={:?}",
                left_node.id, right_node.id
            ));
        }
        if left_node.fragment != right_node.fragment {
            return Some(format!("node {index} fragment differs"));
        }
        if left_node.kind != right_node.kind {
            return Some(format!(
                "node {index} kind differs: left={:?} right={:?}",
                left_node.kind, right_node.kind
            ));
        }
        if left_node.weight != right_node.weight {
            return Some(format!(
                "node {index} weight differs: left={:?} right={:?}",
                left_node.weight, right_node.weight
            ));
        }
        if left.provenance_record_count(left_node.id).ok()
            != right.provenance_record_count(right_node.id).ok()
        {
            return Some(format!(
                "node {index} provenance count differs: left={:?} right={:?}",
                left.provenance_record_count(left_node.id).ok(),
                right.provenance_record_count(right_node.id).ok()
            ));
        }
    }

    if let Some((index, (left_edge, right_edge))) = left
        .edges()
        .iter()
        .zip(right.edges())
        .enumerate()
        .find(|(_, (left_edge, right_edge))| left_edge != right_edge)
    {
        return Some(format!(
            "edge {index} differs: left={:?} right={:?}",
            left_edge, right_edge
        ));
    }

    None
}

#[derive(Clone, Debug)]
struct GraphNodeNeighborhood {
    node_id: NodeId,
    weight: u64,
    parents: Vec<String>,
    children: Vec<String>,
}

#[derive(Clone, Debug)]
struct GraphKeyGroup {
    total_weight: u64,
    neighborhoods: Vec<GraphNodeNeighborhood>,
}

#[derive(Clone, Debug)]
struct CountTraceDifferenceAnalysis {
    differing_key_count: usize,
    trace_excess_nodes: usize,
    preserved_weight_key_count: usize,
    trace_multiplicity_histogram: BTreeMap<usize, usize>,
    first_topological_mismatch: String,
    fragment_kind_deltas: Vec<String>,
    examples: Vec<String>,
}

#[derive(Clone, Debug)]
struct TraceSecondaryMergeRiskAnalysis {
    duplicate_key_count: usize,
    duplicate_node_count: usize,
    conflicting_neighborhood_key_count: usize,
    multiplicity_histogram: BTreeMap<usize, usize>,
    examples: Vec<String>,
}

#[derive(Clone, Debug)]
struct BranchConstraintDeltaAnalysis {
    left_branch_nodes: usize,
    right_branch_nodes: usize,
    left_fan_in_nodes: usize,
    right_fan_in_nodes: usize,
    left_fan_out_nodes: usize,
    right_fan_out_nodes: usize,
    left_excess_branch_pattern_nodes: usize,
    examples: Vec<String>,
}

fn analyze_count_vs_trace_merge_difference(
    count_graph: &FtoDag,
    trace_graph: &FtoDag,
) -> CountTraceDifferenceAnalysis {
    let count_groups = graph_key_groups(count_graph);
    let trace_groups = graph_key_groups(trace_graph);
    let mut keys = trace_groups.keys().cloned().collect::<Vec<_>>();
    keys.sort_unstable();

    let mut differing_key_count = 0;
    let mut trace_excess_nodes = 0;
    let mut preserved_weight_key_count = 0;
    let mut trace_multiplicity_histogram = BTreeMap::new();
    let mut examples = Vec::new();

    for key in keys {
        let Some(trace_group) = trace_groups.get(&key) else {
            continue;
        };
        let count_group = count_groups.get(&key);
        let trace_len = trace_group.neighborhoods.len();
        let count_len = count_group.map_or(0, |group| group.neighborhoods.len());
        if trace_len == count_len {
            continue;
        }

        differing_key_count += 1;
        trace_excess_nodes += trace_len.saturating_sub(count_len);
        *trace_multiplicity_histogram.entry(trace_len).or_insert(0) += 1;

        let count_weight = count_group.map_or(0, |group| group.total_weight);
        if count_weight == trace_group.total_weight {
            preserved_weight_key_count += 1;
        }

        if examples.len() < 12 {
            let trace_nodes = trace_group
                .neighborhoods
                .iter()
                .map(format_node_neighborhood)
                .collect::<Vec<_>>()
                .join(" || ");
            let count_nodes = count_group
                .map(|group| {
                    group
                        .neighborhoods
                        .iter()
                        .map(format_node_neighborhood)
                        .collect::<Vec<_>>()
                        .join(" || ")
                })
                .unwrap_or_else(|| "none".to_string());
            examples.push(format!(
                "key={key} trace_nodes={trace_len} count_nodes={count_len} trace_weight={} count_weight={} trace_details=[{}] count_details=[{}]",
                trace_group.total_weight,
                count_weight,
                trace_nodes,
                count_nodes
            ));
        }
    }

    CountTraceDifferenceAnalysis {
        differing_key_count,
        trace_excess_nodes,
        preserved_weight_key_count,
        trace_multiplicity_histogram,
        first_topological_mismatch: first_topological_mismatch(count_graph, trace_graph),
        fragment_kind_deltas: fragment_kind_count_deltas(count_graph, trace_graph),
        examples,
    }
}

fn analyze_trace_secondary_merge_risk(trace_graph: &FtoDag) -> TraceSecondaryMergeRiskAnalysis {
    let groups = graph_key_groups(trace_graph);
    let mut duplicate_key_count = 0;
    let mut duplicate_node_count = 0;
    let mut conflicting_neighborhood_key_count = 0;
    let mut multiplicity_histogram = BTreeMap::new();
    let mut examples = Vec::new();

    let mut keys = groups.keys().cloned().collect::<Vec<_>>();
    keys.sort_unstable();
    for key in keys {
        let group = &groups[&key];
        let multiplicity = group.neighborhoods.len();
        if multiplicity <= 1 {
            continue;
        }
        duplicate_key_count += 1;
        duplicate_node_count += multiplicity;
        *multiplicity_histogram.entry(multiplicity).or_insert(0) += 1;

        let unique_neighborhoods = group
            .neighborhoods
            .iter()
            .map(neighborhood_signature)
            .collect::<HashSet<_>>();
        if unique_neighborhoods.len() > 1 {
            conflicting_neighborhood_key_count += 1;
        }
        if examples.len() < 12 {
            let detail = group
                .neighborhoods
                .iter()
                .map(format_node_neighborhood)
                .collect::<Vec<_>>()
                .join(" || ");
            examples.push(format!(
                "key={key} multiplicity={} total_weight={} unique_neighborhoods={} details=[{}]",
                multiplicity,
                group.total_weight,
                unique_neighborhoods.len(),
                detail
            ));
        }
    }

    TraceSecondaryMergeRiskAnalysis {
        duplicate_key_count,
        duplicate_node_count,
        conflicting_neighborhood_key_count,
        multiplicity_histogram,
        examples,
    }
}

fn analyze_branch_constraint_delta(
    left_graph: &FtoDag,
    right_graph: &FtoDag,
) -> BranchConstraintDeltaAnalysis {
    let left_patterns = graph_branch_pattern_counts(left_graph);
    let right_patterns = graph_branch_pattern_counts(right_graph);

    let left_branch_nodes = count_branch_nodes(left_graph);
    let right_branch_nodes = count_branch_nodes(right_graph);
    let left_fan_in_nodes = count_fan_in_nodes(left_graph);
    let right_fan_in_nodes = count_fan_in_nodes(right_graph);
    let left_fan_out_nodes = count_fan_out_nodes(left_graph);
    let right_fan_out_nodes = count_fan_out_nodes(right_graph);

    let mut deltas = Vec::new();
    let mut left_excess_branch_pattern_nodes = 0;
    for (key, left_count) in left_patterns {
        let right_count = right_patterns.get(&key).copied().unwrap_or(0);
        if left_count > right_count {
            let delta = left_count - right_count;
            left_excess_branch_pattern_nodes += delta;
            deltas.push((delta, left_count, right_count, key));
        }
    }
    deltas.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then(right.1.cmp(&left.1))
            .then_with(|| left.3.cmp(&right.3))
    });

    let examples = deltas
        .into_iter()
        .take(12)
        .map(|(delta, left_count, right_count, key)| {
            format!(
                "pattern={key} left_count={left_count} right_count={right_count} left_minus_right={delta}"
            )
        })
        .collect();

    BranchConstraintDeltaAnalysis {
        left_branch_nodes,
        right_branch_nodes,
        left_fan_in_nodes,
        right_fan_in_nodes,
        left_fan_out_nodes,
        right_fan_out_nodes,
        left_excess_branch_pattern_nodes,
        examples,
    }
}

fn graph_branch_pattern_counts(graph: &FtoDag) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for node in graph.nodes() {
        let parent_count = graph.parents(node.id).expect("branch parents exist").len();
        let child_count = graph
            .children(node.id)
            .expect("branch children exist")
            .len();
        if parent_count <= 1 && child_count <= 1 {
            continue;
        }
        let key = format!(
            "fragment={:?}|kind={:?}|in={parent_count}|out={child_count}",
            node.fragment, node.kind
        );
        *counts.entry(key).or_insert(0) += 1;
    }
    counts
}

fn count_branch_nodes(graph: &FtoDag) -> usize {
    graph
        .nodes()
        .iter()
        .filter(|node| {
            let parent_count = graph.parents(node.id).expect("branch parents exist").len();
            let child_count = graph
                .children(node.id)
                .expect("branch children exist")
                .len();
            parent_count > 1 || child_count > 1
        })
        .count()
}

fn count_fan_in_nodes(graph: &FtoDag) -> usize {
    graph
        .nodes()
        .iter()
        .filter(|node| graph.parents(node.id).expect("fan-in parents exist").len() > 1)
        .count()
}

fn count_fan_out_nodes(graph: &FtoDag) -> usize {
    graph
        .nodes()
        .iter()
        .filter(|node| {
            graph
                .children(node.id)
                .expect("fan-out children exist")
                .len()
                > 1
        })
        .count()
}

fn simulate_count_only_merge_without_secondary(left: FtoDag, right: FtoDag) -> FtoDag {
    let selected = simulate_count_only_merge_plan(left, right);
    apply_count_only_merge_plan_without_secondary(
        selected.base_graph,
        selected.add_graph,
        &selected.plan,
    )
}

#[derive(Clone, Debug)]
struct SelectedCountOnlyMergePlan {
    base_graph: FtoDag,
    add_graph: FtoDag,
    plan: PlannedCountOnlyMergeForTest,
    main_path_len: usize,
    main_path_weight: u64,
    total_add_weight: u64,
    anchored_add_weight: u64,
    anchored_start_count: usize,
    anchored_internal_count: usize,
    anchored_end_count: usize,
    anchored_root_count: usize,
    anchored_sink_count: usize,
}

fn simulate_count_only_merge_plan(left: FtoDag, right: FtoDag) -> SelectedCountOnlyMergePlan {
    let right_into_left =
        plan_count_only_merge_for_test(&left, &right, MergeOrientation::RightIntoLeft);
    let left_into_right =
        plan_count_only_merge_for_test(&right, &left, MergeOrientation::LeftIntoRight);
    if prefer_count_only_plan_for_test(&left_into_right, &right_into_left) {
        build_selected_count_only_plan(right, left, left_into_right)
    } else {
        build_selected_count_only_plan(left, right, right_into_left)
    }
}

#[derive(Clone, Debug)]
struct PlannedCountOnlyMergeForTest {
    orientation: MergeOrientation,
    anchors: AnchorMap,
    score: u64,
    matched_nodes: usize,
    main_path: Vec<NodeId>,
}

fn plan_count_only_merge_for_test(
    base: &FtoDag,
    add: &FtoDag,
    orientation: MergeOrientation,
) -> PlannedCountOnlyMergeForTest {
    let base_topology = DagTopology::try_from_graph(base).unwrap();
    let add_topology = DagTopology::try_from_graph(add).unwrap();
    let add_main_path = max_weight_path_for_test(add, &add_topology);
    let candidate_sets =
        base_main_path_candidate_sets_for_test(base, &base_topology, add, &add_main_path);
    let anchor_path = select_monotone_anchor_path_with_graph(&candidate_sets, base);
    let anchors = anchor_map_from_main_path_for_test(&add_main_path, &anchor_path);
    let score = anchor_score_for_test(base, add, &anchors);
    PlannedCountOnlyMergeForTest {
        orientation,
        matched_nodes: anchors.pairs.len(),
        anchors,
        score,
        main_path: add_main_path,
    }
}

fn prefer_count_only_plan_for_test(
    candidate: &PlannedCountOnlyMergeForTest,
    current: &PlannedCountOnlyMergeForTest,
) -> bool {
    candidate.score > current.score
        || (candidate.score == current.score && candidate.matched_nodes > current.matched_nodes)
        || (candidate.score == current.score
            && candidate.matched_nodes == current.matched_nodes
            && matches!(candidate.orientation, MergeOrientation::RightIntoLeft)
            && matches!(current.orientation, MergeOrientation::LeftIntoRight))
}

fn build_selected_count_only_plan(
    base_graph: FtoDag,
    add_graph: FtoDag,
    plan: PlannedCountOnlyMergeForTest,
) -> SelectedCountOnlyMergePlan {
    let total_add_weight = add_graph.nodes().iter().map(|node| node.weight.raw()).sum();
    let main_path_weight = plan
        .main_path
        .iter()
        .map(|node_id| add_graph.node(*node_id).unwrap().weight.raw())
        .sum();
    let anchored_add_weight = plan
        .anchors
        .pairs
        .iter()
        .map(|(add_node, _)| add_graph.node(*add_node).unwrap().weight.raw())
        .sum();
    let mut anchored_start_count = 0;
    let mut anchored_internal_count = 0;
    let mut anchored_end_count = 0;
    let mut anchored_root_count = 0;
    let mut anchored_sink_count = 0;
    let root_set: HashSet<NodeId> = add_graph
        .endpoints()
        .structural_roots()
        .iter()
        .copied()
        .collect();
    let sink_set: HashSet<NodeId> = add_graph
        .endpoints()
        .structural_sinks()
        .iter()
        .copied()
        .collect();
    for (add_node, _) in &plan.anchors.pairs {
        let node = add_graph.node(*add_node).unwrap();
        match node.kind {
            NodeKind::Start => anchored_start_count += 1,
            NodeKind::Internal => anchored_internal_count += 1,
            NodeKind::End => anchored_end_count += 1,
            NodeKind::Singleton => {}
        }
        if root_set.contains(add_node) {
            anchored_root_count += 1;
        }
        if sink_set.contains(add_node) {
            anchored_sink_count += 1;
        }
    }

    SelectedCountOnlyMergePlan {
        base_graph,
        add_graph,
        main_path_len: plan.main_path.len(),
        main_path_weight,
        total_add_weight,
        anchored_add_weight,
        anchored_start_count,
        anchored_internal_count,
        anchored_end_count,
        anchored_root_count,
        anchored_sink_count,
        plan,
    }
}

fn apply_count_only_merge_plan_without_secondary(
    mut base: FtoDag,
    add: FtoDag,
    plan: &PlannedCountOnlyMergeForTest,
) -> FtoDag {
    let add_topology = DagTopology::try_from_graph(&add).unwrap();
    let mut remap = vec![None; add.node_count()];
    for (add_node, base_node) in &plan.anchors.pairs {
        remap[add_node.to_usize()] = Some(*base_node);
    }

    for add_node in add_topology.topological_order() {
        if remap[add_node.to_usize()].is_some() {
            continue;
        }
        let source = add.node(*add_node).unwrap();
        let merged = base.add_node(source.fragment.clone(), source.kind).unwrap();
        remap[add_node.to_usize()] = Some(merged);
    }

    for edge in add.edges() {
        let parent = remap[edge.key.parent.to_usize()].unwrap();
        let child = remap[edge.key.child.to_usize()].unwrap();
        if parent == child {
            continue;
        }
        base.add_or_increment_edge(parent, child, edge.weight)
            .unwrap();
    }

    for node in add.nodes() {
        let merged = remap[node.id.to_usize()].unwrap();
        base.add_provenance_count(merged, node.weight.raw())
            .unwrap();
    }

    base
}

fn max_weight_path_for_test(graph: &FtoDag, topology: &DagTopology) -> Vec<NodeId> {
    let mut scores = vec![0_u64; graph.node_count()];
    let mut previous = vec![None; graph.node_count()];

    for node_id in topology.topological_order().iter().copied() {
        let node_score = graph.node(node_id).unwrap().weight.raw();
        let mut best_parent_score = 0_u64;
        let mut best_parent = None;
        for parent in graph.parents(node_id).unwrap().iter().copied() {
            let candidate_score = scores[parent.to_usize()];
            if candidate_score > best_parent_score
                || (candidate_score == best_parent_score
                    && best_parent.is_none_or(|current| parent < current))
            {
                best_parent_score = candidate_score;
                best_parent = Some(parent);
            }
        }
        scores[node_id.to_usize()] = best_parent_score + node_score;
        previous[node_id.to_usize()] = best_parent;
    }

    let mut best_end = None;
    for node_id in graph.endpoints().structural_sinks().iter().copied() {
        let score = scores[node_id.to_usize()];
        if best_end.is_none_or(|current: NodeId| {
            score > scores[current.to_usize()]
                || (score == scores[current.to_usize()] && node_id < current)
        }) {
            best_end = Some(node_id);
        }
    }
    let mut cursor = best_end.or_else(|| topology.topological_order().last().copied());
    let mut path = Vec::new();
    while let Some(node_id) = cursor {
        path.push(node_id);
        cursor = previous[node_id.to_usize()];
    }
    path.reverse();
    path
}

fn base_main_path_candidate_sets_for_test(
    base: &FtoDag,
    base_topology: &DagTopology,
    add: &FtoDag,
    add_main_path: &[NodeId],
) -> Vec<AnchorCandidateSet> {
    let mut sets = Vec::with_capacity(add_main_path.len());
    for (position, add_node_id) in add_main_path.iter().copied().enumerate() {
        let add_node = add.node(add_node_id).unwrap();
        let mut candidates = Vec::new();
        for base_node_id in base
            .fragment_index()
            .nodes_for_stored(&add_node.fragment, add_node.kind)
            .iter()
            .copied()
        {
            let base_node = base.node(base_node_id).unwrap();
            candidates.push(AnchorCandidate {
                node: base_node_id,
                kind: base_node.kind,
                coordinate: Some(base_topology.forward_coordinate(base_node_id).unwrap()),
                reverse_coordinate: Some(base_topology.reverse_coordinate(base_node_id).unwrap()),
                weight: base_node.weight,
            });
        }
        sets.push(AnchorCandidateSet {
            position,
            kind: add_node.kind,
            fragment: add_node.fragment.to_fragment_key(),
            candidates,
        });
    }
    sets
}

fn anchor_map_from_main_path_for_test(
    add_main_path: &[NodeId],
    anchor_path: &AnchorPath,
) -> AnchorMap {
    AnchorMap {
        pairs: anchor_path
            .decisions
            .iter()
            .enumerate()
            .filter_map(|(index, decision)| match decision {
                AnchorDecision::Matched(base_node) => Some((add_main_path[index], *base_node)),
                AnchorDecision::Unmatched(_) => None,
            })
            .collect(),
    }
}

fn anchor_score_for_test(base: &FtoDag, add: &FtoDag, anchors: &AnchorMap) -> u64 {
    anchors
        .pairs
        .iter()
        .map(|(add_node, base_node)| {
            add.node(*add_node).unwrap().weight.raw() + base.node(*base_node).unwrap().weight.raw()
        })
        .sum()
}

fn graph_kind_summary(graph: &FtoDag) -> String {
    let starts = graph
        .nodes()
        .iter()
        .filter(|node| matches!(node.kind, NodeKind::Start))
        .count();
    let internals = graph
        .nodes()
        .iter()
        .filter(|node| matches!(node.kind, NodeKind::Internal))
        .count();
    let ends = graph
        .nodes()
        .iter()
        .filter(|node| matches!(node.kind, NodeKind::End))
        .count();
    let singletons = graph
        .nodes()
        .iter()
        .filter(|node| matches!(node.kind, NodeKind::Singleton))
        .count();
    format!(
        "start={} internal={} end={} singleton={} roots={} sinks={}",
        starts,
        internals,
        ends,
        singletons,
        graph.endpoints().structural_roots().len(),
        graph.endpoints().structural_sinks().len(),
    )
}

fn first_topological_mismatch(count_graph: &FtoDag, trace_graph: &FtoDag) -> String {
    let count_len = count_graph.node_count();
    let trace_len = trace_graph.node_count();
    let shared = count_len.min(trace_len);
    for index in 0..shared {
        let count_node = &count_graph.nodes()[index];
        let trace_node = &trace_graph.nodes()[index];
        if count_node.fragment != trace_node.fragment
            || count_node.kind != trace_node.kind
            || count_node.weight != trace_node.weight
        {
            let count_window = node_window(count_graph, index, 3);
            let trace_window = node_window(trace_graph, index, 3);
            return format!(
                "index={} count_node={:?}/{:?}/{:?} trace_node={:?}/{:?}/{:?} count_window={:?} trace_window={:?}",
                index,
                count_node.fragment,
                count_node.kind,
                count_node.weight,
                trace_node.fragment,
                trace_node.kind,
                trace_node.weight,
                count_window,
                trace_window
            );
        }
    }
    if count_len != trace_len {
        return format!(
            "all shared prefix nodes match; lengths differ count={} trace={}",
            count_len, trace_len
        );
    }
    "none".to_string()
}

fn node_window(graph: &FtoDag, center: usize, radius: usize) -> Vec<String> {
    let start = center.saturating_sub(radius);
    let end = (center + radius + 1).min(graph.node_count());
    graph.nodes()[start..end]
        .iter()
        .map(|node| {
            format!(
                "{}:{:?}:{:?}:{:?}",
                node.id.to_usize(),
                node.fragment,
                node.kind,
                node.weight
            )
        })
        .collect()
}

fn fragment_kind_count_deltas(count_graph: &FtoDag, trace_graph: &FtoDag) -> Vec<String> {
    let mut count_map = HashMap::<String, usize>::new();
    let mut trace_map = HashMap::<String, usize>::new();
    for node in count_graph.nodes() {
        *count_map
            .entry(format!("{:?}|{:?}", node.fragment, node.kind))
            .or_insert(0) += 1;
    }
    for node in trace_graph.nodes() {
        *trace_map
            .entry(format!("{:?}|{:?}", node.fragment, node.kind))
            .or_insert(0) += 1;
    }
    let mut keys = trace_map.keys().cloned().collect::<Vec<_>>();
    keys.sort_unstable();
    let mut deltas = keys
        .into_iter()
        .filter_map(|key| {
            let trace = trace_map.get(&key).copied().unwrap_or(0);
            let count = count_map.get(&key).copied().unwrap_or(0);
            (trace != count).then(|| (trace.saturating_sub(count), trace, count, key))
        })
        .collect::<Vec<_>>();
    deltas.sort_by(|left, right| right.cmp(left));
    deltas
        .into_iter()
        .take(12)
        .map(|(delta, trace, count, key)| {
            format!("key={key} trace_count={trace} count_count={count} trace_minus_count={delta}")
        })
        .collect()
}

fn graph_key_groups(graph: &FtoDag) -> HashMap<String, GraphKeyGroup> {
    let topology = DagTopology::try_from_graph(graph).expect("graph is acyclic for diff analysis");
    let mut groups = HashMap::new();
    for node_id in topology.topological_order().iter().copied() {
        let node = graph.node(node_id).expect("diff node exists");
        let key = format!(
            "fragment={:?}|kind={:?}|forward={}|reverse={}",
            node.fragment,
            node.kind,
            topology
                .forward_coordinate(node_id)
                .expect("forward coordinate exists")
                .raw(),
            topology
                .reverse_coordinate(node_id)
                .expect("reverse coordinate exists")
                .raw(),
        );
        let group = groups.entry(key).or_insert_with(|| GraphKeyGroup {
            total_weight: 0,
            neighborhoods: Vec::new(),
        });
        group.total_weight += node.weight.raw();
        group.neighborhoods.push(GraphNodeNeighborhood {
            node_id,
            weight: node.weight.raw(),
            parents: graph
                .parents(node_id)
                .expect("diff parents exist")
                .iter()
                .copied()
                .map(|parent| node_signature(graph, &topology, parent))
                .collect(),
            children: graph
                .children(node_id)
                .expect("diff children exist")
                .iter()
                .copied()
                .map(|child| node_signature(graph, &topology, child))
                .collect(),
        });
    }
    for group in groups.values_mut() {
        group
            .neighborhoods
            .sort_by_key(|neighborhood| neighborhood.node_id.to_usize());
    }
    groups
}

fn node_signature(graph: &FtoDag, topology: &DagTopology, node_id: NodeId) -> String {
    let node = graph.node(node_id).expect("signature node exists");
    format!(
        "{}:{:?}:{:?}:f{}:r{}",
        node_id.to_usize(),
        node.fragment,
        node.kind,
        topology
            .forward_coordinate(node_id)
            .expect("forward coordinate exists")
            .raw(),
        topology
            .reverse_coordinate(node_id)
            .expect("reverse coordinate exists")
            .raw(),
    )
}

fn format_node_neighborhood(node: &GraphNodeNeighborhood) -> String {
    format!(
        "id={} weight={} parents={:?} children={:?}",
        node.node_id.to_usize(),
        node.weight,
        node.parents,
        node.children,
    )
}

fn neighborhood_signature(node: &GraphNodeNeighborhood) -> String {
    format!(
        "weight={} parents={:?} children={:?}",
        node.weight, node.parents, node.children
    )
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

fn endpoint_index_difference(left: &EndpointIndex, right: &EndpointIndex) -> String {
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

    if left_sequence_starts != right_sequence_starts {
        return format!(
            "sequence starts differ: left={:?} right={:?}",
            left_sequence_starts, right_sequence_starts
        );
    }
    if left_sequence_ends != right_sequence_ends {
        return format!(
            "sequence ends differ: left={:?} right={:?}",
            left_sequence_ends, right_sequence_ends
        );
    }
    if left_roots != right_roots {
        return format!(
            "roots differ: left={:?} right={:?}",
            left_roots, right_roots
        );
    }
    if left_sinks != right_sinks {
        return format!(
            "sinks differ: left={:?} right={:?}",
            left_sinks, right_sinks
        );
    }
    "none".to_string()
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
