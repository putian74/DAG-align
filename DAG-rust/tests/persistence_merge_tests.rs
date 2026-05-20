use dag_rust::prelude::*;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn build_graph(sequences: &[&str], provenance: ProvenanceStorageStrategy) -> FtoDag {
    let alphabet = BuiltinAlphabet::dna_exact();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let mut config = BuildConfig::new(3);
    config.provenance_storage_strategy = provenance;
    config.min_initial_match_ratio = Some(SimilarityThreshold::none());
    let mut builder = FtoDagBuilder::new(config);

    for (index, sequence) in sequences.iter().enumerate() {
        let encoded = EncodedSequence::encode(
            SequenceRecord::new(format!("seq-{index}"), (*sequence).to_string()),
            &alphabet,
        )
        .unwrap();
        builder
            .add_sequence_from_encoded(SequenceId::try_from(index).unwrap(), &encoded, &encoder)
            .unwrap();
    }

    builder.finalize_graph().unwrap()
}

fn key(raw: u16) -> FragmentKey {
    FragmentKey::symbols(vec![SymbolId::new(raw)])
}

fn build_branching_count_only_graph() -> FtoDag {
    let mut graph = FtoDag::with_provenance_storage(1, ProvenanceStorageStrategy::CountOnly);
    let start = graph.add_node(key(0), NodeKind::Start).unwrap();
    let branch_a = graph.add_node(key(1), NodeKind::Internal).unwrap();
    let branch_b = graph.add_node(key(2), NodeKind::Internal).unwrap();
    let end = graph.add_node(key(3), NodeKind::End).unwrap();

    graph
        .add_or_increment_edge(start, branch_a, Weight::new(1))
        .unwrap();
    graph
        .add_or_increment_edge(start, branch_b, Weight::new(1))
        .unwrap();
    graph
        .add_or_increment_edge(branch_a, end, Weight::new(1))
        .unwrap();
    graph
        .add_or_increment_edge(branch_b, end, Weight::new(1))
        .unwrap();

    for (position, node) in [start, branch_a, branch_b, end].into_iter().enumerate() {
        graph
            .add_provenance_record(
                node,
                ProvenanceRecord {
                    sequence_id: SequenceId::new(0),
                    position: ProvenancePosition::new(position as u64),
                },
            )
            .unwrap();
    }

    graph
}

fn temp_graph_path(label: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    env::temp_dir().join(format!("dag-rust-{label}-{nanos}.dagrs"))
}

fn assert_graph_structure_eq(left: &FtoDag, right: &FtoDag) {
    assert_eq!(
        left.fragment_len(),
        right.fragment_len(),
        "fragment length differs"
    );
    assert_eq!(
        left.provenance_storage_strategy(),
        right.provenance_storage_strategy(),
        "provenance storage differs"
    );
    assert_eq!(
        left.edge_index_strategy(),
        right.edge_index_strategy(),
        "edge index strategy differs"
    );
    assert_eq!(left.node_count(), right.node_count(), "node count differs");
    assert_eq!(left.edge_count(), right.edge_count(), "edge count differs");
    assert_endpoint_index_eq(left.endpoints(), right.endpoints());
    for (index, (left_node, right_node)) in left.nodes().iter().zip(right.nodes()).enumerate() {
        assert_eq!(left_node, right_node, "node {index} differs");
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

#[test]
fn native_storage_round_trips_full_record_graphs() {
    let graph = build_graph(
        &["ACGTA", "ACCTA", "ACGTT"],
        ProvenanceStorageStrategy::FullRecords,
    );
    let storage = NativeGraphStorage;
    let path = temp_graph_path("full-records");

    storage
        .save_graph(&graph, &path, StorageConfig::default())
        .unwrap();
    let loaded = storage.load_graph(&path).unwrap();

    assert_graph_structure_eq(&graph, &loaded);
    assert!(loaded.validate().is_valid());
    fs::remove_file(path).unwrap();
}

#[test]
fn native_storage_round_trips_trace_path_graphs() {
    let graph = build_graph(
        &["ACGTA", "ACCTA", "ACGTT"],
        ProvenanceStorageStrategy::TracePaths,
    );
    let storage = NativeGraphStorage;
    let path = temp_graph_path("trace-paths");

    storage
        .save_graph(&graph, &path, StorageConfig::default())
        .unwrap();
    let loaded = storage.load_graph(&path).unwrap();

    assert_graph_structure_eq(&graph, &loaded);
    for sequence in 0..3 {
        let sequence_id = SequenceId::new(sequence);
        assert_eq!(
            graph.sequence_trace_path(sequence_id).unwrap(),
            loaded.sequence_trace_path(sequence_id).unwrap(),
            "sequence path {sequence} differs"
        );
    }
    assert!(loaded.validate().is_valid());
    fs::remove_file(path).unwrap();
}

#[test]
fn merge_graphs_replays_trace_paths_into_expected_graph() {
    let left_sequences = ["ACGTA", "ACGTT"];
    let right_sequences = ["ACCTA", "ACCTT"];
    let left = build_graph(&left_sequences, ProvenanceStorageStrategy::TracePaths);
    let right = build_graph(&right_sequences, ProvenanceStorageStrategy::TracePaths);
    let expected = build_graph(
        &["ACGTA", "ACGTT", "ACCTA", "ACCTT"],
        ProvenanceStorageStrategy::TracePaths,
    );

    let merged = merge_graphs(left, right, MergeConfig::default()).unwrap();

    assert_graph_structure_eq(&merged, &expected);
    for sequence in 0..4 {
        let sequence_id = SequenceId::new(sequence);
        assert_eq!(
            merged.sequence_trace_path(sequence_id).unwrap(),
            expected.sequence_trace_path(sequence_id).unwrap(),
            "merged sequence path {sequence} differs"
        );
    }
    assert!(merged.validate().is_valid());
}

#[test]
fn merge_trace_path_graphs_as_count_only_matches_direct_count_only_build() {
    let left_sequences = ["ACGTA", "ACGTT"];
    let right_sequences = ["ACCTA", "ACCTT"];
    let left = build_graph(&left_sequences, ProvenanceStorageStrategy::TracePaths);
    let right = build_graph(&right_sequences, ProvenanceStorageStrategy::TracePaths);
    let expected = build_graph(
        &["ACGTA", "ACGTT", "ACCTA", "ACCTT"],
        ProvenanceStorageStrategy::CountOnly,
    );

    let merged =
        merge_trace_path_graphs_as_count_only(left, right, MergeConfig::default()).unwrap();

    assert_graph_structure_eq(&merged, &expected);
    assert_eq!(
        merged.provenance_storage_strategy(),
        ProvenanceStorageStrategy::CountOnly
    );
    assert!(merged.validate().is_valid());
}

#[test]
fn merge_graphs_count_only_is_stubbed_for_now() {
    let left = build_graph(&["AACCA"], ProvenanceStorageStrategy::CountOnly);
    let right = build_graph(&["GACCG"], ProvenanceStorageStrategy::CountOnly);

    let error = merge_graphs(left, right, MergeConfig::default()).unwrap_err();
    assert!(
        matches!(error, DagError::UnsupportedOperation(message) if message.contains("CountOnly graph merge is a stub"))
    );
}

#[test]
fn secondary_merge_graph_count_only_is_noop_stub() {
    let graph = build_branching_count_only_graph();
    let expected_node_count = graph.node_count();
    let expected_edge_count = graph.edge_count();
    let expected_total_provenance = total_provenance_records(&graph);
    let merged = secondary_merge_graph(graph, SecondaryMergeConfig::default()).unwrap();

    assert_eq!(merged.node_count(), expected_node_count);
    assert_eq!(merged.edge_count(), expected_edge_count);
    assert_eq!(total_provenance_records(&merged), expected_total_provenance);
    assert!(merged.validate().is_valid());
}

fn total_provenance_records(graph: &FtoDag) -> usize {
    graph
        .nodes()
        .iter()
        .map(|node| node.weight.raw() as usize)
        .sum()
}
