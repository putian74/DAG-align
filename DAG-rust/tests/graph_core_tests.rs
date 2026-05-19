use dag_rust::prelude::*;

fn key(raw: u16) -> FragmentKey {
    FragmentKey::symbols(vec![SymbolId::new(raw)])
}

#[test]
fn endpoint_index_separates_sequence_and_structural_endpoints() {
    let mut graph = FtoDag::new(1);
    let start = graph.add_node(key(0), NodeKind::Start).unwrap();
    let internal = graph.add_node(key(1), NodeKind::Internal).unwrap();
    let end = graph.add_node(key(2), NodeKind::End).unwrap();

    assert_eq!(graph.endpoints().sequence_starts(), &[start]);
    assert_eq!(graph.endpoints().sequence_ends(), &[end]);
    assert_eq!(
        graph.endpoints().structural_roots(),
        &[start, internal, end]
    );
    assert_eq!(
        graph.endpoints().structural_sinks(),
        &[start, internal, end]
    );

    graph
        .add_or_increment_edge(start, internal, Weight::new(1))
        .unwrap();
    let update = graph
        .add_or_increment_edge(start, internal, Weight::new(2))
        .unwrap();
    assert!(!update.inserted);
    assert_eq!(update.weight, Weight::new(3));
    assert_eq!(
        graph.edge_weight(EdgeKey {
            source: start,
            target: internal,
        }),
        Some(Weight::new(3))
    );
    graph
        .add_or_increment_edge(internal, end, Weight::new(1))
        .unwrap();

    assert_eq!(graph.endpoints().sequence_starts(), &[start]);
    assert_eq!(graph.endpoints().sequence_ends(), &[end]);
    assert_eq!(graph.endpoints().structural_roots(), &[start]);
    assert_eq!(graph.endpoints().structural_sinks(), &[end]);
    assert!(graph.validate().is_valid());
    assert_eq!(graph.edge_count(), 2);
}

#[test]
fn hybrid_edge_index_increments_existing_edges() {
    let mut graph = FtoDag::with_source_and_edge_storage(
        1,
        SourceStorageStrategy::FullRecords,
        EdgeIndexStrategy::LowDegreeHybrid,
    );
    let left = graph.add_node(key(1), NodeKind::Start).unwrap();
    let right = graph.add_node(key(2), NodeKind::End).unwrap();

    let first = graph
        .add_or_increment_edge(left, right, Weight::new(1))
        .unwrap();
    let second = graph
        .add_or_increment_edge(left, right, Weight::new(3))
        .unwrap();

    assert_eq!(
        graph.edge_index_strategy(),
        EdgeIndexStrategy::LowDegreeHybrid
    );
    assert!(first.inserted);
    assert!(!second.inserted);
    assert_eq!(graph.edge_count(), 1);
    assert_eq!(
        graph.edge_weight(EdgeKey {
            source: left,
            target: right,
        }),
        Some(Weight::new(4))
    );
    assert!(graph.validate().is_valid());
}

#[test]
fn fragment_index_keeps_repeated_same_kind_nodes_distinct() {
    let mut graph = FtoDag::new(1);
    let repeated = key(7);
    let first = graph
        .add_node(repeated.clone(), NodeKind::Internal)
        .unwrap();
    let second = graph
        .add_node(repeated.clone(), NodeKind::Internal)
        .unwrap();
    let start = graph.add_node(repeated.clone(), NodeKind::Start).unwrap();

    assert_eq!(
        graph
            .fragment_index()
            .nodes_for(&repeated, NodeKind::Internal),
        &[first, second]
    );
    assert_eq!(
        graph.fragment_index().nodes_for(&repeated, NodeKind::Start),
        &[start]
    );
}

#[test]
fn fragment_index_uses_packed_inline_keys() {
    let mut graph = FtoDag::new(31);
    let fragment = FragmentKey::packed_inline(2, 31, 123_456);
    let first = graph
        .add_node(fragment.clone(), NodeKind::Internal)
        .unwrap();
    let second = graph
        .add_node(fragment.clone(), NodeKind::Internal)
        .unwrap();

    assert_eq!(
        graph
            .fragment_index()
            .nodes_for(&fragment, NodeKind::Internal),
        &[first, second]
    );
}

#[test]
fn graph_owned_source_table_tracks_node_source_ranges() {
    let mut graph = FtoDag::new(1);
    let left = graph.add_node(key(1), NodeKind::Start).unwrap();
    let right = graph.add_node(key(2), NodeKind::End).unwrap();

    graph
        .add_source_record(
            left,
            SourceRecord {
                sequence_id: SequenceId::new(0),
                position: SourcePosition::new(10),
            },
        )
        .unwrap();
    graph
        .add_source_record(
            right,
            SourceRecord {
                sequence_id: SequenceId::new(0),
                position: SourcePosition::new(11),
            },
        )
        .unwrap();
    graph
        .add_source_record(
            left,
            SourceRecord {
                sequence_id: SequenceId::new(1),
                position: SourcePosition::new(20),
            },
        )
        .unwrap();

    let left_sources = graph.source_records(left).unwrap();
    assert_eq!(left_sources.len(), 2);
    assert_eq!(left_sources[0].sequence_id, SequenceId::new(0));
    assert_eq!(left_sources[1].sequence_id, SequenceId::new(1));
    assert_eq!(graph.nodes()[left.to_usize()].weight, Weight::new(2));
    assert!(graph.validate().is_valid());
}

#[test]
fn packed32_source_storage_round_trips_records() {
    let mut graph = FtoDag::with_source_storage(1, SourceStorageStrategy::Packed32);
    let node = graph.add_node(key(1), NodeKind::Singleton).unwrap();
    let records = [
        SourceRecord {
            sequence_id: SequenceId::new(12),
            position: SourcePosition::new(34),
        },
        SourceRecord {
            sequence_id: SequenceId::new(56),
            position: SourcePosition::new(78),
        },
    ];

    for record in records {
        graph.add_source_record(node, record).unwrap();
    }

    assert_eq!(
        graph.source_storage_strategy(),
        SourceStorageStrategy::Packed32
    );
    assert!(graph.retains_source_records());
    assert_eq!(graph.source_record_count(node).unwrap(), records.len());
    assert_eq!(graph.source_records(node).unwrap(), records);
    assert!(graph.validate().is_valid());
}

#[test]
fn count_only_source_storage_tracks_counts_without_records() {
    let mut graph = FtoDag::with_source_storage(1, SourceStorageStrategy::CountOnly);
    let node = graph.add_node(key(1), NodeKind::Singleton).unwrap();
    graph
        .add_source_record(
            node,
            SourceRecord {
                sequence_id: SequenceId::new(0),
                position: SourcePosition::new(10),
            },
        )
        .unwrap();
    graph
        .add_source_record(
            node,
            SourceRecord {
                sequence_id: SequenceId::new(1),
                position: SourcePosition::new(11),
            },
        )
        .unwrap();

    assert_eq!(
        graph.source_storage_strategy(),
        SourceStorageStrategy::CountOnly
    );
    assert_eq!(graph.source_record_count(node).unwrap(), 2);
    assert!(graph.source_records(node).is_err());
    assert!(graph.validate().is_valid());
}

#[test]
fn trace_path_source_storage_tracks_sequence_paths_without_node_records() {
    let mut graph = FtoDag::with_source_storage(1, SourceStorageStrategy::TracePaths);
    let left = graph.add_node(key(1), NodeKind::Start).unwrap();
    let right = graph.add_node(key(2), NodeKind::End).unwrap();
    graph
        .add_source_record(
            left,
            SourceRecord {
                sequence_id: SequenceId::new(0),
                position: SourcePosition::new(0),
            },
        )
        .unwrap();
    graph
        .add_source_record(
            right,
            SourceRecord {
                sequence_id: SequenceId::new(0),
                position: SourcePosition::new(1),
            },
        )
        .unwrap();

    assert_eq!(
        graph.source_storage_strategy(),
        SourceStorageStrategy::TracePaths
    );
    assert!(!graph.retains_source_records());
    assert!(graph.retains_sequence_trace_paths());
    assert_eq!(graph.source_record_count(left).unwrap(), 1);
    assert!(graph.source_records(left).is_err());
    assert_eq!(
        graph.sequence_trace_path(SequenceId::new(0)).unwrap(),
        &[left, right]
    );
    assert!(graph.validate().is_valid());
}

#[test]
fn validation_reports_duplicate_sequence_sources() {
    let mut graph = FtoDag::new(1);
    let node = graph.add_node(key(1), NodeKind::Singleton).unwrap();
    for position in [0, 1] {
        graph
            .add_source_record(
                node,
                SourceRecord {
                    sequence_id: SequenceId::new(7),
                    position: SourcePosition::new(position),
                },
            )
            .unwrap();
    }

    let report = graph.validate();
    assert!(
        report
            .errors
            .contains(&GraphValidationError::DuplicateSequenceSource {
                node: node.to_usize(),
                sequence: 7,
            })
    );
}

#[test]
fn validation_reports_cycles() {
    let mut graph = FtoDag::new(1);
    let left = graph.add_node(key(1), NodeKind::Start).unwrap();
    let right = graph.add_node(key(2), NodeKind::End).unwrap();
    graph
        .add_or_increment_edge(left, right, Weight::new(1))
        .unwrap();
    graph
        .add_or_increment_edge(right, left, Weight::new(1))
        .unwrap();

    let report = graph.validate();
    assert!(report.errors.contains(&GraphValidationError::CycleDetected));
}
