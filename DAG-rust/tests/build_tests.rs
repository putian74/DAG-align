use dag_rust::prelude::*;

#[test]
fn first_sequence_initialization_preserves_repeated_occurrences() {
    let alphabet = BuiltinAlphabet::dna_exact();
    let sequence = EncodedSequence::encode(SequenceRecord::new("repeat", "AAAAAA"), &alphabet)
        .expect("sequence encodes");
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let mut builder = FtoDagBuilder::new(BuildConfig::new(3));

    let result = builder
        .initialize_from_encoded(SequenceId::new(0), &sequence, &encoder)
        .expect("first sequence initializes");
    assert_eq!(result.node_path.len(), 4);
    assert_eq!(result.inserted_edges, 3);
    assert_eq!(result.provenance_records_added, 4);
    assert_eq!(result.block_plan.new_nodes, result.node_path);

    let graph = builder.graph();
    assert_eq!(graph.node_count(), 4);
    assert_eq!(graph.edge_count(), 3);
    assert_eq!(graph.endpoints().sequence_starts().len(), 1);
    assert_eq!(graph.endpoints().sequence_ends().len(), 1);

    let repeated_key = graph.nodes()[1].fragment.clone();
    let internal_repeats = graph
        .fragment_index()
        .nodes_for_stored(&repeated_key, NodeKind::Internal);
    assert_eq!(internal_repeats.len(), 2);
    assert_ne!(internal_repeats[0], internal_repeats[1]);

    for (position, node) in graph.nodes().iter().enumerate() {
        assert_eq!(node.weight, Weight::new(1));
        let provenance = graph
            .provenance_records(node.id)
            .expect("provenance range exists");
        assert_eq!(provenance.len(), 1);
        assert_eq!(provenance[0].sequence_id, SequenceId::new(0));
        assert_eq!(
            provenance[0].position,
            ProvenancePosition::new(position as u64)
        );
    }

    assert!(graph.validate().is_valid());
}

#[test]
fn first_sequence_initialization_populates_topology_coordinates() {
    let alphabet = BuiltinAlphabet::dna_canonical();
    let sequence =
        EncodedSequence::encode(SequenceRecord::new("linear", "ACGT"), &alphabet).unwrap();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let mut builder = FtoDagBuilder::new(BuildConfig::new(2));

    let result = builder
        .initialize_from_encoded(SequenceId::new(7), &sequence, &encoder)
        .unwrap();
    assert_eq!(result.node_path.len(), 3);

    let topology = DagTopology::try_from_graph(builder.graph()).expect("acyclic topology");
    let order = topology.topological_order();
    assert_eq!(order.len(), 3);
    assert_eq!(
        topology.forward_coordinate(order[0]).unwrap(),
        TopologicalCoordinate::new(1)
    );
    assert_eq!(
        topology.forward_coordinate(order[1]).unwrap(),
        TopologicalCoordinate::new(2)
    );
    assert_eq!(
        topology.forward_coordinate(order[2]).unwrap(),
        TopologicalCoordinate::new(3)
    );
    assert_eq!(
        topology.reverse_coordinate(order[2]).unwrap(),
        TopologicalCoordinate::new(1)
    );
}

#[test]
fn initial_match_threshold_rejects_low_similarity_without_mutation() {
    let alphabet = BuiltinAlphabet::dna_canonical();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let seed = EncodedSequence::encode(SequenceRecord::new("seed", "ACGTACGT"), &alphabet).unwrap();
    let distant =
        EncodedSequence::encode(SequenceRecord::new("distant", "TTTTTTTT"), &alphabet).unwrap();
    let mut config = BuildConfig::new(3);
    config.min_initial_match_ratio = Some(SimilarityThreshold::from_basis_points(5_000).unwrap());
    config.graph_id = GraphId::new(42);
    let mut builder = FtoDagBuilder::new(config);
    builder
        .initialize_from_encoded(SequenceId::new(0), &seed, &encoder)
        .unwrap();
    let before = builder.graph().stats();

    let decision = builder
        .initial_match_decision(SequenceId::new(1), &distant, &encoder)
        .unwrap();

    assert_eq!(
        decision,
        IntegrationDecision::Reject(RejectedSequence {
            sequence_id: SequenceId::new(1),
            subgraph_id: GraphId::new(42),
        })
    );
    assert_eq!(builder.graph().stats(), before);
}

#[test]
fn add_sequence_records_minimal_rejection_report() {
    let alphabet = BuiltinAlphabet::dna_canonical();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let seed = EncodedSequence::encode(SequenceRecord::new("seed", "ACGTACGT"), &alphabet).unwrap();
    let distant =
        EncodedSequence::encode(SequenceRecord::new("distant", "TTTTTTTT"), &alphabet).unwrap();
    let mut config = BuildConfig::new(3);
    config.graph_id = GraphId::new(9);
    config.min_initial_match_ratio = Some(SimilarityThreshold::from_basis_points(5_000).unwrap());
    let mut builder = FtoDagBuilder::new(config);
    builder
        .initialize_from_encoded(SequenceId::new(0), &seed, &encoder)
        .unwrap();

    let outcome = builder
        .add_sequence_from_encoded(SequenceId::new(3), &distant, &encoder)
        .unwrap();

    assert_eq!(
        outcome,
        SequenceBuildOutcome::Rejected(RejectedSequence {
            sequence_id: SequenceId::new(3),
            subgraph_id: GraphId::new(9),
        })
    );
    assert_eq!(
        builder.report().rejected_sequences,
        vec![RejectedSequence {
            sequence_id: SequenceId::new(3),
            subgraph_id: GraphId::new(9),
        }]
    );
    assert!(builder.report().integrated_sequences.is_empty());
    assert_eq!(builder.report().attempted_sequences, 1);
    assert_eq!(builder.report().total_nodes_created, 0);
}

#[test]
fn build_from_input_encodes_records_and_returns_cumulative_report() {
    let alphabet = BuiltinAlphabet::dna_canonical();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let mut config = BuildConfig::new(3);
    config.graph_id = GraphId::new(11);
    config.min_initial_match_ratio = Some(SimilarityThreshold::from_basis_points(2_000).unwrap());
    let mut builder = FtoDagBuilder::new(config);
    let mut input = VecSequenceInput::new(vec![
        SequenceRecord::new("seed", "ACGTACGT"),
        SequenceRecord::new("similar", "ACGTTCGT"),
        SequenceRecord::new("distant", "TTTTTTTT"),
    ]);

    let report = builder
        .build_from_input(&mut input, &alphabet, &encoder)
        .unwrap();

    assert_eq!(
        report.integrated_sequences,
        vec![SequenceId::new(0), SequenceId::new(1)]
    );
    assert_eq!(
        report.rejected_sequences,
        vec![RejectedSequence {
            sequence_id: SequenceId::new(2),
            subgraph_id: GraphId::new(11),
        }]
    );
    assert_eq!(report.attempted_sequences, 3);
    assert_eq!(report.total_nodes_created, builder.graph().node_count());
    assert_eq!(report.total_provenance_records_added, 12);
    assert!(builder.graph().validate().is_valid());
}

#[test]
fn initial_match_threshold_allows_similar_sequence() {
    let alphabet = BuiltinAlphabet::dna_canonical();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let seed = EncodedSequence::encode(SequenceRecord::new("seed", "ACGTACGT"), &alphabet).unwrap();
    let similar =
        EncodedSequence::encode(SequenceRecord::new("similar", "ACGTTCGT"), &alphabet).unwrap();
    let mut config = BuildConfig::new(3);
    config.min_initial_match_ratio = Some(SimilarityThreshold::from_basis_points(2_000).unwrap());
    let mut builder = FtoDagBuilder::new(config);
    builder
        .initialize_from_encoded(SequenceId::new(0), &seed, &encoder)
        .unwrap();

    let decision = builder
        .initial_match_decision(SequenceId::new(1), &similar, &encoder)
        .unwrap();

    assert_eq!(decision, IntegrationDecision::Proceed);
}

#[test]
fn collect_anchor_candidates_includes_coordinates_and_weights() {
    let alphabet = BuiltinAlphabet::dna_canonical();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let seed = EncodedSequence::encode(SequenceRecord::new("seed", "ACGTACGT"), &alphabet).unwrap();
    let query =
        EncodedSequence::encode(SequenceRecord::new("query", "ACGTTCGT"), &alphabet).unwrap();
    let mut builder = FtoDagBuilder::new(BuildConfig::new(3));
    builder
        .initialize_from_encoded(SequenceId::new(0), &seed, &encoder)
        .unwrap();
    let topology = DagTopology::try_from_graph(builder.graph()).unwrap();

    let candidate_sets = builder
        .collect_anchor_candidates(&query, &encoder, Some(&topology))
        .unwrap();

    assert_eq!(candidate_sets.len(), 6);
    assert_eq!(candidate_sets[0].position, 0);
    assert_eq!(candidate_sets[0].kind, NodeKind::Start);
    assert_eq!(candidate_sets[0].candidates.len(), 1);
    assert_eq!(candidate_sets[0].candidates[0].weight, Weight::new(1));
    assert_eq!(
        candidate_sets[0].candidates[0].coordinate,
        Some(TopologicalCoordinate::new(1))
    );
    assert!(
        candidate_sets.iter().any(|set| set.candidates.is_empty()),
        "SNP-like query should leave at least one unmatched fragment"
    );
}

#[test]
fn monotone_anchor_selection_keeps_increasing_injective_chain() {
    let n0 = NodeId::new(0);
    let n1 = NodeId::new(1);
    let n2 = NodeId::new(2);
    let candidate_sets = vec![
        AnchorCandidateSet {
            position: 0,
            kind: NodeKind::Internal,
            candidates: vec![AnchorCandidate {
                node: n1,
                kind: NodeKind::Internal,
                coordinate: Some(TopologicalCoordinate::new(2)),
                reverse_coordinate: None,
                weight: Weight::new(10),
            }],
        },
        AnchorCandidateSet {
            position: 1,
            kind: NodeKind::Internal,
            candidates: vec![
                AnchorCandidate {
                    node: n0,
                    kind: NodeKind::Internal,
                    coordinate: Some(TopologicalCoordinate::new(1)),
                    reverse_coordinate: None,
                    weight: Weight::new(20),
                },
                AnchorCandidate {
                    node: n2,
                    kind: NodeKind::Internal,
                    coordinate: Some(TopologicalCoordinate::new(3)),
                    reverse_coordinate: None,
                    weight: Weight::new(1),
                },
            ],
        },
    ];

    let path = select_monotone_anchor_path(&candidate_sets);

    assert_eq!(
        path.decisions,
        vec![AnchorDecision::Matched(n1), AnchorDecision::Matched(n2)]
    );
}

#[test]
fn continuity_scoring_prefers_existing_path_over_isolated_high_weight_hits() {
    let fragment = FragmentKey::symbols(vec![SymbolId::new(0)]);
    let mut graph = FtoDag::new(1);
    let coherent_left = graph
        .add_node(fragment.clone(), NodeKind::Internal)
        .unwrap();
    let coherent_right = graph
        .add_node(fragment.clone(), NodeKind::Internal)
        .unwrap();
    let isolated_left = graph
        .add_node(fragment.clone(), NodeKind::Internal)
        .unwrap();
    let isolated_right = graph
        .add_node(fragment.clone(), NodeKind::Internal)
        .unwrap();
    graph
        .add_or_increment_edge(coherent_left, coherent_right, Weight::new(1))
        .unwrap();
    let candidate_sets = vec![
        AnchorCandidateSet {
            position: 0,
            kind: NodeKind::Internal,
            candidates: vec![
                AnchorCandidate {
                    node: coherent_left,
                    kind: NodeKind::Internal,
                    coordinate: Some(TopologicalCoordinate::new(1)),
                    reverse_coordinate: None,
                    weight: Weight::new(1),
                },
                AnchorCandidate {
                    node: isolated_left,
                    kind: NodeKind::Internal,
                    coordinate: Some(TopologicalCoordinate::new(1)),
                    reverse_coordinate: None,
                    weight: Weight::new(100),
                },
            ],
        },
        AnchorCandidateSet {
            position: 1,
            kind: NodeKind::Internal,
            candidates: vec![
                AnchorCandidate {
                    node: coherent_right,
                    kind: NodeKind::Internal,
                    coordinate: Some(TopologicalCoordinate::new(2)),
                    reverse_coordinate: None,
                    weight: Weight::new(1),
                },
                AnchorCandidate {
                    node: isolated_right,
                    kind: NodeKind::Internal,
                    coordinate: Some(TopologicalCoordinate::new(2)),
                    reverse_coordinate: None,
                    weight: Weight::new(100),
                },
            ],
        },
    ];

    let unscored = select_monotone_anchor_path(&candidate_sets);
    let scored = select_monotone_anchor_path_with_graph(&candidate_sets, &graph);

    assert_eq!(
        unscored.decisions,
        vec![
            AnchorDecision::Matched(isolated_left),
            AnchorDecision::Matched(isolated_right),
        ]
    );
    assert_eq!(
        scored.decisions,
        vec![
            AnchorDecision::Matched(coherent_left),
            AnchorDecision::Matched(coherent_right),
        ]
    );
}

#[test]
fn bounded_reuse_prefers_coordinate_safe_existing_path_candidate() {
    let fragment = FragmentKey::symbols(vec![SymbolId::new(0)]);
    let mut graph = FtoDag::new(1);
    let previous = graph
        .add_node(fragment.clone(), NodeKind::Internal)
        .unwrap();
    let coherent = graph
        .add_node(fragment.clone(), NodeKind::Internal)
        .unwrap();
    let isolated = graph
        .add_node(fragment.clone(), NodeKind::Internal)
        .unwrap();
    let next = graph
        .add_node(fragment.clone(), NodeKind::Internal)
        .unwrap();
    graph
        .add_or_increment_edge(previous, coherent, Weight::new(2))
        .unwrap();
    graph
        .add_or_increment_edge(coherent, next, Weight::new(2))
        .unwrap();
    let candidate_set = AnchorCandidateSet {
        position: 1,
        kind: NodeKind::Internal,
        candidates: vec![
            AnchorCandidate {
                node: coherent,
                kind: NodeKind::Internal,
                coordinate: Some(TopologicalCoordinate::new(2)),
                reverse_coordinate: None,
                weight: Weight::new(1),
            },
            AnchorCandidate {
                node: isolated,
                kind: NodeKind::Internal,
                coordinate: Some(TopologicalCoordinate::new(3)),
                reverse_coordinate: None,
                weight: Weight::new(100),
            },
        ],
    };

    let reuse = select_bounded_reuse_candidate(
        &candidate_set,
        CoordinateInterval {
            left: Some(TopologicalCoordinate::new(1)),
            right: Some(TopologicalCoordinate::new(4)),
        },
        &[],
        &graph,
        Some(previous),
        Some(next),
    )
    .expect("bounded reuse candidate exists");

    assert_eq!(reuse.node, coherent);
    assert!(
        select_bounded_reuse_candidate(
            &candidate_set,
            CoordinateInterval {
                left: Some(TopologicalCoordinate::new(2)),
                right: Some(TopologicalCoordinate::new(4)),
            },
            &[],
            &graph,
            Some(previous),
            Some(next),
        )
        .is_some_and(|candidate| candidate.node == isolated)
    );
    assert!(
        select_bounded_reuse_candidate(
            &candidate_set,
            CoordinateInterval {
                left: Some(TopologicalCoordinate::new(1)),
                right: Some(TopologicalCoordinate::new(4)),
            },
            &[coherent, isolated],
            &graph,
            Some(previous),
            Some(next),
        )
        .is_none()
    );
}

#[test]
fn add_sequence_reuses_monotone_anchors_and_inserts_unanchored_block() {
    let alphabet = BuiltinAlphabet::dna_canonical();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let seed = EncodedSequence::encode(SequenceRecord::new("seed", "ACGTACGT"), &alphabet).unwrap();
    let query =
        EncodedSequence::encode(SequenceRecord::new("query", "ACGTTCGT"), &alphabet).unwrap();
    let mut builder = FtoDagBuilder::new(BuildConfig::new(3));
    builder
        .initialize_from_encoded(SequenceId::new(0), &seed, &encoder)
        .unwrap();

    let outcome = builder
        .add_sequence_from_encoded(SequenceId::new(1), &query, &encoder)
        .unwrap();
    let SequenceBuildOutcome::Integrated(result) = outcome else {
        panic!("similar sequence should integrate");
    };

    assert_eq!(result.node_path.len(), 6);
    assert_eq!(result.provenance_records_added, 6);
    assert_eq!(result.inserted_edges, 4);
    assert_eq!(builder.graph().node_count(), 9);
    assert_eq!(builder.graph().edge_count(), 9);
    assert_eq!(result.block_plan.new_nodes.len(), 3);
    assert!(result.block_plan.reused_nodes.is_empty());
    assert_eq!(result.block_plan.accepted_anchors.len(), 2);
    assert_eq!(
        result.block_plan.unanchored_blocks,
        vec![UnanchoredBlock {
            path_range: PathRange::new(2, 5),
            coordinate_interval: CoordinateInterval {
                left: Some(TopologicalCoordinate::new(2)),
                right: Some(TopologicalCoordinate::new(6)),
            },
        }]
    );
    assert_eq!(
        builder.report().integrated_sequences,
        vec![SequenceId::new(1)]
    );
    assert_eq!(builder.report().attempted_sequences, 1);
    assert_eq!(builder.report().total_nodes_created, 3);
    assert_eq!(builder.report().total_edges_inserted, 4);
    assert_eq!(builder.report().total_provenance_records_added, 6);
    assert!(builder.graph().validate().is_valid());
}

#[test]
fn add_sequence_summary_reports_counts_without_full_diagnostics() {
    let alphabet = BuiltinAlphabet::dna_canonical();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let seed = EncodedSequence::encode(SequenceRecord::new("seed", "ACGTACGT"), &alphabet).unwrap();
    let query =
        EncodedSequence::encode(SequenceRecord::new("query", "ACGTTCGT"), &alphabet).unwrap();
    let mut builder = FtoDagBuilder::new(BuildConfig::new(3));
    builder
        .initialize_from_encoded(SequenceId::new(0), &seed, &encoder)
        .unwrap();

    let outcome = builder
        .add_sequence_from_encoded_summary(SequenceId::new(1), &query, &encoder)
        .unwrap();
    let SequenceBuildSummaryOutcome::Integrated(result) = outcome else {
        panic!("similar sequence should integrate");
    };

    assert_eq!(result.sequence_id, SequenceId::new(1));
    assert_eq!(result.provenance_records_added, 6);
    assert_eq!(result.inserted_edges, 4);
    assert_eq!(result.new_nodes_created, 3);
    assert_eq!(result.reused_nodes, 0);
    assert_eq!(builder.graph().node_count(), 9);
    assert_eq!(builder.graph().edge_count(), 9);
    assert_eq!(
        builder.report().integrated_sequences,
        vec![SequenceId::new(1)]
    );
    assert_eq!(builder.report().attempted_sequences, 1);
    assert_eq!(builder.report().total_nodes_created, 3);
    assert_eq!(builder.report().total_edges_inserted, 4);
    assert_eq!(builder.report().total_provenance_records_added, 6);
    assert!(builder.graph().validate().is_valid());
}

#[test]
fn topology_update_strategies_match_on_small_build() {
    let alphabet = BuiltinAlphabet::dna_canonical();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let sequences = ["ACGTACGT", "ACGTTCGT", "ACGTTTGT", "ACGTACGT"];
    let mut full_config = BuildConfig::new(3);
    full_config.topology_update_strategy = TopologyUpdateStrategy::FullRebuild;
    full_config.collect_topology_counters = true;
    let mut affected_config = full_config;
    affected_config.topology_update_strategy = TopologyUpdateStrategy::IncrementalAffectedRegion;
    let mut forward_config = full_config;
    forward_config.topology_update_strategy = TopologyUpdateStrategy::IncrementalForwardOnly;
    let mut rescan_config = full_config;
    rescan_config.topology_update_strategy = TopologyUpdateStrategy::IncrementalForwardRescan;
    let mut relaxation_config = full_config;
    relaxation_config.topology_update_strategy =
        TopologyUpdateStrategy::IncrementalForwardRelaxation;
    let mut full_builder = FtoDagBuilder::new(full_config);
    let mut affected_builder = FtoDagBuilder::new(affected_config);
    let mut forward_builder = FtoDagBuilder::new(forward_config);
    let mut rescan_builder = FtoDagBuilder::new(rescan_config);
    let mut relaxation_builder = FtoDagBuilder::new(relaxation_config);

    for (index, sequence) in sequences.iter().enumerate() {
        let encoded = EncodedSequence::encode(
            SequenceRecord::new(format!("s{index}"), *sequence),
            &alphabet,
        )
        .unwrap();
        full_builder
            .add_sequence_from_encoded(SequenceId::try_from(index).unwrap(), &encoded, &encoder)
            .unwrap();
        affected_builder
            .add_sequence_from_encoded(SequenceId::try_from(index).unwrap(), &encoded, &encoder)
            .unwrap();
        forward_builder
            .add_sequence_from_encoded(SequenceId::try_from(index).unwrap(), &encoded, &encoder)
            .unwrap();
        rescan_builder
            .add_sequence_from_encoded(SequenceId::try_from(index).unwrap(), &encoded, &encoder)
            .unwrap();
        relaxation_builder
            .add_sequence_from_encoded(SequenceId::try_from(index).unwrap(), &encoded, &encoder)
            .unwrap();
    }

    assert_eq!(
        full_builder.graph().stats(),
        affected_builder.graph().stats()
    );
    assert_eq!(
        full_builder.graph().stats(),
        forward_builder.graph().stats()
    );
    assert_eq!(full_builder.graph().stats(), rescan_builder.graph().stats());
    assert_eq!(
        full_builder.graph().stats(),
        relaxation_builder.graph().stats()
    );
    assert_eq!(
        full_builder.report().integrated_sequences,
        affected_builder.report().integrated_sequences
    );
    assert_eq!(
        full_builder.report().integrated_sequences,
        forward_builder.report().integrated_sequences
    );
    assert_eq!(
        full_builder.report().integrated_sequences,
        rescan_builder.report().integrated_sequences
    );
    assert_eq!(
        full_builder.report().integrated_sequences,
        relaxation_builder.report().integrated_sequences
    );
    assert_eq!(
        full_builder.report().rejected_sequences,
        affected_builder.report().rejected_sequences
    );
    assert_eq!(
        full_builder.report().rejected_sequences,
        forward_builder.report().rejected_sequences
    );
    assert_eq!(
        full_builder.report().rejected_sequences,
        rescan_builder.report().rejected_sequences
    );
    assert_eq!(
        full_builder.report().rejected_sequences,
        relaxation_builder.report().rejected_sequences
    );
    assert!(full_builder.graph().validate().is_valid());
    assert!(affected_builder.graph().validate().is_valid());
    assert!(forward_builder.graph().validate().is_valid());
    assert!(rescan_builder.graph().validate().is_valid());
    assert!(relaxation_builder.graph().validate().is_valid());
    let full_topology = DagTopology::try_from_graph(full_builder.graph()).unwrap();
    let affected_topology = DagTopology::try_from_graph(affected_builder.graph()).unwrap();
    let forward_topology = DagTopology::try_from_graph(forward_builder.graph()).unwrap();
    let rescan_topology = DagTopology::try_from_graph(rescan_builder.graph()).unwrap();
    let relaxation_topology = DagTopology::try_from_graph(relaxation_builder.graph()).unwrap();
    assert_eq!(
        full_topology.topological_order().len(),
        affected_topology.topological_order().len()
    );
    assert_eq!(
        full_topology.topological_order().len(),
        forward_topology.topological_order().len()
    );
    assert_eq!(
        full_topology.topological_order().len(),
        rescan_topology.topological_order().len()
    );
    assert_eq!(
        full_topology.topological_order().len(),
        relaxation_topology.topological_order().len()
    );
    assert!(
        relaxation_builder
            .report()
            .topology_counters
            .forward_relax_attempts
            > 0
    );
    assert_eq!(
        forward_builder
            .report()
            .topology_counters
            .forward_relax_attempts,
        0
    );
    assert_eq!(
        rescan_builder
            .report()
            .topology_counters
            .forward_relax_attempts,
        0
    );
}

#[test]
fn add_identical_repeat_sequence_uses_distinct_repeated_nodes() {
    let alphabet = BuiltinAlphabet::dna_exact();
    let encoder =
        DefaultFragmentEncoder::packed(BitWidth::new(alphabet.bits_per_symbol()).unwrap());
    let sequence = EncodedSequence::encode(SequenceRecord::new("repeat", "AAAAAA"), &alphabet)
        .expect("sequence encodes");
    let mut builder = FtoDagBuilder::new(BuildConfig::new(3));
    builder
        .initialize_from_encoded(SequenceId::new(0), &sequence, &encoder)
        .unwrap();

    let outcome = builder
        .add_sequence_from_encoded(SequenceId::new(1), &sequence, &encoder)
        .unwrap();
    let SequenceBuildOutcome::Integrated(result) = outcome else {
        panic!("identical sequence should integrate");
    };

    assert_eq!(builder.graph().node_count(), 4);
    assert_eq!(builder.graph().edge_count(), 3);
    assert_eq!(result.block_plan.new_nodes.len(), 0);
    assert_eq!(result.block_plan.reused_nodes.len(), 0);
    assert_eq!(
        result.node_path,
        vec![
            NodeId::new(0),
            NodeId::new(1),
            NodeId::new(2),
            NodeId::new(3),
        ]
    );
    for node in builder.graph().nodes() {
        let provenance = builder.graph().provenance_records(node.id).unwrap();
        assert_eq!(provenance.len(), 2);
        assert_eq!(node.weight, Weight::new(2));
    }
    assert!(builder.graph().validate().is_valid());
}
