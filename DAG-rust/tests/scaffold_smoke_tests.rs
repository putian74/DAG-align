use dag_rust::prelude::*;

#[test]
fn public_prelude_supports_basic_data_flow() {
    let alphabet = BuiltinAlphabet::new(
        AlphabetKind::Dna,
        NormalizationPolicy::CaseFold,
        AmbiguityPolicy::ExactSymbol,
    );
    let record = SequenceRecord::new("seq1", "ACGT");
    let encoded = EncodedSequence::encode(record, &alphabet).expect("sequence encodes");
    assert_eq!(encoded.len(), 4);

    let encoder = DefaultFragmentEncoder::packed(BitWidth::new(4).expect("valid width"));
    let windows = FragmentWindows::new(&encoded, 2, &encoder)
        .expect("valid fragment windows")
        .collect::<Result<Vec<_>>>()
        .expect("windows encode");
    assert_eq!(windows.len(), 3);

    let mut graph = FtoDag::new(2);
    let left = graph
        .add_node(windows[0].1.clone(), NodeKind::Start)
        .expect("left node inserted");
    let right = graph
        .add_node(windows[1].1.clone(), NodeKind::End)
        .expect("right node inserted");
    graph
        .add_or_increment_edge(left, right, Weight::new(1))
        .expect("edge inserted");

    let report = graph.validate();
    assert!(report.is_valid(), "{report:?}");
    assert_eq!(graph.stats().node_count, 2);
    assert_eq!(graph.stats().edge_count, 1);
}

#[test]
fn cli_help_smoke_test() {
    let code = dag_rust::interfaces::cli::run_from(["dag-rust", "--help"]).expect("help works");
    assert_eq!(code, 0);
}
