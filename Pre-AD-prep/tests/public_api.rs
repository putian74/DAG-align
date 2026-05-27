use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use pre_ad_prep::{
    ArraySpec, ArtifactValidationLevel, CoordinateMode, DataType, InitializationBundle,
    InitializationTrack, LegacyAdapter, LegacyConversionOptions, LegacyDagAlignAdapter,
    LegacyDagAlignInput, PackedStateWindows, PathAlignmentOp, PreAdPrepError,
    ReferenceMsaColumnKind, ReferenceMsaConfig, ReferencePath, RepresentativeGlobalGraphConfig,
    SequenceTable, SourceFormat, SourceRecordTable, StateInterval, StateIntervalSemantics,
    TensorGraph, TensorGraphArtifact, TensorGraphManifest, Validate, WeightedSequencePath,
    WindowBuildConfig, build_edge_window_overlaps, build_global_coordinates, build_packed_windows,
    build_reference_msa, build_representative_global_graph, read_npy_2d_f64,
    select_max_weight_reference_path, validate_tensor_graph_artifact,
    write_representative_global_graph_artifact,
};

#[test]
fn tensor_graph_public_api_validates_minimal_dag() {
    let graph = TensorGraph::new(
        vec![0, 1],
        vec![1.0, 1.0],
        vec![0],
        vec![1],
        vec![1.0],
        vec![0, 1],
    );

    graph.validate_with_global_states(None).unwrap();
}

#[test]
fn packed_state_windows_use_half_open_lengths() {
    let windows = PackedStateWindows::from_intervals(vec![
        StateInterval::new(0, 2),
        StateInterval::new(2, 5),
    ]);

    assert_eq!(windows.lengths, vec![2, 3]);
    windows.validate(5).unwrap();
    assert!(StateInterval::new(6, 6).validate(5).is_err());
}

#[test]
fn manifest_exposes_ad_phmm_contract_names() {
    let mut manifest =
        TensorGraphManifest::new_v1(SourceFormat::DagAlignLegacy, 2, 1).with_global_state_count(5);
    manifest.add_array(ArraySpec::new(
        "node_window_left",
        "coordinates/node_window_left.npy",
        DataType::U64,
        vec![2],
    ));

    assert_eq!(manifest.format_name, "ad_phmm_tensor_graph");
    assert_eq!(manifest.format_version, 1);
    assert_eq!(
        manifest.state_interval_semantics,
        StateIntervalSemantics::HalfOpen
    );
    assert!(manifest.require_array("node_window_left").is_some());
}

#[test]
fn manifest_round_trips_through_json() {
    let root = unique_temp_dir("pre_ad_prep_manifest");
    fs::create_dir_all(&root).unwrap();
    let manifest_path = root.join("manifest.json");

    let mut manifest =
        TensorGraphManifest::new_v1(SourceFormat::Synthetic, 2, 1).with_sequence_count(3);
    manifest = manifest.with_source_graph_dir("/tmp/source-graph");
    manifest.symbol_encoding = vec![("A".into(), 0), ("C".into(), 1)];
    manifest.legacy_metadata = vec![("edge_sort".into(), "src_dst_weight".into())];
    manifest.add_array(ArraySpec::new(
        "node_symbol",
        "graph/node_symbol.npy",
        DataType::U16,
        vec![2],
    ));

    manifest.write_to_path(&manifest_path).unwrap();
    let round_trip = TensorGraphManifest::read_from_path(&manifest_path).unwrap();

    assert_eq!(round_trip.source_format, SourceFormat::Synthetic);
    assert_eq!(round_trip.sequence_count, 3);
    assert_eq!(
        round_trip.source_graph_dir,
        Some(PathBuf::from("/tmp/source-graph"))
    );
    assert!(
        round_trip
            .symbol_encoding
            .iter()
            .any(|(symbol, value)| symbol == "A" && *value == 0)
    );
    assert_eq!(
        round_trip.legacy_metadata,
        vec![("edge_sort".into(), "src_dst_weight".into())]
    );

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn sequence_and_source_tables_validate_basic_shapes() {
    let sequences = SequenceTable {
        sequence_ids: vec![0, 1],
        sequence_name_offsets: vec![0, 4, 8],
        sequence_name_bytes: b"seqAseqB".to_vec(),
    };
    sequences.validate().unwrap().into_result().unwrap();

    let records = SourceRecordTable {
        packed_records: vec![1, 2, 3],
        source_sequence_id: Some(vec![0, 0, 1]),
        source_position: Some(vec![10, 11, 20]),
        ..SourceRecordTable::default()
    };
    records.validate().unwrap().into_result().unwrap();
}

#[test]
fn global_coordinates_follow_reference_path_order() {
    let graph = TensorGraph::new(
        vec![0, 1, 2],
        vec![1.0, 1.0, 1.0],
        vec![0, 1],
        vec![1, 2],
        vec![1.0, 1.0],
        vec![0, 1, 2],
    );
    let output = build_global_coordinates(
        &graph,
        ReferencePath {
            node_ids: vec![Some(0), Some(1), Some(2)],
        },
        CoordinateMode::Hmm,
    )
    .unwrap();

    assert_eq!(output.global_state_count, 3);
    assert_eq!(output.node_intervals[0], StateInterval::new(0, 1));
    assert_eq!(output.node_intervals[1], StateInterval::new(1, 2));
    assert_eq!(output.node_intervals[2], StateInterval::new(2, 2));
}

#[test]
fn packed_windows_and_edge_overlaps_follow_intersections() {
    let graph = TensorGraph::new(
        vec![0, 1, 2],
        vec![1.0, 1.0, 1.0],
        vec![0, 1],
        vec![1, 2],
        vec![1.0, 1.0],
        vec![0, 1, 2],
    );
    let coordinates = build_global_coordinates(
        &graph,
        ReferencePath {
            node_ids: vec![Some(0), Some(1), Some(2)],
        },
        CoordinateMode::Hmm,
    )
    .unwrap();
    let windows = build_packed_windows(
        &coordinates,
        WindowBuildConfig {
            left_padding: 0,
            right_padding: 1,
        },
    )
    .unwrap();
    let overlaps = build_edge_window_overlaps(&graph, &windows).unwrap();

    assert_eq!(windows.intervals[0], StateInterval::new(0, 2));
    assert_eq!(windows.intervals[1], StateInterval::new(1, 3));
    assert_eq!(overlaps.overlaps.len(), 2);
    assert_eq!(overlaps.overlaps[0].src_offset, 1);
    assert_eq!(overlaps.overlaps[0].dst_offset, 0);
    assert_eq!(overlaps.overlaps[0].len, 1);
}

#[test]
fn reference_msa_keeps_preprocessing_logic_in_pre_ad_prep() {
    let paths = vec![
        WeightedSequencePath {
            representative_sequence_id: 0,
            sequence_ids: vec![0, 1],
            node_ids: vec![0, 1, 2, 3],
            symbols: vec![0, 1, 2, 3],
            weight: 2,
        },
        WeightedSequencePath {
            representative_sequence_id: 2,
            sequence_ids: vec![2],
            node_ids: vec![10, 12, 13],
            symbols: vec![0, 2, 3],
            weight: 1,
        },
        WeightedSequencePath {
            representative_sequence_id: 3,
            sequence_ids: vec![3],
            node_ids: vec![20, 21, 22, 23, 24],
            symbols: vec![0, 1, 3, 2, 3],
            weight: 1,
        },
    ];

    let reference_index = select_max_weight_reference_path(&paths).unwrap();
    assert_eq!(reference_index, 0);

    let result =
        build_reference_msa(&paths, reference_index, &ReferenceMsaConfig::default()).unwrap();
    assert_eq!(
        result.reference_path.node_ids,
        vec![Some(0), Some(1), Some(2), Some(3)]
    );
    assert_eq!(result.total_weight, 4);
    assert_eq!(result.longest_insertions, vec![0, 0, 1, 0, 0]);
    assert_eq!(
        result.columns[2].kind,
        ReferenceMsaColumnKind::Insert {
            anchor_index: 2,
            slot: 0
        }
    );
    assert_eq!(result.columns[2].primary_node_id, Some(22));
    assert_eq!(result.columns[1].gap_fraction_basis_points, 2500);
    assert_eq!(result.columns[2].gap_fraction_basis_points, 7500);
    assert_eq!(
        result.state_bands.minimum.included_column_indices,
        vec![0, 1, 3, 4]
    );
    assert_eq!(
        result.state_bands.canonical.included_column_indices,
        vec![0, 1, 3, 4]
    );
    assert_eq!(
        result.state_bands.maximum.included_column_indices,
        vec![0, 1, 3, 4]
    );

    let insertion_alignment = result
        .path_alignments
        .iter()
        .find(|alignment| alignment.sequence_ids == vec![3])
        .unwrap();
    assert!(insertion_alignment.operations.iter().any(|operation| {
        matches!(
            operation,
            PathAlignmentOp::Insert {
                anchor_index: 2,
                slot: 0,
                ..
            }
        )
    }));
}

#[test]
fn reference_msa_reference_path_feeds_coordinate_builder() {
    let graph = TensorGraph::new(
        vec![0, 1, 2, 3],
        vec![1.0, 1.0, 1.0, 1.0],
        vec![0, 1, 2],
        vec![1, 2, 3],
        vec![1.0, 1.0, 1.0],
        vec![0, 1, 2, 3],
    );
    let result = build_reference_msa(
        &[WeightedSequencePath {
            representative_sequence_id: 0,
            sequence_ids: vec![0],
            node_ids: vec![0, 1, 2, 3],
            symbols: vec![0, 1, 2, 3],
            weight: 1,
        }],
        0,
        &ReferenceMsaConfig::default(),
    )
    .unwrap();

    let coordinates =
        build_global_coordinates(&graph, result.reference_path.clone(), CoordinateMode::Hmm)
            .unwrap();
    assert_eq!(coordinates.global_state_count, 4);
    assert_eq!(coordinates.reference_path, result.reference_path);
    assert_eq!(coordinates.node_intervals[0], StateInterval::new(0, 1));
    assert_eq!(coordinates.node_intervals[3], StateInterval::new(3, 3));
}

#[test]
fn representative_global_graph_exports_reference_msa_initialization() {
    let paths = vec![
        WeightedSequencePath {
            representative_sequence_id: 0,
            sequence_ids: vec![0, 1, 2, 3, 4],
            node_ids: vec![10, 11, 12],
            symbols: vec![0, 1, 2],
            weight: 5,
        },
        WeightedSequencePath {
            representative_sequence_id: 5,
            sequence_ids: vec![5, 6],
            node_ids: vec![20, 21, 22],
            symbols: vec![0, 3, 2],
            weight: 2,
        },
    ];
    let config = RepresentativeGlobalGraphConfig {
        source_format: SourceFormat::Synthetic,
        alphabet: vec!['A', 'B', 'C', 'D'],
        symbol_encoding: vec![
            ("A".into(), 0),
            ("B".into(), 1),
            ("C".into(), 2),
            ("D".into(), 3),
        ],
        ..RepresentativeGlobalGraphConfig::default()
    };

    let build = build_representative_global_graph(&paths, &config).unwrap();

    assert_eq!(build.graph.node_symbol, vec![0, 1, 2, 3, 2]);
    assert_eq!(build.graph.node_weight, vec![7.0, 5.0, 5.0, 2.0, 2.0]);
    assert_eq!(build.coordinates.global_state_count, 3);
    assert_eq!(
        build.reference_msa.reference_path.node_ids,
        vec![Some(0), Some(1), Some(2)]
    );

    let root = unique_temp_dir("pre_ad_prep_representative_graph");
    let output_dir = root.join("tensor_graph.v1");
    fs::create_dir_all(&output_dir).unwrap();
    let exported =
        write_representative_global_graph_artifact(&output_dir, &build, &config).unwrap();

    assert_eq!(
        exported.initialization.track,
        InitializationTrack::ReferenceMsa
    );
    assert!(output_dir.join("manifest.json").exists());
    assert!(
        output_dir
            .join("initialization/reference_msa/manifest.json")
            .exists()
    );

    let artifact = TensorGraphArtifact::read_manifest(&output_dir).unwrap();
    assert_eq!(artifact.manifest.sequence_count, 7);
    validate_tensor_graph_artifact(&artifact, ArtifactValidationLevel::TrainingReady)
        .unwrap()
        .into_result()
        .unwrap();
    let bundle = InitializationBundle::load_from_root(&output_dir).unwrap();
    let reference_msa = bundle.get(InitializationTrack::ReferenceMsa).unwrap();
    assert_eq!(reference_msa.global_state_count, 3);
    assert_eq!(reference_msa.alphabet_size, 4);
    assert!(reference_msa.metadata.iter().any(
        |(key, value)| key == "source" && value == "representative_global_graph_reference_msa"
    ));

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn legacy_input_paths_are_consistent() {
    let input = LegacyDagAlignInput::from_graph_dir(PathBuf::from("graph-dir"));
    let options = LegacyConversionOptions::default();

    assert_eq!(input.graph_pkl, PathBuf::from("graph-dir/graph.pkl"));
    assert_eq!(input.data_npz, PathBuf::from("graph-dir/data.npz"));
    assert!(!options.allow_python_object_bridge);
}

#[test]
fn legacy_adapter_call_signature_returns_explicit_unsupported() {
    let adapter = LegacyDagAlignAdapter;
    let input = LegacyDagAlignInput::from_graph_dir("graph-dir");
    let error = adapter
        .convert(
            &input,
            PathBuf::from("tensor_graph.v1"),
            LegacyConversionOptions::default(),
        )
        .unwrap_err();

    assert!(matches!(error, PreAdPrepError::Unsupported(_)));
}

#[test]
fn legacy_adapter_converts_tiny_dag_align_graph_with_object_bridge() {
    if !python_numpy_available() {
        eprintln!("skipping legacy bridge test because python3/numpy is unavailable");
        return;
    }
    let root = unique_temp_dir("pre_ad_prep_legacy_bridge");
    let graph_dir = root.join("legacy");
    let output_dir = root.join("tensor_graph.v1");
    fs::create_dir_all(&graph_dir).unwrap();
    fs::write(graph_dir.join("graph.pkl"), b"placeholder").unwrap();
    create_tiny_legacy_npz(&graph_dir);

    let adapter = LegacyDagAlignAdapter;
    let input = LegacyDagAlignInput::from_graph_dir(&graph_dir);
    let output = adapter
        .convert(
            &input,
            output_dir.clone(),
            LegacyConversionOptions {
                allow_python_object_bridge: true,
                ..LegacyConversionOptions::default()
            },
        )
        .unwrap();

    assert_eq!(output.graph.node_count(), 3);
    assert_eq!(output.graph.edge_count(), 2);
    assert_eq!(output.graph.topo_order, vec![0, 1, 2]);
    assert!(output.artifact.root.ends_with("tensor_graph.v1"));
    assert_eq!(
        output.artifact.manifest.source_graph_dir.as_deref(),
        Some(graph_dir.as_path())
    );
    assert!(
        output
            .artifact
            .manifest
            .symbol_encoding
            .iter()
            .any(|(symbol, value)| symbol == "A" && *value == 0)
    );
    assert!(output.diagnostics.profiling.is_some());
    assert_eq!(output.initializations.tracks.len(), 2);
    assert!(output_dir.join("manifest.json").exists());
    assert!(output_dir.join("graph/node_symbol.npy").exists());
    assert!(
        output_dir
            .join("coordinates/node_coordinate_left.npy")
            .exists()
    );
    assert!(output_dir.join("coordinates/node_window_left.npy").exists());
    assert!(
        output_dir
            .join("coordinates/edge_state_overlap_len.npy")
            .exists()
    );
    assert!(output_dir.join("reference/ref_node_ids.npy").exists());
    assert!(
        output_dir
            .join("initialization/legacy_current/manifest.json")
            .exists()
    );
    assert!(
        output_dir
            .join("initialization/reference_msa/manifest.json")
            .exists()
    );
    assert!(output_dir.join("source/sequence_id.npy").exists());
    assert!(output_dir.join("source/node_source_offset.npy").exists());
    assert!(output_dir.join("source/source_sequence_id.npy").exists());
    assert!(output_dir.join("source/source_position.npy").exists());
    assert!(
        output
            .artifact
            .manifest
            .require_array("sequence_id")
            .is_some()
    );
    assert!(
        output
            .artifact
            .manifest
            .require_array("source_packed")
            .is_some()
    );
    let reference_msa = output
        .initializations
        .get(InitializationTrack::ReferenceMsa)
        .unwrap();
    let legacy_current = output
        .initializations
        .get(InitializationTrack::LegacyCurrent)
        .unwrap();
    assert_eq!(legacy_current.global_state_count, 3);
    assert_eq!(legacy_current.alphabet_size, 6);
    assert!(
        legacy_current
            .metadata
            .iter()
            .any(|(key, value)| key == "normalized_to_graph_encoding" && value == "true")
    );
    assert_eq!(reference_msa.global_state_count, 3);
    assert_eq!(reference_msa.alphabet_size, 6);
    assert!(
        reference_msa
            .metadata
            .iter()
            .any(|(key, value)| key == "source" && value == "reference_path_msa")
    );
    let (legacy_match_rows, legacy_match_cols, legacy_match) =
        read_npy_2d_f64(output_dir.join("initialization/legacy_current/match_emission.npy"))
            .unwrap();
    assert_eq!((legacy_match_rows, legacy_match_cols), (3, 6));
    assert!((legacy_match[0].exp() - 0.9).abs() < 1e-12);
    assert!((legacy_match[1].exp() - 0.03).abs() < 1e-12);
    assert!((legacy_match[2].exp() - 0.02).abs() < 1e-12);
    assert!((legacy_match[3].exp() - 0.05).abs() < 1e-12);
    assert!(legacy_match[4] < -30.0);
    assert!(legacy_match[5] < -30.0);

    let artifact = TensorGraphArtifact::read_manifest(&output_dir).unwrap();
    let graph_report =
        validate_tensor_graph_artifact(&artifact, ArtifactValidationLevel::GraphCore).unwrap();
    graph_report.into_result().unwrap();

    let training_report =
        validate_tensor_graph_artifact(&artifact, ArtifactValidationLevel::TrainingReady).unwrap();
    training_report.into_result().unwrap();

    fs::remove_dir_all(root).unwrap();
}

fn python_numpy_available() -> bool {
    Command::new("python3")
        .arg("-c")
        .arg("import numpy")
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("{prefix}_{nanos}"))
}

fn create_tiny_legacy_npz(graph_dir: &Path) {
    let status = Command::new("python3")
        .arg("-c")
        .arg(
            r#"
import numpy as np
import pickle
from types import SimpleNamespace
from pathlib import Path
graph_dir = Path(__import__('sys').argv[1])
graph = SimpleNamespace(firstBitofOSM=2, allBitofOSM=4)
with (graph_dir / "graph.pkl").open("wb") as handle:
    pickle.dump(graph, handle)
edgeWeightDict = np.array([[0, 1, 2], [1, 2, 3]], dtype=object)
fragments = np.array(["AA", "AC", "AG"])
weights = np.array([2, 3, 4], dtype=np.uint32)
startNodeSet = np.array([0], dtype=np.uint32)
endNodeSet = np.array([2], dtype=np.uint32)
np.savez(
    graph_dir / "data.npz",
    edgeWeightDict=edgeWeightDict,
    fragments=fragments,
    weights=weights,
    startNodeSet=startNodeSet,
    endNodeSet=endNodeSet,
)
np.save(graph_dir / "v_id.npy", np.array([["seq1", "g_0"], ["seq2", "g_1"]], dtype=object))
np.save(
    graph_dir / "osm.npy",
    np.array([
        np.array([0, 4], dtype=np.uint64),
        np.array([1], dtype=np.uint64),
        np.array([2, 5], dtype=np.uint64),
    ], dtype=object),
)
np.savez(
    graph_dir / "thr_0.01.npz",
    ref_seq=np.array("ATC"),
    ref_node_list=np.array([0, 1, 2], dtype=np.int64),
    emProbMatrix=np.array(
        [
            [0.9, 0.1, 0.1],
            [0.05, 0.8, 0.1],
            [0.03, 0.05, 0.7],
            [0.02, 0.05, 0.1],
        ],
        dtype=np.float64,
    ),
    insert_range=np.array([[1, 2]], dtype=np.int64),
)
ini_dir = graph_dir / "ini"
ini_dir.mkdir(exist_ok=True)
parameter_dict = {
    "_mm": np.array([-1.0, -0.1, -0.1, -0.2], dtype=np.float64),
    "_md": np.array([-1.0, -2.0, -2.0, -2.0], dtype=np.float64),
    "_mi": np.array([-1.0, -3.0, -3.0, -1.5], dtype=np.float64),
    "_dm": np.array([-np.inf, -0.5, -0.5, 0.0], dtype=np.float64),
    "_dd": np.array([-np.inf, -0.7, -0.7, -np.inf], dtype=np.float64),
    "_di": np.array([-np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float64),
    "_im": np.array([-0.2, -0.2, -0.2, -0.2], dtype=np.float64),
    "_id": np.array([-np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float64),
    "_ii": np.array([-0.6, -0.6, -0.6, -0.6], dtype=np.float64),
    "match_emission": np.log(
        np.array(
            [
                [0.9, 0.05, 0.03, 0.02],
                [0.1, 0.8, 0.05, 0.05],
                [0.1, 0.1, 0.7, 0.1],
            ],
            dtype=np.float64,
        )
    ),
    "insert_emission": np.log(np.full((4, 4), 0.25, dtype=np.float64)),
}
np.save(ini_dir / "init_demo.npy", parameter_dict)
"#,
        )
        .arg(graph_dir)
        .status()
        .unwrap();
    assert!(status.success());
}
