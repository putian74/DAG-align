//! Interfaces for current DAG-align legacy graph conversion.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Deserialize;

use crate::coordinates::{
    CoordinateMode, ReferencePath, WindowBuildConfig, build_edge_window_overlaps,
    build_global_coordinates, build_packed_windows,
};
use crate::diagnostics::{ConversionDiagnostics, Diagnostic, DiagnosticSeverity, ProfilingSummary};
use crate::error::{PreAdPrepError, Result};
use crate::export::{
    ArraySpec, DataType, SourceFormat, TensorGraphArtifact, TensorGraphManifest, read_npy_1d_u64,
    read_npy_2d_f64, write_npy_1d,
};
use crate::graph::{AdjacencyCsr, NodeFlags, TensorGraph};
use crate::init::{
    InitialPhmmManifest, InitializationBundle, InitializationTrack, TRANSITION_ORDER,
    write_initialization_track,
};
use crate::reference_msa::{
    ReferenceMsaConfig, ReferenceMsaResult, ReferenceMsaScoring,
    build_reference_msa_against_reference, collect_weighted_sequence_paths,
    emission_probability_tables, insertion_ranges,
};
use crate::validate::{ArtifactValidationLevel, validate_tensor_graph_artifact};

const LEGACY_PHMM_BASE_ORDER: [&str; 4] = ["A", "T", "C", "G"];
const LEGACY_SYMBOL_FLOOR_LOG_PROB: f64 = -36.841361487904734;
const LEGACY_SYMBOL_FLOOR_DESCRIPTION: &str = "log(1e-16)";

/// Input paths for one current DAG-align graph directory.
#[derive(Debug, Clone)]
pub struct LegacyDagAlignInput {
    pub graph_dir: PathBuf,
    pub graph_pkl: PathBuf,
    pub data_npz: PathBuf,
    pub osm_npy: Option<PathBuf>,
    pub onm_npy: Option<PathBuf>,
    pub onm_index_npy: Option<PathBuf>,
    pub v_id_npy: Option<PathBuf>,
    pub traceability_path_npy: Option<PathBuf>,
}

impl LegacyDagAlignInput {
    pub fn from_graph_dir(graph_dir: impl Into<PathBuf>) -> Self {
        let graph_dir = graph_dir.into();
        Self {
            graph_pkl: graph_dir.join("graph.pkl"),
            data_npz: graph_dir.join("data.npz"),
            osm_npy: Some(graph_dir.join("osm.npy")),
            onm_npy: Some(graph_dir.join("onm.npy")),
            onm_index_npy: Some(graph_dir.join("onm_index.npy")),
            v_id_npy: Some(graph_dir.join("v_id.npy")),
            traceability_path_npy: Some(graph_dir.join("Traceability_path.npy")),
            graph_dir,
        }
    }
}

/// Options controlling the transitional legacy adapter.
#[derive(Debug, Clone, Copy)]
pub struct LegacyConversionOptions {
    pub include_initialization: bool,
    pub allow_python_object_bridge: bool,
    pub require_state_windows: bool,
}

impl Default for LegacyConversionOptions {
    fn default() -> Self {
        Self {
            include_initialization: true,
            allow_python_object_bridge: false,
            require_state_windows: false,
        }
    }
}

/// Output produced by a legacy graph conversion.
#[derive(Debug, Clone)]
pub struct LegacyConversionOutput {
    pub graph: TensorGraph,
    pub artifact: TensorGraphArtifact,
    pub diagnostics: ConversionDiagnostics,
    pub initializations: InitializationBundle,
}

/// Public adapter interface for current DAG-align graph conversion.
pub trait LegacyAdapter {
    fn convert(
        &self,
        input: &LegacyDagAlignInput,
        output_dir: PathBuf,
        options: LegacyConversionOptions,
    ) -> Result<LegacyConversionOutput>;
}

/// Placeholder adapter. Concrete parsing is added after typed fixtures exist.
#[derive(Debug, Clone, Default)]
pub struct LegacyDagAlignAdapter;

impl LegacyAdapter for LegacyDagAlignAdapter {
    fn convert(
        &self,
        input: &LegacyDagAlignInput,
        output_dir: PathBuf,
        options: LegacyConversionOptions,
    ) -> Result<LegacyConversionOutput> {
        if !options.allow_python_object_bridge {
            return Err(PreAdPrepError::Unsupported(
                "current DAG-align data.npz uses pickle/object arrays; set allow_python_object_bridge=true for the transitional converter"
                    .into(),
            ));
        }
        validate_legacy_input(input)?;
        fs::create_dir_all(&output_dir)?;
        run_python_bridge(input, &output_dir)?;

        let core = read_graph_core(&output_dir.join("diagnostics").join("graph_core.json"))?;
        core.validate_contract()?;
        let mut graph = core.to_tensor_graph();

        let reference_core_path = output_dir.join("diagnostics").join("reference_core.json");
        let mut pending_reference_messages: Vec<(String, String)> = Vec::new();
        let reference_path = Some(derive_reference_path_from_graph(&graph));

        let mut global_state_count = None;
        if let Some(reference_path) = reference_path {
            let coordinates =
                build_global_coordinates(&graph, reference_path.clone(), CoordinateMode::Hmm)?;
            let windows = build_packed_windows(
                &coordinates,
                WindowBuildConfig {
                    left_padding: 100,
                    right_padding: 100,
                },
            )?;
            let overlaps = build_edge_window_overlaps(&graph, &windows)?;
            write_coordinate_artifacts(
                &output_dir,
                &coordinates,
                &windows,
                &overlaps,
                graph.edge_count(),
            )?;
            write_reference_artifacts_from_coordinates(&output_dir, &coordinates, &graph)?;
            graph.state_windows = Some(windows);
            graph.edge_overlaps = Some(overlaps);
            global_state_count = Some(coordinates.global_state_count);
            if options.include_initialization {
                if reference_core_path.exists() {
                    pending_reference_messages.push((
                        "legacy_reference_artifact_retained".into(),
                        "legacy thr_*.npz reference artifacts were preserved for diagnostics, but max-weight graph reference path is used for coordinates and reference_msa initialization".into(),
                    ));
                }
                if let Some(alphabet_size) = initialization_alphabet_size(&core, &graph) {
                    if try_write_reference_msa_initialization(
                        &output_dir,
                        &graph,
                        &coordinates.reference_path,
                        alphabet_size,
                    )? {
                        pending_reference_messages.push((
                            "reference_msa_generated".into(),
                            "reference-path MSA initialization was generated from decoded source provenance".into(),
                        ));
                    } else {
                        pending_reference_messages.push((
                            "reference_msa_fallback".into(),
                            "decoded source provenance was unavailable; preserved existing bootstrap reference_msa initialization if present".into(),
                        ));
                    }
                    pending_reference_messages.extend(normalize_legacy_initialization_tracks(
                        &output_dir,
                        alphabet_size,
                        &core.alphabet,
                        &core.symbol_encoding,
                    )?);
                }
            }
        } else if options.require_state_windows {
            return Err(PreAdPrepError::Validation(
                "state windows were required but no reference path could be constructed".into(),
            ));
        }
        graph.validate_with_global_states(global_state_count)?;

        let mut manifest = TensorGraphManifest::new_v1(
            SourceFormat::DagAlignLegacy,
            graph.node_count(),
            graph.edge_count(),
        )
        .with_sequence_count(core.sequence_count)
        .with_source_graph_dir(core.source_graph_dir.clone());
        manifest.alphabet = core
            .alphabet
            .iter()
            .filter_map(|symbol| symbol.chars().next())
            .collect();
        manifest.symbol_encoding = core
            .symbol_encoding
            .iter()
            .map(|(symbol, value)| (symbol.clone(), *value))
            .collect();
        manifest.legacy_metadata = core
            .legacy
            .iter()
            .map(|(key, value)| (key.clone(), json_value_to_metadata_string(value)))
            .collect();
        if let Some(global_state_count) = global_state_count {
            manifest.global_state_count = Some(global_state_count);
            manifest
                .legacy_metadata
                .push(("state_windows_present".into(), "true".into()));
            manifest
                .legacy_metadata
                .push(("window_padding".into(), "100".into()));
        }
        for spec in core.arrays {
            manifest.add_array(spec.try_into()?);
        }
        add_generated_manifest_arrays(
            &output_dir,
            &mut manifest,
            graph.node_count(),
            graph.edge_count(),
            global_state_count,
        )?;

        let mut diagnostics = ConversionDiagnostics::default();
        for diagnostic in core.diagnostics {
            diagnostics.report.push(diagnostic.into());
        }
        for (code, message) in pending_reference_messages {
            diagnostics.report.warning(code, message);
        }
        if global_state_count.is_some() {
            diagnostics.report.warning(
                "state_windows_generated",
                "state intervals, packed windows, and edge overlaps were exported during legacy conversion",
            );
        }
        diagnostics.profiling = core.profiling.map(Into::into);
        let initializations = if options.include_initialization {
            InitializationBundle::load_from_root(&output_dir)?
        } else {
            InitializationBundle::default()
        };
        if !initializations.tracks.is_empty() {
            diagnostics.report.warning(
                "initialization_exported",
                format!(
                    "exported initialization tracks: {}",
                    initializations
                        .tracks
                        .iter()
                        .map(|track| track.track.as_manifest_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            );
        }
        diagnostics.write_to_path(output_dir.join("diagnostics").join("conversion.json"))?;

        let artifact = TensorGraphArtifact::new(output_dir, manifest);
        artifact.write_manifest()?;
        validate_tensor_graph_artifact(
            &artifact,
            if global_state_count.is_some() {
                ArtifactValidationLevel::TrainingReady
            } else {
                ArtifactValidationLevel::GraphCore
            },
        )?
        .into_result()?;

        Ok(LegacyConversionOutput {
            graph,
            artifact,
            diagnostics,
            initializations,
        })
    }
}

fn validate_legacy_input(input: &LegacyDagAlignInput) -> Result<()> {
    if !input.graph_dir.exists() {
        return Err(PreAdPrepError::Validation(format!(
            "legacy graph directory does not exist: {}",
            input.graph_dir.display()
        )));
    }
    if !input.data_npz.exists() {
        return Err(PreAdPrepError::Validation(format!(
            "legacy graph data.npz does not exist: {}",
            input.data_npz.display()
        )));
    }
    if !input.graph_pkl.exists() {
        return Err(PreAdPrepError::Validation(format!(
            "legacy graph graph.pkl does not exist: {}",
            input.graph_pkl.display()
        )));
    }
    Ok(())
}

fn run_python_bridge(input: &LegacyDagAlignInput, output_dir: &Path) -> Result<()> {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tools")
        .join("legacy_bridge.py");
    let output = Command::new("python3")
        .arg(script)
        .arg("--graph-dir")
        .arg(&input.graph_dir)
        .arg("--output-dir")
        .arg(output_dir)
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(PreAdPrepError::Validation(format!(
            "legacy Python bridge failed with status {}. stdout: {} stderr: {}",
            output.status, stdout, stderr
        )));
    }
    Ok(())
}

fn read_graph_core(path: &Path) -> Result<GraphCoreJson> {
    let contents = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&contents)?)
}

#[derive(Debug, Deserialize)]
struct GraphCoreJson {
    node_symbol: Vec<u16>,
    node_weight: Vec<f32>,
    node_flags: Vec<u32>,
    edge_src: Vec<usize>,
    edge_dst: Vec<usize>,
    edge_weight: Vec<f32>,
    topo_order: Vec<usize>,
    source_format: String,
    source_graph_dir: PathBuf,
    sequence_count: usize,
    alphabet: Vec<String>,
    symbol_encoding: BTreeMap<String, u16>,
    state_interval_semantics: String,
    arrays: Vec<ArraySpecJson>,
    legacy: BTreeMap<String, serde_json::Value>,
    diagnostics: Vec<DiagnosticJson>,
    profiling: Option<ProfilingJson>,
}

impl GraphCoreJson {
    fn validate_contract(&self) -> Result<()> {
        if self.source_format != SourceFormat::DagAlignLegacy.as_manifest_str() {
            return Err(PreAdPrepError::Validation(format!(
                "legacy bridge emitted unexpected source_format: {}",
                self.source_format
            )));
        }
        if self.state_interval_semantics != "half_open" {
            return Err(PreAdPrepError::Validation(format!(
                "legacy bridge emitted unexpected state interval semantics: {}",
                self.state_interval_semantics
            )));
        }
        Ok(())
    }

    fn to_tensor_graph(&self) -> TensorGraph {
        let mut graph = TensorGraph::new(
            self.node_symbol.clone(),
            self.node_weight.clone(),
            self.edge_src.clone(),
            self.edge_dst.clone(),
            self.edge_weight.clone(),
            self.topo_order.clone(),
        );
        graph.node_flags = self.node_flags.iter().copied().map(NodeFlags).collect();
        graph.csr = Some(build_adjacency(
            graph.node_count(),
            &graph.edge_src,
            &graph.edge_dst,
        ));
        graph.csc = Some(build_adjacency(
            graph.node_count(),
            &graph.edge_dst,
            &graph.edge_src,
        ));
        graph
    }
}

fn derive_reference_path_from_graph(graph: &TensorGraph) -> ReferencePath {
    let node_count = graph.node_count();
    if node_count == 0 {
        return ReferencePath {
            node_ids: Vec::new(),
        };
    }

    let mut best_score = vec![f64::NEG_INFINITY; node_count];
    let mut best_parent = vec![None; node_count];
    for &node in &graph.topo_order {
        let mut score = f64::from(graph.node_weight[node]);
        let mut parent_choice = None;
        if let Some(csc) = &graph.csc {
            let start = csc.indptr[node];
            let end = csc.indptr[node + 1];
            let mut parent_best = f64::NEG_INFINITY;
            for index in start..end {
                let parent = csc.indices[index];
                let candidate = best_score[parent];
                if candidate > parent_best {
                    parent_best = candidate;
                    parent_choice = Some(parent);
                }
            }
            if parent_choice.is_some() {
                score += parent_best;
            }
        }
        best_score[node] = score;
        best_parent[node] = parent_choice;
    }
    let mut end_node = graph.topo_order[0];
    for &node in &graph.topo_order {
        if best_score[node] > best_score[end_node] {
            end_node = node;
        }
    }
    let mut path = Vec::new();
    let mut cursor = Some(end_node);
    while let Some(node) = cursor {
        path.push(Some(node));
        cursor = best_parent[node];
    }
    path.reverse();
    ReferencePath { node_ids: path }
}

fn write_coordinate_artifacts(
    output_dir: &Path,
    coordinates: &crate::coordinates::GlobalCoordinateOutput,
    windows: &crate::coordinates::PackedStateWindows,
    overlaps: &crate::coordinates::EdgeWindowOverlaps,
    edge_count: usize,
) -> Result<()> {
    let coordinates_dir = output_dir.join("coordinates");
    let node_coordinate_left: Vec<u64> = coordinates
        .node_intervals
        .iter()
        .map(|interval| interval.left as u64)
        .collect();
    let node_coordinate_right: Vec<u64> = coordinates
        .node_intervals
        .iter()
        .map(|interval| interval.right as u64)
        .collect();
    let node_window_left: Vec<u64> = windows
        .intervals
        .iter()
        .map(|interval| interval.left as u64)
        .collect();
    let node_window_right: Vec<u64> = windows
        .intervals
        .iter()
        .map(|interval| interval.right as u64)
        .collect();
    let node_state_offset: Vec<u64> = windows.offsets.iter().map(|&value| value as u64).collect();
    let node_state_len: Vec<u64> = windows.lengths.iter().map(|&value| value as u64).collect();
    let mut edge_state_src_offset = vec![0u64; edge_count];
    let mut edge_state_dst_offset = vec![0u64; edge_count];
    let mut edge_state_overlap_len = vec![0u64; edge_count];
    for overlap in &overlaps.overlaps {
        edge_state_src_offset[overlap.edge_id] = overlap.src_offset as u64;
        edge_state_dst_offset[overlap.edge_id] = overlap.dst_offset as u64;
        edge_state_overlap_len[overlap.edge_id] = overlap.len as u64;
    }

    write_npy_1d(
        coordinates_dir.join("node_coordinate_left.npy"),
        &node_coordinate_left,
    )?;
    write_npy_1d(
        coordinates_dir.join("node_coordinate_right.npy"),
        &node_coordinate_right,
    )?;
    write_npy_1d(
        coordinates_dir.join("node_window_left.npy"),
        &node_window_left,
    )?;
    write_npy_1d(
        coordinates_dir.join("node_window_right.npy"),
        &node_window_right,
    )?;
    write_npy_1d(
        coordinates_dir.join("node_state_offset.npy"),
        &node_state_offset,
    )?;
    write_npy_1d(coordinates_dir.join("node_state_len.npy"), &node_state_len)?;
    write_npy_1d(
        coordinates_dir.join("edge_state_src_offset.npy"),
        &edge_state_src_offset,
    )?;
    write_npy_1d(
        coordinates_dir.join("edge_state_dst_offset.npy"),
        &edge_state_dst_offset,
    )?;
    write_npy_1d(
        coordinates_dir.join("edge_state_overlap_len.npy"),
        &edge_state_overlap_len,
    )?;
    Ok(())
}

fn write_reference_artifacts_from_coordinates(
    output_dir: &Path,
    coordinates: &crate::coordinates::GlobalCoordinateOutput,
    graph: &TensorGraph,
) -> Result<()> {
    let reference_dir = output_dir.join("reference");
    let ref_node_path = reference_dir.join("ref_node_ids.npy");
    let ref_symbols_path = reference_dir.join("ref_sequence_symbols.npy");
    let ref_node_ids: Vec<i64> = coordinates
        .reference_path
        .node_ids
        .iter()
        .map(|node| node.map(|node| node as i64).unwrap_or(-1))
        .collect();
    let ref_sequence_symbols: Vec<u16> = coordinates
        .reference_path
        .node_ids
        .iter()
        .map(|node| node.map(|node| graph.node_symbol[node]).unwrap_or_default())
        .collect();
    write_npy_1d(ref_node_path, &ref_node_ids)?;
    write_npy_1d(ref_symbols_path, &ref_sequence_symbols)?;
    Ok(())
}

fn try_write_reference_msa_initialization(
    output_dir: &Path,
    graph: &TensorGraph,
    reference_path: &ReferencePath,
    alphabet_size: usize,
) -> Result<bool> {
    let source_sequence_id_path = output_dir.join("source").join("source_sequence_id.npy");
    let source_position_path = output_dir.join("source").join("source_position.npy");
    let node_source_offset_path = output_dir.join("source").join("node_source_offset.npy");
    let node_source_len_path = output_dir.join("source").join("node_source_len.npy");
    if !source_sequence_id_path.exists()
        || !source_position_path.exists()
        || !node_source_offset_path.exists()
        || !node_source_len_path.exists()
    {
        return Ok(false);
    }

    let source_sequence_id = read_npy_1d_u64(&source_sequence_id_path)?;
    let source_position = read_npy_1d_u64(&source_position_path)?;
    let node_source_offset = to_usize_vec(read_npy_1d_u64(&node_source_offset_path)?)?;
    let node_source_len = to_usize_vec(read_npy_1d_u64(&node_source_len_path)?)?;
    let paths = collect_weighted_sequence_paths(
        graph,
        &source_sequence_id,
        &source_position,
        &node_source_offset,
        &node_source_len,
    )?;
    if paths.is_empty() {
        return Ok(false);
    }

    let reference_symbols = reference_path
        .node_ids
        .iter()
        .map(|node_id| {
            node_id
                .map(|node_id| graph.node_symbol[node_id])
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    let config = ReferenceMsaConfig {
        scoring: ReferenceMsaScoring::identity(alphabet_size, 2, -1, -5, -1),
        ..ReferenceMsaConfig::default()
    };
    let result =
        build_reference_msa_against_reference(&paths, reference_path, &reference_symbols, &config)?;
    let global_state_count = reference_path.global_state_count();
    if result.reference_path.global_state_count() != global_state_count {
        return Err(PreAdPrepError::Validation(
            "reference_msa result disagrees with coordinate reference path length".into(),
        ));
    }

    let (match_probs, insert_probs) = emission_probability_tables(&result, alphabet_size, 1e-3)?;
    let insert_ranges = insertion_ranges(&result.longest_insertions);
    write_reference_insert_ranges(output_dir, &insert_ranges)?;
    let transition_logits = bootstrap_transition_logits(global_state_count, &insert_ranges);
    let match_emission = match_probs
        .into_iter()
        .map(|value| value.max(1e-16).ln())
        .collect::<Vec<_>>();
    let insert_emission = insert_probs
        .into_iter()
        .map(|value| value.max(1e-16).ln())
        .collect::<Vec<_>>();
    let metadata = reference_msa_metadata(&result, alphabet_size, &insert_ranges);
    let manifest = write_initialization_track(
        output_dir,
        InitializationTrack::ReferenceMsa,
        global_state_count,
        alphabet_size,
        &match_emission,
        &insert_emission,
        &transition_logits,
        metadata,
    )?;
    let diagnostics_path = output_dir
        .join("diagnostics")
        .join("init_reference_msa.json");
    fs::write(diagnostics_path, serde_json::to_vec_pretty(&manifest)?)?;
    Ok(true)
}

fn normalize_legacy_initialization_tracks(
    output_dir: &Path,
    target_alphabet_size: usize,
    graph_alphabet: &[String],
    symbol_encoding: &BTreeMap<String, u16>,
) -> Result<Vec<(String, String)>> {
    let mut messages = Vec::new();
    for track in [
        InitializationTrack::LegacyCurrent,
        InitializationTrack::ReferenceMsa,
    ] {
        if normalize_legacy_initialization_track(
            output_dir,
            track,
            target_alphabet_size,
            graph_alphabet,
            symbol_encoding,
        )? {
            messages.push((
                format!("{}_normalized", track.as_manifest_str()),
                format!(
                    "{} initialization was normalized from legacy A,T,C,G emissions into graph symbol encoding [{}]",
                    track.as_manifest_str(),
                    graph_alphabet.join(",")
                ),
            ));
        }
    }
    Ok(messages)
}

fn normalize_legacy_initialization_track(
    output_dir: &Path,
    track: InitializationTrack,
    target_alphabet_size: usize,
    graph_alphabet: &[String],
    symbol_encoding: &BTreeMap<String, u16>,
) -> Result<bool> {
    let manifest_path = InitialPhmmManifest::manifest_path(output_dir, track);
    if !manifest_path.exists() {
        return Ok(false);
    }
    let manifest = InitialPhmmManifest::read_from_path(&manifest_path)?;
    if manifest.alphabet_size == target_alphabet_size {
        return Ok(false);
    }
    if manifest.alphabet_size != LEGACY_PHMM_BASE_ORDER.len() {
        return Err(PreAdPrepError::Validation(format!(
            "initialization track {} has alphabet_size {}, but only {}-column legacy A,T,C,G normalization to graph alphabet size {} is supported",
            track.as_manifest_str(),
            manifest.alphabet_size,
            LEGACY_PHMM_BASE_ORDER.len(),
            target_alphabet_size
        )));
    }

    let (match_rows, match_cols, match_emission) =
        read_npy_2d_f64(output_dir.join(&manifest.match_emission.path))?;
    let (insert_rows, insert_cols, insert_emission) =
        read_npy_2d_f64(output_dir.join(&manifest.insert_emission.path))?;
    let (transition_rows, transition_cols, transition_logits) =
        read_npy_2d_f64(output_dir.join(&manifest.transition_logits.path))?;

    if match_rows != manifest.global_state_count || match_cols != manifest.alphabet_size {
        return Err(PreAdPrepError::Validation(format!(
            "initialization track {} match_emission array shape [{}, {}] does not match manifest [{}, {}]",
            track.as_manifest_str(),
            match_rows,
            match_cols,
            manifest.global_state_count,
            manifest.alphabet_size
        )));
    }
    if insert_rows != manifest.global_state_count + 1 || insert_cols != manifest.alphabet_size {
        return Err(PreAdPrepError::Validation(format!(
            "initialization track {} insert_emission array shape [{}, {}] does not match manifest [{}, {}]",
            track.as_manifest_str(),
            insert_rows,
            insert_cols,
            manifest.global_state_count + 1,
            manifest.alphabet_size
        )));
    }
    if transition_rows != manifest.global_state_count + 1
        || transition_cols != TRANSITION_ORDER.len()
    {
        return Err(PreAdPrepError::Validation(format!(
            "initialization track {} transition_logits array shape [{}, {}] does not match manifest [{}, {}]",
            track.as_manifest_str(),
            transition_rows,
            transition_cols,
            manifest.global_state_count + 1,
            TRANSITION_ORDER.len()
        )));
    }

    let normalized_match = expand_legacy_emission_table(
        &match_emission,
        match_rows,
        match_cols,
        target_alphabet_size,
        symbol_encoding,
    )?;
    let normalized_insert = expand_legacy_emission_table(
        &insert_emission,
        insert_rows,
        insert_cols,
        target_alphabet_size,
        symbol_encoding,
    )?;
    let mut metadata = manifest.metadata;
    metadata.push((
        "source_alphabet_order".into(),
        LEGACY_PHMM_BASE_ORDER.join(","),
    ));
    metadata.push(("normalized_to_graph_encoding".into(), "true".into()));
    metadata.push(("graph_alphabet".into(), graph_alphabet.join(",")));
    metadata.push((
        "unsupported_symbol_fill".into(),
        LEGACY_SYMBOL_FLOOR_DESCRIPTION.into(),
    ));
    let normalized_manifest = write_initialization_track(
        output_dir,
        track,
        manifest.global_state_count,
        target_alphabet_size,
        &normalized_match,
        &normalized_insert,
        &transition_logits,
        metadata,
    )?;
    fs::write(
        output_dir
            .join("diagnostics")
            .join(format!("init_{}.json", track.as_manifest_str())),
        serde_json::to_vec_pretty(&normalized_manifest)?,
    )?;
    Ok(true)
}

fn expand_legacy_emission_table(
    source: &[f64],
    rows: usize,
    cols: usize,
    target_alphabet_size: usize,
    symbol_encoding: &BTreeMap<String, u16>,
) -> Result<Vec<f64>> {
    if cols == target_alphabet_size {
        return Ok(source.to_vec());
    }
    if cols != LEGACY_PHMM_BASE_ORDER.len() {
        return Err(PreAdPrepError::Validation(format!(
            "cannot normalize legacy emission table with {} columns into graph alphabet size {}",
            cols, target_alphabet_size
        )));
    }
    let mut expanded = vec![LEGACY_SYMBOL_FLOOR_LOG_PROB; rows * target_alphabet_size];
    for (source_col, symbol) in LEGACY_PHMM_BASE_ORDER.iter().enumerate() {
        let target_col = usize::from(*symbol_encoding.get(*symbol).ok_or_else(|| {
            PreAdPrepError::Validation(format!(
                "graph symbol encoding is missing canonical legacy symbol {}",
                symbol
            ))
        })?);
        if target_col >= target_alphabet_size {
            return Err(PreAdPrepError::Validation(format!(
                "graph symbol {} maps to column {}, outside target alphabet size {}",
                symbol, target_col, target_alphabet_size
            )));
        }
        for row in 0..rows {
            expanded[row * target_alphabet_size + target_col] = source[row * cols + source_col];
        }
    }
    Ok(expanded)
}

fn initialization_alphabet_size(core: &GraphCoreJson, graph: &TensorGraph) -> Option<usize> {
    let from_encoding = core
        .symbol_encoding
        .values()
        .copied()
        .max()
        .map(|value| value as usize + 1);
    let from_graph = graph
        .node_symbol
        .iter()
        .copied()
        .max()
        .map(|value| value as usize + 1);
    from_encoding.or(from_graph)
}

fn to_usize_vec(values: Vec<u64>) -> Result<Vec<usize>> {
    values
        .into_iter()
        .map(|value| {
            usize::try_from(value).map_err(|_| {
                PreAdPrepError::Validation(format!(
                    "NumPy value {} cannot fit into usize on this platform",
                    value
                ))
            })
        })
        .collect()
}

fn write_reference_insert_ranges(
    output_dir: &Path,
    insert_ranges: &[(usize, usize)],
) -> Result<()> {
    if insert_ranges.is_empty() {
        return Ok(());
    }
    let reference_dir = output_dir.join("reference");
    let left = insert_ranges
        .iter()
        .map(|(start, _)| *start as i64)
        .collect::<Vec<_>>();
    let right = insert_ranges
        .iter()
        .map(|(_, end)| *end as i64)
        .collect::<Vec<_>>();
    write_npy_1d(reference_dir.join("insert_region_left.npy"), &left)?;
    write_npy_1d(reference_dir.join("insert_region_right.npy"), &right)?;
    Ok(())
}

fn bootstrap_transition_logits(
    global_state_count: usize,
    insert_ranges: &[(usize, usize)],
) -> Vec<f64> {
    let me = (0.5_f64).ln();
    let md = -2.0_f64;
    let mi = -5.0_f64;
    let ii = (0.5_f64).ln();
    let dm = (0.5_f64).ln();
    let pi_mid = [1.0_f64, 1.0_f64, 1.0_f64];

    let n_positions = global_state_count + 1;
    let mm_base = (1.0 - md.exp() - mi.exp()).ln();
    let im_base = (1.0 - ii.exp()).ln();
    let dd_base = (1.0 - dm.exp()).ln();
    let high_mi = (mi.exp() + 0.1).ln();

    let mut _mi = vec![mi; n_positions];
    let mut _md = vec![md; n_positions];
    let mut _mm = vec![mm_base; n_positions];
    for &(left, right) in insert_ranges {
        let start = left.min(n_positions);
        let stop = right.min(n_positions);
        for value in &mut _mi[start..stop] {
            *value = high_mi;
        }
    }

    let pi_sum = pi_mid.iter().sum::<f64>();
    _mi[0] = (pi_mid[1] / pi_sum).ln();
    _md[0] = (pi_mid[2] / pi_sum).ln();
    _mm[0] = (pi_mid[0] / pi_sum).ln();
    _mm[n_positions - 1] = me;
    _mi[n_positions - 1] = (1.0 - me.exp()).ln();

    let _ii = vec![ii; n_positions];
    let _im = vec![im_base; n_positions];
    let _id = vec![f64::NEG_INFINITY; n_positions];
    let mut _dm = vec![dm; n_positions];
    let mut _dd = vec![dd_base; n_positions];
    let _di = vec![f64::NEG_INFINITY; n_positions];
    _dm[0] = f64::NEG_INFINITY;
    _dd[0] = f64::NEG_INFINITY;
    _dd[n_positions - 1] = f64::NEG_INFINITY;
    _dm[n_positions - 1] = 0.0;

    let columns = [&_mm, &_md, &_mi, &_dm, &_dd, &_di, &_im, &_id, &_ii];
    let mut values = Vec::with_capacity(n_positions * TRANSITION_ORDER.len());
    for row in 0..n_positions {
        for column in &columns {
            values.push(column[row]);
        }
    }
    values
}

fn reference_msa_metadata(
    result: &ReferenceMsaResult,
    alphabet_size: usize,
    insert_ranges: &[(usize, usize)],
) -> Vec<(String, String)> {
    vec![
        ("source".into(), "reference_path_msa".into()),
        ("alphabet_size".into(), alphabet_size.to_string()),
        (
            "path_count".into(),
            result.path_alignments.len().to_string(),
        ),
        ("total_weight".into(), result.total_weight.to_string()),
        ("insert_range_count".into(), insert_ranges.len().to_string()),
        (
            "minimum_state_count".into(),
            result
                .state_bands
                .minimum
                .included_column_indices
                .len()
                .to_string(),
        ),
        (
            "canonical_state_count".into(),
            result
                .state_bands
                .canonical
                .included_column_indices
                .len()
                .to_string(),
        ),
        (
            "maximum_state_count".into(),
            result
                .state_bands
                .maximum
                .included_column_indices
                .len()
                .to_string(),
        ),
    ]
}

fn add_generated_manifest_arrays(
    output_dir: &Path,
    manifest: &mut TensorGraphManifest,
    node_count: usize,
    edge_count: usize,
    global_state_count: Option<usize>,
) -> Result<()> {
    let _ = output_dir;
    if let Some(global_state_count) = global_state_count {
        push_array_if_missing(
            manifest,
            ArraySpec::new(
                "node_coordinate_left",
                "coordinates/node_coordinate_left.npy",
                DataType::U64,
                vec![node_count],
            ),
        );
        push_array_if_missing(
            manifest,
            ArraySpec::new(
                "node_coordinate_right",
                "coordinates/node_coordinate_right.npy",
                DataType::U64,
                vec![node_count],
            ),
        );
        push_array_if_missing(
            manifest,
            ArraySpec::new(
                "node_window_left",
                "coordinates/node_window_left.npy",
                DataType::U64,
                vec![node_count],
            ),
        );
        push_array_if_missing(
            manifest,
            ArraySpec::new(
                "node_window_right",
                "coordinates/node_window_right.npy",
                DataType::U64,
                vec![node_count],
            ),
        );
        push_array_if_missing(
            manifest,
            ArraySpec::new(
                "node_state_offset",
                "coordinates/node_state_offset.npy",
                DataType::U64,
                vec![node_count],
            ),
        );
        push_array_if_missing(
            manifest,
            ArraySpec::new(
                "node_state_len",
                "coordinates/node_state_len.npy",
                DataType::U64,
                vec![node_count],
            ),
        );
        push_array_if_missing(
            manifest,
            ArraySpec::new(
                "edge_state_src_offset",
                "coordinates/edge_state_src_offset.npy",
                DataType::U64,
                vec![edge_count],
            ),
        );
        push_array_if_missing(
            manifest,
            ArraySpec::new(
                "edge_state_dst_offset",
                "coordinates/edge_state_dst_offset.npy",
                DataType::U64,
                vec![edge_count],
            ),
        );
        push_array_if_missing(
            manifest,
            ArraySpec::new(
                "edge_state_overlap_len",
                "coordinates/edge_state_overlap_len.npy",
                DataType::U64,
                vec![edge_count],
            ),
        );
        push_array_if_missing(
            manifest,
            ArraySpec::new(
                "ref_node_ids",
                "reference/ref_node_ids.npy",
                DataType::I64,
                vec![global_state_count],
            )
            .optional(),
        );
        push_array_if_missing(
            manifest,
            ArraySpec::new(
                "ref_sequence_symbols",
                "reference/ref_sequence_symbols.npy",
                DataType::U16,
                vec![global_state_count],
            )
            .optional(),
        );
    }
    Ok(())
}

fn push_array_if_missing(manifest: &mut TensorGraphManifest, spec: ArraySpec) {
    if manifest
        .arrays
        .iter()
        .all(|existing| existing.name != spec.name)
    {
        manifest.add_array(spec);
    }
}

fn build_adjacency(node_count: usize, edge_src: &[usize], edge_dst: &[usize]) -> AdjacencyCsr {
    let edge_count = edge_src.len();
    let mut counts = vec![0usize; node_count];
    for &src in edge_src {
        if src < node_count {
            counts[src] += 1;
        }
    }
    let mut indptr = vec![0usize; node_count + 1];
    for node in 0..node_count {
        indptr[node + 1] = indptr[node] + counts[node];
    }
    let mut cursor = indptr[..node_count].to_vec();
    let mut indices = vec![0usize; edge_count];
    let mut edge_ids = vec![0usize; edge_count];
    for (edge_id, (&src, &dst)) in edge_src.iter().zip(edge_dst.iter()).enumerate() {
        if src < node_count {
            let pos = cursor[src];
            indices[pos] = dst;
            edge_ids[pos] = edge_id;
            cursor[src] += 1;
        }
    }
    AdjacencyCsr {
        indptr,
        indices,
        edge_ids,
    }
}

#[derive(Debug, Deserialize)]
struct ArraySpecJson {
    name: String,
    path: PathBuf,
    dtype: String,
    shape: Vec<usize>,
    required: bool,
}

impl TryFrom<ArraySpecJson> for ArraySpec {
    type Error = PreAdPrepError;

    fn try_from(value: ArraySpecJson) -> Result<Self> {
        Ok(Self {
            name: value.name,
            path: value.path,
            dtype: parse_dtype(&value.dtype)?,
            shape: value.shape,
            required: value.required,
            description: None,
        })
    }
}

fn parse_dtype(dtype: &str) -> Result<DataType> {
    match dtype {
        "uint8" | "u8" => Ok(DataType::U8),
        "uint16" | "u16" => Ok(DataType::U16),
        "uint32" | "u32" => Ok(DataType::U32),
        "uint64" | "u64" => Ok(DataType::U64),
        "int64" | "i64" => Ok(DataType::I64),
        "float32" | "f32" => Ok(DataType::F32),
        "float64" | "f64" => Ok(DataType::F64),
        "utf8_bytes" => Ok(DataType::Utf8Bytes),
        other => Err(PreAdPrepError::Validation(format!(
            "unsupported dtype in legacy bridge output: {other}"
        ))),
    }
}

fn json_value_to_metadata_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

#[derive(Debug, Deserialize)]
struct ProfilingJson {
    wall_seconds: Option<f64>,
    cpu_seconds: Option<f64>,
    peak_rss_bytes: Option<u64>,
}

impl From<ProfilingJson> for ProfilingSummary {
    fn from(value: ProfilingJson) -> Self {
        Self {
            wall_seconds: value.wall_seconds,
            cpu_seconds: value.cpu_seconds,
            peak_rss_bytes: value.peak_rss_bytes,
        }
    }
}

#[derive(Debug, Deserialize)]
struct DiagnosticJson {
    severity: String,
    code: String,
    message: String,
}

impl From<DiagnosticJson> for Diagnostic {
    fn from(value: DiagnosticJson) -> Self {
        let severity = match value.severity.as_str() {
            "warning" => DiagnosticSeverity::Warning,
            "error" => DiagnosticSeverity::Error,
            _ => DiagnosticSeverity::Info,
        };
        Diagnostic::new(severity, value.code, value.message)
    }
}
