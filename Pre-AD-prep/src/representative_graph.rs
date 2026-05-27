//! Representative global-graph construction and export.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::coordinates::{
    CoordinateMode, EdgeWindowOverlaps, GlobalCoordinateOutput, PackedStateWindows, ReferencePath,
    WindowBuildConfig, build_edge_window_overlaps, build_global_coordinates, build_packed_windows,
};
use crate::error::{PreAdPrepError, Result};
use crate::export::{
    ArraySpec, DataType, SourceFormat, TensorGraphArtifact, TensorGraphManifest, write_npy_1d,
};
use crate::graph::{AdjacencyCsr, NodeFlags, TensorGraph};
use crate::init::{
    InitialPhmmManifest, InitializationTrack, TRANSITION_ORDER, write_initialization_track,
};
use crate::reference_msa::{
    ReferenceMsaConfig, ReferenceMsaResult, WeightedSequencePath,
    build_reference_msa_against_reference, emission_probability_tables, insertion_ranges,
};
use crate::validate::{ArtifactValidationLevel, validate_tensor_graph_artifact};

/// Configuration for representative-global-graph construction and export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepresentativeGlobalGraphConfig {
    pub source_format: SourceFormat,
    pub source_graph_dir: Option<PathBuf>,
    pub coordinate_mode: CoordinateMode,
    pub window: WindowBuildConfig,
    pub reference_msa: ReferenceMsaConfig,
    pub alphabet: Vec<char>,
    pub symbol_encoding: Vec<(String, u16)>,
}

impl Default for RepresentativeGlobalGraphConfig {
    fn default() -> Self {
        Self {
            source_format: SourceFormat::Synthetic,
            source_graph_dir: None,
            coordinate_mode: CoordinateMode::Hmm,
            window: WindowBuildConfig::default(),
            reference_msa: ReferenceMsaConfig::default(),
            alphabet: Vec::new(),
            symbol_encoding: Vec::new(),
        }
    }
}

/// In-memory representative-global-graph pipeline outputs.
#[derive(Debug, Clone)]
pub struct RepresentativeGlobalGraphBuild {
    pub graph: TensorGraph,
    pub coordinates: GlobalCoordinateOutput,
    pub windows: PackedStateWindows,
    pub edge_overlaps: EdgeWindowOverlaps,
    pub reference_msa: ReferenceMsaResult,
}

/// On-disk representative-global-graph export summary.
#[derive(Debug, Clone)]
pub struct RepresentativeGlobalGraphExport {
    pub artifact: TensorGraphArtifact,
    pub initialization: InitialPhmmManifest,
}

#[derive(Default)]
struct RepresentativeGraphBuilder {
    node_symbol: Vec<u16>,
    node_weight: Vec<f32>,
    node_flags: Vec<NodeFlags>,
    edge_src: Vec<usize>,
    edge_dst: Vec<usize>,
    edge_weight: Vec<f32>,
    next_node: BTreeMap<(Option<usize>, u16), usize>,
    next_edge: BTreeMap<(usize, usize), usize>,
}

impl RepresentativeGraphBuilder {
    fn integrate_path(&mut self, path: &WeightedSequencePath) -> Result<()> {
        path.validate()?;
        let mut parent = None;
        let weight = path.weight as f32;
        for &symbol in &path.symbols {
            let node_id = *self.next_node.entry((parent, symbol)).or_insert_with(|| {
                let new_id = self.node_symbol.len();
                self.node_symbol.push(symbol);
                self.node_weight.push(0.0);
                self.node_flags.push(NodeFlags::default());
                new_id
            });
            self.node_weight[node_id] += weight;
            if parent.is_none() {
                self.node_flags[node_id].0 |= NodeFlags::START;
            }
            if let Some(parent_id) = parent {
                let edge_id = *self
                    .next_edge
                    .entry((parent_id, node_id))
                    .or_insert_with(|| {
                        let new_id = self.edge_src.len();
                        self.edge_src.push(parent_id);
                        self.edge_dst.push(node_id);
                        self.edge_weight.push(0.0);
                        new_id
                    });
                self.edge_weight[edge_id] += weight;
            }
            parent = Some(node_id);
        }
        if let Some(node_id) = parent {
            self.node_flags[node_id].0 |= NodeFlags::END;
        }
        Ok(())
    }

    fn finish(self) -> Result<TensorGraph> {
        if self.node_symbol.is_empty() {
            return Err(PreAdPrepError::Validation(
                "representative global graph requires at least one non-empty path".into(),
            ));
        }
        let node_count = self.node_symbol.len();
        let edge_count = self.edge_src.len();
        let csr = build_adjacency(node_count, &self.edge_src, &self.edge_dst);
        let csc = build_adjacency(node_count, &self.edge_dst, &self.edge_src);
        let topo_order = (0..node_count).collect();
        let mut graph = TensorGraph::new(
            self.node_symbol,
            self.node_weight,
            self.edge_src,
            self.edge_dst,
            self.edge_weight,
            topo_order,
        );
        graph.node_flags = self.node_flags;
        graph.csr = Some(csr);
        graph.csc = Some(csc);
        debug_assert_eq!(graph.edge_count(), edge_count);
        Ok(graph)
    }
}

/// Merge weighted paths into a representative global graph and derive coarse-MSA artifacts on it.
pub fn build_representative_global_graph(
    paths: &[WeightedSequencePath],
    config: &RepresentativeGlobalGraphConfig,
) -> Result<RepresentativeGlobalGraphBuild> {
    if paths.is_empty() {
        return Err(PreAdPrepError::Validation(
            "representative global graph requires at least one weighted path".into(),
        ));
    }
    config.reference_msa.validate()?;

    let mut builder = RepresentativeGraphBuilder::default();
    for path in paths {
        builder.integrate_path(path)?;
    }
    let mut graph = builder.finish()?;
    let reference_path = derive_reference_path_from_graph(&graph)?;
    let reference_symbols = reference_path
        .node_ids
        .iter()
        .map(|node_id| {
            node_id
                .map(|node_id| graph.node_symbol[node_id])
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    for &node_id in reference_path.node_ids.iter().flatten() {
        graph.node_flags[node_id].0 |= NodeFlags::REFERENCE;
    }

    let reference_msa = build_reference_msa_against_reference(
        paths,
        &reference_path,
        &reference_symbols,
        &config.reference_msa,
    )?;
    let coordinates = build_global_coordinates(
        &graph,
        reference_msa.reference_path.clone(),
        config.coordinate_mode,
    )?;
    let windows = build_packed_windows(&coordinates, config.window)?;
    let edge_overlaps = build_edge_window_overlaps(&graph, &windows)?;
    graph.state_windows = Some(windows.clone());
    graph.edge_overlaps = Some(edge_overlaps.clone());
    graph.validate_with_global_states(Some(coordinates.global_state_count))?;

    Ok(RepresentativeGlobalGraphBuild {
        graph,
        coordinates,
        windows,
        edge_overlaps,
        reference_msa,
    })
}

/// Export a representative global graph on the standard `tensor_graph.v1` contract.
pub fn write_representative_global_graph_artifact(
    output_dir: impl AsRef<Path>,
    build: &RepresentativeGlobalGraphBuild,
    config: &RepresentativeGlobalGraphConfig,
) -> Result<RepresentativeGlobalGraphExport> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir.join("graph"))?;
    fs::create_dir_all(output_dir.join("coordinates"))?;
    fs::create_dir_all(output_dir.join("reference"))?;
    fs::create_dir_all(output_dir.join("diagnostics"))?;

    write_graph_arrays(output_dir, &build.graph)?;
    write_coordinate_artifacts(
        output_dir,
        &build.coordinates,
        &build.windows,
        &build.edge_overlaps,
        build.graph.edge_count(),
    )?;
    write_reference_artifacts(output_dir, &build.coordinates, &build.graph)?;

    let insert_ranges = insertion_ranges(&build.reference_msa.longest_insertions);
    write_insert_range_artifacts(output_dir, &insert_ranges)?;
    let alphabet_size = infer_alphabet_size(&build.graph, config);
    let (match_probs, insert_probs) =
        emission_probability_tables(&build.reference_msa, alphabet_size, 1e-3)?;
    let match_emission = match_probs
        .into_iter()
        .map(|value| value.max(1e-16).ln())
        .collect::<Vec<_>>();
    let insert_emission = insert_probs
        .into_iter()
        .map(|value| value.max(1e-16).ln())
        .collect::<Vec<_>>();
    let transition_logits =
        bootstrap_transition_logits(build.coordinates.global_state_count, &insert_ranges);
    let initialization = write_initialization_track(
        output_dir,
        InitializationTrack::ReferenceMsa,
        build.coordinates.global_state_count,
        alphabet_size,
        &match_emission,
        &insert_emission,
        &transition_logits,
        reference_msa_metadata(&build.reference_msa, alphabet_size, &insert_ranges),
    )?;

    let sequence_count = usize::try_from(build.reference_msa.total_weight).map_err(|_| {
        PreAdPrepError::Validation(format!(
            "representative global graph total weight {} cannot fit into sequence_count",
            build.reference_msa.total_weight
        ))
    })?;
    let mut manifest = TensorGraphManifest::new_v1(
        config.source_format,
        build.graph.node_count(),
        build.graph.edge_count(),
    )
    .with_sequence_count(sequence_count)
    .with_global_state_count(build.coordinates.global_state_count);
    if let Some(source_graph_dir) = &config.source_graph_dir {
        manifest = manifest.with_source_graph_dir(source_graph_dir);
    }
    manifest.alphabet = config.alphabet.clone();
    manifest.symbol_encoding = config.symbol_encoding.clone();
    manifest.legacy_metadata = vec![
        ("construction".into(), "representative_global_graph".into()),
        (
            "reference_path_length".into(),
            build.coordinates.global_state_count.to_string(),
        ),
    ];
    add_manifest_arrays(&mut manifest, build, insert_ranges.len());

    let artifact = TensorGraphArtifact::new(output_dir, manifest);
    artifact.write_manifest()?;
    validate_tensor_graph_artifact(&artifact, ArtifactValidationLevel::TrainingReady)?
        .into_result()?;
    fs::write(
        output_dir
            .join("diagnostics")
            .join("representative_global_graph.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "path_count": build.reference_msa.path_alignments.len(),
            "sequence_count": build.reference_msa.total_weight,
            "global_state_count": build.coordinates.global_state_count,
        }))?,
    )?;

    Ok(RepresentativeGlobalGraphExport {
        artifact,
        initialization,
    })
}

fn derive_reference_path_from_graph(graph: &TensorGraph) -> Result<ReferencePath> {
    if graph.node_count() == 0 {
        return Err(PreAdPrepError::Validation(
            "cannot derive a reference path from an empty graph".into(),
        ));
    }
    let parents = graph.csc.as_ref().ok_or_else(|| {
        PreAdPrepError::Validation("representative graph is missing CSC adjacency".into())
    })?;
    let mut best_score = vec![f64::NEG_INFINITY; graph.node_count()];
    let mut best_parent = vec![None; graph.node_count()];
    for &node in &graph.topo_order {
        let mut score = f64::from(graph.node_weight[node]);
        let mut parent_choice = None;
        let start = parents.indptr[node];
        let end = parents.indptr[node + 1];
        let mut parent_score = f64::NEG_INFINITY;
        let mut parent_edge_weight = f64::NEG_INFINITY;
        for index in start..end {
            let parent = parents.indices[index];
            let edge_id = parents.edge_ids[index];
            let candidate_parent_score = best_score[parent];
            let candidate_edge_weight = f64::from(graph.edge_weight[edge_id]);
            if candidate_parent_score > parent_score
                || (candidate_parent_score == parent_score
                    && candidate_edge_weight > parent_edge_weight)
                || (candidate_parent_score == parent_score
                    && candidate_edge_weight == parent_edge_weight
                    && Some(parent) < parent_choice)
            {
                parent_score = candidate_parent_score;
                parent_edge_weight = candidate_edge_weight;
                parent_choice = Some(parent);
            }
        }
        if parent_choice.is_some() {
            score += parent_score.max(0.0) + parent_edge_weight.max(0.0);
        }
        best_score[node] = score;
        best_parent[node] = parent_choice;
    }

    let mut end_node = graph.topo_order[0];
    for &node in &graph.topo_order[1..] {
        if best_score[node] > best_score[end_node]
            || (best_score[node] == best_score[end_node] && node < end_node)
        {
            end_node = node;
        }
    }

    let mut node_ids = Vec::new();
    let mut cursor = Some(end_node);
    while let Some(node_id) = cursor {
        node_ids.push(Some(node_id));
        cursor = best_parent[node_id];
    }
    node_ids.reverse();
    Ok(ReferencePath { node_ids })
}

fn build_adjacency(node_count: usize, primary: &[usize], secondary: &[usize]) -> AdjacencyCsr {
    let edge_count = primary.len();
    let mut counts = vec![0usize; node_count];
    for &src in primary {
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
    for (edge_id, (&src, &dst)) in primary.iter().zip(secondary.iter()).enumerate() {
        let pos = cursor[src];
        indices[pos] = dst;
        edge_ids[pos] = edge_id;
        cursor[src] += 1;
    }
    AdjacencyCsr {
        indptr,
        indices,
        edge_ids,
    }
}

fn write_graph_arrays(output_dir: &Path, graph: &TensorGraph) -> Result<()> {
    let graph_dir = output_dir.join("graph");
    let node_flags = graph
        .node_flags
        .iter()
        .map(|flag| flag.0)
        .collect::<Vec<_>>();
    let edge_src = graph
        .edge_src
        .iter()
        .map(|&value| value as u64)
        .collect::<Vec<_>>();
    let edge_dst = graph
        .edge_dst
        .iter()
        .map(|&value| value as u64)
        .collect::<Vec<_>>();
    let topo_order = graph
        .topo_order
        .iter()
        .map(|&value| value as u64)
        .collect::<Vec<_>>();
    let csr = graph.csr.as_ref().ok_or_else(|| {
        PreAdPrepError::Validation("representative graph is missing CSR adjacency".into())
    })?;
    let csc = graph.csc.as_ref().ok_or_else(|| {
        PreAdPrepError::Validation("representative graph is missing CSC adjacency".into())
    })?;
    let csr_indptr = csr
        .indptr
        .iter()
        .map(|&value| value as u64)
        .collect::<Vec<_>>();
    let csr_indices = csr
        .indices
        .iter()
        .map(|&value| value as u64)
        .collect::<Vec<_>>();
    let csr_edge_id = csr
        .edge_ids
        .iter()
        .map(|&value| value as u64)
        .collect::<Vec<_>>();
    let csc_indptr = csc
        .indptr
        .iter()
        .map(|&value| value as u64)
        .collect::<Vec<_>>();
    let csc_indices = csc
        .indices
        .iter()
        .map(|&value| value as u64)
        .collect::<Vec<_>>();
    let csc_edge_id = csc
        .edge_ids
        .iter()
        .map(|&value| value as u64)
        .collect::<Vec<_>>();

    write_npy_1d(graph_dir.join("node_symbol.npy"), &graph.node_symbol)?;
    write_npy_1d(graph_dir.join("node_weight.npy"), &graph.node_weight)?;
    write_npy_1d(graph_dir.join("node_flags.npy"), &node_flags)?;
    write_npy_1d(graph_dir.join("edge_src.npy"), &edge_src)?;
    write_npy_1d(graph_dir.join("edge_dst.npy"), &edge_dst)?;
    write_npy_1d(graph_dir.join("edge_weight.npy"), &graph.edge_weight)?;
    write_npy_1d(graph_dir.join("topo_order.npy"), &topo_order)?;
    write_npy_1d(graph_dir.join("csr_indptr.npy"), &csr_indptr)?;
    write_npy_1d(graph_dir.join("csr_indices.npy"), &csr_indices)?;
    write_npy_1d(graph_dir.join("csr_edge_id.npy"), &csr_edge_id)?;
    write_npy_1d(graph_dir.join("csc_indptr.npy"), &csc_indptr)?;
    write_npy_1d(graph_dir.join("csc_indices.npy"), &csc_indices)?;
    write_npy_1d(graph_dir.join("csc_edge_id.npy"), &csc_edge_id)?;
    Ok(())
}

fn write_coordinate_artifacts(
    output_dir: &Path,
    coordinates: &GlobalCoordinateOutput,
    windows: &PackedStateWindows,
    overlaps: &EdgeWindowOverlaps,
    edge_count: usize,
) -> Result<()> {
    let coordinates_dir = output_dir.join("coordinates");
    let node_coordinate_left = coordinates
        .node_intervals
        .iter()
        .map(|interval| interval.left as u64)
        .collect::<Vec<_>>();
    let node_coordinate_right = coordinates
        .node_intervals
        .iter()
        .map(|interval| interval.right as u64)
        .collect::<Vec<_>>();
    let node_window_left = windows
        .intervals
        .iter()
        .map(|interval| interval.left as u64)
        .collect::<Vec<_>>();
    let node_window_right = windows
        .intervals
        .iter()
        .map(|interval| interval.right as u64)
        .collect::<Vec<_>>();
    let node_state_offset = windows
        .offsets
        .iter()
        .map(|&value| value as u64)
        .collect::<Vec<_>>();
    let node_state_len = windows
        .lengths
        .iter()
        .map(|&value| value as u64)
        .collect::<Vec<_>>();
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

fn write_reference_artifacts(
    output_dir: &Path,
    coordinates: &GlobalCoordinateOutput,
    graph: &TensorGraph,
) -> Result<()> {
    let reference_dir = output_dir.join("reference");
    let ref_node_ids = coordinates
        .reference_path
        .node_ids
        .iter()
        .map(|node| node.map(|node_id| node_id as i64).unwrap_or(-1))
        .collect::<Vec<_>>();
    let ref_sequence_symbols = coordinates
        .reference_path
        .node_ids
        .iter()
        .map(|node| {
            node.map(|node_id| graph.node_symbol[node_id])
                .unwrap_or_default()
        })
        .collect::<Vec<_>>();
    write_npy_1d(reference_dir.join("ref_node_ids.npy"), &ref_node_ids)?;
    write_npy_1d(
        reference_dir.join("ref_sequence_symbols.npy"),
        &ref_sequence_symbols,
    )?;
    Ok(())
}

fn write_insert_range_artifacts(output_dir: &Path, insert_ranges: &[(usize, usize)]) -> Result<()> {
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

fn add_manifest_arrays(
    manifest: &mut TensorGraphManifest,
    build: &RepresentativeGlobalGraphBuild,
    insert_range_count: usize,
) {
    let node_count = build.graph.node_count();
    let edge_count = build.graph.edge_count();
    let global_state_count = build.coordinates.global_state_count;
    let mut push = |spec: ArraySpec| manifest.add_array(spec);

    push(ArraySpec::new(
        "node_symbol",
        "graph/node_symbol.npy",
        DataType::U16,
        vec![node_count],
    ));
    push(ArraySpec::new(
        "node_weight",
        "graph/node_weight.npy",
        DataType::F32,
        vec![node_count],
    ));
    push(ArraySpec::new(
        "node_flags",
        "graph/node_flags.npy",
        DataType::U32,
        vec![node_count],
    ));
    push(ArraySpec::new(
        "edge_src",
        "graph/edge_src.npy",
        DataType::U64,
        vec![edge_count],
    ));
    push(ArraySpec::new(
        "edge_dst",
        "graph/edge_dst.npy",
        DataType::U64,
        vec![edge_count],
    ));
    push(ArraySpec::new(
        "edge_weight",
        "graph/edge_weight.npy",
        DataType::F32,
        vec![edge_count],
    ));
    push(ArraySpec::new(
        "topo_order",
        "graph/topo_order.npy",
        DataType::U64,
        vec![node_count],
    ));
    push(ArraySpec::new(
        "csr_indptr",
        "graph/csr_indptr.npy",
        DataType::U64,
        vec![node_count + 1],
    ));
    push(ArraySpec::new(
        "csr_indices",
        "graph/csr_indices.npy",
        DataType::U64,
        vec![edge_count],
    ));
    push(ArraySpec::new(
        "csr_edge_id",
        "graph/csr_edge_id.npy",
        DataType::U64,
        vec![edge_count],
    ));
    push(ArraySpec::new(
        "csc_indptr",
        "graph/csc_indptr.npy",
        DataType::U64,
        vec![node_count + 1],
    ));
    push(ArraySpec::new(
        "csc_indices",
        "graph/csc_indices.npy",
        DataType::U64,
        vec![edge_count],
    ));
    push(ArraySpec::new(
        "csc_edge_id",
        "graph/csc_edge_id.npy",
        DataType::U64,
        vec![edge_count],
    ));
    push(ArraySpec::new(
        "node_coordinate_left",
        "coordinates/node_coordinate_left.npy",
        DataType::U64,
        vec![node_count],
    ));
    push(ArraySpec::new(
        "node_coordinate_right",
        "coordinates/node_coordinate_right.npy",
        DataType::U64,
        vec![node_count],
    ));
    push(ArraySpec::new(
        "node_window_left",
        "coordinates/node_window_left.npy",
        DataType::U64,
        vec![node_count],
    ));
    push(ArraySpec::new(
        "node_window_right",
        "coordinates/node_window_right.npy",
        DataType::U64,
        vec![node_count],
    ));
    push(ArraySpec::new(
        "node_state_offset",
        "coordinates/node_state_offset.npy",
        DataType::U64,
        vec![node_count],
    ));
    push(ArraySpec::new(
        "node_state_len",
        "coordinates/node_state_len.npy",
        DataType::U64,
        vec![node_count],
    ));
    push(ArraySpec::new(
        "edge_state_src_offset",
        "coordinates/edge_state_src_offset.npy",
        DataType::U64,
        vec![edge_count],
    ));
    push(ArraySpec::new(
        "edge_state_dst_offset",
        "coordinates/edge_state_dst_offset.npy",
        DataType::U64,
        vec![edge_count],
    ));
    push(ArraySpec::new(
        "edge_state_overlap_len",
        "coordinates/edge_state_overlap_len.npy",
        DataType::U64,
        vec![edge_count],
    ));
    push(
        ArraySpec::new(
            "ref_node_ids",
            "reference/ref_node_ids.npy",
            DataType::I64,
            vec![global_state_count],
        )
        .optional(),
    );
    push(
        ArraySpec::new(
            "ref_sequence_symbols",
            "reference/ref_sequence_symbols.npy",
            DataType::U16,
            vec![global_state_count],
        )
        .optional(),
    );
    push(
        ArraySpec::new(
            "insert_region_left",
            "reference/insert_region_left.npy",
            DataType::I64,
            vec![insert_range_count],
        )
        .optional(),
    );
    push(
        ArraySpec::new(
            "insert_region_right",
            "reference/insert_region_right.npy",
            DataType::I64,
            vec![insert_range_count],
        )
        .optional(),
    );
}

fn infer_alphabet_size(graph: &TensorGraph, config: &RepresentativeGlobalGraphConfig) -> usize {
    let from_graph = graph
        .node_symbol
        .iter()
        .copied()
        .max()
        .map(|value| value as usize + 1)
        .unwrap_or(config.reference_msa.scoring.alphabet_size);
    let declared = config.alphabet.len().max(config.symbol_encoding.len());
    from_graph.max(declared)
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
        (
            "source".into(),
            "representative_global_graph_reference_msa".into(),
        ),
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
