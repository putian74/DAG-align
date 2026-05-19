//! Validation report types and shared validation helpers.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::Read;

use crate::error::{PreAdPrepError, Result};
use crate::export::{ArraySpec, TensorGraphArtifact, TensorGraphManifest};

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationSeverity {
    Warning,
    Error,
}

/// One validation finding with a stable code for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationIssue {
    pub severity: ValidationSeverity,
    pub code: String,
    pub message: String,
}

impl ValidationIssue {
    pub fn error(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: ValidationSeverity::Error,
            code: code.into(),
            message: message.into(),
        }
    }

    pub fn warning(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            severity: ValidationSeverity::Warning,
            code: code.into(),
            message: message.into(),
        }
    }
}

/// Accumulates validation issues without hiding non-fatal warnings.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, issue: ValidationIssue) {
        self.issues.push(issue);
    }

    pub fn error(&mut self, code: impl Into<String>, message: impl Into<String>) {
        self.push(ValidationIssue::error(code, message));
    }

    pub fn warning(&mut self, code: impl Into<String>, message: impl Into<String>) {
        self.push(ValidationIssue::warning(code, message));
    }

    pub fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|issue| issue.severity == ValidationSeverity::Error)
    }

    pub fn into_result(self) -> Result<()> {
        if self.has_errors() {
            let message = self
                .issues
                .iter()
                .filter(|issue| issue.severity == ValidationSeverity::Error)
                .map(|issue| format!("{}: {}", issue.code, issue.message))
                .collect::<Vec<_>>()
                .join("; ");
            Err(PreAdPrepError::Validation(message))
        } else {
            Ok(())
        }
    }
}

/// Public validation contract implemented by artifact structs.
pub trait Validate {
    fn validate(&self) -> Result<ValidationReport>;
}

/// Validation strictness used by CLI and library callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArtifactValidationLevel {
    GraphCore,
    TrainingReady,
}

/// Validate a tensor graph artifact directory against its manifest.
pub fn validate_tensor_graph_artifact(
    artifact: &TensorGraphArtifact,
    level: ArtifactValidationLevel,
) -> Result<ValidationReport> {
    let mut report = ValidationReport::new();
    validate_manifest_structure(&artifact.manifest, &mut report);
    validate_required_array_contracts(&artifact.manifest, level, &mut report);

    for array in &artifact.manifest.arrays {
        validate_array_file(artifact, array, &mut report)?;
    }
    validate_related_array_shapes(&artifact.manifest, &mut report);

    Ok(report)
}

fn validate_manifest_structure(manifest: &TensorGraphManifest, report: &mut ValidationReport) {
    if manifest.format_name != "ad_phmm_tensor_graph" {
        report.error(
            "manifest_format_name",
            format!("unexpected format_name {}", manifest.format_name),
        );
    }
    if manifest.format_version != 1 {
        report.error(
            "manifest_format_version",
            format!("unexpected format_version {}", manifest.format_version),
        );
    }

    let mut array_names = BTreeSet::new();
    let mut array_paths = BTreeSet::new();
    for array in &manifest.arrays {
        if !array_names.insert(array.name.clone()) {
            report.error(
                "manifest_duplicate_array_name",
                format!("duplicate array name {}", array.name),
            );
        }
        if !array_paths.insert(array.path.clone()) {
            report.error(
                "manifest_duplicate_array_path",
                format!("duplicate array path {}", array.path.display()),
            );
        }
        validate_array_shape_contract(manifest, array, report);
    }
}

fn validate_array_shape_contract(
    manifest: &TensorGraphManifest,
    array: &ArraySpec,
    report: &mut ValidationReport,
) {
    match array.name.as_str() {
        "node_symbol"
        | "node_weight"
        | "node_flags"
        | "node_coordinate_left"
        | "node_coordinate_right"
        | "node_window_left"
        | "node_window_right"
        | "node_state_offset"
        | "node_state_len" => {
            if array.shape.first().copied() != Some(manifest.node_count) {
                report.error(
                    "manifest_node_array_shape",
                    format!(
                        "{} first dimension must equal node_count {}",
                        array.name, manifest.node_count
                    ),
                );
            }
        }
        "edge_src"
        | "edge_dst"
        | "edge_weight"
        | "csr_indices"
        | "csr_edge_id"
        | "csc_indices"
        | "csc_edge_id"
        | "edge_state_src_offset"
        | "edge_state_dst_offset"
        | "edge_state_overlap_len" => {
            if array.shape.first().copied() != Some(manifest.edge_count) {
                report.error(
                    "manifest_edge_array_shape",
                    format!(
                        "{} first dimension must equal edge_count {}",
                        array.name, manifest.edge_count
                    ),
                );
            }
        }
        "csr_indptr" | "csc_indptr" => {
            if array.shape.first().copied() != Some(manifest.node_count + 1) {
                report.error(
                    "manifest_indptr_shape",
                    format!(
                        "{} first dimension must equal node_count + 1 ({})",
                        array.name,
                        manifest.node_count + 1
                    ),
                );
            }
        }
        "node_source_offset" | "node_source_len" => {
            if array.shape.first().copied() != Some(manifest.node_count) {
                report.error(
                    "manifest_node_source_shape",
                    format!(
                        "{} first dimension must equal node_count {}",
                        array.name, manifest.node_count
                    ),
                );
            }
        }
        "sequence_id" => {
            if array.shape.first().copied() != Some(manifest.sequence_count) {
                report.error(
                    "manifest_sequence_id_shape",
                    format!(
                        "sequence_id first dimension must equal sequence_count {}",
                        manifest.sequence_count
                    ),
                );
            }
        }
        "sequence_name_offset" => {
            if array.shape.first().copied() != Some(manifest.sequence_count + 1) {
                report.error(
                    "manifest_sequence_name_offset_shape",
                    format!(
                        "sequence_name_offset first dimension must equal sequence_count + 1 ({})",
                        manifest.sequence_count + 1
                    ),
                );
            }
        }
        _ => {}
    }
}

fn validate_required_array_contracts(
    manifest: &TensorGraphManifest,
    level: ArtifactValidationLevel,
    report: &mut ValidationReport,
) {
    for required in graph_core_required_arrays() {
        if manifest.require_array(required).is_none() {
            report.error(
                "manifest_missing_graph_array",
                format!("required graph array missing from manifest: {required}"),
            );
        }
    }
    if level == ArtifactValidationLevel::TrainingReady {
        for required in training_ready_required_arrays() {
            if manifest.require_array(required).is_none() {
                report.error(
                    "manifest_missing_training_array",
                    format!("training-ready array missing from manifest: {required}"),
                );
            }
        }
    }
}

fn validate_array_file(
    artifact: &TensorGraphArtifact,
    array: &ArraySpec,
    report: &mut ValidationReport,
) -> Result<()> {
    let path = artifact.root.join(&array.path);
    if !path.exists() {
        if array.required {
            report.error(
                "array_file_missing",
                format!("required array file is missing: {}", path.display()),
            );
        } else {
            report.warning(
                "array_file_missing_optional",
                format!("optional array file is missing: {}", path.display()),
            );
        }
        return Ok(());
    }

    let header = read_npy_header(&path)?;
    if !array.dtype.matches_npy_descr(&header.descr) {
        report.error(
            "array_dtype_mismatch",
            format!(
                "{} dtype mismatch: manifest {:?} vs file {}",
                array.name, array.dtype, header.descr
            ),
        );
    }
    if array.shape != header.shape {
        report.error(
            "array_shape_mismatch",
            format!(
                "{} shape mismatch: manifest {:?} vs file {:?}",
                array.name, array.shape, header.shape
            ),
        );
    }
    Ok(())
}

fn graph_core_required_arrays() -> &'static [&'static str] {
    &[
        "node_symbol",
        "node_weight",
        "node_flags",
        "edge_src",
        "edge_dst",
        "edge_weight",
        "csr_indptr",
        "csr_indices",
        "csr_edge_id",
        "csc_indptr",
        "csc_indices",
        "csc_edge_id",
        "topo_order",
    ]
}

fn training_ready_required_arrays() -> &'static [&'static str] {
    &[
        "node_coordinate_left",
        "node_coordinate_right",
        "node_window_left",
        "node_window_right",
        "node_state_offset",
        "node_state_len",
        "edge_state_src_offset",
        "edge_state_dst_offset",
        "edge_state_overlap_len",
    ]
}

fn validate_related_array_shapes(manifest: &TensorGraphManifest, report: &mut ValidationReport) {
    let packed = manifest.require_array("source_packed");
    let seq = manifest.require_array("source_sequence_id");
    let pos = manifest.require_array("source_position");
    if let (Some(packed), Some(seq)) = (packed, seq)
        && packed.shape.first() != seq.shape.first()
    {
        report.error(
            "manifest_source_sequence_id_shape",
            "source_sequence_id length must match source_packed length",
        );
    }
    if let (Some(packed), Some(pos)) = (packed, pos)
        && packed.shape.first() != pos.shape.first()
    {
        report.error(
            "manifest_source_position_shape",
            "source_position length must match source_packed length",
        );
    }
    for (left_name, right_name) in [
        ("node_coordinate_left", "node_coordinate_right"),
        ("node_window_left", "node_window_right"),
    ] {
        if let (Some(left), Some(right)) = (
            manifest.require_array(left_name),
            manifest.require_array(right_name),
        ) && left.shape != right.shape
        {
            report.error(
                "manifest_interval_pair_shape",
                format!("{left_name} and {right_name} must have identical shapes"),
            );
        }
    }
}

#[derive(Debug)]
struct NpyHeader {
    descr: String,
    shape: Vec<usize>,
}

fn read_npy_header(path: &std::path::Path) -> Result<NpyHeader> {
    let mut file = File::open(path)?;
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)?;
    if &magic != b"\x93NUMPY" {
        return Err(PreAdPrepError::Validation(format!(
            "file is not a NumPy .npy array: {}",
            path.display()
        )));
    }

    let mut version = [0u8; 2];
    file.read_exact(&mut version)?;
    let header_len = match version {
        [1, 0] => {
            let mut len_bytes = [0u8; 2];
            file.read_exact(&mut len_bytes)?;
            u16::from_le_bytes(len_bytes) as usize
        }
        [2, 0] | [3, 0] => {
            let mut len_bytes = [0u8; 4];
            file.read_exact(&mut len_bytes)?;
            u32::from_le_bytes(len_bytes) as usize
        }
        other => {
            return Err(PreAdPrepError::Unsupported(format!(
                "unsupported .npy version {}.{}",
                other[0], other[1]
            )));
        }
    };

    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)?;
    let header = String::from_utf8(header_bytes).map_err(|error| {
        PreAdPrepError::Validation(format!("invalid .npy header utf8: {error}"))
    })?;

    Ok(NpyHeader {
        descr: extract_header_value(&header, "'descr':")?
            .trim_matches('\'')
            .to_string(),
        shape: parse_shape(&extract_header_value(&header, "'shape':")?)?,
    })
}

fn extract_header_value(header: &str, key: &str) -> Result<String> {
    let start = header
        .find(key)
        .ok_or_else(|| PreAdPrepError::Validation(format!("NumPy header missing key {key}")))?
        + key.len();
    let remainder = &header[start..];
    let trimmed = remainder.trim_start();
    if key == "'shape':" {
        let end = trimmed.find(')').ok_or_else(|| {
            PreAdPrepError::Validation("NumPy header shape tuple is not closed".into())
        })?;
        Ok(trimmed[..=end].trim().to_string())
    } else {
        let end = trimmed.find(',').ok_or_else(|| {
            PreAdPrepError::Validation(format!("NumPy header key {key} has no terminator"))
        })?;
        Ok(trimmed[..end].trim().to_string())
    }
}

fn parse_shape(raw: &str) -> Result<Vec<usize>> {
    let inner = raw
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .filter_map(|item| {
            let item = item.trim();
            (!item.is_empty()).then_some(item)
        })
        .map(|item| {
            item.parse::<usize>().map_err(|error| {
                PreAdPrepError::Validation(format!("invalid NumPy shape item {item}: {error}"))
            })
        })
        .collect()
}
