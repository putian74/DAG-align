//! Manifest and export-directory contracts for `tensor_graph.v1`.

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{PreAdPrepError, Result};

/// Dtypes allowed in typed artifact manifests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataType {
    U8,
    U16,
    U32,
    U64,
    I64,
    F32,
    F64,
    Utf8Bytes,
}

impl DataType {
    pub fn as_manifest_str(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::U16 => "u16",
            Self::U32 => "u32",
            Self::U64 => "u64",
            Self::I64 => "i64",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Utf8Bytes => "utf8_bytes",
        }
    }

    pub fn matches_npy_descr(self, descr: &str) -> bool {
        match self {
            Self::U8 | Self::Utf8Bytes => matches!(descr, "|u1" | "<u1" | "u1"),
            Self::U16 => matches!(descr, "<u2" | "|u2" | "u2"),
            Self::U32 => matches!(descr, "<u4" | "|u4" | "u4"),
            Self::U64 => matches!(descr, "<u8" | "|u8" | "u8"),
            Self::I64 => matches!(descr, "<i8" | "|i8" | "i8"),
            Self::F32 => matches!(descr, "<f4" | "|f4" | "f4"),
            Self::F64 => matches!(descr, "<f8" | "|f8" | "f8"),
        }
    }
}

/// Source graph family that produced a tensor artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceFormat {
    DagAlignLegacy,
    DagRust,
    Synthetic,
}

impl SourceFormat {
    pub fn as_manifest_str(self) -> &'static str {
        match self {
            Self::DagAlignLegacy => "dag_align_legacy",
            Self::DagRust => "dag_rust",
            Self::Synthetic => "synthetic",
        }
    }
}

/// PHMM coordinate semantics shared with AD-PHMM-align.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateIntervalSemantics {
    HalfOpen,
}

impl StateIntervalSemantics {
    pub fn as_manifest_str(self) -> &'static str {
        match self {
            Self::HalfOpen => "half_open",
        }
    }
}

/// One manifest entry for a typed array file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArraySpec {
    pub name: String,
    pub path: PathBuf,
    pub dtype: DataType,
    pub shape: Vec<usize>,
    pub required: bool,
    #[serde(default)]
    pub description: Option<String>,
}

impl ArraySpec {
    pub fn new(
        name: impl Into<String>,
        path: impl Into<PathBuf>,
        dtype: DataType,
        shape: Vec<usize>,
    ) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            dtype,
            shape,
            required: true,
            description: None,
        }
    }

    pub fn optional(mut self) -> Self {
        self.required = false;
        self
    }

    pub fn described(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Manifest contract consumed by AD-PHMM-align.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorGraphManifest {
    pub format_name: String,
    pub format_version: u32,
    pub source_format: SourceFormat,
    pub source_graph_dir: Option<PathBuf>,
    pub node_count: usize,
    pub edge_count: usize,
    pub sequence_count: usize,
    pub global_state_count: Option<usize>,
    pub alphabet: Vec<char>,
    pub symbol_encoding: Vec<(String, u16)>,
    pub state_interval_semantics: StateIntervalSemantics,
    pub arrays: Vec<ArraySpec>,
    pub legacy_metadata: Vec<(String, String)>,
}

impl TensorGraphManifest {
    pub fn new_v1(source_format: SourceFormat, node_count: usize, edge_count: usize) -> Self {
        Self {
            format_name: "ad_phmm_tensor_graph".into(),
            format_version: 1,
            source_format,
            source_graph_dir: None,
            node_count,
            edge_count,
            sequence_count: 0,
            global_state_count: None,
            alphabet: Vec::new(),
            symbol_encoding: Vec::new(),
            state_interval_semantics: StateIntervalSemantics::HalfOpen,
            arrays: Vec::new(),
            legacy_metadata: Vec::new(),
        }
    }

    pub fn with_sequence_count(mut self, sequence_count: usize) -> Self {
        self.sequence_count = sequence_count;
        self
    }

    pub fn with_global_state_count(mut self, global_state_count: usize) -> Self {
        self.global_state_count = Some(global_state_count);
        self
    }

    pub fn with_source_graph_dir(mut self, source_graph_dir: impl Into<PathBuf>) -> Self {
        self.source_graph_dir = Some(source_graph_dir.into());
        self
    }

    pub fn add_array(&mut self, spec: ArraySpec) {
        self.arrays.push(spec);
    }

    pub fn require_array(&self, name: &str) -> Option<&ArraySpec> {
        self.arrays.iter().find(|spec| spec.name == name)
    }

    pub fn write_to_path(&self, path: impl AsRef<Path>) -> Result<()> {
        let raw: RawTensorGraphManifest = self.clone().into();
        let bytes = serde_json::to_vec_pretty(&raw)?;
        fs::write(path, bytes)?;
        Ok(())
    }

    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let raw: RawTensorGraphManifest = serde_json::from_str(&contents)?;
        raw.try_into()
    }
}

/// Filesystem root plus manifest for an exported tensor graph.
#[derive(Debug, Clone)]
pub struct TensorGraphArtifact {
    pub root: PathBuf,
    pub manifest: TensorGraphManifest,
}

impl TensorGraphArtifact {
    pub fn new(root: impl Into<PathBuf>, manifest: TensorGraphManifest) -> Self {
        Self {
            root: root.into(),
            manifest,
        }
    }

    pub fn manifest_path(&self) -> PathBuf {
        self.root.join("manifest.json")
    }

    pub fn write_manifest(&self) -> Result<()> {
        self.manifest.write_to_path(self.manifest_path())
    }

    pub fn read_manifest(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let manifest = TensorGraphManifest::read_from_path(root.join("manifest.json"))?;
        Ok(Self { root, manifest })
    }
}

/// Primitive dtypes supported by the minimal Rust `.npy` writer.
pub trait NpyElement {
    fn descr() -> &'static str;
    fn write_one<W: Write>(&self, writer: &mut W) -> Result<()>;
}

impl NpyElement for u8 {
    fn descr() -> &'static str {
        "|u1"
    }

    fn write_one<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&[*self])?;
        Ok(())
    }
}

impl NpyElement for u16 {
    fn descr() -> &'static str {
        "<u2"
    }

    fn write_one<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }
}

impl NpyElement for u32 {
    fn descr() -> &'static str {
        "<u4"
    }

    fn write_one<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }
}

impl NpyElement for u64 {
    fn descr() -> &'static str {
        "<u8"
    }

    fn write_one<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }
}

impl NpyElement for i64 {
    fn descr() -> &'static str {
        "<i8"
    }

    fn write_one<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }
}

impl NpyElement for f64 {
    fn descr() -> &'static str {
        "<f8"
    }

    fn write_one<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }
}

impl NpyElement for f32 {
    fn descr() -> &'static str {
        "<f4"
    }

    fn write_one<W: Write>(&self, writer: &mut W) -> Result<()> {
        writer.write_all(&self.to_le_bytes())?;
        Ok(())
    }
}

pub fn write_npy_1d<T: NpyElement>(path: impl AsRef<Path>, values: &[T]) -> Result<()> {
    write_npy(path, values, &[values.len()])
}

pub fn write_npy_2d<T: NpyElement>(
    path: impl AsRef<Path>,
    values: &[T],
    rows: usize,
    cols: usize,
) -> Result<()> {
    if rows.checked_mul(cols) != Some(values.len()) {
        return Err(PreAdPrepError::Validation(
            "2D .npy writer shape does not match value count".into(),
        ));
    }
    write_npy(path, values, &[rows, cols])
}

pub fn read_npy_1d_u64(path: impl AsRef<Path>) -> Result<Vec<u64>> {
    read_npy_1d(path, DataType::U64, |bytes| {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(bytes);
        Ok(u64::from_le_bytes(raw))
    })
}

pub fn read_npy_2d_f64(path: impl AsRef<Path>) -> Result<(usize, usize, Vec<f64>)> {
    read_npy_2d(path, DataType::F64, |bytes| {
        let mut raw = [0u8; 8];
        raw.copy_from_slice(bytes);
        Ok(f64::from_le_bytes(raw))
    })
}

fn read_npy_1d<T>(
    path: impl AsRef<Path>,
    expected_dtype: DataType,
    decode: impl Fn(&[u8]) -> Result<T>,
) -> Result<Vec<T>> {
    let path = path.as_ref();
    let mut file = fs::File::open(path)?;
    let header = read_npy_header(&mut file, path, expected_dtype)?;
    if header.shape.len() != 1 {
        return Err(PreAdPrepError::Validation(format!(
            "NumPy array {} must be one-dimensional, found shape {:?}",
            path.display(),
            header.shape
        )));
    }
    let payload = fs::read(path)?;
    let payload = &payload[header.data_offset..];
    let element_size = dtype_width(expected_dtype);
    if payload.len() % element_size != 0 {
        return Err(PreAdPrepError::Validation(format!(
            "NumPy payload length {} is not divisible by element size {} for {}",
            payload.len(),
            element_size,
            path.display()
        )));
    }
    payload
        .chunks_exact(element_size)
        .map(decode)
        .collect::<Result<Vec<_>>>()
}

fn read_npy_2d<T>(
    path: impl AsRef<Path>,
    expected_dtype: DataType,
    decode: impl Fn(&[u8]) -> Result<T>,
) -> Result<(usize, usize, Vec<T>)> {
    let path = path.as_ref();
    let mut file = fs::File::open(path)?;
    let header = read_npy_header(&mut file, path, expected_dtype)?;
    if header.shape.len() != 2 {
        return Err(PreAdPrepError::Validation(format!(
            "NumPy array {} must be two-dimensional, found shape {:?}",
            path.display(),
            header.shape
        )));
    }
    let payload = fs::read(path)?;
    let payload = &payload[header.data_offset..];
    let element_size = dtype_width(expected_dtype);
    if payload.len() % element_size != 0 {
        return Err(PreAdPrepError::Validation(format!(
            "NumPy payload length {} is not divisible by element size {} for {}",
            payload.len(),
            element_size,
            path.display()
        )));
    }
    let values = payload
        .chunks_exact(element_size)
        .map(decode)
        .collect::<Result<Vec<_>>>()?;
    if header.shape[0].checked_mul(header.shape[1]) != Some(values.len()) {
        return Err(PreAdPrepError::Validation(format!(
            "NumPy array {} shape {:?} does not match payload element count {}",
            path.display(),
            header.shape,
            values.len()
        )));
    }
    Ok((header.shape[0], header.shape[1], values))
}

fn read_npy_header(
    file: &mut fs::File,
    path: &Path,
    expected_dtype: DataType,
) -> Result<NpyHeader> {
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

    let descr = extract_header_value(&header, "'descr':")?
        .trim_matches('\'')
        .to_string();
    if !expected_dtype.matches_npy_descr(&descr) {
        return Err(PreAdPrepError::Validation(format!(
            "NumPy array {} has dtype {}, expected {}",
            path.display(),
            descr,
            expected_dtype.as_manifest_str()
        )));
    }
    let shape = parse_shape(&extract_header_value(&header, "'shape':")?)?;
    Ok(NpyHeader {
        data_offset: 6 + 2 + if version == [1, 0] { 2 } else { 4 } + header_len,
        shape,
    })
}

struct NpyHeader {
    data_offset: usize,
    shape: Vec<usize>,
}

fn dtype_width(dtype: DataType) -> usize {
    match dtype {
        DataType::U8 | DataType::Utf8Bytes => 1,
        DataType::U16 => 2,
        DataType::U32 | DataType::F32 => 4,
        DataType::U64 | DataType::I64 | DataType::F64 => 8,
    }
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

fn write_npy<T: NpyElement>(path: impl AsRef<Path>, values: &[T], shape: &[usize]) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[1, 0])?;

    let shape_repr = match shape {
        [n] => format!("({},)", n),
        [rows, cols] => format!("({}, {})", rows, cols),
        _ => {
            return Err(PreAdPrepError::Unsupported(
                "minimal .npy writer currently supports only 1D and 2D arrays".into(),
            ));
        }
    };
    let mut header = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        T::descr(),
        shape_repr
    );
    while (10 + header.len() + 1) % 16 != 0 {
        header.push(' ');
    }
    header.push('\n');

    let header_len = u16::try_from(header.len())
        .map_err(|_| PreAdPrepError::Validation("NumPy header too long for v1 format".into()))?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    for value in values {
        value.write_one(&mut file)?;
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawTensorGraphManifest {
    format_name: String,
    format_version: u32,
    source_format: SourceFormat,
    #[serde(default)]
    source_graph_dir: Option<PathBuf>,
    node_count: usize,
    edge_count: usize,
    sequence_count: usize,
    #[serde(default)]
    global_state_count: Option<usize>,
    #[serde(default)]
    alphabet: Vec<String>,
    #[serde(default)]
    symbol_encoding: BTreeMap<String, u16>,
    state_interval_semantics: StateIntervalSemantics,
    #[serde(default)]
    arrays: Vec<RawArraySpec>,
    #[serde(default, rename = "legacy")]
    legacy_metadata: BTreeMap<String, serde_json::Value>,
}

impl From<TensorGraphManifest> for RawTensorGraphManifest {
    fn from(value: TensorGraphManifest) -> Self {
        let mut symbol_encoding = BTreeMap::new();
        for (symbol, encoded) in value.symbol_encoding {
            symbol_encoding.insert(symbol, encoded);
        }
        let mut legacy_metadata = BTreeMap::new();
        for (key, metadata) in value.legacy_metadata {
            legacy_metadata.insert(key, serde_json::Value::String(metadata));
        }
        Self {
            format_name: value.format_name,
            format_version: value.format_version,
            source_format: value.source_format,
            source_graph_dir: value.source_graph_dir,
            node_count: value.node_count,
            edge_count: value.edge_count,
            sequence_count: value.sequence_count,
            global_state_count: value.global_state_count,
            alphabet: value
                .alphabet
                .into_iter()
                .map(|symbol| symbol.to_string())
                .collect(),
            symbol_encoding,
            state_interval_semantics: value.state_interval_semantics,
            arrays: value.arrays.into_iter().map(Into::into).collect(),
            legacy_metadata,
        }
    }
}

impl TryFrom<RawTensorGraphManifest> for TensorGraphManifest {
    type Error = PreAdPrepError;

    fn try_from(value: RawTensorGraphManifest) -> Result<Self> {
        let mut manifest = Self::new_v1(value.source_format, value.node_count, value.edge_count)
            .with_sequence_count(value.sequence_count);
        manifest.format_name = value.format_name;
        manifest.format_version = value.format_version;
        manifest.source_graph_dir = value.source_graph_dir;
        manifest.global_state_count = value.global_state_count;
        manifest.alphabet = value
            .alphabet
            .iter()
            .map(|symbol| {
                let mut chars = symbol.chars();
                let Some(ch) = chars.next() else {
                    return Err(PreAdPrepError::Validation(
                        "alphabet entry cannot be empty".into(),
                    ));
                };
                if chars.next().is_some() {
                    return Err(PreAdPrepError::Validation(format!(
                        "alphabet entry must be one symbol wide: {symbol}"
                    )));
                }
                Ok(ch)
            })
            .collect::<Result<Vec<_>>>()?;
        manifest.symbol_encoding = value.symbol_encoding.into_iter().collect();
        manifest.state_interval_semantics = value.state_interval_semantics;
        manifest.arrays = value
            .arrays
            .into_iter()
            .map(TryInto::try_into)
            .collect::<Result<_>>()?;
        manifest.legacy_metadata = value
            .legacy_metadata
            .into_iter()
            .map(|(key, metadata)| (key, json_value_to_string(&metadata)))
            .collect();
        Ok(manifest)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawArraySpec {
    name: String,
    path: PathBuf,
    dtype: String,
    shape: Vec<usize>,
    required: bool,
    #[serde(default)]
    description: Option<String>,
}

impl From<ArraySpec> for RawArraySpec {
    fn from(value: ArraySpec) -> Self {
        Self {
            name: value.name,
            path: value.path,
            dtype: value.dtype.as_manifest_str().into(),
            shape: value.shape,
            required: value.required,
            description: value.description,
        }
    }
}

impl TryFrom<RawArraySpec> for ArraySpec {
    type Error = PreAdPrepError;

    fn try_from(value: RawArraySpec) -> Result<Self> {
        Ok(Self {
            name: value.name,
            path: value.path,
            dtype: parse_dtype(&value.dtype)?,
            shape: value.shape,
            required: value.required,
            description: value.description,
        })
    }
}

fn json_value_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        other => other.to_string(),
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
            "unsupported manifest dtype: {other}"
        ))),
    }
}
