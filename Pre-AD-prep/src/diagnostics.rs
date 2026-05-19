//! Diagnostics and profiling summaries emitted during preprocessing.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Severity of a diagnostic message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

/// One diagnostic event with stable code and human-readable message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub severity: DiagnosticSeverity,
    pub code: String,
    pub message: String,
}

impl Diagnostic {
    pub fn new(
        severity: DiagnosticSeverity,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            severity,
            code: code.into(),
            message: message.into(),
        }
    }
}

/// Aggregate diagnostics emitted by one conversion/preprocessing run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiagnosticReport {
    pub diagnostics: Vec<Diagnostic>,
}

impl DiagnosticReport {
    pub fn push(&mut self, diagnostic: Diagnostic) {
        self.diagnostics.push(diagnostic);
    }

    pub fn warning(&mut self, code: impl Into<String>, message: impl Into<String>) {
        self.push(Diagnostic::new(DiagnosticSeverity::Warning, code, message));
    }

    pub fn write_to_path(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_vec_pretty(self)?;
        fs::write(path, data)?;
        Ok(())
    }

    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let data = fs::read(path)?;
        Ok(serde_json::from_slice(&data)?)
    }
}

/// Lightweight CPU/memory profiling summary for future optimization baselines.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct ProfilingSummary {
    pub wall_seconds: Option<f64>,
    pub cpu_seconds: Option<f64>,
    pub peak_rss_bytes: Option<u64>,
}

/// Diagnostics attached to a graph conversion or preprocessing output.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConversionDiagnostics {
    pub report: DiagnosticReport,
    pub profiling: Option<ProfilingSummary>,
}

impl ConversionDiagnostics {
    pub fn write_to_path(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_vec_pretty(self)?;
        fs::write(path, data)?;
        Ok(())
    }

    pub fn read_from_path(path: impl AsRef<Path>) -> Result<Self> {
        let data = fs::read(path)?;
        Ok(serde_json::from_slice(&data)?)
    }
}
