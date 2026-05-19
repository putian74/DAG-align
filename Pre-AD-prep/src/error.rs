//! Error types shared across preprocessing modules.

use std::fmt::{Display, Formatter};

/// Convenient crate-wide result type.
pub type Result<T> = std::result::Result<T, PreAdPrepError>;

/// Errors surfaced by Pre-AD-prep public APIs.
#[derive(Debug)]
pub enum PreAdPrepError {
    /// Input or generated artifact failed a contract check.
    Validation(String),
    /// The requested input/output format is intentionally not implemented yet.
    Unsupported(String),
    /// Filesystem or stream operation failed.
    Io(std::io::Error),
    /// JSON metadata parsing or writing failed.
    Json(serde_json::Error),
}

impl Display for PreAdPrepError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Validation(message) => write!(f, "validation error: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported operation: {message}"),
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::Json(error) => write!(f, "JSON error: {error}"),
        }
    }
}

impl std::error::Error for PreAdPrepError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Json(error) => Some(error),
            Self::Validation(_) | Self::Unsupported(_) => None,
        }
    }
}

impl From<std::io::Error> for PreAdPrepError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for PreAdPrepError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}
