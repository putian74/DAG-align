//! Command-line entry point for Pre-AD-prep.

use std::env;
use std::path::PathBuf;

use pre_ad_prep::{
    ArtifactValidationLevel, ConversionDiagnostics, LegacyAdapter, LegacyConversionOptions,
    LegacyDagAlignAdapter, LegacyDagAlignInput, TensorGraphArtifact,
    validate_tensor_graph_artifact,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("--version") | Some("-V") => {
            println!("{}", env!("CARGO_PKG_VERSION"));
        }
        Some("convert-legacy") => {
            let graph_dir = PathBuf::from(args.next().ok_or("missing <graph-dir>")?);
            let output_dir = PathBuf::from(args.next().ok_or("missing <output-dir>")?);
            let mut options = LegacyConversionOptions {
                allow_python_object_bridge: true,
                ..LegacyConversionOptions::default()
            };
            for flag in args {
                match flag.as_str() {
                    "--skip-initialization" => options.include_initialization = false,
                    "--require-state-windows" => options.require_state_windows = true,
                    other => return Err(format!("unknown flag for convert-legacy: {other}").into()),
                }
            }
            let adapter = LegacyDagAlignAdapter;
            let output = adapter.convert(
                &LegacyDagAlignInput::from_graph_dir(graph_dir),
                output_dir.clone(),
                options,
            )?;
            println!(
                "converted {} nodes and {} edges into {}",
                output.graph.node_count(),
                output.graph.edge_count(),
                output_dir.display()
            );
        }
        Some("validate") => {
            let artifact_dir = PathBuf::from(args.next().ok_or("missing <artifact-dir>")?);
            let training_ready = args.any(|flag| flag == "--training-ready");
            let artifact = TensorGraphArtifact::read_manifest(&artifact_dir)?;
            let level = if training_ready {
                ArtifactValidationLevel::TrainingReady
            } else {
                ArtifactValidationLevel::GraphCore
            };
            validate_tensor_graph_artifact(&artifact, level)?.into_result()?;
            println!("validated {}", artifact_dir.display());
        }
        Some("diagnose") => {
            let artifact_dir = PathBuf::from(args.next().ok_or("missing <artifact-dir>")?);
            let diagnostics = ConversionDiagnostics::read_from_path(
                artifact_dir.join("diagnostics").join("conversion.json"),
            )?;
            println!(
                "diagnostics={} profiling={}",
                diagnostics.report.diagnostics.len(),
                diagnostics.profiling.is_some()
            );
        }
        Some(command) => {
            return Err(format!("unknown command: {command}").into());
        }
        None => {
            println!("pre-ad-prep <convert-legacy|validate|diagnose|--version> [arguments...]");
        }
    }
    Ok(())
}
