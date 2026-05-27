"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import List, Optional

from ad_phmm_align import __version__
from ad_phmm_align.io import InitializationTrack, TensorGraphArtifactLoader
from ad_phmm_align.phmm import PhmmParameterSet, load_initial_parameters
from ad_phmm_align.train import Trainer, TrainingConfig


def _serialize(value):
    if is_dataclass(value):
        return _serialize(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # pragma: no cover - defensive scalar fallback
            return value
    return value


def _emit_json(payload, output: Optional[Path]) -> int:
    text = json.dumps(_serialize(payload), indent=2, sort_keys=True)
    if output is None:
        print(text)
        return 0
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    return 0


def _parse_track(value: Optional[str]) -> Optional[InitializationTrack]:
    if value is None:
        return None
    return InitializationTrack(value)


def _build_runtime_trainer(
    graph_path: Path, initialization_path: Path, output_dir: Path
) -> tuple[Trainer, object, object]:
    trainer = Trainer(
        TrainingConfig(
            graph_path=graph_path,
            initialization_path=initialization_path,
            output_dir=output_dir,
        )
    )
    graph = TensorGraphArtifactLoader(graph_path).load_graph()
    trainer.validate_training_artifact(graph)
    initial_parameters = load_initial_parameters(initialization_path)
    trainer.validate_runtime_compatibility(graph, initial_parameters)
    return trainer, graph, initial_parameters


def _run_validate_artifacts(args: argparse.Namespace) -> int:
    graph_path = Path(args.graph)
    artifact = TensorGraphArtifactLoader(graph_path).load_artifact()
    payload = {
        "graph": {
            "graph_id": artifact.graph.metadata.graph_id,
            "node_count": artifact.graph.node_count,
            "edge_count": artifact.graph.edge_count,
            "sequence_count": artifact.manifest.sequence_count,
            "global_state_count": artifact.graph.metadata.global_state_count,
            "source_format": artifact.manifest.source_format.value,
        }
    }
    if args.initialization is not None:
        init_path = Path(args.initialization)
        trainer = Trainer(
            TrainingConfig(
                graph_path=graph_path,
                initialization_path=init_path,
                output_dir=Path(args.output).parent
                if args.output is not None
                else graph_path / "cli-output",
            )
        )
        initial_parameters = load_initial_parameters(
            init_path, track=_parse_track(args.track)
        )
        trainer.validate_runtime_compatibility(artifact.graph, initial_parameters)
        payload["initialization"] = {
            "track": initial_parameters.track.value,
            "global_state_count": initial_parameters.graph.global_state_count,
            "alphabet_size": initial_parameters.metadata.get("alphabet_size"),
        }
    return _emit_json(payload, None if args.output is None else Path(args.output))


def _run_train(args: argparse.Namespace) -> int:
    trainer = Trainer(
        TrainingConfig(
            graph_path=Path(args.graph),
            initialization_path=Path(args.initialization),
            output_dir=Path(args.output_dir),
            device=args.device,
            max_steps=args.max_steps,
        )
    )
    result = trainer.fit()
    payload = {
        "steps_completed": result.steps_completed,
        "final_loss": result.final_loss,
        "metrics": dict(result.metrics),
        "metadata": dict(result.metadata),
        "checkpoint_path": result.checkpoint_path,
    }
    return _emit_json(payload, None if args.output is None else Path(args.output))


def _run_decode(args: argparse.Namespace) -> int:
    graph_path = Path(args.graph)
    init_path = Path(args.initialization)
    output_dir = Path(args.output).parent if args.output is not None else graph_path / "cli-output"
    trainer = Trainer(
        TrainingConfig(
            graph_path=graph_path,
            initialization_path=init_path,
            output_dir=output_dir,
        )
    )
    graph = TensorGraphArtifactLoader(graph_path).load_graph()
    trainer.validate_training_artifact(graph)
    initial_parameters = load_initial_parameters(init_path, track=_parse_track(args.track))
    trainer.validate_runtime_compatibility(graph, initial_parameters)
    parameters = PhmmParameterSet.from_initial_parameters(initial_parameters)
    hard = trainer.run_hard_inference(graph, parameters)
    payload = {
        "decode_status": hard.decode_status,
        "decode_error": hard.decode_error,
        "score": hard.viterbi_result.score,
        "state_path": list(hard.viterbi_result.states),
        "node_ids": None
        if hard.viterbi_result.node_ids is None
        else hard.viterbi_result.node_ids.tolist(),
        "global_state_ids": None
        if hard.viterbi_result.global_state_ids is None
        else hard.viterbi_result.global_state_ids.tolist(),
        "node_assignment_count": 0
        if hard.viterbi_result.node_assignments is None
        else len(hard.viterbi_result.node_assignments),
        "metrics": dict(hard.metrics),
    }
    if hard.decoded_alignment is not None:
        payload["alignment"] = {
            "column_count": len(hard.decoded_alignment.column_keys),
            "sequence_names": list(hard.decoded_alignment.sequence_names),
        }
    return _emit_json(payload, None if args.output is None else Path(args.output))


def build_parser() -> argparse.ArgumentParser:
    """Build the AD-PHMM-align CLI parser."""

    parser = argparse.ArgumentParser(prog="ad-phmm-align")
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command")

    subcommands.add_parser(
        "validate-artifacts",
        help="Validate graph and initialization artifacts.",
    )
    validate = subcommands.choices["validate-artifacts"]
    validate.add_argument("--graph", required=True, help="Path to tensor_graph.v1 root.")
    validate.add_argument(
        "--initialization",
        help="Path to initialization root/track/manifest for compatibility checks.",
    )
    validate.add_argument(
        "--track",
        choices=[track.value for track in InitializationTrack],
        help="Initialization track to load when --initialization points at the graph root.",
    )
    validate.add_argument("--output", help="Optional JSON output path.")

    train = subcommands.add_parser("train", help="Run the current baseline trainer.")
    train.add_argument("--graph", required=True, help="Path to tensor_graph.v1 root.")
    train.add_argument("--initialization", required=True, help="Initialization root or manifest.")
    train.add_argument("--output-dir", required=True, help="Trainer output directory.")
    train.add_argument("--max-steps", type=int, default=0, help="Number of training steps.")
    train.add_argument("--device", default="cpu", help="Requested device label.")
    train.add_argument("--output", help="Optional JSON output path.")

    decode = subcommands.add_parser("decode", help="Decode alignments with current PHMM parameters.")
    decode.add_argument("--graph", required=True, help="Path to tensor_graph.v1 root.")
    decode.add_argument("--initialization", required=True, help="Initialization root or manifest.")
    decode.add_argument(
        "--track",
        choices=[track.value for track in InitializationTrack],
        help="Initialization track to load when --initialization points at the graph root.",
    )
    decode.add_argument("--output", help="Optional JSON output path.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "validate-artifacts":
        return _run_validate_artifacts(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "decode":
        return _run_decode(args)
    raise NotImplementedError(f"CLI command is not implemented yet: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
