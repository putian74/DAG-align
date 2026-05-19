"""Training loop placeholder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ad_phmm_align.exceptions import ArtifactValidationError
from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.io.artifact_loader import TensorGraphArtifactLoader
from ad_phmm_align.io.schema import InitialPhmmParameters
from ad_phmm_align.phmm.initialization import load_initial_parameters
from ad_phmm_align.phmm.parameters import PhmmParameterSet
from ad_phmm_align.train.config import TrainingConfig
from ad_phmm_align.train.dataset import PreparedTrainingBatch, TensorGraphArtifact
from ad_phmm_align.train.profiling import ProfilingResult


@dataclass(frozen=True)
class TrainingRuntimeArtifacts:
    """Resolved graph and initialization artifacts for one training run."""

    graph_artifact: TensorGraphArtifact
    initial_parameters: InitialPhmmParameters
    parameters: PhmmParameterSet


@dataclass(frozen=True)
class TrainingStepInput:
    """Inputs needed to run one differentiable PHMM training step."""

    batch: PreparedTrainingBatch
    parameters: PhmmParameterSet
    step_index: int
    active_state_mask: Optional[Any] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingStepResult:
    """Outputs from one training step."""

    step_index: int
    batch_id: str
    loss: Any
    loss_components: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    profiling: Optional[ProfilingResult] = None
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FitResult:
    """Summary returned after training completes."""

    steps_completed: int
    final_loss: Optional[Any] = None
    metrics: Mapping[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    profiling: Optional[ProfilingResult] = None


class Trainer:
    """Coordinate AD-PHMM training."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def load_runtime_artifacts(self) -> TrainingRuntimeArtifacts:
        """Load, validate, and materialize graph plus initialization artifacts."""

        graph_artifact = TensorGraphArtifactLoader(self.config.graph_path).load_artifact()
        self.validate_training_artifact(graph_artifact.graph)
        initial_parameters = load_initial_parameters(self.config.initialization_path)
        parameters = PhmmParameterSet.from_initial_parameters(initial_parameters)
        self.validate_runtime_compatibility(graph_artifact.graph, initial_parameters)
        return TrainingRuntimeArtifacts(
            graph_artifact=graph_artifact,
            initial_parameters=initial_parameters,
            parameters=parameters,
        )

    def fit(self) -> None:
        """Run training."""

        raise NotImplementedError("Trainer.fit is not implemented yet")

    def training_step(self, step_input: TrainingStepInput) -> TrainingStepResult:
        """Run one training step and return loss, metrics, and profiling."""

        raise NotImplementedError("Trainer.training_step is not implemented yet")

    def validate_training_artifact(self, graph: TensorDag) -> None:
        """Fail fast when a graph lacks arrays required by the config."""

        graph.validate()
        if self.config.sampling.require_state_windows and graph.state_windows is None:
            raise ArtifactValidationError(
                "training config requires packed state windows, but graph has none"
            )
        if self.config.sampling.require_edge_overlaps and graph.edge_overlaps is None:
            raise ArtifactValidationError(
                "training config requires edge-window overlaps, but graph has none"
            )

    def validate_runtime_compatibility(
        self, graph: TensorDag, initial_parameters: InitialPhmmParameters
    ) -> None:
        """Check graph/init compatibility before DP or SGD starts."""

        if graph.metadata.global_state_count != initial_parameters.graph.global_state_count:
            raise ArtifactValidationError(
                "graph and initialization artifacts disagree on global_state_count"
            )
        if (
            graph.metadata.state_interval_semantics
            != initial_parameters.graph.state_interval_semantics
        ):
            raise ArtifactValidationError(
                "graph and initialization artifacts disagree on state interval semantics"
            )
        alphabet_size = initial_parameters.metadata.get("alphabet_size")
        if alphabet_size is not None and graph.metadata.alphabet is not None:
            if len(tuple(graph.metadata.alphabet)) != int(alphabet_size):
                raise ArtifactValidationError(
                    "graph alphabet length does not match initialization alphabet_size"
                )
        transition_order = tuple(
            str(name) for name in initial_parameters.metadata.get("transition_order", ())
        )
        if not transition_order:
            raise ArtifactValidationError(
                "initialization artifact must declare transition_order metadata"
            )
