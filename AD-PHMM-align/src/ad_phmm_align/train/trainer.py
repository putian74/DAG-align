"""Training loop placeholder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from ad_phmm_align.exceptions import ArtifactValidationError
from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.phmm.parameters import PhmmParameterSet
from ad_phmm_align.train.config import TrainingConfig
from ad_phmm_align.train.dataset import PreparedTrainingBatch
from ad_phmm_align.train.profiling import ProfilingResult


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
