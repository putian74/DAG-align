"""Training configuration schema."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from ad_phmm_align.train.profiling import ProfilingConfig


@dataclass(frozen=True)
class OptimizerConfig:
    """Optimizer settings for PHMM training."""

    name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)


@dataclass(frozen=True)
class LossWeights:
    """Weights for differentiable PHMM training objectives."""

    negative_log_likelihood: float = 1.0
    soft_viterbi: float = 0.0
    entropy: float = 0.0
    pairwise: float = 0.0
    transition_regularization: float = 0.0
    emission_regularization: float = 0.0
    active_state_penalty: float = 0.0


@dataclass(frozen=True)
class SubgraphSamplingConfig:
    """Configuration for SGD subgraph sampling."""

    strategy: str = "sequence_batch"
    sequence_batch_size: Optional[int] = None
    max_nodes: Optional[int] = None
    max_edges: Optional[int] = None
    require_state_windows: bool = True
    require_edge_overlaps: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for AD-PHMM training orchestration."""

    graph_path: Path
    initialization_path: Path
    output_dir: Path
    device: str = "cuda"
    seed: int = 1
    max_steps: Optional[int] = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    losses: LossWeights = field(default_factory=LossWeights)
    sampling: SubgraphSamplingConfig = field(default_factory=SubgraphSamplingConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    metadata: Mapping[str, object] = field(default_factory=dict)
