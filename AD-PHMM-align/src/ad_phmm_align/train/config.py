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
    transition_anchor: float = 0.0
    emission_anchor: float = 0.0
    emission_smooth: float = 0.0
    logit_l2: float = 0.0
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
class MultiStartConfig:
    """Configuration for replica-based local-minimum escape."""

    enabled: bool = False
    replicas: int = 1
    transition_logit_std: float = 0.0
    emission_logit_std: float = 0.0
    temperature_std: float = 0.0


@dataclass(frozen=True)
class WarmRestartConfig:
    """Configuration for restart-based optimizer schedules."""

    enabled: bool = False
    strategy: str = "cosine"
    first_cycle_steps: Optional[int] = None
    cycle_multiplier: float = 1.0
    min_learning_rate: float = 0.0
    restart_max_lr_scale: float = 1.0


@dataclass(frozen=True)
class AnnealingConfig:
    """Configuration for objective and temperature continuation."""

    enabled: bool = False
    initial_temperature: float = 1.0
    final_temperature: float = 1.0
    temperature_decay_steps: Optional[int] = None
    entropy_warmup_steps: int = 0
    pairwise_warmup_steps: int = 0


@dataclass(frozen=True)
class AnchorReleaseConfig:
    """Configuration for scheduled release of initialization anchoring."""

    enabled: bool = False
    initial_scale: float = 1.0
    final_scale: float = 1.0
    decay_steps: Optional[int] = None
    relax_on_plateau: bool = True
    plateau_scale: float = 0.5


@dataclass(frozen=True)
class BranchPerturbConfig:
    """Configuration for plateau-triggered branching and perturbation."""

    enabled: bool = False
    monitor_window: int = 0
    max_branches: int = 1
    perturbation_scale: float = 0.0
    learning_rate_boost: float = 1.0


@dataclass(frozen=True)
class SupportExpansionConfig:
    """Configuration for widening sampled state support during recovery."""

    enabled: bool = False
    expansion_factor: float = 1.0
    recovery_steps: int = 0


@dataclass(frozen=True)
class EscapeConfig:
    """Independent switches for optimization escape mechanisms."""

    multi_start: MultiStartConfig = field(default_factory=MultiStartConfig)
    warm_restarts: WarmRestartConfig = field(default_factory=WarmRestartConfig)
    annealing: AnnealingConfig = field(default_factory=AnnealingConfig)
    anchor_release: AnchorReleaseConfig = field(default_factory=AnchorReleaseConfig)
    branch_perturb: BranchPerturbConfig = field(default_factory=BranchPerturbConfig)
    support_expansion: SupportExpansionConfig = field(default_factory=SupportExpansionConfig)


@dataclass(frozen=True)
class RegularizationConfig:
    """Independent switches for regularization ablation studies."""

    transition_anchor: bool = True
    emission_anchor: bool = True
    emission_smooth: bool = True
    logit_l2: bool = True
    active_state_penalty: bool = True


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
    escape: EscapeConfig = field(default_factory=EscapeConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    metadata: Mapping[str, object] = field(default_factory=dict)
