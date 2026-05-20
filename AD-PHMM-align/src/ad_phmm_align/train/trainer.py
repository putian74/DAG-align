"""Training loop orchestration scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ad_phmm_align.exceptions import ArtifactValidationError
from ad_phmm_align.graph.subgraph import SubgraphBatch
from ad_phmm_align.graph.tensor_dag import TensorDag
from ad_phmm_align.io.artifact_loader import TensorGraphArtifactLoader
from ad_phmm_align.io.schema import InitialPhmmParameters
from ad_phmm_align.eval import decode_alignment, summarize_alignment_metrics
from ad_phmm_align.losses import (
    active_state_regularization,
    emission_anchor_regularization,
    emission_smoothness_regularization,
    logit_l2_regularization,
    negative_log_likelihood,
    soft_alignment_entropy,
    soft_pairwise_score,
    transition_anchor_regularization,
)
from ad_phmm_align.phmm.hard_path import viterbi_decode
from ad_phmm_align.phmm.initialization import load_initial_parameters
from ad_phmm_align.phmm.parameters import PhmmParameterSet, TransitionLogitView
from ad_phmm_align.phmm.soft_path import (
    backward_log_likelihood,
    forward_log_likelihood,
    posterior_occupancy,
    soft_viterbi_score,
)
from ad_phmm_align.sampling import SubgraphSampler
from ad_phmm_align.train.config import LossWeights, TrainingConfig
from ad_phmm_align.train.dataset import PreparedTrainingBatch, TensorGraphArtifact
from ad_phmm_align.train.inference import (
    HardInferenceSummary,
    InferenceSummary,
    SoftInferenceSummary,
)
from ad_phmm_align.train.profiling import ProfilingResult, ProfilingTimer


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
    loss: Optional[Any]
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
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class _ScheduledLossWeights:
    negative_log_likelihood: float
    soft_viterbi: float
    entropy: float
    pairwise: float
    transition_anchor: float
    emission_anchor: float
    emission_smooth: float
    logit_l2: float
    active_state_penalty: float

    def as_mapping(self) -> Mapping[str, float]:
        return {
            "negative_log_likelihood": self.negative_log_likelihood,
            "soft_viterbi": self.soft_viterbi,
            "entropy": self.entropy,
            "pairwise": self.pairwise,
            "transition_anchor": self.transition_anchor,
            "emission_anchor": self.emission_anchor,
            "emission_smooth": self.emission_smooth,
            "logit_l2": self.logit_l2,
            "active_state_penalty": self.active_state_penalty,
        }


@dataclass(frozen=True)
class _TrainerReplica:
    replica_id: str
    parameters: PhmmParameterSet
    temperature_multiplier: float = 1.0
    metadata: Mapping[str, object] = field(default_factory=dict)


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

    def fit(self) -> FitResult:
        """Run baseline trainer orchestration over configured replicas/steps."""

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.profiling.enabled:
            with ProfilingTimer(self.config.profiling) as timer:
                result = self._fit_impl()
            return FitResult(
                steps_completed=result.steps_completed,
                final_loss=result.final_loss,
                metrics=result.metrics,
                checkpoint_path=result.checkpoint_path,
                profiling=timer.result,
                metadata=result.metadata,
            )
        return self._fit_impl()

    def training_step(self, step_input: TrainingStepInput) -> TrainingStepResult:
        """Run one CPU reference training step and report scalar losses/metrics."""

        graph = step_input.metadata.get("_graph")
        if not isinstance(graph, TensorDag):
            raise ArtifactValidationError(
                "training step metadata must include a TensorDag under '_graph'"
            )

        def _execute() -> TrainingStepResult:
            scheduled = step_input.metadata.get("scheduled_loss_weights")
            if not isinstance(scheduled, _ScheduledLossWeights):
                raise ArtifactValidationError(
                    "training step metadata must include scheduled loss weights"
                )
            reference_parameters = step_input.metadata.get("_reference_parameters")
            if not isinstance(reference_parameters, PhmmParameterSet):
                raise ArtifactValidationError(
                    "training step metadata must include reference PHMM parameters"
                )

            temperature = float(step_input.metadata.get("scheduled_temperature", 1.0))
            inference = self.run_inference_paths(
                graph,
                step_input.parameters,
                effective_support=step_input.active_state_mask,
                temperature=temperature,
            )
            soft_summary = inference.soft
            hard_summary = inference.hard
            forward_result = soft_summary.forward_result
            backward_result = soft_summary.backward_result
            posterior = soft_summary.posterior
            soft_viterbi_result = soft_summary.soft_viterbi_result
            viterbi_result = hard_summary.viterbi_result

            n_emit = max(1, int(step_input.batch.node_symbol.shape[0]))
            nll = float(soft_summary.negative_log_likelihood)
            entropy = float(soft_summary.entropy)
            pairwise = float(soft_summary.pairwise_score)
            soft_viterbi_component = float(-soft_viterbi_result.score / n_emit)
            transition_anchor = float(
                transition_anchor_regularization(step_input.parameters, reference_parameters)
            )
            emission_anchor = float(
                emission_anchor_regularization(step_input.parameters, reference_parameters)
            )
            emission_smooth = float(
                emission_smoothness_regularization(step_input.parameters)
            )
            logit_l2 = float(logit_l2_regularization(step_input.parameters))
            active_state_penalty = float(
                active_state_regularization(
                    posterior.posterior_support.active_global_state_ids(),
                    int(graph.metadata.global_state_count or 0),
                )
            )

            loss_components = {
                "negative_log_likelihood": nll,
                "soft_viterbi": soft_viterbi_component,
                "entropy": entropy,
                "pairwise": pairwise,
                "transition_anchor": transition_anchor,
                "emission_anchor": emission_anchor,
                "emission_smooth": emission_smooth,
                "logit_l2": logit_l2,
                "active_state_penalty": active_state_penalty,
            }
            total_loss = (
                scheduled.negative_log_likelihood * nll
                + scheduled.soft_viterbi * soft_viterbi_component
                + scheduled.entropy * entropy
                - scheduled.pairwise * pairwise
                + scheduled.transition_anchor * transition_anchor
                + scheduled.emission_anchor * emission_anchor
                + scheduled.emission_smooth * emission_smooth
                + scheduled.logit_l2 * logit_l2
                + scheduled.active_state_penalty * active_state_penalty
            )

            posterior_states = posterior.posterior_support.active_global_state_ids()
            metrics = {
                "optimization_ready": False,
                "cpu_reference_ready": True,
                "node_count": graph.node_count,
                "edge_count": graph.edge_count,
                "global_state_count": graph.metadata.global_state_count,
                "forward_active_packed_states": forward_result.input.forward_support.active_count,
                "backward_active_packed_states": backward_result.input.backward_support.active_count,
                "posterior_active_packed_states": posterior.posterior_support.active_count,
                "posterior_active_global_states": int(posterior_states.shape[0]),
                "viterbi_active_packed_states": int(
                    viterbi_result.effective_support.active_count
                    if viterbi_result.effective_support is not None
                    else 0
                ),
                "viterbi_node_assignments": 0
                if viterbi_result.node_assignments is None
                else len(viterbi_result.node_assignments),
                "scheduled_temperature": temperature,
                "log_likelihood": float(forward_result.log_likelihood),
                "backward_log_likelihood": float(backward_result.log_likelihood),
                "likelihood_gap": abs(
                    float(forward_result.log_likelihood)
                    - float(backward_result.log_likelihood)
                ),
                "entropy": entropy,
                "soft_entropy": entropy,
                "pairwise_score": pairwise,
                "soft_pairwise_score": pairwise,
                "viterbi_score": float(viterbi_result.score),
                "soft_viterbi_score": float(soft_viterbi_result.score),
                "hard_decode_status": hard_summary.decode_status,
            }
            if hard_summary.alignment_metrics is not None:
                metrics = {
                    **metrics,
                    "hard_entropy": hard_summary.alignment_metrics.entropy,
                    "hard_pairwise_score": hard_summary.alignment_metrics.pairwise_score,
                    "hard_core_entropy": hard_summary.alignment_metrics.core_entropy,
                    "hard_alignment_length": hard_summary.alignment_metrics.alignment_length,
                    "hard_core_column_count": hard_summary.alignment_metrics.core_column_count,
                    "hard_sequence_count": hard_summary.alignment_metrics.sequence_count,
                }
            metadata = {
                "phase": step_input.metadata.get("phase"),
                "replica_id": step_input.metadata.get("replica_id"),
                "active_escape_mechanisms": tuple(
                    step_input.metadata.get("active_escape_mechanisms", ())
                ),
                "regularization_switches": dict(
                    step_input.metadata.get("regularization_switches", {})
                ),
                "active_regularizers": tuple(
                    step_input.metadata.get("active_regularizers", ())
                ),
                "scheduled_loss_weights": scheduled.as_mapping(),
                "implementation": "cpu_dense_reference",
                "viterbi_state_path": tuple(viterbi_result.states),
                "viterbi_node_ids": tuple(
                    np.asarray(viterbi_result.node_ids, dtype=np.int64).tolist()
                ),
                "viterbi_node_assignment_count": 0
                if viterbi_result.node_assignments is None
                else len(viterbi_result.node_assignments),
                "hard_decode_status": hard_summary.decode_status,
                "hard_decode_error": hard_summary.decode_error,
            }
            return TrainingStepResult(
                step_index=step_input.step_index,
                batch_id=step_input.batch.batch_id,
                loss=float(total_loss),
                loss_components=loss_components,
                metrics=metrics,
                metadata=metadata,
            )

        if self.config.profiling.enabled:
            with ProfilingTimer(self.config.profiling) as timer:
                result = _execute()
            return TrainingStepResult(
                step_index=result.step_index,
                batch_id=result.batch_id,
                loss=result.loss,
                loss_components=result.loss_components,
                metrics=result.metrics,
                profiling=timer.result,
                metadata=result.metadata,
            )
        return _execute()

    def run_soft_inference(
        self,
        graph: TensorDag,
        parameters: PhmmParameterSet,
        *,
        effective_support: Optional[Any] = None,
        temperature: float = 1.0,
    ) -> SoftInferenceSummary:
        """Run the complete differentiable/soft inference path."""

        forward_result = forward_log_likelihood(
            graph,
            parameters,
            effective_support=effective_support,
        )
        backward_result = backward_log_likelihood(
            graph,
            parameters,
            effective_support=effective_support,
        )
        posterior = posterior_occupancy(
            graph,
            parameters,
            forward_result=forward_result,
            backward_result=backward_result,
        )
        soft_viterbi_result = soft_viterbi_score(
            graph,
            parameters,
            temperature=temperature,
            effective_support=effective_support,
        )
        n_emit = max(1, int(graph.node_count))
        pairwise_score_matrix = self.default_pairwise_score_matrix(
            int(np.asarray(parameters.match_emission).shape[1])
        )
        negative_ll = float(
            negative_log_likelihood(forward_result.log_likelihood, normalizer=n_emit)
        )
        entropy = float(soft_alignment_entropy(posterior))
        pairwise = float(soft_pairwise_score(posterior, pairwise_score_matrix))
        return SoftInferenceSummary(
            forward_result=forward_result,
            backward_result=backward_result,
            posterior=posterior,
            soft_viterbi_result=soft_viterbi_result,
            negative_log_likelihood=negative_ll,
            entropy=entropy,
            pairwise_score=pairwise,
            metrics={
                "log_likelihood": float(forward_result.log_likelihood),
                "backward_log_likelihood": float(backward_result.log_likelihood),
                "likelihood_gap": abs(
                    float(forward_result.log_likelihood)
                    - float(backward_result.log_likelihood)
                ),
                "soft_entropy": entropy,
                "soft_pairwise_score": pairwise,
                "soft_viterbi_score": float(soft_viterbi_result.score),
            },
        )

    def run_hard_inference(
        self,
        graph: TensorDag,
        parameters: PhmmParameterSet,
        *,
        effective_support: Optional[Any] = None,
        log_likelihood: Optional[float] = None,
    ) -> HardInferenceSummary:
        """Run the hard/max-product inference path and materialize hard metrics."""

        viterbi_result = viterbi_decode(
            graph,
            parameters,
            effective_support=effective_support,
        )
        try:
            decoded_alignment = decode_alignment(graph, viterbi_result)
            alignment_metrics = summarize_alignment_metrics(
                decoded_alignment,
                log_likelihood=log_likelihood,
            )
            return HardInferenceSummary(
                viterbi_result=viterbi_result,
                decoded_alignment=decoded_alignment,
                alignment_metrics=alignment_metrics,
                decode_status="decoded_alignment",
                metrics={
                    "hard_entropy": alignment_metrics.entropy,
                    "hard_pairwise_score": alignment_metrics.pairwise_score,
                    "hard_core_entropy": alignment_metrics.core_entropy,
                    "hard_alignment_length": alignment_metrics.alignment_length,
                    "hard_core_column_count": alignment_metrics.core_column_count,
                    "hard_sequence_count": alignment_metrics.sequence_count,
                },
            )
        except ImportError as exc:
            return HardInferenceSummary(
                viterbi_result=viterbi_result,
                decode_status="viterbi_only",
                decode_error=str(exc),
                metrics={},
            )

    def run_inference_paths(
        self,
        graph: TensorDag,
        parameters: PhmmParameterSet,
        *,
        effective_support: Optional[Any] = None,
        temperature: float = 1.0,
    ) -> InferenceSummary:
        """Run both soft and hard inference paths for one graph/batch."""

        soft = self.run_soft_inference(
            graph,
            parameters,
            effective_support=effective_support,
            temperature=temperature,
        )
        hard = self.run_hard_inference(
            graph,
            parameters,
            effective_support=effective_support,
            log_likelihood=float(soft.forward_result.log_likelihood),
        )
        return InferenceSummary(soft=soft, hard=hard)

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

    def prepare_full_graph_batch(self, graph: TensorDag) -> PreparedTrainingBatch:
        """Materialize the baseline full-graph batch used before subgraph sampling."""

        node_ids = np.arange(graph.node_count, dtype=np.int64)
        edge_ids = np.arange(graph.edge_count, dtype=np.int64)
        global_state_ids = None
        if graph.metadata.global_state_count is not None:
            global_state_ids = np.arange(graph.metadata.global_state_count, dtype=np.int64)
        subgraph = SubgraphBatch(
            batch_id=f"{graph.metadata.graph_id}:full_graph",
            node_ids=node_ids,
            edge_ids=edge_ids,
            node_coordinate_left=graph.node_coordinate_left.astype(np.int64, copy=False),
            node_coordinate_right=graph.node_coordinate_right.astype(np.int64, copy=False),
            node_window_left=graph.node_window_left.astype(np.int64, copy=False),
            node_window_right=graph.node_window_right.astype(np.int64, copy=False),
            state_windows=graph.state_windows,
            edge_overlaps=graph.edge_overlaps,
            global_state_ids=global_state_ids,
            local_to_global_state=global_state_ids,
            node_local_index=node_ids,
            edge_local_index=edge_ids,
            metadata={"strategy": "full_graph"},
        )
        return PreparedTrainingBatch(
            subgraph=subgraph,
            node_symbol=graph.node_symbol,
            node_weight=graph.node_weight,
            edge_src=graph.edge_src,
            edge_dst=graph.edge_dst,
            edge_weight=graph.edge_weight,
            node_coordinate_left=graph.node_coordinate_left,
            node_coordinate_right=graph.node_coordinate_right,
            node_window_left=graph.node_window_left,
            node_window_right=graph.node_window_right,
            node_flags=graph.node_flags,
            topo_order=graph.topo_order,
            csr_indptr=graph.csr_indptr,
            csr_indices=graph.csr_indices,
            csc_indptr=graph.csc_indptr,
            csc_indices=graph.csc_indices,
            state_window_offset=None
            if graph.state_windows is None
            else graph.state_windows.offset,
            state_window_length=None
            if graph.state_windows is None
            else graph.state_windows.length,
            edge_overlap_src_offset=None
            if graph.edge_overlaps is None
            else graph.edge_overlaps.src_offset,
            edge_overlap_dst_offset=None
            if graph.edge_overlaps is None
            else graph.edge_overlaps.dst_offset,
            edge_overlap_length=None
            if graph.edge_overlaps is None
            else graph.edge_overlaps.length,
            active_global_states=global_state_ids,
            sequence_ids=None
            if graph.extra is None
            else graph.extra.get("sequence_id"),
            metadata={
                "graph_id": graph.metadata.graph_id,
                "sampling_strategy": "full_graph",
            },
        )

    def prepare_subgraph_batch(
        self,
        sampled_graph: TensorDag,
        subgraph: SubgraphBatch,
    ) -> PreparedTrainingBatch:
        """Materialize a prepared training batch for a sampled subgraph graph."""

        return PreparedTrainingBatch(
            subgraph=subgraph,
            node_symbol=sampled_graph.node_symbol,
            node_weight=sampled_graph.node_weight,
            edge_src=sampled_graph.edge_src,
            edge_dst=sampled_graph.edge_dst,
            edge_weight=sampled_graph.edge_weight,
            node_coordinate_left=sampled_graph.node_coordinate_left,
            node_coordinate_right=sampled_graph.node_coordinate_right,
            node_window_left=sampled_graph.node_window_left,
            node_window_right=sampled_graph.node_window_right,
            node_flags=sampled_graph.node_flags,
            topo_order=sampled_graph.topo_order,
            csr_indptr=sampled_graph.csr_indptr,
            csr_indices=sampled_graph.csr_indices,
            csc_indptr=sampled_graph.csc_indptr,
            csc_indices=sampled_graph.csc_indices,
            state_window_offset=None
            if sampled_graph.state_windows is None
            else sampled_graph.state_windows.offset,
            state_window_length=None
            if sampled_graph.state_windows is None
            else sampled_graph.state_windows.length,
            edge_overlap_src_offset=None
            if sampled_graph.edge_overlaps is None
            else sampled_graph.edge_overlaps.src_offset,
            edge_overlap_dst_offset=None
            if sampled_graph.edge_overlaps is None
            else sampled_graph.edge_overlaps.dst_offset,
            edge_overlap_length=None
            if sampled_graph.edge_overlaps is None
            else sampled_graph.edge_overlaps.length,
            active_global_states=subgraph.global_state_ids,
            sequence_ids=subgraph.sequence_ids,
            metadata={
                "graph_id": sampled_graph.metadata.graph_id,
                "sampling_strategy": subgraph.metadata.get("sampling_strategy", "sequence_batch"),
            },
        )

    def build_subgraph_sampler(self, graph: TensorDag) -> SubgraphSampler:
        """Construct the configured subgraph sampler for the current graph."""

        sampling = self.config.sampling
        return SubgraphSampler(
            graph,
            sequence_batch_size=sampling.sequence_batch_size,
            state_mask_spec=None,
            max_nodes=sampling.max_nodes,
            max_edges=sampling.max_edges,
            seed=self.config.seed,
        )

    def build_step_input(
        self,
        runtime: TrainingRuntimeArtifacts,
        replica: _TrainerReplica,
        step_index: int,
    ) -> TrainingStepInput:
        """Assemble one validated training-step input with scheduled metadata."""

        graph = runtime.graph_artifact.graph
        graph_for_step = graph
        batch = self.prepare_full_graph_batch(graph)
        use_sequence_batch = (
            self.config.sampling.strategy == "sequence_batch"
            and self.config.sampling.sequence_batch_size is not None
            and graph.extra is not None
            and "sequence_id" in graph.extra
            and int(np.asarray(graph.extra["sequence_id"]).shape[0])
            > int(self.config.sampling.sequence_batch_size)
        )
        if use_sequence_batch:
            sampled = self.build_subgraph_sampler(graph).sample_graph(step_index=step_index)
            graph_for_step = sampled.graph
            batch = self.prepare_subgraph_batch(sampled.graph, sampled.subgraph)
        scheduled_loss_weights = self.scheduled_loss_weights(step_index)
        return TrainingStepInput(
            batch=batch,
            parameters=replica.parameters,
            step_index=step_index,
            metadata={
                "_graph": graph_for_step,
                "_reference_parameters": runtime.parameters,
                "graph_id": graph_for_step.metadata.graph_id,
                "initialization_track": runtime.initial_parameters.track.value,
                "phase": self.training_phase(step_index),
                "replica_id": replica.replica_id,
                "scheduled_temperature": self.scheduled_temperature(step_index, replica),
                "scheduled_loss_weights": scheduled_loss_weights,
                "active_escape_mechanisms": self.active_escape_mechanisms(),
                "regularization_switches": self.regularization_switches(),
                "active_regularizers": self.active_regularizers(step_index),
                "replica_metadata": dict(replica.metadata),
            },
        )

    def active_escape_mechanisms(self) -> tuple[str, ...]:
        """Return the enabled escape-mechanism names for logging and checkpoints."""

        config = self.config.escape
        names = []
        if config.multi_start.enabled:
            names.append("multi_start")
        if config.warm_restarts.enabled:
            names.append("warm_restarts")
        if config.annealing.enabled:
            names.append("annealing")
        if config.anchor_release.enabled:
            names.append("anchor_release")
        if config.branch_perturb.enabled:
            names.append("branch_perturb")
        if config.support_expansion.enabled:
            names.append("support_expansion")
        return tuple(names)

    def regularization_switches(self) -> Mapping[str, bool]:
        """Return the on/off state of every regularizer for ablation tracking."""

        regularization = self.config.regularization
        return {
            "transition_anchor": regularization.transition_anchor,
            "emission_anchor": regularization.emission_anchor,
            "emission_smooth": regularization.emission_smooth,
            "logit_l2": regularization.logit_l2,
            "active_state_penalty": regularization.active_state_penalty,
        }

    def active_regularizers(self, step_index: int) -> tuple[str, ...]:
        """Return regularizers that are both switched on and currently weighted."""

        switches = self.regularization_switches()
        scheduled = self.scheduled_loss_weights(step_index).as_mapping()
        names = [
            name
            for name in switches
            if switches[name] and float(scheduled.get(name, 0.0)) != 0.0
        ]
        return tuple(names)

    def training_phase(self, step_index: int) -> str:
        """Return the coarse training phase for logging and scheduling."""

        warmup_end = max(
            self.config.escape.annealing.entropy_warmup_steps,
            self.config.escape.annealing.pairwise_warmup_steps,
        )
        total_steps = self.config.max_steps
        if step_index < warmup_end:
            return "likelihood_warmup"
        if total_steps is not None and total_steps > 0:
            fine_tuning_start = max(warmup_end, int(np.floor(0.8 * total_steps)))
            if step_index >= fine_tuning_start:
                return "fine_tuning"
        return "alignment_shaping"

    def scheduled_temperature(self, step_index: int, replica: _TrainerReplica) -> float:
        """Return the step-specific soft-Viterbi temperature."""

        annealing = self.config.escape.annealing
        if not annealing.enabled:
            return float(replica.temperature_multiplier)
        initial = float(annealing.initial_temperature)
        final = float(annealing.final_temperature)
        decay_steps = annealing.temperature_decay_steps
        if decay_steps is None or decay_steps <= 0:
            base_temperature = final if step_index > 0 else initial
        else:
            progress = min(1.0, float(step_index) / float(decay_steps))
            base_temperature = initial + (final - initial) * progress
        return max(1e-6, base_temperature * float(replica.temperature_multiplier))

    def scheduled_loss_weights(self, step_index: int) -> _ScheduledLossWeights:
        """Return per-step loss weights after annealing and ablation gates."""

        base = self.config.losses
        switches = self.regularization_switches()
        anchor_scale = self.anchor_scale(step_index)
        return _ScheduledLossWeights(
            negative_log_likelihood=float(base.negative_log_likelihood),
            soft_viterbi=float(base.soft_viterbi),
            entropy=self._ramped_weight(
                float(base.entropy),
                step_index,
                self.config.escape.annealing.entropy_warmup_steps,
            ),
            pairwise=self._ramped_weight(
                float(base.pairwise),
                step_index,
                self.config.escape.annealing.pairwise_warmup_steps,
            ),
            transition_anchor=(
                float(base.transition_anchor) * anchor_scale
                if switches["transition_anchor"]
                else 0.0
            ),
            emission_anchor=(
                float(base.emission_anchor) * anchor_scale
                if switches["emission_anchor"]
                else 0.0
            ),
            emission_smooth=(
                float(base.emission_smooth) if switches["emission_smooth"] else 0.0
            ),
            logit_l2=float(base.logit_l2) if switches["logit_l2"] else 0.0,
            active_state_penalty=(
                float(base.active_state_penalty)
                if switches["active_state_penalty"]
                else 0.0
            ),
        )

    def anchor_scale(self, step_index: int) -> float:
        """Return the scheduled scale applied to anchoring regularizers."""

        config = self.config.escape.anchor_release
        if not config.enabled:
            return 1.0
        initial = float(config.initial_scale)
        final = float(config.final_scale)
        decay_steps = config.decay_steps
        if decay_steps is None or decay_steps <= 0:
            return final if step_index > 0 else initial
        progress = min(1.0, float(step_index) / float(decay_steps))
        return initial + (final - initial) * progress

    def build_replicas(self, parameters: PhmmParameterSet) -> tuple[_TrainerReplica, ...]:
        """Construct baseline and perturbed replicas for multi-start training."""

        multi_start = self.config.escape.multi_start
        if not multi_start.enabled:
            return (
                _TrainerReplica(
                    replica_id="replica-0",
                    parameters=self.clone_parameters(parameters),
                    temperature_multiplier=1.0,
                    metadata={"baseline": True},
                ),
            )

        if multi_start.replicas < 1:
            raise ValueError("multi_start.replicas must be at least 1 when enabled")

        rng = np.random.default_rng(self.config.seed)
        replicas = []
        for replica_index in range(int(multi_start.replicas)):
            is_baseline = replica_index == 0
            transition_std = 0.0 if is_baseline else float(multi_start.transition_logit_std)
            emission_std = 0.0 if is_baseline else float(multi_start.emission_logit_std)
            temperature_multiplier = 1.0
            if not is_baseline and float(multi_start.temperature_std) != 0.0:
                temperature_multiplier = max(
                    1e-3, 1.0 + float(rng.normal(0.0, float(multi_start.temperature_std)))
                )
            replicas.append(
                _TrainerReplica(
                    replica_id=f"replica-{replica_index}",
                    parameters=self.clone_parameters(
                        parameters,
                        transition_std=transition_std,
                        emission_std=emission_std,
                        rng=rng,
                    ),
                    temperature_multiplier=temperature_multiplier,
                    metadata={
                        "baseline": is_baseline,
                        "transition_logit_std": transition_std,
                        "emission_logit_std": emission_std,
                    },
                )
            )
        return tuple(replicas)

    def clone_parameters(
        self,
        parameters: PhmmParameterSet,
        transition_std: float = 0.0,
        emission_std: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> PhmmParameterSet:
        """Clone parameter tensors and optionally add small Gaussian perturbations."""

        transition_tensor = self._clone_array(parameters.transition_logits.tensor)
        match_emission = self._clone_array(parameters.match_emission)
        insert_emission = self._clone_array(parameters.insert_emission)
        if transition_std != 0.0:
            transition_tensor = self._perturb_array(transition_tensor, transition_std, rng)
        if emission_std != 0.0:
            match_emission = self._perturb_array(match_emission, emission_std, rng)
        return PhmmParameterSet(
            match_emission=match_emission,
            insert_emission=insert_emission,
            transition_logits=TransitionLogitView(
                tensor=transition_tensor,
                order=tuple(parameters.transition_logits.order),
            ),
            metadata=dict(parameters.metadata),
        )

    def _fit_impl(self) -> FitResult:
        runtime = self.load_runtime_artifacts()
        replicas = self.build_replicas(runtime.parameters)
        if self.config.max_steps is None:
            steps_target = 0
        else:
            steps_target = int(self.config.max_steps)
        if steps_target < 0:
            raise ValueError("max_steps must be non-negative")

        steps_completed = 0
        last_result: Optional[TrainingStepResult] = None
        for replica in replicas:
            for step_index in range(steps_target):
                step_input = self.build_step_input(runtime, replica, step_index)
                last_result = self.training_step(step_input)
                steps_completed += 1

        summary_metrics = {
            "optimization_ready": False,
            "cpu_reference_ready": True,
            "graph_id": runtime.graph_artifact.graph.metadata.graph_id,
            "replica_count": len(replicas),
            "configured_steps_per_replica": steps_target,
            "executed_steps": steps_completed,
        }
        if last_result is not None:
            summary_metrics = {**summary_metrics, **last_result.metrics}

        return FitResult(
            steps_completed=steps_completed,
            final_loss=None if last_result is None else last_result.loss,
            metrics=summary_metrics,
            metadata={
                "initialization_track": runtime.initial_parameters.track.value,
                "active_escape_mechanisms": self.active_escape_mechanisms(),
                "regularization_switches": self.regularization_switches(),
                "active_regularizers": self.active_regularizers(
                    max(0, steps_target - 1) if steps_target else 0
                ),
                "replica_ids": tuple(replica.replica_id for replica in replicas),
                "scheduled_loss_weights": self.scheduled_loss_weights(
                    max(0, steps_target - 1) if steps_target else 0
                ).as_mapping(),
                "last_step_metadata": {}
                if last_result is None
                else dict(last_result.metadata),
            },
        )

    def _ramped_weight(self, base: float, step_index: int, warmup_steps: int) -> float:
        if base == 0.0:
            return 0.0
        if warmup_steps <= 0:
            return base
        if step_index < warmup_steps:
            return 0.0
        total_steps = self.config.max_steps
        if total_steps is None or total_steps <= warmup_steps:
            return base
        progress = (step_index - warmup_steps + 1) / float(total_steps - warmup_steps)
        return base * min(1.0, max(0.0, progress))

    @staticmethod
    def default_pairwise_score_matrix(alphabet_size: int) -> np.ndarray:
        """Return the default mismatch/identity score matrix."""

        if alphabet_size <= 0:
            raise ValueError("alphabet_size must be positive")
        matrix = np.full((alphabet_size, alphabet_size), -1.0, dtype=np.float64)
        np.fill_diagonal(matrix, 1.0)
        return matrix

    @staticmethod
    def _clone_array(value: Any) -> Any:
        array = np.asarray(value)
        return np.array(array, copy=True)

    @staticmethod
    def _perturb_array(
        value: np.ndarray, std: float, rng: Optional[np.random.Generator]
    ) -> np.ndarray:
        if std == 0.0:
            return value
        generator = np.random.default_rng() if rng is None else rng
        return value + generator.normal(0.0, std, size=value.shape)
