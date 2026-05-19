# AD-PHMM-align plan

## Purpose

AD-PHMM-align is the Python/PyTorch project for differentiable PHMM training and alignment on DAGs. It consumes typed artifacts from `Pre-AD-prep` and keeps graph conversion, reference-MSA construction, global coordinate construction, and CPU-heavy preprocessing outside the training loop.

## Project boundary

- `Pre-AD-prep` produces `tensor_graph.v1` artifacts, initialization tracks, packed state windows, edge-window overlaps, global/subgraph projections, and diagnostics.
- `AD-PHMM-align` loads those artifacts, validates contracts, instantiates trainable PHMM tensors, runs differentiable forward/backward and Viterbi-style programs, trains with SGD, records profiling data, decodes alignments, and reports metrics.
- Mature AD-PHMM-align workflows should not parse legacy DAG-align pickle/object arrays directly.
- The first end-to-end training path should use `legacy_current` initialization as the baseline; `reference_msa` should be treated as a later comparison initializer once baseline training is working.

## Coordinate convention

All PHMM state intervals are half-open: `[left, right)`.

This convention matches NumPy/PyTorch slicing, makes packed-window lengths exactly `right - left`, and avoids inclusive-bound off-by-one ambiguity. `Pre-AD-prep` and AD-PHMM-align must use the same convention in manifests, node windows, edge overlaps, subgraph projections, and sampling ranges.

## Core artifact contracts

AD-PHMM-align should treat the following as first-class public contracts:

- tensor graph metadata and manifest information;
- node arrays, edge arrays, CSR/CSC adjacency, topological order, and optional topological levels;
- half-open node state windows;
- packed window offsets and lengths;
- edge-window overlap offsets and lengths;
- source-format, alphabet, symbol-encoding, and array-identity manifest fields shared with `Pre-AD-prep`;
- sequence/source provenance needed for likelihood scaling and alignment-quality losses;
- initialization tracks: baseline `legacy_current` plus comparative `reference_msa`;
- global-to-subgraph state projections and active-state masks.

Training must refuse to run when required window/projection arrays are absent, while validation and diagnostics may still inspect partial artifacts.

## Data-preparation module design

Initial public interfaces should support:

- loading a typed graph artifact into a `TensorDag`;
- validating half-open intervals, packed windows, and edge-window overlaps;
- representing sampled subgraphs without losing global state IDs;
- constructing prepared training batches with tensor-like fields but without importing PyTorch at package import time;
- carrying batch metadata, sequence counts, likelihood scaling weights, and optional profiling information.

## Training module design

Initial public interfaces should support:

- `TrainingConfig` with optimizer, loss weights, subgraph sampling, device, reproducibility, and profiling options;
- `TrainingStepInput` as the boundary between data preparation and PHMM training;
- `TrainingStepResult` with total loss, component losses, metrics, step index, batch ID, and profiling results;
- `FitResult` for high-level training summaries and checkpoint metadata;
- trainer placeholders that make future implementation expectations explicit.

## Profiling from the start

CPU time and memory profiling should be available before heavy optimization begins. Early training and data-preparation steps should be able to record wall time, process CPU time, peak resident memory, and optional device-memory metrics. These results should flow through training-step outputs and progress diagnostics so later optimizations have a baseline.

## Baseline execution strategy

The first end-to-end training path should be built in the following order:

1. load and validate `tensor_graph.v1` plus `legacy_current` initialization artifacts;
2. build a CPU reference implementation for forward, backward, posterior summaries, soft-Viterbi, and hard Viterbi on tiny DAGs;
3. lift the same packed-window recurrences to PyTorch tensors without changing the graph/coordinate contracts;
4. add CUDA-oriented wavefront scheduling and kernel-friendly tensor layouts only after the reference path is numerically stable;
5. use `reference_msa` initialization only after the baseline path is already training correctly.

This keeps correctness, range semantics, and branching/merging behavior fixed before heavy GPU optimization begins.

## Algorithm priorities

### 1. Forward/backward with branching and merging DAG paths

The forward and backward algorithms should treat the graph as a DAG of emission sites and the PHMM position axis as the second dynamic-programming dimension.

Core implementation rules:

- Process forward messages strictly in topological order and backward messages in reverse topological order.
- Keep one shared recurrence layout for `M`, `I`, and `D` channels over each node's packed state window.
- At branch points, a source node should fan out messages independently along each outgoing edge.
- At merge points, destination-node messages should combine all predecessor contributions with log-sum-exp for likelihood and posterior computation.
- Hard Viterbi should reuse the same data flow but replace log-sum-exp merge reduction with max reduction plus sparse backpointers.
- Soft-Viterbi should reuse the same layout again, differing only in the temperature-controlled merge/transition reduction.

The plan should avoid separate graph-specific and sequence-specific algorithm families. There should be one DAG-aware recurrence interface with pluggable reduction behavior:

- forward/backward: log-sum-exp;
- hard Viterbi: max + argmax;
- soft Viterbi: temperature-smoothed log-sum-exp/max interpolation.

### 2. Effective node/position ranges in forward, backward, and Viterbi

Static state windows exported by `Pre-AD-prep` are necessary but not sufficient. AD-PHMM-align should also compute pass-specific effective ranges so kernels do not spend work on unreachable node/position pairs.

The implementation should distinguish four layers:

1. **Raw coordinate span**: the propagated full-graph coordinate/legal interval for each node.
2. **Static DP window**: the padded half-open `[left, right)` interval actually paired with packed offsets/lengths.
3. **Edge transfer support**: overlap metadata that maps source packed states into destination packed states.
4. **Pass-specific effective support**: the subset of each node window actually reached in forward, backward, or Viterbi for the current subgraph/batch.

Planned behavior:

- Preserve the raw coordinate span for analysis, diagnostics, and future sampling logic, but drive DP kernels from the static DP window.
- Start from the static node window as the legal bound for all kernels.
- During forward, derive effective support at node `v` from the union of predecessor-transferred support intersected with `v`'s legal window.
- During backward, derive effective support symmetrically from successor-transferred support.
- For Viterbi, keep the same effective support rules and only allocate backpointer storage for reachable packed cells.
- Preserve exact reachability with boolean masks when branch/merge structure creates holes inside a node window; do not rely only on coarse `[min, max)` tightening.
- Optionally cache coarse contiguous spans alongside masks for kernel launch sizing and diagnostics.

This means the project needs both an explicit coordinate-vs-window distinction and an explicit range/mask layer, rather than treating one exported interval pair as both the raw coordinate span and the packed DP window.

### 3. CUDA parallelization

CUDA work should be planned around the DAG dependency structure, not around dense rectangular DP tables.

The intended execution model is:

- flatten node-state work into packed ragged buffers using the exported window offsets and lengths;
- schedule nodes by topological levels or dependency-safe wavefront frontiers;
- compute edge-local transfers in parallel across edges whose source values are ready;
- reduce incoming edge contributions into destination node-state buffers with segmented log-sum-exp or max reductions;
- keep channel-major or state-major memory layout consistent across forward, backward, and Viterbi to avoid repeated transposes.

The first GPU path should prefer PyTorch tensor ops and fused reductions over custom CUDA code. Only after profiling should the project decide whether custom CUDA/Triton kernels are needed for:

- segmented merge reductions at high in-degree nodes;
- repeated edge-overlap gather/scatter;
- fused transition + emission + reduction kernels for large packed windows.

## Alignment assembly, scoring, and training objective

### 1. Hard-Viterbi alignment assembly

AD-PHMM-align needs a concrete path from hard Viterbi output to alignment artifacts, not just a best-score state trace.

Planned decode flow:

1. run hard Viterbi on the packed DAG/state layout;
2. backtrack one best state/channel path per sequence-induced graph traversal or sampled subgraph view;
3. project local packed states back to global PHMM state IDs;
4. merge sequence-wise path assignments into alignment columns keyed by global PHMM state positions plus insertion slots;
5. emit both:
   - a compact internal alignment-column representation for metrics/losses;
   - a materialized alignment artifact for evaluation/export.

This means `eval/decode.py` should eventually distinguish:

- **state traceback**: packed/local/global PHMM state path;
- **alignment assembly**: column-centric representation suitable for entropy and SP scoring;
- **format export**: FASTA/zip-like output for reporting and comparison with legacy DAG-align.

### 2. Entropy and sum-of-pairs scoring

The current DAG-align evaluation path computes:

- a scaled sum-of-pairs score from alignment columns;
- total entropy over alignment columns;
- average core-column entropy for low-gap columns.

AD-PHMM-align should preserve that separation:

- **hard decoded metrics** for reporting, checkpoint selection, and baseline comparison;
- **differentiable surrogates** for training.

Planned metric split:

- `eval/alignment_metrics.py`:
  - hard decoded SP score;
  - hard decoded total entropy;
  - optional core-column entropy;
  - alignment length and gap diagnostics.
- `losses/entropy.py`:
  - posterior/state-column entropy surrogate;
  - optional concentration penalty over ambiguous state occupancy.
- `losses/pairwise.py`:
  - differentiable sum-of-pairs approximation from soft co-assignment / posterior column occupancy;
  - hard decoded SP as a detached monitoring metric.

The first implementation should avoid trying to backpropagate through hard column assembly. Instead:

- use forward/backward or soft-Viterbi outputs to define soft occupancy/co-assignment tensors;
- compute differentiable entropy and SP surrogates there;
- use hard decoded alignment only for evaluation and progress diagnostics.

Detailed design choice for the first differentiable version:

- hard decoded metrics should continue to use the fully assembled alignment;
- differentiable entropy/SP surrogates should first operate on **global match-state columns** only, because those columns have stable identities across subgraphs and batches;
- insert-slot-aware differentiable scoring can be added later after the match-column path is numerically stable.

Planned soft column statistics for a batch:

- let `gamma[u, m]` be posterior occupancy of emitted observation/node `u` at global match state `m`;
- let `base(u)` be the emitted symbol for `u`;
- define soft match-column counts

```text
c[m, a] = sum_{u in batch, base(u)=a} gamma[u, m]
```

- define soft column occupancy

```text
n[m] = sum_a c[m, a]
rho[m] = n[m] / max(1, n_seq_batch)
```

- define a soft core-column weight that mirrors the legacy gap-threshold idea (`gap_fraction < 0.7`, equivalently occupancy above about `0.3`):

```text
w_core[m] = sigmoid(alpha_core * (rho[m] - tau_core))
```

with initial defaults:

- `tau_core = 0.3`
- `alpha_core = 10`

Planned differentiable entropy surrogate:

```text
p[m, a] = c[m, a] / max(eps, n[m])
H[m] = -sum_a p[m, a] * log(p[m, a] + eps)
L_entropy = sum_m w_core[m] * H[m] / sum_m w_core[m]
L_entropy_norm = L_entropy / log(|alphabet|)
```

This keeps entropy smaller for cleaner/more decisive columns and normalizes it into a roughly `[0, 1]` range.

Planned differentiable scaled-SP surrogate:

For a base-pair score matrix `S[a, b]` matching the legacy positive/negative scoring tables,

```text
SP_col[m]
  = sum_a 0.5 * c[m, a] * max(c[m, a] - 1, 0) * S[a, a]
  + sum_{a < b} c[m, a] * c[m, b] * S[a, b]

R_sp
  = 2 * sum_m SP_col[m]
    / max(eps, n_seq_batch * (n_seq_batch - 1) * max(1, n_match_active))
```

where `n_match_active` is the number of match columns with non-trivial occupancy.

This mirrors the structure of the legacy scaled-SP score while staying differentiable through the soft counts `c[m, a]`. Since larger SP is better, this quantity should enter the total loss with a negative sign.

### 3. Loss composition

The training loss should be a weighted composition of:

- **negative log-likelihood** from forward log-likelihood;
- **entropy term** from posterior or soft alignment occupancy;
- **sum-of-pairs term** from a differentiable soft alignment surrogate;
- **regularization** informed by initialization priors and model complexity.

The important rule is: do **not** combine raw NLL, raw total entropy, and raw SP directly. They have different units, different scaling with batch size/alignment length, and NLL is correlated with alignment quality but not strictly monotonic with entropy/SP.

Planned normalized objective:

```text
L_nll
  = -log p(batch | theta) / max(1, n_emit_batch)

L_total
  = w_nll * L_nll
  + w_entropy * L_entropy_norm
  - w_sp * R_sp
  + w_tr_anchor * R_tr_anchor
  + w_em_anchor * R_em_anchor
  + w_em_smooth * R_em_smooth
  + w_logit_l2 * R_logit_l2
  + w_active_state * R_active_state
```

where:

- `n_emit_batch` is the number of emitted observations/nodes in the batch;
- `L_entropy_norm` is the normalized soft core-column entropy above;
- `R_sp` is the differentiable scaled-SP reward above;
- `R_active_state` is zero until state-position sampling is enabled.

Target sign convention:

```text
total_loss
  = w_nll * negative_log_likelihood
  + w_entropy * entropy_penalty
  - w_sp * soft_sum_of_pairs_score
  + w_reg_transition * transition_regularization
  + w_reg_emission * emission_regularization
  + w_active_state * active_state_penalty
```

Implementation note: scoring-style terms and penalty-style terms should keep their natural meanings internally, and the loss builder should apply the sign. That avoids confusing configuration semantics later.

Planned training schedule:

1. **likelihood warmup**
   - optimize `L_nll +` anchoring regularizers only;
   - keep `w_entropy = 0`, `w_sp = 0`.
2. **alignment shaping**
   - linearly ramp `w_entropy` and `w_sp` from zero to target values after likelihood stabilizes;
   - keep anchoring regularizers active but decaying.
3. **fine-tuning**
   - lower anchoring weights further;
   - keep `w_sp` moderate and `w_entropy` small but non-zero;
   - optionally enable state-position-sampling penalties.

This makes NLL the primary optimization driver early on and prevents the alignment-score surrogates from destabilizing the model before the PHMM has learned a sensible likelihood landscape.

### 4. Escape mechanisms beyond plain SGD

The design should not rely on SGD noise alone to escape poor local minima. The first optimization stack should include explicit exploration and recovery mechanisms.

#### A. Multi-start from a common structured initializer

For each graph/training run, launch a small ensemble of runs from the same chosen initialization track (`legacy_current` first), but with controlled structured perturbations:

- **transition perturbation**
  - add small zero-mean noise to transition logits within each normalized transition family before softmax;
- **emission perturbation**
  - add smaller noise to match-emission logits, with stronger damping in highly confident initialized columns;
- **temperature perturbation**
  - start soft-Viterbi / soft occupancy with slightly different temperatures across replicas.

Design rule:

- keep perturbations small enough that every replica remains in the same broad basin as the initialization;
- use 3-5 replicas first, not a large population;
- select checkpoints by a monitored Pareto view of `L_nll`, hard scaled-SP, and hard entropy, not NLL alone.

This is the cleanest non-destructive escape mechanism because it preserves the initialization prior while exploring nearby basins.

#### B. Cosine restart schedule for the optimizer

Use cosine decay with warm restarts rather than one monotone learning-rate schedule.

Planned behavior:

- each cycle decays the learning rate;
- on restart, raise the learning rate again but not necessarily to the original maximum;
- decay anchor weights across cycles so later cycles explore more freely than early ones.

Why:

- warm restarts are a simple basin-escape mechanism that does not require changing the probabilistic model;
- they fit naturally with the staged objective where early cycles are likelihood-dominated and later cycles include SP/entropy shaping.

#### C. Objective annealing / smoothing continuation

Treat the training objective itself as a continuation path:

1. smoother objective first;
2. sharper objective later.

Concretely:

- start with higher-temperature soft reductions for soft-Viterbi / posterior-derived soft alignment surrogates;
- ramp down temperature gradually;
- ramp `w_sp` and `w_entropy` up only after the likelihood landscape has become stable.

This gives a smoother optimization surface early on and reduces the chance of committing too early to brittle alignments.

#### D. Controlled anchor release

Anchoring regularizers help avoid bad regions early, but they can also trap the model near a mediocre initializer if kept too strong.

So anchoring should follow a release schedule:

- strong in warmup;
- moderate in alignment-shaping;
- weak in fine-tuning;
- temporarily relaxed further during restart cycles if plateau is detected.

This creates an explicit mechanism for leaving a mediocre initialization basin without discarding the value of initialization altogether.

#### E. Plateau-triggered branch-and-perturb

Add a stagnation detector on a moving window of:

- normalized NLL;
- hard scaled-SP;
- hard entropy.

If all monitored metrics plateau or if NLL improves while alignment metrics stall/regress for several windows:

1. branch from the best recent checkpoint;
2. apply a small logit perturbation;
3. temporarily increase learning rate and soften anchoring;
4. continue training as a new branch.

Keep only a small number of active branches and prune clearly inferior ones.

This is the main explicit recovery path when the objective appears trapped in a poor compromise between likelihood and alignment quality.

#### F. Sampling-support expansion on demand

Once PHMM state-position sampling is enabled, overly narrow sampled supports can create artificial local minima by excluding the right alignment states.

So if plateau is detected under sampled-state training:

- temporarily widen candidate state ranges;
- increase proposal coverage around high-entropy regions;
- rerun a few recovery steps before shrinking the support again.

This is important because some failures will come from support truncation rather than parameter optimization alone.

#### G. What to use first

The initial implementation should include these escape mechanisms in priority order:

1. multi-start replicas with small structured perturbations;
2. cosine warm restarts;
3. objective/temperature annealing;
4. controlled anchor release;
5. branch-and-perturb recovery;
6. sampled-support expansion later, once state-position sampling exists.

That gives the project multiple escape routes without making the first trainer overly complex.

### 5. Regularization design

Regularization should be tied to the initialization pathways and profile structure rather than treated as generic weight decay.

The first exact regularization set should be:

#### A. Transition-anchor regularization

Use KL-to-initialization on each normalized outgoing transition distribution.

For each transition family/state group `k`, let:

- `pi_k(theta)` be the current transition distribution after softmax/log-softmax;
- `pi_k^(0)` be the corresponding initialized distribution from `legacy_current` or `reference_msa`.

Define:

```text
R_tr_anchor = mean_k KL(pi_k^(0) || pi_k(theta))
```

Apply this to:

- start-state probabilities;
- `M -> {M, I, D, End}` families where applicable;
- `I -> {M, I, D, End}` families where applicable;
- `D -> {M, D, I, End}` families if the chosen parameterization exposes them.

Why this exact form:

- it matches the meaning of the legacy `trProbAdds_*` log-add priors;
- it keeps the model near the graph-derived baseline early on;
- it still allows the trained model to move away when the likelihood term provides evidence.

#### B. Emission-anchor regularization

Use KL-to-initialization on match emissions, plus background anchoring for insert emissions if inserts are parameterized separately.

Let:

- `e_m(theta)` be the current emission distribution for match state `m`;
- `e_m^(0)` be the initialized emission distribution for `m`;
- `b_bg` be the alphabet background distribution.

Define:

```text
R_em_anchor
  = mean_m KL(e_m^(0) || e_m(theta))
    + lambda_insert_bg * mean_i KL(b_bg || e_i(theta))
```

Why this exact form:

- it preserves the reference/global-graph initialization signal;
- it mirrors the role of `emProbAdds_Match*` smoothing in legacy DAG-align;
- it prevents insert emissions from drifting into implausibly sharp sequence-specific profiles.

#### C. Adjacent-match emission smoothness

Use Jensen-Shannon divergence between neighboring match-state emission distributions:

```text
R_em_smooth = mean_m JS(e_m(theta), e_{m+1}(theta))
```

Why this exact form:

- adjacent PHMM match positions usually vary smoothly in realistic profiles;
- JS works directly on probability distributions and is symmetric and bounded;
- it is easier to interpret than raw logit second-difference penalties.

This should be weaker than emission anchoring and mainly act as a stability prior.

#### D. Small logit-norm safety penalty

Use a very small quadratic penalty on unconstrained logits:

```text
R_logit_l2 = mean(logits_transition^2) + mean(logits_emission^2)
```

This is not the main prior. It is only a numerical safety term to discourage runaway logits and overconfident degenerate solutions.

#### E. Active-state / sampled-range penalty

This term is inactive until PHMM state-position sampling is implemented.

For hard contiguous sampled intervals:

```text
R_active_state = mean_batch sampled_width / max(1, proposal_width)
```

For a later soft-gated sampler:

```text
R_active_state = mean_batch sum_s gate[s] / max(1, proposal_width)
```

Why this exact form:

- it encourages the sampler to use the narrowest state range that still explains the batch;
- it makes reference-MSA proposal ranges useful as informative but non-binding supports;
- it prevents the sampling path from collapsing back to always using nearly the full PHMM axis.

#### F. What not to regularize first

The initial implementation should **not** start with:

- a direct penalty on hard Viterbi paths;
- a penalty based on hard decoded alignment entropy;
- aggressive delete-state occupancy penalties;
- many separate handcrafted head/tail penalties inside AD training.

Those are better handled by:

- initialization metadata imported from preprocessing;
- the main probabilistic objective;
- the simpler anchoring/smoothness priors above.

## State-position sampling and global/subgraph mapping

### 1. Sampling target

Sampling in AD-PHMM-align is not just subgraph sampling; it must also support sampling PHMM state positions while preserving the mapping through:

- the full global PHMM state axis;
- the global graph artifact;
- the sampled training subgraph;
- the local packed window layout used by DP kernels.

This requires two distinct but linked sampling layers:

1. **subgraph sampling** over graph nodes/edges/sequences;
2. **state-position sampling** over global PHMM states or candidate state ranges.

### 2. Mapping requirements

Every training batch should eventually carry:

- sampled graph nodes/edges;
- local packed node windows;
- `global_state_ids` / `local_to_global_state`;
- active-state masks or candidate ranges;
- enough metadata to reconstruct how sampled states relate to the full graph and PHMM.

The mapping contract should support:

- full-range training with no state subsampling;
- contiguous candidate-range training;
- sparse active-state masks with holes;
- future reference-MSA-driven proposals.

### 3. Sampling strategy progression

Planned progression:

1. **baseline training**
   - no PHMM state-position sampling;
   - use full exported static windows and effective-support masks only.
2. **coarse contiguous sampling**
   - sample one or a few global PHMM intervals per batch;
   - intersect with node windows to build local active-state masks.
3. **proposal-driven sampling**
   - use reference-MSA-derived candidate ranges, uncertainty scores, insert burdens, and occupancy diagnostics.
4. **adaptive sampling**
   - bias toward uncertain/high-entropy/high-gradient regions once the basic training loop is stable.

This avoids coupling the very first training path to the still-unimplemented proposal machinery.

## Initialization pathways and legacy factors

### 1. Baseline `legacy_current`

The baseline path should explicitly preserve the main factors used by current DAG-align:

- **global graph estimation**
  - graph reduction/merging is performed before PHMM initialization;
  - a thresholded coarse reference graph is built from that merged global graph.
- **reference-derived emission initialization**
  - `thr_*.npz` provides `ref_seq`, `ref_node_list`, `emProbMatrix`, and insert ranges;
  - the initial PHMM is seeded from those global graph/reference statistics.
- **empirical priors / smoothing**
  - `init_M2D`, `init_M2I`, `init_I2I`, `init_D2M`, `init_M2End`;
  - `trProbAdds_*` transition priors;
  - `emProbAdds_Match*` emission smoothing;
  - head/tail-specific heuristics.

In AD-PHMM-align these factors should be separated into:

- initialization tensors loaded from `legacy_current`;
- explicit metadata about the priors/smoothing that produced them;
- regularization terms that can optionally keep training near that baseline early on.

### 2. Comparative `reference_msa`

The later `reference_msa` path should provide two outputs, not one:

1. initialization tensors;
2. candidate sampling ranges for PHMM state positions.

The coarse reference-derived MSA should therefore feed:

- match/insert emission priors;
- transition priors if supported by the MSA statistics;
- proposal scores for state-position sampling ranges;
- diagnostics describing support, uncertainty, insert-heavy regions, and gap structure.

### 3. Implementation rule

Do not entangle the baseline training path with the `reference_msa` path at first.

The correct order is:

1. finish baseline training with `legacy_current`;
2. make alignment assembly and hard metrics stable;
3. add differentiable entropy/SP losses;
4. add state-position sampling;
5. then compare `reference_msa` initialization and proposal-driven sampling against the baseline.

## Planned module expansion

The current scaffold already has the right top-level packages. The next implementation split should be:

- `io/`: manifest and initialization loaders for `tensor_graph.v1`;
- `phmm/forward_backward.py`: CPU reference forward/backward plus a backend-neutral recurrence interface;
- `phmm/viterbi.py`: hard-Viterbi decode on the same packed graph/state layout;
- `phmm/soft_viterbi.py`: soft-Viterbi score on the same recurrence core;
- `phmm/parameters.py`: tensor materialization, validation, and transition-view helpers;
- `graph/` or `sampling/`: effective-range propagation, packed masks, and subgraph-to-global state mapping utilities;
- `train/trainer.py`: baseline training step orchestration once loaders and DP kernels are in place.

If the recurrence code grows, it should split further into:

- `phmm/ranges.py` for effective support propagation and mask construction;
- `phmm/backend.py` for NumPy/PyTorch backend helpers;
- `phmm/wavefront.py` for level scheduling and merge reduction planning.

## Implementation phases

1. Keep the scaffold importable with NumPy-only core dependencies and optional PyTorch training extras.
2. Finalize typed artifact, half-open interval, packed-window, edge-overlap, and subgraph batch contracts.
3. Implement artifact loaders for `tensor_graph.v1` and initialization manifests, starting with `legacy_current`.
4. Add effective-range propagation utilities for forward, backward, and Viterbi support masks on packed windows.
5. Implement CPU reference forward/backward on tiny DAGs, including branch fan-out and merge log-sum-exp.
6. Implement posterior summaries and likelihood losses on top of the CPU reference path.
7. Implement hard and soft Viterbi on the same packed recurrence layout, including sparse backpointer storage for reachable packed cells.
8. Lift the reference recurrences to PyTorch tensors with dependency-safe wavefront scheduling.
9. Profile the PyTorch backend, then optimize the CUDA-oriented path for packed windows, segmented merge reductions, and overlap transfer kernels.
10. Add losses, metrics, subgraph SGD, checkpointing, and profiling reports on top of the stabilized PyTorch backend.
11. Compare `reference_msa` initialization against the `legacy_current` baseline once the end-to-end training path is stable.
12. Iterate with `Pre-AD-prep` whenever training needs additional exported arrays, diagnostics, or layout changes.

## Detailed execution plan

### Phase A: loaders and shared tensor views

Implement:

- tensor-graph manifest loader;
- `.npy` array loader and dtype validation;
- initialization manifest loader for `legacy_current`;
- helpers that expose transition tensors in consistent named views rather than raw dictionary access.

Acceptance targets:

- one tiny training-ready fixture loads end to end;
- graph windows and edge overlaps validate on load;
- initialization tensor shapes match `global_state_count` and alphabet size.

### Phase B: effective-range engine

Implement:

- packed boolean active masks over node windows;
- coarse span summaries for diagnostics and launch sizing;
- forward-reachable support propagation from source nodes;
- backward-reachable support propagation from sink nodes;
- intersection helpers for posterior and Viterbi support.

Acceptance targets:

- chain and diamond DAG fixtures produce hand-checked reachable masks;
- branch/merge cases with disconnected packed cells keep exact masks rather than being widened silently;
- diagnostics report active packed-state counts per batch.

### Phase C: CPU reference forward/backward

Implement:

- NumPy log-space forward pass over packed node windows and topological order;
- reverse-topological backward pass on the same layout;
- per-edge transfer using overlap metadata;
- merge reduction with log-sum-exp;
- source/sink handling and graph-level likelihood aggregation.

Acceptance targets:

- chain-graph results agree with a non-DAG PHMM baseline when the DAG degenerates to a path;
- diamond-graph likelihood matches hand-computed small examples;
- forward and backward agree on total log-likelihood within numerical tolerance.

### Phase D: CPU hard/soft Viterbi

Implement:

- hard-Viterbi score/path on the same packed buffers;
- soft-Viterbi score with temperature parameter;
- sparse backpointer storage keyed by reachable packed cells only;
- parity checks between hard-Viterbi, soft-Viterbi at low temperature, and forward likelihood bounds.

Acceptance targets:

- decoded states always lie inside effective ranges;
- low-temperature soft-Viterbi approaches hard-Viterbi on tiny fixtures;
- branch/merge backpointers reconstruct valid DAG-respecting paths.

### Phase E: PyTorch wavefront backend

Implement:

- PyTorch tensor version of the packed recurrences;
- dependency-safe execution by topological levels or computed frontiers;
- batched edge-transfer gather/scatter;
- segmented merge reductions for both forward/backward and Viterbi-style kernels.

Acceptance targets:

- numerical parity with CPU reference on tiny and medium fixtures;
- autograd flows through forward likelihood and soft-Viterbi objectives;
- profiling captures per-step wall/CPU/device memory metrics.

### Phase F: alignment assembly and metric stack

Implement:

- hard-Viterbi traceback to global PHMM state paths;
- alignment-column assembly from decoded paths;
- hard decoded entropy and SP metrics compatible with legacy reporting;
- detached monitoring metrics separated from differentiable training losses.

Acceptance targets:

- decoded alignments can be materialized from Viterbi outputs on tiny fixtures;
- hard SP/entropy agree with hand-computed small examples;
- decoded alignments preserve global PHMM state identity through subgraph projection.

### Phase G: loss and regularization stack

Implement:

- posterior entropy loss;
- differentiable soft SP approximation;
- transition/emission regularization informed by initialization metadata;
- one loss builder that combines likelihood, entropy, SP, and regularization with explicit sign handling.

Acceptance targets:

- every component can be enabled/disabled independently;
- hard metrics and soft loss terms are logged separately;
- loss decomposition is stable enough for step-by-step debugging.

### Phase H: state-position sampling

Implement:

- no-subsampling baseline mode;
- contiguous global-range proposals intersected with local node windows;
- explicit global-to-local state projections in training batches;
- placeholder interfaces for later `reference_msa` proposal-driven sampling.

Acceptance targets:

- sampled state ranges map consistently through graph/subgraph/local packed layouts;
- training batches can recover global PHMM state IDs exactly;
- baseline training still runs with full windows when sampling is disabled.

### Phase I: CUDA-oriented optimization

Focus areas:

- flatten packed `(node, state, channel)` work into kernel-friendly contiguous buffers;
- reduce launch overhead by batching nodes within the same dependency frontier;
- fuse emission lookup, transition application, and edge-merge reduction where profiling justifies it;
- keep one memory layout for forward, backward, and Viterbi kernels to minimize reshaping overhead.

Optimization guardrails:

- no custom CUDA path before CPU/PyTorch parity tests pass;
- every optimization must preserve effective-range masks exactly;
- merge-node reductions and overlap transfers are the first profiling hotspots to target.

## Test and validation priorities

The first AD implementation pass should add focused correctness fixtures for:

- linear DAG/path equivalence;
- diamond branch/merge graphs;
- nodes with narrow windows and zero-length overlaps;
- branch/merge cases that create holes inside otherwise broad legal windows;
- CPU vs PyTorch parity for forward likelihood, posterior support, hard Viterbi, and soft Viterbi;
- hard decoded alignment assembly on tiny fixtures;
- hard vs soft entropy/SP monitoring consistency on simple cases;
- full-range vs sampled-range batch mapping consistency;
- gradient sanity checks for the differentiable likelihood and soft-Viterbi objectives.
