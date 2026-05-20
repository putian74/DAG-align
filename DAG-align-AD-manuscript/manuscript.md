# DAG-align-AD: Differentiable Profile Hidden Markov Models on Directed Acyclic Graphs

**Draft status:** paper scaffold  
**Scope:** differentiable PHMM training, inference, decoding, and evaluation on directed acyclic graphs  
**Out of scope for this paper:** graph construction and graph merging internals, which belong to the DAG-rust paper

---

## Abstract

**Scaffold note:** keep the abstract centered on the model and inference contribution rather than the graph engine.

Planned abstract flow:

1. State the need for probabilistic alignment and training directly on sequence DAGs rather than only on linear references or static MSAs.
2. Introduce **DAG-align-AD** as a differentiable PHMM framework that consumes typed DAG artifacts and performs forward/backward, hard Viterbi, soft Viterbi, decoding, and training on graph-structured inputs.
3. Emphasize the key technical ideas:
   - DAG-aware dynamic programs over packed state windows;
   - merge-safe reductions at branch/gather points;
   - explicit effective-support masks/ranges;
   - differentiable entropy and scaled-SP style objectives;
   - subgraph-aware training and decoding.
4. Conclude with the main result theme: a complete and extensible differentiable alignment framework on DAGs that separates graph preprocessing from model training while preserving exact graph-aware semantics.

## 1. Introduction

The original DAG-align work established that sequence DAGs can compress large collections of related sequences and support scalable alignment workflows. However, the probabilistic modeling layer of that system remained tied to a more conventional training setup. The next step is to move from a graph-aware alignment pipeline to a **differentiable probabilistic modeling framework on DAGs**, so that training objectives, inference procedures, and alignment quality metrics can be optimized within a unified model-driven system.

This manuscript introduces **DAG-align-AD**, a follow-up framework that treats the DAG not merely as a compressed preprocessing artifact, but as the native substrate for profile hidden Markov model inference and training. The core goal is to define a PHMM formulation that operates directly on directed acyclic graphs, admits graph-aware forward/backward and decoding recurrences, and can be optimized with modern automatic differentiation tooling while preserving the structural constraints exported by the preprocessing pipeline.

The scope of this paper is intentionally narrower than a full platform paper and intentionally different from the DAG-rust paper. DAG-rust is about graph construction, graph merging, persistence, and efficient graph-native systems design. DAG-align-AD is about the **probabilistic model on top of those graphs**: how typed graph artifacts are loaded, how dynamic programs are executed on packed graph/state layouts, how hard and soft decoding are defined, how training objectives are formulated, and how the whole model is trained and evaluated.

### 1.1. Positioning relative to the DAG-rust paper

The DAG-rust paper and the DAG-align-AD paper should read as complementary but clearly separate contributions.

- **DAG-rust** answers: how should large sequence DAGs be built, merged, stored, and exported safely and efficiently?
- **DAG-align-AD** answers: given a typed DAG representation, how should a differentiable PHMM be defined, trained, decoded, and evaluated directly on that graph?

That boundary should be maintained consistently throughout the manuscript. Graph construction details should only appear here when they are necessary to define the artifact contract or to justify model-side assumptions.

## 2. Core claim and contributions

**Core claim:** profile hidden Markov models can be trained and decoded directly on directed acyclic graphs within a differentiable framework, provided that graph support, state-window structure, and merge/branch reductions are handled explicitly rather than forced into a linear-sequence approximation.

Planned contribution list:

1. **A differentiable PHMM formulation on DAGs.**
2. **Graph-aware forward/backward recurrences over packed node-state windows.**
3. **Hard and soft Viterbi decoding on the same DAG-aware dynamic-programming substrate.**
4. **Explicit effective-support masks and ranges for reachable node/state pairs.**
5. **A training framework that separates typed graph preprocessing from PyTorch-side optimization.**
6. **Subgraph-aware training and decoding interfaces that preserve global PHMM state identity.**
7. **A model/evaluation stack that connects posterior summaries, hard decoded alignments, entropy-like objectives, scaled-SP style objectives, and regularization.**

## 3. Problem formulation

This section should formalize the computational objects used by DAG-align-AD.

### 3.1. Inputs

- a typed sequence DAG exported by preprocessing;
- node and edge arrays;
- topological order and optional wavefront/group metadata;
- per-node raw coordinate spans and static DP windows;
- edge-window overlap metadata;
- sequence/source provenance metadata;
- initialization tracks such as `legacy_current` and later `reference_msa`.

### 3.2. Model

Define the PHMM state family over global state positions:

- match states;
- insertion states;
- deletion states;
- start/end handling;
- parameter tensors for transitions and emissions;
- graph-conditioned legality constraints induced by static windows and overlaps.

### 3.3. Outputs

- forward likelihood summaries;
- backward/posterior summaries;
- hard Viterbi paths;
- soft Viterbi scores or distributions;
- hard decoded alignment artifacts;
- differentiable loss values and trainer-side metrics.

## 4. Artifact and runtime interface

This section should explain the boundary between preprocessing and model execution.

### 4.1. Why typed artifacts matter

The differentiable model should not parse legacy Python pickle/object-array graph data directly. Instead, it should consume typed artifacts produced upstream. This keeps graph construction concerns outside the training loop, stabilizes contracts between projects, and makes the model layer easier to validate and optimize independently.

### 4.2. Required contracts

Describe the artifact fields that are essential to model execution:

- graph structure and adjacency;
- packed state-window offsets and lengths;
- edge-overlap mappings;
- source/provenance tables;
- initialization track metadata;
- global-to-subgraph state projections.

### 4.3. Coordinate semantics

State explicitly that all PHMM intervals are half-open `[left, right)`, and distinguish:

1. raw coordinate span;
2. static DP window;
3. edge transfer support;
4. pass-specific effective support.

This distinction is central and should appear early in the paper.

## 5. DAG-aware inference algorithms

This should be the technical center of the paper.

### 5.1. Forward recursion on DAGs

Planned points:

- process nodes in topological order;
- operate on packed node-state windows rather than dense rectangles;
- propagate messages along graph edges;
- combine incoming contributions at merge points with log-sum-exp;
- separate legality constraints from actual reachability.

### 5.2. Backward recursion on DAGs

Planned points:

- reverse-topological traversal;
- symmetric treatment of successor contributions;
- compatibility with the same packed support structures used by forward.

### 5.3. Posterior summaries

Planned points:

- posterior state occupancy;
- posterior transition usage;
- graph-aware aggregation needed for training losses and diagnostics.

### 5.4. Hard Viterbi

Planned points:

- replace log-sum-exp reduction with max reduction;
- keep sparse backpointers only for reachable cells;
- support traceback in packed/local/global state coordinates.

### 5.5. Soft Viterbi

Planned points:

- temperature-controlled interpolation between probabilistic and hard reductions;
- role in diagnostics, training surrogates, or annealed decoding.

## 6. Effective support, masks, and range control

This section should explain why static windows alone are insufficient.

### 6.1. Static legality versus dynamic reachability

Static windows define where computation is allowed. Effective-support masks define where computation is actually reachable in a given pass and subgraph.

### 6.2. Exact masks versus coarse spans

The manuscript should emphasize that branch/merge structure can create holes inside a node window, so exact boolean support masks are needed even when coarse contiguous spans are also cached for efficiency.

### 6.3. Why this matters

This is likely one of the most distinctive technical points of the paper because it separates DAG-aware inference from a naive rectangular or purely banded approximation.

## 7. Training objectives and optimization

This section should define what is optimized and how.

### 7.1. Parameterization

- transition logits;
- emission logits;
- initialization tracks;
- optional constraints or anchored parameters.

### 7.2. Objective components

Planned objective pieces:

- likelihood or negative log-likelihood;
- entropy-style objectives or penalties;
- scaled-SP style objectives or surrogates;
- regularizers on transitions/emissions/active states;
- optional curriculum or annealing terms.

### 7.3. Optimization strategy

Explain the staged path:

1. CPU reference path first;
2. validate numerical behavior and metrics;
3. lift the same recurrences into PyTorch tensors;
4. optimize wavefront execution later without changing the semantics.

## 8. Hard decode and alignment assembly

This section should connect state-path inference to final alignment objects.

### 8.1. From traceback to alignment columns

Planned points:

- map packed/local states back to global PHMM states;
- assemble sequence-wise traces into alignment columns;
- preserve insertion-slot structure;
- represent the result in a sparse alignment form suitable for metrics and export.

### 8.2. Hard metrics versus differentiable surrogates

Keep the distinction clear:

- hard decoded alignment metrics for reporting and comparison;
- differentiable surrogates for training.

## 9. Subgraph sampling and scalable training

This section should explain how the model is expected to scale beyond toy full-graph runs.

### 9.1. Why sampling is needed

Full-graph training can be too expensive for large datasets, especially when graph/state products become large.

### 9.2. Sampled TensorDag views

Describe the role of provenance-aware sampled subgraphs and optional state-mask clipping.

### 9.3. Global-state identity preservation

The manuscript should emphasize that local sampled views must preserve mapping to global PHMM state identities so losses and decoded results remain comparable across batches.

## 10. Implementation architecture

This section should describe the software stack briefly.

Suggested subsections:

- loader and validation layer;
- runtime graph/batch representation;
- reference CPU implementation;
- PyTorch execution path;
- trainer/evaluation modules;
- profiling hooks and diagnostics.

The implementation section should be concrete enough to support reproducibility, but should not turn into a software manual.

## 11. Experimental plan

The experimental section should be written around the model paper’s own claims rather than around graph-construction metrics.

### 11.1. Core experimental questions

1. Does the DAG-aware differentiable PHMM execute correctly end to end?
2. Do forward/backward, posterior, hard Viterbi, and soft Viterbi agree with the intended semantics on controlled DAG cases?
3. How much do effective-support masks and packed windows reduce work relative to naive dense layouts?
4. How do differentiable objectives and initialization tracks affect alignment quality and training behavior?
5. How well does sampled/subgraph training preserve useful global behavior?

### 11.2. Minimum result blocks for the paper

- correctness on tiny synthetic DAGs;
- end-to-end inference baseline on exported preprocessing artifacts;
- ablation of hard versus soft paths;
- ablation of support masks/windows;
- initialization comparison: `legacy_current` versus `reference_msa` when available;
- runtime and memory trends for CPU reference and later PyTorch execution.

### 11.3. Comparisons

The comparison strategy should be chosen carefully so the paper remains distinct from the manuscript under review. This paper’s natural baselines are:

- internal ablations;
- CPU reference versus tensorized implementation;
- alternative initialization tracks;
- objective/regularization ablations;
- possibly Baum-Welch-style or non-AD internal baselines when available.

## 12. Discussion

The discussion should stress that DAG-align-AD is not just “PHMM training after graph construction.” Its contribution is the explicit formulation of a differentiable probabilistic model whose native domain is a DAG with packed state support, merge-aware reductions, and subgraph-aware execution.

The paper should also discuss the engineering choice to separate:

- graph construction and persistence in DAG-rust / preprocessing;
- model training and decoding in DAG-align-AD.

That separation is part of the scientific contribution because it makes the probabilistic layer cleaner to analyze, optimize, and extend.

## 13. Immediate writing tasks

1. Turn the abstract scaffold into a first full abstract once the target result claims are fixed.
2. Write the formal notation section for graph nodes, state windows, and overlap mappings.
3. Draft the forward/backward and Viterbi sections with equations.
4. Decide how much of the training objective to present in the main text versus Methods.
5. Specify the minimum experimental story required for a first complete manuscript draft.
