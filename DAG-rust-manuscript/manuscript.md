# DAG-rust: A Safe and Scalable Rust Engine for Sequence DAG Construction and Merging

**Draft status:** initial manuscript draft for the DAG-rust paper  
**Scope:** graph construction, graph merging, persistence, determinism, memory efficiency, and safety  
**Out of scope for this paper:** AD-based PHMM training and alignment, which will be developed as a separate paper

---

## Abstract

Graph-based compression of large sequence collections can remove vast amounts of repeated computation, but practical use at ultralarge scale requires more than algorithmic novelty alone. The implementation must also be fast, memory efficient, deterministic, and robust enough to serve as a reusable systems foundation. We present **DAG-rust**, an idiomatic Rust re-engineering of the graph construction and merging core behind DAG-align. In contrast to the earlier Python-centered implementation, DAG-rust makes implementation safety, predictable resource usage, and graph-native persistence first-class design goals.

DAG-rust separates the graph layer from downstream probabilistic training, introduces typed data models for nodes, edges, provenance, and reference metadata, and adopts memory-conscious encodings for sequence fragments and provenance records. The system supports DNA, RNA, protein, and custom alphabets through explicit normalization and ambiguity policies rather than ad hoc string handling. It also exposes deterministic graph build and merge paths, configurable topology maintenance strategies, multiple provenance storage modes, and a versioned native storage format for graph exchange and downstream preprocessing.

This paper positions DAG-rust as a graph and systems upgrade of the original DAG-align work. The central claim is not a new end-to-end alignment pipeline, but a safer and more scalable graph engine that enables faster execution, smaller memory footprint, cleaner package boundaries, and more reliable downstream integration. We describe the design of DAG-rust, the algorithmic and implementation changes that distinguish it from the previous generation, and the evaluation framework used to quantify improvements in speed, memory use, determinism, and extensibility.

## 1. Introduction

The original DAG-align work introduced a graph-centric strategy for accurate multiple sequence alignment at ultralarge scale. Its core insight was that a directed acyclic graph can compress repeated fragments shared across many related sequences and thereby remove a large fraction of redundant computation that burdens conventional alignment pipelines. That work established the conceptual value of the FTO-DAG representation and its downstream use in probabilistic alignment. However, as the project matured, it became clear that the next bottleneck was not only algorithmic. It was also architectural.

The earlier implementation was centered on a Python codebase that mixed graph construction, graph manipulation, persistence, and downstream alignment concerns. That design was effective for rapid research iteration, but it becomes increasingly costly as workloads, graph sizes, and supported data types grow. For ultralarge sequence collections, graph construction and merging must operate under strict memory pressure, avoid accidental nondeterminism, and provide stronger correctness guarantees than dynamic object-heavy implementations typically offer. In addition, future work on differentiable PHMM training benefits from a clean boundary in which graph construction and persistence are handled independently of model training.

These pressures motivate **DAG-rust**, an idiomatic Rust implementation of the graph core. The goal of DAG-rust is to upgrade the graph layer of DAG-align along three main axes:

1. **Faster execution**, by reducing dynamic-language overhead, improving locality, and exposing data structures that better match the graph operations used during construction and merging.
2. **Smaller memory footprint**, by replacing object-heavy representations with compact typed layouts, bit-packed fragment encodings, and configurable provenance storage modes.
3. **Better safety**, by relying on Rust's ownership and type system, explicit validation, versioned storage, and deterministic defaults to rule out broad classes of implementation errors.

This paper therefore has a narrower and cleaner scope than the original DAG-align manuscript. It focuses on the graph engine itself: how sequence fragments are encoded, how nodes and edges are represented, how incremental sequence integration and graph merging are organized, how provenance is stored, how graphs are serialized, and how these choices improve speed, memory behavior, and implementation safety. We intentionally leave AD-based PHMM training and alignment to a separate follow-up paper so that this manuscript can be judged as a graph, algorithms, and systems contribution in its own right.

### 1.1. Paper positioning relative to the earlier DAG-align manuscript

This manuscript should be read as a **graph-engine upgrade paper**, not as a replacement version of the original end-to-end DAG-align paper. The earlier paper introduced the broader conceptual pipeline: FTO-DAG construction, tiled PHMM training, DAG-aware decoding, and large-scale alignment benchmarks. The present paper isolates one layer of that stack and rebuilds it as a standalone software and algorithmic system.

That distinction matters for both novelty and evaluation. The main question here is not whether graph-aware alignment is useful in principle; the earlier work already answered that. The question here is whether the graph layer can be redesigned so that it is substantially more efficient, safer, more modular, and easier to extend to broader biological alphabets and downstream workflows. DAG-rust is our answer to that question.

## 2. Contributions

The main contributions of DAG-rust are as follows.

1. **A graph-native Rust engine for sequence DAG construction and merging.** DAG-rust isolates the graph layer from the legacy Python alignment code and from future AD-based training code, giving the project a cleaner systems boundary.
2. **Memory-conscious representations for fragments, edges, and provenance.** The implementation uses packed fragment windows, typed node and edge records, and multiple provenance storage strategies to reduce memory overhead while preserving required traceability.
3. **Explicit support for DNA, RNA, protein, and custom alphabets.** Alphabet handling is elevated into a typed API with explicit normalization and ambiguity policies, rather than being treated as a special case of string processing.
4. **Deterministic defaults for build and merge behavior.** DAG-rust treats reproducibility as a design goal rather than an afterthought, while leaving room for similarity-aware ordering policies when they are justified by evaluation.
5. **Native graph persistence with versioned storage.** DAG-rust introduces a dedicated binary storage format for graphs and associated provenance metadata, enabling reliable interchange with downstream preprocessing and training components.
6. **A cleaner foundation for future graph-to-model workflows.** Reference paths, export profiles, coordinate-related metadata, and graph snapshots are structured explicitly so the graph engine can serve as a stable producer for later PHMM-oriented tooling.

## 3. Design goals

The design of DAG-rust is guided by several engineering principles that directly reflect the limitations of the previous generation.

### 3.1. Safety by construction

In a graph engine that will eventually process millions of sequences and produce large persistent artifacts, implementation safety is not a secondary concern. DAG-rust therefore replaces many implicit assumptions with typed interfaces: node identifiers, sequence identifiers, provenance positions, topological coordinates, graph IDs, and weights are represented as dedicated types rather than interchangeable raw integers. Similarly, graph storage is versioned explicitly, and invalid configurations such as mismatched fragment lengths or incompatible provenance strategies are surfaced as typed errors rather than latent runtime corruption.

Rust does not guarantee algorithmic correctness by itself, but it does let us remove a large class of memory-management and aliasing failures from the implementation space. For this project, that matters because graph construction and merging involve many interdependent updates to nodes, edges, indices, provenance records, and topological metadata. A language-level safety baseline makes those updates easier to reason about and easier to validate.

### 3.2. Memory efficiency as a first-class requirement

The graph layer is intended for datasets whose scale makes even moderate constant factors important. DAG-rust therefore treats memory use as a design constraint from the start. Fragment keys can be represented in multiple forms, including packed inline windows when symbol width and fragment length allow it. Provenance can likewise be stored as full records, compact packed records, sequence trace paths, or count-only summaries depending on the downstream need. These choices let the implementation trade off fidelity, traceability, and footprint explicitly rather than paying a uniform cost for all workflows.

### 3.3. Determinism and reproducibility

The project treats deterministic behavior as the default for both graph building and graph merging. Deterministic binary ordering is the baseline merge policy, while similarity-aware ordering remains an explicit policy choice rather than an uncontrolled side effect of container iteration or ingestion order. This is important for scientific reproducibility, for debugging large integrations, and for making benchmark comparisons interpretable.

### 3.4. Clean separation of graph and model layers

The graph layer should not depend on the details of downstream PHMM training code. DAG-rust therefore exposes graph-only and graph-with-reference export profiles, while later preprocessing and training stages can consume those outputs without entangling their internal logic with graph construction internals. This separation is central to the larger reorganization of the project into DAG-rust, Pre-AD-prep, and AD-PHMM-align.

## 4. System overview

At a high level, DAG-rust takes encoded biological sequences, converts them into fragment occurrences under an explicit alphabet and fragment encoder, integrates those occurrences into an FTO-DAG through incremental construction or graph merging, maintains graph and provenance indices, validates structural invariants, and persists the result in a native versioned format.

The implementation is organized into several conceptual layers:

- **Sequence model:** alphabets, normalization policies, ambiguity handling, encoded sequences, and fragment-window generation.
- **Graph model:** typed nodes and edges, node kinds, edge indices, fragment indices, provenance tables, endpoint bookkeeping, and reference-related metadata.
- **Algorithms:** incremental graph construction, similarity gating, topology maintenance strategies, merge planning and replay, scheduling, and graph postprocessing.
- **Persistence and interfaces:** native binary storage, export profiles, CLI entry points, and graph snapshots.

This modularization is important because it lets us discuss and evaluate graph construction independently from downstream model training. It also makes it possible to benchmark individual layers rather than only measuring an opaque end-to-end runtime.

## 5. Data model and implementation details

### 5.1. Typed biological alphabets

DAG-rust treats alphabet support as a core abstraction rather than a thin parser convenience. The implementation currently exposes built-in alphabets for DNA, RNA, and proteins, as well as a path for custom alphabets. Each alphabet carries an explicit normalization policy and ambiguity policy. For example, RNA-to-DNA and DNA-to-RNA normalization can be requested explicitly, case folding is not conflated with biological normalization, and ambiguity handling can either preserve exact ambiguous symbols or reject them.

This design serves two purposes. First, it broadens the scope of the graph engine beyond the viral DNA/RNA workloads emphasized in the earlier DAG-align work. Second, it prevents alphabet semantics from leaking implicitly into fragment construction and merge logic. That is important for a reusable graph engine, especially if the eventual downstream tasks span nucleotide and protein workflows.

### 5.2. Memory-conscious fragment encodings

The fragment representation is one of the most important places to reduce overhead. DAG-rust therefore introduces a `FragmentKey` abstraction with multiple concrete layouts. When fragment length and bits-per-symbol permit, a window can be packed inline into a compact integer representation. Longer or less compact cases can fall back to wider packed words or explicit symbol vectors. The default fragment encoder can exploit packed inline windows directly during sliding-window generation, which reduces allocation and avoids repeatedly materializing short vectors for the most common cases.

This is a concrete example of the design philosophy behind DAG-rust. The point is not merely to translate Python code into Rust syntax. The point is to redesign the data path so that frequent graph operations such as fragment lookup, anchor search, and node insertion work with compact, typed representations that match the structure of the computation.

### 5.3. Graph records, indices, and update strategies

The core graph type stores typed nodes and weighted directed edges. Nodes carry a fragment key, node kind, weight, flags, and a provenance range. Edges carry typed parent and child identifiers plus weights. On top of this base representation, DAG-rust maintains indices tailored to the operations required during construction and merging.

One important example is edge indexing. DAG-rust currently supports a global hash-based index as well as a low-degree hybrid strategy that keeps small adjacency lists inline and spills higher-degree cases into overflow storage. This reflects a common pattern in sequence DAGs: many nodes have modest degree, and paying the full cost of a large generic hash structure for every adjacency can be wasteful. The hybrid index gives the system a path toward better locality and smaller footprint on low-degree regions without giving up a general fallback for more complex neighborhoods.

### 5.4. Provenance storage modes

Not all downstream workflows need the same provenance fidelity. DAG-rust therefore supports multiple provenance storage strategies, including full records, packed 32-bit records, explicit sequence trace paths, and count-only storage. This is both a memory optimization and a separation-of-concerns decision. Some workflows need exact source-position traceability, while others only need a compressed summary sufficient for graph statistics or downstream export. Making provenance strategy explicit lets the engine fit different workloads rather than forcing all users into a single high-cost representation.

### 5.5. Native graph persistence

The earlier project relied heavily on Python-centric object formats and ad hoc graph materialization. DAG-rust replaces that style with a native binary graph format carrying a magic header, explicit format version, graph dimensions, storage strategy metadata, node and edge records, optional sequence linkage, and provenance snapshots. This approach has several advantages: it makes graph I/O more predictable, reduces dependence on language-specific serialization behavior, and provides a more stable interchange layer for later preprocessing and training stages.

For a project that is being split across Rust graph construction, preprocessing, and differentiable model training, this persistence layer is not a convenience feature. It is part of the core architecture.

## 6. Construction and merging algorithms

### 6.1. Incremental sequence integration

The build path in DAG-rust is centered on incremental integration of encoded sequences into an FTO-DAG. The construction configuration includes fragment length, ordering policy, optional similarity thresholding, rejection policy, topology update strategy, provenance storage strategy, and edge index strategy. This makes the build path configurable in ways that directly affect speed, memory use, and final graph structure.

A useful change relative to a monolithic implementation is that topology maintenance is itself made explicit. DAG-rust distinguishes several topology update strategies, including full rebuilds and increasingly local incremental approaches. This is important because the cost of maintaining acyclicity and topological consistency can dominate graph construction if it is handled too conservatively. Exposing the strategy at the configuration level lets us evaluate trade-offs rather than burying them inside a single hard-coded implementation.

### 6.2. Similarity gating and controlled rejection

Large-scale graph construction benefits from a lightweight early decision on whether a sequence should be integrated immediately, deferred, or skipped under the current configuration. DAG-rust formalizes this through similarity thresholds and rejection policies. This is especially relevant for datasets that contain a dominant cluster of closely related sequences together with more divergent outliers. By making the initial match criterion explicit, the system avoids conflating graph growth with graph filtering and makes benchmark behavior easier to interpret.

### 6.3. Deterministic graph merging

Merging is treated as a first-class graph operation rather than a one-off utility. DAG-rust exposes merge policies and merge configurations explicitly, with deterministic binary ordering as the default behavior. In the current design, trace-path-based graphs can be replayed into a base graph under controlled merge logic, while more aggressive or lossy graph-only merge modes can be introduced with their own semantics and validation criteria.

This explicitness is important scientifically. Merge order can influence intermediate graph structure, compression behavior, and even downstream evaluation. A deterministic default therefore improves reproducibility, while similarity-aware scheduling can be studied as a deliberate optimization rather than an undocumented implementation artifact.

## 7. Evaluation plan

The evaluation section of the final paper should be organized around the central claim of this manuscript: **DAG-rust is a graph-engine upgrade that delivers faster speed, smaller memory footprint, and better safety than the previous generation.**

Accordingly, the main evaluation axes should be:

1. **Speed:** graph construction time, merge time, serialization time, and selected scaling curves.
2. **Memory footprint:** peak resident memory during construction and merging, storage size on disk, and the effect of different provenance strategies.
3. **Safety and robustness:** deterministic repeatability, validation failures caught explicitly, storage round-trip integrity, and failure behavior on invalid inputs.
4. **Generality:** successful operation on DNA, RNA, and protein alphabets with the same graph core.

The final benchmark suite should include:

- comparisons against the legacy Python graph core where possible;
- sensitivity analyses over fragment length, provenance mode, and topology update strategy;
- build and merge measurements on representative SARS-CoV-2 subsets and at least one non-DNA workload;
- persistence and reload benchmarks for native graph storage;
- ablations of edge index strategy and packed versus unpacked fragment handling.

**[TODO: insert finalized benchmark tables and figures once the measurement pipeline is frozen.]**

## 8. Discussion

DAG-rust is best understood as a systems and architecture paper built on top of the earlier DAG-align concept. Its importance lies not only in making the existing graph workflow faster, but also in making the graph layer a durable piece of infrastructure. A safer and more modular graph engine reduces the cost of future algorithmic work, makes downstream preprocessing cleaner, and gives later PHMM-oriented work a stronger and more stable foundation.

The broader implication is that large-scale biological graph software cannot remain a collection of loosely connected research scripts if it is expected to support ultralarge datasets, wider alphabet support, and reproducible downstream modeling. The graph core must itself be designed as a rigorous software system. DAG-rust is a step in that direction.

At the same time, this paper should remain disciplined in scope. Its goal is not to claim completion of the entire future DAG-align stack. Instead, it argues that the graph layer is now important enough to deserve a standalone treatment, both because it introduces substantial systems advances and because it changes the practical ceiling of what the broader pipeline can support.

## 9. Draft outline for the remainder of the paper

The full manuscript can be expanded from this draft using the following structure.

1. **Abstract**
2. **Introduction**
3. **Contributions and positioning**
4. **System overview**
5. **Data model and implementation details**
6. **Construction and merging algorithms**
7. **Experimental setup**
8. **Results**
9. **Discussion**
10. **Methods**

## 10. Immediate writing tasks

- tighten the abstract once benchmark numbers are available;
- add a dedicated related-work section contrasting DAG-rust with POA-style graph aligners, pangenome graph systems, and language-level graph toolchains;
- write the experimental setup around the legacy Python comparison and the DAG-rust ablations;
- decide whether the final narrative should emphasize "engine", "graph core", or "construction and merging system" in the introduction and results;
- insert concrete benchmark numbers for speed, memory, and storage once measurements are finalized.
