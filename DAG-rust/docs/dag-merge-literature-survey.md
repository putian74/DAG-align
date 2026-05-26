# DAG merge literature survey

This note captures a short literature survey for improving `DAG-rust` merge
performance. The current exact TracePaths merge remains path-replay heavy: it
iterates add-graph trace paths, attempts exact/long-path integration, and falls
back to occurrence-summary replay when exact reuse misses.

## Current code context

- `src/algorithms/merge.rs` drives TracePaths merge through
  `replay_trace_path_graphs`.
- `src/algorithms/build.rs` uses anchor-candidate dynamic programming for exact
  path integration.
- `tests/dataset_smoke_tests.rs` contains the main local exact merge benchmark
  for `1000 + 1000 -> 2000` SARS-CoV-2 TracePaths graphs.

That makes the main optimization target clear: reduce the amount of replayed
path work before trying to tune local path heuristics further.

## High-value directions

### 1. Transitive reduction

**Idea:** remove edges already implied by other directed paths while preserving
reachability.

**Why it may help here:** if merge cost is dominated by replaying paths through
shortcut edges, a reachability-equivalent but sparser DAG should cut candidate
sets, anchor ambiguity, and replay volume before merge starts.

**Assessment:** high-priority exact pre-pass, assuming merge semantics depend on
reachability and support rather than explicit retention of every transitive edge.

**References**

- A. V. Aho, M. R. Garey, J. D. Ullman, *The Transitive Reduction of a Directed
  Graph*, SIAM Journal on Computing, 1972.
  <https://doi.org/10.1137/0201008>
- E. W. Myers, *The fragment assembly string graph*, Bioinformatics, 2005.
  <https://doi.org/10.1093/bioinformatics/bti1114>

### 2. Structural hashing / hash-consing

**Idea:** canonicalize nodes or local subgraphs by structural descriptors such
as `(fragment, kind, normalized predecessors/successors, local attrs)` and
intern identical structures once.

**Why it may help here:** exact merge currently rediscovers identical structure
through replay. Structural hashing can turn repeated equal subgraphs into direct
lookups instead of repeated path-by-path reconstruction.

**Assessment:** highest-priority engineering direction. It aligns naturally with
the existing exact merge goal and should combine well with current fragment-based
matching.

**References**

- R. E. Bryant, *Graph-Based Algorithms for Boolean Function Manipulation*,
  IEEE Transactions on Computers, 1986.
  <https://doi.org/10.1109/TC.1986.1676819>
- J.-C. Filliâtre, S. Conchon, *Type-safe modular hash-consing*, 2006.
  <https://doi.org/10.1145/1159876.1159880>
- N.-F. Zhou, C. T. Have, *Efficient tabling of structured data with enhanced
  hash-consing*, Theory and Practice of Logic Programming, 2012.
  <https://doi.org/10.1017/S1471068412000178>

### 3. Bisimulation / partition-refinement quotienting

**Idea:** collapse behaviorally equivalent nodes into quotient classes before
exact merge.

**Why it may help here:** repeated suffix/fanout structure may be compressible
without losing exact merge semantics if equivalence classes are defined
carefully.

**Assessment:** promising after structural hashing; likely best as a reduction
step rather than as the first optimization.

**References**

- R. Paige, R. E. Tarjan, *Three Partition Refinement Algorithms*, SIAM Journal
  on Computing, 1987. <https://doi.org/10.1137/0216062>
- A. Dovier, C. Piazza, A. Policriti, *An efficient algorithm for computing
  bisimulation equivalence*, Theoretical Computer Science, 2004.
  <https://doi.org/10.1016/S0304-3975(03)00361-X>

### 4. Reachability / ancestry indexing

**Idea:** precompute labels or summaries that reject impossible node matches
cheaply.

**Why it may help here:** current exact matching spends real work after fragment
matches already exist. Reachability-aware blockers can reduce candidate sets
before anchor DP and replay.

**Assessment:** strong supporting optimization, but not a replacement for replay.

**References**

- E. Cohen, E. Halperin, H. Kaplan, U. Zwick, *Reachability and Distance Queries
  via 2-Hop Labels*, SIAM Journal on Computing, 2003.
  <https://doi.org/10.1137/S0097539702403098>
- R. Bramandia, B. Choi, W. K. Ng, *Incremental Maintenance of 2-Hop Labeling of
  Large Graphs*, IEEE TKDE, 2010. <https://doi.org/10.1109/TKDE.2009.117>

### 5. Direct DAG dynamic programming instead of replay

**Idea:** solve matching or merge subproblems on graph states directly instead
of replaying linearized paths.

**Why it may help here:** the literature on partial-order alignment and
sequence-to-graph alignment repeatedly avoids path materialization because it
blows up on rich graph structure.

**Assessment:** likely the best long-term reformulation, but a larger change
than the first branch goals.

**References**

- C. Lee, C. Grasso, M. F. Sharlow, *Multiple sequence alignment using partial
  order graphs*, Bioinformatics, 2002.
  <https://doi.org/10.1093/bioinformatics/18.3.452>
- C. Grasso, C. Lee, *Combining partial order alignment and progressive multiple
  sequence alignment increases alignment speed and scalability to very large
  alignment problems*, Bioinformatics, 2004.
  <https://doi.org/10.1093/bioinformatics/bth126>
- M. Rautiainen, T. Marschall, *Aligning sequences to general graphs in
  O(V + mE) time*, 2017. <https://doi.org/10.1101/216127>
- M. Rautiainen, T. Marschall, *GraphAligner: rapid and versatile
  sequence-to-graph alignment*, Genome Biology, 2020.
  <https://doi.org/10.1186/s13059-020-02157-2>

## Directions that look less attractive for the first pass

- Whole-graph canonical labeling as a default merge engine: useful on small
  ambiguous regions, but too expensive as the main strategy.
- Generic subgraph isomorphism: exact but search-heavy.
- Tree-edit-distance style approaches: good for tree-like regions, poor fit for
  heavily shared DAG structure.

## Recommended branch order

The first branch should focus on the two lowest-risk, highest-return changes:

1. **Transitive reduction pre-pass**
   - define the exact invariants the reduced graph must preserve for merge;
   - measure edge-count and replay-count reduction on the local benchmark;
   - keep the transform independently switchable for validation.
2. **Structural hashing / exact subgraph interning**
   - define canonical keys for merge-relevant node structure;
   - cache and reuse exact equal subgraphs across base/add graphs;
   - use it before fallback replay rather than after replay already started.

## Immediate implementation sketch

### Transitive reduction

- start with a safe DAG-only reduction pass after topology is available;
- prefer exact removal of only provably transitive edges;
- record per-graph stats: edges examined, edges removed, node count unchanged,
  benchmark merge time change.

### Structural hashing

- start with a conservative key built from stable node identity fields and
  normalized adjacent structure;
- make hashing exact only for cases where all merge-relevant structure is known;
- integrate it as a fast path before occurrence-summary replay.

## Practical expectation

The literature suggests the first big wins will come from **shrinking the graph
and reusing repeated structure before replay**, not from refining corridor/path
heuristics alone. In this codebase, that means:

1. make the graph sparser without changing exact semantics;
2. detect structurally identical regions early;
3. leave graph-DP reformulation for a later, larger merge redesign.
