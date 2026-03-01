# PCT Top-Down Partitioning

This document describes the top-down partitioning algorithm that builds the initial structure of a Principal Component Tree (PCT). Partitioning discovers hierarchical subspace structure by recursively splitting data and deciding where branches should diverge.

**Prerequisites**: This document assumes familiarity with [Principal Component Trees](principal-component-trees.md).

**What comes next**: After partitioning, the tree undergoes EM refinement to optimize principal vectors and branch depths. See [PCT EM Training](pct-em-training.md).

The partitioning algorithm is implemented in `training.py`, with the split-or-extend decision driven by `subspace_sifting.py` (see [Subspace Sifting](pct-subspace-sifting.md)).

## 1. Overview: What Partitioning Accomplishes

Top-down partitioning answers two questions simultaneously:

1. **How many branches?** Where should the tree split to capture distinct subspace clusters?
2. **How much sharing?** How many principal directions should branches share before diverging?

The algorithm starts with all data and the full SVD, then iteratively expands leaf nodes—either **extending** (adding a single child to capture the next principal direction) or **splitting** (branching into multiple children via subspace clustering). The result is a tree structure where:

- Shared ancestors capture directions common to multiple data subpopulations
- Branches specialize to distinct subspace clusters
- The tree depth at each branch reflects how much variance that subpopulation contains

A key insight: **we work in progressively smaller subspaces as we descend the tree**. Each partition's SVD is truncated to capture a target percentage of that partition's variance (controlled by `variance_capture_percent`). This means deep nodes might operate in 50 dimensions even if the original data lives in 4096 dimensions—a critical runtime optimization.

## 2. The Core Loop

The `PCTreeTrainer_Partitioning` class orchestrates the expansion process.

### Initialization

```python
def fit(self, verbose=0):
    # Full-data SVD, truncated to variance_capture_percent
    U, s, VT = my_full_svd(self.X)
    effective_rank = rank_to_explain_variance_given_s(s, self.options.variance_capture_percent)
    U, s, VT = as_low_dimentions(U, s, VT, effective_rank)

    # Single root node containing all data
    initial_root = PCTreeNode_Training(self, None, self.X, self.options,
                                        np.ones(self.n_data, dtype=bool), U, s, VT)
    self.leaves = [initial_root]
```

The root node holds the truncated SVD of all data. Its first principal vector becomes the tree's root.

### The Expansion Loop

```python
while still_training_with_leaves:
    still_training_with_leaves = self.expand_next()
```

Each iteration of `expand_next()`:

1. **Selects** which leaf to expand (via greedy budgeting)
2. **Expands** that leaf (extend or split)
3. **Updates** the leaf set with new nodes
4. **Checks** stopping conditions (`max_nodes`, `max_width`)

### Greedy Budget-Based Expansion

Rather than BFS or DFS, the algorithm uses `PCTreeTrainingBudgeter` to select the most valuable leaf to expand:

```python
def next_node_to_expand(self, remaining_budget: int) -> tuple[PCTreeNode_Training, int]:
    next_node, next_node_budget = self.options.budgeter.next_node(self.leaves, remaining_budget)
    return next_node, next_node_budget
```

The budgeter builds a temporary tree from all leaves' remaining singular values and applies `EfficientEncodingPruner` to identify which leaf would contribute most to compression quality if expanded. This greedy approach prioritizes leaves with high singular values and substantial data—the nodes where additional principal components would capture the most variance.

**Runtime optimization note**: The budgeter constructs a simplified "leaf tree" that ignores the actual tree structure, treating each leaf's remaining dimensions as an independent chain. This avoids recomputing the full tree's efficient encoding at each step.

## 3. The Extend vs Split Decision

At each leaf, the algorithm must decide: should this node **extend** (add a single child capturing the next principal direction) or **split** (branch into multiple children for distinct subspace clusters)?

This decision is driven by `sift_subspace()`, which detects clustering structure in the residual data. See [Subspace Sifting](subspace_sifting.md) for the full algorithm.

### The Sifting Test

```python
test_result = sift_subspace(self.U[:, :used_test_dim], self.s[:used_test_dim],
                            self.opt.subspace_sifter_options)
clustering_detected = test_result.primary_significant
```

The test returns two key outputs:

1. **`primary_significant`**: Is there detectable clustering structure? (Based on both p-value and Cohen's d effect size thresholds)
2. **`intersecting_dimensions`**: How many leading principal directions are shared across clusters before they diverge?

### The Decision Matrix

| Clustering Detected? | Intersection Dimensions | Action |
|---------------------|------------------------|--------|
| No | — | Extend (single child) |
| Yes | 0 | Split immediately |
| Yes | N > 0 | Set fuse to N-1, extend, then split when fuse expires |

The third case is crucial: when clusters share leading directions, we must capture that shared structure before splitting. This is handled by the **fuse mechanism**.

### Bypass Conditions

Several conditions cause the algorithm to extend without running the sifting test:

```python
other_bypass = (reached_max_siblings or
                reached_max_width or
                not splittable_height or
                not under_min_chord_length or
                compr_fuse_running)
```

- **`reached_max_siblings`**: This node's parent already has `max_children` children
- **`reached_max_width`**: The tree already has `max_width` leaves
- **`compr_fuse_running`**: A fuse is counting down (split already scheduled)

When bypassed, the node extends with a reason logged (e.g., "MaxSiblings", "MaxWidthReached").

## 4. The Fuse Mechanism: Capturing Shared Structure

When sifting detects clustering with nonzero intersection dimensions, the tree shouldn't split immediately. It must first capture the shared subspace—the principal directions common to all clusters.

### Why Intersections Matter

Consider word embeddings for "north", "south", "red", "blue". Directional words cluster together; color words cluster together. But both clusters share structure related to "being an adjective" or "being a concrete concept." A PCT should capture this shared structure in ancestor nodes before the branches diverge.

Shared ancestor nodes represent common structure across subpopulations, enabling more parameter-efficient representations in downstream applications.

### The Fuse Lifecycle

The `compr_test_fuse` field acts as a countdown timer:

```
Step 1: Sifting detects clustering with intersecting_dimensions = 3
        Node sets compr_test_fuse = 2 (one less than intersection count)

Step 2: Next expansion - fuse is 2 > 0
        Node EXTENDS (adds single child), decrements fuse to 1
        No sifting test run

Step 3: Next expansion - fuse is 1 > 0
        Node EXTENDS again, decrements fuse to 0

Step 4: Next expansion - fuse is 0 (expired)
        Node SPLITS via cluster_expand()
```

**Concrete example**: Suppose we're processing LLM token embeddings and sifting returns `intersecting_dimensions = 3`. This means the first 3 principal directions are shared across the detected clusters—perhaps capturing general "token-ness" before specializing to nouns vs. verbs. The fuse ensures we add 2 more shared nodes (the current node plus 2 children) before splitting into specialized branches.

### Connection to `sifting_effect_size`

The `sifting_effect_size` hyperparameter controls how much structure reduction (measured by Cohen's d) must occur before sifting declares "the shared structure has been captured, time to split."

- **Higher values** (e.g., 0.8): Require more dramatic structure loss → detect more intersection dimensions → more shared nodes before splitting → deeper, narrower trees
- **Lower values** (e.g., 0.2): Trigger splits with less structure loss → fewer intersection dimensions → split sooner → wider, shallower trees

This is the most important hyperparameter for controlling tree shape organically, based on data geometry rather than hard limits.

### Fuse in Code

```python
# When fuse is running, skip sifting and extend
compr_fuse_running = (self.compr_test_fuse is not None) and (self.compr_test_fuse > 0)
next_compr_fuse = None if not compr_fuse_running else self.compr_test_fuse - 1

if next_compr_fuse is not None:
    # Extend without sifting test
    U_next, s_next, VT_next = self.U[:, 1:], self.s[1:], self.VT[1:]
    next_node = PCTreeNode_Training(..., compr_test_fuse=next_compr_fuse)
    self.children = [next_node]
    return PCTNode_ExpansionResult(False, [next_node],
                                    no_split_reason=f"BreatheDeeper: Fuse={next_compr_fuse}")

# When fuse expires, force split
if compr_fuse_expired and not other_bypass:
    result = self.cluster_expand(budget)
    return result
```

## 5. The Split Operation

When a node splits, `cluster_expand()` partitions the data into two clusters and creates child nodes for each.

### The Clustering Step

```python
def cluster_expand(self, budget: int) -> PCTNode_ExpansionResult:
    # Determine clustering dimension based on intrinsic dimensionality
    max_dim_of_ss_clusters = min(self.abid_dim, budget, MAX_CLUSTER_DIM)
    clustering_dim = min(self.rank, 2 * self.abid_dim, MAX_CLUSTER_DIM)

    # Project data to clustering dimension
    X_cluster = self.U[:, :clustering_dim] * self.s[:clustering_dim]

    # Binary subspace clustering
    kss_initial = BinaryBuckshotSubspaceClusterer(max_dim_of_ss_clusters, 20)
    kss_initial.fit(X_cluster)
    ass_split = kss_initial.predict(X_cluster)
```

The `BinaryBuckshotSubspaceClusterer` finds a binary partition where each cluster lies near a low-dimensional subspace. The specific algorithm choice is secondary—any subspace clustering method could substitute here.

### Intrinsic Dimensionality Estimation

The `abid_dim` field estimates the data's intrinsic dimensionality using ABID (Angle-Based Intrinsic Dimensionality):

```python
def _calculate_abid_dim(self, X_og: NDArray) -> float:
    if self.rank == 1:
        return 1
    n_points_for_abid = min(self.N_data, min(500, 2 * self.rank))
    return abid_dim_estimation(X_og[random_mask(self.N_data, n_points_for_abid)])
```

This estimate guides how many dimensions to use for clustering—enough to capture cluster structure without overfitting to noise.

### Per-Cluster SVD

After clustering, each partition needs its own SVD:

```python
unnormalized_U = self.U * self.s
UX0 = unnormalized_U[ass_split == 0]
UX1 = unnormalized_U[ass_split == 1]

# Efficient split SVD using covariance subtraction
full_U_cov = np.diag(self.s ** 2)
(U0, s0, VT0), (U1, s1, VT1) = efficient_split_svd(full_U_cov, UX0, UX1)
```

**Runtime optimization**: `efficient_split_svd` exploits the fact that we already have the combined covariance. Rather than computing each cluster's covariance from scratch, it computes the smaller cluster's covariance and subtracts from the total. This halves the covariance computation cost.

### Projection to Ambient Space

The per-cluster SVDs are computed in the reduced dimension. We project back to ambient space:

```python
s0a, VT0_ambient = project_vt_up(self.VT, VT0, s0)
s1a, VT1_ambient = project_vt_up(self.VT, VT1, s1)
```

### Variance Truncation

Each child's SVD is truncated to `variance_capture_percent`:

```python
effective_rank_0 = rank_to_explain_variance_given_s(s0a, self.opt.variance_capture_percent)
U0b, s0b, VT0_ambient = as_low_dimentions(U0, s0a, VT0, effective_rank_0)
```

This is where the "shrinking subspace" principle manifests: each partition keeps only the dimensions needed to explain (e.g.) 99.9% of its variance, which may be far fewer than the parent had.

### Creating Child Nodes

```python
mask0 = apply_sequential_masks(self.data_mask, ass_split == 0)
mask1 = apply_sequential_masks(self.data_mask, ass_split == 1)

splits = [
    PCTreeNode_Training(self.ctx, self.parent, self.X_og, self.opt,
                        mask0, U0b, s0c, VT0_ambient),
    PCTreeNode_Training(self.ctx, self.parent, self.X_og, self.opt,
                        mask1, U1b, s1c, VT1_ambient)
]
```

Note that splits replace the current node—they become children of the current node's **parent**, not of the current node itself. The current node is removed from the tree.

## 6. Working in Shrinking Subspaces

A critical design principle: **each partition operates in the smallest subspace that captures its variance**.

### The `variance_capture_percent` Hyperparameter

This parameter (typically 0.999 or 99.9%) controls dimension truncation at every level:

```python
effective_rank = rank_to_explain_variance_given_s(s, self.options.variance_capture_percent)
U, s, VT = as_low_dimentions(U, s, VT, effective_rank)
```

### Why This Matters

Consider training on 4096-dimensional LLM embeddings with `max_nodes=10000`:

1. The root SVD might truncate to rank 800 (capturing 99.9% of total variance)
2. After a split, cluster A might need only 400 dimensions; cluster B only 350
3. Deep in the tree, a specialized cluster might need only 50 dimensions

Without truncation, every SVD operates in 4096D. With truncation, most operate in far smaller spaces.

**Runtime impact**: The algorithm performs hundreds of SVDs during partitioning. SVD cost scales with dimension. Reducing working dimension from 4096 to 400 provides roughly 10× speedup per SVD—compounding to orders of magnitude overall.

**Alignment with subspace clustering**: The goal of subspace clustering is to find partitions where each lies near a low-dimensional subspace. Truncating to `variance_capture_percent` operationalizes this: if a partition truly lies near a k-dimensional subspace, truncation naturally discovers this.

### The `rank_to_explain_variance_given_s` Utility

```python
def rank_to_explain_variance_given_s(s: NDArray, target_variance: float) -> int:
    cumulative_variance = np.cumsum(s ** 2) / np.sum(s ** 2)
    return np.searchsorted(cumulative_variance, target_variance) + 1
```

Given singular values, returns the rank needed to capture the target percentage of total variance.

## 7. Controlling Tree Shape

Several hyperparameters control tree structure. Understanding their interactions is essential.

### Primary Parameters

**`max_nodes`**: The total node budget. This is the primary stopping criterion—expansion continues until the tree has this many nodes (or until other constraints prevent further growth).

**`max_width`**: Maximum number of leaves (branches). This is critical for several reasons:

- **Downstream applications**: Many applications need a manageable number of branches for routing or assignment
- **Runtime optimization**: Once `max_width` is reached, no more splits occur; the algorithm rapidly extends remaining leaves and proceeds to EM

**Typical behavior**: The partitioning algorithm tends to split early and often, producing wide, shallow trees. With LLM token embeddings, a `max_width` of 100 might be reached after only ~400 nodes when targeting 10,000. The remaining ~9,600 nodes come from extending these 100 branches deeper.

```
Without max_width constraint:     With max_width=100:
        ___________                      _______
       /  /  \  \  \                    /   |   \
      /  |    |  |  \                  /    |    \
     ... (200+ branches)              ...  (100 branches,
                                           each extended deep)
```

**`max_children`**: Maximum children per split. Limits local branching factor. With binary subspace clustering, this is effectively 2, but the constraint prevents pathological cases.

### Organic Control via `sifting_effect_size`

While `max_width` and `max_children` are hard constraints, `sifting_effect_size` provides organic, data-driven control:

| Value | Behavior | Result |
|-------|----------|--------|
| Low (0.2) | Split readily, few shared nodes | Wide, shallow trees |
| Medium (0.5) | Balanced | Moderate width and depth |
| High (0.8) | Require strong evidence to split | Narrow, deep trees with more sharing |

This parameter shapes the tree based on actual data geometry. If clusters genuinely share structure, high `sifting_effect_size` captures it; if they don't, even low values won't force artificial sharing.

### Parameter Interaction Example

Consider targeting a 10,000-node PCT on LLM embeddings:

```
max_nodes = 10000
max_width = 100
sifting_effect_size = 0.5
variance_capture_percent = 0.999
```

Typical progression:
1. First ~400 nodes: Splits dominate, tree reaches 100 branches
2. Remaining ~9,600 nodes: Extensions only (max_width reached)
3. Result: 100 branches averaging 100 nodes each

Changing `sifting_effect_size` to 0.8:
1. First ~600 nodes: Fewer splits due to higher intersection detection, tree reaches 50 branches
2. Remaining ~9,400 nodes: Extensions only
3. Result: 50 branches averaging 200 nodes each, more shared structure

## 8. The Training Node: `PCTreeNode_Training`

Each node during training carries substantial state.

### Core Fields

```python
class PCTreeNode_Training:
    # Tree structure
    parent: PCTreeNode_Training
    children: list[PCTreeNode_Training]
    ctx: PCTreeTrainer_Partitioning  # Reference to trainer context

    # This node's principal direction
    v: NDArray      # [dim] principal vector
    sigma: float    # Singular value

    # Partition's SVD (in reduced dimension)
    U: NDArray      # [N_data × rank] left singular vectors
    s: NDArray      # [rank] singular values
    VT: NDArray     # [rank × dim] right singular vectors

    # Data membership
    data_mask: NDArray  # [total_N] boolean mask
    N_data: int         # Number of points in this partition

    # Fuse state
    compr_test_fuse: int | None  # Countdown to forced split

    # Dimensionality estimates
    rank: int       # Current working rank
    abid_dim: int   # Estimated intrinsic dimensionality
```

### Key Methods

**`expand(self, budget: int) -> PCTNode_ExpansionResult`**: Main expansion logic. Runs sifting test (unless bypassed), decides extend vs split, returns new nodes.

**`cluster_expand(self, budget: int) -> PCTNode_ExpansionResult`**: Performs binary subspace clustering split. Called when split decision is made.

**`expand_no_more_splits(self) -> PCTreeNode_Training`**: Extends without possibility of future splits. Used when reaching end of tree growth.

### Memory Management

The `kms()` method ("kill my state") clears large arrays after expansion:

```python
def kms(self):
    self.U = None
    del self.U
```

**Runtime optimization**: `U` matrices are the largest memory consumers (N_data × rank). After a node expands, its `U` is no longer needed—children have their own. Clearing promptly prevents memory exhaustion during deep tree construction.

### Tree Navigation

```python
def inclusive_siblings(self) -> list:
    """All children of this node's parent, including self."""
    if self.parent is not None:
        return self.parent.children
    return [n for n in self.ctx.node_set if n.parent is None]

def height(self) -> int:
    """Distance from root."""
    if self.parent is None:
        return 0
    return self.parent.height() + 1
```

## 9. From Training Nodes to PCT: `to_pctree()`

After partitioning completes, the `PCTreeNode_Training` graph is converted to a standard `PCTree` object.

```python
def to_pctree(self, extend_leaves: bool = False, verbose: int = 0)
    -> tuple[PCTree, NDArray, NDArray]:
```

### The Conversion Process

The method traverses training nodes and builds arrays:

```python
while len(stack) > 0:
    curr = stack.pop()

    if curr_extend_leaf:
        # Add ALL remaining principal directions
        V = np.vstack([V, curr.VT])
        s = np.concatenate([s, curr.s])
    else:
        # Add just the first principal direction
        V = np.vstack([V, curr.v])
        s = np.concatenate([s, [curr.sigma]])
```

### The `extend_leaves` Parameter

When `extend_leaves=True`, each leaf's entire remaining SVD is appended to the tree—not just the first principal vector. This creates an "oversized" tree where every branch extends to its full effective rank.

**Why this matters**: The EM phase requires equal-length branches to prevent "withering" (see [PCT EM Training](pct-em-training.md)). Extending leaves here provides the raw material for EM to prune back intelligently.

Example sizes:
- Without extension: 500-node tree (partitioning result)
- With extension: 50,000-node tree (full effective ranks)
- After EM pruning: 10,000-node tree (target size)

### Return Values

```python
return result_tree, training_data_assignments, data_prop
```

- **`result_tree`**: The constructed `PCTree`
- **`training_data_assignments`**: Per-point branch assignments from partitioning
- **`data_prop`**: Per-node data proportion (fraction of data touching each node)

## 10. Class and Method Reference

### Core Classes

| Class | Purpose | Key Fields |
|-------|---------|------------|
| `PCTreeTrainer` | Top-level trainer | `options`, `root_nodes` |
| `PCTreeTrainer_Partitioning` | Manages expansion loop | `X`, `leaves`, `node_set` |
| `PCTreeNode_Training` | Per-node training state | `U`, `s`, `VT`, `data_mask`, `compr_test_fuse` |
| `PCTreeTrainerOptions` | Hyperparameter container | `max_nodes`, `max_width`, `sifting_effect_size` |
| `PCTreeTrainingBudgeter` | Greedy node selection | `subtree_selector` |

### Key Methods

| Method | Signature | Purpose |
|--------|-----------|---------|
| `PCTreeTrainer.fit_partition` | `(X, verbose) -> None` | Run top-down partitioning |
| `PCTreeTrainer.fit_em` | `(X, X_topdown, n_iters, ...) -> PCTree` | Run partitioning + EM refinement |
| `PCTreeTrainer.to_pctree` | `(extend_leaves, verbose) -> (PCTree, NDArray, NDArray)` | Convert to PCT |
| `PCTreeTrainer_Partitioning.expand_next` | `() -> bool` | Expand one node, return whether to continue |
| `PCTreeNode_Training.expand` | `(budget) -> PCTNode_ExpansionResult` | Main expansion logic |
| `PCTreeNode_Training.cluster_expand` | `(budget) -> PCTNode_ExpansionResult` | Binary split |

### Runtime Optimizations Summary

| Optimization | Location | Impact |
|--------------|----------|--------|
| Variance truncation | Every SVD | 10× speedup per SVD |
| `efficient_split_svd` | `cluster_expand()` | 2× speedup for split SVDs |
| Chunked outer products | `sift_subspace()` | O(N) instead of O(N²) comparisons |
| Memory clearing (`kms`) | After expansion | Prevents OOM on deep trees |
| Greedy budgeting | `PCTreeTrainingBudgeter` | Avoids full tree recomputation |

## 11. Summary: The Algorithm at a Glance

```
INPUT: Data X [N × dim], options (max_nodes, max_width, sifting_effect_size, ...)

1. INITIALIZE
   - Compute truncated SVD of X (variance_capture_percent)
   - Create root node with full data

2. EXPANSION LOOP (until max_nodes reached or no valid leaves)

   a. SELECT next leaf via greedy budgeting

   b. DECIDE: extend or split?
      - If fuse running: EXTEND, decrement fuse
      - If fuse expired: SPLIT
      - If bypass condition: EXTEND
      - Else: run sift_subspace()
        - No clustering: EXTEND
        - Clustering, 0 intersection: SPLIT
        - Clustering, N intersection: set fuse=N-1, EXTEND

   c. EXTEND:
      - Create single child with next principal direction
      - Child inherits data and truncated SVD

   c. SPLIT:
      - Binary subspace clustering
      - Per-cluster SVD (truncated to variance_capture_percent)
      - Create two children, remove current node

3. FINALIZE
   - Convert PCTreeNode_Training graph to PCTree
   - Optionally extend leaves to full rank for EM

OUTPUT: PCTree structure, branch assignments, data proportions
```

### Quick Reference: Hyperparameters

| Parameter | Type | Effect |
|-----------|------|--------|
| `max_nodes` | int | Total node budget (primary stopping criterion) |
| `max_width` | int | Max leaves; triggers end of splitting phase |
| `max_children` | int | Max children per node |
| `variance_capture_percent` | float (0.99-0.999) | Controls subspace shrinkage |
| `sifting_effect_size` | float (0.2-0.8) | **Most important**. Higher = fewer splits, more sharing |
| `subspace_sifter_max_dim` | int | Max dimensions for sifting test |

---

*See also: [Subspace Sifting](pct-subspace-sifting.md) for the split detection algorithm, [PCT EM Training](pct-em-training.md) for post-partitioning refinement.*
