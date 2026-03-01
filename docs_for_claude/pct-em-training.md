# PCT Training: Expectation-Maximization Refinement

This document explains the EM (Expectation-Maximization) refinement phase of Principal Component Tree training. After top-down partitioning creates an initial tree structure, EM iteratively improves it by alternating between assigning data to branches and recomputing principal vectors.

**Prerequisite**: This document assumes familiarity with [PCT Routing and Pruning](pct-routing-and-pruning.md), particularly the `EfficientEncodingRouter` and `EfficientEncodingPruner`.

The EM algorithms are implemented in [reweighting.py](../src/pctree/reweighting.py), with supporting routines in [branches.py](../src/pctree/branches.py).

## Why EM Refinement?

Top-down partitioning builds a PCT by recursively splitting data and computing local SVDs. This is fast and produces reasonable structure, but has limitations:

1. **Premature termination**: Partitioning often hits the branch limit (`max_width`) well before reaching the target node count (`max_nodes`), leaving tendrils underdeveloped.

2. **Local decisions**: Each split uses only the data in that partition—there's no global optimization across branches.

3. **Fixed assignments**: Data points are locked to their partition from the split that created it, even if a sibling branch would fit better.

EM refinement addresses all three by applying the **k-subspaces principle**: iteratively reassign data to best-fitting branches (E-step), then recompute each branch's principal vectors from its newly assigned data (M-step).

## The Withering Problem

A naive EM approach on unequal-length branches leads to **branch withering**—shorter branches progressively lose data until they become vestigial.

```
Initial state:          After naive EM iterations:

Branch A: ████████      Branch A: ██████████████
Branch B: ████          Branch B: █
Branch C: ██████        Branch C: ███
```

**The mechanism**:
1. Longer branches accumulate more AUC during routing (even with length normalization, they have more nodes contributing variance)
2. Branches with fewer assigned points produce weaker singular values in the M-step
3. Weaker singular values → even fewer assignments next iteration
4. Feedback loop until short branches capture almost no data

**The solution**: Before EM begins, extend all tendrils to their full effective rank via `to_pctree(extend_leaves=True)`. This puts all branches on an equal footing—they compete based on how well they fit the data, not on structural accidents from partitioning. Pruning after EM determines final branch lengths based on actual data affinity.

## The EM Training Flow

The `fit_em()` method orchestrates EM refinement:

```
┌─────────────────────────────────────────────────────┐
│  Top-down partitioning (fit_partition)              │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  Extend tendrils to full rank                       │
│  to_pctree(extend_leaves=True)                      │
│  Creates large "oversized" tree (can be 100k+ nodes)│
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  Initial reweight + prune                           │
│  reweight_single_flex_given_big_tree()              │
│  Prunes back to target_size                         │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  Fixed-structure EM iterations                      │
│  reweight() with n_iter iterations                  │
└─────────────────────┬───────────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────────┐
│  Final refined PCT                                  │
└─────────────────────────────────────────────────────┘
```

The extended tree can be very large (e.g., 120k nodes for 200-dimensional embeddings), making operations expensive. This motivates `ReducedBranchRouter` for fast approximate routing during EM.

## The E-Step: Branch Assignment

The E-step assigns each data point to its best-fitting branch. See [PCT Routing and Pruning](pct-routing-and-pruning.md) for full details on `EfficientEncodingRouter`.

In the EM context, two additional concerns arise:

**Performance on large trees**: Extended trees are expensive to route through. `ReducedBranchRouter` provides fast approximate routing by truncating tendrils to a variance threshold (e.g., 90%) while keeping all internal structure:

```python
router = ReducedBranchRouter(extended_tree, tendril_varcap=0.95)
assignments = router.predict_batches(X)
```

**Converting to node proportions**: The M-step and pruning need per-node data flow, not per-point assignments. `branch_assignments_to_node_data_proportions()` computes what fraction of data touches each node:

```python
def branch_assignments_to_node_data_proportions(pct, branch_assignments):
    result = np.zeros(pct.N)
    for bi, branch in enumerate(pct.branches()):
        n_assigned = (branch_assignments == bi).sum()
        result[branch] += n_assigned  # All nodes in branch get credit
    return result / len(branch_assignments)
```

A node shared by multiple branches accumulates proportions from all of them—this is how internal nodes naturally get higher proportions than tendril nodes.

## The M-Step: In-Place Reweighting

Given branch assignments, the M-step recomputes the tree's principal vectors (`V`) and singular values (`s`) to optimally fit the assigned data. The key insight: operate at the **chord level** (maximal single-child chains) rather than node-by-node, and enforce orthogonality via null space projection.

### Covariance Aggregation

First, compute covariance matrices at multiple granularities:

```python
# Per-branch covariance
for br_i, branch in enumerate(pct.branches()):
    branch_covar[br_i] = cov(X[assignments == br_i])

# Per-chord covariance = sum over branches containing that chord
for chord_i, chord in enumerate(pct.chords()):
    chord_covar[chord_i] = sum(branch_covar[br_i]
                                for br_i in pct.chord_branches(chord_i))

# Total covariance for working dimension
total_covar = sum(branch_covar.values())
```

### Hierarchical SVD with Null Space Projection

The reweighting proceeds chord-by-chord, enforcing that each chord's principal vectors are orthogonal to all ancestors. This is achieved by projecting onto the parent's **null space**—the subspace orthogonal to what the parent already captures.

Let $\Sigma_c$ denote chord $c$'s covariance matrix, and let $P_c$ denote the projection matrix onto the null space available to chord $c$.

**Algorithm**:

1. **Initialize**: SVD of total covariance gives top directions. Keep enough to capture `varcap` (e.g., 99%) of total variance:
   $$U_\Omega \Lambda_\Omega U_\Omega^T = \text{SVD}(\Sigma_\text{total})$$
   The initial working space is $P_\text{root} = U_\Omega[:, :k]$ where $k$ is the rank for `varcap`.

2. **For each chord** (in tree traversal order):
   - Get parent's null space: $P_\text{parent}$ (or $P_\text{root}$ if root chord)
   - Project chord's covariance onto this space:
     $$\tilde{\Sigma}_c = P_\text{parent}^T \Sigma_c P_\text{parent}$$
   - SVD of projected covariance:
     $$U_c \Lambda_c U_c^T = \text{SVD}(\tilde{\Sigma}_c)$$
   - Store top $|c|$ vectors (chord size) as this chord's principal vectors:
     $$V[c] = U_c[:, :|c|]^T \cdot P_\text{parent}$$
     $$s[c] = \sqrt{\text{diag}(\Lambda_c)[:, :|c|]}$$
   - Pass remaining directions as null space to children:
     $$P_c = U_c[:, |c|:]^T \cdot P_\text{parent}$$

This enforces **ancestral orthogonality**: each node's principal vector lies in the null space of all its ancestors, making projection trivial (just matrix multiply, no inversion).

### Simplified Code

```python
def in_place_reweight(pct, X, assignments, varcap=0.99):
    # 1. Compute covariances
    branch_covar = {bi: cov(X[assignments == bi])
                    for bi in range(len(pct.branches()))}
    chord_covar = {ci: sum(branch_covar[bi] for bi in pct.chord_branches(ci))
                   for ci in range(len(pct.chords()))}
    total_covar = sum(branch_covar.values())

    # 2. Initial working space from total covariance
    s_total, VT_total = svd(total_covar)
    working_dim = rank_for_varcap(s_total, varcap)

    null_spans = {}  # chord_idx -> available projection space
    root_span = VT_total[:working_dim]

    # 3. Process each chord
    for chord_i, chord in enumerate(pct.chords()):
        chord_size = chord.stop - chord.start
        parent_span = root_span if no_parent(chord_i) else null_spans[parent(chord_i)]

        # Project covariance onto parent's null space
        covar_projected = parent_span @ chord_covar[chord_i] @ parent_span.T
        s, VT_local = svd(covar_projected)

        # Store principal vectors (projected back to ambient space)
        pct.s[chord] = s[:chord_size]
        pct.V[chord] = VT_local[:chord_size] @ parent_span

        # Null space for children
        null_spans[chord_i] = VT_local[chord_size:] @ parent_span

    return pct
```

## Tendril Flexibility and Pruning

### The Flex-Then-Prune Pattern

Rather than fixing branch lengths from partitioning, EM uses a **flex-then-prune** approach:

1. **Extend** all tendrils to full effective rank (via `branch_fill_tree()`)
2. **Reweight** on the extended tree—let singular values reflect actual data fit
3. **Prune** back to target size using `EfficientEncodingPruner`

This lets data determine optimal branch depths. A tendril that captures little incremental variance gets pruned short; one that keeps capturing variance stays deep.

### Preserving All Branches

`modify_subtree_selection_to_require_all_branches()` ensures every tendril keeps at least one node after pruning. Without this, `EfficientEncodingPruner` might eliminate low-traffic branches entirely, causing:
- Branch index inconsistencies across iterations
- Loss of specialized subspaces that serve small but distinct data clusters

The function swaps out low-value selected nodes for required tendril roots:

```python
def modify_subtree_selection_to_require_all_branches(pct, initial_mask):
    required = first_node_of_each_tendril(pct)
    if all(required in initial_mask):
        return initial_mask

    # Swap: drop lowest-variance non-required nodes, add missing required ones
    to_drop = lowest_variance_non_required(initial_mask, required)
    return initial_mask - to_drop + required
```

### The Double-Reweight Pattern

When flexing tendrils in `reweight_single()`, two reweighting passes occur:

1. **First reweight**: Update V/s based on incoming assignments
2. **Re-route**: Compute new assignments on the just-reweighted tree
3. **Second reweight**: Update V/s based on fresh assignments

Why? The first reweight can significantly change singular values, making the incoming assignments stale. The second pass ensures V/s reflect the tree's current routing behavior.

## Complete EM Iteration

The `reweight()` method runs fixed-structure EM:

```python
def reweight(self, pct, X, n_iter=1):
    tree = pct
    router = self.branch_router_gen(tree)
    assignments = router.predict_batches(X)  # Initial E-step

    for i in range(n_iter):
        # M-step: reweight given current assignments
        tree = self.reweight_single(tree, X, assignments)

        # E-step: reassign based on updated tree
        router = self.branch_router_gen(tree)
        assignments = router.predict_batches(X)

    return tree
```

**Typical usage**: 1-3 iterations suffice for convergence. Assignments typically stabilize after the first iteration; subsequent iterations refine singular values.

## Summary

| Component | Role in EM | Key Parameters |
|-----------|-----------|----------------|
| `extend_leaves=True` | Prevent withering by equalizing branch lengths | — |
| `ReducedBranchRouter` | Fast E-step on large extended trees | `tendril_varcap` |
| `in_place_reweight()` | M-step: hierarchical SVD with null space projection | `varcap` |
| `branch_fill_tree()` | Extend tendrils to target dimension | target dimension |
| `EfficientEncodingPruner` | Prune extended tree to target size | `size`, data proportions |
| `modify_subtree_selection_to_require_all_branches()` | Prevent branch elimination | — |
| `reweight()` | Complete EM loop | `n_iter` |

The EM phase transforms a rough partitioning-based PCT into a refined structure where branch depths and principal vectors reflect actual data geometry rather than partitioning accidents.
