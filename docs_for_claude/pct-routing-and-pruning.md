# PCT Routing and Pruning

This document explains the routing and pruning algorithms used in Principal Component Tree training and inference. These components address two fundamental challenges:

- **Routing**: Given a data point and a PCT with multiple branches, which branch best represents it?
- **Pruning**: Given a full PCT, how do we select a compact subtree that preserves quality?

Both challenges share a unifying principle: **Efficient Encoding**—maximize variance captured per unit of encoding cost.

The routing algorithms are implemented in [branches.py](../src/pctree/branches.py), and pruning algorithms in [pruning.py](../src/pctree/pruning.py).

## The Efficient Encoding Principle

The naive approach to branch assignment or node selection is to maximize total variance captured. But this can be misleading when comparing options of different sizes.

Consider compression as an **encoding problem**. When we compress a dataset using a subspace:
- We pay a cost for each coefficient we compute and store
- We gain variance captured by that coefficient

The **Progressive Compression Curve** visualizes this trade-off:
- **Y-axis**: Cumulative variance captured
- **X-axis**: Cumulative encoding cost (coefficients spent)

The **Area Under the Curve (AUC)** measures encoding efficiency—how much "compression value" we get across the entire encoding process.

### Why AUC Beats Total Variance

A subspace that captures variance **earlier** (steeper initial curve) may be preferable to one with higher total variance but slower ramp-up.

Consider two branches for encoding a data point:

| Branch | Rank 1 Var | Rank 2 Var | Rank 3 Var | Total Var | AUC |
|--------|-----------|-----------|-----------|-----------|-----|
| A (short) | 50 | - | - | 50 | 50 |
| B (long) | 20 | 20 | 30 | 70 | 20 + 40 + 70 = 130... |

Wait—this seems to favor the longer branch. The key insight is that we need to **normalize for length**. If we extend Branch A's curve as if it continues at 100% capture:

| Branch | Cumulative @ Rank 1 | @ Rank 2 | @ Rank 3 | AUC (sum of cumulatives) |
|--------|---------------------|----------|----------|--------------------------|
| A | 50 | 50 | 50 | 150 |
| B | 20 | 40 | 70 | 130 |

Now Branch A wins despite lower total variance, because it **front-loads** the variance capture.

### Numerical Example: Node Selection

When selecting nodes for a subtree, we also consider **data proportions**—what fraction of the dataset uses each node. Here's an example comparing three candidate nodes:

```
Node A: variance=10, proportion=0.1 (high variance, few data points)
Node B: variance=11, proportion=0.2 (slightly more variance, more data)
Node C: variance=5,  proportion=0.3 (less variance, most data)
```

The encoding AUC formula scores each node:
```
triangle_part = (variance × proportion) / 2
remaining_part = variance × (max_proportion - proportion)
score = triangle_part + remaining_part
```

With `max_proportion = 0.3`:

| Node | Triangle | Remaining | Total Score |
|------|----------|-----------|-------------|
| A | 0.5 | 2.0 | **2.5** |
| B | 1.1 | 1.1 | 2.2 |
| C | 0.75 | 0.0 | 0.75 |

Node A wins despite having less variance than B, because it provides high variance for a small data fraction—efficient encoding.

If we increase `max_proportion` to 0.9 (simulating a frontier with a high-proportion node):

| Node | Triangle | Remaining | Total Score |
|------|----------|-----------|-------------|
| A | 0.5 | 8.0 | **8.5** |
| B | 1.1 | 7.7 | 8.8 |

Now B wins—when the X-axis extends further, B's slight absolute variance advantage compounds over the extended range.

## Branch Routing

### EfficientEncodingRouter

The primary router assigns each data point to its best-matching branch using the AUC principle.

**Algorithm**:

1. Project all data onto all principal vectors: `C = X @ pct.V.T`
2. For each branch:
   - Extract the coefficients for nodes in that branch
   - Square them to get variance captured: `branch_coeffs ** 2`
   - Compute cumulative sum along the branch depth
   - Sum all cumulative values = **branch AUC** for each data point
3. Assign each point to the branch with highest AUC

```python
def predict(self, X: NDArray):
    C = X @ self.pct.V.T
    branch_aucs = np.zeros((n_data, n_branches))

    for bi, branch_indices in enumerate(pct.branches()):
        branch_coeffs = C[:, branch_indices] ** 2
        branch_cumm_coeffs = branch_coeffs.cumsum(axis=1)
        total_var = branch_cumm_coeffs[:, -1]

        # Sum of cumulatives = AUC
        branch_coeffs_cumm_part_auc = np.sum(branch_cumm_coeffs, axis=1)

        # Normalize for branch length
        already_captured_variance = total_var * (max_branch_height - branch_height)
        branch_aucs[:, bi] = branch_coeffs_cumm_part_auc + already_captured_variance

    return branch_aucs.argmax(axis=1)
```

**Branch Length Normalization**:

The line `already_captured_variance = total_var * (max_branch_height - branch_height)` extends shorter branches as if they continue at 100% variance capture. Without this:
- Longer branches accumulate more AUC simply by having more terms
- In iterative route→prune→route cycles, shorter branches "wither away" as they lose data assignments
- The tree degenerates toward a single long branch

### ReducedBranchRouter

For large trees, computing full branch assignments is expensive. `ReducedBranchRouter` provides a fast approximation.

**Approach**:
1. Use `TendrilVariancePruner` to create a reduced "structural skeleton"
2. Apply `EfficientEncodingRouter` on the reduced tree

```python
class ReducedBranchRouter(PCTreeBranchRouter):
    def __init__(self, pct: PCTree, tendril_varcap: float = 0.9, verbose: int = 0):
        reduced_tree_mask = TendrilVariancePruner(tendril_varcap, verbose).subtree(pct)
        self.reduced_branch_tree = pct.subtree(reduced_tree_mask)
        self.reduced_tree_brancher = EfficientEncodingRouter(self.reduced_branch_tree)
```

The reduced tree keeps all internal (shared) nodes but truncates the tendrils to capture only `tendril_varcap` (e.g., 90%) of their variance. This dramatically reduces computation while preserving routing accuracy—the "structural" parts of the tree determine routing, while deep tendrils are just refinement.

## Subtree Pruning

### EfficientEncodingPruner

Selects a size-limited subtree that balances variance captured against data distribution.

**Input**: Requires `data_proportions`—the fraction of data flowing through each node, computed from prior branch assignments via `branch_assignments_to_node_data_proportions()`.

**Algorithm** (greedy frontier-based selection):

1. Initialize frontier with root node(s)
2. Score each frontier node using `encoding_auc()`
3. Select the highest-scoring node
4. Expand frontier to include that node's children
5. Repeat until reaching the size limit

```python
def subtree(self, size: int, pct: PCTree, data_proportions: NDArray):
    selected_subtree = np.zeros(pct.s.shape, dtype=bool)
    curr_nodes_to_inspect = pct.root_mask()  # Frontier starts at root

    while selected_subtree.sum() < size:
        scores = self.encoding_auc(pct, data_proportions, selected_subtree, curr_nodes_to_inspect)
        next_node = scores.argmax()

        selected_subtree[next_node] = True
        curr_nodes_to_inspect[next_node] = False
        curr_nodes_to_inspect[pct.child_mask(next_node)] = True  # Expand frontier

    return selected_subtree
```

This respects tree structure—you cannot select a child before its parent—while optimizing for encoding efficiency rather than raw variance.

The `encoding_auc()` function scores nodes using the triangle + remaining formula described in the Efficient Encoding Principle section. The `max_data_prop` from the current frontier defines the common X-axis endpoint for fair comparison.

### MostVariancePruner

A simpler alternative that just selects the top-k nodes by singular value.

```python
def subtree(self, size: int, pct: PCTree, data_proportions: NDArray):
    node_order = pct.s.argsort()[::-1][:size]
    selected_subtree[node_order] = True
    return selected_subtree, node_order
```

**Theoretical justification** (Property 2 from the PCT paper): A node's singular value is always ≤ its ancestors'. Therefore, selecting the top-k nodes by singular value is **guaranteed to form a valid subtree** with no gaps—you'll never select a child without its parent.

**When to use**: When you want optimal reconstruction error without considering data distribution. This gives the mathematically best size-n approximation of the full tree, but may over-invest in high-traffic branches at the expense of smaller efficient ones.

### TendrilVariancePruner

Unlike the other pruners, this doesn't target a specific size—it prunes by **variance threshold**.

**Approach**:
- **Internal chords** (shared by multiple branches): Keep all nodes
- **Tendrils** (leaf chords, unique to one branch): Keep only enough nodes to capture `tendril_varcap` (e.g., 90%) of that tendril's variance

```python
def subtree(self, pct: PCTree):
    for br_i, br in enumerate(pct.branches()):
        # Keep all internal chords
        internal_chords = [pct.chords()[ci] for ci in pct.branch_chords(br_i)[:-1]]
        for ic in internal_chords:
            internal_mask[ic] = True

        # Truncate tendril by variance threshold
        tendril = pct.chords()[pct.branch_chords(br_i)[-1]]
        tendril_cap_dim = rank_to_explain_variance_given_s(pct.s[tendril], self.tendril_varcap)
        selected_tendril_mask[tendril.start:tendril.start+tendril_cap_dim] = True

    return internal_mask + selected_tendril_mask
```

**Why keep all internal nodes**: Internal chords define the tree's branching structure. Removing them can cause branches to disappear entirely, leading to index consistency issues. The tendrils are where most nodes live anyway, so truncating them provides the compression benefit.

**Primary use**: Creating reduced trees for fast routing (see `ReducedBranchRouter`).

## Role in Training

Routing and pruning are used iteratively during PCT training:

1. **Route**: Assign data to branches using `EfficientEncodingRouter`
2. **Compute proportions**: Convert assignments to node proportions via `branch_assignments_to_node_data_proportions()`
3. **Prune**: Select subtree using `EfficientEncodingPruner` with those proportions
4. **Repeat** with the pruned tree

For inference, `ReducedBranchRouter` provides one-time pruning with fast subsequent routing.

Detailed documentation of the end-to-end training loop will be covered in a separate document.

## Summary

| Component | Purpose | Key Input | Use Case |
|-----------|---------|-----------|----------|
| `EfficientEncodingRouter` | Assign points to branches | PCT | Training & inference |
| `ReducedBranchRouter` | Fast approximate routing | PCT, `tendril_varcap` | Inference speedup |
| `EfficientEncodingPruner` | Encoding-efficient subtree | `size`, `data_proportions` | Training iterations |
| `MostVariancePruner` | Variance-maximizing subtree | `size` | Optimal reconstruction |
| `TendrilVariancePruner` | Truncate tendrils by variance | `tendril_varcap` | Creating reduced routing trees |