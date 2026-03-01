# Principal Component Trees

This document explains Principal Component Trees (PCTs), a data structure for discovering hierarchical, intersecting subspace structure in high-dimensional data.

## What is a Principal Component Tree?

A Principal Component Tree is a data structure that generalizes PCA to capture **hierarchical, intersecting subspace structure** in high-dimensional data. Where PCA learns a single subspace to approximate all data, and subspace clustering learns multiple disjoint subspaces, PCT learns a tree of subspaces where **branches can share common ancestors**.

### Core Structure

A PCT consists of:
- **Nodes**: Each node holds a principal vector `v` and singular value `σ`
- **Edges**: Parent-child re    lationships forming a tree
- **Branches**: Paths from root to leaf, each defining a subspace

```python
class PCTree:
    V: NDArray   # [N × dim] principal vectors
    E: NDArray   # [N] parent indices (-1 for roots)
    s: NDArray   # [N] singular values
```

### Key Properties

1. **Ancestral Orthogonality**: Each node's principal vector is orthogonal to all its ancestors and descendants. This makes projection efficient—just matrix multiply, no inversion needed.

2. **Non-decreasing Singular Values**: A node's singular value is ≤ its parent's. The most important directions appear near the root.

3. **PCA as Special Case**: A PCT with no splits (single chain of nodes) is exactly PCA.

4. **Subspace Clustering as Special Case**: A PCT that splits only at the root gives disjoint subspace clustering.

## How PCTs Compress Data Better Than PCA

The key insight: **different data points lie near different subspaces**. PCA forces all data through the same set of principal components. PCT routes each data point to its best-fitting branch.

### Empirical Results

From the demo notebooks on MNIST and LLaMA token embeddings:

**MNIST** (demo_start_here.ipynb):
```
PCTree: 3.5 scalars per data point
PCA:    9.46 scalars for same reconstruction quality
```

**LLaMA Token Embeddings** (demo_compression.ipynb):
```
PCTree: 184 scalars average (4000 nodes, 30 branches)
PCA:    551 scalars for same 95.9% variance captured
```

PCT achieves ~3x compression advantage by specializing branches to different data subpopulations.

### Why This Works

Consider word embeddings for "north", "south", "east", "west". These directional words form a tight cluster that lies on a low-dimensional subspace. A PCT branch specialized for this cluster needs only ~8 dimensions to capture most of the variance, while PCA—optimizing for all words—would need ~40 dimensions.

The compression demo shows this visually: for target words, the specialized branch captures variance much faster than PCA (steeper curve on the rank vs. variance-captured plot).

## PCT Construction Algorithm

The `PCTreeTrainer` builds PCTs through a top-down partitioning process:

### 1. Start with PCA
```python
U, s, VT = svd(X)
root = PCTreeNode(V=VT[0], s=s[0], data=all_data)
```

### 2. For Each Leaf, Decide: Extend or Split?

**Extend** (add child along same direction): When the residual data still lies near a single subspace. Detected via angle distribution hypothesis testing—if pairwise angles match the uniform distribution expected for a single subspace, don't split.

**Split** (create multiple children): When the residual shows subspace cluster structure. Detected by comparing observed angle distribution to null hypothesis.

```python
def expand(node):
    test_result = sift_subspace(node.U, node.s, options)
    if test_result.clustering_detected:
        return cluster_expand(node)  # Split into 2+ children
    else:
        return extend(node)  # Single child, next PC
```

### 3. Subspace Clustering for Splits

When splitting, use binary subspace clustering on the residual's low-dimensional representation:
```python
X_residual = U[:, 1:] * s[1:]  # Remove first PC
clusterer = BinaryBuckshotSubspaceClusterer()
assignments = clusterer.fit_predict(X_residual)
```

Each cluster becomes a new branch with its own SVD.

### 4. Iterative Refinement (EM)

After initial partitioning, refine with expectation-maximization:
1. **E-step**: Reassign data to best-fitting branches
2. **M-step**: Recompute principal vectors for each branch

## Branch Assignment and Routing

Given a PCT and data point `x`, which branch best represents it?

### Efficient Encoding Router

The `EfficientEncodingRouter` picks the branch that captures variance fastest:

```python
def predict(self, X):
    C = X @ self.pct.V.T  # Project onto all nodes
    for branch in branches:
        coeffs = C[:, branch] ** 2
        cumulative = coeffs.cumsum(axis=1)
        # Score by area under cumulative variance curve
        auc = cumulative.sum(axis=1)
    return branch_aucs.argmax(axis=1)
```

Intuition: A good branch captures most variance in its first few nodes. The "area under curve" metric rewards branches that front-load variance capture.

## Chords: Maximal Single-Child Chains

A **chord** is a maximal chain of nodes where each has exactly one child. Chords are the "straight sections" between splits in the tree.

```
Tree:           Chords:
    A              [A-B-C] (chord 0)
    |              [D-E]   (chord 1)
    B              [F]     (chord 2)
    |
    C
   / \
  D   F
  |
  E
```

The PCT library provides infrastructure for chord-based operations:
```python
tree.chords()           # List of all chords
tree.branch_chords(i)   # Which chords make up branch i
tree.chord_branches(i)  # Which branches share chord i
```

Chords are useful for downstream applications where you want to train models on contiguous sequences of principal components, particularly when those sequences need to be uniform in length for hardware efficiency.

## Library Reference

The `pctree` library provides:

| Module | Purpose |
|--------|---------|
| `core.py` | `PCTree`, `PCTreeCoefficients` classes |
| `training.py` | `PCTreeTrainer`, top-down construction |
| `branches.py` | `EfficientEncodingRouter`, branch assignment |
| `subspace_sifting.py` | Structure detection tests for split decisions |
| `reweighting.py` | EM refinement of principal vectors |

## Summary

Principal Component Trees extend PCA to capture hierarchical subspace structure with intersections:

1. **Hierarchical subspace discovery**: PCT finds which data points cluster together and how clusters relate hierarchically
2. **Branches capture specialization**: Each branch captures a distinct subspace with its own dimensionality
3. **Shared ancestors capture commonality**: Parameter efficiency through shared structure
4. **Geometric routing**: Branch assignment via projection-based measures, no learned routing needed

PCTs provide a foundation for applications requiring hierarchical data organization, variable-depth representations, and structured compression.

---

*Reference: "Principal Component Trees and Their Persistent Homology" (Kizaric & Pimentel-Alarcón, AAAI 2024)*
