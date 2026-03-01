> **Note**: This document describes a future application of PCTs. The Tree of Orthogonal Experts (ToOE) architecture discussed here is not yet implemented in this codebase.

# Fixed-Size PCT Chords for Hardware-Efficient ToOE

This document explains why Principal Component Tree (PCT) training should support fixed-size chords, motivated by the hardware requirements of the Tree of Orthogonal Experts (ToOE) architecture.

## The ToOE Architecture in Brief

A Tree of Orthogonal Experts (ToOE) is a mixture-of-experts architecture where experts are arranged in a tree structure that mirrors a PCT. Each node in the PCT becomes an expert—a small "bowtie" MLP that handles a slice of the output rank space. Tokens are routed down tree branches, activating all experts along their path. The key property: expert outputs are orthogonal, enabling simple additive combination without learned mixing weights.

**Why tree structure?** It provides three benefits:
1. **Parameter sharing**: Tokens on sibling branches share ancestor experts (like DeepSeek's shared experts)
2. **Variable compute**: Simple tokens (shallow branches) use fewer experts than complex tokens (deep branches)
3. **Specialization**: Different branches handle different data subpopulations

## The Hardware Constraint: All Experts Must Be Identical Shape

Modern MoE implementations (e.g., Mixtral, DeepSeek-V2) require all experts to have identical dimensions. This enables:

- **Batched operations**: Stack all expert weights into 3D tensors, process multiple tokens in parallel
- **Custom kernels**: Triton/CUDA kernels optimized for fixed-shape operations
- **Efficient routing**: Tokens can be dynamically grouped and processed in batches

Variable-sized experts force sequential processing:
```python
# This is unacceptably slow:
for expert_id in range(n_experts):
    mask = (assignments == expert_id)
    output[mask] = experts[expert_id](X[mask])
```

Fixed-size experts enable parallelization:
```python
# This is fast:
expert_outputs = einsum('bte,eoi->bto', tokens_per_expert, stacked_expert_weights)
```

## From PCT Nodes to Experts: The Mapping Challenge

In ToOE, each PCT **chord** (maximal chain of single-child nodes) maps to a sequence of experts. Consider this PCT structure:

```
    A           Chords:
    |           [A-B-C]  (chord 0, length 3)
    B           [D-E]    (chord 1, length 2)
    |           [F-G-H-I](chord 2, length 4)
    C
   / \
  D   F
  |   |
  E   G
      |
      H
      |
      I
```

The naive mapping assigns one expert per node, creating variable-length expert sequences:
- Branch using chord 0 + chord 1: 5 experts total
- Branch using chord 0 + chord 2: 7 experts total

**Problem**: These branches require different compute (different numbers of matrix multiplications). This breaks the batching requirement.

## The Fixed-Size Chord Solution

**Core idea**: Constrain PCT training so all chords have the same length (or are integer multiples of a base size).

With fixed chord size `k=2`:
```
    A-B         Chords:
     |          [A-B]  (chord 0, length 2)
     C-D        [C-D]  (chord 1, length 2)
    /   \       [E-F]  (chord 2, length 2)
  E-F   G-H     [G-H]  (chord 3, length 2)
```

Now all branches have lengths that are multiples of 2:
- Branch 0 + 2: 4 experts (2 sequences of 2)
- Branch 0 + 3: 4 experts (2 sequences of 2)

**Each chord maps to a fixed-size expert**. Variable compute comes from using a variable *number* of same-sized expert sequences, not variable-sized experts.

## Why Integer Multiples Work

Allowing chord lengths to be integer multiples of a base size `k` provides flexibility while maintaining hardware efficiency:

- Base chord: length `k` → 1 expert
- Double chord: length `2k` → 2 identical experts in sequence
- Triple chord: length `3k` → 3 identical experts in sequence

This handles the natural variance in PCT structure (some branches capture more variance and should go deeper) while keeping expert shapes uniform.

## Hardware Efficiency Benefits

**Before (variable-size chords):**
- 30 unique expert shapes across 100 nodes
- Cannot batch across different expert types
- Memory fragmentation from irregular tensor sizes
- Difficult to optimize with custom kernels

**After (fixed-size chords with k=32):**
- 1 expert shape (all handle 32 output ranks)
- All experts stack into `[num_experts, hidden_dim, 32]` tensor
- Batched inference across all active experts
- Single optimized kernel for all expert computations

## Parameter Efficiency of Smaller Experts

Surprisingly, using more smaller fixed-size experts **reduces total parameters** compared to fewer larger variable-size experts covering the same output ranks.

For an expert with ambient dimension `d`, input dimension `k_in`, hidden dimension `k_up`, and output dimension `k_out`:
- Outer parameters: `2·d·k_up + d·k_out` (projection to/from ambient space)
- Inner parameters: `2·k_in·k_up + k_up·k_out` (internal MLP)

The outer term dominates and scales with `k_up`. Splitting one large expert into `n` smaller experts reduces `k_up` by factor `n`, dramatically shrinking the outer parameters.

**Concrete example** (d=2048, covering 128 output ranks):
- 1 expert (k_in=128, k_up=512, k_out=128): 2,555,904 parameters
- 4 experts (k_in=32, k_up=128, k_out=32 each): 2,457,600 parameters (-4%)
- 64 experts covering full LLaMA MLP: 38M parameters vs 50M original (-23%)

Smaller, uniform experts are both more parameter-efficient and hardware-friendly.

## Summary

| Requirement | Solution | Benefit |
|-------------|----------|---------|
| All experts must be identical shape | Fixed-size PCT chords | Enables batched operations |
| Variable compute per token | Variable *number* of fixed-size experts | Deep branches use more experts, shallow use fewer |
| Parameter efficiency | Smaller uniform experts | Fewer total parameters than variable-size experts |
| Discover data structure | PCT with chord-size constraint | Natural subspace clustering preserved |

Fixed-size chords are not a limitation—they're a design constraint that aligns PCT's geometric structure discovery with modern hardware requirements, enabling efficient deployment of tree-structured MoEs at scale.
