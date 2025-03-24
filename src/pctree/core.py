import numpy as np
from numpy.typing import NDArray
from typing import Self

# mini-util
def in_slice(inds: NDArray, slicey: slice) -> NDArray:
    return (inds >= slicey.start) * (inds < slicey.stop)


class PCTree:
    ROOT_FLAG = -1

    # These three are "Core"
    V: NDArray = None
    E: NDArray = None
    s: NDArray = None

    # These are derived from the core
    N: int = None
    dim: int = None

    _branches: list[NDArray] = None
    _chords: list[slice] = None
    # branch_idx -> list of chords that are involved in the branch. Populated all at once.
    _branch_chords: list[list[int]] = None
    #chord_idx -> list of branches that are involved in the chord. Populated all at once.
    _chord_branches: list[set[int]] = None
    # chord idx -> list of chords that descend from chord i
    _chord_descendents: list[set[int]] = None
    _chord_anscestors: list[set[int]] = None
    

    def __init__(self, V: NDArray, E: NDArray, s: NDArray) -> None:
        self.V = V
        self.E = E
        self.s = s
        assert V.shape[0] == E.shape[0]
        assert V.shape[0] == len(s)

        self.N = V.shape[0]
        self.dim = V.shape[1]

    def _path_to_root(self, node_idx: int):
        route_to_root = []
        curr_node_idx = node_idx
        is_root = lambda n: self.E[n] == PCTree.ROOT_FLAG

        while not is_root(curr_node_idx):
            route_to_root.append(curr_node_idx)
            curr_node_idx = self.parent(curr_node_idx)
            #curr_node_idx = np.where(self.E[curr_node_idx])[0][0]
        route_to_root.append(curr_node_idx)

        return np.array(list(reversed(route_to_root)))

    def branches(self) -> list[NDArray]:
        if self._branches is None:
            result = []
            for leaf_i in self.leaf_inds():
                result.append(self._path_to_root(leaf_i))
            self._branches = result
        return self._branches
    
    def _chords_recursive(self: Self, accumulator: list[slice], curr_node_idx: int, curr_chord: slice):
        next_children = self.child_inds(curr_node_idx)
        #next_children = list(np.where(curr_node_child_mask)[0])

        if curr_chord is None: # if this node is the start of a new chord.
            curr_chord = slice(curr_node_idx, curr_node_idx)

        if len(next_children) == 0: # reached a leaf
            curr_chord = slice(curr_chord.start, curr_node_idx)
            accumulator.append(curr_chord) # finished with chord
        if len(next_children) == 1: # continuation of chord
            curr_chord = slice(curr_chord.start, curr_node_idx)
            self._chords_recursive(accumulator, next_children[0], curr_chord)
        if len(next_children) > 1: # This is the last node of a chord. Start a new chord for each child.
            curr_chord = slice(curr_chord.start, curr_node_idx) # finish the chord
            accumulator.append(curr_chord)
            for child_node_i in next_children:
                self._chords_recursive(accumulator, child_node_i, None)
        return

    def chords(self) -> list[slice]:
        if self._chords is None:
            result_inclusive: list[slice] = []
            roots_idx = self.root_inds()
            #roots_idx = list(np.where(roots_mask)[0])
            for root_i in roots_idx:
                self._chords_recursive(result_inclusive, root_i, None)
            result_proper_slice = []
            for chord in result_inclusive:
                result_proper_slice.append(slice(chord.start, chord.stop + 1))
            self._chords = result_proper_slice
            # a quick test
            coverage = np.zeros(self.N, dtype=int)
            for chord in self._chords:
                coverage[chord] += 1
            assert (coverage == 1).sum() == self.N
        return self._chords
    
    def _compute_chord_branch_relations(self):
        chords = self.chords()
        branches = self.branches()
        # [set()] * len(chords)  no work - each item is the same set!
        self._chord_branches = [set() for i in range(len(chords))] 
        self._branch_chords = [[] for i in range(len(branches))] 
        for chord_idx, chord in enumerate(chords):
            for branch_idx, branch in enumerate(branches):
                if in_slice(branch, chord).sum() > 0:
                    self._chord_branches[chord_idx].add(branch_idx)
                    self._branch_chords[branch_idx].append(chord_idx)
        for branch_idx, branch in enumerate(branches):
            self._branch_chords[branch_idx] = list(set(self._branch_chords[branch_idx]))
            self._branch_chords[branch_idx] = list(sorted(self._branch_chords[branch_idx], key=lambda chord_i: chords[chord_i].start))
    
    def chord_branches(self, chord_idx: int) -> set[int]: # branch_indexes
        if (self._chord_branches is None) or (self._branch_chords is None):
            self._compute_chord_branch_relations()
        return self._chord_branches[chord_idx]
    
    def branch_chords(self, branch_idx: int) -> list[int]: # chord_indexes
        if (self._chord_branches is None) or (self._branch_chords is None):
            self._compute_chord_branch_relations()
        return self._branch_chords[branch_idx]
    
    def _compute_chord_chord_relations(self):
        chords = self.chords()
        self._chord_anscestors = [set() for i in range(len(chords))] 
        self._chord_descendents = [set() for i in range(len(chords))] 
        for chord_a_idx, chord_a in enumerate(chords):
            chord_a_branches = self.chord_branches(chord_a_idx)
            for chord_b_idx, chord_b in enumerate(chords):
                chord_b_branches = self.chord_branches(chord_b_idx)
                if chord_a_idx == chord_b_idx:
                    continue
                if chord_a_branches.isdisjoint(chord_b_branches):
                    continue
                if chord_a.start < chord_b.start:
                    self._chord_descendents[chord_a_idx].add(chord_b_idx)
                    self._chord_anscestors[chord_b_idx].add(chord_a_idx)
                if chord_b.start < chord_a.start:
                    self._chord_descendents[chord_b_idx].add(chord_a_idx)
                    self._chord_anscestors[chord_a_idx].add(chord_b_idx)
    
    def chord_descendents(self, chord_idx: int) -> list[int]: # chord_indexes, not including the input.
        if (self._chord_anscestors is None) or (self._chord_descendents is None):
            self._compute_chord_chord_relations()
        return self._chord_descendents[chord_idx]
    
    def chord_anscestors(self, chord_idx: int) -> list[int]: # chord_indexes, not including the input.
        if (self._chord_anscestors is None) or (self._chord_descendents is None):
            self._compute_chord_chord_relations()
        return self._chord_anscestors[chord_idx]
    
    def chord_parent(self, chord_idx: int) -> int:
        anscestors = self.chord_anscestors(chord_idx)
        if len(anscestors) == 0:
            return None
        anscestors_order = list(sorted(anscestors, key=lambda l: self.chords()[l].start))
        return anscestors_order[-1]
    
    def subtree(self, nodes_selected_mask: NDArray):
        # Subtree is the one operation where switching from sparse binary matrix to list of parents is more complicated, 
        # and probably slower.
        curr_e = self.E.copy()
        nodes_removed = np.sort(np.where(~nodes_selected_mask)[0])[::-1] # descending
        n_nodes_removed = len(nodes_removed)
        for i in range(n_nodes_removed):
            node_to_remove = nodes_removed[i] # the largest remaining node
            curr_e[curr_e > node_to_remove] -= 1
            curr_e[node_to_remove] = -99
        curr_e = curr_e[curr_e != -99]
        return PCTree(self.V[nodes_selected_mask].copy(), curr_e, self.s[nodes_selected_mask].copy())
    
    def root_mask(self):
        return self.E == PCTree.ROOT_FLAG
    
    def root_inds(self):
        result = np.arange(self.N, dtype=int)
        return result[self.root_mask()]

    def non_leaf_inds(self) -> NDArray:
        # NO ORDER
        return np.sort(np.unique(self.E[self.E >= 0]))
    
    def leaf_inds(self) -> NDArray:
        # NO ORDER
        result = np.arange(self.N, dtype=int)
        return result[self.leaf_mask()]

    def leaf_mask(self) -> NDArray:
        non_leaf_mask = np.zeros(self.N, dtype=bool)
        non_leaf_mask[self.non_leaf_inds()] = True
        return ~non_leaf_mask
    
    def child_mask(self, parent_i: int) -> NDArray:
        return self.E == parent_i

    def child_inds(self, parent_i: int) -> NDArray:
        result = np.arange(self.N)
        return result[self.child_mask(parent_i)]

    def children_of_mask(self, node_mask: NDArray) -> NDArray:
        return self.E[:, node_mask].sum(axis=1) > 0
    
    def traverse(self) -> NDArray:
        result: list[int] = []
        roots_idx = self.root_inds()
        #roots_idx = np.where(roots_mask)[0]
        node_stack = list(roots_idx)
        while(len(node_stack) > 0):
            next_node_idx = node_stack.pop(0)
            result.append(next_node_idx)
            next_children_inds = self.child_inds(next_node_idx)
            if len(next_children_inds) == 0:
                #print(next_node_idx, "leaf")
                continue
            #else:
            #    print(next_node_idx, "not leaf")
            #next_child_idx = np.where(next_child_mask)[0]
            node_stack = list(next_children_inds) + node_stack
        return result
    
    def parent(self, node_idx: int) -> int | None:
        p = self.E[node_idx]
        return None if p == -1 else p
    
    def is_only_child(self, node_idx: int) -> bool:
        parent = self.parent(node_idx)
        if parent == None:
            return False
        children_of_parent = self.child_inds(parent)
        if len(children_of_parent) != 1:
            return False
        assert children_of_parent[0] == node_idx
        return True
    
def expand_tree_chord_with_zeros(tree: PCTree, chord_i: int, target_size: int):
    chords = tree.chords()
    assert chord_i < len(chords)
    assert chord_i >= 0
    target_chord = chords[chord_i]
    current_size = target_chord.stop - target_chord.start
    assert target_size >= current_size
    if target_size == current_size:
        return PCTree(tree.V.copy(), tree.E.copy(), tree.s.copy())
    fill_size = target_size - current_size
    last_ind_in_target = target_chord.stop - 1
    V_result = np.vstack([
        tree.V[: last_ind_in_target + 1],
        np.zeros((fill_size, tree.dim)),
        tree.V[last_ind_in_target+1:]
    ])

    s_result = np.concatenate([
        tree.s[: last_ind_in_target + 1],
        np.zeros((fill_size)),
        tree.s[last_ind_in_target+1:]
    ])

    E_result = np.copy(tree.E)
    E_result[E_result >= last_ind_in_target] += fill_size
    E_result = np.concatenate([
        E_result[: last_ind_in_target + 1],
        np.arange(last_ind_in_target, last_ind_in_target + fill_size),
        E_result[last_ind_in_target+1:]
    ])
    return PCTree(V_result, E_result, s_result)
    
class PCTreeCoefficients:
    branches: list[NDArray]
    included_branches: set[int]
    included_nodes = set[int]
    pct: PCTree
    isEmpty: bool # Used in generative models. All coefficients start as zeros. 

    branch_assignments: NDArray
    _branch_coeffs: dict[int, NDArray] # branch index -> 2D array of coeffs. Always populated!
    _branch_counts: NDArray
    _node_coeffs: dict[int, NDArray] # node index -> 1D array of coeffs. Not populated until requested. 
    _node_counts: NDArray

    def __init__(self, pct: PCTree, X: NDArray = None, branch_assignments: NDArray = None, isEmpty: bool = False):
        self.branches = pct.branches()
        self.branch_assignments = branch_assignments
        self.pct = pct
        self.isEmpty = isEmpty
        #self._check_branch_order()

        self.included_branches = set(np.unique(branch_assignments))
        self._compute_included_nodes()
        self._branch_coeffs = {}
        self._branch_counts = np.zeros(len(self.branches), dtype=int)
        self._node_counts = np.zeros(self.pct.N, dtype=int)
        self._compute_branch_coeffs(X)
        self._node_coeffs = {}

    def _compute_included_nodes(self):
        result: set[int] = set()
        for br_i in self.included_branches:
            if br_i >= len(self.branches):
                raise Exception(f"{br_i} not in {len(self.branches)}")
            for ni in self.branches[br_i]:
                result.add(ni)
        self.included_nodes = result

    def _compute_branch_coeffs(self, X: NDArray = None):
        self._branch_coeffs = {}
        for br_i in self.included_branches:
            mask = self.branch_assignments == br_i
            branch_count = mask.sum()
            if not self.isEmpty:
                assert X is not None
                self._branch_coeffs[br_i] = X[mask] @ self.pct.V[self.branches[br_i], :].T
            else:
                self._branch_coeffs[br_i] = np.zeros((branch_count, len(self.branches[br_i])))
            self._branch_counts[br_i] = branch_count
            self._node_counts[self.branches[br_i]] += branch_count
            assert self._branch_coeffs[br_i].shape == (branch_count, len(self.branches[br_i]))
    
    def _compute_node_coeffs(self, node_idx: int):
        values = []
        for br_i in self.included_branches:
            if node_idx not in self.branches[br_i]:
                continue
            node_index_in_branch = (self.branches[br_i] == node_idx).argmax()
            values.append(self._branch_coeffs[br_i][:, node_index_in_branch])
        return np.concatenate(values)
    
    def _check_branch_order(self):
        branch_strings = []
        for br in self.branches:
            branch_strings.append(" ".join(str(n) for n in br))
        assert (np.argsort(branch_strings) != np.arange(len(self.branches))).sum() == 0

    def branch(self, idx: int) -> tuple[NDArray, NDArray]:
        assert idx in self.included_branches
        return self.branches[idx], self._branch_coeffs[idx]
    
    def node(self, idx: int) -> tuple[NDArray, NDArray]:
        assert idx in self.included_nodes

        if idx not in self._node_coeffs:
            self._node_coeffs[idx] = self._compute_node_coeffs(idx)
        return self._node_coeffs[idx]
    
    def node_branches(self, node_idx: int):
        result = []
        for br_i in self.included_branches:
            if node_idx in self.branches[br_i]:
                result.append(br_i)
        return result
    
    def node_branch(self, node_idx: int, branch_index: int) -> tuple[NDArray, NDArray]:
        assert node_idx in self.included_nodes
        assert branch_index in self.included_branches

        node_index_in_branch = (self.branches[branch_index] == node_idx).argmax()
        return self._branch_coeffs[branch_index][:, node_index_in_branch]
    
    def set_node(self, node_idx: int, values: NDArray):
        assert node_idx in self.included_nodes
        assert len(values) == self._node_counts[node_idx]
        value_offset = 0
        for br_i, br in enumerate(self.branches):
            if node_idx not in self.branches[br_i]:
                continue
            node_index_in_branch = (self.branches[br_i] == node_idx).argmax()
            self._branch_coeffs[br_i][:, node_index_in_branch] = values[value_offset:value_offset+self._branch_counts[br_i]]
            value_offset += self._branch_counts[br_i]
        self._node_coeffs[node_idx] = values

    def set_branch(self, branch_idx: int, values: NDArray):
        assert branch_idx in self.included_branches
        assert values.shape[0] == self._branch_counts[branch_idx]
        assert values.shape[1] == self._branch_coeffs[branch_idx].shape[1]
        self._branch_coeffs[branch_idx] = values

    def reconstruct(self) -> NDArray:
        result = np.zeros((len(self.branch_assignments), self.pct.dim))
        for br_i in self.included_branches:
            result[self.branch_assignments == br_i] = self._branch_coeffs[br_i] @ self.pct.V[self.branches[br_i], :]
        return result
    
    def sum_over_branches(self) -> NDArray:
        result = np.zeros(len(self.branch_assignments))
        for br_i in self.included_branches:
            result[self.branch_assignments == br_i] = self._branch_coeffs[br_i].sum(axis=1)
        return result
    
    def save(self, npz_path: str):
        assert npz_path.endswith(".npz")
        arrays = {}
        arrays["tree_V"] = self.pct.V
        arrays["tree_s"] = self.pct.s
        arrays["tree_E"] = self.pct.E

        arrays["branch-assignments"] = self.branch_assignments
        for br_i, br_c in self._branch_coeffs.items():
            arrays[f"branch_{br_i}"] = br_c

        np.savez(npz_path, **arrays)

    def load_npz(npz_path: str):
        npz = np.load(npz_path)
        tree_V = npz["tree_V"]
        tree_s = npz["tree_s"]
        tree_E = npz["tree_E"]
        tree = PCTree(tree_V, tree_E, tree_s)
        
        result = PCTreeCoefficients(tree, X=None, branch_assignments=npz["branch-assignments"], isEmpty=True)
        for array_name in npz.keys():
            if array_name.startswith("branch_"):
                br_i = int(array_name.split("_")[-1])
                result._branch_coeffs[br_i] = npz[array_name]
        return result
    
    def total_scalars_used(self):
        total = 0
        for br_i, br_c in self._branch_coeffs.items():
            total += br_c.shape[0] * br_c.shape[1]
        return total
    
    def average_scalars_used(self):
        return self.total_scalars_used() / len(self.branch_assignments)