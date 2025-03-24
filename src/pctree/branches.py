
from numpy.typing import NDArray
import numpy as np
import math
from time import time

from pctree.logging_utils import print_for_verbosity
from pctree.core import PCTree, expand_tree_chord_with_zeros
from pctree.pruning import TendrilVariancePruner

class PCTreeBranchRouter:
    pct: PCTree
    n_branches: int
    is_fit: bool
    verbose: int

    def __init__(self, pct: PCTree, verbose: int = 0) -> None:
        self.pct = pct
        self.is_fit = False
        self.verbose = verbose

    def fit(self, X: NDArray):
        self.is_fit = True
        pass

    def predict(self, X: NDArray) -> NDArray:
        pass

    def predict_batches(self, X: NDArray, batch_size: int = 10_000) -> NDArray:
        n_batch = math.ceil(X.shape[0] / batch_size)
        batch_assignments = []
        print_for_verbosity(self.verbose, first_level="[Branch Assignment]", second_level="")
        print_for_verbosity(self.verbose, second_level=f"Assigning data to one of {len(self.pct.branches())} branches of size-{self.pct.N} pct in batches of {batch_size}. \n")
        print_for_verbosity(self.verbose, second_level=f"Log Key: '|' logged for every batch assigned. \n")
        t0 = time()
        for bi in range(n_batch):
            batch_assignments.append(self.predict(X[bi*batch_size:(bi+1)*batch_size]))
            print_for_verbosity(self.verbose, "|")
        print_for_verbosity(self.verbose, second_level=f"\n")
        t1 = time()
        print_for_verbosity(self.verbose, second_level=f" {X.shape[0]} points assigned in {round(t1 - t0, 2)}s. \n")
        print_for_verbosity(self.verbose, first_level=f" {X.shape[0]} points assigned in {round(t1 - t0, 2)}s. \n", second_level="")
        return np.concatenate(batch_assignments)

    def clear(self):
        self.is_fit = False

class EfficientEncodingRouter(PCTreeBranchRouter):
    def __init__(self, pct: PCTree, verbose: int = 0):
        super().__init__(pct, verbose)

    def fit(self, X: NDArray):
        super().fit(X)

    def predict(self, X: NDArray):
        C = X @ self.pct.V.T

        n_data = X.shape[0]
        pct_branches = self.pct.branches()
        n_branches = len(pct_branches)
        max_branch_height = max(len(b) for b in pct_branches)
        branch_aucs = np.zeros((n_data, n_branches))
        for bi, branch_indices in enumerate(pct_branches):
            branch_height = len(branch_indices)
            branch_coeffs = C[:, branch_indices]
            assert branch_coeffs.shape[0] == X.shape[0]
            assert branch_coeffs.shape[1] == branch_height
            branch_coeffs = branch_coeffs ** 2
            branch_cumm_coeffs = branch_coeffs.cumsum(axis=1)
            total_var = branch_cumm_coeffs[:, -1]
            branch_coeffs_cumm_part_auc = np.sum(branch_cumm_coeffs, axis=1)
            already_captured_variance = total_var * (max_branch_height - branch_height)
            branch_aucs[:, bi] = branch_coeffs_cumm_part_auc + already_captured_variance
        return branch_aucs.argmax(axis=1)
    
    def clear(self):
        super().clear()


class SteepestBranchRouter(PCTreeBranchRouter):
    def __init__(self, pct: PCTree):
        super().__init__(pct)

    def fit(self, X: NDArray):
        super().fit(X)

    def predict(self, X: NDArray):
        C = X @ self.pct.V.T
        pct_branches = self.pct.branches()
        n_data = X.shape[0]
        n_branches = len(pct_branches)
        steepness = np.zeros((n_data, n_branches))
        for bi, branch_indices in enumerate(pct_branches):
            branch_height = len(branch_indices)
            branch_coeffs = C[:, branch_indices]
            branch_coeffs = branch_coeffs ** 2
            branch_varcap = branch_coeffs.sum(axis=1)
            branch_steepness = branch_varcap / branch_height
            steepness[:, bi] = branch_steepness
        return steepness.argmax(axis=1)
    
    def clear(self):
        super().clear()

def branch_assignments_to_node_data_proportions(pct: PCTree, branch_assignments: NDArray):
    n_data = len(branch_assignments)
    result = np.zeros(pct.N)
    branches = pct.branches()
    assert len(set(branch_assignments)) <= len(branches)
    for bi, br in enumerate(branches):
        n_branch = (branch_assignments == bi).sum()
        result[br] += n_branch
    return result / n_data


def branch_fill_tree(pct: PCTree, fill_target: int) -> PCTree:
    curr_pct = pct
    non_filled_branches = [(br_i, br) for br_i, br in enumerate(curr_pct.branches()) if len(br) < fill_target]

    while len(non_filled_branches) > 0:
        print(curr_pct.N, len(non_filled_branches))
        target_branch_i, target_branch = non_filled_branches[0]
        branch_tendril = curr_pct.branch_chords(target_branch_i)[-1]
        curr_pct = expand_tree_chord_with_zeros(curr_pct, branch_tendril, fill_target)
        non_filled_branches = [(br_i, br) for br_i, br in enumerate(curr_pct.branches()) if len(br) < fill_target]
        
    return curr_pct
    
class ReducedBranchRouter(PCTreeBranchRouter):
    reduced_branch_tree: PCTree
    reduced_tree_brancher: EfficientEncodingRouter
    def __init__(self, pct: PCTree, tendril_varcap: float = 0.9, verbose: int = 0):
        super().__init__(pct, verbose=verbose)
        reduced_tree_mask = TendrilVariancePruner(tendril_varcap, verbose).subtree(pct)
        self.reduced_branch_tree = pct.subtree(reduced_tree_mask)
        self.reduced_tree_brancher = EfficientEncodingRouter(self.reduced_branch_tree, verbose)

    def fit(self, X: NDArray):
        super().fit(X)

    def predict(self, X: NDArray):
        return self.reduced_tree_brancher.predict(X)
    
    def clear(self):
        super().clear()