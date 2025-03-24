import numpy as np
from numpy.typing import NDArray 
from typing import Callable
from time import time

from pctree.core import PCTree
from pctree.linalg_utils import my_cov, my_svd_sigma_VT_given_cov, rank_to_explain_variance_given_s
from pctree.branches import PCTreeBranchRouter, branch_fill_tree, branch_assignments_to_node_data_proportions
from pctree.pruning import PCTreePruner, EfficientEncodingPruner
from pctree.logging_utils import print_for_verbosity

def in_place_reweight(pct: PCTree, X: NDArray, branch_assignments: NDArray, varcap: float = 0.99, verbose: int = 0) -> PCTree:
    chord_covar: dict[int, NDArray] = {}
    branch_covar: dict[int, NDArray] = {}
    t0 = time()
    total_covar = np.zeros((pct.dim, pct.dim))
    print_for_verbosity(verbose, second_level=f"Fixed-shape reweighting / EM iteration of size-{pct.N}. Using {X.shape[0]} points. \n")
    print_for_verbosity(verbose, second_level="Log Key: 'Σ' logged each time a covariance matrix is calculated. 'λ' logged each time an eigendecomposition is performed. \n")
    print_for_verbosity(verbose, first_level="[Fixed-shape PCT Reweighting] ", second_level="")
    for br_i, branch in enumerate(pct.branches()):
        data_for_branch = X[branch_assignments == br_i]
        branch_covar[br_i] = my_cov(data_for_branch)
        total_covar += branch_covar[br_i]
        print_for_verbosity(verbose, "Σ")

    for chord_i, chord in enumerate(pct.chords()):
        chord_covar[chord_i] = np.zeros((pct.dim, pct.dim))
        for br_i in pct.chord_branches(chord_i):
            chord_covar[chord_i] += branch_covar[br_i]

    null_spans: dict[int, NDArray] = {}
    s_omega, VT_omega = my_svd_sigma_VT_given_cov(total_covar)
    sufficient_depth = rank_to_explain_variance_given_s(s_omega, varcap)
    
    VT_omega_span = VT_omega[:sufficient_depth]
    
    t1 = time()
    for chord_i, chord in enumerate(pct.chords()):
        chord_size = chord.stop - chord.start
        chord_parent = pct.chord_parent(chord_idx=chord_i)
        parent_span = VT_omega_span if chord_parent is None else null_spans[chord_parent]
        covar = chord_covar[chord_i]
        covar_on_span = parent_span @ covar @ parent_span.T
        s, VT_on_parent_span = my_svd_sigma_VT_given_cov(covar_on_span)
        VT_ambient = VT_on_parent_span @ parent_span
        try:
            effective_chord_size = s[:chord_size].shape[0]
            pct.s[chord.start:chord.start+effective_chord_size] = s[:chord_size]
            pct.V[chord.start:chord.start+effective_chord_size] = VT_ambient[:chord_size]
        except Exception as err:
            print(err)
        rank_to_varcap = rank_to_explain_variance_given_s(s[chord_size:], varcap)
        null_spans[chord_i] = VT_ambient[chord_size:chord_size + rank_to_varcap]
        print_for_verbosity(verbose, "λ")
    t2 = time()
    print_for_verbosity(verbose, first_level=f" Reweighting using {X.shape[0]} points in {round(t2 - t0, 2)}s. \n", second_level="")
    print_for_verbosity(verbose, second_level=f" Covariance matrix Time: {round(t1 - t0, 2)}s. Eigendecompostion Time: {round(t2 - t1, 2)}s \n")
    return pct


class PCTreeReweighter:
    branch_router_gen: Callable[[PCTree], PCTreeBranchRouter]
    flex_tendrils: bool
    varcap: float
    target_size: int
    verbose: int

    flex_tendrils_base_size: int = None
    def __init__(self, branch_router_gen: Callable[[PCTree], PCTreeBranchRouter], 
                 flex_tendrils: bool = False, 
                 varcap: float = None,
                 target_size: int = None,
                 verbose: int = 0) -> None:
        self.branch_router_gen = branch_router_gen
        self.flex_tendrils = flex_tendrils
        self.varcap = varcap
        self.target_size = target_size
        self.verbose = verbose

    def reweight_single_flex_given_big_tree(self, pct: PCTree, X: NDArray, branch_assignments: NDArray) -> PCTree:
        result_pct = pct

        use_assignments = branch_assignments
        extended_assigner = self.branch_router_gen(result_pct)
        use_assignments = extended_assigner.predict_batches(X)

        result_pct = in_place_reweight(result_pct, X, use_assignments, self.varcap, self.verbose)

        selec = EfficientEncodingPruner()
        use_treesize = pct.N if self.target_size is None else self.target_size # should only be used in conjunction with flex_tentrils
        subtree_mask, _ = selec.subtree(use_treesize, result_pct, branch_assignments_to_node_data_proportions(result_pct, use_assignments))
        subtree_mask_clip = PCTreePruner.modify_subtree_selection_to_require_all_branches(result_pct, subtree_mask)
        result_pct = result_pct.subtree(subtree_mask_clip)
        return result_pct

    def reweight_single(self, pct: PCTree, X: NDArray, branch_assignments: NDArray) -> PCTree:
        result_pct = PCTree(
            V = np.zeros(pct.V.shape),
            E = pct.E, 
            s = np.zeros(pct.s.shape))
        
        #no need to made it excessively large.
        if self.flex_tendrils:
            result_pct = branch_fill_tree(result_pct, result_pct.dim)
        
        result_pct = in_place_reweight(result_pct, X, branch_assignments, self.varcap)

        if self.flex_tendrils:
            extended_assigner = self.branch_router_gen(result_pct)
            use_assignments = extended_assigner.predict_batches(X)
            result_pct = in_place_reweight(result_pct, X, use_assignments, self.varcap)

            selec = EfficientEncodingPruner()
            use_treesize = pct.N if self.target_size is None else self.target_size # should only be used in conjunction with flex_tentrils
            subtree_mask, _ = selec.subtree(use_treesize, result_pct, branch_assignments_to_node_data_proportions(result_pct, use_assignments))
            subtree_mask_clip = PCTreePruner.modify_subtree_selection_to_require_all_branches(result_pct, subtree_mask)
            result_pct = result_pct.subtree(subtree_mask_clip)
        return result_pct


    def reweight(self, pct: PCTree, X: NDArray, n_iter: int = 1) -> PCTree:
        curr_tree = pct
        curr_assigner = self.branch_router_gen(pct)
        curr_assigner.fit(X)
        curr_assignment = curr_assigner.predict_batches(X)

        for i in range(n_iter):
            next_tree = self.reweight_single(curr_tree, X, curr_assignment)
            curr_assigner = self.branch_router_gen(next_tree)
            curr_assigner.fit(X)
            if i == n_iter - 1:
                next_assignment = curr_assigner.predict_batches(X)
                curr_assignment = next_assignment
            curr_tree = next_tree

        return curr_tree
