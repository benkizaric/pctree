from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Self, Literal
from time import time
import math

from pctree.linalg_utils import *
from pctree.subspace_clustering import BinaryBuckshotSubspaceClusterer
from pctree.core import PCTree
from pctree.subspace_sifting import sift_subspace, SubspaceSiftResult, SubspaceSifterOptions
from pctree.logging_utils import print_for_verbosity
from pctree.pruning import PCTreePruner, EfficientEncodingPruner
from pctree.reweighting import PCTreeReweighter
from pctree.branches import ReducedBranchRouter, branch_assignments_to_node_data_proportions

class PCTNode_ExpansionResult:
    split: bool
    next: list[any] # PCTNode_Training
    finished: bool
    no_split_reason: str
    def __init__(self, split: bool, next: list[any], finished: bool = False, no_split_reason: str = "") -> None:
        self.split = split
        self.next = next
        self.finished = finished
        self.no_split_reason = no_split_reason    

class PCTNode_Training_Runtime:
    init_time: float
    test_time: float
    other_split_time: float
    cluster_compute_time: float
    cluster_shared_time: float
    cluster_nonshared_time: float
    cluster_svd_time: float
    def __init__(self) -> None:
        self.other_expand_time = None
        self.init_time = None
        self.test_time = None
        self.cluster_compute_time = None
        self.cluster_nonshared_time = None
        self.cluster_shared_time = None
        self.cluster_assign_time = None
        self.cluster_svd_time = None

class PCTreeNode_Training:
    ctx: PCTreeTrainer_Partitioning = None
    parent: any # PCTNode_Training
    children: list[any] # list[PCTNode_Training]

    opt: PCTreeTrainerOptions
    data_mask: NDArray
    N_data: int
    X_og: NDArray
    U: NDArray
    s: NDArray
    VT: NDArray
    max_test_dim: int
    rank: int
    abid_dim: int

    v: NDArray
    sigma: float
    rss: float

    compr_test_fuse: int

    runtime: PCTNode_Training_Runtime

    def __init__(self, ctx: PCTreeTrainer_Partitioning, parent: any, 
                 X_og: NDArray, options: PCTreeTrainerOptions, data_mask: NDArray, 
                 U: NDArray, s: NDArray, VT: NDArray, 
                 any_more_splits: bool = True, compr_test_fuse: int = None) -> None:
        t0 = time()
        self.ctx = ctx
        self.runtime = PCTNode_Training_Runtime()
        self.parent = parent
        self.children = []
        self.opt = options
        self.data_mask = data_mask
        self.N_data = data_mask.sum()
        self.X_og = X_og
        assert (s > 0).all()
        assert VT.shape[0] == s.shape[0]
        assert data_mask.shape[0] == X_og.shape[0]
        assert VT.shape[1] == X_og.shape[1]
        assert self.N_data >= s.shape[0]
        assert U.shape[1] == VT.shape[0]
        self.rank = s.shape[0]
        self.s = s
        self.sigma = s[0]
        self.rank = (self.s > 0).sum()
        self.VT = VT
        self.v = VT[0]
        self.U = U
        if not any_more_splits:
            self.runtime.init_time = time() - t0
            return
        
        self.max_test_dim = min(self.opt.subspace_sifter_max_dim, self.rank)
        self.rss = (s[1:] ** 2).sum()
        self.runtime.init_time = time() - t0

        # handle fuse...
        self.compr_test_fuse = compr_test_fuse
        # self._verify_passed_U_only_if_running_fuse(U)

    def _verify_passed_U_only_if_running_fuse(self, U: NDArray = None):
        if U is None:
            self.U = None
            return
        assert U.shape[1] == self.s.shape[0] == self.VT.shape[0], f"Something wrong with {(U.shape[1], self.s.shape[0], self.VT.shape[0])}"
        assert (self.compr_test_fuse is not None) and (self.compr_test_fuse >= 0), f"Something wrong with {self.compr_test_fuse}"
        self.U = U

    def _calculate_U_abid(self):
        if self.U is None:
            self.U = my_svd_U_given_s_VT_pos_eig(self.ctx.X[self.data_mask], self.s, self.VT)
        
        marginal_abid = math.ceil(self._calculate_abid_dim(self.U * self.s))
        self.abid_dim = marginal_abid

    def _calculate_abid_dim(self, X_og: NDArray) -> float:
        if self.rank == 1:
            return 1
        n_points_for_abid = min(self.N_data, min(500, 2*self.rank))
        return abid_dim_estimation(X_og[random_mask(self.N_data, n_points_for_abid)])
        
    def _common_singular_values_for_clustering_comparison(self) -> NDArray:
        s1 = self.s[:-1]
        s1 = s1 / s1.max()
        s2 = self.s[1:]
        s2 = s2 / s2.max()
        return (s1 + s2) / 2

    def cluster_expand(self, budget: int) -> PCTNode_ExpansionResult:
        MAX_CLUSTER_DIM = 150
        t0 = time()

        max_dim_of_ss_clusters = min(self.abid_dim, budget, MAX_CLUSTER_DIM)
        clustering_dim = min(self.rank, 2*self.abid_dim, MAX_CLUSTER_DIM)
        X_cluster = self.U[:, :clustering_dim] * self.s[:clustering_dim]
        kss_initial = BinaryBuckshotSubspaceClusterer(max_dim_of_ss_clusters, 20)
        kss_initial.fit(X_cluster)

        t1 = time()
        self.runtime.cluster_compute_time = t1 - t0

        ass_split = kss_initial.predict(X_cluster)
        
        # =============================================================================================
        
        unnormalized_U = self.U * self.s
        UX0 = unnormalized_U[ass_split == 0]
        UX1 = unnormalized_U[ass_split == 1]

        tb = time()
        full_U_cov = np.diag(self.s ** 2)
        (U0, s0, VT0), (U1, s1, VT1) = efficient_split_svd(full_U_cov, UX0, UX1)

        tc = time()
        svd_time = tc - tb
        pred_time = tb - t1


        s0a, VT0_ambient = project_vt_up(self.VT, VT0, s0)
        s1a, VT1_ambient = project_vt_up(self.VT, VT1, s1)
        mask0 = apply_sequential_masks(self.data_mask, ass_split == 0)
        mask1 = apply_sequential_masks(self.data_mask, ass_split == 1)

        effective_rank_0 = rank_to_explain_variance_given_s(s0a, self.opt.variance_capture_percent)
        U0b, s0b, VT0_ambient = as_low_dimentions(U0, s0a, VT0, effective_rank_0)
        effective_rank_1 = rank_to_explain_variance_given_s(s1a, self.opt.variance_capture_percent)
        U1b, s1b, VT1_ambient = as_low_dimentions(U1, s1a, VT1, effective_rank_1)

        s0c, VT0_ambient = project_vt_up(self.VT, VT0_ambient, s0b)
        s1c, VT1_ambient = project_vt_up(self.VT, VT1_ambient, s1b)

        splits = [PCTreeNode_Training(self.ctx, self.parent, self.X_og, self.opt, mask0, U0b, s0c, VT0_ambient), PCTreeNode_Training(self.ctx, self.parent, self.X_og, self.opt, mask1, U1b, s1c, VT1_ambient)]
        n0 = (ass_split == 0).sum()
        n1 = (ass_split == 1).sum()
        if self.parent is not None:
            self.parent.children.remove(self)
            self.parent.children.extend(splits)

        self.runtime.cluster_nonshared_time = time() - t1

        print_for_verbosity(self.ctx.verbose, second_level=f"[Split Info]   Dim:{self.U.shape[1]} -> {s0c.shape[0], s1c.shape[0]} (Alt, {rank_to_explain_variance_given_s(self.s, self.opt.variance_capture_percent)}).   N: {X_cluster.shape[0]} -> {n0, n1}.  TimeCluster: {round(t1 - t0, 3)}, TimeNonCluster: {round(self.runtime.cluster_nonshared_time, 3)}.    ABID Dim {self.abid_dim} \n")
        return PCTNode_ExpansionResult(True, splits)
    
    def inclusive_siblings(self) -> list:
        if self.parent is not None:
            return self.parent.children
        return [n for n in self.ctx.node_set if n.parent is None]
    
    def height(self) -> int:
        if self.parent is None:
            return 0
        return self.parent.height() + 1
    
    def allowed_to_split_becasue_of_min_chord_length(self) -> int:
        if self.ctx.options.min_chord_length is None:
            return True
        if self.parent is None:
            return True
        anscestors = []
        curr = self
        while curr.parent is not None:
            anscestors.append(curr.parent)
            curr = curr.parent
        anscestors_have_multiple_children = np.array([len(ansc.children) > 1 for ansc in anscestors])
        if anscestors_have_multiple_children.sum() == 0:
            return True
        return anscestors_have_multiple_children[:self.ctx.options.min_chord_length].sum() == 0
    
    def total_tree_width(self) -> int:
        return len(self.ctx.leaves)
    
    def expand_no_more_splits(self) -> any:
        U_next, s_next, VT_next = self.U[:, 1:], self.s[1:], self.VT[1:]
        result = PCTreeNode_Training(self.ctx, self, self.X_og, self.opt, self.data_mask, U_next, s_next, VT_next, any_more_splits=False)
        self.children = [result]
        self.kms()
        
        return result
    
    def run_compr_test(self) -> SubspaceSiftResult:
        return sift_subspace(self.U, self.s, 500, 8)

    def kms(self):
        self.U = None
        #self.VT = None
        #self.data_mask = None
        del self.U
        #del self.VT
        #del self.data_mask

    def expand(self, budget: int) -> PCTNode_ExpansionResult: # list of selfz
        t0 = time()
        self._calculate_U_abid()

        if self.rank == 1:
            return PCTNode_ExpansionResult(False, [], finished=True, no_split_reason="Rank1")
        splittable_height = self.height() % self.ctx.options.split_on_levels == 0
        reached_max_siblings = len(self.inclusive_siblings()) >= self.opt.max_children if self.opt.max_children is not None else False
        reached_max_width = self.total_tree_width() >= self.opt.max_width if self.opt.max_width is not None else False
        under_min_chord_length = self.allowed_to_split_becasue_of_min_chord_length()
        
        
        # Handle "deeper comparison fuse": If a previous comparison test has already told us that we must "go deeper" into the data
        # before splitting, then there's some things we can bypass. 
        compr_fuse_running = (self.compr_test_fuse is not None) and (self.compr_test_fuse > 0)
        next_compr_fuse = None if not compr_fuse_running else self.compr_test_fuse - 1
        compr_fuse_expired = (self.compr_test_fuse is not None) and (self.compr_test_fuse == 0)
        if next_compr_fuse is not None:
            U_next, s_next, VT_next = self.U[:, 1:], self.s[1:], self.VT[1:]
            next_node = PCTreeNode_Training(self.ctx, self, self.X_og, self.opt, self.data_mask, U_next, s_next, VT_next, compr_test_fuse=next_compr_fuse)
            self.children = [next_node]
            result = PCTNode_ExpansionResult(False, [next_node], no_split_reason=f"BreatheDeeper: Fuse= {next_compr_fuse}.")
            self.kms()
            return result

        t0, t1 = 0, 0
        other_bypass = reached_max_siblings or reached_max_width or not splittable_height or not under_min_chord_length or compr_fuse_running

        if compr_fuse_expired and not other_bypass:
            result = self.cluster_expand(budget)
            del self.U
            return result

        clustering_detected = "N/A"
        if not other_bypass:
            ta = time()
            used_test_dim = round(min(self.abid_dim * 1.5, self.max_test_dim))
            test_result = sift_subspace(self.U[:, :used_test_dim], self.s[:used_test_dim], self.opt.subspace_sifter_options)
            clustering_detected = test_result.primary_significant
            split_now_test = test_result.primary_significant and test_result.intersecting_dimenstions == 0
            next_compr_fuse = None if (not test_result.primary_significant or split_now_test) else test_result.intersecting_dimenstions - 1

            tb = time()
            print_for_verbosity(self.ctx.verbose, second_level=f"Subspace Test In: {round(tb - ta, 5)} sec \n")

        self.runtime.test_time = time() - t0
        override = False
        if override or (not clustering_detected) or (next_compr_fuse is not None) or (budget == 1) or reached_max_siblings or reached_max_width or not splittable_height or not under_min_chord_length:
            reason = ""
            if override:
                reason = override
            if next_compr_fuse is not None:
                reason += f"BreatheDeeper: Fuse= {next_compr_fuse}, "
            if not clustering_detected:
                reason += "UniformAngles, "
            if budget == 1:
                reason += "OutOfBudget, "
            if reached_max_siblings:
                reason += "MaxSiblings, "
            if not splittable_height:
                reason += "NotSplittableHeight, "
            if reached_max_width:
                reason += "MaxWidthReached, "
            if not under_min_chord_length:
                reason += "MinChordLength, "
            U_next, s_next, VT_next = self.U[:, 1:], self.s[1:], self.VT[1:]
            next_node = PCTreeNode_Training(self.ctx, self, self.X_og, self.opt, self.data_mask, U_next, s_next, VT_next, compr_test_fuse=next_compr_fuse)
            self.children = [next_node]
            result = PCTNode_ExpansionResult(False, [next_node], no_split_reason=reason)
        else:
            result = self.cluster_expand(budget)
        self.kms()
        
        return result

class PCTreeTrainingBudgeter:
    subtree_selector: PCTreePruner
    build_leaf_tree_time: float
    budget_many_time: float
    budget_single_time: float

    def __init__(self, subtree_selector: PCTreePruner) -> None:
        self.subtree_selector = subtree_selector
        self.build_leaf_tree_time = 0
        self.budget_many_time = 0
        self.budget_single_time = 0

    def _build_leaf_tree(self, leaf_nodes: list[PCTreeNode_Training]) -> tuple[PCTree, NDArray, dict[PCTreeNode_Training, int]]:
        # 1. build a non-hierarchichal tree out of the leaves
        # A couple things to note: 
        #   each leaf node can have it'e own dimentionality it's working in, but all we care about are the singular values?
        #   It's assumed that the first singular value / vector of each leaf is already captured, moe onto the remaining ones.
        total_nodes: int = sum([ln.VT[1:].shape[0] for ln in leaf_nodes])
        fake_ambient_dim = 10
        V = np.zeros((total_nodes, fake_ambient_dim))
        s = np.zeros(total_nodes)
        #E = np.zeros(total_nodes, dtype=int)
        E = np.arange(total_nodes) - 1

        data_prop = np.zeros(total_nodes)
        gbl_node_idx = 0
        leaf_node_temp_pct_masks: dict[PCTreeNode_Training, NDArray] = {}
        for ln in leaf_nodes:
            ln_size = ln.VT[1:].shape[0]
            s[gbl_node_idx:gbl_node_idx+ln_size] = ln.s[1:]
            data_prop[gbl_node_idx:gbl_node_idx+ln_size] = ln.N_data
            E[gbl_node_idx] = PCTree.ROOT_FLAG
            leaf_node_temp_pct_masks[ln] = np.zeros(total_nodes, dtype=bool)
            leaf_node_temp_pct_masks[ln][gbl_node_idx:gbl_node_idx+ln_size] = True
            
            gbl_node_idx += ln_size

        data_prop = data_prop / data_prop.max() # I don't think having this not literally be scaled correctly will matter? It's all relative lol.
        leaf_tree = PCTree(V, E, s)
        return leaf_tree, data_prop, leaf_node_temp_pct_masks
    
    def _budget_from_tree(self, leaf_tree: PCTree, data_proportions: NDArray, leaf_pct_masks: dict[PCTreeNode_Training, int], total_budget: int) -> dict[PCTreeNode_Training, int]:
        # select the optimal subtree with some algorithm
        subtree_mask, subtree_order = self.subtree_selector.subtree(total_budget, leaf_tree, data_proportions)

        # return how many descendents are in each (disjoint) branch of the optimal subtree
        result: dict[PCTreeNode_Training, int] = {}
        for ln in leaf_pct_masks.keys():
            budgey = subtree_mask[leaf_pct_masks[ln]].sum()
            result[ln] = budgey
        return result
    
    def _budget_next_from_tree(self, leaf_tree: PCTree, data_proportions: NDArray, leaf_pct_masks: dict[PCTreeNode_Training, int], total_budget: int) -> dict[PCTreeNode_Training, int]:
        # select the optimal subtree with some algorithm
        subtree_mask, subtree_order = self.subtree_selector.subtree(total_budget, leaf_tree, data_proportions)
        next_node = subtree_order[0]
        for ln, leaf_tree_mask in leaf_pct_masks.items():
            if leaf_tree_mask[next_node]:
                return ln, subtree_mask[leaf_tree_mask].sum()

    def budget(self, leaf_nodes: list[PCTreeNode_Training], total_budget: int) -> dict[PCTreeNode_Training, int]:
        leaf_tree, data_prop, leaf_masks = self._build_leaf_tree(leaf_nodes)
        return self._budget_from_tree(leaf_tree, data_prop, leaf_masks, total_budget)
    
    def next_node(self, leaf_nodes: list[PCTreeNode_Training], total_budget: int) -> tuple[PCTreeNode_Training, int]:
        t0 = time()
        leaf_tree, data_prop, leaf_masks = self._build_leaf_tree(leaf_nodes)
        t1 = time()
        next_leaf_new, next_leaf_budget = self._budget_next_from_tree(leaf_tree, data_prop, leaf_masks, min(total_budget, leaf_tree.N))
        t2 = time()
        self.build_leaf_tree_time += t1 - t0
        self.budget_many_time += t2 - t1
        return next_leaf_new, next_leaf_budget

class PCTreeTrainerOptions:
    max_nodes: int
    max_children: int | None
    max_width: int | None
    variance_capture_percent: float
    subspace_sifter_max_dim: int
    pruner: PCTreePruner
    budgeter: PCTreeTrainingBudgeter
    split_on_levels: int
    min_chord_length: int
    subspace_sifter_options: SubspaceSifterOptions
    
    def __init__(self, 
            max_nodes: int,
            max_width: int,
            sifting_effect_size: float = 0.5,
            pruner: PCTreePruner = EfficientEncodingPruner(),
            variance_capture_percent: float = 0.999,
            subspace_sifter_max_dim: int = 50,
            max_children: int | None = None,
            split_on_levels: int = 1,
            min_chord_length: int = None,
            subspace_sifter_options: SubspaceSifterOptions = None
            ) -> None:
        self.max_nodes = max_nodes
        self.max_width = max_width
        self.max_children = max_children
        self.pruner = pruner
        self.budgeter = PCTreeTrainingBudgeter(pruner)
        self.variance_capture_percent = variance_capture_percent
        self.subspace_sifter_max_dim = subspace_sifter_max_dim
        self.subspace_sifter_options = subspace_sifter_options
        self.split_on_levels = split_on_levels
        self.min_chord_length = min_chord_length
        if subspace_sifter_options is None:
            self.subspace_sifter_options = SubspaceSifterOptions(sifting_effect_size)
    
class PCTreeTrainer_Partitioning:
    X: NDArray
    n_data: int
    ambient_dim: int
    init_time: float
    chose_next_time: float
    options: PCTreeTrainerOptions
    leaves: list[PCTreeNode_Training]
    node_set: set[PCTreeNode_Training]
    verbose: int

    def __init__(self, X: NDArray, training_options) -> None:
        self.X = X
        self.n_data, self.ambient_dim = X.shape
        self.options = training_options
        self.node_set = set()
        self.leaves = []
        self.init_time = 0
        self.chose_next_time = 0
        self.verbose = 0
        
    def fit(self, verbose = 0):
        self.verbose = verbose
        t0 = time()
        U, s, VT = my_full_svd(self.X)
        effective_rank = rank_to_explain_variance_given_s(s, self.options.variance_capture_percent)
        U, s, VT = as_low_dimentions(U, s, VT, effective_rank)
        t1 = time()
        self.init_time += t1 - t0
        initial_root = PCTreeNode_Training(self, None, self.X, self.options, np.ones(self.n_data, dtype=bool), U, s, VT)
        self.leaves.append(initial_root)
        self.node_set = set([initial_root])
        self.all_leaves_ever = []
        
        print_for_verbosity(self.verbose, "Performing top-down partitioning to create initial PCT. \n")
        print_for_verbosity(self.verbose, first_level="Log Key: `-` is a node expanded by adding one additional child. `<` is a node expanded by spliting into two nodes. \n", second_level="")
        still_training_with_leaves = True
        while still_training_with_leaves:
            still_training_with_leaves = self.expand_next()
        print_for_verbosity(self.verbose, first_level="\n", second_level="")
        

    def roots(self) -> list[PCTreeNode_Training]:
        return [n for n in self.node_set if n.parent is None]
    
    def next_node_to_expand(self, remaining_budget: int) -> tuple[PCTreeNode_Training, int]:
        next_node, next_node_budget = self.options.budgeter.next_node(self.leaves, remaining_budget)
        return next_node, next_node_budget

    def expand_next(self) -> bool:
        remaining_budget = self.options.max_nodes - len(self.node_set)
        if remaining_budget < 1:
            return False
        if self.options.max_width is not None and len(self.leaves) >= self.options.max_width:
            return False
        t0 = time()
        next_node, next_node_budget = self.next_node_to_expand(remaining_budget)
        self.chose_next_time += time() - t0
        self.leaves.remove(next_node)
        self.all_leaves_ever.append(next_node)
        next_expansion = next_node.expand(next_node_budget)
        second_level_log = "[Split Expansion] \n" if next_expansion.split else f"[Non-Split Expansion] ({next_expansion.no_split_reason}) \n"
        print_for_verbosity(self.verbose, "<" if next_expansion.split else "-", second_level=second_level_log)
        if next_expansion.finished:
            return len(self.node_set) <= self.options.max_nodes
        if next_expansion.split:
            self.node_set.remove(next_node)
        for new_leaf in next_expansion.next:
            if new_leaf.rss > 0:
                self.leaves.append(new_leaf)
            self.node_set.add(new_leaf)
        return len(self.node_set) <= self.options.max_nodes

class PCTreeTrainer:
    options: PCTreeTrainerOptions
    root_nodes: list[PCTreeNode_Training]
    _n_partition_data: int
    _dim: int 
    def __init__(self, trainer_options: PCTreeTrainerOptions) -> None:
        self.options = trainer_options
        
    def fit_partition(self, X: NDArray, verbose: int = 0):
        self._n_partition_data = X.shape[0]
        self._dim = X.shape[1]
        trainer_partitioning = PCTreeTrainer_Partitioning(X, self.options)
        trainer_partitioning.fit(verbose)
        self.root_nodes = trainer_partitioning.roots()

    def fit_em(self, X: NDArray, X_topdown: NDArray = None, n_iters: int = 1, flex_tendrils: bool = True, tendril_varcap = 0.95, verbose: int = 0) -> PCTree:
        tree: PCTree = None
        reweighter = PCTreeReweighter(
            branch_router_gen=lambda tr: ReducedBranchRouter(tr, tendril_varcap=tendril_varcap, verbose=verbose),
            flex_tendrils=False,
            varcap=self.options.variance_capture_percent,
            target_size=self.options.max_nodes,
            verbose=verbose
        )

        tree_full, topdown_training_branch_assignments, _ = self.to_pctree(extend_leaves=True, verbose=verbose)
        if flex_tendrils:
            tree = reweighter.reweight_single_flex_given_big_tree(tree_full, X_topdown, topdown_training_branch_assignments)
        else:
            node_data_proportion = branch_assignments_to_node_data_proportions(tree_full, topdown_training_branch_assignments)
            selected_mask, _ = self.options.pruner.subtree(self.options.max_nodes, tree_full, node_data_proportion)
            tree = tree_full.subtree(selected_mask)

        tree = reweighter.reweight(tree, X, n_iter=n_iters)
        return tree  
    
    def to_pctree(self, extend_leaves: bool = False, verbose: int = 0) -> tuple[PCTree, NDArray, NDArray]:
        node_to_index: dict[PCTreeNode_Training, int] = {}
        stack = [n for n in self.root_nodes]
        E_list = []
        n_nodes_processed = 0
        leaf_node_index_to_selected_data_inds: dict[int, NDArray] = {}

        print_for_verbosity(verbose, second_level="Log Key: Each | is a training node being added to a standard PCT. \n")

        while len(stack) > 0:
            curr = stack.pop()
            curr_is_a_root = curr.parent is None
            curr_is_a_leaf = curr.children is None or len(curr.children) == 0
            curr_extend_leaf = curr_is_a_leaf and extend_leaves
            print_for_verbosity(verbose, second_level="|")

            if n_nodes_processed == 0: # handle very first node... ug, what if it's a leaf
                V = curr.v.reshape((1, self._dim))
                s = np.array([curr.sigma])
                E_list.append(PCTree.ROOT_FLAG)
                data_prop = np.array([curr.N_data / self._n_partition_data])
            else: # handle other nodes
                if curr_extend_leaf:
                    V = np.vstack([V, curr.VT])
                    s = np.concatenate([s, curr.s])
                    data_prop = np.concatenate([data_prop, [curr.N_data / self._n_partition_data] * curr.VT.shape[0]])
                    if not curr_is_a_root:
                        parent_index = node_to_index[curr.parent]
                        E_list.append(parent_index)
                    else:
                        E_list.append(PCTree.ROOT_FLAG)
                    E_list.extend(list(np.arange(n_nodes_processed, n_nodes_processed + curr.VT.shape[0]-1)))
                else:
                    V = np.vstack([V, curr.v])
                    s = np.concatenate([s, [curr.sigma]])
                    data_prop = np.concatenate([data_prop, [curr.N_data / self._n_partition_data]])
                    if not curr_is_a_root: 
                        parent_index = node_to_index[curr.parent]
                        E_list.append(parent_index)
                    else: # 
                        E_list.append(PCTree.ROOT_FLAG)

            node_to_index[curr] = n_nodes_processed
            if not curr_is_a_leaf:
                for n in curr.children:
                    stack.append(n)
                n_nodes_processed += 1
            else:
                if curr_extend_leaf:
                    n_nodes_processed += curr.VT.shape[0]
                    leaf_node_index_to_selected_data_inds[n_nodes_processed - 1] = np.where(curr.data_mask)[0]
                if not curr_extend_leaf:
                    leaf_node_index_to_selected_data_inds[n_nodes_processed] = np.where(curr.data_mask)[0]
                    n_nodes_processed += 1
                #n_nodes_processed += 1 if not curr_extend_leaf else curr.VT.shape[0]

        result_tree = PCTree(V, np.array(E_list, dtype=int), s)
        training_data_assignments = np.zeros(self._n_partition_data, dtype=int) - 1
        for br_i, br in enumerate(result_tree.branches()):
            branch_leaf_index = br[-1]
            training_data_assignments[leaf_node_index_to_selected_data_inds[branch_leaf_index]] = br_i
        return result_tree, training_data_assignments, data_prop
    
    def from_pca(X: NDArray, n_components: int):
        assert n_components <= X.shape[0]
        s, VT = my_svd_sigma_VT_given_X(X)
        e = np.arange(n_components) - 1
        return PCTree(VT[:n_components], e, s[:n_components])