import numpy as np
from numpy.typing import NDArray

from pctree.core import PCTree
from pctree.linalg_utils import rank_to_explain_variance_given_s
from pctree.logging_utils import print_for_verbosity

class PCTreePruner():
    def __init__(self) -> None:
        pass

    def subtree(self, size: int, pct: PCTree, data_proportions: NDArray) -> tuple[NDArray, NDArray]:
        # return the actual subtree, the subtree mask, and the order of nodes
        pass

    def modify_subtree_selection_to_require_all_branches(
            pct: PCTree, initial_subtree_mask: NDArray) -> NDArray:
        subtree_size = initial_subtree_mask.sum()
        # every tentril has to have at least 1 node.
        required_mask = np.zeros(pct.N, dtype=bool)
        for chord_i, chord in enumerate(pct.chords()):
            is_tentril = len(pct.chord_branches(chord_i)) == 1
            if is_tentril:
                required_mask[chord.start] = True
            else:
                required_mask[chord] = True
        n_required = required_mask.sum()
        if (n_required > subtree_size):
            raise "Impossible to fufill!"
        intersection = required_mask * initial_subtree_mask
        if intersection.sum() == n_required:
            return initial_subtree_mask # no work needed.
        # if we are here, then we failed to include som required nodes.
        result_mask = initial_subtree_mask.copy()

        required_nodes_we_didnt_select = required_mask * ~initial_subtree_mask
        n_missed_required_nodes = required_nodes_we_didnt_select.sum()
        non_required_nodes_we_selected = initial_subtree_mask * ~required_mask
        s_non_required_nodes_we_selected = pct.s.copy()
        s_non_required_nodes_we_selected[~non_required_nodes_we_selected] = pct.s.max() * 10 # so that the sort will only select what we want
        non_required_nodes_we_selected_to_drop = s_non_required_nodes_we_selected.argsort()[:n_missed_required_nodes]

        result_mask[required_mask] = True
        result_mask[initial_subtree_mask] = True
        result_mask[non_required_nodes_we_selected_to_drop] = False
        assert result_mask.sum() == subtree_size
        return result_mask

class EfficientEncodingPruner(PCTreePruner):
    def __init__(self) -> None:
        pass

    def encoding_auc(self, pct: PCTree, data_proportions: NDArray, selected_subtree: NDArray, nodes_to_inspect: NDArray):
        var_cap = (pct.s ** 2).copy()
        max_data_prop = data_proportions[nodes_to_inspect].max()
        var_cap[~nodes_to_inspect] = 0
        triangle_part = var_cap * data_proportions / 2
        remaining_part = var_cap * (max_data_prop - data_proportions)
        encoding_auc = triangle_part + remaining_part
        encoding_auc[~nodes_to_inspect] = 0
        encoding_auc[selected_subtree] = 0
        return encoding_auc

    def subtree(self, size: int, pct: PCTree, data_proportions: NDArray) -> tuple[NDArray, NDArray]:
        # return the actual subtree, the subtree mask, and the order of nodes
        # reapeast until complete...
        # get the unadded nodes.
        # consider the additionl variance, and the proportions of each
        # pick the one that gives the best tradeoff.
        if size >= pct.N:
            return np.ones(pct.N, dtype=bool), np.argsort(pct.s)[::-1]
        
        selected_subtree = np.zeros(pct.s.shape, dtype=bool)
        curr_nodes_to_inspect = pct.root_mask()
        selected_node_order = []
        while(selected_subtree.sum() < size):
            encoding_auc_of_remaining = self.encoding_auc(pct, data_proportions, selected_subtree, curr_nodes_to_inspect)
            next_node_to_pick = encoding_auc_of_remaining.argmax()
            selected_node_order.append(next_node_to_pick)
            selected_subtree[next_node_to_pick] = True                # Mark the node as chosen
            curr_nodes_to_inspect[next_node_to_pick] = False          # Remove it from "frontier"
            curr_nodes_to_inspect[pct.child_mask(next_node_to_pick)] = True  # Add it's children to be inspected next
        return selected_subtree, np.array(selected_node_order)
        
class MostVariancePruner(PCTreePruner):
    def __init__(self) -> None:
        pass

    def subtree(self, size: int, pct: PCTree, data_proportions: NDArray) ->  tuple[NDArray, NDArray]:
        # reapeast until complete...
        # get the unadded nodes.
        # consider the additionl variance, and the proportions of each
        # pick the one that gives the best tradeoff.
        if size >= pct.N:
            return np.ones(pct.N, dtype=bool), pct.s.argsort()[::-1]
        
        selected_subtree = np.zeros(pct.s.shape, dtype=bool)
        node_order = pct.s.argsort()[::-1][:size]
        selected_subtree[node_order] = True
        return selected_subtree, node_order
    
# removes all nodes greater than size, primary use is in persistent homology measurements.
class TallestFirstPruner(PCTreePruner):
    def __init__(self):
        super().__init__()

    def subtree(self, size: int, pct: PCTree, data_proportions: NDArray) ->  tuple[NDArray, NDArray]:
        node_heights = np.zeros(pct.N)
        for ni in range(pct.N):
            node_heights[ni] = len(pct._path_to_root(ni))
        node_order = np.argsort(node_heights)
        selected_subtree = node_heights <= size
        return selected_subtree, node_order
                
    
# doesnt inherit from base because it is not trying to select a specific size. 
class TendrilVariancePruner:
    verbose: int
    tendril_varcap: float
    def __init__(self, tendril_varcap: float = 0.9, verbose: int = 0) -> None:
        self.verbose = verbose
        self.tendril_varcap = tendril_varcap

    def subtree(self, pct: PCTree) ->  tuple[NDArray, NDArray]:
        internal_mask = np.zeros(pct.N, dtype=bool)
        selected_tendril_mask = np.zeros(pct.N, dtype=bool)
        for br_i, br in enumerate(pct.branches()):
            internal_chords = [pct.chords()[ci] for ci in pct.branch_chords(br_i)[:-1]]
            for ic in internal_chords:
                internal_mask[ic] = True
            tendril_i = pct.branch_chords(br_i)[-1]
            tendril = pct.chords()[tendril_i]
            tendril_cap_dim = rank_to_explain_variance_given_s(pct.s[tendril], self.tendril_varcap)
            selected_tendril_mask[tendril.start:tendril.start+tendril_cap_dim] = True
        if self.verbose > 0:
            total_variance = (pct.s**2).sum()
            n_internal_nodes = internal_mask.sum()
            n_selected_tendril_nodes = selected_tendril_mask.sum()
            internal_node_variance = (pct.s[internal_mask]**2).sum()
            selected_tendril_variance = (pct.s[selected_tendril_mask]**2).sum()
            debug_string = f"[Selecting Tentril Subtree] For 'Full' PCT with {pct.N} nodes, {n_internal_nodes} are internal nodes. (capturing {round((internal_node_variance / total_variance) * 100, 3)}% of variance). Selected a subtree with those internal nodes & more. Size {n_internal_nodes + n_selected_tendril_nodes} nodes ({round(((internal_node_variance+selected_tendril_variance) / total_variance) * 100, 3)}% variance). \n"
            print_for_verbosity(self.verbose, second_level=debug_string)
        return internal_mask + selected_tendril_mask