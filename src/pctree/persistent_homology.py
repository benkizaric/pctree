import numpy as np
from numpy.typing import NDArray

from pctree.core import PCTree
from pctree.pruning import PCTreePruner, MostVariancePruner, TallestFirstPruner

class PCTreeHomology:
    pruner: PCTreePruner

    def __init__(self, pruner: PCTreePruner):
        self.pruner = pruner

    def maximum_curve_sum(self, tree_size: int) -> float:
        pass

    def maximum_curve_value(self, tree_size: int) -> float:
        pass

    def minumum_curve_sum(self, tree_size: int) -> float:
        pass

    def measure_single(self, tree: PCTree) -> float:
        pass

    def measure(self, tree: PCTree, standardize: bool = False) -> tuple[NDArray, float]:
        curr_tree = tree
        m = self.measure_single(curr_tree)
        curve = [m]
        for ni in range(1, tree.N-1):
            next_subtree_mask, _ = self.pruner.subtree(tree.N - ni, curr_tree, None)
            curr_tree = curr_tree.subtree(next_subtree_mask)
            m = self.measure_single(curr_tree)
            curve.append(m)
        curve = curve[::-1]
        curve_arr = np.array(curve, dtype=float)
        summary = np.sum(curve)
        if standardize:
            cumm_min, cumm_max = self.minumum_curve_sum(tree.N), self.maximum_curve_sum(tree.N)
            curve_arr /= self.maximum_curve_value(tree.N)
            summary -= cumm_min
            summary /= (cumm_max - cumm_min)
        return curve_arr, summary

class WidthFiltration(PCTreeHomology):
    def __init__(self):
        super().__init__(MostVariancePruner())
    
    def maximum_curve_sum(self, tree_size: int) -> float:
        n = tree_size
        return (((n-1)**2) / 2) + n
    
    def minumum_curve_sum(self, tree_size: int) -> float:
        n = tree_size
        return n

    def maximum_curve_value(self, tree_size: int) -> float:
        return float(tree_size)
    
    def measure_single(self, tree: PCTree) -> float:
        return float(len(tree.leaf_inds()))
    
class HeightFiltration(PCTreeHomology):
    def __init__(self):
        super().__init__(TallestFirstPruner())

    def maximum_curve_sum(self, tree_size: int) -> float:
        n = tree_size
        return ((n-1)**2) + n
    
    def minumum_curve_sum(self, tree_size: int) -> float:
        n = tree_size
        return (((n-1)**2) / 2) + n

    def maximum_curve_value(self, tree_size: int) -> float:
        return float(tree_size)
    
    def measure_single(self, tree: PCTree) -> float:
        return tree.N