import numpy as np

from pctree.core import PCTree

def load_pctree(npz_path: str) -> PCTree:
    npz = np.load(npz_path)
    tree_V = npz["tree_V"]
    tree_s = npz["tree_s"]
    tree_E = npz["tree_E"]
    return PCTree(tree_V, tree_E, tree_s)

def save_pctree(tree: PCTree, npz_path: str):
    arrays = {}
    arrays["tree_V"] = tree.V
    arrays["tree_s"] = tree.s
    arrays["tree_E"] = tree.E
    np.savez(npz_path, **arrays)