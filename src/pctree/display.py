import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass

from pctree.core import PCTree

def num_descendant_leaves(pct: PCTree) -> dict[int, int]:
    result = {}
    branches = pct.branches()
    for ni in range(pct.N):
        result[ni] = len([br for br in branches if ni in br])
    return result


def node_plot_ranges_recursive(pct: PCTree, node_widths: dict[int, int], result: dict[int, tuple[int, int]], proportion_filled: dict[int, float], node_i: int):
    total_width = node_widths[node_i]

    is_root = node_i in pct.root_inds()
    if is_root:
        parent_span = (0, 1.0)
        parent_relative_width = total_width / len(pct.branches())
        prop_parent_span_already_filled = proportion_filled[-1]
    else:
        par_i = pct.parent(node_i)
        parent_span = result[par_i]
        parent_relative_width = total_width / node_widths[par_i]
        prop_parent_span_already_filled = proportion_filled[par_i]
    parent_span_size = parent_span[-1] - parent_span[0]
    my_span_start = parent_span[0] + (parent_span_size * prop_parent_span_already_filled)
    my_span_end = my_span_start + (parent_span_size * parent_relative_width)
    result[node_i] = (my_span_start, my_span_end)
    proportion_filled[node_i] = 0.0
    if is_root:
        proportion_filled[-1] += parent_relative_width
    else:
        par_i = pct.parent(node_i)
        proportion_filled[par_i] += parent_relative_width
    is_leaf = node_i in pct.leaf_inds()
    if is_leaf:
        return
    else:
        for ni in pct.child_inds(node_i):
            node_plot_ranges_recursive(pct, node_widths, result, proportion_filled, ni)

@dataclass
class PCT_DisplayInfo():
    n_nodes: int
    n_edges: int
    node_coords: NDArray
    edge_coords: NDArray
    node_colors: NDArray
    edge_colors: NDArray

def get_pct_display_info(pct: PCTree, sqrt_weight: bool = False) -> PCT_DisplayInfo:
    node_widths = num_descendant_leaves(pct)
    node_ranges_dict = {}
    proportion_filled = {-1: 0}
    for ni in pct.root_inds():
        node_plot_ranges_recursive(pct, node_widths, node_ranges_dict, proportion_filled, ni)

    opacity = pct.s.copy()
    if sqrt_weight:
        opacity = np.sqrt(opacity)
    opacity /= opacity.max()
    opacity[opacity < 0.1] = 0.1
    opacity = 1-opacity

    node_coords = np.zeros((pct.N, 2))
    node_colors = []
    edge_coords = []
    edge_colors = []
    max_tree_height = max([len(br) for br in pct.branches()]) + 0.5
    for ni in range(pct.N):
        rangex = node_ranges_dict[ni]
        node_coords[ni][0] = (rangex[0] + rangex[1]) / 2
        node_height = len(pct._path_to_root(ni))
        node_coords[ni][1] = (1-(node_height / max_tree_height))
        node_colors.append([opacity[ni], opacity[ni], opacity[ni], 1])
        if ni not in pct.root_inds():
            par_i = pct.parent(ni)
            edge_coords.append([node_coords[ni][0], node_coords[ni][1], node_coords[par_i][0], node_coords[par_i][1]])
            edge_color = (opacity[ni] + opacity[par_i]) / 2
            edge_colors.append([edge_color, edge_color, edge_color, 1])
    node_colors = np.vstack(node_colors)
    edge_coords = np.vstack(edge_coords)
    edge_colors = np.vstack(edge_colors)
    return PCT_DisplayInfo(pct.N, edge_coords.shape[0], node_coords, edge_coords, node_colors, edge_colors)

def plot(disp: PCT_DisplayInfo, ax, size=100):
    ax.scatter(disp.node_coords[:, 0], disp.node_coords[:, 1], s=size, color= disp.node_colors, zorder=2)
    for ei in range(disp.n_edges):
        ax.plot([disp.edge_coords[ei, 0], disp.edge_coords[ei, 2]], [disp.edge_coords[ei, 1], disp.edge_coords[ei, 3]], lw=2, c = disp.edge_colors[ei], zorder=1)

def plot_coeffs(pct: PCTree, disp: PCT_DisplayInfo, x: NDArray, nodes: int, ax):
    coeffs = pct.V @ x
    coeffs_display = np.zeros(pct.N)
    coeffs_display[:] = np.nan
    coeffs_display[nodes] = coeffs[nodes]

    ax.scatter(disp.node_coords[:, 0], disp.node_coords[:, 1], s=110, c=coeffs_display, edgecolors= "grey", cmap="bwr", zorder=2)
    for ei in range(disp.n_edges):
        ax.plot([disp.edge_coords[ei, 0], disp.edge_coords[ei, 2]], [disp.edge_coords[ei, 1], disp.edge_coords[ei, 3]], lw=2, c = [0.5, 0.5, 0.5], zorder=1)