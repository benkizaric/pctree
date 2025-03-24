import numpy as np
from numpy.typing import NDArray
import numpy.linalg as LA
from scipy.stats import ttest_ind, multivariate_normal
from math import floor
from dataclasses import dataclass
from time import time

from pctree.linalg_utils import my_svd_sigma_VT_given_cov, my_cov, my_full_svd, softmax, random_mask

def cohend(d1: NDArray, d2: NDArray, robust="mean"):
    # convention: d2 is null
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    if robust == "mean-median":
        u1 = (np.mean(d1) + np.median(d1)) / 2
        u2 = (np.mean(d2) + np.median(d2)) / 2
    if robust == "median":
        u1 = np.median(d1)
        u2 = np.median(d2)
    # calculate the effect size
    effect = np.abs((u1 - u2)) / s

    return effect

class SubspaceSifterOptions:
    sifting_effect_size: float = None 
    test_pval: float = None
    test_effect_size: float = None
    test_center_measure: str = None
    n_comparisons: int = None
    n_chunks: int = None

    def __init__(self, 
                sifting_effect_size: float, # 100% the most important parameter. Values between 0.2 and 0.8 reccomended. 
                test_pval: float = 0.001,
                test_effect_size: float = 0.2,
                test_center_measure: float = "mean",
                n_comparisons: int = 500,
                n_chunks: int = 8):
        self.sifting_effect_size = sifting_effect_size # 100% the most important parameter.
        self.test_pval = test_pval
        self.test_effect_size = test_effect_size
        self.test_center_measure = test_center_measure
        assert self.test_center_measure in ["mean", "mean-median", "median"]
        self.n_comparisons = n_comparisons
        self.n_chunks = n_chunks

def generate_matching_gaussian(X: NDArray):
    N, dim = X.shape
    mu_obs = X.mean(axis=0)
    cov_obs = my_cov(X - mu_obs) / N
    s_obs, VT_obs = my_svd_sigma_VT_given_cov(cov_obs)

    ord_norm_cent = np.sort(LA.norm(X - mu_obs, axis=1))

    W_pa = multivariate_normal(mean = np.zeros(dim), cov = np.eye(dim)).rvs(N)
    W_pa -= W_pa.mean(axis=0)
    W_pa_U, _, _ = my_full_svd(W_pa)
    W_pa_U_c = W_pa_U @ np.diag(s_obs * (N ** 0.5)) @ VT_obs
    
    # Do the ord norm adjust, and repeat.
    W_pa_U_c_ord = np.sort(LA.norm(W_pa_U_c, axis=1))

    # advanced colored...
    W_adv_colored = W_pa_U_c[np.argsort(LA.norm(W_pa_U_c, axis=1))] * (ord_norm_cent / W_pa_U_c_ord)[:, np.newaxis]

    W_ac_U, _, _ = my_full_svd(W_adv_colored)
    W_ac_U_c = W_ac_U @ np.diag(s_obs * (N ** 0.5)) @ VT_obs
    W_ac_U_c += mu_obs
    return W_ac_U_c

def random_masks(N: int, take: int, n_masks: int, seed= None):
    if N == take:
        return np.ones(N, dtype=bool)
    assert N >= take
    rng = np.random.default_rng(seed)
    arr = np.zeros((n_masks, N), dtype=bool)
    arr[:, :take]=True
    return rng.permuted(arr, axis=1)

def subspace_structure_index(X_outer: NDArray, theta = 3) -> tuple[NDArray, NDArray, NDArray]:
    N, _ = X_outer.shape
    t1 = time()
    avg_repr = X_outer.mean()
    outer_sm_weights = softmax(X_outer, axis=1, theta=theta)
    outer_sm = (outer_sm_weights * X_outer).sum(axis=1)
    return outer_sm, avg_repr

def soft_unit_normalize(X: NDArray, unit_normalize_factor: float = 4):
    X_norm = LA.norm(X, axis=1)
    median_norm = np.median(X_norm)
    X_norm_median_relative = X_norm / median_norm

    X_norm_median_relative_softened = 1 - np.exp(-unit_normalize_factor * X_norm_median_relative)
    X_unit = X / LA.norm(X, axis=1)[:, np.newaxis]
    X_soft_unit = X_unit * X_norm_median_relative_softened[:, np.newaxis]
    return X_soft_unit

def multi_chunk_outer_product_chunk_masks(X: NDArray, n_comparisons: int, n_chunks: int):
    N_use = n_comparisons * n_chunks
    N, d = X.shape
    assert N_use == N
    assert N_use <= N
    result = np.zeros((n_chunks, N_use), dtype=bool)
    use_indexes = np.arange(N_use)
    np.random.shuffle(use_indexes)
    for ci in range(n_chunks):
        chunk_start = ci * n_comparisons
        chunk_end = (ci+1) * n_comparisons
        result[ci, use_indexes[chunk_start:chunk_end]] = True
    return result

def multi_chunk_outer_product(X: NDArray, n_comparisons: int, n_chunks: int, chunk_masks: NDArray):
    N_use = n_comparisons * n_chunks
    N, d = X.shape
    assert N_use == N
    assert N_use <= N
    result = np.zeros((N_use, n_comparisons))
    
    tri_mask = np.eye(n_comparisons).astype(bool)
    X_unit = X / LA.norm(X, axis=1)[:, np.newaxis]

    for ci in range(n_chunks):
        chunk_start = ci * n_comparisons
        chunk_end = (ci+1) * n_comparisons
        X_chunk = X[chunk_masks[ci]]
        X_chunk_unit = X_unit[chunk_masks[ci]]
        outer_chunk = np.square(X_chunk @ X_chunk_unit.T)
        outer_chunk[tri_mask] = 0
        result[chunk_start:chunk_end] = outer_chunk
    return result

@dataclass
class SubspaceSiftResult():
    primary_significant: bool
    intersecting_dimenstions: bool
    primary_pval: float
    primary_cohend: float
    deeper_pvals: list[float]
    deeper_cohends: list[float]

@dataclass
class TestEffectSize():
    is_significant: bool
    pval: float
    effect_size: float

def remove_high_outliers(v: NDArray):
    # values should be only positive
    ceil = np.median(v) * 50
    return v[v < ceil]

def test_with_effectsize(values: NDArray, null_values: NDArray, alpha: float, effect_size_threshold: float, clamp_values: bool = False, robust_effect: str = "mean") -> TestEffectSize:
    val_clamp = values
    null_clamp = null_values
    if clamp_values:
        val_clamp = remove_high_outliers(values)
        null_clamp = remove_high_outliers(null_values)

    pval = ttest_ind(val_clamp, null_clamp, alternative="greater").pvalue
    effect = cohend(val_clamp, null_clamp, robust_effect)
    if np.isnan(effect):
        print(effect)
    significant = (pval <= alpha and effect >= effect_size_threshold)

    return TestEffectSize(
        is_significant= significant,
        pval= pval,
        effect_size= effect
    )

def sift_subspace(U: NDArray, s: NDArray, hyperparams: SubspaceSifterOptions = SubspaceSifterOptions(0.5)):
    N, d = U.shape
    n_comparisons_use = min(hyperparams.n_comparisons, N)
    n_chunks_use = min(hyperparams.n_chunks, floor(N / n_comparisons_use))
    half_dim = d // 2
    assert d == len(s)

    U_scale = U * s
    
    use_mask = random_mask(N, n_chunks_use * n_comparisons_use)
    U_scale_use = U_scale[use_mask]
    mvn = generate_matching_gaussian(U_scale_use)

    U_scale_soft_unit = soft_unit_normalize(U_scale_use[:, :half_dim])
    mvn_scale_unit = soft_unit_normalize(mvn[:, :half_dim])

    chunk_masks = multi_chunk_outer_product_chunk_masks(U_scale_soft_unit, n_comparisons_use, n_chunks_use)
    U_outer = multi_chunk_outer_product(U_scale_soft_unit, n_comparisons_use, n_chunks_use, chunk_masks)
    mvn_outer = multi_chunk_outer_product(mvn_scale_unit, n_comparisons_use, n_chunks_use, chunk_masks)

    U_compress_index_base, _ = subspace_structure_index(U_outer)
    mvn_compress_index_base, _ = subspace_structure_index(mvn_outer)

    hypothesis_test_result = test_with_effectsize(U_compress_index_base, mvn_compress_index_base, hyperparams.test_pval, hyperparams.test_effect_size, robust_effect=hyperparams.test_center_measure)

    if not hypothesis_test_result.is_significant:
        return SubspaceSiftResult(
            primary_significant = False,
            intersecting_dimenstions = 0,
            primary_pval = hypothesis_test_result.pval,
            primary_cohend = hypothesis_test_result.effect_size,
            deeper_pvals = [],
            deeper_cohends = []
        )

    max_additional_depth = d - 2
    deeper_pvals = []
    deeper_cohens = []
    past_effectsizes = []
    for sift_depth in range(1, max_additional_depth):
        deeper_norm = soft_unit_normalize(U_scale_use[:, sift_depth:sift_depth+half_dim])
        u_outer_deep = multi_chunk_outer_product(deeper_norm, n_comparisons_use, n_chunks_use, chunk_masks)
        U_deep_compress_index, _ = subspace_structure_index(u_outer_deep)

        # compared to the original data, how much less (or more?) subspace structure
        # is the data after we shave away leading dimensions.
        structure_index_reduction = U_compress_index_base - U_deep_compress_index
        structure_index_reduction_effectsize = (structure_index_reduction.mean()) / np.std(structure_index_reduction)
        effectsize_decreased = structure_index_reduction_effectsize < past_effectsizes[-1] if len(past_effectsizes) > 0 else False 
        past_effectsizes.append(structure_index_reduction_effectsize)

        # keep going until there is a significant drop in
        # performance by sifting away shallow dimentions
        if (structure_index_reduction_effectsize > hyperparams.sifting_effect_size) or effectsize_decreased:
            return SubspaceSiftResult(
                primary_significant = True,
                intersecting_dimenstions = sift_depth - 1,
                primary_pval = hypothesis_test_result.pval,
                primary_cohend = hypothesis_test_result.effect_size,
                deeper_pvals = deeper_pvals,
                deeper_cohends = deeper_cohens
            )

    return SubspaceSiftResult(
        primary_significant = True,
        intersecting_dimenstions = max_additional_depth,
        primary_pval = hypothesis_test_result.pval,
        primary_cohend = hypothesis_test_result.effect_size,
        deeper_pvals = deeper_pvals,
        deeper_cohends = deeper_cohens
    )