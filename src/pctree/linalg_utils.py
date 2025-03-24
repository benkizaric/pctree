import numpy as np
from numpy.typing import NDArray
import numpy.linalg as LA

from scipy.linalg import subspace_angles

def abid_dim_estimation(X):
    N = X.shape[0]
    X_norm = X / LA.norm(X, axis=1)[:, np.newaxis]
    angles = X_norm @ X_norm.T
    angz = angles[~np.eye(N, dtype=bool)].ravel()
    variance = np.var(angz)
    return 1 / variance

def my_cov(X: NDArray) -> NDArray:
    return np.dot(X.T, X)

def my_svd_sigma_VT_given_cov(cov: NDArray):
    eigvals, V = np.linalg.eigh(cov)
    idx = eigvals.argsort()[::-1]
    eigenvalues = eigvals[idx]
    V = V[:, idx]
    eig_vals_pos = eigenvalues > 0
    Sigma = np.zeros(eigenvalues.shape)
    Sigma[eig_vals_pos] = np.sqrt(eigenvalues[eig_vals_pos])
    return Sigma, V.T

def my_svd_U_given_sigma_VT(X: NDArray, s: NDArray, VT: NDArray):
    eig_pos = s > 0
    n_zero_eig = (s == 0).sum()
    U_unnormalized = np.dot(X, VT.T)
    U_pos = U_unnormalized[:, eig_pos] / s[eig_pos]
    return np.hstack([U_pos, np.zeros((U_pos.shape[0], n_zero_eig))])

def rank_1_approx(X: NDArray, v: NDArray) -> NDArray:
    return np.dot(X, v.reshape(-1, 1)) @ v.reshape((1, len(v)))

def my_svd_U_sigma_given_VT(X: NDArray, VT: NDArray):
    U_unnormalized = np.dot(X, VT.T)
    s = LA.norm(U_unnormalized, axis=0)
    return U_unnormalized / s, s

def my_full_svd(X: NDArray) -> tuple[NDArray, NDArray, NDArray]:
    cov = my_cov(X)
    s, VT = my_svd_sigma_VT_given_cov(cov)
    U = my_svd_U_given_sigma_VT(X, s, VT)
    return U, s, VT

def my_svd_sigma_VT_given_X(X: NDArray) -> tuple[NDArray, NDArray]:
    cov = my_cov(X)
    s, VT = my_svd_sigma_VT_given_cov(cov)
    return s, VT

def as_low_dimentions(U: NDArray, s: NDArray, VT: NDArray, low_dim: int = None) -> tuple[NDArray, NDArray, NDArray]:
    s_low = np.copy(s)
    if low_dim is not None:
        s_low[low_dim:] = 0
    eig_pos = s_low > 0
    return U[:, eig_pos], s[eig_pos], VT[eig_pos]

def my_svd_raw_U_given_s_VT_pos_eig(X: NDArray, s: NDArray, VT: NDArray) -> NDArray:
    eig_pos = s > 0
    U_unnormalized = np.dot(X, VT[eig_pos].T)
    return U_unnormalized

def my_svd_U_given_s_VT_pos_eig(X: NDArray, s: NDArray, VT: NDArray):
    eig_pos = s > 0
    U_unnormalized = np.dot(X, VT[eig_pos].T)
    return U_unnormalized / s[eig_pos]

def my_svd_raw_U_given_s_VT(X: NDArray, s: NDArray, VT: NDArray):
    U_unnormalized = np.dot(X, VT.T)
    return U_unnormalized

def my_svd_cov_given_s_VT(s: NDArray, VT: NDArray):
    return VT.T @ np.diag(s**2) @ VT

def project_vt_up(VT_1: NDArray, VT_2: NDArray, s2: NDArray) -> tuple[NDArray, NDArray]:
    low_dim_1, ambient = VT_1.shape
    low_dim_2, low_dim_1_prime = VT_2.shape
    assert low_dim_1 == low_dim_1_prime
    assert len(s2.shape) == 1
    s_result = np.zeros(ambient)
    s_result[:len(s2)] = s2
    return s2, VT_2 @ VT_1

def random_mask(N: int, take: int, seed: int = None):
    if N == take:
        return np.ones(N, dtype=bool)
    assert N >= take
    rng = np.random.default_rng(seed)
    rnd = rng.random(N)
    take_val = np.quantile(rnd, take / N)
    return rnd < take_val

def random_mask2(N: int, take: int, seed: int = None):
    if N == take:
        return np.ones(N, dtype=bool)
    assert N >= take
    arr = np.zeros(N, dtype=bool)
    arr[:take]=True
    np.random.shuffle(arr)
    return arr



def rank_to_explain_variance_given_s(s: NDArray, variance_explained: float) -> int:
    assert (variance_explained > 0.0) and (variance_explained < 1.0)
    max_rank = s.shape[0]
    s_sqr = s ** 2
    total_var = np.sum(s_sqr)
    percent_variance_explained = np.cumsum(s_sqr) / total_var
    return min((percent_variance_explained <= variance_explained).sum() + 1, max_rank)


def apply_sequential_masks(outer_mask, inner_mask):
    assert outer_mask.sum() == inner_mask.shape[0]
    # Find the indices where mask1 is True
    indices_mask1 = np.where(outer_mask)[0]
    
    # Use the indices from mask1 to extract elements from mask2
    masked_indices = indices_mask1[inner_mask]
    
    # Use the indices from mask1 to obtain the final result
    result = np.zeros_like(outer_mask)
    result[masked_indices] = True
    
    return result

def abs_cosine_similarity(X):
    X_norm = X / LA.norm(X, axis=1)[:, np.newaxis]
    return np.abs(X_norm @ X_norm.T)

def cosine_similarity(X):
    X_norm = X / LA.norm(X, axis=1)[:, np.newaxis]
    return X_norm @ X_norm.T

def grassmannian_distance(VA, VB):
    # Step 1: Compute the matrix M = VA^T * VB
    theta = subspace_angles(VA.copy(), VB.copy())
    return (theta ** 2).sum() ** 0.5

def ss_similarity(VA, VB):
    s = np.linalg.svd(VA.T.copy() @ VB.copy(), compute_uv=False)
    return LA.norm(s)

def weighted_ss_similarity(VA, sa, VB, sb):
    sa_norm = sa / sa.max()
    sb_norm = sb / sb.max()

    VA_w = VA.copy() * sa_norm
    VB_w = VB.copy() * sb_norm
    s = np.linalg.svd(VA_w.T @ VB_w, compute_uv=False)
    return LA.norm(s)

def rags(VA: NDArray, VB: NDArray, sigma: NDArray):
    # Representation aware grassmanian similarity
    sig_norm = sigma / sigma.max()
    VA_dim = VA.shape[1]
    VB_dim = VB.shape[1]

    VA_w = VA.copy() * sig_norm[:VA_dim]
    VB_w = VB.copy() * sig_norm[:VB_dim]
    s = np.linalg.svd(VA_w.T @ VB_w, compute_uv=False)
    return LA.norm(s)

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    #y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p

def binary_nmi(x, y):
    """
    Calculates the binary normalized mutual information (NMI) between two binary arrays.
    """

    # Ensure that x and y are binary
    x = np.asarray(x).astype(bool)
    y = np.asarray(y).astype(bool)

    # Calculate the contingency matrix
    n = x.size
    tp = np.sum(np.logical_and(x, y))
    tn = np.sum(np.logical_and(np.logical_not(x), np.logical_not(y)))
    fp = np.sum(np.logical_and(np.logical_not(x), y))
    fn = np.sum(np.logical_and(x, np.logical_not(y)))

    # Calculate entropy
    def entropy(p):
        if p == 0 or p == 1:
            return 0
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    hx = entropy(np.mean(x))
    hy = entropy(np.mean(y))

    # Calculate mutual information
   
    mi_components = [
        tp / n * np.log2((n * tp) / ((tp + fp) * (tp + fn))),
        fn / n * np.log2((n * fn) / ((fn + tn) * (tp + fn))),
        fp / n * np.log2((n * fp) / ((fp + tp) * (fp + tn))),
        tn / n * np.log2((n * tn) / ((tn + fn) * (tn + fp)))
    ]
    mi = sum([m for m in mi_components if not np.isnan(m)])

    # Calculate normalized mutual information
    if hx == 0 or hy == 0:
        return 0
    nmi = 2 * mi / (hx + hy)
    return nmi

def efficient_split_svd(full_cov: NDArray, X0: NDArray, X1: NDArray) -> tuple[tuple[NDArray, NDArray, NDArray], tuple[NDArray, NDArray, NDArray]]:
    assert full_cov.shape[0] == full_cov.shape[1]
    assert X0.shape[1] == X1.shape[1]
    cov0: NDArray = None
    cov1: NDArray = None
    if X0.shape[0] <= X1.shape[0]:
        cov0 = my_cov(X0)
        cov1 = full_cov - cov0
    else:
        cov1 = my_cov(X1)
        cov0 = full_cov - cov1
    s0, VT0 = my_svd_sigma_VT_given_cov(cov0)
    U0 = my_svd_U_given_sigma_VT(X0, s0, VT0)
    s1, VT1 = my_svd_sigma_VT_given_cov(cov1)
    U1 = my_svd_U_given_sigma_VT(X1, s1, VT1)
    return (U0, s0, VT0), (U1, s1, VT1)

        