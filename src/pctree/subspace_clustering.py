import numpy as np
import numpy.linalg as LA
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering
from time import time

from pctree.linalg_utils import my_svd_U_given_sigma_VT, my_svd_sigma_VT_given_cov, my_cov, abs_cosine_similarity, random_mask, my_cov, my_svd_sigma_VT_given_cov
from pctree.pruning import EfficientEncodingPruner
from pctree.core import PCTree

class HACSubspaceClusterer:
    def __init__(self) -> None:
        self.training_points_unit: NDArray = None
        self.training_points_assignments: NDArray = None

    def fit(self, U: NDArray):
        self.training_points_unit = U / LA.norm(U, axis=1)[:, np.newaxis]
        clusterer = AgglomerativeClustering(n_clusters=2, metric="euclidean", linkage="ward")
        abs_cosine_sim = abs_cosine_similarity(self.training_points_unit)
        self.training_points_assignments = clusterer.fit_predict(abs_cosine_sim)

    def predict(self, U: NDArray):
        U_unit = U / LA.norm(U, axis=1)[:, np.newaxis]
        cosine_sim_to_training_points = np.abs(U_unit @ self.training_points_unit.T)
        nearest_training_points = cosine_sim_to_training_points.argmax(axis=1)
        return self.training_points_assignments[nearest_training_points]


class BinarySSClustering:
    s0: NDArray
    VT0: NDArray
    n0: int
    s1: NDArray
    VT1: NDArray
    n1: int

    def __init__(self, s0, VT0, n0, s1, VT1, n1) -> None:
        self.s0: NDArray = s0
        self.VT0: NDArray = VT0
        self.n0: int = n0
        self.s1: NDArray = s1
        self.VT1: NDArray = VT1
        self.n1: int = n1

class BinaryBuckshotSubspaceClusterer:
    N: int
    s0: NDArray
    VT0: NDArray
    s1: NDArray
    VT1: NDArray
    cov0: NDArray
    cov1: NDArray
    U_test: NDArray
    X_test: NDArray
    trial_assigments: NDArray
    trail_buckshots: list[HACSubspaceClusterer]
    chosen_buckshot: HACSubspaceClusterer

    cov_eval: NDArray
    
    dim: int
    n_trials: int

    cov_time_trial: float
    cov_time_full: float
    eig_time: float
    eig_time_trial: float
    buck_train_time: float
    pred_time_buck_trail: float
    pred_time_auc_trail: float
    pred_time_buck_full: float
    pred_time_auc_full: float
    tree_time: float

    pred_mul_time: float
    cluster_sim_time: float

    subtree_selector: EfficientEncodingPruner

    ope: list[tuple[NDArray, NDArray, float, NDArray, NDArray, float]]
    trial_eval_assignments: list[NDArray]
    trial_eval_assignments_buck: list[NDArray]
    aucs: list[int]
    big_props: list[float]

    def __init__(self, dim: int, n_trials: int) -> None:
        self.full_cov: NDArray
        self.full_s: NDArray
        self.full_VT: NDArray
        self.native_dim: int = 0

        self.cov_time_trial: float = 0.0
        self.cov_time_full: float = 0.0
        self.eig_time: float = 0.0
        self.eig_time_trial: float = 0.0
        self.buck_train_time: float = 0.0
        self.pred_time_buck_trail: float = 0.0
        self.pred_time_auc_trail: float = 0.0
        self.pred_time_buck_full: float = 0.0
        self.pred_time_auc_full: float = 0.0
        self.tree_time: float = 0.0


        self.dim = dim
        self.n_trials = n_trials
        self.subtree_selector = EfficientEncodingPruner()
        self.trial_eval_assignments = []
        self.trial_eval_assignments_buck = []
        self.aucs = []
        self.big_props = []
    
    def _predict(self, X_in: NDArray, VT0: NDArray, VT1: NDArray) -> NDArray:
        N = X_in.shape[0]
        affinities = np.zeros((2, N))
        effective_dim  = min(VT0[:self.dim].shape[0], VT1[:self.dim].shape[0])
        affinities[0] = np.square(X_in @ VT0[:effective_dim].T).cumsum(axis=1).sum(axis=1)
        affinities[1] = np.square(X_in @ VT1[:effective_dim].T).cumsum(axis=1).sum(axis=1)
        result = affinities.argmax(axis=0)
        return result
    
    def predict(self, X: NDArray):
        two = self._predict(X, self.VT0, self.VT1)
        return two
    
    def buckshot_size(self) -> int:
        return min(round((self.N*3) ** 0.5), 400)
    
    def training_pool_size(self) -> int:
        return self.N
    
    def eval_size(self) -> int:
        return min(round(self.buckshot_size() * self.n_trials), self.N)
    
    def eval_dim(self) -> int:
        assert self.native_dim >= 1
        return round((self.dim + self.native_dim) / 2)
    
    def fit(self, X: NDArray):
        self.N = X.shape[0]
        self.native_dim = X.shape[1]
        t0 = time()
        self.full_cov = my_cov(X)
        t1 = time()
        self.full_s, self.full_VT = my_svd_sigma_VT_given_cov(self.full_cov)
        t2 = time()
        U = my_svd_U_given_sigma_VT(X, self.full_s, self.full_VT)
        US = U * self.full_s
        eval_mask = random_mask(self.N, self.eval_size())
        training_pool_mask = random_mask(self.N, self.training_pool_size())
        self.U_train = US[training_pool_mask]
        self.X_train = X[training_pool_mask]
        self.U_eval = US[eval_mask]
        self.X_eval = X[eval_mask, :self.eval_dim()]
        t3 = time()
        self.cov_eval = my_cov(self.X_eval)
        t4 = time()

        self.cov_time_full += t1 - t0
        self.eig_time += t2 - t1
        self.eig_time += t4 - t3

        bucks: list[HACSubspaceClusterer] = []
        eval_assignments = []
        trial_quality = np.zeros(self.n_trials)
        trial_smaller_prop = np.zeros(self.n_trials)
        for i in range(self.n_trials):
            buck, eval_assignment, quality, smaller_prop = self.run_trial()
            bucks.append(buck)
            eval_assignments.append(eval_assignment)
            trial_quality[i] = quality
            trial_smaller_prop[i] = smaller_prop

        quality_score = trial_quality
        quality_score = trial_smaller_prop

        chosen_buck = bucks[np.argmax(quality_score)]
        _, _, clustering = self.apply_buck(chosen_buck, US, X, self.full_cov)
        self.s0 = clustering.s0
        self.VT0 = clustering.VT0
        self.s1 = clustering.s1
        self.VT1 = clustering.VT1
        if False:
            perf = {
                "cov_time_trail": round(self.cov_time_trial, 4),
                "cov_time_full": round(self.cov_time_full, 4),
                "eig_time": round(self.eig_time, 4),
                "eig_time_trial": round(self.eig_time_trial, 4),
                "buck_train_time": round(self.buck_train_time, 4),
                "pred_time_buck_trail": round(self.pred_time_buck_trail, 4),
                "pred_time_auc_trail": round(self.pred_time_auc_trail, 4),
                "pred_time_buck_full": round(self.pred_time_buck_full, 4),
                "pred_time_auc_full": round(self.pred_time_auc_full, 4),
                "tree_time": round(self.tree_time, 4)
            }
            print(perf)

            

    def run_trial(self) -> tuple[HACSubspaceClusterer, NDArray, NDArray]:
        trail_train_mask = random_mask(self.training_pool_size(), self.buckshot_size())
        trail_train_U = self.U_train[trail_train_mask, :self.dim]
        
        buck = HACSubspaceClusterer()
        t0 = time()
        buck.fit(trail_train_U)
        t1 = time()
        eval_ass_hac = buck.predict(self.U_eval[:, :self.dim])
        t2 = time()

        cov0_hac = my_cov(self.X_eval[eval_ass_hac == 0])
        cov1_hac = self.cov_eval - cov0_hac
        t3 = time()
        s0_hac, VT0_hac = my_svd_sigma_VT_given_cov(cov0_hac)
        s1_hac, VT1_hac = my_svd_sigma_VT_given_cov(cov1_hac)
        t4 = time()

        eval_ass = self._predict(self.X_eval, VT0_hac, VT1_hac)
        t5 = time()
        
        cov0 = my_cov(self.X_eval[eval_ass == 0])
        cov1 = self.cov_eval - cov0
        t6 = time()
        s0, VT0 = my_svd_sigma_VT_given_cov(cov0)
        s1, VT1 = my_svd_sigma_VT_given_cov(cov1)
        t7 = time()

        n_data0, n_data1 = (eval_ass == 0).sum(), (eval_ass == 1).sum()
        tree_trail, data_prop_trail = single_binary_ss_clusters_to_pct(VT0, s0, VT1, s1, n_data0, n_data1)
        _, order = self.subtree_selector.subtree(tree_trail.N, tree_trail, data_prop_trail)
        t8 = time()

        curve_x = data_prop_trail[order].cumsum()
        curve_y = (tree_trail.s**2)[order].cumsum()

        n = len(eval_ass)
        n0_eval, n1_eval = (eval_ass == 0).sum(), (eval_ass == 1).sum()
        smaller_prop_eval = round(min(n0_eval, n1_eval) / n, 4)

        self.buck_train_time += t1 - t0
        self.pred_time_buck_trail += t2 - t1
        self.cov_time_trial += t3 - t2
        self.eig_time_trial += t4 - t3
        self.pred_time_auc_trail += t5 - t4
        self.cov_time_trial += t6 - t5
        self.eig_time_trial += t7 - t6
        self.tree_time += t8 - t7

        return buck, eval_ass, np.trapz(curve_y, curve_x), smaller_prop_eval
    
    def apply_buck(self, buck: HACSubspaceClusterer, U_test: NDArray, X_test: NDArray, X_test_cov: NDArray) -> tuple[NDArray, NDArray, BinarySSClustering]:
        t0 = time()
        ass_buck = buck.predict(U_test[:, :self.dim])
        t1 = time()
        cov0 = my_cov(X_test[ass_buck == 0])
        cov1 = X_test_cov - cov0
        t2 = time()
        s0, VT0 = my_svd_sigma_VT_given_cov(cov0)
        s1, VT1 = my_svd_sigma_VT_given_cov(cov1)
        t3 = time()
        
        ass_auc = self._predict(X_test, VT0, VT1)
        t4 = time()
        cov0 = my_cov(X_test[ass_auc == 0])
        cov1 = X_test_cov - cov0
        t5 = time()
        s0, VT0 = my_svd_sigma_VT_given_cov(cov0)
        s1, VT1 = my_svd_sigma_VT_given_cov(cov1)
        t6 = time()

        self.pred_time_buck_full += t1 - t0
        self.cov_time_full += t2 - t1
        self.eig_time += t3 - t2
        self.pred_time_auc_full += t4 - t3
        self.cov_time_full += t5 - t4
        self.eig_time += t6 - t5

        n_data0, n_data1 = (ass_auc == 0).sum(), (ass_auc == 1).sum()
        big_n = max(n_data0, n_data1)
        big_prop = big_n / (n_data0 + n_data1)

        tree_trail, data_prop_trail = single_binary_ss_clusters_to_pct(VT0, s0, VT1, s1, n_data0, n_data1)
        _, order = self.subtree_selector.subtree(tree_trail.N, tree_trail, data_prop_trail)
        
        curve_x = data_prop_trail[order].cumsum()
        curve_y = (tree_trail.s**2)[order].cumsum()

        return curve_x, curve_y, BinarySSClustering(s0, VT0, n_data0, s1, VT1, n_data1)


def single_binary_ss_clusters_to_pct(VT0: NDArray, s0: NDArray, VT1: NDArray, s1: NDArray, n_data0: int, n_data1: int) -> tuple[PCTree, NDArray]:
    assert VT0.shape[0] == len(s0)
    assert VT1.shape[0] == len(s1)
    n0 = VT0.shape[0]
    n1 = VT1.shape[0]

    V = np.vstack([VT0, VT1])
    s = np.concatenate([s0, s1])

    E = np.arange(n0 + n1)
    E[0] = PCTree.ROOT_FLAG
    E[n0] = PCTree.ROOT_FLAG

    prop0, prop1 = n_data0 / (n_data0+n_data1), n_data1 / (n_data0+n_data1)
    data_prop = np.zeros(n0 + n1)
    data_prop[:n0] = prop0
    data_prop[n0:] = prop1
    return PCTree(V, E, s), data_prop