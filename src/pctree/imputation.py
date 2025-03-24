import numpy as np
from numpy.typing import NDArray
from typing import Callable

from pctree.core import PCTree, PCTreeCoefficients
from pctree.training import PCTreeTrainer, PCTreeTrainerOptions
from pctree.pruning import MostVariancePruner
from pctree.branches import EfficientEncodingRouter
from pctree.reweighting import PCTreeReweighter, in_place_reweight

class PCTreeImputerOptions:
    pct_training_options: PCTreeTrainerOptions = None
    n_free_iters: int = None
    n_reweight_iters: int = None
    n_pca_iters: int = None
    def __init__(self, pct_training_options, n_free_iters, n_reweight_iters, n_pca_iters):
        self.pct_training_options = pct_training_options
        self.n_free_iters = n_free_iters
        self.n_reweight_iters = n_reweight_iters
        self.n_pca_iters = n_pca_iters

class PCTImputer:
    options: PCTreeImputerOptions
    X_mean: NDArray = None
    verbose: int = 0
    def __init__(self, training_options: PCTreeImputerOptions):
        self.options = training_options

    def fit_predict(self, X: NDArray, verbose = 0):
        self.verbose = verbose
        is_nan = np.isnan(X)
        X_0 = np.zeros(X.shape)
        X_0[~is_nan] = X[~is_nan]
        self.X_mean = X_0.mean(axis=0)
        X_0_cent = X_0 - self.X_mean

        curr_best_guess = X_0_cent

        for _ in range(self.options.n_pca_iters):
            curr_best_guess = self._impute_pca_iter(curr_best_guess, is_nan)

        curr_best_tree = None
        for _ in range(self.options.n_free_iters):
            curr_best_guess, curr_best_tree = self._impute_iter_free(curr_best_guess, is_nan)

        for _ in range(self.options.n_reweight_iters):
            curr_best_guess, curr_best_tree = self._imput_iter_reweight(curr_best_guess, curr_best_tree, is_nan)

        return curr_best_guess + self.X_mean

    def _impute_pca_iter(self, X_best_guess: NDArray, is_nan: NDArray) -> NDArray:
        max_nodes = self.options.pct_training_options.max_nodes
        tree_pca = PCTreeTrainer.from_pca(X_best_guess, n_components=max_nodes)

        C = PCTreeCoefficients(tree_pca, X_best_guess, np.zeros(X_best_guess.shape[0], dtype=int))
        X_new_approx = C.reconstruct()
        result = np.copy(X_best_guess)
        result[is_nan] = X_new_approx[is_nan]
        
        return result
        
    def _impute_iter_free(self, X_best_guess: NDArray, is_nan: NDArray) -> NDArray:
        # build a pct from best guess
        proto_tree = PCTreeTrainer(self.options.pct_training_options)
        proto_tree.fit_partition(X_best_guess, self.verbose)
        tree = proto_tree.fit_em(X_best_guess, X_best_guess, self.verbose)

        best_branch = EfficientEncodingRouter(tree).predict(X_best_guess)
        C = PCTreeCoefficients(tree, X_best_guess, best_branch)
        X_new_approx = C.reconstruct()
        result = np.copy(X_best_guess)
        result[is_nan] = X_new_approx[is_nan]

        return result, tree
    
    def _imput_iter_reweight(self, X_best_guess: NDArray, tree: PCTree, is_nan: NDArray):
        best_branch = EfficientEncodingRouter(tree).predict(X_best_guess)
        tree = in_place_reweight(tree, X_best_guess, best_branch, self.options.pct_training_options.variance_capture_percent)

        selec = EfficientEncodingRouter(tree)
        best_branch = selec.predict(X_best_guess)
        C = PCTreeCoefficients(tree, X_best_guess, best_branch)
        X_new_approx = C.reconstruct()
        result = np.copy(X_best_guess)
        result[is_nan] = X_new_approx[is_nan]

        return result, tree