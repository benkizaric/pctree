from scipy.stats import norm
from typing import Callable
from sklearn.mixture import GaussianMixture
import numpy as np
from numpy.typing import NDArray

from pctree.core import PCTree, PCTreeCoefficients

class NodeGenerativeModel:
    def __init__(self):
        pass
    def fit(self, values: NDArray):
        # values is 1-D array
        pass

    def loglikelihood(self, values: NDArray) -> NDArray:
        pass

    def generate(self, values: NDArray) -> NDArray:
        pass

    def num_params(self) -> int:
        pass

class NormalGenerativeModel(NodeGenerativeModel):
    mean: float
    std: float
    def __init__(self):
        pass
    def fit(self, values: NDArray):
        self.mean = values.mean()
        self.std = values.std()

    def loglikelihood(self, values: NDArray) -> NDArray:
        return norm(self.mean, self.std).logpdf(values)

    def generate(self, N: int) -> NDArray:
        return norm(self.mean, self.std).rvs(N)
    
    def num_params(self) -> int:
        return 2
    
class GMM1DGenerativeModel(NodeGenerativeModel):
    core_model: GaussianMixture
    n_components: int
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.core_model = GaussianMixture(n_components)

    def fit(self, values: NDArray):
        self.core_model.fit(values.reshape(-1, 1))

    def loglikelihood(self, values: NDArray) -> NDArray:
        return self.core_model.score_samples(values.reshape(-1, 1))

    def generate(self, N: int) -> NDArray:
        values, labels =  self.core_model.sample(N)
        result = values.ravel()
        np.random.shuffle(result)
        return result
    
    def num_params(self) -> int:
        return self.core_model._n_parameters()
    
class BranchAssignmentGenerator:
    ass: NDArray
    probs: NDArray
    def __init__(self, assignments: NDArray):
        n = len(assignments)
        ass_distinct = list(set(list(assignments)))
        self.ass = np.zeros(len(ass_distinct), dtype=int)
        self.probs = np.zeros(len(ass_distinct))
        for idx, a in enumerate(ass_distinct):
            self.ass[idx] = a
            self.probs[idx] = (assignments == a).sum() / n
        self.probs /= self.probs.sum()

    def generate(self, n: int) -> NDArray:
        return np.random.choice(self.ass, size=n, p=self.probs)

class BranchGenerativeModel:
    def __init__(self):
        pass
    def fit(self, values: NDArray):
        # values is 1-D array
        pass

    def loglikelihood(self, C: NDArray) -> NDArray:
        pass

    def generate(self, values: NDArray) -> NDArray:
        pass 

    def num_params(self) -> int:
        pass

class BayesBranchGenerativeModel(BranchGenerativeModel):
    make_model: Callable[[], NodeGenerativeModel]
    # node space is branch-relative!
    node_models: dict[int, NodeGenerativeModel]
    branch_dim: int
    def __init__(self, make_model: Callable[[], NodeGenerativeModel]):
        self.make_model = make_model
    
    def fit(self, X: NDArray):
        self.branch_dim = X.shape[1]
        self.node_models = {}
        for ni in range(self.branch_dim):
            self.node_models[ni] = self.make_model()
            self.node_models[ni].fit(X[:, ni])
    
    def generate(self, n: int) -> NDArray:
        result = np.zeros((n, self.branch_dim))
        for ni in self.node_models:
            generated_values = self.node_models[ni].generate(n)
            result[:, ni] = generated_values
        return result
    
    def loglikelihood(self, C: NDArray) -> NDArray:
        assert len(self.node_models) == C.shape[1]
        nodewise_log_likelihood = np.zeros((C.shape[0], self.branch_dim))
        for ni in self.node_models:
            nodewise_log_likelihood[:, ni] = self.node_models[ni].loglikelihood(C[:, ni])
        return np.sum(nodewise_log_likelihood, axis=1)
    
    def num_params(self) -> int:
        result = 0
        for ni, node_model in self.node_models.items():
            result += node_model.num_params()
        return result
    
class GMMBranchGenerativeModel(BranchGenerativeModel):
    core_model: GaussianMixture
    branch_dim: int
    def __init__(self, n_components: int, covariance_type: int):
        self.core_model = GaussianMixture(n_components=n_components, covariance_type=covariance_type)
    
    def fit(self, X: NDArray):
        self.core_model.fit(X)
    
    def generate(self, n: int) -> NDArray:
        values, labels = self.core_model.sample(n)
        np.random.shuffle(values)
        return values
    
    def loglikelihood(self, C: NDArray) -> NDArray:
        return self.core_model.score_samples(C)
    
    def num_params(self) -> int:
        return self.core_model._n_parameters()

class NodeWiseGenerativeModel:
    make_model: Callable[[], NodeGenerativeModel]
    pct: PCTree
    node_models: dict[int, NodeGenerativeModel]
    def __init__(self, make_model: Callable[[], NodeGenerativeModel]):
        self.make_model = make_model

    def fit(self, coeff: PCTreeCoefficients):
        self.node_models = {}
        self.pct = coeff.pct
        assert coeff.pct.N == len(coeff.included_nodes)
        for ni in coeff.included_nodes:
            node_values = coeff.node(ni)
            self.node_models[ni] = self.make_model()
            self.node_models[ni].fit(node_values)
    
    def generate(self, branch_assignments: NDArray) -> PCTreeCoefficients:
        result = PCTreeCoefficients(self.pct, None, branch_assignments, isEmpty=True)
        for ni in self.node_models:
            generated_values = self.node_models[ni].generate(result._node_counts[ni])
            result.set_node(ni, generated_values)
        return result
    
    def loglikelihood(self, X: NDArray, branch_assignments: NDArray) -> NDArray:
        coeffs = PCTreeCoefficients(self.pct, X, branch_assignments)
        result_log_likelihood = PCTreeCoefficients(self.pct, None, branch_assignments, isEmpty=True)
        for ni in coeffs.included_nodes:
            node_values = coeffs.node(ni)
            result_log_likelihood.set_node(ni, self.node_models[ni].loglikelihood(node_values))
        return result_log_likelihood.sum_over_branches()
    
    def num_params(self) -> int:
        result = 0
        for ni, node_model in self.node_models.items():
            result += node_model.num_params()
        return result

    
class BranchwiseGenerativeModel:
    make_model: Callable[[], BranchGenerativeModel]
    pct: PCTree
    branch_models: dict[int, BranchGenerativeModel]
    def __init__(self, make_model: Callable[[], BranchGenerativeModel]):
        self.make_model = make_model

    def fit(self, coeff: PCTreeCoefficients):
        self.branch_models = {}
        self.pct = coeff.pct
        assert coeff.pct.N == len(coeff.included_nodes)
        for br_i in coeff.included_branches:
            _, branch_coeffs = coeff.branch(br_i)
            self.branch_models[br_i] = self.make_model()
            self.branch_models[br_i].fit(branch_coeffs)
    
    def generate(self, branch_assignments: NDArray) -> PCTreeCoefficients:
        result = PCTreeCoefficients(self.pct, None, branch_assignments, isEmpty=True)
        for br_i in self.branch_models:
            generated_values = self.branch_models[br_i].generate(result._branch_counts[br_i])
            result.set_branch(br_i, generated_values)
        return result
    
    def loglikelihood(self, X: NDArray, branch_assignments: NDArray) -> NDArray:
        result = np.zeros(X.shape[0])
        coeffs = PCTreeCoefficients(self.pct, X, branch_assignments)
        for br_i in coeffs.included_branches:
            _, CB = coeffs.branch(br_i)
            result[branch_assignments == br_i] = self.branch_models[br_i].loglikelihood(CB)
        return result
    
    def num_params(self) -> int:
        result = 0
        for br_i, branch_model in self.branch_models.items():
            result += branch_model.num_params()
        return result