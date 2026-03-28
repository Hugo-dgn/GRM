from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from scipy.stats import multivariate_normal

import numpy as np

import utils
import inference

class Graph:
    def __init__(self, n):
        self.nodes = []
        self.adj = defaultdict(list)
        self.phi = {}
        self.psi = {}
        self.rho = {}
        self.n = n

    def add_node(self, node, unary):
        if node in self.nodes:
            raise ValueError(f'Node {node} already exists.')
        self.nodes.append(node)
        self.phi[node] = unary

    def add_edge(self, node1, node2, psi12, rho=1.0):
        if not (0 < rho <= 1):
            raise ValueError(f'rho must be in (0, 1], got {rho}')
        self.adj[node1].append(node2)
        self.adj[node2].append(node1)
        self.psi[(node1, node2)] = psi12
        self.psi[(node2, node1)] = psi12.T
        self.rho[(node1, node2)] = rho
        self.rho[(node2, node1)] = rho
    
    def compute_rho_uniform_spanning_tree(self):
        """
        Compute edge appearance probabilities under the uniform spanning tree
        distribution using the Matrix-Tree theorem.
        
        For each edge (i,j): rho_ij = L^+[i,i] + L^+[j,j] - 2*L^+[i,j]
        where L^+ is the pseudoinverse of the Laplacian.
        """
        nodes = self.nodes
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}

        # Build the Laplacian (unweighted: all edge weights = 1)
        L = np.zeros((n, n))
        for node in nodes:
            for nb in self.adj[node]:
                i, j = node_idx[node], node_idx[nb]
                L[i, i] += 1
                L[i, j] -= 1
        L /= 2  # each edge was counted twice

        # Pseudoinverse of the Laplacian
        L_pinv = np.linalg.pinv(L)

        # Edge appearance probability: rho_ij = L^+_ii + L^+_jj - 2*L^+_ij
        rho = {}
        for node in nodes:
            for nb in self.adj[node]:
                if (nb, node) in rho:
                    rho[(node, nb)] = rho[(nb, node)]
                else:
                    i, j = node_idx[node], node_idx[nb]
                    r = L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j]
                    r = float(np.clip(r, 1e-6, 1.0))
                    rho[(node, nb)] = r
                    rho[(nb, node)] = r

        self.rho = rho

class DensityDistance(BaseEstimator, ClassifierMixin):
    def __init__(self, reg=1e-6):
        self.reg = reg
        
        self.models_ = [None, None]

    def fit(self, X, y):
        """
        X: (n_samples, 3)
        y: (n_samples,) binary labels
        """
        _, n_features = X.shape

        for c in range(2):
            X_c = X[y == c]

            mean = X_c.mean(axis=0)
            cov = np.cov(X_c, rowvar=False)
            cov += self.reg * np.eye(n_features)
            self.models_[c] = multivariate_normal(mean=mean, cov=cov)

        return self

    def predict_proba(self, X):
        """
        Returns: (n_samples, n_classes)
        """
        probs = []
        
        for model in self.models_:
            probs.append(model.pdf(X))

        probs = np.stack(probs)
        probs = probs / (np.sum(probs, axis=0) + 1e-6)
        
        return probs.T

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class Segment():
    
    def __init__(self, model, beta=1, sigma=100):
        
        self.model = Pipeline([('scaler', StandardScaler()), ('model', model)])
        
        self.beta = beta
        self.sigma = sigma
    
    def fit(self, features, labels):
        self.model.fit(features, labels)
    
    def predict(self, features):
        return self.model.predict(features)

    def predict_image(self, feature_map):
        H, W, C = feature_map.shape
        X_flat = feature_map.reshape(-1, C)
        predictions = self.predict(X_flat).reshape(H, W, -1)
        return predictions[..., 0]
    
    def __call__(self, features, max_iter, trw=False):
        graph = utils.segmentation_graph_from_image(self.model, features, self.beta, self.sigma)
        if trw:
            graph.compute_rho_uniform_spanning_tree()
            marginals = inference.trw_bp(graph, max_iter)
        else:
            marginals = inference.loopy_bp(graph, max_iter)
        mask = utils.image_from_marginals(features, marginals)
        return mask

class DensityMask(Segment):
    
    def __init__(self, beta=1, sigma=100):
        model = DensityDistance()
        Segment.__init__(self, model, beta, sigma)
        
class LogisticMask(Segment):
    
    def __init__(self, beta=1, sigma=100, solver="lbfgs"):
        model = LogisticRegression(solver=solver)
        Segment.__init__(self, model, beta, sigma)
        

class SuperSegment():
    
    def __init__(self, model, beta=1, sigma=100, n_segments=100, compactness=20, agg_func=None):
        
        self.model = Pipeline([('scaler', StandardScaler()), ('model', model)])
        
        self.beta = beta
        self.sigma = sigma
        
        self.n_segments = n_segments
        self.compactness = compactness
        
        self.agg_func = agg_func
    
    def fit(self, features, labels):
        self.model.fit(features, labels)
    
    def predict(self, features):
        return self.model.predict(features)

    def predict_image(self, feature_map):
        H, W, C = feature_map.shape
        X_flat = feature_map.reshape(-1, C)
        predictions = self.predict(X_flat).reshape(H, W, -1)
        return predictions[..., 0]
    
    def __call__(self, features, max_iter, trw=False):
        graph, segments, pixels, edges = utils.get_super_pixels_graph(self.model, 
                                                                      features, 
                                                                      self.beta,
                                                                      self.sigma,
                                                                      n_segments=self.n_segments,
                                                                      compactness=self.compactness,
                                                                      agg_func=self.agg_func)
        if trw:
            graph.compute_rho_uniform_spanning_tree()
            marginals = inference.trw_bp(graph, max_iter)
        else:
            marginals = inference.loopy_bp(graph, max_iter)
        
        return utils.image_from_super_marginals(features, marginals, segments)
    
    
class SuperDensityMask(SuperSegment):
    
    def __init__(self, beta=1, sigma=100, n_segments=100, compactness=20, agg_func=None):
        model = DensityDistance()
        SuperSegment.__init__(self, model, beta, sigma, n_segments, compactness, agg_func=agg_func)
        
class SuperLogisticMask(SuperSegment):
    
    def __init__(self, beta=1, sigma=100, n_segments=100, compactness=20, solver="lbfgs", agg_func=None):
        model = LogisticRegression(solver=solver)
        SuperSegment.__init__(self, model, beta, sigma, n_segments, compactness, agg_func=agg_func)