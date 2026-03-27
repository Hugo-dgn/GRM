from collections import defaultdict

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

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
        
        self.n = n
    
    def add_node(self, node, unary):
        if node in self.nodes:
            raise ValueError(f'Node {node} already exists.')
        self.nodes.append(node)
        self.phi[node] = unary
    
    def add_edge(self, node1, node2, psi12):
        self.adj[node1].append(node2)
        self.adj[node2].append(node1)
        
        self.psi[(node1, node2)] = psi12
        self.psi[(node2, node1)] = psi12.T

class DensityDistance(BaseEstimator, ClassifierMixin):
    def __init__(self, reg=1e-6):
        self.reg = reg
        
        self.models = [None, None]

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
            self.models[c] = multivariate_normal(mean=mean, cov=cov)

        return self

    def predict_proba(self, X):
        """
        Returns: (n_samples, n_classes)
        """
        probs = []
        
        for model in self.models:
            probs.append(model.pdf(X))

        probs = np.stack(probs)
        probs = probs / np.sum(probs, axis=0)
        
        return probs.T

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


class Segment():
    
    def __init__(self, size, model, beta=1, sigma=100):
        self.size = size
        self.model = model
        self.scaler = StandardScaler()
        
        self.beta = beta
        self.sigma = sigma
    
    def fit(self, features, labels):
        features = self.scaler.fit_transform(features)
        self.model.fit(features, labels)
    
    def predict(self, features):
        features = self.scaler.transform(features)
        return self.model.predict(features)

    def predict_image(self, feature_map):
        H, W, C = feature_map.shape
        X_flat = feature_map.reshape(-1, C)
        predictions = self.predict(X_flat).reshape(H, W, -1)
        return predictions[..., 0]
    
    def __call__(self, features, max_iter):
        graph = utils.segmentation_graph_from_image(self.model, features, self.beta, self.sigma, transform=self.scaler)
        marginals = inference.loopy_bp(graph, max_iter)
        mask = utils.image_from_marginals(features, marginals)
        return mask

class DensityMask(Segment):
    
    def __init__(self, size, beta=1, sigma=100):
        model = DensityDistance()
        Segment.__init__(self, size, model, beta, sigma)
        
class LogisticMask(Segment):
    
    def __init__(self, size, beta=1, sigma=100, solver="lbfgs"):
        model = LogisticRegression(solver=solver)
        Segment.__init__(self, size, model, beta, sigma)