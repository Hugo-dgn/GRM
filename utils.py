import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator

import models

def plot_graph(graph):

    G = nx.Graph()

    # add nodes
    for node in graph.nodes:
        G.add_node(node)

    # add edges
    for node in graph.nodes:
        for neigh in graph.adj[node]:
            if not G.has_edge(node, neigh):
                G.add_edge(node, neigh)

    pos = nx.spring_layout(G)

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=800,
        node_color="lightblue",
        font_size=12
    )

    plt.show()
    
    
def noise_graph_from_image(image : np.array, alpha : float, beta : float):
    graph = models.Graph(2)
    
    pairwise = np.exp(-np.array([[0, beta],
                                [beta, 0]]))
    
    H, W = image.shape
    
    def flat_idx(i, j):
        return i*W + j
    
    for i in range(H):
        for j in range(W):
            x = image[i, j]
            u = np.zeros(2)
            u[1 - int(x)] = alpha
            p = np.exp(-u)
            p = p / p.sum()
            k = flat_idx(i, j)
            graph.add_node(k, p)
            
            if j+1 < W:
                graph.add_edge(k, flat_idx(i, j+1), pairwise)
            if i+1 < H:
                graph.add_edge(k, flat_idx(i+1, j), pairwise)

    
    return graph


def segmentation_graph_from_image(model: BaseEstimator, image: np.ndarray,
                                  beta: float, sigma: float, scaler=None):
    H, W, C = image.shape
    N = H * W
    graph = models.Graph(2)

    flat = np.arange(N).reshape(H, W)

    X_flat = image.reshape(-1, C)
    
    if scaler is not None:
        X_flat = scaler.transform(X_flat)
    
    probas = model.predict_proba(X_flat).reshape(H, W, -1)
    probas /= probas.sum(axis=-1, keepdims=True) + 1e-10

    for k in range(N):
        graph.add_node(k, probas.reshape(-1, probas.shape[-1])[k])

    pairwise = np.exp(-np.array([[0, beta],
                                 [beta, 0]]))


    diff_h = np.linalg.norm(image[:, :-1, :] - image[:, 1:, :], axis=-1)
    diff_v = np.linalg.norm(image[:-1, :, :] - image[1:, :, :], axis=-1)

    omega_h = np.exp(-(diff_h ** 2) / (sigma ** 2))
    omega_v = np.exp(-(diff_v ** 2) / (sigma ** 2))

    src_h = flat[:, :-1].ravel()
    dst_h = flat[:, 1:].ravel()
    for k1, k2, w in zip(src_h, dst_h, omega_h.ravel()):
        graph.add_edge(int(k1), int(k2), w * pairwise)

    src_v = flat[:-1, :].ravel()
    dst_v = flat[1:, :].ravel()
    for k1, k2, w in zip(src_v, dst_v, omega_v.ravel()):
        graph.add_edge(int(k1), int(k2), w * pairwise)

    return graph

def image_from_marginals(image, marginals):
    H, W = image.shape[:2]
    im = np.zeros((H, W))
    
    for k, (_, x) in marginals.items():
        i = k // W
        j = k % W
        im[i, j] = 0 if x < 0.5 else 1
    
    return im