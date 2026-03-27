import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import BaseEstimator
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic

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
                                  beta: float, sigma: float, transform=None):
    H, W, C = image.shape
    N = H * W
    graph = models.Graph(2)

    flat = np.arange(N).reshape(H, W)

    X_flat = image.reshape(-1, C)
    
    if transform is not None:
        X_flat = transform.transform(X_flat)
    
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


def get_superpixels(image, n_segments=200, compactness=10, start_label=0):
    segments = slic(np.array(image), n_segments=n_segments, compactness=compactness, start_label=start_label)

    pixels = np.unique(segments)
    edges_h = np.stack([segments[:, :-1], segments[:, 1:]], axis=-1)
    edges_h = edges_h[edges_h[:, :, 0] != edges_h[:, :, 1]]
    edges_v = np.stack([segments[:-1, :], segments[1:, :]], axis=-1)
    edges_v = edges_v[edges_v[:, :, 0] != edges_v[:, :, 1]]
    edges = np.vstack((edges_h.reshape(-1, 2), edges_v.reshape(-1, 2)))
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    
    return segments, pixels, edges

def process_super_pixel(image, mask, segments, pixels, threshold=0.5):
    
    if mask is not None:
        mask = np.array(mask)
        labels = np.stack([mask[segments == p].mean() for p in pixels]) > threshold
        super_mask = np.zeros_like(mask)
        for i, p in enumerate(pixels):
            super_mask[segments == p] = labels[i]
    else:
        labels = None
        super_mask = None
    if image is not None:
        image = np.array(image)
        features = np.stack([image[segments == p].mean(axis=0) for p in pixels])
    else:
        features = None
    
    return features, labels, super_mask

def get_super_pixels_graph(model, image, beta, n_segments=200, compactness=10, start_label=0, transform=None):
    graph = models.Graph(2)
    
    pairwise = np.exp(-np.array([[0, beta],
                                 [beta, 0]]))
    segments, pixels, edges = get_superpixels(image, n_segments, compactness, start_label)
    
    X, _, _ = process_super_pixel(image, None, segments, pixels)
    
    if transform is not None:
        X = transform.transform(X)
    
    probas = model.predict_proba(X)
    probas /= probas.sum(axis=-1, keepdims=True) + 1e-10

    for i, k in enumerate(pixels):
        graph.add_node(k, probas[i])
    
    for k1, k2 in edges:
        graph.add_edge(int(k1), int(k2), pairwise)
    
    return graph

def compute_centroids(segments):
    centroids = {}
    
    for label in np.unique(segments):
        coords = np.column_stack(np.where(segments == label))
        y, x = coords.mean(axis=0)  # careful: (row, col) = (y, x)
        centroids[label] = (x, y)   # switch to (x, y) for plotting
        
    return centroids

def plot_superpixels(image, segments, edges):
    centroids = compute_centroids(segments)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
 
    for u, v in edges:
        x1, y1 = centroids[u]
        x2, y2 = centroids[v]
        plt.plot([x1, x2], [y1, y2], 'b-', linewidth=1)

    xs = [centroids[l][0] for l in centroids]
    ys = [centroids[l][1] for l in centroids]
    plt.scatter(xs, ys, c='red', s=10)
    
    plt.imshow(mark_boundaries(image, segments))

    plt.axis('off')
    plt.show()