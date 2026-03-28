import numpy as np
from sklearn.base import BaseEstimator
from skimage.segmentation import slic

import models
    
    
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

def image_from_super_marginals(image, marginals, segments):
    mask = np.zeros(image.shape[:2])
    for pixel, label in marginals.items():
        mask[segments==pixel] = np.argmax(label)
    return mask


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

def mean_agg(x):
    return np.mean(x, axis=0)

def process_super_pixel(image, mask, segments, pixels, threshold=0.5, agg_func=mean_agg):
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
        features = np.stack([agg_func(image[segments == p]) for p in pixels])
    else:
        features = None
    
    return features, labels, super_mask

def get_super_pixels_graph(model, image, beta, sigma, n_segments=200, compactness=10, start_label=0, transform=None, agg_func=None):
    graph = models.Graph(2)
    
    pairwise = np.exp(-np.array([[0, beta],
                                 [beta, 0]]))
    segments, pixels, edges = get_superpixels(image, n_segments, compactness, start_label)
    
    if agg_func is None:
        agg_func = mean_agg
    X, _, _ = process_super_pixel(image, None, segments, pixels, agg_func=agg_func)
    
    if transform is not None:
        X = transform.transform(X)
    
    probas = model.predict_proba(X)
    probas /= probas.sum(axis=-1, keepdims=True) + 1e-10

    for i, k in enumerate(pixels):
        graph.add_node(k, probas[i])
    
    for k1, k2 in edges:
        i1 = np.where(pixels==k1)[0][0]
        i2 = np.where(pixels==k2)[0][0]
        
        x1 = X[i1]
        x2 = X[i2]
        
        omega = np.exp(-np.linalg.norm(((x1 - x2)) ** 2) / (sigma ** 2))
        graph.add_edge(int(k1), int(k2), omega*pairwise)
    
    return graph, segments, pixels, edges

def IoU(mask1, mask2):
    mask1 = np.array(mask1, dtype=float)
    mask2 = np.array(mask2, dtype=float)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    return np.sum(intersection) / np.sum(union)

def preprocess(mask):
    return mask.point(lambda p: 1 if p == 1 or p == 3 else 0)

def extract_super_features(dataset, n_segments, compactness, agg_func): 
    features = []
    labels = []

    for image, (mask, cat) in dataset:
        mask = preprocess(mask)
        segments, pixels, edges = get_superpixels(image, n_segments=n_segments, compactness=compactness)
        feature, label, _ = process_super_pixel(image, mask, segments, pixels, agg_func=agg_func)
        features.append(feature)
        labels.append(label)

    features = np.concatenate(features)
    labels = np.concatenate(labels)
    
    return features, labels

def balance_data(features, labels):
    idx_0 = np.where(labels == 0)[0]
    idx_1 = np.where(labels == 1)[0]

    n = min(len(idx_0), len(idx_1))

    idx_0_balanced = np.random.choice(idx_0, size=n, replace=False)
    idx_1_balanced = np.random.choice(idx_1, size=n, replace=False)

    balanced_idx = np.concatenate([idx_0_balanced, idx_1_balanced])
    np.random.shuffle(balanced_idx)

    features = features[balanced_idx]
    labels = labels[balanced_idx]
    
    return features, labels