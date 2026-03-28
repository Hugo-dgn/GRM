import networkx as nx
import matplotlib.pyplot as plt

import numpy as np

from skimage.segmentation import mark_boundaries

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

def plot_mask(image, mask):
    mask = np.array(mask)
    masked = np.ma.masked_where(mask == 0, mask)
    plt.imshow(np.array(image))
    plt.imshow(masked, cmap='jet', alpha=0.5)
    plt.axis('off')
    plt.show()