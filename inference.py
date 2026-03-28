from collections import defaultdict
from tqdm.auto import tqdm

import numpy as np

import utils

def tree_bp(tree, root):
    
    messages = {}
    
    def collect(node, parent):
        
        h = tree.phi[node]
        for neigh in tree.adj[node]:
            if neigh == parent:
                continue
            collect(neigh, node)
            h = h * messages[(neigh, node)]
        
        if parent is None:
            return
        msg = tree.psi[(node, parent)].T @ h
        msg /= msg.sum()
        messages[(node, parent)] = msg
    
    collect(root, None)
    
    marginals = {}
    
    def backward(node, parent):
        
        
        msgs = {}
        
        p = tree.phi[node]
        for neigh in tree.adj[node]:
            msg = messages[(neigh, node)]
            p = p * msg
            msgs[neigh] = msg
        
        p = p / p.sum()
        marginals[node] = p
        
        for neigh in tree.adj[node]:
            if neigh == parent:
                continue
            h = tree.phi[node]
            for name, msg in msgs.items():
                if name == neigh:
                    continue
                h = h * msg
        
            msg = tree.psi[(node, neigh)].T @ h
            msg /= msg.sum()
            messages[(node, neigh)] = msg
            
            backward(neigh, node)
    
    backward(root, None)
    
    return marginals

def loopy_bp(graph, max_iter=50, tol=1e-3):
    n_states = graph.n
    nodes = graph.nodes
    adj = graph.adj
    phi = graph.phi
    psi = graph.psi

    uniform = np.ones(n_states) / n_states
    prev_messages = {u: {v: uniform.copy() for v in adj[u]} for u in nodes}

    for _ in range(max_iter):
        diffs = []

        for node in nodes:
            neighs = adj[node]
            phi_node = phi[node]

            if len(neighs) == 0:
                continue

            msgs_in = np.stack([prev_messages[nb][node] for nb in neighs], axis=0)
            
            total_prod = phi_node * msgs_in.prod(axis=0)

            for t_idx, target in enumerate(neighs):
                h = total_prod / np.clip(msgs_in[t_idx], 1e-10, np.inf)
                msg = psi[(node, target)].T @ h
                msg /= msg.sum() + 1e-10

                new_msg = msg
                diffs.append(np.max(np.abs(new_msg - prev_messages[node][target])))
                prev_messages[node][target] = new_msg

        if np.mean(diffs) < tol:
            break

    marginals = {}
    for node in nodes:
        neighs = adj[node]
        p = phi[node].copy()
        if neighs:
            p *= np.stack([prev_messages[nb][node] for nb in neighs]).prod(axis=0)
        p /= p.sum() + 1e-10
        marginals[node] = p

    return marginals


def trw_bp(graph, max_iter=50, tol=1e-3):
    n_states = graph.n
    nodes = graph.nodes
    adj = graph.adj
    phi = graph.phi
    psi = graph.psi
    rho = graph.rho

    uniform = np.ones(n_states) / n_states
    prev_messages = {u: {v: uniform.copy() for v in adj[u]} for u in nodes}

    for _ in range(max_iter):
        diffs = []

        for node in nodes:
            neighs = adj[node]
            phi_node = phi[node]

            if len(neighs) == 0:
                continue

            for t_idx, target in enumerate(neighs):
                rho_ij = rho[(node, target)]

                # Each incoming message is raised to its own rho weight
                # (rho_ki for neighbor k -> node i)
                weighted_msgs_in = np.stack([
                    prev_messages[nb][node] ** rho[(nb, node)]
                    for nb in neighs
                ], axis=0)

                # Cavity: product of all weighted incoming messages except from target
                cavity = phi_node * weighted_msgs_in.prod(axis=0)
                cavity /= np.clip(prev_messages[target][node] ** rho_ij, 1e-10, np.inf)

                # Edge potential is reweighted by rho_ij
                psi_rho = psi[(node, target)] ** rho_ij  # shape: (n_states, n_states)

                # Marginalize: new message is psi_rho.T @ cavity, then raised to 1/rho_ij
                raw = psi_rho.T @ cavity
                msg = raw ** (1.0 / rho_ij)
                msg /= msg.sum() + 1e-10

                diffs.append(np.max(np.abs(msg - prev_messages[node][target])))
                prev_messages[node][target] = msg

        if np.mean(diffs) < tol:
            break

    # Beliefs: incoming messages weighted by their own rho
    marginals = {}
    for node in nodes:
        neighs = adj[node]
        p = phi[node].copy()
        if neighs:
            p *= np.stack([
                prev_messages[nb][node] ** rho[(nb, node)]
                for nb in neighs
            ]).prod(axis=0)
        p /= p.sum() + 1e-10
        marginals[node] = p

    return marginals


def run_single_image(image, mask, model, max_iter, trw):
    mask = utils.preprocess(mask)
    image = np.array(image)

    predictions = model(image, max_iter, trw=trw)
    return utils.IoU(mask, predictions)

def sequential_segmentation(dataset, model, max_iter, trw):
    results = []
    for image, (mask, cat) in tqdm(dataset):
        iou = run_single_image(image, mask, model, max_iter, trw)
        results.append(iou)
    return np.array(results)