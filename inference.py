from collections import defaultdict

import numpy as np
from tqdm.auto import tqdm

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


def loopy_bp(tree, max_iter, alpha = 0.5):
    n = tree.n
    messages = defaultdict(lambda : np.ones(n) / n)
    for _ in range(max_iter):
        for node in tree.nodes:
            for target in tree.adj[node]:
                h = tree.phi[node]
                for neigh in tree.adj[node]:
                    if neigh == target:
                        continue
                    h = h * messages[(neigh, node)]
                    
                msg = tree.psi[(node, target)].T @ h
                msg /= (msg.sum() + 1e-10)
                msg = (1-alpha)*msg + alpha*messages[(node,target)]
                messages[(node, target)] = msg
    
    marginals = {}
    
    for node in tree.nodes:
        p = tree.phi[node]
        for neigh in tree.adj[node]:
            p = p * messages[(neigh, node)]
        
        p = p / p.sum()
        marginals[node] = p
    
    return marginals