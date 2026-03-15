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