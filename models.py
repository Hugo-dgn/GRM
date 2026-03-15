import numpy as np
from collections import defaultdict

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