'''
Created on 1 Jun 2020

@author: snake91
'''


import networkx as nx
import numpy as np
import itertools as it
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd


def reshape(p):

    ltriangle = np.array(p)
    n = len(nodes)
    idx = np.tril_indices(n, k=-1, m=n)
    matrix = np.zeros((n,n)).astype(float)
    matrix[idx] = ltriangle
    
    matrix = matrix + matrix.T
    
    return matrix

flatten = lambda l: [item for sublist in l for item in sublist]


if __name__ == "__main__":
    
    nodes = np.random.normal(size = (100,2))
    perm = it.combinations(nodes, 2) 
    
    perm = np.array(list(perm))
    
    undirected = True 
    
    dist = np.array(list(map(lambda x: np.sqrt((x[0][0] - x[0][1])**2 + (x[1][0] - x[1][1])**2), perm)))

    mean = np.mean(dist)
    var = np.var(dist)
#  
    p = st.chi2.cdf(dist, df = 1)
    puniform = np.random.uniform(size = len(dist))
#     
    p = np.array([1 if p[i] > puniform[i] else 0 for i in range(len(p))]) #edges that should get created



    
    matrix = reshape(dist)
    adj_matrix = reshape(p)
    
    graph = nx.from_numpy_matrix(adj_matrix)
    
    nx.set_node_attributes(graph, 0, 'infected')
    nx.set_node_attributes(graph, 0, 'position')
    
    pos = {i : np.array([nodes[i][0], nodes[i][1]]) for i in range(len(graph.nodes.keys()))}
    nx.set_node_attributes(graph, pos, 'position')

    infected0 = st.binom.rvs(1, 0.05, size = len(graph.nodes))   
    cmap_infected = np.where(infected0 == 1, "red", "blue")
    
#     infected_idx = np.where(infected0 == 1)[0]
    
    infected = {i : infected0[i] for i in range(len(infected0)) }
    nx.set_node_attributes(graph, infected, "infected")
    
#     nx.draw_networkx(graph, pos, node_size = 10, with_labels= False, width = 0, node_color =  cmap_infected)
#     plt.show()

    c = 0
    while True:

        infected = {i : graph.nodes[i]["infected"] for i in graph.nodes.keys() if graph.nodes[i]["infected"] == 1}

        neighbors = [list(graph.neighbors(i)) for i in infected.keys()]
        
        
        
        cmap_infected = np.where(infected == 1, "red", "blue")
#         nx.draw_networkx(graph, pos, node_size = 10, with_labels= False, width = 0, node_color =  cmap_infected)
    
        nodes = np.random.normal(size = (100,2))
        perm = it.combinations(nodes, 2)        
        
        graph = nx.from_numpy_matrix(adj_matrix)
        
        pos = {i : np.array([nodes[i][0], nodes[i][1]]) for i in range(len(graph.nodes.keys()))}
        nx.set_node_attributes(graph, pos, 'position')

        infected_candidates = set(flatten(neighbors))
        
        infected = np.random.uniform(size = len(infected_candidates))
        infected = list(map(lambda x: 0 if x < 0.5 else 1, infected))
    
    
    
        print("")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    