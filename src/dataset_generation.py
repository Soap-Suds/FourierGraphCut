import numpy as np
import networkx as nx
import random

def randdataset(N, m, D, dataset=None, seed=None, weights = None, name = None): 

    if seed is not None:
            random.seed(seed)  # Set seed for reproducibility
            np.random.seed(seed)

    if dataset is None:  # Initialize the dataset dictionary
        dataset = {}
    
    if m == 0:  # Base case: when no more graphs need to be generated
        return [name, dataset]
    
    # Generate a random graph
    G = nx.gnp_random_graph(N[-1], D[-1])  
    if weights is not None:
        for (u, v) in G.edges():
            G[u][v]['weight'] = random.uniform(weights[0], weights[1])
    
    N = N[:-1]  # Remove the last element of N
    D = D[:-1]  # Remove the last element of D
    A = nx.to_numpy_array(G, dtype=float)  
    
    # Store it in the dataset
    dataset[m - 1] = (A, G)  # Using m-1 to maintain consistency with the original for-loop

    # Recursive call for the remaining m-1 graphs
    return randdataset(N, m - 1, D, dataset, name = name)

def generate_unweighted_datasets(N, m, I = None, T = None, ES = None, VS = 1, MS = None, MD = None, VD = None):
    datasets = {}
    if I:
        datasets['Isolated'] = randdataset(N, m, [1/i for i in N], name='Isolated')
    if T:
        datasets['Transitioning to Connectivity'] = randdataset(N, m, [float(np.log2(i)/i) for i in N], name='Transitioning to Connectivity')
    if ES:
        datasets['Extremely Sparse'] = randdataset(N, m, [float(1/np.sqrt(i)) for i in N], name='Extremely Sparse')
    if VS:
        datasets['Very Sparse'] = randdataset(N, m, [0.1 for i in N], name='Very Sparse')
    if MS:
        datasets['Moderately Sparse'] = randdataset(N, m, [0.25 for i in N], name='Moderately Sparse')
    if MD:
        datasets['Moderately Dense'] = randdataset(N, m, [0.5 for i in N], name='Moderately Dense')
    if VD:
        datasets['Very Dense'] = randdataset(N, m, [0.75 for i in N], name='Very Dense')
    return datasets

def generate_weighted_datasets(N, m, w, W, I = None, T = None, ES = None, VS = 1, MS = None, MD = None, VD = None):
    datasets = {}
    if I:
        datasets['Isolated'] = randdataset(N, m, [1/i for i in N], weights=(w, W), name='Isolated')
    if T:
        datasets['Transitioning to Connectivity'] = randdataset(N, m, [float(np.log2(i)/i) for i in N], weights=(w, W), name='Transitioning to Connectivity')
    if ES:
        datasets['Extremely Sparse'] = randdataset(N, m, [float(1/np.sqrt(i)) for i in N], weights=(w, W), name='Extremely Sparse')
    if VS:
        datasets['Very Sparse'] = randdataset(N, m, [0.1 for i in N], weights=(w, W), name='Very Sparse')
    if MS:
        datasets['Moderately Sparse'] = randdataset(N, m, [0.25 for i in N], weights=(w, W), name='Moderately Sparse')
    if MD:
        datasets['Moderately Dense'] = randdataset(N, m, [0.5 for i in N], weights=(w, W), name='Moderately Dense')
    if VD:
        datasets['Very Dense'] = randdataset(N, m, [0.75 for i in N], weights=(w, W), name='Very Dense')
    return datasets
 