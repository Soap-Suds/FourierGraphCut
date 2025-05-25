"""
Graph generation utilities for benchmarking Max-k-Cut algorithms
"""

import numpy as np
import networkx as nx
import random


def generate_random_graph(n, p, weights=None, seed=None):
    """
    Generate a single random graph
    
    Parameters:
    n (int): Number of vertices
    p (float): Edge probability
    weights (tuple): (min_weight, max_weight) for weighted graphs
    seed (int): Random seed
    
    Returns:
    numpy.ndarray: Adjacency matrix
    networkx.Graph: Graph object
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate random graph
    G = nx.gnp_random_graph(n, p)
    
    if weights is not None:
        for (u, v) in G.edges():
            G[u][v]['weight'] = random.uniform(weights[0], weights[1])
    
    A = nx.to_numpy_array(G, dtype=float)
    
    return A, G


def create_dataset(name, sizes, edge_probs, weights=None, seed=None):
    """
    Create a dataset of random graphs
    
    Parameters:
    name (str): Dataset name
    sizes (list): List of graph sizes
    edge_probs (list): List of edge probabilities
    weights (tuple): (min_weight, max_weight) for weighted graphs
    seed (int): Random seed
    
    Returns:
    dict: Dataset with metadata and graphs
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    dataset = {
        'name': name,
        'weighted': weights is not None,
        'graphs': []
    }
    
    for n, p in zip(sizes, edge_probs):
        A, G = generate_random_graph(n, p, weights)
        dataset['graphs'].append({
            'n': n,
            'p': p,
            'adjacency': A,
            'networkx': G
        })
    
    return dataset


def generate_benchmark_datasets(m=50, v_min=15, v_max=20, w_min=0.1, w_max=1.0, seed=None):
    """
    Generate standard benchmark datasets
    
    Parameters:
    m (int): Number of graphs per dataset
    v_min (int): Minimum vertices
    v_max (int): Maximum vertices
    w_min (float): Minimum edge weight
    w_max (float): Maximum edge weight
    seed (int): Random seed
    
    Returns:
    dict: Dictionary of unweighted datasets
    dict: Dictionary of weighted datasets
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate vertex sizes
    sizes = [random.randint(v_min, v_max) for _ in range(m)]
    
    # Dataset configurations
    configs = {
        'Isolated': lambda n: 1/n,
        'Transitioning': lambda n: np.log2(n)/n,
        'Extremely Sparse': lambda n: 1/np.sqrt(n),
        'Very Sparse': lambda n: 0.1,
        'Moderately Sparse': lambda n: 0.25,
        'Moderately Dense': lambda n: 0.5,
        'Very Dense': lambda n: 0.75,
    }
    
    unweighted_datasets = {}
    weighted_datasets = {}
    
    for name, prob_func in configs.items():
        edge_probs = [prob_func(n) for n in sizes]
        
        # Create unweighted dataset
        unweighted_datasets[name] = create_dataset(
            name, sizes, edge_probs, weights=None, seed=seed
        )
        
        # Create weighted dataset
        weighted_datasets[name] = create_dataset(
            name, sizes, edge_probs, weights=(w_min, w_max), seed=seed
        )
    
    return unweighted_datasets, weighted_datasets


def load_real_world_graph(path):
    """Load a real-world graph from file"""
    # Implement based on your file format
    pass


def generate_special_graphs(n):
    """Generate special graph structures for testing"""
    graphs = {}
    
    # Complete graph
    graphs['complete'] = nx.to_numpy_array(nx.complete_graph(n))
    
    # Cycle
    graphs['cycle'] = nx.to_numpy_array(nx.cycle_graph(n))
    
    # Star
    graphs['star'] = nx.to_numpy_array(nx.star_graph(n-1))
    
    # Path
    graphs['path'] = nx.to_numpy_array(nx.path_graph(n))
    
    # Random regular
    if n % 2 == 0:  # Regular graphs need even n for odd degree
        graphs['regular'] = nx.to_numpy_array(nx.random_regular_graph(3, n))
    
    return graphs
