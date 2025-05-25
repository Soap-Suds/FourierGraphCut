"""
Implementation of various Max-k-Cut algorithms including:
- Goemans-Williamson SDP relaxation
- Spectral clustering approach
"""

import numpy as np
import cvxpy as cvx
from scipy import linalg
from sklearn.cluster import KMeans


def goemans_williamson_max_cut(W):
    """
    Implements the Goemans-Williamson algorithm for max-cut approximation
    
    Parameters:
    W (numpy.ndarray): Symmetric weight matrix where W[i,j] is the weight of edge (i,j)
    
    Returns:
    list: Binary partition of vertices (0 or 1 for each vertex)
    float: Value of the cut
    """
    n = W.shape[0]
    
    # Step 1: Solve the SDP relaxation using CVXPY
    X = cvx.Variable((n, n), symmetric=True)
    
    # Vectorized constraints: X is PSD and diagonal elements are 1
    constraints = [
        X >> 0,  # X is positive semidefinite
        cvx.diag(X) == np.ones(n)  # All diagonal elements equal to 1
    ]
    
    # Vectorized objective function
    objective = 0.25 * cvx.sum(cvx.multiply(W, (1 - X)))
    prob = cvx.Problem(cvx.Maximize(objective), constraints)
    
    # Solve the SDP
    prob.solve()
    
    if prob.status != cvx.OPTIMAL:
        raise ValueError("SDP relaxation failed to solve optimally")
    
    X_opt = X.value
    
    # Step 2: Get the vector representation using eigendecomposition
    eigvals, eigvecs = linalg.eigh(X_opt)
    
    # Filter out negative or very small eigenvalues (numerical issues)
    tol = 1e-8
    pos_indices = eigvals > tol
    eigvals_filtered = eigvals[pos_indices]
    eigvecs_filtered = eigvecs[:, pos_indices]
    
    # Construct the vectors: V = U * sqrt(D)
    vectors = eigvecs_filtered @ np.diag(np.sqrt(eigvals_filtered))
    
    # Step 3: Random hyperplane rounding
    r = np.random.randn(vectors.shape[1])
    r = r / np.linalg.norm(r)
    
    # Vectorized dot product and sign operation
    signs = np.sign(vectors @ r)
    
    # Handle possible zeros in signs
    if np.any(signs == 0):
        signs[signs == 0] = np.random.choice([-1, 1], size=np.sum(signs == 0))
    
    # Convert from {-1, 1} to {0, 1}
    partition = ((signs + 1) / 2).astype(int)
    
    # Step 4: Calculate the cut value
    cut_value = calculate_cut_value(W, partition)
    
    return partition.tolist(), cut_value


def max_k_cut_sdp(W, k):
    """
    Implements a generalized Goemans-Williamson algorithm for max-k-cut approximation
    
    Parameters:
    W (numpy.ndarray): Symmetric weight matrix where W[i,j] is the weight of edge (i,j)
    k (int): Number of partitions
    
    Returns:
    list: Partition assignment for each vertex (values 0 to k-1)
    float: Value of the cut
    """
    if k == 2:
        return goemans_williamson_max_cut(W)
    
    n = W.shape[0]
    
    # Step 1: Solve the SDP relaxation using CVXPY
    X = cvx.Variable((n, n), symmetric=True)
    
    # Vectorized constraints
    constraints = [
        X >> 0,  # X is positive semidefinite
        cvx.diag(X) == np.ones(n)  # All diagonal elements equal to 1
    ]
    
    # For k>2, the correct SDP relaxation (Frieze and Jerrum, 1997)
    objective = cvx.sum(cvx.multiply(W, (1 - X/(k-1))))
    prob = cvx.Problem(cvx.Maximize(objective), constraints)
    
    # Solve the SDP
    prob.solve()
    
    if prob.status != cvx.OPTIMAL:
        raise ValueError("SDP relaxation failed to solve optimally")
    
    X_opt = X.value
    
    # Step 2: Get the vector representation using eigendecomposition
    eigvals, eigvecs = linalg.eigh(X_opt)
    
    # Filter out negative or very small eigenvalues
    tol = 1e-8
    pos_indices = eigvals > tol
    eigvals_filtered = eigvals[pos_indices]
    eigvecs_filtered = eigvecs[:, pos_indices]
    
    # Construct the vectors: V = U * sqrt(D)
    vectors = eigvecs_filtered @ np.diag(np.sqrt(eigvals_filtered))
    
    # Step 3: Use k-means clustering for k>2
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    partition = kmeans.fit_predict(vectors)
    
    # Step 4: Calculate the cut value
    cut_value = calculate_cut_value(W, partition)
    
    return partition.tolist(), cut_value


def spectral_cut(A, k, max_iter=1000):
    """
    Spectral clustering approach to max-k-cut
    
    Parameters:,,
    A (numpy.ndarray): Adjacency matrix
    k (int): Number of partitions
    max_iter (int): Maximum iterations for k-means
    
    Returns:
    list: Partition assignment for each vertex
    float: Value of the cut
    """
    n = A.shape[0]
    
    # Calculate degrees of vertices
    degrees = np.sum(A, axis=1)
    
    # Create the degree matrix
    D = np.diag(degrees)
    
    # Calculate the Laplacian matrix
    L = D - A
    
    # Get the eigenvectors of the Laplacian
    eigvals, eigvecs = np.linalg.eigh(L)
    
    idx = np.argsort(eigvals)[-k:-1]  # Get k-1 largest (excluding the very largest)
    
    # Get the eigenvectors corresponding to these eigenvalues
    X = eigvecs[:, idx]
    
    # Use sklearn's KMeans for consistency
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, 
                    max_iter=max_iter, random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Calculate cut value
    cut_value = calculate_cut_value(A, clusters)
    
    return clusters.tolist(), cut_value

    


def calculate_cut_value(W, partition):
    """
    Calculate the value of a k-cut given a partition
    
    Parameters:
    W (numpy.ndarray): Weight matrix
    partition (array-like): Partition assignment for each vertex
    
    Returns:
    float: Value of the cut
    """
    n = W.shape[0]
    partition = np.array(partition)
    
    # Create a matrix where entry is 1 if vertices are in different partitions
    different_partitions = (partition.reshape(-1, 1) != partition).astype(int)
    
    # Use only upper triangular part (to count each edge once)
    upper_tri_mask = np.triu(np.ones((n, n)), k=1).astype(bool)
    
    # Multiply by the weight matrix and sum
    cut_value = np.sum(W[upper_tri_mask] * different_partitions[upper_tri_mask])
    
    return float(cut_value)


# Additional baseline methods

def random_cut(W, k):
    """Random partition baseline"""
    n = W.shape[0]
    partition = np.random.randint(0, k, size=n)
    cut_value = calculate_cut_value(W, partition)
    return partition.tolist(), cut_value


def greedy_local_search(W, k, max_iter=100):
    """Simple greedy local search"""
    n = W.shape[0]
    
    # Start with random partition
    partition = np.random.randint(0, k, size=n)
    cut_value = calculate_cut_value(W, partition)
    
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            best_part = partition[i]
            best_value = cut_value
            
            # Try moving vertex i to each partition
            for j in range(k):
                if j != partition[i]:
                    partition[i] = j
                    new_value = calculate_cut_value(W, partition)
                    if new_value > best_value:
                        best_value = new_value
                        best_part = j
                        improved = True
            
            partition[i] = best_part
            cut_value = best_value
        
        if not improved:
            break
    
    return partition.tolist(), cut_value
