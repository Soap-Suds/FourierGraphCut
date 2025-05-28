"""
Memory-efficient benchmarking framework for Max-k-Cut algorithms
"""

import numpy as np
import pandas as pd
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Tuple
import json
import gc

from greedy_cut import GreedyFourierSearch
from other_methods import (
    goemans_williamson_max_cut, max_k_cut_sdp, spectral_cut,
    random_cut, greedy_local_search
)


class MemoryEfficientBenchmark:
    """Memory-efficient benchmarking class for Max-k-Cut algorithms"""
    
    def __init__(self, save_partitions=False):
        self.algorithms = {
            'greedy': self._run_greedy,
            'spectral': self._run_spectral,
            'gw_sdp': self._run_gw_sdp,
            'random': self._run_random,
            'local_search': self._run_local_search
        }
        self.save_partitions = save_partitions
        # Don't store full results in memory - write to disk immediately
        self.stats_buffer = []
        self.buffer_size = 10  # Write to disk every 10 results
    
    def _run_greedy(self, A: np.ndarray, k: int) -> Tuple[List[int], float, float]:
        """Run greedy Fourier search algorithm"""
        start_time = time.time()
        solver = GreedyFourierSearch(A, k)
        partition, cut_value = solver.solve()
        runtime = time.time() - start_time
        
        # Force garbage collection after greedy algorithm
        del solver
        gc.collect()
        
        return partition, cut_value, runtime
    
    def _run_spectral(self, A: np.ndarray, k: int) -> Tuple[List[int], float, float]:
        """Run spectral clustering algorithm"""
        start_time = time.time()
        partition, cut_value = spectral_cut(A, k)
        runtime = time.time() - start_time
        
        # Force garbage collection
        gc.collect()
        
        return partition, cut_value, runtime
    
    def _run_gw_sdp(self, A: np.ndarray, k: int) -> Tuple[List[int], float, float]:
        """Run Goemans-Williamson SDP algorithm"""
        start_time = time.time()
        partition, cut_value = max_k_cut_sdp(A, k)
        runtime = time.time() - start_time
        return partition, cut_value, runtime
    
    def _run_random(self, A: np.ndarray, k: int) -> Tuple[List[int], float, float]:
        """Run random partition baseline"""
        start_time = time.time()
        partition, cut_value = random_cut(A, k)
        runtime = time.time() - start_time
        return partition, cut_value, runtime
    
    def _run_local_search(self, A: np.ndarray, k: int) -> Tuple[List[int], float, float]:
        """Run greedy local search"""
        start_time = time.time()
        partition, cut_value = greedy_local_search(A, k)
        runtime = time.time() - start_time
        return partition, cut_value, runtime
    
    def add_algorithm(self, name: str, func: Callable):
        """Add a custom algorithm to benchmark"""
        self.algorithms[name] = func
    
    def run_single_test(self, A: np.ndarray, k: int, algorithms: List[str] = None) -> Dict:
        """
        Run a single test on one graph
        Only returns cut values and runtimes (not partitions unless save_partitions=True)
        """
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        
        results = {}
        for alg_name in algorithms:
            if alg_name not in self.algorithms:
                print(f"Warning: Algorithm {alg_name} not found")
                continue
            
            try:
                partition, cut_value, runtime = self.algorithms[alg_name](A, k)
                
                # Only save essential information
                results[alg_name] = {
                    'cut_value': cut_value,
                    'runtime': runtime
                }
                
                # Only save partition if explicitly requested
                if self.save_partitions:
                    results[alg_name]['partition'] = partition
                
                # Clear partition from memory if not needed
                del partition
                
            except Exception as e:
                print(f"Error running {alg_name}: {str(e)}")
                results[alg_name] = {
                    'cut_value': None,
                    'runtime': None,
                    'error': str(e)
                }
        
        # Force garbage collection after each test
        gc.collect()
        
        return results
    
    def run_dataset_test_streaming(self, dataset: Dict, k_values: List[int], 
                                  algorithms: List[str], output_dir: str):
        """
        Run tests on dataset with streaming output to disk
        Processes one graph at a time and writes results immediately
        """
        dataset_name = dataset['name']
        results_file = f"{output_dir}/{dataset_name}_streaming_results.jsonl"
        
        with open(results_file, 'w') as f:
            for i, graph_data in enumerate(dataset['graphs']):
                A = graph_data['adjacency']
                n = graph_data['n']
                
                print(f"\rProcessing graph {i+1}/{len(dataset['graphs'])}", end='', flush=True)
                
                for k in k_values:
                    if k > n:
                        continue
                    
                    test_results = self.run_single_test(A, k, algorithms)
                    
                    # Write results immediately to disk
                    result_entry = {
                        'graph_index': i,
                        'n': n,
                        'p': graph_data['p'],
                        'k': k,
                        'results': test_results
                    }
                    
                    f.write(json.dumps(result_entry) + '\n')
                    f.flush()  # Ensure it's written to disk
                
                # Clear the adjacency matrix from memory
                del A
                gc.collect()
        
        print()  # New line after progress
        return results_file
    
    def compute_statistics_from_file(self, results_file: str) -> pd.DataFrame:
        """
        Compute statistics from streaming results file
        """
        stats_data = defaultdict(lambda: defaultdict(list))
        
        with open(results_file, 'r') as f:
            for line in f:
                entry = json.loads(line)
                k = entry['k']
                
                for alg_name, alg_results in entry['results'].items():
                    if alg_results['cut_value'] is not None:
                        stats_data[k][alg_name].append({
                            'cut_value': alg_results['cut_value'],
                            'runtime': alg_results['runtime']
                        })
        
        # Compute statistics
        stats_list = []
        for k, k_data in stats_data.items():
            for alg_name, results in k_data.items():
                cut_values = [r['cut_value'] for r in results]
                runtimes = [r['runtime'] for r in results]
                
                stats_list.append({
                    'k': k,
                    'algorithm': alg_name,
                    'mean_cut': np.mean(cut_values),
                    'median_cut': np.median(cut_values),
                    'std_cut': np.std(cut_values),
                    'mean_runtime': np.mean(runtimes),
                    'total_runtime': np.sum(runtimes)
                })
        
        return pd.DataFrame(stats_list)


# Memory-efficient versions of algorithms

def memory_efficient_spectral_cut(A, k, max_iter=1000):
    """
    Memory-efficient spectral clustering for large graphs
    Uses sparse matrices and iterative eigensolvers when beneficial
    """
    n = A.shape[0]
    
    # For very large graphs, consider using sparse matrices
    if n > 500:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        # Convert to sparse if beneficial
        sparsity = np.count_nonzero(A) / (n * n)
        if sparsity < 0.3:  # If less than 30% non-zero
            A_sparse = csr_matrix(A)
            degrees = np.array(A_sparse.sum(axis=1)).flatten()
            D_sparse = csr_matrix((degrees, (range(n), range(n))), shape=(n, n))
            L_sparse = D_sparse - A_sparse
            
            # Use iterative solver for largest eigenvalues
            eigvals, eigvecs = eigsh(L_sparse, k=k, which='LA')
            idx = np.argsort(eigvals)[1:]  # Exclude smallest
            X = eigvecs[:, idx]
        else:
            # Use original method for dense graphs
            return spectral_cut(A, k, max_iter)
    else:
        # Use original method for small graphs
        return spectral_cut(A, k, max_iter)
    
    # Rest of the algorithm remains the same
    from sklearn.cluster import MiniBatchKMeans
    
    # Use MiniBatchKMeans for large datasets (more memory efficient)
    kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', n_init=10, 
                             batch_size=min(100, n), random_state=42)
    clusters = kmeans.fit_predict(X)
    
    # Calculate cut value
    from other_methods import calculate_cut_value
    cut_value = calculate_cut_value(A, clusters)
    
    return clusters.tolist(), cut_value