"""
Benchmarking framework for Max-k-Cut algorithms
"""

import numpy as np
import pandas as pd
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Callable, Tuple
import json

from greedy_cut import GreedyFourierSearch
from other_methods import (
    goemans_williamson_max_cut, max_k_cut_sdp, spectral_cut,
    random_cut, greedy_local_search
)


class MaxKCutBenchmark:
    """Main benchmarking class for Max-k-Cut algorithms"""
    
    def __init__(self):
        self.algorithms = {
            'greedy': self._run_greedy,
            'spectral': self._run_spectral,
            'gw_sdp': self._run_gw_sdp,
            'random': self._run_random,
            'local_search': self._run_local_search
        }
        self.results = defaultdict(lambda: defaultdict(dict))
    
    def _run_greedy(self, A: np.ndarray, k: int) -> Tuple[List[int], float, float]:
        """Run greedy Fourier search algorithm"""
        start_time = time.time()
        solver = GreedyFourierSearch(A, k)
        partition, cut_value = solver.solve()
        runtime = time.time() - start_time
        return partition, cut_value, runtime
    
    def _run_spectral(self, A: np.ndarray, k: int) -> Tuple[List[int], float, float]:
        """Run spectral clustering algorithm"""
        start_time = time.time()
        partition, cut_value = spectral_cut(A, k)
        runtime = time.time() - start_time
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
        
        Parameters:
        A: Adjacency matrix
        k: Number of partitions
        algorithms: List of algorithm names to run (None = all)
        
        Returns:
        Dictionary of results
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
                results[alg_name] = {
                    'partition': partition,
                    'cut_value': cut_value,
                    'runtime': runtime
                }
            except Exception as e:
                print(f"Error running {alg_name}: {str(e)}")
                results[alg_name] = {
                    'partition': None,
                    'cut_value': None,
                    'runtime': None,
                    'error': str(e)
                }
        
        return results
    
    def run_dataset_test(self, dataset: Dict, k_values: List[int], 
                        algorithms: List[str] = None) -> Dict:
        """
        Run tests on an entire dataset
        
        Parameters:
        dataset: Dataset dictionary from graph_generation
        k_values: List of k values to test
        algorithms: List of algorithm names to run
        
        Returns:
        Dictionary of results organized by k and algorithm
        """
        dataset_name = dataset['name']
        results = defaultdict(lambda: defaultdict(list))
        
        for graph_data in dataset['graphs']:
            A = graph_data['adjacency']
            n = graph_data['n']
            
            for k in k_values:
                if k > n:
                    continue
                
                test_results = self.run_single_test(A, k, algorithms)
                
                for alg_name, alg_results in test_results.items():
                    results[k][alg_name].append(alg_results)
        
        self.results[dataset_name] = results
        return results
    
    def compute_statistics(self, results: Dict) -> pd.DataFrame:
        """
        Compute statistics from results
        
        Parameters:
        results: Results dictionary from run_dataset_test
        
        Returns:
        DataFrame with statistics
        """
        stats_data = []
        
        for k, k_results in results.items():
            for alg_name, alg_results in k_results.items():
                cut_values = [r['cut_value'] for r in alg_results if r['cut_value'] is not None]
                runtimes = [r['runtime'] for r in alg_results if r['runtime'] is not None]
                
                if cut_values:
                    stats_data.append({
                        'k': k,
                        'algorithm': alg_name,
                        'mean_cut': np.mean(cut_values),
                        'median_cut': np.median(cut_values),
                        'std_cut': np.std(cut_values),
                        'mean_runtime': np.mean(runtimes) if runtimes else None,
                        'total_runtime': np.sum(runtimes) if runtimes else None
                    })
        
        return pd.DataFrame(stats_data)
    
    def compare_algorithms(self, dataset_results: Dict) -> pd.DataFrame:
        """
        Compare algorithm performance across all tests
        
        Parameters:
        dataset_results: Results from multiple datasets
        
        Returns:
        DataFrame comparing algorithms
        """
        comparison_data = []
        
        for dataset_name, results in dataset_results.items():
            stats = self.compute_statistics(results)
            stats['dataset'] = dataset_name
            comparison_data.append(stats)
        
        return pd.concat(comparison_data, ignore_index=True)
    
    def plot_results(self, stats_df: pd.DataFrame, metric: str = 'mean_cut'):
        """
        Plot benchmark results
        
        Parameters:
        stats_df: Statistics DataFrame
        metric: Metric to plot ('mean_cut', 'mean_runtime', etc.)
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot by k value
        for k in stats_df['k'].unique():
            k_data = stats_df[stats_df['k'] == k]
            axes[0].bar(k_data['algorithm'], k_data[metric], 
                       label=f'k={k}', alpha=0.7)
        
        axes[0].set_xlabel('Algorithm')
        axes[0].set_ylabel(metric)
        axes[0].set_title(f'{metric} by Algorithm and k')
        axes[0].legend()
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot runtime vs quality
        if 'mean_runtime' in stats_df.columns:
            for alg in stats_df['algorithm'].unique():
                alg_data = stats_df[stats_df['algorithm'] == alg]
                axes[1].scatter(alg_data['mean_runtime'], alg_data['mean_cut'], 
                              label=alg, s=100)
        
        axes[1].set_xlabel('Mean Runtime (s)')
        axes[1].set_ylabel('Mean Cut Value')
        axes[1].set_title('Runtime vs Quality Trade-off')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename: str):
        """Save results to JSON file"""
        # Convert results to serializable format
        serializable_results = {}
        for dataset, k_results in self.results.items():
            serializable_results[dataset] = {}
            for k, alg_results in k_results.items():
                serializable_results[dataset][str(k)] = {}
                for alg, results_list in alg_results.items():
                    serializable_results[dataset][str(k)][alg] = results_list
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
    
    def load_results(self, filename: str):
        """Load results from JSON file"""
        with open(filename, 'r') as f:
            loaded = json.load(f)
        
        # Convert back to defaultdict structure
        self.results = defaultdict(lambda: defaultdict(dict))
        for dataset, k_results in loaded.items():
            for k_str, alg_results in k_results.items():
                k = int(k_str)
                for alg, results_list in alg_results.items():
                    self.results[dataset][k][alg] = results_list