"""
Example usage of the benchmarking framework and how to add custom algorithms
"""

import numpy as np
from benchmarking import MaxKCutBenchmark
from graph_generation import generate_random_graph
from other_methods import calculate_cut_value
import time


def my_custom_algorithm(A, k):
    """
    Example custom algorithm - implement your own logic here
    
    Parameters:
    A: Adjacency matrix
    k: Number of partitions
    
    Returns:
    partition: List of partition assignments
    cut_value: Value of the cut
    runtime: Time taken
    """
    start_time = time.time()
    
    n = A.shape[0]
    
    # Your algorithm logic here...
    # This is just a placeholder example
    partition = np.random.randint(0, k, size=n)
    cut_value = calculate_cut_value(A, partition)
    
    runtime = time.time() - start_time
    return partition.tolist(), cut_value, runtime


def main():
    # Example 1: Running a single test
    print("Example 1: Single graph test")
    print("-" * 50)
    
    # Generate a random graph
    A, G = generate_random_graph(n=20, p=0.5, weights=None, seed=42)
    k = 3
    
    # Initialize benchmark
    benchmark = MaxKCutBenchmark()
    
    # Run single test
    results = benchmark.run_single_test(A, k, algorithms=['greedy', 'spectral'])
    
    for alg_name, alg_results in results.items():
        print(f"{alg_name}: cut_value = {alg_results['cut_value']:.2f}, "
              f"runtime = {alg_results['runtime']:.4f}s")
    
    # Example 2: Adding a custom algorithm
    print("\n\nExample 2: Adding custom algorithm")
    print("-" * 50)
    
    # Add custom algorithm to benchmark
    benchmark.add_algorithm('my_custom', my_custom_algorithm)
    
    # Run test including custom algorithm
    results = benchmark.run_single_test(A, k, algorithms=['greedy', 'my_custom'])
    
    for alg_name, alg_results in results.items():
        print(f"{alg_name}: cut_value = {alg_results['cut_value']:.2f}, "
              f"runtime = {alg_results['runtime']:.4f}s")
    
    # Example 3: Running on a small dataset
    print("\n\nExample 3: Small dataset test")
    print("-" * 50)
    
    # Create a small dataset
    small_dataset = {
        'name': 'small_test',
        'weighted': False,
        'graphs': []
    }
    
    for i in range(5):
        A, G = generate_random_graph(n=15, p=0.3, weights=None, seed=i)
        small_dataset['graphs'].append({
            'n': 15,
            'p': 0.3,
            'adjacency': A,
            'networkx': G
        })
    
    # Run benchmark on dataset
    results = benchmark.run_dataset_test(
        small_dataset, 
        k_values=[2, 3], 
        algorithms=['greedy', 'spectral', 'gw_sdp']
    )
    
    # Compute statistics
    stats = benchmark.compute_statistics(results)
    print("\nDataset Statistics:")
    print(stats.to_string())
    
    # Example 4: Analyzing specific properties
    print("\n\nExample 4: Analyzing algorithm properties")
    print("-" * 50)
    
    # Test on graphs with different densities
    densities = [0.1, 0.3, 0.5, 0.7, 0.9]
    n = 20
    k = 3
    
    density_results = {}
    for p in densities:
        A, _ = generate_random_graph(n=n, p=p, weights=None)
        results = benchmark.run_single_test(A, k, algorithms=['greedy', 'spectral'])
        density_results[p] = results
    
    print("Cut values by graph density:")
    print("Density | Greedy | Spectral")
    print("-" * 30)
    for p, results in density_results.items():
        greedy_cut = results['greedy']['cut_value']
        spectral_cut = results['spectral']['cut_value']
        print(f"{p:7.1f} | {greedy_cut:6.1f} | {spectral_cut:8.1f}")
    
    # Example 5: Performance comparison
    print("\n\nExample 5: Performance comparison")
    print("-" * 50)
    
    # Test scalability
    sizes = [10, 20, 30, 40, 50]
    k = 3
    
    print("Runtime comparison (seconds):")
    print("Size | Greedy | Spectral")
    print("-" * 25)
    
    for n in sizes:
        A, _ = generate_random_graph(n=n, p=0.3, weights=None)
        results = benchmark.run_single_test(A, k, algorithms=['greedy', 'spectral'])
        
        greedy_time = results['greedy']['runtime']
        spectral_time = results['spectral']['runtime']
        print(f"{n:4d} | {greedy_time:6.4f} | {spectral_time:8.4f}")


if __name__ == "__main__":
    main()
