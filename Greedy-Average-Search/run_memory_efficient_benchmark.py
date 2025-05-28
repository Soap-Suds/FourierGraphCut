"""
Memory-efficient script to run Max-k-Cut benchmarks
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import gc
import psutil
import os
import random
import time
import sys

# Increase recursion limit for large graphs
sys.setrecursionlimit(10000)

from graph_generation import generate_benchmark_datasets
from memory_efficient_benchmarking import MemoryEfficientBenchmark, memory_efficient_spectral_cut


def print_memory_usage():
    """Print current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024 / 1024:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description='Run memory-efficient Max-k-Cut benchmarks')
    parser.add_argument('--m', type=int, default=50, help='Number of graphs per dataset')
    parser.add_argument('--v_range', type=int, nargs=2, default=[15, 20], 
                        help='Vertex range [min max]')
    parser.add_argument('--k_values', type=int, nargs='+', default=[2, 3, 4], 
                        help='Values of k to test')
    parser.add_argument('--algorithms', type=str, nargs='+', 
                        default=['greedy', 'spectral'],
                        help='Algorithms to benchmark')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results')
    parser.add_argument('--datasets', type=str, nargs='+',
                        help='Specific datasets to test (default: all)')
    parser.add_argument('--memory_efficient', action='store_true',
                        help='Use memory-efficient spectral algorithm')
    
    args = parser.parse_args()
    
    v_min, v_max = args.v_range
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = {
        'm': args.m,
        'v_min': v_min,
        'v_max': v_max,
        'k_values': args.k_values,
        'algorithms': args.algorithms,
        'seed': args.seed,
        'timestamp': timestamp,
        'memory_efficient': args.memory_efficient
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Memory-efficient benchmark")
    print(f"Vertex range: {v_min} to {v_max}")
    print(f"Graphs per dataset: {args.m}")
    print(f"Testing k values: {args.k_values}")
    print(f"Algorithms: {args.algorithms}")
    print("-" * 50)
    
    # Initialize benchmark
    benchmark = MemoryEfficientBenchmark(save_partitions=False)
    
    # Use memory-efficient spectral if requested
    if args.memory_efficient and 'spectral' in args.algorithms:
        def spectral_wrapper(A, k):
            start_time = time.time()
            partition, cut_value = memory_efficient_spectral_cut(A, k)
            runtime = time.time() - start_time
            return partition, cut_value, runtime
        
        benchmark.algorithms['spectral'] = spectral_wrapper
        print("Using memory-efficient spectral algorithm")
    
    print("Initial memory usage:")
    print_memory_usage()
    
    # Process datasets one at a time
    dataset_configs = {
        'Isolated': lambda n: 1/n,
        'Transitioning': lambda n: np.log2(n)/n,
        'Extremely Sparse': lambda n: 1/np.sqrt(n),
        'Very Sparse': lambda n: 0.1,
        'Moderately Sparse': lambda n: 0.25,
        'Moderately Dense': lambda n: 0.5,
        'Very Dense': lambda n: 0.75,
    }
    
    # Filter datasets if specified
    if args.datasets:
        dataset_configs = {k: v for k, v in dataset_configs.items() 
                          if k in args.datasets}
    
    all_stats = []
    
    for dataset_name, prob_func in dataset_configs.items():
        print(f"\nProcessing dataset: {dataset_name}")
        print_memory_usage()
        
        # Generate graphs one at a time and process immediately
        from graph_generation import generate_random_graph
        
        # Create temporary dataset structure
        dataset = {
            'name': dataset_name,
            'weighted': False,
            'graphs': []
        }
        
        # Generate and process graphs in batches
        batch_size = 10
        results_file = output_dir / f'{dataset_name}_results.jsonl'
        
        with open(results_file, 'w') as f:
            for i in range(args.m):
                if i % batch_size == 0:
                    print(f"\rProcessing graphs {i+1}-{min(i+batch_size, args.m)}/{args.m}", 
                          end='', flush=True)
                
                # Generate single graph
                n = random.randint(v_min, v_max)
                p = prob_func(n)
                A, G = generate_random_graph(n, p, weights=None, seed=args.seed + i)
                
                # Process immediately
                for k in args.k_values:
                    if k > n:
                        continue
                    
                    test_results = benchmark.run_single_test(A, k, args.algorithms)
                    
                    # Write results immediately
                    result_entry = {
                        'graph_index': i,
                        'n': n,
                        'p': p,
                        'k': k,
                        'results': test_results
                    }
                    
                    f.write(json.dumps(result_entry) + '\n')
                    f.flush()
                
                # Clear graph from memory
                del A, G
                
                # Periodic garbage collection
                if i % batch_size == 0:
                    gc.collect()
        
        print()  # New line
        
        # Compute statistics from file
        stats = benchmark.compute_statistics_from_file(results_file)
        stats['dataset'] = dataset_name
        all_stats.append(stats)
        
        print(f"Completed {dataset_name}")
        print_memory_usage()
        
        # Force garbage collection between datasets
        gc.collect()
    
    # Combine all statistics
    combined_stats = pd.concat(all_stats, ignore_index=True)
    combined_stats.to_csv(output_dir / 'combined_stats.csv', index=False)
    
    # Create comparison tables
    for k in args.k_values:
        k_data = combined_stats[combined_stats['k'] == k]
        if len(k_data) > 0:
            pivot = k_data.pivot_table(
                index='dataset',
                columns='algorithm',
                values='mean_cut'
            )
            pivot.to_csv(output_dir / f'k{k}_comparison.csv')
            print(f"\nk={k} comparison:")
            print(pivot.round(2).to_string())
    
    print(f"\n\nResults saved to {output_dir}")
    print("Final memory usage:")
    print_memory_usage()


if __name__ == "__main__":
    main()