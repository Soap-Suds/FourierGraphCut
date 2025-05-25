"""
Main script to run Max-k-Cut benchmarks
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from graph_generation import generate_benchmark_datasets
from benchmarking import MaxKCutBenchmark


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Max-k-Cut benchmarks')
    parser.add_argument('--m', type=int, default=50, help='Number of graphs per dataset')
    parser.add_argument('--v_range', type=int, nargs=2, default=[15, 20], 
                        help='Vertex range [min max]')
    parser.add_argument('--v_min', type=int, help='Minimum vertices (deprecated, use --v_range)')
    parser.add_argument('--v_max', type=int, help='Maximum vertices (deprecated, use --v_range)')
    parser.add_argument('--k_values', type=int, nargs='+', default=[2, 3, 4], 
                        help='Values of k to test')
    parser.add_argument('--algorithms', type=str, nargs='+', 
                        default=['greedy', 'spectral', 'gw_sdp'],
                        help='Algorithms to benchmark')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results')
    parser.add_argument('--weighted', action='store_true', help='Test weighted graphs')
    parser.add_argument('--unweighted', action='store_true', help='Test unweighted graphs')
    
    args = parser.parse_args()
    
    # Handle vertex range - support both old and new syntax
    if args.v_min is not None or args.v_max is not None:
        print("Warning: --v_min and --v_max are deprecated. Use --v_range instead.")
        v_min = args.v_min if args.v_min is not None else args.v_range[0]
        v_max = args.v_max if args.v_max is not None else args.v_range[1]
    else:
        v_min, v_max = args.v_range
    
    # Default to both if neither specified
    if not args.weighted and not args.unweighted:
        args.weighted = True
        args.unweighted = True
    
    # Create output directory with timestamp to avoid overwriting
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save run configuration
    config = {
        'm': args.m,
        'v_min': v_min,
        'v_max': v_max,
        'k_values': args.k_values,
        'algorithms': args.algorithms,
        'seed': args.seed,
        'weighted': args.weighted,
        'unweighted': args.unweighted,
        'timestamp': timestamp
    }
    
    import json
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Generating benchmark datasets with {args.m} graphs each...")
    print(f"Vertex range: {v_min} to {v_max}")
    print(f"Testing k values: {args.k_values}")
    print(f"Algorithms: {args.algorithms}")
    print(f"Results will be saved to: {output_dir}")
    print("-" * 50)
    
    # Generate datasets
    unweighted_datasets, weighted_datasets = generate_benchmark_datasets(
        m=args.m, v_min=v_min, v_max=v_max, seed=args.seed
    )
    
    # Initialize benchmark
    benchmark = MaxKCutBenchmark()
    
    # Run benchmarks on unweighted graphs
    if args.unweighted:
        print("\nRunning benchmarks on unweighted graphs...")
        unweighted_results = {}
        
        for dataset_name, dataset in unweighted_datasets.items():
            print(f"\nDataset: {dataset_name}")
            results = benchmark.run_dataset_test(
                dataset, args.k_values, args.algorithms
            )
            unweighted_results[dataset_name] = results
            
            # Compute and display statistics
            stats = benchmark.compute_statistics(results)
            print(stats.to_string())
            
            # Save dataset-specific results
            stats.to_csv(output_dir / f'unweighted_{dataset_name}_stats.csv')
        
        # Save combined results
        benchmark.save_results(str(output_dir / 'unweighted_results.json'))
        
        # Create comparison across all datasets
        all_unweighted_stats = []
        for dataset_name, results in unweighted_results.items():
            stats = benchmark.compute_statistics(results)
            stats['dataset'] = dataset_name
            all_unweighted_stats.append(stats)
        
        combined_unweighted = pd.concat(all_unweighted_stats, ignore_index=True)
        combined_unweighted.to_csv(output_dir / 'unweighted_combined_stats.csv')
        
        # Create pivot tables for easier analysis
        for k in args.k_values:
            k_data = combined_unweighted[combined_unweighted['k'] == k]
            pivot = k_data.pivot_table(
                index='dataset', 
                columns='algorithm', 
                values='mean_cut'
            )
            pivot.to_csv(output_dir / f'unweighted_k{k}_comparison.csv')
            print(f"\nUnweighted k={k} comparison:")
            print(pivot.to_string())
    
    # Run benchmarks on weighted graphs
    if args.weighted:
        print("\n\nRunning benchmarks on weighted graphs...")
        weighted_results = {}
        
        for dataset_name, dataset in weighted_datasets.items():
            print(f"\nDataset: {dataset_name}")
            results = benchmark.run_dataset_test(
                dataset, args.k_values, args.algorithms
            )
            weighted_results[dataset_name] = results
            
            # Compute and display statistics
            stats = benchmark.compute_statistics(results)
            print(stats.to_string())
            
            # Save dataset-specific results
            stats.to_csv(output_dir / f'weighted_{dataset_name}_stats.csv')
        
        # Save combined results
        benchmark.save_results(str(output_dir / 'weighted_results.json'))
        
        # Create comparison across all datasets
        all_weighted_stats = []
        for dataset_name, results in weighted_results.items():
            stats = benchmark.compute_statistics(results)
            stats['dataset'] = dataset_name
            all_weighted_stats.append(stats)
        
        combined_weighted = pd.concat(all_weighted_stats, ignore_index=True)
        combined_weighted.to_csv(output_dir / 'weighted_combined_stats.csv')
        
        # Create pivot tables for easier analysis
        for k in args.k_values:
            k_data = combined_weighted[combined_weighted['k'] == k]
            pivot = k_data.pivot_table(
                index='dataset', 
                columns='algorithm', 
                values='mean_cut'
            )
            pivot.to_csv(output_dir / f'weighted_k{k}_comparison.csv')
            print(f"\nWeighted k={k} comparison:")
            print(pivot.to_string())
    
    print(f"\n\nResults saved to {output_dir}")
    print("Summary files:")
    print("- config.json: Run configuration")
    print("- *_combined_stats.csv: All statistics for each graph type")
    print("- *_k{k}_comparison.csv: Algorithm comparison for each k value")
    print("- *_results.json: Complete results including partitions")


if __name__ == "__main__":
    main()