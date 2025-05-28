"""
Analyze and compare results from multiple benchmark runs
"""

import pandas as pd
import json
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np


def load_run_results(run_dir):
    """Load results from a single run directory"""
    run_dir = Path(run_dir)
    
    # Load configuration
    with open(run_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    results = {
        'config': config,
        'unweighted': {},
        'weighted': {}
    }
    
    # Load unweighted results if they exist
    if (run_dir / 'unweighted_combined_stats.csv').exists():
        results['unweighted']['combined'] = pd.read_csv(run_dir / 'unweighted_combined_stats.csv')
        
        # Load k-specific comparisons
        for k in config['k_values']:
            k_file = run_dir / f'unweighted_k{k}_comparison.csv'
            if k_file.exists():
                results['unweighted'][f'k{k}'] = pd.read_csv(k_file, index_col=0)
    
    # Load weighted results if they exist
    if (run_dir / 'weighted_combined_stats.csv').exists():
        results['weighted']['combined'] = pd.read_csv(run_dir / 'weighted_combined_stats.csv')
        
        # Load k-specific comparisons
        for k in config['k_values']:
            k_file = run_dir / f'weighted_k{k}_comparison.csv'
            if k_file.exists():
                results['weighted'][f'k{k}'] = pd.read_csv(k_file, index_col=0)
    
    return results


def compare_runs(results_dir, runs=None):
    """Compare results from multiple runs"""
    results_dir = Path(results_dir)
    
    # Find all run directories
    if runs is None:
        run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])
    else:
        run_dirs = [results_dir / run for run in runs]
    
    all_results = {}
    for run_dir in run_dirs:
        if run_dir.exists():
            print(f"Loading results from {run_dir.name}...")
            all_results[run_dir.name] = load_run_results(run_dir)
    
    return all_results


def plot_algorithm_comparison(all_results, graph_type='unweighted', k=2):
    """Plot algorithm performance across different runs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Collect data for plotting
    algorithms = set()
    datasets = set()
    
    for run_name, results in all_results.items():
        if graph_type in results and 'combined' in results[graph_type]:
            df = results[graph_type]['combined']
            k_data = df[df['k'] == k]
            algorithms.update(k_data['algorithm'].unique())
            datasets.update(k_data['dataset'].unique())
    
    algorithms = sorted(list(algorithms))
    
    # Define the preferred order for datasets
    dataset_order = ["Isolated", "Transitioning", "Extremely Sparse", 
                     "Very Sparse", "Moderately Sparse", "Moderately Dense", "Very Dense"]
    
    # Sort datasets according to the preferred order, keeping any extras at the end
    ordered_datasets = []
    for ds in dataset_order:
        if ds in datasets:
            ordered_datasets.append(ds)
    
    # Add any datasets not in the preferred order at the end
    for ds in sorted(list(datasets)):
        if ds not in ordered_datasets:
            ordered_datasets.append(ds)
    
    datasets = ordered_datasets
    
    # Plot 1: Algorithm performance by dataset
    for alg in algorithms:
        means = []
        for dataset in datasets:
            values = []
            for run_name, results in all_results.items():
                if graph_type in results and 'combined' in results[graph_type]:
                    df = results[graph_type]['combined']
                    mask = (df['k'] == k) & (df['algorithm'] == alg) & (df['dataset'] == dataset)
                    if len(df[mask]) > 0:
                        values.append(df[mask]['mean_cut'].values[0])
            
            if values:
                means.append(np.mean(values))
            else:
                means.append(0)
        
        ax1.plot(datasets, means, marker='o', label=alg)
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Mean Cut Value')
    ax1.set_title(f'{graph_type.capitalize()} Graphs - k={k} - Algorithm Performance')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Runtime comparison
    for alg in algorithms:
        runtimes = []
        cut_values = []
        
        for run_name, results in all_results.items():
            if graph_type in results and 'combined' in results[graph_type]:
                df = results[graph_type]['combined']
                alg_data = df[(df['k'] == k) & (df['algorithm'] == alg)]
                if len(alg_data) > 0:
                    runtimes.extend(alg_data['mean_runtime'].values)
                    cut_values.extend(alg_data['mean_cut'].values)
        
        if runtimes and cut_values:
            ax2.scatter(runtimes, cut_values, label=alg, s=100, alpha=0.6)
    
    ax2.set_xlabel('Mean Runtime (s)')
    ax2.set_ylabel('Mean Cut Value')
    ax2.set_title(f'{graph_type.capitalize()} Graphs - k={k} - Runtime vs Quality')
    ax2.legend()
    ax2.set_xscale('log')
    
    plt.tight_layout()
    return fig


def generate_summary_report(all_results, output_file='summary_report.txt'):
    """Generate a text summary of all runs"""
    with open(output_file, 'w') as f:
        f.write("BENCHMARK SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for run_name, results in sorted(all_results.items()):
            config = results['config']
            f.write(f"Run: {run_name}\n")
            f.write(f"Timestamp: {config['timestamp']}\n")
            f.write(f"Vertex range: {config['v_min']}-{config['v_max']}\n")
            f.write(f"Graphs per dataset: {config['m']}\n")
            f.write(f"k values: {config['k_values']}\n")
            f.write(f"Algorithms: {config['algorithms']}\n")
            f.write("-" * 40 + "\n")
            
            # Best performing algorithm for each k
            for graph_type in ['unweighted', 'weighted']:
                if graph_type in results and 'combined' in results[graph_type]:
                    f.write(f"\n{graph_type.upper()} GRAPHS:\n")
                    df = results[graph_type]['combined']
                    
                    for k in config['k_values']:
                        k_data = df[df['k'] == k]
                        if len(k_data) > 0:
                            # Find best algorithm by mean cut value
                            best_idx = k_data.groupby('algorithm')['mean_cut'].mean().idxmax()
                            best_value = k_data.groupby('algorithm')['mean_cut'].mean().max()
                            f.write(f"  k={k}: Best algorithm = {best_idx} (avg cut = {best_value:.2f})\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    print(f"Summary report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Analyze benchmark results')
    parser.add_argument('--dir', type=str, default='results', 
                        help='Directory containing benchmark results')
    parser.add_argument('--runs', type=str, nargs='+', 
                        help='Specific runs to compare (default: all)')
    parser.add_argument('--plot', action='store_true', 
                        help='Generate comparison plots')
    parser.add_argument('--k', type=int, default=2, 
                        help='k value for plots')
    parser.add_argument('--summary', action='store_true', 
                        help='Generate summary report')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.dir}...")
    all_results = compare_runs(args.dir, args.runs)
    
    if not all_results:
        print("No results found!")
        return
    
    print(f"Found {len(all_results)} benchmark runs")
    
    # Generate summary report
    if args.summary:
        generate_summary_report(all_results)
    
    # Generate plots
    if args.plot:
        for graph_type in ['unweighted', 'weighted']:
            fig = plot_algorithm_comparison(all_results, graph_type, args.k)
            plt.savefig(f'comparison_{graph_type}_k{args.k}.png', dpi=150, bbox_inches='tight')
            print(f"Saved plot: comparison_{graph_type}_k{args.k}.png")
        
        plt.show()
    
    # Print quick comparison
    print("\nQuick Comparison (mean cut values):")
    print("-" * 60)
    
    for run_name in sorted(all_results.keys()):
        results = all_results[run_name]
        config = results['config']
        print(f"\n{run_name} (v={config['v_min']}-{config['v_max']}):")
        
        for graph_type in ['unweighted', 'weighted']:
            if graph_type in results and 'combined' in results[graph_type]:
                df = results[graph_type]['combined']
                pivot = df.pivot_table(values='mean_cut', index='algorithm', columns='k', aggfunc='mean')
                print(f"\n  {graph_type.upper()}:")
                print(pivot.round(2).to_string(index=True))


if __name__ == "__main__":
    main()