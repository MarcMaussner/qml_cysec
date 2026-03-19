import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns

def aggregate_results(directory):
    pattern = os.path.join(directory, "benchmark_*.yaml")
    files = glob.glob(pattern)
    print(f"Aggregating results from {len(files)} files...")
    
    aggregated_data = []
    
    for fpath in files:
        try:
            with open(fpath, 'r') as f:
                data = yaml.safe_load(f)
            
            metadata = data.get('metadata', {})
            benchmarks = data.get('benchmarks', {})
            
            target_qubits = metadata.get('target_qubits')
            threads = metadata.get('threads_tested')
            
            for backend, methods in benchmarks.items():
                if isinstance(methods, dict) and any(m in methods for m in ['automatic', 'statevector', 'matrix_product_state']):
                    for method_name, results in methods.items():
                        aggregated_data.append({
                            'Backend': backend,
                            'Method': method_name,
                            'Qubits': target_qubits,
                            'Threads': threads,
                            'M_Avg': results['multi_thread']['avg_time'],
                            'Speedup': results.get('speedup', 0)
                        })
                else:
                    results = methods
                    aggregated_data.append({
                        'Backend': backend,
                        'Method': 'automatic',
                        'Qubits': target_qubits,
                        'Threads': threads,
                        'M_Avg': results['multi_thread']['avg_time'],
                        'Speedup': results.get('speedup', 0)
                    })
        except Exception as e:
            print(f"Warning: Failed to parse {fpath}: {e}")
            
    return pd.DataFrame(aggregated_data)

def generate_comprehensive_plots(df, output_file):
    if df.empty:
        print("No data to plot.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # 1. Top-Left: Backend Comparison
    # Instead of a fixed common_q, we'll pick a thread count (usually max common, say 4 or 1)
    # and for each backend, we take its own max qubit count at that thread count.
    # This allows comparing "best effort" performance.
    
    # Let's find a thread count that exists for most backends
    potential_threads = [4, 1]
    common_t = df['Threads'].max() # default
    for t in potential_threads:
        if t in df['Threads'].values:
            common_t = t
            break
            
    tl_data = []
    for backend in df['Backend'].unique():
        b_df = df[(df['Backend'] == backend) & (df['Threads'] == common_t) & (df['Method'] == 'automatic')]
        if not b_df.empty:
            # Pick the row with highest qubit count for this backend at this thread level
            best_row = b_df.loc[b_df['Qubits'].idxmax()]
            tl_data.append(best_row)
    
    tl_df = pd.DataFrame(tl_data)
    
    if not tl_df.empty:
        sns.barplot(x='Backend', y='M_Avg', data=tl_df, ax=axes[0, 0], hue='Backend', palette='viridis', legend=False)
        axes[0, 0].set_yscale('log')
        axes[0, 0].set_title(f'Execution Time Comparison (Log Scale)\nVariables Qubits (Max), {common_t} Threads', fontsize=14)
        axes[0, 0].set_ylabel('Average Time (s)')
        
        # Add labels to show which qubit count was used
        for i, row in enumerate(tl_df.sort_values('Backend').itertuples()):
            axes[0, 0].text(i, row.M_Avg * 1.1, f'Q:{row.Qubits}', ha='center', fontsize=10, fontweight='bold')
    else:
        axes[0, 0].text(0.5, 0.5, 'Data not found (Log Scale)', ha='center')

    # 2. Top-Right: Speedup Efficiency
    # Fixed at max qubits per backend
    for backend in df['Backend'].unique():
        b_df = df[(df['Backend'] == backend) & (df['Method'] == 'automatic')].sort_values('Threads')
        # Selecting highest qubit count available for each backend to show "hardest" scaling
        max_q = b_df['Qubits'].max()
        bq_df = b_df[b_df['Qubits'] == max_q]
        if not bq_df.empty:
            axes[0, 1].plot(bq_df['Threads'], bq_df['Speedup'], marker='o', label=f"{backend} (Q:{max_q})")
    
    axes[0, 1].plot([1, df['Threads'].max()], [1, df['Threads'].max()], 'k--', alpha=0.5, label='Ideal Scaling')
    axes[0, 1].set_title('Multicore Speedup Efficiency', fontsize=14)
    axes[0, 1].set_xlabel('Number of Threads')
    axes[0, 1].set_ylabel('Speedup (vs Single Thread)')
    axes[0, 1].legend()

    # 3. Bottom-Left: Qubit Complexity Scaling
    # Plot scaling for each backend at its *own* max thread count
    for backend in df['Backend'].unique():
        max_t_backend = df[df['Backend'] == backend]['Threads'].max()
        b_df = df[(df['Backend'] == backend) & (df['Threads'] == max_t_backend) & (df['Method'] == 'automatic')].sort_values('Qubits')
        if not b_df.empty:
            axes[1, 0].plot(b_df['Qubits'], b_df['M_Avg'], marker='s', label=f"{backend} ({max_t_backend}T)")
            
    axes[1, 0].set_title('Qubit Scaling Impact (Max Threads)', fontsize=14)
    axes[1, 0].set_xlabel('Number of Qubits')
    axes[1, 0].set_ylabel('Time (s)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()

    # 4. Bottom-Right: Method Comparison
    # We'll take Torino as the "expert" case if it exists, else use the one with most methods
    methods_count = df.groupby('Backend')['Method'].nunique()
    ref_backend = methods_count.idxmax()
    if 'FakeTorino' in df['Backend'].values:
        ref_backend = 'FakeTorino'
        
    ref_q = df[df['Backend'] == ref_backend]['Qubits'].max()
    max_t_ref = df[df['Backend'] == ref_backend]['Threads'].max()
    br_df = df[(df['Backend'] == ref_backend) & (df['Qubits'] == ref_q) & (df['Threads'] == max_t_ref)]
    
    if not br_df.empty:
        sns.barplot(x='Method', y='M_Avg', data=br_df, ax=axes[1, 1], hue='Method', palette='magma', legend=False)
        axes[1, 1].set_title(f'Simulation Method Comparison\n{ref_backend} ({ref_q} Qubits, {max_t_ref} Threads)', fontsize=14)
        axes[1, 1].set_ylabel('Time (s)')
    else:
        axes[1, 1].text(0.5, 0.5, 'Method data not found', ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Quantum ML Simulation: Comprehensive Backend Benchmark Analysis', fontsize=20)
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Comprehensive plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Benchmark Visualization")
    parser.add_argument("--dir", default=".", help="Directory containing benchmark_*.yaml files")
    parser.add_argument("--output", default="comprehensive_comparison.png", help="Output file name")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.dir):
        print(f"Error: Directory {args.dir} not found.")
    else:
        results_df = aggregate_results(args.dir)
        generate_comprehensive_plots(results_df, args.output)
        print("Done.")
