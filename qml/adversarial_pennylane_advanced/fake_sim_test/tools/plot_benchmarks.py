import yaml
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

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
                # Handle both new (dict of methods) and old (dict of results) formats
                if isinstance(methods, dict) and any(m in methods for m in ['automatic', 'statevector', 'matrix_product_state']):
                    for method_name, results in methods.items():
                        aggregated_data.append({
                            'Backend': backend,
                            'Method': method_name,
                            'Qubits': target_qubits,
                            'Threads': threads,
                            'S_Avg': results['single_thread']['avg_time'],
                            'S_Total': results['single_thread']['total_time'],
                            'M_Avg': results['multi_thread']['avg_time'],
                            'M_Total': results['multi_thread']['total_time'],
                            'Speedup': results.get('speedup', 0)
                        })
                else:
                    results = methods
                    aggregated_data.append({
                        'Backend': backend,
                        'Method': 'automatic',
                        'Qubits': target_qubits,
                        'Threads': threads,
                        'S_Avg': results['single_thread']['avg_time'],
                        'S_Total': results['single_thread']['total_time'],
                        'M_Avg': results['multi_thread']['avg_time'],
                        'M_Total': results['multi_thread']['total_time'],
                        'Speedup': results.get('speedup', 0)
                    })
        except Exception as e:
            print(f"Warning: Failed to parse {fpath}: {e}")
            
    return pd.DataFrame(aggregated_data)

def generate_plots(df, output_dir):
    if df.empty:
        print("No data to plot.")
        return

    # 1. Per-Backend Detailed Timing Plots (Single vs Multi, Avg vs Total)
    backends = df['Backend'].unique()
    for backend in backends:
        b_df = df[df['Backend'] == backend]
        methods = b_df['Method'].unique()
        
        for method in methods:
            bm_df = b_df[b_df['Method'] == method].sort_values(['Qubits', 'Threads'])
            if bm_df.empty: continue
            
            print(f"Generating timing comparison for {backend} ({method})...")
            
            # Use max threads available for the comparison bar plot if multiple exist for same qubit count
            # Or just plot for each configuration
            for qubits in bm_df['Qubits'].unique():
                q_df = bm_df[bm_df['Qubits'] == qubits]
                for threads in q_df['Threads'].unique():
                    row = q_df[q_df['Threads'] == threads].iloc[0]
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    # Avg Time Plot
                    ax1.bar(['Single-Thread', 'Multi-Thread'], [row['S_Avg'], row['M_Avg']], color=['#4C72B0', '#55A868'])
                    ax1.set_title(f'Avg Time ({qubits} Qubits, {threads} Threads)')
                    ax1.set_ylabel('Time (s)')
                    ax1.grid(axis='y', linestyle='--', alpha=0.7)

                    # Total Time Plot
                    ax2.bar(['Single-Thread', 'Multi-Thread'], [row['S_Total'], row['M_Total']], color=['#4C72B0', '#55A868'])
                    ax2.set_title(f'Total Time ({qubits} Qubits, {threads} Threads)')
                    ax2.set_ylabel('Time (s)')
                    ax2.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    fig.suptitle(f"Backend: {backend} | Method: {method}", fontsize=14)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    
                    out_path = os.path.join(output_dir, f"plot_{backend}_{method}_q{qubits}_t{threads}_timing.png")
                    plt.savefig(out_path)
                    plt.close()

    # 2. Cross-Backend Comparison (Avg and Total)
    # Plotting for max qubits and common threads
    max_q = df['Qubits'].max()
    common_threads = df['Threads'].unique()
    
    for method in df['Method'].unique():
        for threads in common_threads:
            cb_df = df[(df['Method'] == method) & (df['Qubits'] == max_q) & (df['Threads'] == threads)]
            if cb_df.empty: continue
            
            print(f"Generating cross-backend comparison (Method: {method}, Qubits: {max_q}, Threads: {threads})...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Avg Time Comparison
            ax1.bar(cb_df['Backend'], cb_df['M_Avg'], color=plt.cm.Paired(range(len(cb_df))))
            ax1.set_title(f'Avg Multi-Thread Time ({threads} Threads)')
            ax1.set_ylabel('Time (s)')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(axis='y', linestyle='--', alpha=0.7)

            # Total Time Comparison
            ax2.bar(cb_df['Backend'], cb_df['M_Total'], color=plt.cm.Paired(range(len(cb_df))))
            ax2.set_title(f'Total Multi-Thread Time ({threads} Threads)')
            ax2.set_ylabel('Time (s)')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(axis='y', linestyle='--', alpha=0.7)
            
            fig.suptitle(f"Cross-Backend Comparison | Method: {method} | Qubits: {max_q}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            out_cb = os.path.join(output_dir, f"plot_cross_backend_{method}_q{max_q}_t{threads}.png")
            plt.savefig(out_cb)
            plt.close()

    # 3. Traditional Scaling Plots (Speedup vs Threads)
    for backend in backends:
        b_df = df[df['Backend'] == backend]
        plt.figure(figsize=(10, 6))
        for method in b_df['Method'].unique():
            # For speedup, we might have multiple qubit counts. We'll pick max qubits for clarity or plot all.
            max_q_b = b_df[b_df['Method'] == method]['Qubits'].max()
            m_df = b_df[(b_df['Method'] == method) & (b_df['Qubits'] == max_q_b)].sort_values('Threads')
            if not m_df.empty:
                plt.plot(m_df['Threads'], m_df['Speedup'], marker='o', label=f"{method} (Qubits: {max_q_b})")
        
        plt.title(f"Multicore Speedup Scaling - {backend}")
        plt.xlabel("Number of Threads")
        plt.ylabel("Speedup (vs Single Thread)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"plot_{backend}_speedup_summary.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregated Plotting for Quantum Benchmarks")
    parser.add_argument("--dir", default=".", help="Directory containing benchmark_*.yaml files")
    parser.add_argument("--output", default=".", help="Directory to save plots")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.dir):
        print(f"Error: Directory {args.dir} not found.")
    else:
        results_df = aggregate_results(args.dir)
        generate_plots(results_df, args.output)
        print("Done.")
