import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def calculate_resources(num_qubits):
    results = []
    
    # Common parameters based on existing scripts
    asp_layers = 2
    # Pattern: 4 qubits -> 16 layers, 8 qubits -> 32 layers => classifier_layers = 4 * num_qubits
    classifier_layers = 4 * num_qubits
    
    # 1. Exact State Preparation (SBM)
    # Theoretically O(2^n)
    exact_cnot = 2**num_qubits - 2 if num_qubits < 60 else float('inf') # placeholder for overflow
    exact_depth = 2**num_qubits # rough approximation
    
    # 2. ASP (Approximate State Preparation)
    # ASP Ansatz: num_layers * num_qubits CNOTs
    # Classifier: num_layers * num_qubits CNOTs
    asp_cnot = num_qubits * (asp_layers + classifier_layers)
    # Depth: 2 per ASP layer + 4 per classifier layer (3 rotations + 1 CX)
    asp_depth = (2 * asp_layers) + (4 * classifier_layers)
    
    # 3. Stochastic ASP / QDA-ASP
    # Same CNOT count as ASP
    # Depth: ASP Depth + 1 (for the noise layer of rotations)
    counter_cnot = asp_cnot
    counter_depth = asp_depth + 1
    
    return {
        'Qubits': num_qubits,
        'Exact_CNOT': exact_cnot,
        'Exact_Depth': exact_depth,
        'ASP_CNOT': asp_cnot,
        'ASP_Depth': asp_depth,
        'Counter_CNOT': counter_cnot,
        'Counter_Depth': counter_depth
    }

def main():
    qubit_counts = [4, 8, 12, 16, 20, 24, 28, 32, 100]
    data = [calculate_resources(n) for n in qubit_counts]
    df = pd.DataFrame(data)
    
    output_dir = "pictures_benchmarks"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set plot style (Standard light background for alignment)
    colors = {
        'Exact': '#d62728',   # Standard Red
        'ASP': '#2ca02c',     # Standard Green
        'Counter': '#1f77b4'  # Standard Blue
    }
    
    # Plot 1: Size (CNOT count)
    fig, ax = plt.subplots(figsize=(6, 4)) # Adjusted size for better integration
    
    # For plotting, cap Exact at a readable value for log scale
    plot_df = df[df['Qubits'] <= 32]
    
    ax.plot(plot_df['Qubits'], plot_df['Exact_CNOT'], marker='o', label='Exact (SBM)', color=colors['Exact'], linewidth=1.5)
    ax.plot(df['Qubits'], df['ASP_CNOT'], marker='s', label='ASP', color=colors['ASP'], linewidth=1.5)
    ax.plot(df['Qubits'], df['Counter_CNOT'], marker='^', linestyle='--', label='Countermeasures', color=colors['Counter'], linewidth=1.5)
    
    ax.set_yscale('log')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('CNOT Gate Count (Log)')
    ax.set_title('Circuit Size Scaling', fontweight='bold')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(fontsize='small')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/resource_scaling_size.png", dpi=150) # Adjusted DPI
    print(f"Saved {output_dir}/resource_scaling_size.png")
    
    # Plot 2: Depth
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(plot_df['Qubits'], plot_df['Exact_Depth'], marker='o', label='Exact (SBM)', color=colors['Exact'], linewidth=1.5)
    ax.plot(df['Qubits'], df['ASP_Depth'], marker='s', label='ASP', color=colors['ASP'], linewidth=1.5)
    ax.plot(df['Qubits'], df['Counter_Depth'], marker='^', linestyle='--', label='Countermeasures', color=colors['Counter'], linewidth=1.5)
    
    ax.set_yscale('log')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Circuit Depth (Log)')
    ax.set_title('Circuit Depth Scaling', fontweight='bold')
    ax.grid(True, which="both", ls="-", alpha=0.3)
    ax.legend(fontsize='small')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/resource_scaling_depth.png", dpi=150)
    print(f"Saved {output_dir}/resource_scaling_depth.png")
    
    # Save CSV
    df.to_csv(f"{output_dir}/resource_benchmarks.csv", index=False)
    print(f"Saved {output_dir}/resource_benchmarks.csv")

if __name__ == "__main__":
    main()
