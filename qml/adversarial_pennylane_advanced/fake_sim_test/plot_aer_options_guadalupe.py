import yaml
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_benchmark_results(input_file, output_image):
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Run the benchmark script first.")
        return

    with open(input_file, 'r') as f:
        data = yaml.safe_load(f)

    benchmarks = data.get('benchmarks', {})
    qubits_keys = sorted(benchmarks.keys(), key=lambda x: int(x[1:]))
    methods = data['metadata']['methods_tested']

    # Initialize structure to hold times
    # results[method] = [time_for_q1, time_for_q6, time_for_q8]
    results = {method: [] for method in methods}
    valid_qubits = []

    for q_key in qubits_keys:
        q_data = benchmarks[q_key]
        if not q_data:
            continue
        valid_qubits.append(q_key)
        for method in methods:
            time = q_data.get(method, {}).get('avg_time', 0)
            results[method].append(time)

    if not valid_qubits:
        print("No benchmark data found to plot.")
        return

    # Plotting
    x = np.arange(len(valid_qubits))
    width = 0.25
    multiplier = 0
    
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')

    for method, times in results.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, times, width, label=method)
        ax.bar_label(rects, padding=3, fmt='%.2fs')
        multiplier += 1

    ax.set_ylabel('Average Execution Time (s)')
    ax.set_title(f"Aer Simulation Options Benchmark - {data['metadata']['backend']}")
    ax.set_xticks(x + width, valid_qubits)
    ax.legend(loc='upper left', ncols=3)
    
    # Ensure Y-axis is reasonable
    all_times = [t for times_list in results.values() for t in times_list]
    if all_times:
        ax.set_ylim(0, max(all_times) * 1.3)

    plt.savefig(output_image)
    print(f"Plot saved to {output_image}")

if __name__ == "__main__":
    input_file = "results_aer_simulation_options_guadalupe.yaml"
    output_image = "benchmark_aer_guadalupe_plot.png"
    plot_benchmark_results(input_file, output_image)
