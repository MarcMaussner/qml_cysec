import time
import pennylane as qml
import numpy as np
import yaml
import sys
import argparse
import importlib
from qiskit_ibm_runtime import fake_provider

def get_backend_by_name(backend_name):
    """Dynamically load a backend class from qiskit_ibm_runtime.fake_provider."""
    if hasattr(fake_provider, backend_name):
        return getattr(fake_provider, backend_name)()
    
    try:
        base_module = "qiskit_ibm_runtime.fake_provider.backends"
        module_name = backend_name.replace('Fake', '').replace('V2', '').lower()
        full_module_path = f"{base_module}.{module_name}.fake_{module_name}"
        
        module = importlib.import_module(full_module_path)
        if hasattr(module, backend_name):
            return getattr(module, backend_name)()
        alt_name = backend_name.replace('V2', '')
        if hasattr(module, alt_name):
            return getattr(module, alt_name)()
    except ImportError:
        pass

    raise ValueError(f"Backend '{backend_name}' not found in qiskit_ibm_runtime.fake_provider")

def run_benchmark(backend_name, n_wires, method="automatic", n_runs=5, n_layers=3, n_shots=1000):
    print(f"\n>>> [START] Benchmarking {backend_name} | Method: {method} | Wires: {n_wires}")
    
    print(f"DEBUG: [1/7] Fetching backend {backend_name}...")
    backend_instance = get_backend_by_name(backend_name)
    
    from qiskit_aer import AerSimulator
    from qiskit import QuantumCircuit, transpile
    
    print(f"DEBUG: [2/7] Creating AerSimulator from {backend_name}...")
    sim = AerSimulator.from_backend(backend_instance)
    
    print(f"DEBUG: [3/7] Configuring simulator (method={method})...")
    sim.set_options(method=method)
    
    if n_wires > backend_instance.num_qubits:
        print(f"Warning: Requested {n_wires} wires, but {backend_name} has {backend_instance.num_qubits}. Using {backend_instance.num_qubits}.")
        n_wires = backend_instance.num_qubits

    print(f"DEBUG: [4/7] Constructing Qiskit QuantumCircuit...")
    qc = QuantumCircuit(n_wires)
    # Simple circuit: Rotations + CNOT chain
    for i in range(n_wires):
        qc.rx(np.pi/4, i)
        qc.ry(np.pi/4, i)
    for i in range(n_wires - 1):
        qc.cx(i, i+1)
    qc.measure_all()

    print(f"DEBUG: [5/7] Transpiling circuit for the backend...")
    compiled_qc = transpile(qc, sim)
    
    print(f"DEBUG: [6/7] Running warmup execution...")
    try:
        start_warmup = time.time()
        sim.run(compiled_qc, shots=n_shots).result()
        print(f"DEBUG: Warmup success. Time: {time.time() - start_warmup:.4f}s")
    except Exception as e:
        print(f"CRITICAL ERROR during warmup for {method}: {e}")
        return None
    
    print(f"DEBUG: [7/7] Executing benchmark loop ({n_runs} repetitions)...")
    times = []
    for i in range(n_runs):
        start_run = time.time()
        sim.run(compiled_qc, shots=n_shots).result()
        duration = time.time() - start_run
        times.append(duration)
        print(f"  - Repetition {i+1}/{n_runs}: {duration:.4f}s")
    
    avg_time = np.mean(times)
    total_time = np.sum(times)
    print(f">>> [DONE] Results for {method}: Avg {avg_time:.4f}s | Total {total_time:.4f}s")
    
    return {
        "avg_time": float(avg_time),
        "total_time": float(total_time),
        "qubits": int(backend_instance.num_qubits),
        "sim_wires": int(n_wires)
    }

if __name__ == "__main__":
    print("=== Aer Simulation Benchmark Start ===")
    backend_name = "FakeGuadalupeV2"
    qubits_list = [1, 6, 8]
    methods = ["automatic", "matrix_product_state", "statevector"]
    runs = 3
    output_file = "results_aer_simulation_options_guadalupe.yaml"
    
    all_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "backend": backend_name,
            "qubits_list": qubits_list,
            "n_runs": runs,
            "methods_tested": methods,
            "sim_options": "coupling_map=None"
        },
        "benchmarks": {}
    }

    total_start = time.time()
    for n_qubits in qubits_list:
        qubit_key = f"q{n_qubits}"
        all_results["benchmarks"][qubit_key] = {}
        print(f"\n" + "="*50)
        print(f"BEGIN TESTING: {n_qubits} QUBITS")
        print("="*50)
        
        for method in methods:
            try:
                res = run_benchmark(backend_name, n_wires=n_qubits, method=method, n_runs=runs, n_layers=3)
                if res:
                    all_results["benchmarks"][qubit_key][method] = {
                        "qubits_available": res["qubits"],
                        "qubits_benchmarked": res["sim_wires"],
                        "avg_time": res["avg_time"],
                        "total_time": res["total_time"]
                    }
            except Exception as e:
                print(f"ERROR: Global exception during {n_qubits}q / {method}: {e}")

    print(f"\n" + "="*50)
    print(f"FINISHING: Saving results...")
    with open(output_file, "w") as f:
        yaml.dump(all_results, f, default_flow_style=False)
    
    print(f"Results successfully saved to: {output_file}")
    print(f"Total benchmark duration: {time.time() - total_start:.2f}s")
    print("=== Aer Simulation Benchmark End ===")
