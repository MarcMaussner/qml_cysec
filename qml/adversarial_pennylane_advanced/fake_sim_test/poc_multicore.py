import time
import pennylane as qml
import numpy as np
import yaml
import sys
import argparse
import importlib
import inspect
from qiskit_ibm_runtime import fake_provider

def get_backend_by_name(backend_name):
    """Dynamically load a backend class from qiskit_ibm_runtime.fake_provider."""
    # First check top-level fake_provider
    if hasattr(fake_provider, backend_name):
        return getattr(fake_provider, backend_name)()
    
    # Otherwise deep dive into backends directory
    # (Simplified for the PoC, usually we'd use the logic from our lister)
    try:
        # Example: FakeTorino -> backends.torino.fake_torino.FakeTorino
        # We'll use a search strategy
        base_module = "qiskit_ibm_runtime.fake_provider.backends"
        module_name = backend_name.replace('Fake', '').replace('V2', '').lower()
        full_module_path = f"{base_module}.{module_name}.fake_{module_name}"
        
        module = importlib.import_module(full_module_path)
        if hasattr(module, backend_name):
            return getattr(module, backend_name)()
        # Some are just 'FakeName' without V2 suffix in the class name itself
        alt_name = backend_name.replace('V2', '')
        if hasattr(module, alt_name):
            return getattr(module, alt_name)()
    except ImportError:
        pass

    raise ValueError(f"Backend '{backend_name}' not found in qiskit_ibm_runtime.fake_provider")

def run_benchmark(backend_name, num_threads, n_wires, method="automatic", n_runs=3, n_layers=5, n_shots=1000):
    print(f"\n>>> Benchmarking {backend_name} | Method: {method} | Threads: {num_threads} | Wires: {n_wires}")
    
    backend_instance = get_backend_by_name(backend_name)
    backend_instance.set_options(
        max_parallel_threads=num_threads, 
        max_parallel_experiments=1,
        method=method
    )
    
    if n_wires > backend_instance.num_qubits:
        print(f"Warning: Requested {n_wires} wires, but {backend_name} only has {backend_instance.num_qubits}. Using {backend_instance.num_qubits}.")
        n_wires = backend_instance.num_qubits

    dev = qml.device('qiskit.remote', wires=n_wires, backend=backend_instance)
    
    @qml.qnode(dev, shots=n_shots)
    def circuit(params):
        qml.StronglyEntanglingLayers(params, wires=range(n_wires))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_wires)]

    batch_size = 20
    shape = (batch_size,) + qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_wires)
    params_batch = np.random.random(shape)
    
    # Warmup
    try:
        circuit(params_batch)
    except Exception as e:
        print(f"Error during warmup with method {method}: {e}")
        return None
    
    start_time = time.time()
    for _ in range(n_runs):
        circuit(params_batch)
    
    total_time = time.time() - start_time
    avg_time = total_time / n_runs
    print(f"Average time: {avg_time:.4f}s | Total: {total_time:.4f}s")
    return {
        "avg_time": float(avg_time),
        "total_time": float(total_time),
        "qubits": int(backend_instance.num_qubits),
        "sim_wires": int(n_wires)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake Backend Multicore & Method PoC")
    parser.add_argument("backends", nargs="+", help="List of backend names")
    parser.add_argument("--qubits", type=int, default=6, help="Number of qubits to benchmark")
    parser.add_argument("--threads", type=int, default=6, help="Number of threads for multicore test")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per configuration")
    parser.add_argument("--methods", nargs="+", default=["automatic"], help="Simulation methods (automatic, statevector, matrix_product_state, etc.)")
    parser.add_argument("--output", default="benchmark_results.yaml", help="Output YAML file")
    
    args = parser.parse_args()
    
    all_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "threads_tested": args.threads,
            "target_qubits": args.qubits,
            "n_runs": args.runs,
            "methods_tested": args.methods
        },
        "benchmarks": {}
    }

    for bname in args.backends:
        all_results["benchmarks"][bname] = {}
        for method in args.methods:
            try:
                print(f"\n--- {bname} [ {method} ] ---")
                res_single = run_benchmark(bname, num_threads=1, n_wires=args.qubits, method=method, n_runs=args.runs)
                if res_single is None: continue
                
                res_multi = run_benchmark(bname, num_threads=args.threads, n_wires=args.qubits, method=method, n_runs=args.runs)
                if res_multi is None: continue
                
                all_results["benchmarks"][bname][method] = {
                    "qubits_available": res_single["qubits"],
                    "qubits_benchmarked": res_single["sim_wires"],
                    "single_thread": {
                        "avg_time": res_single["avg_time"],
                        "total_time": res_single["total_time"]
                    },
                    "multi_thread": {
                        "avg_time": res_multi["avg_time"],
                        "total_time": res_multi["total_time"]
                    },
                    "speedup": float(res_single["avg_time"] / res_multi["avg_time"]) if res_multi["avg_time"] > 0 else 0
                }
            except Exception as e:
                print(f"Error benchmarking {bname} with method {method}: {e}")

    with open(args.output, "w") as f:
        yaml.dump(all_results, f, default_flow_style=False)
    
    print(f"\nResults saved to {args.output}")
