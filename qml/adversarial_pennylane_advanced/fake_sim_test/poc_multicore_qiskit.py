import time
import numpy as np
import yaml
import sys
import argparse
import importlib
from qiskit_ibm_runtime import fake_provider
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector

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

def create_strongly_entangling_circuit(n_qubits, n_layers):
    """Equivalent of PennyLane's StronglyEntanglingLayers."""
    qc = QuantumCircuit(n_qubits)
    num_params = n_layers * n_qubits * 3
    params = ParameterVector('p', num_params)
    
    param_idx = 0
    for l in range(n_layers):
        # Rotations
        for i in range(n_qubits):
            qc.u(params[param_idx+1], params[param_idx+2], params[param_idx], i)
            param_idx += 3
        
        # Entanglers
        if n_qubits > 1:
            shift = (l % (n_qubits - 1)) + 1
            for i in range(n_qubits):
                # Circular entangler with shift
                src = i
                tgt = (i + shift) % n_qubits
                qc.cx(src, tgt)
    
    qc.measure_all()
    return qc, params

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

    # Create template circuit
    qc_template, params = create_strongly_entangling_circuit(n_wires, n_layers)
    
    # Transpile once for the backend
    transpiled_qc = transpile(qc_template, backend_instance)
    
    batch_size = 20
    # Generate random parameters (3 params per qubit per layer)
    params_values = np.random.random((batch_size, len(params))) * 2 * np.pi
    
    # Create batch of parameter-bound circuits
    circuits_batch = []
    for val in params_values:
        bound_qc = transpiled_qc.assign_parameters(val)
        circuits_batch.append(bound_qc)

    # Warmup
    try:
        job = backend_instance.run(circuits_batch, shots=n_shots)
        job.result()
    except Exception as e:
        print(f"Error during warmup with method {method}: {e}")
        return None
    
    start_time = time.time()
    for _ in range(n_runs):
        job = backend_instance.run(circuits_batch, shots=n_shots)
        job.result()
    
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
    parser = argparse.ArgumentParser(description="Fake Backend Multicore & Method PoC (Qiskit-only)")
    parser.add_argument("backends", nargs="+", help="List of backend names")
    parser.add_argument("--qubits", type=int, default=6, help="Number of qubits to benchmark")
    parser.add_argument("--threads", type=int, default=6, help="Number of threads for multicore test")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per configuration")
    parser.add_argument("--methods", nargs="+", default=["automatic"], help="Simulation methods")
    parser.add_argument("--output", default="benchmark_results_qiskit.yaml", help="Output YAML file")
    
    args = parser.parse_args()
    
    all_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "threads_tested": args.threads,
            "target_qubits": args.qubits,
            "n_runs": args.runs,
            "methods_tested": args.methods,
            "framework": "qiskit-only"
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
