# Analysis of FakeLimaV2 Multicore Execution

This document provides an analysis of how to enable multicore execution for the `FakeLimaV2` device when used with PennyLane and Qiskit.

## 1. Underlying Simulator
The `FakeLimaV2` device from `qiskit_ibm_runtime.fake_provider` is a noisy simulator that mimics the IBM Lima quantum processor. Under the hood, it uses the **Qiskit Aer** (`AerSimulator`) to perform the quantum simulations, provided that `qiskit-aer` is installed in the environment.

## 2. Enabling Multicore Support
Qiskit Aer supports parallelization at several levels. To make the device run on multiple cores, you can configure the following options:

### A. Key Parallelization Options
*   `max_parallel_threads`: Specifies the number of CPU threads used for a single circuit simulation.
*   `max_parallel_experiments`: Specifies the number of circuits to simulate in parallel. This is particularly useful in PennyLane when executing batches of circuits (e.g., during gradient computation or multiple observable measurements).
*   `max_parallel_shots`: Specifies the number of shots to simulate in parallel for methods that support it (like `density_matrix`).

### B. Implementation via PennyLane Device
The recommended way to apply these settings in your PennyLane script is during the device initialization. The `qiskit.remote` device implementation in PennyLane-Qiskit passes additional keyword arguments to the Qiskit backend options.

```python
import pennylane as qml
from qiskit_ibm_runtime.fake_provider import FakeLimaV2

# Initialize the fake backend
fake_backend = FakeLimaV2()

# Create the PennyLane device with multicore options
dev = qml.device(
    'qiskit.remote', 
    wires=5, 
    backend=fake_backend,
    # Multicore options
    max_parallel_threads=8,     # Use 8 threads for single circuits
    max_parallel_experiments=4  # Run up to 4 circuits in parallel
)
```

### C. Direct Backend Configuration
Alternatively, you can set the options directly on the `fake_backend` instance before passing it to the PennyLane device:

```python
fake_backend = FakeLimaV2()
fake_backend.set_options(max_parallel_threads=8, max_parallel_experiments=4)

dev = qml.device('qiskit.remote', wires=5, backend=fake_backend)
```

## 3. High-Level PennyLane Parallelization
If your script involves heavy gradient calculations, you can also consider PennyLane's high-level parallelization features. For example, using `jax.jit` or `torch.multiprocessing` can help, but for a single QNode execution, the backend options above are the most direct way to leverage multicore CPUs.

## 4. Performance Considerations
*   **Density Matrix Simulation**: Since `FakeLimaV2` includes noise, it usually defaults to `density_matrix` simulation, which is computationally expensive ($4^n$ memory/time scaling). Multicore support is critical here.
*   **Circuit Batching**: PennyLane naturally batches circuits. Setting `max_parallel_experiments` to match your CPU core count can significantly speed up the evaluation of batches.

---
**Status**: Analysis complete.
**Author**: Antigravity
**Date**: 2026-02-10
