# Estimated Runtime Analysis: 4 Qubits vs 8 Qubits

Based on actual measured runtime of the golden sample (`PGD_Attack_Reduced_Noise_Fake_Guadalupe_4qubits.py`) on `FakeGuadalupeV2`, and empirical benchmark data.

**Measured 4-qubit runtime: 220,705 seconds (≈ 2.56 days)**

## Summary of Changes
| Feature | 4-Qubit Run (Actual) | 8-Qubit Run (Estimated) | Scaling Factor |
| :--- | :--- | :--- | :--- |
| **Qubits** | 4 | 8 | ~2.6× (Empirical) |
| **Circuit Depth** | 16 Layers | 32 Layers | 2.0× |
| **Parameters** | 192 | 768 | 4.0× |
| **Evaluations** | ~300k (est) | ~1.2M (est) | 4.0× |
| **Parallelization** | 1.0× (No speedup) | ~1.9× (Speedup) | 0.53× (Reduction) |
| **Total Runtime** | **2.56 days** (measured) | **~27.8 days** | **~10.86×** |

## Detailed Breakdown

### 1. Circuit Evaluations (Gradient Complexity)
The experiment uses `torch` with `pennylane`. Since the fake device is a noisy simulator, it uses **Parameter-Shift** or **Finite-Difference** for gradients.
- The number of circuit evaluations per gradient step scales linearly with the number of parameters.
- **4-Qubit script**: 16 layers × 4 qubits × 3 params/gate = **192 parameters**.
- **8-Qubit script**: 32 layers × 8 qubits × 3 params/gate = **768 parameters**.
- This results in a **4.0× increase** in circuit executions.

### 2. Simulation Time per Circuit
Simulation complexity scales roughly exponentially ($2^n$), but for noisy simulations on small qubit counts, noise model overhead is dominant.
- **Empirical Scaling**: Benchmarks show that for `FakeGuadalupeV2`, increasing from 4 to 8 qubits increases average execution time from **5.09s** to **13.14s** (a factor of **2.58×**).
- **Depth Scaling**: The 8-qubit script also doubles the layers (16 → 32), increasing execution time by an additional **2×**.
- **Net Per-Circuit Factor**: $2.58 \times 2 = \mathbf{5.16×}$.

### 3. Multithreading Speedup
At higher qubit counts, Qiskit Aer's parallelization becomes more effective.
- **4 Qubits**: Benchmarks show **no speedup** from multithreading (5.09s vs 5.09s).
- **8 Qubits**: Benchmarks show a **1.9× speedup** (13.14s → 6.89s) with `max_parallel_threads=8`.
- This partially offsets the complexity, reducing total time by a factor of ~0.53.

### 4. Final Calculation
$$\text{Total Scaling} = \frac{\text{Eval Scaling} \times \text{Sim Scaling}}{\text{Thread Speedup}} = \frac{4 \times 5.16}{1.9} \approx \mathbf{10.86×}$$

$$2.56 \text{ days} \times 10.86 \approx \mathbf{27.8 \text{ days}}$$

## Recommendation
Running the experiment for 8 qubits with the current settings is estimated to take ~**28 days**. To make this feasible, consider:
1. **Reducing Samples**: Decreasing the training subset from 200 to 50 samples → ~7 days.
2. **Reducing Layers**: Keeping 16 layers (same as 4-qubit run) → ~14 days.
3. **Both together**: ~3.5 days — comparable to the 4-qubit runtime.
4. **Adjoint Differentiation**: `diff_method="adjoint"` is significantly faster but may not work with the full `FakeGuadalupeV2` noise model.

*Last updated: 2026-03-02 (based on actual measured 4-qubit runtime of 220,705 s)*
