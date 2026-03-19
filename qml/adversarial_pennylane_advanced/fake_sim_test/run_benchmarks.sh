#!/bin/bash

# Configuration
PYTHON="./venv/bin/python"
SCRIPT="poc_multicore.py"

# Verification of environment
if [ ! -f "$PYTHON" ]; then
    echo "Error: Virtual environment python not found at $PYTHON"
    exit 1
fi

if [ ! -f "$SCRIPT" ]; then
    echo "Error: Benchmark script not found at $SCRIPT"
    exit 1
fi

echo "Starting Quantum ML Benchmarking Suite..."

# Run Set 1:
# Backends: "FakeLimaV2", "FakeJakartaV2", "FakeTorino"
# Qubits: (1, 4)
# Threads: (1, 4)
SET1_BACKENDS=("FakeLimaV2" "FakeGuadalupeV2" "FakeTorino")
SET1_QUBITS=(1 4)
SET1_THREADS=(1 4)

echo ">>> Standard Runs (Set 1)"
for backend in "${SET1_BACKENDS[@]}"; do
    for qubits in "${SET1_QUBITS[@]}"; do
        for threads in "${SET1_THREADS[@]}"; do
            OUTPUT="benchmark_${backend}_q${qubits}_t${threads}.yaml"
            echo "Executing: $backend | Wires: $qubits | Threads: $threads"
            $PYTHON "$SCRIPT" "$backend" --qubits "$qubits" --threads "$threads" --output "$OUTPUT"
        done
    done
done

# Run Set 2:
# Backends: "FakeJakartaV2", "FakeTorino"
# Qubits: (6, 8)
# Threads: (1, 6, 8)
SET2_BACKENDS=("FakeGuadalupeV2" "FakeTorino")
SET2_QUBITS=(6 8)
SET2_THREADS=(1 6 8)

echo ">>> Advanced Runs (Set 2)"
for backend in "${SET2_BACKENDS[@]}"; do
    for qubits in "${SET2_QUBITS[@]}"; do
        for threads in "${SET2_THREADS[@]}"; do
            OUTPUT="benchmark_${backend}_q${qubits}_t${threads}.yaml"
            echo "Executing: $backend | Wires: $qubits | Threads: $threads"
            $PYTHON "$SCRIPT" "$backend" --qubits "$qubits" --threads "$threads" --output "$OUTPUT"
        done
    done
done

echo "Benchmarking suite completed."
