import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, fidelity
from scipy.optimize import minimize
import time

def get_ansatz(params, num_qubits, num_layers):
    qc = QuantumCircuit(num_qubits)
    p = params.reshape((num_layers, num_qubits, 3))
    for l in range(num_layers):
        for q in range(num_qubits):
            qc.u(p[l, q, 0], p[l, q, 1], p[l, q, 2], q)
        for q in range(num_qubits):
            qc.cx(q, (q + 1) % num_qubits)
    return qc

def objective(params, target_sv, num_qubits, num_layers):
    qc = get_ansatz(params, num_qubits, num_layers)
    approx_sv = Statevector.from_instruction(qc)
    return 1 - fidelity(approx_sv, target_sv)

num_qubits = 8
num_layers = 2
target_vector = np.random.rand(2**num_qubits)
target_vector /= np.linalg.norm(target_vector)
target_sv = Statevector(target_vector)

start = time.time()
initial_params = np.random.rand(num_layers * num_qubits * 3) * 2 * np.pi
res = minimize(objective, initial_params, args=(target_sv, num_qubits, num_layers), method='SLSQP', options={'maxiter': 5})
print(f"Fitting took {time.time() - start:.2f} seconds for 5 iterations.")
print(f"Final fidelity: {1 - res.fun:.4f}")
