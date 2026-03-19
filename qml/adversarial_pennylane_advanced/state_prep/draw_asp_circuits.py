import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit

# Re-implementing the core circuit building logic from the ASP scripts
class VariationalASP:
    def __init__(self, num_qubits, num_layers=2):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
    def get_ansatz(self, params):
        qc = QuantumCircuit(self.num_qubits, name="ASP Ansatz")
        if torch.is_tensor(params):
            p_np = params.detach().numpy().reshape((self.num_layers, self.num_qubits, 3))
        else:
            p_np = params.reshape((self.num_layers, self.num_qubits, 3))
            
        for l in range(self.num_layers):
            for q in range(self.num_qubits):
                qc.u(float(p_np[l, q, 0]), float(p_np[l, q, 1]), float(p_np[l, q, 2]), q)
            for q in range(self.num_qubits):
                qc.cx(q, (q + 1) % self.num_qubits)
        return qc

def build_full_circuit(num_qubits, num_layers, num_reup, asp_layers=2):
    asp = VariationalASP(num_qubits, num_layers=asp_layers)
    
    # Dummy parameters for visualization
    asp_params = np.random.rand(asp_layers * num_qubits * 3)
    weights = np.random.rand(num_layers, num_qubits, 3)
    bias = np.random.rand(num_layers, num_qubits, 3)
    inputs = np.random.rand(num_layers, num_qubits, 3)
    
    qc = QuantumCircuit(num_qubits)
    
    # 1. ASP Part
    asp_qc = asp.get_ansatz(asp_params)
    qc.compose(asp_qc, inplace=True)
    qc.barrier()
    
    # 2. Classifier Part (Data Re-uploading)
    for l in range(num_layers):
        for q in range(num_qubits):
            val = weights[l, q] * inputs[l, q] + bias[l, q]
            qc.rz(val[0], q)
            qc.ry(val[1], q)
            qc.rz(val[2], q)
        
        r = (l % (num_qubits - 1)) + 1
        for q in range(num_qubits):
            qc.cx(q, (q + r) % num_qubits)
        qc.barrier()
        
    qc.measure_all()
    return qc

def main():
    num_qubits = 4
    num_layers = 16 # Original hyperparameter
    num_reup = 3    # Original hyperparameter (though used for input stacking in code)
    
    # Note: The code uses 16 layers in build_circuit, but for a clear visualization 
    # of the structure, we might want to draw a smaller version or just one block.
    # However, I will draw the full structural version but perhaps with fewer layers 
    # for clarity in the PNG if it gets too long, or just the full one.
    
    print("Generating Qiskit ASP 4-qubit circuit...")
    qc_asp = build_full_circuit(num_qubits, num_layers=2, num_reup=3)
    
    # Save ASP circuit
    fig_asp = qc_asp.draw(output='mpl', style='iqp')
    fig_asp.suptitle("Qiskit ASP 4-qubit Circuit Structure", fontsize=16)
    fig_asp.savefig("qiskit_asp_4q_circuit.png", bbox_inches='tight')
    plt.close(fig_asp)
    print("Saved qiskit_asp_4q_circuit.png")

    print("Generating Qiskit Stochastic ASP 4-qubit circuit...")
    qc_stochastic = build_full_circuit(num_qubits, num_layers=2, num_reup=3)
    fig_stochastic = qc_stochastic.draw(output='mpl', style='iqp')
    fig_stochastic.suptitle("Qiskit Stochastic ASP 4-qubit Circuit Structure (Noisy Params)", fontsize=16)
    fig_stochastic.savefig("qiskit_stochastic_asp_4q_circuit.png", bbox_inches='tight')
    plt.close(fig_stochastic)
    print("Saved qiskit_stochastic_asp_4q_circuit.png")

if __name__ == "__main__":
    main()
