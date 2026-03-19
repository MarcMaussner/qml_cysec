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
    
    # 2. QDA Noise Injection (The distinguishing part)
    for q in range(num_qubits):
        qc.rx(np.random.normal(0, 0.05), q)
        qc.ry(np.random.normal(0, 0.05), q)
        qc.rz(np.random.normal(0, 0.05), q)
    qc.barrier()
    
    # 3. Classifier Part (Data Re-uploading)
    for l in range(num_layers):
        for q in range(num_qubits):
            # In QDA, input noise is added during training, but the circuit structure is the same.
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
    num_layers = 16 # Original hyperparameter for QDA-ASP
    
    output_dir = "pictures_qiskit_qda_asp_4q"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating Qiskit QDA-ASP 4-qubit circuit...")
    # Use a smaller version for clear visualization if needed, but structure is what matters.
    # Here we draw 2 layers to show the repeating block structure.
    qc_qda = build_full_circuit(num_qubits, num_layers=2, num_reup=3)
    
    fig_qda = qc_qda.draw(output='mpl', style='iqp')
    fig_qda.suptitle("Qiskit QDA-ASP 4-qubit Circuit Structure", fontsize=16)
    
    save_path = os.path.join(output_dir, "qiskit_qda_asp_4q_circuit.png")
    fig_qda.savefig(save_path, bbox_inches='tight')
    plt.close(fig_qda)
    
    print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
