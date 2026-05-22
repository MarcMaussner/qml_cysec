#!/usr/bin/env python
# coding: utf-8

import os
import time
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
import argparse
import yaml
from qiskit import QuantumCircuit, transpile
from torchmetrics.classification import MulticlassConfusionMatrix

# Total execution timer
total_start = time.time()

# Create pictures directory if it doesn't exist
os.makedirs("pictures_qiskit", exist_ok=True)

# Helper for timing
class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        print(f"--- Starting {self.name} ---")
        return self
    def __exit__(self, *args):
        self.end = time.time()
        print(f"--- {self.name} took {self.end - self.start:.2f} seconds ---")

# Dataset loading and preprocessing
def load_dataset():
    dataset_file = "datasets/plus-minus/plus-minus.h5"
    if not os.path.exists(dataset_file):
        # Look in alternative paths to avoid re-downloading if already present
        alt_paths = [
            "../fake_sim/datasets/plus-minus/plus-minus.h5",
            "../datasets/plus-minus/plus-minus.h5",
            "../../fake_sim/datasets/plus-minus/plus-minus.h5"
        ]
        for alt in alt_paths:
            if os.path.exists(alt):
                dataset_file = alt
                break

    if not os.path.exists(dataset_file):
        # Fallback to downloading if not found locally
        import pennylane as qml
        print("Local dataset not found, downloading via PennyLane...")
        [pm] = qml.data.load('other', name='plus-minus', directory="datasets")
        X_train_orig = pm.img_train
        X_test_orig = pm.img_test
        Y_train = pm.labels_train
        Y_test = pm.labels_test
    else:
        print(f"Loading local dataset from: {dataset_file}")
        with h5py.File(dataset_file, "r") as f:
            X_train_orig = np.array(f['img_train'])
            X_test_orig = np.array(f['img_test'])
            Y_train = np.array(f['labels_train'])
            Y_test = np.array(f['labels_test'])
    
    # Resize to 8x8
    def resize_img(imgs):
        imgs = np.transpose(imgs, (1, 2, 0))
        imgs = skimage.transform.resize(imgs, (8, 8))
        return np.transpose(imgs, (2, 0, 1))

    X_train = resize_img(X_train_orig)
    X_test = resize_img(X_test_orig)
    
    return X_train, X_test, Y_train, Y_test

def visualize_data(x, y, pred=None, save_name=None):
    n_img = len(x)
    labels_list = ["\u2212", "\u002b", "\ua714", "\u02e7"]
    fig, axes = plt.subplots(1, 4, figsize=(8, 2))
    for i in range(min(n_img, 4)):
        axes[i].imshow(x[i], cmap="gray")
        if pred is None:
            axes[i].set_title("Label: {}".format(labels_list[y[i]]))
        else:
            axes[i].set_title("Label: {}, Pred: {}".format(labels_list[y[i]], labels_list[pred[i]]))
    plt.tight_layout(w_pad=2)
    if save_name:
        plt.savefig(os.path.join("pictures_qiskit", save_name))
        print(f"Saved image to pictures_qiskit/{save_name}")
    plt.close()

# Hyperparameters
input_dim = 8*8
num_classes = 4
num_layers = 16
num_qubits = 4
num_reup = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global placeholder for active backend
active_backend = None

# Qiskit-Torch Integration
class QiskitQuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias):
        ctx.save_for_backward(inputs, weights, bias)
        
        inputs_reshaped = inputs.detach().numpy().reshape((num_layers, num_qubits, 3))
        w = weights.detach().numpy()
        b = bias.detach().numpy()
        
        qc = QiskitQuantumFunction._build_circuit(inputs_reshaped, w, b)
        t_qc = transpile(qc, active_backend)
        result = active_backend.run(t_qc, shots=1024).result()
        counts = result.get_counts()
        
        return torch.tensor(QiskitQuantumFunction._get_expvals(counts), dtype=torch.float32, device=device)

    @staticmethod
    def _get_expvals(counts):
        expvals = []
        if not isinstance(counts, list): counts = [counts]
        
        for c in counts:
            sample_expvals = []
            for q in range(num_classes):
                z_exp = 0
                for bitstring, count in c.items():
                    val = 1 if bitstring[-(q+1)] == '0' else -1
                    z_exp += val * (count / 1024.0)
                sample_expvals.append(z_exp)
            expvals.append(sample_expvals)
            
        return expvals[0] if len(expvals) == 1 else expvals

    @staticmethod
    def _build_circuit(inputs, weights, bias, shifts=None):
        qc = QuantumCircuit(num_qubits)
        if shifts is None:
            shifts = np.zeros((num_layers, num_qubits, 3))
            
        for l in range(num_layers):
            for q in range(num_qubits):
                val = weights[l, q] * inputs[l, q] + bias[l, q] + shifts[l, q]
                qc.rz(val[0], q)
                qc.ry(val[1], q)
                qc.rz(val[2], q)
            r = (l % (num_qubits - 1)) + 1
            for q in range(num_qubits):
                qc.cx(q, (q + r) % num_qubits)
        qc.measure_all()
        return qc

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights, bias = ctx.saved_tensors
        grad_inputs = torch.zeros_like(inputs)
        grad_weights = torch.zeros_like(weights)
        grad_bias = torch.zeros_like(bias)
        
        shift_val = np.pi / 2
        
        inputs_np = inputs.detach().numpy().reshape(num_layers, num_qubits, 3)
        weights_np = weights.detach().numpy().reshape(num_layers, num_qubits, 3)
        bias_np = bias.detach().numpy().reshape(num_layers, num_qubits, 3)
        
        circuits = []
        circ_idx_map = []

        with Timer("Building Gradient Circuits"):
            for l in range(num_layers):
                for q in range(num_qubits):
                    for i in range(3):
                        # Plus shift
                        s_p = np.zeros_like(weights_np)
                        s_p[l, q, i] = shift_val
                        circuits.append(QiskitQuantumFunction._build_circuit(inputs_np, weights_np, bias_np, s_p))
                        
                        # Minus shift
                        s_m = np.zeros_like(weights_np)
                        s_m[l, q, i] = -shift_val
                        circuits.append(QiskitQuantumFunction._build_circuit(inputs_np, weights_np, bias_np, s_m))
                        
                        circ_idx_map.append((l, q, i))

        if not circuits:
            return grad_inputs, grad_weights, grad_bias

        with Timer(f"Batch Execution ({len(circuits)} circuits)"):
            t_circuits = transpile(circuits, active_backend)
            results = active_backend.run(t_circuits, shots=1024).result()
            all_counts = results.get_counts()
            all_expvals = QiskitQuantumFunction._get_expvals(all_counts)
        
        grad_inputs_reshaped = np.zeros((num_layers, num_qubits, 3))
        
        for idx, (l, q, i) in enumerate(circ_idx_map):
            e_p = torch.tensor(all_expvals[2*idx], dtype=torch.float32, device=device)
            e_m = torch.tensor(all_expvals[2*idx+1], dtype=torch.float32, device=device)
            
            dL_dTheta = torch.sum(grad_output * 0.5 * (e_p - e_m)).item()
            
            grad_weights[l, q, i] = float(dL_dTheta * inputs_np[l, q, i])
            grad_bias[l, q, i] = float(dL_dTheta)
            grad_inputs_reshaped[l, q, i] = dL_dTheta * weights_np[l, q, i]

        grad_inputs = torch.from_numpy(grad_inputs_reshaped.flatten()).to(device).to(torch.float32)

        return grad_inputs, grad_weights, grad_bias

class QML_classifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_qubits, num_layers):
        super().__init__()
        torch.manual_seed(1337)
        self.num_qubits = num_qubits
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.weights = torch.nn.Parameter(0.1 * torch.rand((num_layers, num_qubits, 3)))
        self.bias = torch.nn.Parameter(0.1 * torch.rand((num_layers, num_qubits, 3)))

    def forward(self, x):
        inputs_stack = torch.cat([x] * num_reup)
        return QiskitQuantumFunction.apply(inputs_stack, self.weights, self.bias)

def export_circuit_diagram():
    qc = QuantumCircuit(num_qubits)
    for l in range(2):
        for q in range(num_qubits):
            qc.rz(0.1, q); qc.ry(0.1, q); qc.rz(0.1, q)
        r = (l % (num_qubits - 1)) + 1
        for q in range(num_qubits):
            qc.cx(q, (q + r) % num_qubits)
    qc.draw(output='mpl', filename='pictures_qiskit/classifier_circuit.png')
    print("Saved circuit diagram to pictures_qiskit/classifier_circuit.png")

def main():
    parser = argparse.ArgumentParser(description="Quantum Machine Learning Model Training (Qiskit/PyTorch)")
    parser.add_argument("--backend_name", type=str, default="FakeGuadalupeV2",
                        help="Qiskit fake backend class name from qiskit_ibm_runtime.fake_provider (e.g., FakeGuadalupeV2, FakeLimaV2) or 'AerSimulator'")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    args = parser.parse_args()

    global active_backend

    # 1. Setup Backend
    backend_name = args.backend_name
    print(f"Setting up Qiskit backend: {backend_name}...")
    if backend_name.lower() == "aersimulator":
        from qiskit_aer import AerSimulator
        active_backend = AerSimulator()
    else:
        from qiskit_ibm_runtime import fake_provider
        try:
            backend_cls = getattr(fake_provider, backend_name)
            active_backend = backend_cls()
        except AttributeError:
            raise ValueError(f"Unknown fake backend: {backend_name}. Ensure it's in qiskit_ibm_runtime.fake_provider.")
    
    # Configure threading options
    active_backend.set_options(max_parallel_threads=8, method="automatic")

    # 2. Load dataset
    X_train, X_test, Y_train, Y_test = load_dataset()
    visualize_data(X_train[:4], Y_train[:4], save_name="initial_data_visualization.png")

    # 3. Instantiate model
    model = QML_classifier(input_dim, num_classes, num_qubits, num_layers)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Export reference 2-layer diagram
    export_circuit_diagram()

    # Data selection (replicating original script slicing)
    feats_train = torch.from_numpy(X_train[:200]).reshape(200, -1).to(torch.float32)
    feats_test = torch.from_numpy(X_test[:50]).reshape(50, -1).to(torch.float32)
    labels_train = torch.from_numpy(Y_train[:200]).to(torch.long)
    labels_test = torch.from_numpy(Y_test[:50]).to(torch.long)

    def accuracy(labels, predictions):
        acc = 0
        for l, p in zip(labels, predictions):
            if torch.argmax(p) == l:
                acc += 1
        return acc / len(labels)

    def gen_batches(num_samples, num_batches):
        assert num_samples % num_batches == 0
        perm_ind = torch.reshape(torch.randperm(num_samples), (num_batches, -1))
        return perm_ind

    def print_acc(epoch, max_ep=4):
        with torch.no_grad():
            predictions_test = torch.stack([model(f) for f in feats_test])
            acc_test = accuracy(labels_test, predictions_test)
            print(f"Epoch {epoch}/{max_ep} | Acc val: {acc_test:0.4f}")
            return acc_test

    # 4. Training Loop
    print(f"Starting Qiskit training loop on {backend_name} ({num_qubits} qubits, {num_layers} layers)...")
    num_train = feats_train.shape[0]
    num_batches = num_train // args.batch_size

    with Timer("QML Model Training"):
        for ep in range(args.epochs):
            batch_ind = gen_batches(num_train, num_batches)
            print_acc(ep, max_ep=args.epochs)
            
            for it in range(num_batches):
                optimizer.zero_grad()
                batch_feats = feats_train[batch_ind[it]]
                batch_labels = labels_train[batch_ind[it]]
                
                outputs = torch.stack([model(f) for f in batch_feats])
                batch_loss = loss_fn(outputs, batch_labels)
                batch_loss.backward()
                optimizer.step()
                
        final_benign_acc = print_acc(args.epochs, max_ep=args.epochs)

    # 5. Benign Evaluation
    with Timer("Benign Evaluation"):
        predictions_test = torch.stack([model(f) for f in feats_test])
        acc_benign = accuracy(labels_test, predictions_test)
        print(f"Final Benign Accuracy: {acc_benign:0.4f}")

        # Confusion Matrix
        metric = MulticlassConfusionMatrix(num_classes=4)
        preds_max = torch.argmax(predictions_test, dim=1)
        metric.update(preds_max, labels_test)
        fig, ax = metric.plot()
        plt.savefig(f"pictures_qiskit/confusion_matrix_benign_{backend_name}.png")
        plt.close()
        print(f"Saved confusion matrix to pictures_qiskit/confusion_matrix_benign_{backend_name}.png")

    visualize_data(X_test[:4], Y_test[:4], [torch.argmax(p).item() for p in predictions_test[:4]], 
                   save_name=f"benign_data_evaluation_{backend_name}.png")

    # 6. Save parameters to YAML
    yaml_filename = f"{backend_name}_training.yaml"
    print(f"Saving trained model parameters to {yaml_filename}...")
    
    weights_list = model.weights.detach().cpu().numpy().tolist()
    bias_list = model.bias.detach().cpu().numpy().tolist()

    save_data = {
        "backend_name": backend_name,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "final_benign_accuracy": float(final_benign_acc),
        "hyperparameters": {
            "num_layers": num_layers,
            "num_qubits": num_qubits,
            "num_reup": num_reup,
            "input_dim": input_dim,
            "num_classes": num_classes
        },
        "weights": weights_list,
        "bias": bias_list
    }

    with open(yaml_filename, "w") as f:
        yaml.safe_dump(save_data, f, default_flow_style=False)

    print(f"✅ Training script complete! Parameters successfully written to {yaml_filename}")
    print(f"✅ Total execution time: {time.time() - total_start:.2f} seconds")

if __name__ == "__main__":
    main()
