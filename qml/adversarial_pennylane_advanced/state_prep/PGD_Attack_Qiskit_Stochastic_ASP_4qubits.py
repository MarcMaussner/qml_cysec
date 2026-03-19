#!/usr/bin/env python
# coding: utf-8

import os
import time
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import skimage.transform
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector, state_fidelity as fidelity
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
from torchmetrics.classification import MulticlassConfusionMatrix
from scipy.optimize import minimize

# Total execution timer
total_start = time.time()

# Create pictures directory if it doesn't exist
output_dir = "pictures_qiskit_stochastic_asp_4q"
os.makedirs(output_dir, exist_ok=True)

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

# ## Variational ASP Implementation
class VariationalASP:
    def __init__(self, num_qubits, num_layers=2):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        
    def get_ansatz(self, params):
        qc = QuantumCircuit(self.num_qubits)
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

    def fidelity_objective(self, params, target_sv):
        qc = self.get_ansatz(params)
        approx_sv = Statevector.from_instruction(qc)
        return 1 - fidelity(approx_sv, target_sv)

    def fit(self, target_vector, maxiter=50):
        target_vector = target_vector.flatten()
        norm = np.linalg.norm(target_vector)
        if norm > 0:
            target_vector = target_vector / norm
        else:
            target_vector = np.zeros(2**self.num_qubits)
            target_vector[0] = 1.0
            
        target_sv = Statevector(target_vector)
        initial_params = np.random.rand(self.num_layers * self.num_qubits * 3) * 2 * np.pi
        
        res = minimize(
            self.fidelity_objective, 
            initial_params, 
            args=(target_sv,), 
            method='SLSQP', 
            options={'maxiter': maxiter}
        )
        return res.x, 1 - res.fun

# ## Dataset loading and preprocessing
def load_dataset():
    dataset_file = "datasets/plus-minus/plus-minus.h5"
    if not os.path.exists(dataset_file):
        import pennylane as qml
        print("Local dataset not found, downloading via PennyLane...")
        [pm] = qml.data.load('other', name='plus-minus', directory="datasets")
        X_train_orig = pm.img_train
        X_test_orig = pm.img_test
        Y_train = pm.labels_train
        Y_test = pm.labels_test
    else:
        with h5py.File(dataset_file, "r") as f:
            X_train_orig = np.array(f['img_train'])
            X_test_orig = np.array(f['img_test'])
            Y_train = np.array(f['labels_train'])
            Y_test = np.array(f['labels_test'])
    
    # Resize to 8x8 as in original 4-qubit script
    X_train = np.array([skimage.transform.resize(img, (8, 8)) for img in X_train_orig])
    X_test = np.array([skimage.transform.resize(img, (8, 8)) for img in X_test_orig])
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = load_dataset()

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
        plt.savefig(os.path.join(output_dir, save_name))
        print(f"Saved image to {output_dir}/{save_name}")
    plt.close()

visualize_data(X_train[:4], Y_train[:4], save_name="initial_data_visualization.png")

# ## Hyperparameters
input_dim = 8*8
num_classes = 4
num_layers = 16
num_qubits = 4
num_reup = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize ASP
asp = VariationalASP(num_qubits, num_layers=2)

# ## Qiskit Backend Setup
fake_backend = FakeGuadalupeV2()
fake_backend.set_options(max_parallel_threads=8, method="automatic")

# ## Qiskit-Torch Integration
class QiskitQuantumFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, bias, asp_params=None):
        ctx.save_for_backward(inputs, weights, bias)
        ctx.asp_params = asp_params
        
        inputs_reshaped = inputs.detach().numpy().reshape((num_layers, num_qubits, 3))
        w = weights.detach().numpy()
        b = bias.detach().numpy()
        
        qc = QiskitQuantumFunction._build_circuit(inputs_reshaped, w, b, asp_params=asp_params)
        t_qc = transpile(qc, fake_backend)
        result = fake_backend.run(t_qc, shots=1024).result()
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
    def _build_circuit(inputs, weights, bias, shifts=None, asp_params=None):
        qc = QuantumCircuit(num_qubits)
        if asp_params is not None:
            asp_qc = asp.get_ansatz(asp_params)
            qc.compose(asp_qc, inplace=True)
            
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
        asp_params = ctx.asp_params
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
                        s_p = np.zeros_like(weights_np); s_p[l, q, i] = shift_val
                        circuits.append(QiskitQuantumFunction._build_circuit(inputs_np, weights_np, bias_np, s_p, asp_params=asp_params))
                        s_m = np.zeros_like(weights_np); s_m[l, q, i] = -shift_val
                        circuits.append(QiskitQuantumFunction._build_circuit(inputs_np, weights_np, bias_np, s_m, asp_params=asp_params))
                        circ_idx_map.append((l, q, i))

        if not circuits:
            return grad_inputs, grad_weights, grad_bias, None

        with Timer(f"Batch Execution ({len(circuits)} circuits)"):
            t_circuits = transpile(circuits, fake_backend)
            results = fake_backend.run(t_circuits, shots=1024).result()
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
        return grad_inputs, grad_weights, grad_bias, None

class QML_classifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_qubits, num_layers, stochastic_std=0.0):
        super().__init__()
        torch.manual_seed(1337)
        self.num_qubits = num_qubits
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.weights = torch.nn.Parameter(0.1 * torch.rand((num_layers, num_qubits, 3)))
        self.bias = torch.nn.Parameter(0.1 * torch.rand((num_layers, num_qubits, 3)))
        self.asp_cache = {}
        self.stochastic_std = stochastic_std

    def forward(self, x, idx=None):
        if idx is not None and idx in self.asp_cache:
            asp_p = self.asp_cache[idx]
        else:
            # 4 qubits -> 2^4 = 16 amplitudes. Resize 8x8 image to 4x4.
            img_resized = skimage.transform.resize(x.detach().numpy().reshape(8, 8), (4, 4)).flatten()
            p, f = asp.fit(img_resized, maxiter=40)
            asp_p = torch.tensor(p)
            if idx is not None: self.asp_cache[idx] = asp_p
            print(f"   [ASP] Fitted sample {idx if idx is not None else ''} | Fidelity: {f:.4f}")
            
        # Introduce stochasticity in ASP parameters during inference/attack
        if self.stochastic_std > 0:
            # We detached to ensure this is purely a stochastic injection not part of backprop calculation
            noise = torch.randn_like(asp_p) * self.stochastic_std
            asp_p_stochastic = asp_p + noise
        else:
            asp_p_stochastic = asp_p
            
        inputs_stack = torch.cat([x] * num_reup)
        return QiskitQuantumFunction.apply(inputs_stack, self.weights, self.bias, asp_p_stochastic)

# ## Training setup
learning_rate = 0.1
epochs = 4
batch_size = 20

feats_train = torch.from_numpy(X_train[:200]).reshape(200, -1).to(torch.float32)
feats_test = torch.from_numpy(X_test[:50]).reshape(50, -1).to(torch.float32)
labels_train = torch.from_numpy(Y_train[:200]).to(torch.long)
labels_test = torch.from_numpy(Y_test[:50]).to(torch.long)

# Initialize with stochastic_std=0 for normal training
model = QML_classifier(input_dim, num_classes, num_qubits, num_layers, stochastic_std=0.0)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def accuracy(labels, predictions):
    acc = 0
    for l, p in zip(labels, predictions):
        if torch.argmax(p) == l: acc += 1
    return acc / len(labels)

def print_acc(epoch, max_ep=4):
    with torch.no_grad():
        predictions_test = torch.stack([model(f, idx=1000+i) for i, f in enumerate(feats_test)])
        acc_test = accuracy(labels_test, predictions_test)
        print(f"Epoch {epoch}/{max_ep} | Acc val: {acc_test:0.4f}")

# Training Loop
print(f"Starting Qiskit Stochastic ASP training loop ({num_qubits} qubits)...")
num_train = feats_train.shape[0]
num_batches = num_train // batch_size

with Timer("QML Stochastic ASP Model Training"):
    for ep in range(epochs):
        perm = torch.randperm(num_train)
        for it in range(num_batches):
            optimizer.zero_grad()
            idx = perm[it*batch_size : (it+1)*batch_size]
            batch_feats = feats_train[idx]
            batch_labels = labels_train[idx]
            outputs = torch.stack([model(f, idx=i.item()) for i, f in zip(idx, batch_feats)])
            batch_loss = loss_fn(outputs, batch_labels)
            batch_loss.backward()
            optimizer.step()
        print_acc(ep+1, epochs)

# ## Evaluation and Attack
# Now enable stochasticity for evaluation
model.stochastic_std = 0.05 # 5% noise on ASP parameters

with Timer("Benign Evaluation (Stochastic)"):
    predictions_test = torch.stack([model(f, idx=1000+i) for i, f in enumerate(feats_test)])
    acc_benign = accuracy(labels_test, predictions_test)
    print(f"Final Benign Accuracy (Stochastic): {acc_benign:0.4f}")

    # Confusion Matrix
    metric = MulticlassConfusionMatrix(num_classes=4)
    preds_max = torch.argmax(predictions_test, dim=1)
    metric.update(preds_max, labels_test)
    fig, ax = metric.plot()
    plt.savefig(f"{output_dir}/confusion_matrix_benign.png")
    plt.close()

visualize_data(X_test[:4], Y_test[:4], [torch.argmax(p).item() for p in predictions_test[:4]], 
               save_name="benign_data_evaluation.png")

def PGD(model, feats, labels, epsilon=0.1, alpha=0.01, num_iter=10):
    delta = torch.zeros_like(feats, requires_grad=True)
    for t in range(num_iter):
        print(f"   - PGD Iteration {t+1}/{num_iter}...")
        feats_adv = feats + delta
        outputs = torch.stack([model(f, idx=2000+i) for i, f in enumerate(feats_adv)])
        l = loss_fn(outputs, labels)
        l.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()

print("\n--- Running PGD Attack (against Stochastic Model) ---")
attack_samples = 50
with Timer("PGD Attack Generation"):
    perturbations = PGD(model, feats_test[:attack_samples], labels_test[:attack_samples], epsilon=0.1)
    perturbed_x = feats_test[:attack_samples] + perturbations
    adv_preds = torch.stack([model(f, idx=2000+i) for i, f in enumerate(perturbed_x)])
    adv_acc = accuracy(labels_test[:attack_samples], adv_preds)
    print(f"Adversarial Accuracy (Stochastic): {adv_acc:0.4f}")

visualize_data(skimage.transform.resize(perturbed_x.detach().numpy().reshape(-1, 8, 8), (len(perturbed_x), 8, 8))[:4], 
               labels_test[:4].numpy(), [torch.argmax(p).item() for p in adv_preds[:4]], 
               save_name="adversarial_attack_evaluation.png")

# Confusion Matrix for Adversarial Attack
metric_adv = MulticlassConfusionMatrix(num_classes=4)
metric_adv.update(torch.argmax(adv_preds, dim=1), labels_test[:attack_samples])
fig_adv, ax_adv = metric_adv.plot()
plt.savefig(f"{output_dir}/confusion_matrix_adversarial.png")
plt.close()

# ## Adversarial Retraining
print("\n--- Starting Adversarial Retraining ---")
adv_train_samples = 20
perturbations_train = PGD(model, feats_train[:adv_train_samples], labels_train[:adv_train_samples], epsilon=0.1)
feats_retrain = torch.cat([feats_train, feats_train[:adv_train_samples] + perturbations_train])
labels_retrain = torch.cat([labels_train, labels_train[:adv_train_samples]])

with Timer("Adversarial Retraining"):
    retrain_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for ep in range(1):
        perm = torch.randperm(feats_retrain.shape[0])
        for it in range(len(perm)//batch_size):
            retrain_optimizer.zero_grad()
            idx = perm[it*batch_size : (it+1)*batch_size]
            batch_feats = feats_retrain[idx]
            batch_labels = labels_retrain[idx]
            outputs = torch.stack([model(f, idx=3000+i.item()) for i, f in zip(idx, batch_feats)])
            loss = loss_fn(outputs, batch_labels)
            loss.backward()
            retrain_optimizer.step()

    adv_preds_post = torch.stack([model(f, idx=2000+i) for i, f in enumerate(perturbed_x)])
    acc_robust = accuracy(labels_test[:attack_samples], adv_preds_post)
    print(f"Post-Retraining Adversarial Accuracy (Stochastic): {acc_robust:0.4f}")

visualize_data(skimage.transform.resize(perturbed_x.detach().numpy().reshape(-1, 8, 8), (len(perturbed_x), 8, 8))[:4], 
               labels_test[:4].numpy(), [torch.argmax(p).item() for p in adv_preds_post[:4]], 
               save_name="robustness_evaluation_after_retraining.png")

# Confusion Matrix for Robust Model
metric_robust = MulticlassConfusionMatrix(num_classes=4)
metric_robust.update(torch.argmax(adv_preds_post, dim=1), labels_test[:attack_samples])
fig_robust, ax_robust = metric_robust.plot()
plt.savefig(f"{output_dir}/confusion_matrix_robust_model.png")
plt.close()

print(f"\n✅ Total execution time: {time.time() - total_start:.2f} seconds")
