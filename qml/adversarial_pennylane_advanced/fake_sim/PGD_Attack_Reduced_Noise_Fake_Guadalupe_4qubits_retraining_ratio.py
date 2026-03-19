#!/usr/bin/env python
# coding: utf-8

# # Adversarial attacks and robustness for QML - Retraining Ratio Experiment
# This script implements a 50/50 ratio of adversarial to benign samples during retraining.

import pennylane as qml
from pennylane import numpy as np
import torch
from matplotlib import pyplot as plt
import skimage
import os
import time

# Total execution timer
total_start = time.time()

# Create pictures directory if it doesn't exist
os.makedirs("pictures_ratio", exist_ok=True)

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


# ## Datatset and visualization

# Check if dataset is locally stored
dataset_file = "datasets/plus-minus/plus-minus.h5"
if os.path.exists(dataset_file):
    print(f"Loading local dataset from {dataset_file}")
    from pennylane.data import Dataset
    pm = Dataset.open(dataset_file, mode="r")
else:
    print("Local dataset not found, downloading...")
    [pm] = qml.data.load('other', name='plus-minus', directory="datasets")

X_train_orig = pm.img_train  # shape (1000,16,16)
X_test_orig = pm.img_test  # shape (200,16,16)
Y_train = pm.labels_train  # shape (1000,)
Y_test = pm.labels_test  # shape (200,)

X_train = np.transpose(X_train_orig, (1, 2, 0)) # put height and width in front
X_train = skimage.transform.resize(X_train, (8, 8)) 
X_train = np.transpose(X_train, (2, 0, 1) ) # move back 

X_test = np.transpose(X_test_orig, (1, 2, 0)) # put height and width in front
X_test = skimage.transform.resize(X_test, (8, 8)) 
X_test = np.transpose(X_test, (2, 0, 1) ) # move back 

# show one image from each class
x_vis = [
    (X_train[Y_train == 0])[0],
    (X_train[Y_train == 1])[0],
    (X_train[Y_train == 2])[0],
    (X_train[Y_train == 3])[0],
]
y_vis = [0, 1, 2, 3]


def visualize_data(x, y, pred=None, save_name=None):
    n_img = len(x)
    labels_list = ["\u2212", "\u002b", "\ua714", "\u02e7"]
    fig, axes = plt.subplots(1, 4, figsize=(8, 2))
    for i in range(n_img):
        axes[i].imshow(x[i], cmap="gray")
        if pred is None:
            axes[i].set_title("Label: {}".format(labels_list[y[i]]))
        else:
            axes[i].set_title("Label: {}, Pred: {}".format(labels_list[y[i]], labels_list[pred[i]]))
    plt.tight_layout(w_pad=2)
    if save_name:
        plt.savefig(os.path.join("pictures_ratio", save_name))
        print(f"Saved image to pictures_ratio/{save_name}")
    plt.close()


visualize_data(x_vis, y_vis, save_name="initial_data_visualization.png")


# ## QML classification

from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

fake_backend = FakeGuadalupeV2()
fake_backend.set_options(
    max_parallel_threads=4, 
    max_parallel_experiments=1,
    method="automatic"
)

fake_dev = qml.device('qiskit.remote', wires=fake_backend.num_qubits, backend=fake_backend, shots=1024)


# build circuit
#### Hyperparameters ####
input_dim = 8*8
num_classes = 4
num_layers = 16
num_qubits = 4
num_reup = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class QML_classifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim, num_qubits, num_layers):
        super().__init__()
        torch.manual_seed(1337)
        self.num_qubits = num_qubits
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.device = fake_dev
        self.weights_shape = qml.StronglyEntanglingLayers.shape(
            n_layers=self.num_layers, n_wires=self.num_qubits
        )

        @qml.qnode(self.device, interface="torch")
        def circuit(inputs, weights, bias):
            inputs = torch.reshape(inputs, self.weights_shape)
            qml.StronglyEntanglingLayers(
                weights=weights * inputs + bias, wires=range(self.num_qubits)
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(self.output_dim)]

        param_shapes = {"weights": self.weights_shape, "bias": self.weights_shape}
        init_vals = {
            "weights": 0.1 * torch.rand(self.weights_shape),
            "bias": 0.1 * torch.rand(self.weights_shape),
        }

        self.qcircuit = qml.qnn.TorchLayer(
            qnode=circuit, weight_shapes=param_shapes, init_method=init_vals
        )

    def forward(self, x):
        inputs_stack = torch.hstack([x] * num_reup)
        return self.qcircuit(inputs_stack)


# train classifier
#### Hyperparameters ####
learning_rate = 0.1
epochs = 4
batch_size = 20

feats_train = torch.from_numpy(X_train[:200]).reshape(200, -1).to(device)
feats_test = torch.from_numpy(X_test[:50]).reshape(50, -1).to(device)
labels_train = torch.from_numpy(Y_train[:200]).to(device)
labels_test = torch.from_numpy(Y_test[:50]).to(device)
num_train = feats_train.shape[0]

qml_model = QML_classifier(input_dim, num_classes, num_qubits, num_layers)
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(qml_model.parameters(), lr=learning_rate)
num_batches = feats_train.shape[0] // batch_size


def accuracy(labels, predictions):
    acc = 0
    for l, p in zip(labels, predictions):
        if torch.argmax(p) == l:
            acc += 1
    acc = acc / len(labels)
    return acc


def gen_batches(num_samples, num_batches):
    assert num_samples % num_batches == 0
    perm_ind = torch.reshape(torch.randperm(num_samples), (num_batches, -1))
    return perm_ind


def print_acc(epoch, max_ep=4):
    predictions_train = [qml_model(f) for f in feats_train[:50]]
    predictions_test = [qml_model(f) for f in feats_test]
    cost_approx_train = loss(torch.stack(predictions_train), labels_train[:50])
    cost_approx_test = loss(torch.stack(predictions_test), labels_test)
    acc_approx_train = accuracy(labels_train[:50], predictions_train)
    acc_approx_test = accuracy(labels_test, predictions_test)
    print(
        f"Epoch {epoch}/{max_ep} | Approx Cost (train): {cost_approx_train:0.7f} | Cost (val): {cost_approx_test:0.7f} |"
        f" Approx Acc train: {acc_approx_train:0.7f} | Acc val: {acc_approx_test:0.7f}"
    )


print(f"Starting training loop ({num_qubits} qubits, {num_layers} layers)...")

with Timer("QML Model Training (Initial)"):
    for ep in range(0, epochs):
        batch_ind = gen_batches(num_train, num_batches)
        print_acc(epoch=ep)
        for it in range(num_batches):
            optimizer.zero_grad()
            feats_train_batch = feats_train[batch_ind[it]]
            labels_train_batch = labels_train[batch_ind[it]]
            outputs = [qml_model(f) for f in feats_train_batch]
            batch_loss = loss(torch.stack(outputs), labels_train_batch)
            batch_loss.backward()
            optimizer.step()
    print_acc(epochs)


# evaluate benign data
x_vis_torch = torch.from_numpy(np.array(x_vis).reshape(4, -1))
y_vis_torch = torch.from_numpy(np.array(y_vis))
benign_preds = [qml_model(f) for f in x_vis_torch]
benign_class_output = [torch.argmax(p) for p in benign_preds]
visualize_data(x_vis, y_vis, benign_class_output, save_name="benign_data_evaluation.png")


with Timer("Benign Evaluation"):
    predictions_test = [qml_model(f) for f in feats_test]
    acc_approx_test = accuracy(labels_test, predictions_test)
    print(f"Acc val: {acc_approx_test:0.7f}")


# confusion matrix on test set
from torchmetrics.classification import MulticlassConfusionMatrix
metric = MulticlassConfusionMatrix(num_classes=4)
preds_max = [torch.argmax(predictions_test[i]).detach().numpy().item() for i in range(len(predictions_test))]
metric.update(torch.tensor(preds_max), labels_test)
fig_, ax_ = metric.plot()
plt.savefig(os.path.join("pictures_ratio", "confusion_matrix_benign.png"))
plt.close()


# ## now attack
def PGD(model, feats, labels, epsilon=0.1, alpha=0.01, num_iter=10):
    delta = torch.zeros_like(feats, requires_grad=True)
    for t in range(num_iter):
        feats_adv = feats + delta
        outputs = [model(f) for f in feats_adv]
        l = loss(torch.stack(outputs), labels)
        l.backward()
        delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
    return delta.detach()


with Timer("PGD Attack Generation (Vis Samples)"):
    perturbations = PGD(model=qml_model, feats=x_vis_torch, labels=y_vis_torch, epsilon=0.1)
perturbed_x = x_vis_torch + perturbations
adversarial_preds = [qml_model(f) for f in perturbed_x]
adversarial_class_output = [torch.argmax(p) for p in adversarial_preds]
visualize_data(perturbed_x.reshape(-1, 8, 8), y_vis, adversarial_class_output, save_name="adversarial_attack_evaluation.png")


with Timer("Adversarial Evaluation (Full Test Set)"):
    perturbation_test = PGD(model=qml_model, feats=feats_test, labels=labels_test, epsilon=0.1)
    perturbed_x_test = feats_test + perturbation_test
    adversarial_preds_test = [qml_model(f) for f in perturbed_x_test]
acc_approx_test = accuracy(labels_test, adversarial_preds_test)
print(f"Acc val: {acc_approx_test:0.7f}")


# confusion matrix on adversarial test set
from torchmetrics.classification import MulticlassConfusionMatrix
metric = MulticlassConfusionMatrix(num_classes=4)
preds_max = [torch.argmax(adversarial_preds_test[i]).detach().numpy().item() for i in range(len(adversarial_preds_test))]
metric.update(torch.tensor(preds_max), labels_test)
fig_, ax_ = metric.plot()
plt.savefig(os.path.join("pictures_ratio", "confusion_matrix_adversarial.png"))
print("Saved confusion matrix to pictures_ratio/confusion_matrix_adversarial.png")
plt.close()



# ## increase robustness (50/50 split experiment)
with Timer("Full Adversarial Dataset Generation"):
    # Generate adversarial samples for the ENTIRE training set (200 samples)
    adv_dataset = PGD(model=qml_model, feats=feats_train, labels=labels_train, epsilon=0.1) + feats_train

# Create the 50/50 augmented dataset
feats_retrain = torch.cat((feats_train, adv_dataset))
labels_retrain = torch.cat((labels_train, labels_train))
num_train_retrain = feats_retrain.shape[0]  # Should be 400
num_batches_retrain = num_train_retrain // batch_size # Should be 20
epochs_retraining = 2

with Timer("Adversarial Retraining (50/50 Ratio)"):
    for ep in range(0, epochs_retraining):
        batch_ind = gen_batches(num_train_retrain, num_batches_retrain)
        # Use simple print_acc since it evaluates on original test set
        print(f"Epoch {ep}/{epochs_retraining} starting...")

        for it in range(num_batches_retrain):
            optimizer.zero_grad()
            feats_train_batch = feats_retrain[batch_ind[it]]
            labels_train_batch = labels_retrain[batch_ind[it]]
            outputs = [qml_model(f) for f in feats_train_batch]
            batch_loss = loss(torch.stack(outputs), labels_train_batch)
            batch_loss.backward()
            optimizer.step()

# Final evaluation after retraining
adversarial_preds_v_after = [qml_model(f) for f in perturbed_x]
adversarial_class_output_after = [torch.argmax(p) for p in adversarial_preds_v_after]
visualize_data(perturbed_x.reshape(-1, 8, 8), y_vis, adversarial_class_output_after, save_name="robustness_evaluation_after_retraining.png")

with Timer("Robustness Evaluation after Retraining"):
    adversarial_preds_test_increased = [qml_model(f) for f in perturbed_x_test]
    final_acc = accuracy(labels_test, adversarial_preds_test_increased)
    print(f"Robust Acc (50/50 Ratio): {final_acc:0.7f}")

metric = MulticlassConfusionMatrix(num_classes=4)
preds_max = [torch.argmax(adversarial_preds_test_increased[i]).detach().numpy().item() for i in range(len(adversarial_preds_test_increased))]
metric.update(torch.tensor(preds_max), labels_test)
fig_, ax_ = metric.plot()
plt.savefig(os.path.join("pictures_ratio", "confusion_matrix_robust_model.png"))
plt.close()

print(f"\n✅ Total execution time: {time.time() - total_start:.2f} seconds")
