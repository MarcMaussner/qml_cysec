#!/usr/bin/env python
# coding: utf-8

import os
import time
import argparse
import skimage.transform
import torch
import pennylane as qml
from pennylane import numpy as np
from matplotlib import pyplot as plt
from qiskit_ibm_runtime.fake_provider import FakeJakartaV2
from torchmetrics.classification import MulticlassConfusionMatrix

# Total execution timer
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

class DataLoader:
    def __init__(self, resize_dim=(8, 8)):
        self.resize_dim = resize_dim
        self.labels_list = ["\u2212", "\u002b", "\ua714", "\u02e7"]

    def load_and_preprocess(self):
        print("Loading dataset...")
        [pm] = qml.data.load('other', name='plus-minus')
        
        X_train = self._preprocess(pm.img_train)
        X_test = self._preprocess(pm.img_test)
        
        return X_train, pm.labels_train, X_test, pm.labels_test

    def _preprocess(self, data):
        # Resize images: (N, H, W) -> (N, h, w)
        # Using the same logic as original: transpose, resize, transpose back
        data_transposed = np.transpose(data, (1, 2, 0))
        data_resized = skimage.transform.resize(data_transposed, self.resize_dim)
        return np.transpose(data_resized, (2, 0, 1))

    def visualize(self, x, y, pred=None, save_name=None, output_dir="pictures_refined"):
        os.makedirs(output_dir, exist_ok=True)
        n_img = len(x)
        fig, axes = plt.subplots(1, n_img, figsize=(2 * n_img, 2))
        if n_img == 1:
            axes = [axes]
        
        for i in range(n_img):
            img = x[i].reshape(self.resize_dim) if isinstance(x[i], (torch.Tensor, np.ndarray)) else x[i]
            axes[i].imshow(img, cmap="gray")
            title = f"Label: {self.labels_list[y[i]]}"
            if pred is not None:
                title += f", Pred: {self.labels_list[pred[i]]}"
            axes[i].set_title(title)
            axes[i].axis('off')
            
        plt.tight_layout(w_pad=2)
        if save_name:
            path = os.path.join(output_dir, save_name)
            plt.savefig(path)
            print(f"Saved visualization to {path}")
        plt.close()

class QMLClassifier(torch.nn.Module):
    def __init__(self, num_qubits=4, num_layers=16, num_classes=4, num_reup=3, shots=1024):
        super().__init__()
        torch.manual_seed(1337)
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_reup = num_reup
        
        # Setup Fake Backend
        self.fake_backend = FakeJakartaV2()
        self.fake_backend.set_options(
            max_parallel_threads=4, 
            max_parallel_experiments=1,
            method="automatic"
        )
        
        self.q_device = qml.device('qiskit.remote', wires=self.num_qubits, backend=self.fake_backend, shots=shots)
        self.weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=self.num_layers, n_wires=self.num_qubits)

        @qml.qnode(self.q_device, interface="torch")
        def circuit(inputs, weights, bias):
            inputs = torch.reshape(inputs, self.weights_shape)
            qml.StronglyEntanglingLayers(
                weights=weights * inputs + bias, wires=range(self.num_qubits)
            )
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_classes)]

        param_shapes = {"weights": self.weights_shape, "bias": self.weights_shape}
        init_vals = {
            "weights": 0.1 * torch.rand(self.weights_shape),
            "bias": 0.1 * torch.rand(self.weights_shape),
        }

        self.qcircuit = qml.qnn.TorchLayer(
            qnode=circuit, weight_shapes=param_shapes, init_method=init_vals
        )

    def forward(self, x):
        inputs_stack = torch.hstack([x] * self.num_reup)
        return self.qcircuit(inputs_stack)

class PGDManager:
    @staticmethod
    def generate_attack(model, loss_fn, feats, labels, epsilon=0.1, alpha=0.01, num_iter=10):
        delta = torch.zeros_like(feats, requires_grad=True)
        for _ in range(num_iter):
            outputs = [model(f) for f in feats + delta]
            l = loss_fn(torch.stack(outputs), labels)
            l.backward()
            
            delta.data = (delta + alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
        return delta.detach()

class ExperimentRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.data_loader = DataLoader()
        self.model = QMLClassifier(
            num_qubits=args.num_qubits,
            num_layers=args.num_layers,
            num_classes=args.num_classes,
            num_reup=args.num_reup,
            shots=args.shots
        ).to(self.device)
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

    def run(self):
        X_train, Y_train, X_test, Y_test = self.data_loader.load_and_preprocess()
        
        # Select subsets
        train_feats = torch.from_numpy(X_train[:self.args.train_samples]).reshape(self.args.train_samples, -1).to(self.device)
        test_feats = torch.from_numpy(X_test[:self.args.test_samples]).reshape(self.args.test_samples, -1).to(self.device)
        train_labels = torch.from_numpy(Y_train[:self.args.train_samples]).to(self.device).long()
        test_labels = torch.from_numpy(Y_test[:self.args.test_samples]).to(self.device).long()

        # Initial training
        with Timer("Initial Training"):
            self._train(train_feats, train_labels, test_feats, test_labels, epochs=self.args.epochs)

        # Benign Evaluation
        self._evaluate(test_feats, test_labels, "benign")

        # Attack
        with Timer("PGD Attack Generation"):
            perturbation = PGDManager.generate_attack(
                self.model, self.loss_fn, test_feats, test_labels, 
                epsilon=self.args.epsilon, alpha=self.args.alpha, num_iter=self.args.attack_iter
            )
            perturbed_test_feats = test_feats + perturbation

        # Adversarial Evaluation
        self._evaluate(perturbed_test_feats, test_labels, "adversarial")

        # Adversarial Retraining
        if self.args.retrain_epochs > 0:
            with Timer("Adversarial Retraining"):
                adv_samples_count = min(len(train_feats), 20)
                adv_perturbation = PGDManager.generate_attack(
                    self.model, self.loss_fn, train_feats[:adv_samples_count], train_labels[:adv_samples_count],
                    epsilon=self.args.epsilon
                )
                adv_feats = train_feats[:adv_samples_count] + adv_perturbation
                
                retrain_feats = torch.cat((train_feats, adv_feats))
                retrain_labels = torch.cat((train_labels, train_labels[:adv_samples_count]))
                
                self._train(retrain_feats, retrain_labels, test_feats, test_labels, epochs=self.args.retrain_epochs)
                
            # Robustness Evaluation
            self._evaluate(perturbed_test_feats, test_labels, "robust")

    def _train(self, train_feats, train_labels, val_feats, val_labels, epochs):
        num_samples = train_feats.shape[0]
        num_batches = num_samples // self.args.batch_size
        
        for ep in range(epochs):
            perm = torch.randperm(num_samples)
            total_loss = 0
            
            for i in range(num_batches):
                indices = perm[i * self.args.batch_size : (i + 1) * self.args.batch_size]
                batch_feats = train_feats[indices]
                batch_labels = train_labels[indices]
                
                self.optimizer.zero_grad()
                outputs = [self.model(f) for f in batch_feats]
                loss = self.loss_fn(torch.stack(outputs), batch_labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            # Print status
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            val_acc = self._get_accuracy(val_feats, val_labels)
            print(f"Epoch {ep+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    def _get_accuracy(self, feats, labels):
        self.model.eval()
        with torch.no_grad():
            preds = [self.model(f) for f in feats]
            correct = sum(torch.argmax(p) == l for p, l in zip(preds, labels))
            acc = float(correct) / len(labels)
        self.model.train()
        return acc

    def _evaluate(self, feats, labels, mode):
        print(f"Evaluating {mode} performance...")
        self.model.eval()
        with torch.no_grad():
            preds = [self.model(f) for f in feats]
            pred_classes = torch.tensor([torch.argmax(p) for p in preds])
            acc = float(sum(pred_classes == labels.cpu())) / len(labels)
            print(f"{mode.capitalize()} Accuracy: {acc:.4f}")

            # Confusion Matrix
            metric = MulticlassConfusionMatrix(num_classes=self.args.num_classes)
            metric.update(pred_classes, labels.cpu())
            fig, ax = metric.plot()
            output_dir = "pictures_refined"
            os.makedirs(output_dir, exist_ok=True)
            path = os.path.join(output_dir, f"confusion_matrix_{mode}.png")
            plt.savefig(path)
            plt.close()
            print(f"Saved confusion matrix to {path}")

            # Visualization
            vis_count = min(len(feats), 4)
            self.data_loader.visualize(
                feats[:vis_count].cpu().numpy(), 
                labels[:vis_count].cpu().numpy(), 
                pred_classes[:vis_count].numpy(), 
                save_name=f"evaluation_{mode}.png"
            )
        self.model.train()

def main():
    parser = argparse.ArgumentParser(description="PGD Attack on Quantum Machine Learning Model")
    parser.add_argument("--train_samples", type=int, default=200, help="Number of training samples")
    parser.add_argument("--test_samples", type=int, default=50, help="Number of test samples")
    parser.add_argument("--num_qubits", type=int, default=4, help="Number of qubits")
    parser.add_argument("--num_layers", type=int, default=16, help="Number of layers in circuit")
    parser.add_argument("--num_classes", type=int, default=4, help="Number of classes")
    parser.add_argument("--num_reup", type=int, default=3, help="Number of data re-uploading repetitions")
    parser.add_argument("--shots", type=int, default=1024, help="Number of shots for quantum device")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size")
    parser.add_argument("--epsilon", type=float, default=0.1, help="PGD epsilon")
    parser.add_argument("--alpha", type=float, default=0.01, help="PGD alpha")
    parser.add_argument("--attack_iter", type=int, default=10, help="Number of PGD iterations")
    parser.add_argument("--retrain_epochs", type=int, default=2, help="Number of adversarial retraining epochs")
    
    args = parser.parse_args()
    
    total_start = time.time()
    runner = ExperimentRunner(args)
    runner.run()
    print(f"\n✅ Total execution time: {time.time() - total_start:.2f} seconds")

if __name__ == "__main__":
    main()
