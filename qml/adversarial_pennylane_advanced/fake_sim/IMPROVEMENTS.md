# Strategies for Increasing QML Accuracy and Robustness

This document summarizes potential technical approaches to improve the classification accuracy and adversarial robustness of 4-qubit and 8-qubit QNN models running on Qiskit fake-device simulators.

## 1. Optimize Adversarial Retraining Ratio
Currently, the adversarial retraining phase uses a small number of perturbed samples relative to benign samples.
- **Approach:** Transition to a **50/50 split** or train exclusively on adversarial data for the final epochs.
- **Expected Impact:** Forces the model to prioritize noise-invariant features over benign accuracy.

## 2. Increase Model Capacity
Deeper models can better approximate complex decision boundaries in noisy Hilbert spaces.
- **Approach:** Increase **`num_layers`** (e.g., to 32) and **`num_reup`** (Data Re-uploading factor).
- **Constraint:** Ensure `input_dim * num_reup` matches total weight dimensions (`num_layers * num_qubits * 3`).

## 3. Data Augmentation with Hardware-Aware Noise
Static noise models in fake backends can be "learned" by the classical optimizer.
- **Approach:** Inject random Gaussian or salt-and-pepper noise into training images *before* quantum encoding.
- **Expected Impact:** Acts as a "smoothing" regularizer for the decision boundary.

## 4. Learning Rate Scheduling
Fixed learning rates ($0.1$) may lead to oscillations in the noisy loss landscape.
- **Approach:** Implement a **Learning Rate Scheduler** (e.g., `StepLR`).
- **Strategy:** Start at $0.1$ for exploration and decay to $0.01$ or $0.001$ for fine-tuning.

## 5. Non-Linear Feature Mapping
Linear scaling of inputs into rotation gates might lack expressivity.
- **Approach:** Apply non-linear activations like `tanh()` or `sin()` to inputs before encoding them into `StronglyEntanglingLayers`.
- **Expected Impact:** Improves the representation power of the quantum-classical interface.

## 6. Gradient-Based Regularization (Lipschitz)
High sensitivity to input changes is the root cause of adversarial vulnerability.
- **Approach:** Add a regularization term to the loss function that penalizes large gradients with respect to the inputs.
- **Expected Impact:** Mathematically bounds the model's sensitivity to small perturbations.
