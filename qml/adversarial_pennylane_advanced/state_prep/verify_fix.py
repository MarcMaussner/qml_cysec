#!/usr/bin/env python
import torch
import numpy as np
import sys
import os

# Mocking the environment to use the classes from the modified script
sys.path.append('.')
from PGD_Attack_Qiskit_4qubits import QiskitQuantumFunction, QML_classifier

def test_gradients():
    print("Testing Gradients...")
    # Small parameters for quick testing
    num_layers = 2
    num_qubits = 2
    
    # We need to monkeypatch the global variables in PGD_Attack_Qiskit_4qubits or just run a subset
    # Actually, it's easier to just run a small version of the script's logic here
    
    model = QML_classifier(input_dim=4, output_dim=2, num_qubits=4, num_layers=4)
    x = torch.randn(64, requires_grad=True)
    
    print("Forward pass...")
    y = model(x)
    print(f"Output: {y}")
    
    print("Backward pass...")
    loss = torch.sum(y**2)
    loss.backward()
    
    print(f"Input gradient mean: {x.grad.abs().mean():.6f}")
    print(f"Weights gradient mean: {model.weights.grad.abs().mean():.6f}")
    print(f"Bias gradient mean: {model.bias.grad.abs().mean():.6f}")
    
    if x.grad.abs().sum() > 0:
        print("✅ Input gradients are non-zero!")
    else:
        print("❌ Input gradients are zero!")

if __name__ == "__main__":
    test_gradients()
