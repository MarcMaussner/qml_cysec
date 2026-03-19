import torch
import numpy as np
from PGD_Attack_Qiskit_ASP_8qubits import QML_classifier, input_dim, num_classes, num_qubits, num_layers

def smoke_test():
    print("Starting 8-qubit ASP Smoke Test...")
    model = QML_classifier(input_dim, num_classes, num_qubits, num_layers)
    
    # Single sample
    x = torch.randn(1, input_dim)
    
    print("Testing forward pass (includes ASP fitting)...")
    try:
        y = model(x[0], idx=0)
        print(f"Forward pass success! Output shape: {y.shape}")
        
        print("Testing backward pass...")
        target = torch.tensor([0], dtype=torch.long)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(y.unsqueeze(0), target)
        loss.backward()
        
        # Check if gradients exist for weights
        grad_sum = model.weights.grad.abs().sum().item()
        print(f"Backward pass success! Gradient magnitude: {grad_sum:.6f}")
        
        if grad_sum > 0:
            print("✅ Smoke test PASSED!")
        else:
            print("❌ Smoke test FAILED: No gradients found.")
            
    except Exception as e:
        print(f"❌ Smoke test FAILED with error: {e}")

if __name__ == "__main__":
    smoke_test()
