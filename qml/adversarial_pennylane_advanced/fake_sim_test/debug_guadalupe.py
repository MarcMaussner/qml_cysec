import pennylane as qml
from qiskit_ibm_runtime import fake_provider
from qiskit_aer import AerSimulator

backend = fake_provider.FakeGuadalupeV2()
print(f"Backend: {backend.name}")
print(f"Num Qubits: {backend.num_qubits}")
print(f"Basis Gates: {backend.operation_names}")
if backend.coupling_map:
    print(f"Coupling Map: {list(backend.coupling_map.get_edges())[:10]}...")

sim = AerSimulator.from_backend(backend)
print(f"\nSimulator from backend:")
print(f"Operation names: {sim.operation_names}")

try:
    sim.set_options(coupling_map=None)
    print("Set coupling_map=None on simulator options")
except Exception as e:
    print(f"Error setting coupling_map: {e}")

# Check what PennyLane sees
dev = qml.device('qiskit.aer', wires=2, backend=sim)
print(f"\nPennyLane device created with 2 wires.")
print(f"Standard basis gates for this device: {dev.backend.operation_names}")

