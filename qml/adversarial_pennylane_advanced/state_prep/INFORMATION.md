# State Preparation and Data Encoding Information
*Extracted from arXiv:2309.09424 ("Drastic Circuit Depth Reductions with Preserved Adversarial Robustness by Approximate Encoding for Quantum Machine Learning")*

## 1. Core Encoding Techniques
The paper focuses on **Amplitude Encoding** for mapping classical grayscale images into quantum states for Machine Learning.

- **Formula**: A vector $x \in \mathbb{R}^N$ is mapped to $|\psi(x)\rangle = \frac{1}{\|x\|^2} \sum_{j=0}^{2^n - 1} (x_{2j} + i x_{2j+1}) |j\rangle$.
- **Efficiency**: Highly qubit-efficient ($n = \lceil \log_2(N) \rceil - 1$). For example, a $28 \times 28$ image (784 features) can be encoded into 9 qubits.
- **Complexity**: Exact preparation requires $O(2^n)$ gates, often resulting in very deep circuits (e.g., ~512 CNOTs for 9 qubits).

## 2. State Preparation Methods

### Exact Method
- **SBM (State-Base Method)**: The standard algorithm (e.g., `initialize` in Qiskit) that achieves 100% fidelity.
- **Drawback**: Extremely deep circuits that are highly susceptible to noise on NISQ hardware.

### Approximate State Preparation (ASP)
The paper proposes and evaluates methods that provide ~60-70% fidelity but with **two orders of magnitude fewer gates**.

| Method | Description | Characteristics |
| :--- | :--- | :--- |
| **MPS (Matrix Product State)** | Represents states as a tensor network (Matrix Product State) with a fixed entanglement rank. | Deterministic, highly memory-efficient, but fidelity is limited by the entanglement rank $2^{k-1}$. |
| **GASP (Genetic Algorithm)** | Uses a genetic algorithm to evolve both the circuit structure and Gate parameters. | Heuristic, can find extremely compact circuits (often < 50 CNOTs) that are noise-resistant. |
| **Variational Prep** | Optimizes a layered variational circuit $V(\theta)$ to match the target state. | Layers can be added incrementally; requires a classical optimization loop (Adam/SLSQP). |

## 3. Key Findings on Adversarial Robustness
- **Noise Resilience**: Preparing states at lower fidelity acts as a form of "pseudo-random noise" that can drown out adversarial perturbations.
- **Performance**: Approximate encoding at ~60% fidelity maintains high classification accuracy on noise-free data while significantly **improving robustness** against adversarial attacks compared to exact encoding.
- **Hardware Viability**: ASP circuits are short enough to run reliably on near-term hardware (e.g., *ibm_algiers*), whereas exact SBM circuits often fail due to gate errors.

---
**Reference**: [arXiv:2309.09424 [quant-ph]](https://arxiv.org/abs/2309.09424)
