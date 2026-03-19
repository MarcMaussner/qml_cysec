# Fake-Device Simulation Experiment Dashboard (`FAKE_SIM.md`)

This document tracks all experiments conducted in the `fake_sim` and `fake_sim_test` sub-projects.
It is intended as a guide for structuring the research paper on adversarial robustness of QML models on noisy quantum simulators.

---

## Legend
| Symbol | Meaning |
|:---:|:---|
| ✅ | Experiment complete — results available |
| 🔄 | Running or partially complete |
| ❌ | Not yet started (planned) |
| 📊 | Results available (plots/logs) |
| 📝 | Script exists, no output yet |

---

## Part I: Performance Benchmarking (`fake_sim_test`)

### 1.1 Proof of Concept: Multicore PennyLane on FakeLimaV2
**Status: ✅ DONE**

> **Research Question:** How do `max_parallel_threads` and `max_parallel_experiments` affect simulation throughput in PennyLane on fake backends?

**Script:** [poc_multicore.py](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim_test/poc_multicore.py)

**Analysis:** [FakeLimaV2_Multicore_Analysis.md](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim_test/FakeLimaV2_Multicore_Analysis.md)

---

### 1.2 Proof of Concept: Multicore via Qiskit Runtime
**Status: ✅ DONE**

**Script:** [poc_multicore_qiskit.py](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim_test/poc_multicore_qiskit.py)

---

### 1.3 Benchmark: Aer Simulation Methods (Guadalupe)
**Status: ✅ DONE — Plots Available**

> **Research Question:** How do `automatic`, `mps`, and `statevector` Aer methods compare in execution time as qubit count scales (4, 6, 8 qubits) on `FakeGuadalupeV2`?

**Key Finding:** At 8 qubits, `automatic` mode takes ~13.14s/circuit. Multithreading gives ~1.9× speedup. MPS is fastest for low entanglement but may not be accurate under noise.

**Script:** [benchmark_aer_guadalupe.py](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim_test/benchmark_aer_guadalupe.py)

**Plot Script:** [plot_aer_options_guadalupe.py](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim_test/plot_aer_options_guadalupe.py)

**Outputs: 📊** [pictures/](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim_test/pictures/)
- `benchmark_aer_guadalupe_plot.png` — Scaling of simulation time vs. qubits per method
- `comprehensive_comparison_pennylane.png` — Full comparison (PennyLane path)
- `comprehensive_comparison_qiskit.png` — Full comparison (Qiskit path)

---

### 1.4 Comprehensive Parallel Profiling
**Status: ✅ DONE**

> Full profiling of execution time under different thread configurations.

**Script:** [plot_comprehensive.py](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim_test/plot_comprehensive.py)

---

### 1.5 Runtime Estimation: 8-Qubit Experiment
**Status: ✅ DONE — Analysis Document**

> Quantitative scaling analysis showing ~11× total runtime increase when doubling qubits (4 → 8) accounting for circuit depth, parameter count, and parallelization speedup.

**Analysis:** [runtime_analysis_8qubits.md](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/runtime_analysis_8qubits.md)

| Factor | Value |
|:---|:---|
| 4q actual runtime (measured) | **220,705 s ≈ 2.56 days** |
| Parameter scaling (4q→8q) | 192 → 768 params (4×) |
| Simulation time scaling | ~2.6× per circuit |
| Depth scaling (layers) | 2× |
| Thread speedup (8q) | ~1.9× |
| **Total runtime scaling** | **~10.86×** |
| **8q estimated runtime** | **~27.8 days** |

---

## Part II: Noise-Aware Adversarial Robustness (`fake_sim`)

### 2.1 Baseline: PGD Attack on 4-Qubit QNN (FakeGuadalupeV2) ⭐ Golden Sample
**Status: ✅ DONE**

> ⭐ **Golden Sample:** This is the canonical reference implementation for all 4-qubit fake-device experiments. All variations (2.2–2.8) derive from this script and should be compared against it.

| Property | Value |
|:---|:---|
| Backend | `FakeGuadalupeV2` |
| Qubits | 4 |
| Layers | 16 |
| Data Re-uploading | 3× |
| Input | 8×8 grayscale (Plus-Minus dataset) |
| Training/Test Samples | 200 / 50 |
| Epochs | 4 training + 2 retraining |
| Feature Encoding | Linear (identity) |
| Attack | PGD (ε=0.1, α=0.01, 10 iters) |
| Retraining Ratio | ~10% adversarial samples |
| Benign Acc (val) | **96%** |
| Adversarial Acc (val) | **40%** |
| Post-Retraining Acc (val) | **44%** |
| Total Runtime | ~220,705 seconds (~61.3 hours) |

**Training curve:**
| Epoch | Train Acc | Val Acc |
|:---:|:---:|:---:|
| 0 | 28% | 22% |
| 1 | 62% | 58% |
| 2 | 98% | 94% |
| 3 | 92% | 100% |
| 4 | 92% | **94%** |

**Post-Retraining curve:**
| Epoch | Train Acc | Val Acc |
|:---:|:---:|:---:|
| 0 | 92% | 94% |
| 1 | 96% | 90% |
| 2 | 98% | **92%** |

> 📊 **Key Finding:** PGD attack reduces accuracy from 96% → 40%. Adversarial retraining with ~10% adv. samples provides only modest recovery (44%), suggesting the retraining ratio or epoch count may be insufficient for strong robustness.

**Script:** [PGD_Attack_Reduced_Noise_Fake_Guadalupe_4qubits.py](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/PGD_Attack_Reduced_Noise_Fake_Guadalupe_4qubits.py)

**Outputs: 📊** [pictures/](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/pictures/FakeGuadalupe_4_qubits/)
- `initial_data_visualization.png`, `benign_data_evaluation.png`, `adversarial_attack_evaluation.png`, `confusion_matrix_benign.png`, `confusion_matrix_adversarial.png`, `confusion_matrix_robust_model.png`

---

### 2.2 Backend Variation: PGD Attack on FakeLimaV2 (4 Qubits)
**Status: ✅ DONE — Full Results Available**

| Property | Value |
|:---|:---|
| Backend | `FakeLimaV2` |
| Qubits | 4 |
| Architecture | Same as 2.1 |
| Benign Acc (val) | **92%** |
| Adversarial Acc (val) | **26%** |
| Post-Retraining Acc (val) | **52%** |
| Total Runtime | ~80,711 seconds (~22.4 hours) |

**Training curve:**
| Epoch | Train Acc | Val Acc |
|:---:|:---:|:---:|
| 0 | 32% | 18% |
| 1 | 50% | 40% |
| 2 | 94% | 92% |
| 3 | 92% | 92% |
| 4 | 94% | **94%** |

**Post-Retraining curve:**
| Epoch | Train Acc | Val Acc |
|:---:|:---:|:---:|
| 0 | 90% | 96% |
| 1 | 94% | 96% |
| 2 | 98% | **100%** |

> 📊 **Key Finding:** FakeLima shows better adversarial robustness after retraining (52%) than the raw adversarial accuracy (26%). Retraining converges quickly and reaches 100% benign accuracy, indicating the model retains good representational capacity.

**Script:** [PGD_Attack_Reduced_Noise_Fake_Lima_q4.py](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/PGD_Attack_Reduced_Noise_Fake_Lima_q4.py)

**Outputs: 📊** [pictures/FakeLima_4_qubits/](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/pictures/FakeLima_4_qubits/)

---

### 2.3 Experiment: Improved Adversarial Retraining Ratio (50/50 Split)
**Status: ✅ DONE**

> **Research Question:** Does a 50/50 adversarial-to-benign retraining ratio improve post-retraining robustness compared to the ~10% ratio baseline?

| Property | Value |
|:---|:---|
| Backend | `FakeGuadalupeV2` |
| Qubits | 4 |
| Retraining Dataset | 200 benign + 200 adversarial (full training set doubled) |
| Retraining Epochs | 2 |
| Benign Acc (val) | **96%** |
| Adversarial Acc (val) | **14%** |
| Post-Retraining Acc (val) | **84%** |
| Total Runtime | ~484,548 seconds (~134.6 hours) |

**Initial Training curve:**
| Epoch | Train Acc | Val Acc |
|:---:|:---:|:---:|
| 0 | 24% | 26% |
| 1 | 60% | 64% |
| 2 | 88% | 88% |
| 3 | 88% | 88% |
| 4 | 94% | **96%** |

> 📊 **Key Finding:** Doubling the training set with a 50/50 adversarial-to-benign ratio dramatically improves robustness. Post-retraining accuracy reached **84%**, a significant jump from the **44%** achieved in the ~10% ratio baseline (§2.1).

**Script:** [PGD_Attack_Reduced_Noise_Fake_Guadalupe_4qubits_retraining_ratio.py](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/PGD_Attack_Reduced_Noise_Fake_Guadalupe_4qubits_retraining_ratio.py)

**Outputs: 📊** [pictures_ratio/](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/pictures_ratio/)
- `initial_data_visualization.png`, `benign_data_evaluation.png`, `adversarial_attack_evaluation.png`, `confusion_matrix_adversarial.png`, `robustness_evaluation_after_retraining.png`

---

### 2.4 Experiment: Non-Linear Feature Mapping (`tanh`)
**Status: 🔄 SCRIPT UPDATED — Re-Run Required**

> **Research Question:** Does applying a `tanh` activation to inputs before encoding into `StronglyEntanglingLayers` improve expressivity and change adversarial robustness behavior?

> ⚠️ **Script Updated (2026-03-02):** Script was rewritten to fully mirror the golden sample (§2.1) structure — same training loop, adversarial retraining, all confusion matrices. **Only behavioral difference:** `tanh(inputs)` applied inside the QNode before angle encoding. Previous results were from an incomplete version missing the retraining phase.

| Property | Value |
|:---|:---|
| Backend | `FakeGuadalupeV2` |
| Qubits | 4 |
| Feature Map | `tanh(x)` applied before angle encoding (inside QNode) |
| Benign Acc (val) — prev. run | **96%** |
| Adversarial Acc (val) — prev. run | **12%** → significant vulnerability |
| Retraining Acc — prev. run | Not available (script was incomplete) |
| Total Runtime — prev. run | ~41.8 hours |

> 📊 **Previous Key Finding:** Non-linear mapping achieves high benign accuracy but does *not* inherently confer adversarial robustness. The adversarial accuracy collapse (96% → 12%) is more severe than the golden sample. Full retraining comparison pending re-run.

**Script:** [PGD_Attack_Reduced_Noise_Fake_Guadalupe_4qubits_nonlinear.py](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/PGD_Attack_Reduced_Noise_Fake_Guadalupe_4qubits_nonlinear.py)

**Previous Outputs: 📊** [pictures_nonlinear/](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/pictures_nonlinear/)
- `initial_data_visualization.png`, `benign_data_evaluation.png`, `adversarial_attack_evaluation.png`, `confusion_matrix_benign.png`
- Missing (re-run needed): `confusion_matrix_adversarial.png`, `robustness_evaluation_after_retraining.png`, `confusion_matrix_robust_model.png`

**Log (prev. run):** [nohup_nonlinear.out](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/nohup_nonlinear.out)

---

### 2.5 Experiment: Learning Rate Scheduler (`StepLR`)
**Status: ✅ DONE**

> **Research Question:** Does an LR decay schedule (high LR for exploration, low LR for fine-tuning) stabilize training in the noisy quantum loss landscape?

| Property | Value |
|:---|:---|
| Backend | `FakeGuadalupeV2` |
| Qubits | 4 |
| Scheduler | `StepLR` (step decay) |
| Benign Acc (val) | **96%** |
| Adversarial Acc (val) | **14%** |
| Post-Retraining Acc (val) | **12%** |
| Total Runtime | ~305,255 seconds (~84.8 hours) |

**Training curve:**
| Epoch | LR | Train Acc | Val Acc |
|:---:|:---:|:---:|:---:|
| 0 | 0.10000 | 26% | 26% |
| 1 | 0.10000 | 66% | 60% |
| 2 | 0.10000 | 88% | 86% |
| 3 | 0.01000 | 86% | 92% |
| 4 | 0.01000 | 86% | 92% |
| 5 | 0.01000 | 94% | 98% |
| 6 | 0.00100 | 90% | 94% |
| 7 | 0.00100 | 92% | 94% |
| 8 | 0.00100 | 94% | 94% |

> 📊 **Key Finding:** The StepLR scheduler achieves excellent benign accuracy (96%) and converges smoothly. However, the model remains highly vulnerable to PGD attacks (14% acc). Interestingly, standard adversarial retraining at a low learning rate (0.001) did not improve robustness in this configuration, with accuracy actually dipping to 12%, suggesting that higher learning rates or more epochs might be needed during the defense phase when using a scheduler.

**Script:** [PGD_Attack_Reduced_Noise_Fake_Guadalupe_4qubits_scheduler.py](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/PGD_Attack_Reduced_Noise_Fake_Guadalupe_4qubits_scheduler.py)

**Log:** [nohup_scheduler.out](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/pictures_scheduler/nohup_scheduler.out)

**Outputs: 📊** [pictures_scheduler/](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/pictures_scheduler/)
- `initial_data_visualization.png`, `benign_data_evaluation.png`, `adversarial_attack_evaluation.png`, `confusion_matrix_benign.png`, `confusion_matrix_adversarial.png`, `robustness_evaluation_after_retraining.png`, `confusion_matrix_robust_model.png`

---

### 2.6 Experiment: Lipschitz Gradient Regularization
**Status: ✅ DONE**

> **Research Question:** Does penalizing large input gradients (Lipschitz regularization) mathematically bound model sensitivity and improve robustness against PGD attacks?

| Property | Value |
|:---|:---|
| Backend | `FakeGuadalupeV2` |
| Qubits | 4 |
| Layers | 16 |
| Regularization | Lipschitz (λ=0.05) |
| Benign Acc (val) | **94%** |
| Adversarial Acc (val) | **44%** |
| Post-Retraining Acc (val) | **64%** |
| Total Runtime | ~312,371 seconds (~86.8 hours) |

**Training curve:**
| Epoch | Train Acc | Val Acc |
|:---:|:---:|:---:|
| 0 | 24% | 16% |
| 1 | 52% | 68% |
| 2 | 90% | 88% |
| 3 | 84% | 88% |
| 4 | 94% | **94%** |

**Post-Retraining curve:**
| Epoch | Train Acc | Val Acc |
|:---:|:---:|:---:|
| 0 | 90% | 94% |
| 1 | 92% | 92% |
| 2 | 98% | **94%** |

> 📊 **Key Finding:** Lipschitz regularization (λ=0.05) achieves a benign accuracy of 94%. While it does not fully prevent the adversarial collapse (44% adversarial accuracy), it provides a better starting point for retraining than the golden sample (§2.1). Post-retraining robustness reached **64%**, which is a 20% improvement over the baseline (44%).

**Script:** [PGD_Attack_Reduced_Noise_Fake_Guadalupe_4qubits_lipschitz.py](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/PGD_Attack_Reduced_Noise_Fake_Guadalupe_4qubits_lipschitz.py)

**Outputs: 📊** [pictures_lipschitz/](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/pictures_lipschitz/)
- `initial_data_visualization.png`, `benign_data_evaluation.png`, `adversarial_attack_evaluation.png`, `confusion_matrix_benign.png`, `confusion_matrix_adversarial.png`, `robustness_evaluation_after_retraining.png`, `confusion_matrix_robust_model.png`

**Log:** [nohup_4qubib_lipschitz.out](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/pictures_lipschitz/nohup_4qubib_lipschitz.out)

---

### 2.7 Scale-Up: 6-Qubit QNN (FakeGuadalupeV2)
**Status: ❌ PLANNED — Script Exists**

> Scaling up to 6 qubits with adjusted hyperparameters (`num_layers=32`, `num_reup=9`).

**Script:** [PGD_Attack_Reduced_Noise_Fake_Guadalupe_6qubits.py](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/PGD_Attack_Reduced_Noise_Fake_Guadalupe_6qubits.py)

---

### 2.8 Scale-Up: 8-Qubit QNN (FakeGuadalupeV2)
**Status: ❌ PLANNED — Script Exists**

> Estimated runtime: ~27.8 days with current settings (based on measured 2.56-day 4q runtime × 10.86× scaling factor). Requires resource reduction (fewer samples or layers) to be feasible. See §1.5 for the runtime analysis.

**Script:** [PGD_Attack_Reduced_Noise_Fake_Guadalupe_8qubits.py](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/PGD_Attack_Reduced_Noise_Fake_Guadalupe_8qubits.py)

**Reference Analysis:** [runtime_analysis_8qubits.md](file:///home/mms/workspace_qmlsec/qml_cysec/qml/adversarial_pennylane_advanced/fake_sim/runtime_analysis_8qubits.md)

---

## Summary Table

| # | Experiment | Backend | Qubits | Status | Key Script |
|:---:|:---|:---:|:---:|:---:|:---|
| 1.1 | Multicore PoC (PennyLane) | FakeLimaV2 | 5 | ✅ | `poc_multicore.py` |
| 1.2 | Multicore PoC (Qiskit) | FakeLimaV2 | 5 | ✅ | `poc_multicore_qiskit.py` |
| 1.3 | Aer Method Benchmark | FakeGuadalupeV2 | 4/6/8 | ✅ | `benchmark_aer_guadalupe.py` |
| 1.4 | Comprehensive Profiling | FakeGuadalupeV2 | 4/6/8 | ✅ | `plot_comprehensive.py` |
| 1.5 | 8-Qubit Runtime Analysis | — | 8 | ✅ | `runtime_analysis_8qubits.md` |
| 2.1 | ⭐ PGD Baseline (Golden Sample) — 96%→40%→44% | FakeGuadalupeV2 | 4 | ✅ | `...4qubits.py` |
| 2.2 | Backend Variation (Lima) — 92%→26%→52% | FakeLimaV2 | 4 | ✅ | `...Lima_q4.py` |
| 2.3 | Retraining Ratio (50/50) — 96%→14%→84% | FakeGuadalupeV2 | 4 | ✅ | `...retraining_ratio.py` |
| 2.4 | Non-Linear Feature Map (`tanh`) — 96%→12% (partial) | FakeGuadalupeV2 | 4 | 🔄 | `...nonlinear.py` |
| 2.5 | LR Scheduler (`StepLR`) — 96%→14%→12% | FakeGuadalupeV2 | 4 | ✅ | `...scheduler.py` |
| 2.6 | Lipschitz Regularization — 94%→44%→64% | FakeGuadalupeV2 | 4 | ✅ | `...lipschitz.py` |
| 2.7 | Scale-Up | FakeGuadalupeV2 | 6 | ❌ | `...6qubits.py` |
| 2.8 | Scale-Up | FakeGuadalupeV2 | 8 | ❌ | `...8qubits.py` |

---

**Last Updated:** 2026-03-19
**Author:** Antigravity AI Assistant
