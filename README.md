# qml_cysec
cybersecurity aspects in quantum machine learning

## List of cybersecurity attacks for experimentsc
- Crosstalk:
  - create proof of concept with noisy backend -> try to flip one possible qubit
    in accordance with: Abdullah Ash-Saki, Mahabubul Alam, and Swaroop Ghosh. 2020. Analysis of crosstalk in NISQ devices and security implications in multi-programming regime. In Proceedings of the ACM/IEEE International Symposium on Low Power Electronics and Design (ISLPED '20). Association for Computing Machinery, New York, NY, USA, 25–30. https://doi.org/10.1145/3370748.3406570
  - create proof of concept qml -> learn logic gates?
  - create real world qml (MNIST, CISAF10),...
- Transpilation:
  - reuse qiskit transpiler passes to insert X-Gates on purpose of circuit transpilation
- One-Pixel-Attack:
  - create proof of concept based on paper: One pixel attack for fooling deep neural networks (Jiawei Su, Danilo Vasconcellos Vargas, Sakurai Kouichi)
  - try to get into the math behind to construct one-pixel image
- Adversarial attack and increase robustness:
  - https://pennylane.ai/qml/demos/tutorial_adversarial_attacks_QML
- Pennylane-Hw:
  - Simple proof of concept for using noisy simulator and ibm quantum hardware in pennylane

## Conferences:
- Embedded World 2025
- GI Workshop Quantum 2025
- QTML 2025
- QCE 2025
- ECCSW 2025

## Timeline:
- 18.10.2024: Kickoff
  - Gather list of possible attacks
  - Gather list of conferences
- 29.11.2024: FollowUp
  - Think about list extension
  - Present current state of proof of concepts (noisy backend, transpilation)
- 10.01.2025: FollowUp
  - Prepare and present QML exmples and store them in QML folder
    - Sinus learn
    - logic gate learn
    - One-Pixel-Attack:
      - Classification of pictures black/white (AmplitudeEncoding)
      - Try missprediction by changing only one pixel in one picture
    - Adversarial:
      - PGD, FGSM
- 07.03.2025: FollowUp
  - Thoughts:
    1.1) Run Experiments on HW and see with noise pertubations (Adversarial -> PQD, FGSM)
       * Pennylane -> HW?!?
         - default.mixed -> simulator on pennylane with noise
         - https://docs.pennylane.ai/projects/qiskit/en/latest/
    1.2) Run Experiments on Crosstalk on IQM
    3.1) Try attack on kernel
    3.2) Deep dive into math
    2) Transpilation and sine with qiskit -> results always in 1
- 04.04.2025: FollowUp
  - Planning of paper for GI
    - Outline of paper and preparation of template -> Marc
    - Bibliography -> Volker + Marc
    - try sine (mixed, qiskit.remote) -> Marc
    - try "qiskit.local" -> noisy fake device -> Marc
    - try pennylane-hw -> Volker
    - try PGD with mixed-device -> Volker
