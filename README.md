# Quantum-Enhanced Predictive Analytics Engine

[![CI](https://github.com/sanskritifarswal/quantum_variational_classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/sanskritifarswal/quantum_variational_classifier/actions/workflows/ci.yml)

A **Parameterized Quantum Circuit (PQC)** classifier built with PennyLane and PyTorch, benchmarked head-to-head against classical ML baselines. Built as a technical demo for applying quantum machine learning to complex enterprise classification tasks — fraud detection, medical risk scoring, anomaly detection.

![Decision Boundaries](decision_boundaries.png)

---

## The Pitch

Classical ML models — even powerful ones like SVMs — operate in fixed, Euclidean feature spaces. A quantum classifier maps data into an **exponentially large Hilbert space** via a quantum feature map, then learns decision boundaries that are simply unreachable by any classical linear or kernel method.

This project demonstrates that pipeline end-to-end:

```
Classical features
      ↓
AngleEmbedding  →  encodes each feature as a qubit rotation angle (RX gate)
      ↓
StronglyEntanglingLayers  →  trainable rotations + CNOT entanglement across all qubits
      ↓
Pauli-Z measurement  →  collapses quantum state back to a scalar prediction
      ↓
Adam optimizer  →  backpropagates through the circuit via parameter-shift rule
```

The circuit is fully differentiable — gradients flow from the classical loss function through the quantum gates, exactly like training a neural network layer.

---

## Results

On a noisy, non-linearly separable dataset (make_moons, noise=0.20):

| Model               | Accuracy | ROC-AUC |
|---------------------|----------|---------|
| **Quantum PQC**     | ~88%     | ~0.93   |
| RBF-SVM             | ~95%     | ~0.99   |
| Logistic Regression | ~79%     | ~0.91   |

> The quantum classifier achieves competitive accuracy with **72 trainable parameters** — far fewer than a comparable neural network — while operating on a fundamentally different computational substrate.

---

## Architecture

```
Quantum Circuit (4 qubits, 6 layers)
─────────────────────────────────────────────────
q0: ─╭AngleEmbedding─╭StronglyEntanglingLayers─┤ <Z>
q1: ─├AngleEmbedding─├StronglyEntanglingLayers─┤
q2: ─├AngleEmbedding─├StronglyEntanglingLayers─┤
q3: ─╰AngleEmbedding─╰StronglyEntanglingLayers─┤
─────────────────────────────────────────────────

Per layer: Rot(θ,φ,ω) on each qubit  →  CNOT ladder (q0→q1, q1→q2, q2→q3, q3→q0)
Total gates per layer: 4 × Rot + 4 × CNOT = 8 gates
Total trainable params: 6 layers × 4 qubits × 3 angles = 72
```

### Why entanglement matters

Without CNOT gates, each qubit processes its feature independently — the circuit is no better than 4 separate single-variable classifiers. The CNOT ladder creates **entanglement**: the state of every qubit becomes correlated with every other, allowing the circuit to learn joint, non-local patterns across all features simultaneously. This is the quantum analogue of a fully-connected layer.

---

## Quickstart

### Requirements

- Python 3.9+
- Anaconda or a virtual environment recommended

### Install

```bash
git clone https://github.com/sanskritifarswal/quantum_variational_classifier.git
cd quantum-variational-classifier

pip install pennylane torch scikit-learn numpy matplotlib
pip install "autoray==0.6.12"   # pin for PennyLane 0.38 compatibility
```

> **Apple Silicon / ARM Mac note:** If you have `jax`/`jaxlib` installed from another package, uninstall it before running: `pip uninstall jax jaxlib -y`. The script uses PyTorch exclusively and does not require JAX.

### Run

```bash
python quantum_classifier.py
```

### Run tests

```bash
pip install pytest
pytest tests/ -v
```

The test suite covers dataset shape/label integrity, circuit output bounds, gradient flow, parameter count, predict_proba correctness, a loss-decreasing smoke test, and classical baseline sanity checks — all without running a full training job.

Training takes ~5–8 minutes on CPU (the quantum simulator runs every gate numerically). The script outputs:

- Live training progress with accuracy bars in the terminal
- A comparison table: Quantum PQC vs. Logistic Regression vs. RBF-SVM
- A classification report with precision/recall
- `decision_boundaries.png` — a 6-panel visualization ready for slides

---

## Code Structure

```
quantum_classifier.py
├── CONFIG                      # all hyperparameters in one place
├── load_dataset()              # make_moons → StandardScaler → train/test split
│                               # returns both 2-feat (classical) and 4-feat (quantum) arrays
├── quantum_circuit()           # @qml.qnode — AngleEmbedding + StronglyEntanglingLayers
├── QuantumClassifier           # nn.Module wrapping the QNode as a PyTorch layer
│   ├── forward()               # runs circuit sample-by-sample, returns logits
│   └── predict_proba()         # sklearn-compatible interface for plotting
├── train()                     # Adam + CosineAnnealingLR training loop
├── run_classical_baselines()   # LogisticRegression + RBF-SVM on 2-feat data
├── save_figure()               # 6-panel matplotlib figure → decision_boundaries.png
│   ├── plot_boundary()         # decision surface + scatter for one model
│   ├── plot_learning_curve()   # train vs. test accuracy over epochs
│   └── plot_comparison_bar()   # grouped bar: accuracy + AUC across models
└── main()                      # orchestrates the full pipeline
```

---

## Hyperparameter Guide

| Parameter    | Default | Effect |
|-------------|---------|--------|
| `N_QUBITS`  | 4       | Feature dimensions in Hilbert space. More qubits = richer feature map, exponentially larger state space. |
| `N_LAYERS`  | 6       | Ansatz depth. More layers = more expressive circuit, but harder to train (barren plateau risk). |
| `N_SAMPLES` | 400     | Dataset size. Increase for more robust evaluation. |
| `NOISE`     | 0.20    | Dataset difficulty. Higher = harder, more overlap between classes. |
| `EPOCHS`    | 80      | Training duration. Typically converges by epoch 15–20. |
| `LR`        | 0.008   | Adam learning rate. Cosine annealing decays this to near-zero by the final epoch. |

---

## Background: Quantum Advantage for Classification

The theoretical case for quantum classifiers rests on three properties:

1. **Exponential state space** — an n-qubit system lives in a 2ⁿ-dimensional Hilbert space. With 4 qubits, the circuit explores a 16-dimensional space from 4 input features.

2. **Quantum kernel functions** — the inner product `⟨ψ(x)|ψ(x')⟩` between two quantum-encoded states defines a kernel that may be exponentially hard to compute classically (Havlíček et al., 2019).

3. **Entanglement as feature interaction** — CNOT gates create correlations between all features at once, without explicit feature engineering.

> **Honest caveat:** On small, low-dimensional datasets, classical SVMs remain competitive or superior. The quantum advantage is expected to emerge on high-dimensional data where the quantum feature map explores a kernel space that no classical method can efficiently simulate.

---

## References

- Havlíček et al. (2019). [Supervised learning with quantum-enhanced feature spaces](https://www.nature.com/articles/s41586-019-0980-2). *Nature*, 567, 209–212.
- Schuld & Petruccione (2021). *Machine Learning with Quantum Computers*. Springer.
- [PennyLane Documentation](https://docs.pennylane.ai) — Xanadu's quantum ML framework.
- Cerezo et al. (2021). [Variational quantum algorithms](https://www.nature.com/articles/s42254-021-00348-9). *Nature Reviews Physics*.

---

## License

MIT — see [LICENSE](LICENSE).
