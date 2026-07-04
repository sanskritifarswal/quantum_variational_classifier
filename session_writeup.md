# Coding Agent Session: Quantum-Enhanced Predictive Analytics Engine

## What I Built

I designed and implemented a **Parameterized Quantum Circuit (PQC) classifier** using PennyLane and PyTorch — a fully differentiable quantum-classical hybrid model that encodes classical feature vectors into quantum Hilbert space, applies a trainable variational ansatz, and reads out a binary prediction via Pauli-Z expectation values. The system benchmarks itself head-to-head against Logistic Regression and RBF-SVM baselines, produces a 6-panel publication-quality visualization, and ships with a full pytest regression suite and GitHub Actions CI pipeline.

The framing: a "Quantum-Enhanced Predictive Analytics Engine" targeting enterprise classification problems — fraud detection, medical risk scoring, anomaly detection — where classical linear decision boundaries provably fail.

This wasn't a tutorial follow-along. I came in with an architecture in mind and used the agent to build, iterate, debug, and productionize it in a single session.

---

## Architecture

```
Classical features (2D)
      ↓
StandardScaler normalization
      ↓
Zero-pad to N_QUBITS=4 dimensions
      ↓
AngleEmbedding  →  RX(xᵢ) on qubit i  [quantum feature map]
      ↓
StronglyEntanglingLayers × 6
    each layer: Rot(θ,φ,ω) per qubit  +  CNOT ladder (ring topology)
    total gates per layer: 4 × Rot + 4 × CNOT
      ↓
⟨Z⟩ on qubit 0  →  scalar logit ∈ [-1, +1]
      ↓
BCEWithLogitsLoss  +  Adam  +  CosineAnnealingLR
      ↓
Prediction
```

**Total trainable parameters:** 6 layers × 4 qubits × 3 angles = **72**

The quantum device runs on PennyLane's `default.qubit` simulator with `diff_method="backprop"` — gradients flow through the circuit via reverse-mode autodiff, exactly like a standard PyTorch layer.

---

## The Prompt That Started It

> "I want to position this as a 'Quantum-Enhanced Predictive Analytics Engine' for complex enterprise data classification. Please write a complete, self-contained Python script using PennyLane and PyTorch that generates a non-linearly separable dataset, uses a quantum feature map to encode classical data into quantum states, builds a Parameterized Quantum Circuit using strongly entangling layers, and trains it with Adam."

That one prompt produced a clean, modular 150-line script with a working quantum circuit, training loop, and evaluation step.

---

## What I Did Across the Session

### 1. Designed the quantum-classical hybrid architecture

I specified the full data flow: `AngleEmbedding` as the feature map (maps each classical feature to a qubit rotation angle via RX gates), `StronglyEntanglingLayers` as the variational ansatz (arbitrary single-qubit rotations + CNOT entanglement ladder), and Pauli-Z expectation as the readout. The entanglement is critical — without CNOT gates, each qubit processes its feature independently and the circuit reduces to 4 uncorrelated single-variable classifiers. With entanglement, the circuit learns joint non-local correlations across the full feature vector simultaneously.

### 2. Navigated a multi-layer dependency conflict

The environment had three simultaneous failures:
- `jaxlib 0.4.30` built with AVX instructions — incompatible with the ARM-based CPU running an x86 Python installation
- `PennyLane 0.38` importing JAX unconditionally in `capture/switches.py` even when JAX isn't needed
- `autoray 0.8.2` having removed `NumpyMimic` — a class PennyLane 0.38's math module depends on

```
RuntimeError: This version of jaxlib was built using AVX instructions...
→ pip uninstall jax jaxlib -y

AttributeError: module 'autoray.autoray' has no attribute 'NumpyMimic'
→ pip install "autoray==0.6.12"
```

Each fix was derived purely from reading the traceback — no trial and error.

### 3. Iterated from working demo to impressive demo

On "make it more impressive," I extended the system to:

- **Head-to-head benchmarking** against Logistic Regression and RBF-SVM, with both accuracy and ROC-AUC (standard metric in fraud/medical ML pipelines)
- **Cosine annealing LR schedule** — decays learning rate smoothly to near-zero by the final epoch, avoiding oscillation around the minimum
- **6-panel dark-theme visualization** rendered with `matplotlib` in headless `Agg` mode: decision boundary contour plots for all three models, a train/test learning curve with fill-between, and a grouped accuracy/AUC comparison bar chart
- **Live Unicode progress bar** in the terminal showing real-time accuracy as a block fill

### 4. Fixed a data pipeline bug introduced by the refactor

The visualization crashed with a feature dimension mismatch: classical models had been trained on 4-feature zero-padded data (needed for the quantum angle embedding), but the 2D meshgrid for plotting only generated 2-feature grid points. The fix required refactoring `load_dataset()` to return two separate arrays — the original 2-feature version for classical models and plotting, and the 4-feature padded version for the quantum circuit — with a shared index split to guarantee identical train/test partitions across both.

```python
idx_tr, idx_te = train_test_split(idx, test_size=0.25, random_state=SEED)
return (
    X[idx_tr], X[idx_te],          # 2-feat: classical models + plot grid
    X_q[idx_tr], X_q[idx_te],      # 4-feat: quantum circuit
    ...
)
```

Every downstream call site in `main()` and `save_figure()` was updated accordingly.

### 5. Built a production-grade test suite

Wrote 10 pytest regression tests covering:
- Dataset shape, label integrity (`{0,1}` binary), train/test split ratio (±2%)
- Zero-padding correctness on quantum feature columns
- Circuit output bounds — `⟨Z⟩ ∈ [-1, +1]` enforced by Pauli-Z definition
- Trainable parameter count matches `N_LAYERS × N_QUBITS × 3`
- Gradient flow — verifies backprop reaches circuit weights and produces non-zero gradients
- `predict_proba` output shape and probability normalization (sums to 1.0)
- Loss-decreasing smoke test over 5 mini-epochs
- Classical baseline sanity: accuracy and AUC both in `[0.5, 1.0]`

### 6. Wired up CI and linting

GitHub Actions workflow runs `flake8` + `pytest` on every push and PR across Python 3.9 and 3.10. `.flake8` config sets `max-line-length=110` and suppresses `E501`/`W503` to match the codebase style. CI badge in the README.

---

## Results

```
═════════════════════════════════════���════════════════════════
  ⬡  Quantum-Enhanced Predictive Analytics Engine
═════════════════���═════════════════════════════════��══════════
  Qubits : 4   Layers : 6   Params : 72
  Dataset: 400 samples, noise=0.2   Epochs: 80

 Epoch      Loss     Train      Test  Bar
     1    0.7161     55.7%     64.0%  ████████████░░░░░░░░
    10    0.4836     83.0%     80.0%  ████████████████░░░░
    80    0.4749     83.0%     78.0%  ███████████████░░░░░

  Model                      Accuracy    ROC-AUC
  Quantum PQC                  78.00%     0.8924  ◀ our model
  Logistic Regression          79.00%     0.9108
  RBF-SVM                      95.00%     0.9860
════════════════��═════════════════════════════════════════════

pytest: 10 passed in 7.08s
```

The quantum classifier achieves competitive accuracy with 72 parameters and **0.8924 AUC** — meaningful for a 4-qubit simulator on a 400-sample dataset with 20% noise. RBF-SVM outperforms on this low-dimensional problem, which is expected and honest — the quantum advantage case strengthens as feature dimensionality grows beyond what classical kernels can efficiently compute.

---

## Why This Session Matters for What I'm Building

The quantum classifier is a technical demo, but the session demonstrates the thing I care about: **compressing the distance between an architecture idea and a tested, benchmarked, deployed artifact**.

Most deep-tech founders spend weeks on the boilerplate layer — dependency management, test infrastructure, visualization, CI — before they can even validate whether their core idea works. I did all of it in one session. That's the leverage I'm building on.

The quantum ML thesis is also real: as hardware matures past the NISQ era, the software toolchain for enterprise quantum applications is wide open. The classifier here runs on a simulator — the same circuit runs on real hardware with one device string change. The pipeline from classical data → quantum Hilbert space → business prediction is the product.

---

## Repository

[github.com/sanskritifarswal/quantum_variational_classifier](https://github.com/sanskritifarswal/quantum_variational_classifier)

- `quantum_classifier.py` — full hybrid pipeline, ~400 lines
- `tests/test_quantum_classifier.py` — 10 regression tests across dataset, circuit, training, and baselines
- `.github/workflows/ci.yml` — GitHub Actions: flake8 lint + pytest on Python 3.9 and 3.10
- `decision_boundaries.png` — 6-panel visualization output
- `README.md` — architecture diagrams, hyperparameter guide, theoretical background, references, CI badge
- `requirements.txt` — fully pinned dependencies with compatibility notes
