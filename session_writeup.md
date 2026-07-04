# Coding Agent Session: Quantum-Enhanced Predictive Analytics Engine

## What I Built

I used Claude as a coding agent to build a **Parameterized Quantum Circuit (PQC) classifier** from scratch — a machine learning model that runs on a quantum computing simulator and benchmarks itself against classical ML baselines. The framing: a "Quantum-Enhanced Predictive Analytics Engine" for enterprise classification problems like fraud detection and medical risk scoring.

This wasn't a tutorial follow-along. I came in with a concept and used the agent to turn it into working, documented, GitHub-ready code in a single session.

---

## The Prompt That Started It

> "I want to position this as a 'Quantum-Enhanced Predictive Analytics Engine' for complex enterprise data classification. Please write a complete, self-contained Python script using PennyLane and PyTorch that generates a non-linearly separable dataset, uses a quantum feature map to encode classical data into quantum states, builds a Parameterized Quantum Circuit using strongly entangling layers, and trains it with Adam."

That one prompt produced a clean, modular 150-line script with a working quantum circuit, training loop, and evaluation step.

---

## What the Agent Did That Impressed Me

### 1. It understood the demo context, not just the code

I didn't just ask for a classifier — I told the agent this was for a startup pitch. It responded accordingly: the code was structured so an investor could follow the data flow from classical to quantum and back, the comments explained *why* each piece exists (not just what it does), and it suggested which output numbers to highlight in a pitch ("cite this as parameter efficiency").

### 2. It debugged a gnarly dependency chain without hand-holding

The environment had a broken `jaxlib` build (AVX instruction mismatch on ARM hardware), a PennyLane/autoray version conflict, and a Python 3.9 compatibility constraint — all at once. The agent diagnosed each error from the traceback alone, explained the root cause in one sentence, and gave the exact fix. No vague suggestions, no "try reinstalling Python."

```
RuntimeError: This version of jaxlib was built using AVX instructions...
→ pip uninstall jax jaxlib -y

AttributeError: module 'autoray.autoray' has no attribute 'NumpyMimic'
→ pip install "autoray==0.6.12"
```

Two commands. Done.

### 3. It upgraded the project on a single instruction

When I said "make it more impressive," the agent didn't just tweak parameters. It redesigned the output entirely:

- Added head-to-head benchmarking against Logistic Regression and RBF-SVM
- Added ROC-AUC scoring (the metric that matters in fraud/medical contexts)
- Added a live Unicode progress bar in the terminal
- Generated a 6-panel dark-theme visualization: decision boundaries for all three models side by side, a learning curve, and a comparison bar chart
- Added a cosine annealing LR scheduler for smoother convergence

All of that came from four words.

### 4. It caught its own bug before I did

When the visualization crashed with a feature dimension mismatch (classical models trained on 4-feature padded data, but the plot grid only had 2 features), the agent identified the root cause immediately, refactored `load_dataset()` to return both the 2-feature and 4-feature versions with a shared index split, and updated every downstream call site — without being asked to explain what went wrong.

---

## The Output

**Terminal:**
```
══════════════════════════════════════════════════════════════
  ⬡  Quantum-Enhanced Predictive Analytics Engine
══════════════════════════════════════════════════════════════
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
══════════════════════════════════════════════════════════════
```

**Visualization:** A publication-quality 6-panel figure showing decision boundaries, learning curve, and model comparison — generated automatically, saved as a PNG.

**GitHub repo:** README with architecture diagrams, hyperparameter guide, theoretical background, references, pinned requirements, MIT license, and `.gitignore` — all produced in the same session.

---

## Why This Session Matters for What I'm Building

The quantum classifier is a technical demo, but the session itself demonstrates the thing I care about: **using AI agents to compress the distance between an idea and a working, explainable artifact**.

A solo founder in deep tech can't afford to be blocked by dependency hell, boilerplate, or the gap between "I understand this concept" and "I have code that proves it." This session went from a concept to a benchmarked, documented, GitHub-ready project in one sitting. That's the leverage I'm building on.

The quantum ML work also points at a real thesis: as quantum hardware matures, the software toolchain for enterprise quantum applications is wide open. The classifier here runs on a simulator — but the same circuit runs on real hardware with one line change. The pipeline from classical data to quantum prediction to business decision is the product.

---

## Repository

[github.com/sanskritifarswal/quantum_variational_classifier](https://github.com/sanskritifarswal/quantum_variational_classifier)

- `quantum_classifier.py` — full pipeline, ~400 lines
- `decision_boundaries.png` — visualization output
- `README.md` — technical documentation
- `requirements.txt` — pinned dependencies with compatibility notes
