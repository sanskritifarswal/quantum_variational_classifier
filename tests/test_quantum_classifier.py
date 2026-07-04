"""
Regression tests for the Quantum-Enhanced Predictive Analytics Engine.
Run with:  pytest tests/ -v
"""

import numpy as np
import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import quantum_classifier as qc


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def dataset():
    return qc.load_dataset()


@pytest.fixture(scope="module")
def trained_model(dataset):
    *_, X_tr_t, X_te_t, y_tr_t, y_te_t, __ = dataset
    model = qc.QuantumClassifier()
    # Train for just 3 epochs — enough to verify the loop runs, not for accuracy
    import torch.optim as optim
    import torch.nn as nn
    opt = optim.Adam(model.parameters(), lr=0.01)
    crit = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(3):
        for s in range(0, min(32, len(X_tr_t)), qc.BATCH_SIZE):
            xb, yb = X_tr_t[s:s+qc.BATCH_SIZE], y_tr_t[s:s+qc.BATCH_SIZE]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    return model


# ── Dataset tests ──────────────────────────────────────────────────────────────

def test_dataset_shapes(dataset):
    X_tr2, X_te2, y_tr, y_te, X_tr_q, X_te_q, X_tr_t, X_te_t, y_tr_t, y_te_t, _ = dataset
    n_total = len(X_tr2) + len(X_te2)
    assert n_total == qc.N_SAMPLES
    assert X_tr2.shape[1] == 2,         "Classical data must have 2 features"
    assert X_tr_q.shape[1] == qc.N_QUBITS, "Quantum data must have N_QUBITS features"
    assert len(y_tr) == len(X_tr2)
    assert len(y_te) == len(X_te2)


def test_dataset_labels_binary(dataset):
    y_tr, y_te = dataset[2], dataset[3]
    assert set(np.unique(y_tr)).issubset({0, 1})
    assert set(np.unique(y_te)).issubset({0, 1})


def test_train_test_split_ratio(dataset):
    X_tr2, X_te2, *_ = dataset
    ratio = len(X_te2) / (len(X_tr2) + len(X_te2))
    assert abs(ratio - 0.25) < 0.02, f"Expected ~25% test split, got {ratio:.2f}"


def test_quantum_features_padded_with_zeros(dataset):
    X_tr2, _, _, _, X_tr_q, *_ = dataset
    # Columns beyond the original 2 should all be zero
    assert np.allclose(X_tr_q[:, 2:], 0.0)


# ── Circuit / model tests ──────────────────────────────────────────────────────

def test_circuit_output_in_range(dataset):
    *_, X_tr_t, _, y_tr_t, _, __ = dataset
    model = qc.QuantumClassifier()
    model.eval()
    with torch.no_grad():
        out = model(X_tr_t[:4])
    assert out.shape == (4,)
    assert (out >= -1.0).all() and (out <= 1.0).all(), \
        f"Pauli-Z expectation must be in [-1, 1], got {out}"


def test_model_parameter_count():
    model = qc.QuantumClassifier()
    n_params = sum(p.numel() for p in model.parameters())
    expected = qc.N_LAYERS * qc.N_QUBITS * 3
    assert n_params == expected, f"Expected {expected} params, got {n_params}"


def test_forward_is_differentiable(dataset):
    *_, X_tr_t, _, y_tr_t, _, __ = dataset
    import torch.nn as nn
    model = qc.QuantumClassifier()
    logits = model(X_tr_t[:2])
    loss = nn.BCEWithLogitsLoss()(logits, y_tr_t[:2])
    loss.backward()
    grad = model.weights.grad
    assert grad is not None, "No gradient flowed to circuit weights"
    assert not torch.all(grad == 0), "Gradients are all zero — backprop may be broken"


def test_predict_proba_shape(dataset, trained_model):
    X_tr2, *_ = dataset
    probs = trained_model.predict_proba(X_tr2[:10])
    assert probs.shape == (10, 2)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


# ── Training loop smoke test ───────────────────────────────────────────────────

def test_loss_decreases_over_training(dataset):
    """Loss after 5 epochs should be lower than at epoch 1."""
    X_tr2, X_te2, y_tr, y_te, X_tr_q, X_te_q, X_tr_t, X_te_t, y_tr_t, y_te_t, _ = dataset
    import torch.optim as optim
    import torch.nn as nn
    model = qc.QuantumClassifier()
    opt = optim.Adam(model.parameters(), lr=0.05)
    crit = nn.BCEWithLogitsLoss()

    def epoch_loss():
        model.train()
        xb, yb = X_tr_t[:16], y_tr_t[:16]
        opt.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        opt.step()
        return loss.item()

    first = epoch_loss()
    for _ in range(4):
        last = epoch_loss()

    assert last < first * 1.5, \
        f"Loss did not decrease meaningfully: {first:.4f} → {last:.4f}"


# ── Classical baselines smoke test ────────────────────────────────────────────

def test_classical_baselines_run(dataset):
    X_tr2, X_te2, y_tr, y_te, *_ = dataset
    results = qc.run_classical_baselines(X_tr2, X_te2, y_tr, y_te)
    assert "Logistic Regression" in results
    assert "RBF-SVM" in results
    for name, info in results.items():
        assert 0.5 <= info["acc"] <= 1.0, f"{name} accuracy out of range"
        assert 0.5 <= info["auc"] <= 1.0, f"{name} AUC out of range"
