"""
Quantum-Enhanced Predictive Analytics Engine
=============================================
Head-to-head benchmark: Parameterized Quantum Circuit (PQC) vs. classical baselines.

Data flow:
  Classical features → Quantum Feature Map (AngleEmbedding)
                     → Variational Ansatz (StronglyEntanglingLayers)
                     → Pauli-Z Measurement
                     → Classical Adam optimizer (backprop through the circuit)

Run:  python quantum_classifier.py
Output: terminal report + decision_boundaries.png
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import matplotlib
matplotlib.use("Agg")          # headless — works without a display
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

N_QUBITS   = 4
N_LAYERS   = 6       # deeper ansatz → richer expressivity
N_SAMPLES  = 400
NOISE      = 0.20    # deliberately harder dataset
BATCH_SIZE = 32
EPOCHS     = 80
LR         = 0.008
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

COLORS = {"quantum": "#6C63FF", "logreg": "#FF6584", "svm": "#43AA8B"}


# ──────────────────────────────────────────────
# 1. DATASET
# ──────────────────────────────────────────────

def load_dataset():
    X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)           # 2-feature version (for classical models + plotting)
    X_q = np.pad(X, ((0, 0), (0, N_QUBITS - 2)), mode="constant")  # 4-feature for quantum

    # Split indices so both versions share the same train/test split
    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(idx, test_size=0.25, random_state=SEED)

    to_t = lambda a: torch.tensor(a, dtype=torch.float32)
    return (
        X[idx_tr],    X[idx_te],    y[idx_tr],    y[idx_te],    # 2-feat (classical)
        X_q[idx_tr],  X_q[idx_te],                               # 4-feat (quantum, numpy)
        to_t(X_q[idx_tr]), to_t(X_q[idx_te]),                   # 4-feat (quantum, tensor)
        to_t(y[idx_tr]),   to_t(y[idx_te]),
        scaler,
    )


# ──────────────────────────────────────────────
# 2. QUANTUM CIRCUIT
# ──────────────────────────────────────────────

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev, interface="torch", diff_method="backprop")
def quantum_circuit(inputs, weights):
    """
    Feature Map  : AngleEmbedding — maps each feature to an RX rotation angle.
    Ansatz       : StronglyEntanglingLayers — arbitrary rotations + CNOT entanglement.
    Measurement  : ⟨Z⟩ on qubit 0, returning a scalar in [-1, +1].
    """
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="X")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


class QuantumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        shape = qml.StronglyEntanglingLayers.shape(N_LAYERS, N_QUBITS)
        self.weights = nn.Parameter(torch.randn(shape) * 0.1)

    def forward(self, x):
        return torch.stack([quantum_circuit(x[i], self.weights) for i in range(len(x))])

    def predict_proba(self, X_np):
        """Sklearn-compatible interface for decision boundary plotting."""
        self.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_np, dtype=torch.float32)
            logits = self.forward(X_t)
            probs = torch.sigmoid(logits).numpy()
        return np.column_stack([1 - probs, probs])


# ──────────────────────────────────────────────
# 3. TRAINING LOOP
# ──────────────────────────────────────────────

def train(model, X_train_t, y_train_t, X_test_t, y_test_t):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}

    print(f"\n{'Epoch':>6}  {'Loss':>8}  {'Train':>8}  {'Test':>8}  {'Bar'}")
    print("─" * 62)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss, total_acc, n = 0.0, 0.0, 0

        for s in range(0, len(X_train_t), BATCH_SIZE):
            xb, yb = X_train_t[s:s+BATCH_SIZE], y_train_t[s:s+BATCH_SIZE]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_acc  += ((logits >= 0).long() == yb.long()).float().mean().item()
            n += 1

        scheduler.step()

        tr_loss = total_loss / n
        tr_acc  = total_acc  / n

        model.eval()
        with torch.no_grad():
            te_logits = model(X_test_t)
            te_acc = ((te_logits >= 0).long() == y_test_t.long()).float().mean().item()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)

        if epoch % 5 == 0 or epoch == 1:
            bar_len = int(te_acc * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"{epoch:>6}  {tr_loss:>8.4f}  {tr_acc*100:>7.1f}%  {te_acc*100:>7.1f}%  {bar}")

    return history


# ──────────────────────────────────────────────
# 4. CLASSICAL BASELINES
# ──────────────────────────────────────────────

def run_classical_baselines(X_train, X_test, y_train, y_test):
    """X_train/X_test should be the original 2-feature (unpadded) arrays."""
    results = {}

    lr = LogisticRegression(max_iter=1000, random_state=SEED)
    lr.fit(X_train, y_train)
    results["Logistic Regression"] = {
        "acc": accuracy_score(y_test, lr.predict(X_test)),
        "auc": roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1]),
        "model": lr,
        "color": COLORS["logreg"],
    }

    svm = SVC(kernel="rbf", probability=True, random_state=SEED)
    svm.fit(X_train, y_train)
    results["RBF-SVM"] = {
        "acc": accuracy_score(y_test, svm.predict(X_test)),
        "auc": roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1]),
        "model": svm,
        "color": COLORS["svm"],
    }

    return results


# ──────────────────────────────────────────────
# 5. VISUALISATION
# ──────────────────────────────────────────────

def make_grid(X, resolution=120):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    return xx, yy


def plot_boundary(ax, model, xx, yy, X_np, y_np, title, color, is_quantum=False):
    grid_pts = np.c_[xx.ravel(), yy.ravel()]
    if is_quantum:
        # Pad to N_QUBITS features
        grid_pts_q = np.pad(grid_pts, ((0, 0), (0, N_QUBITS - 2)), mode="constant")
        Z = model.predict_proba(grid_pts_q)[:, 1].reshape(xx.shape)
    else:
        Z = model.predict_proba(grid_pts)[:, 1].reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=20, cmap="RdYlBu", alpha=0.75)
    ax.contour(xx, yy, Z, levels=[0.5], colors=color, linewidths=2.5)

    c0 = ax.scatter(X_np[y_np == 0, 0], X_np[y_np == 0, 1],
                    c="#222", edgecolors="white", s=40, linewidth=0.6, label="Class 0", zorder=3)
    c1 = ax.scatter(X_np[y_np == 1, 0], X_np[y_np == 1, 1],
                    c=color, edgecolors="white", s=40, linewidth=0.6, label="Class 1", zorder=3)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[["top","right","bottom","left"]].set_visible(False)


def plot_learning_curve(ax, history):
    epochs = range(1, len(history["train_acc"]) + 1)
    ax.plot(epochs, [a * 100 for a in history["train_acc"]],
            color=COLORS["quantum"], linewidth=2, label="Train acc")
    ax.plot(epochs, [a * 100 for a in history["test_acc"]],
            color=COLORS["quantum"], linewidth=2, linestyle="--", label="Test acc")
    ax.fill_between(epochs,
                    [a * 100 for a in history["train_acc"]],
                    [a * 100 for a in history["test_acc"]],
                    alpha=0.12, color=COLORS["quantum"])
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Quantum Classifier — Learning Curve", fontsize=13, fontweight="bold")
    ax.set_ylim(40, 102)
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(alpha=0.25)


def plot_comparison_bar(ax, q_acc, q_auc, baselines):
    names  = ["Quantum\nPQC"] + [n.replace("-", "-\n") for n in baselines]
    accs   = [q_acc] + [v["acc"] for v in baselines.values()]
    aucs   = [q_auc] + [v["auc"] for v in baselines.values()]
    colors = [COLORS["quantum"], COLORS["logreg"], COLORS["svm"]]

    x = np.arange(len(names))
    w = 0.35
    bars1 = ax.bar(x - w/2, [a*100 for a in accs], w, color=colors, alpha=0.85, label="Accuracy (%)")
    bars2 = ax.bar(x + w/2, [a*100 for a in aucs], w, color=colors, alpha=0.45, label="ROC-AUC (%)")

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(50, 108)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Model Comparison: Accuracy & ROC-AUC", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def save_figure(model, history, baselines, q_acc, q_auc,
                X_train_np, X_test_np, y_train_np, y_test_np):
    fig = plt.figure(figsize=(18, 11), facecolor="#0f0f14")

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.28,
                           left=0.05, right=0.97, top=0.91, bottom=0.07)

    for ax in fig.get_axes():
        ax.set_facecolor("#1a1a24")

    X_all  = np.vstack([X_train_np, X_test_np])
    y_all  = np.concatenate([y_train_np, y_test_np])
    xx, yy = make_grid(X_all[:, :2])

    text_kw = dict(color="white")

    # Decision boundaries — row 0
    for col, (name, info) in enumerate({"Quantum PQC": None,
                                         "Logistic Regression": baselines["Logistic Regression"],
                                         "RBF-SVM": baselines["RBF-SVM"]}.items()):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor("#1a1a24")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        if name == "Quantum PQC":
            plot_boundary(ax, model, xx, yy, X_all[:, :2], y_all,
                          f"Quantum PQC  ✦  {q_acc*100:.1f}% acc",
                          COLORS["quantum"], is_quantum=True)
        else:
            plot_boundary(ax, info["model"], xx, yy, X_all[:, :2], y_all,
                          f"{name}  ✦  {info['acc']*100:.1f}% acc",
                          info["color"])
        ax.title.set_color("white")
        ax.tick_params(colors="white")

    # Learning curve — row 1, col 0-1
    ax_lc = fig.add_subplot(gs[1, 0:2])
    ax_lc.set_facecolor("#1a1a24")
    for spine in ax_lc.spines.values():
        spine.set_edgecolor("#333")
    plot_learning_curve(ax_lc, history)
    ax_lc.title.set_color("white")
    ax_lc.xaxis.label.set_color("white")
    ax_lc.yaxis.label.set_color("white")
    ax_lc.tick_params(colors="white")
    ax_lc.legend(facecolor="#1a1a24", labelcolor="white", edgecolor="#333")

    # Comparison bar — row 1, col 2
    ax_bar = fig.add_subplot(gs[1, 2])
    ax_bar.set_facecolor("#1a1a24")
    for spine in ax_bar.spines.values():
        spine.set_edgecolor("#333")
    plot_comparison_bar(ax_bar, q_acc, q_auc, baselines)
    ax_bar.title.set_color("white")
    ax_bar.xaxis.label.set_color("white")
    ax_bar.yaxis.label.set_color("white")
    ax_bar.tick_params(colors="white")
    ax_bar.legend(facecolor="#1a1a24", labelcolor="white", edgecolor="#333")

    fig.suptitle(
        "Quantum-Enhanced Predictive Analytics Engine  —  PQC vs. Classical Classifiers",
        fontsize=15, fontweight="bold", color="white", y=0.97
    )

    path = "decision_boundaries.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────

def main():
    print("\n" + "═" * 62)
    print("  ⬡  Quantum-Enhanced Predictive Analytics Engine")
    print("     PQC Classifier  ·  PennyLane + PyTorch")
    print("═" * 62)
    print(f"  Qubits : {N_QUBITS}   Layers : {N_LAYERS}   Params : {N_LAYERS * N_QUBITS * 3}")
    print(f"  Dataset: {N_SAMPLES} samples, noise={NOISE}   Epochs: {EPOCHS}   LR: {LR}")
    print("═" * 62)

    # Data
    (X_tr2, X_te2, y_tr, y_te,          # 2-feat numpy (classical models + plotting)
     X_tr_q, X_te_q,                     # 4-feat numpy (quantum numpy)
     X_tr_t, X_te_t,                     # 4-feat tensors (quantum training)
     y_tr_t, y_te_t,
     _) = load_dataset()
    print(f"\n  Train: {len(X_tr2)}  |  Test: {len(X_te2)}")

    # Quantum model
    model = QuantumClassifier()
    t0 = time.time()
    history = train(model, X_tr_t, y_tr_t, X_te_t, y_te_t)
    q_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        te_logits = model(X_te_t)
        q_preds   = (te_logits >= 0).long().numpy()
        q_probs   = torch.sigmoid(te_logits).numpy()

    q_acc = accuracy_score(y_te, q_preds)
    q_auc = roc_auc_score(y_te, q_probs)

    # Classical baselines trained on 2-feature data
    print("\n  Training classical baselines …")
    baselines = run_classical_baselines(X_tr2, X_te2, y_tr, y_te)

    # ── Summary table ──────────────────────────────────────────────────
    print("\n" + "═" * 62)
    print(f"  {'Model':<24} {'Accuracy':>10}  {'ROC-AUC':>9}")
    print("─" * 62)
    print(f"  {'Quantum PQC':<24} {q_acc*100:>9.2f}%  {q_auc:>9.4f}  ◀ our model")
    for name, info in baselines.items():
        print(f"  {name:<24} {info['acc']*100:>9.2f}%  {info['auc']:>9.4f}")
    print("═" * 62)

    print(f"\n  Quantum training time : {q_time:.1f}s")
    print(f"  Parameters            : {N_LAYERS * N_QUBITS * 3}  (vs. 100s in a comparable NN)")

    print("\n  Quantum classifier — detailed report:")
    print(classification_report(y_te, q_preds, target_names=["Class 0", "Class 1"]))

    # ── Circuit diagram ────────────────────────────────────────────────
    print("  Sample circuit (first test point):\n")
    print(qml.draw(quantum_circuit)(X_te_t[0], model.weights.detach()))

    # ── Plot ────────────────────────────────────────────────────────────
    print("\n  Generating visualisation …")
    path = save_figure(
        model, history, baselines, q_acc, q_auc,
        X_tr2, X_te2, y_tr, y_te,
    )
    print(f"  Saved → {path}")
    print("\n  ✓  Pipeline complete.\n")


if __name__ == "__main__":
    main()
