#!/usr/bin/env python
"""Generate Lab 6 (GCN from scratch -> PyG) — student and solution notebooks.

One source of truth: cells below. Exercise cells carry SOLUTION blocks; the student
build replaces them with a TODO stub (tagged `student-todo`), the solution build keeps
them (markers included — the public-repo CI guard greps for those markers, which is why
solutions can never leak). Usage:

    python scripts/labgen/make_lab06.py            # writes both notebooks
"""

from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "labs" / "lab06_gcn.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "labs" / "lab06_gcn.ipynb"

SOL_RE = re.compile(r"[ \t]*### BEGIN SOLUTION.*?### END SOLUTION[ \t]*\n?", re.S)


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(text, stub): return ("code", text, stub)


CELLS = [
    md("""# Lab 6 · GCN from scratch → PyTorch Geometric

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/labs/lab06_gcn.ipynb)

**Week 6 · [lecture](https://lukmanovr.github.io/dkr/lectures/06-gcn.html) · ≈ 25 min of compute (free Colab or CPU)**

You will build this week's lecture, in order: construct the normalized propagation
operator $\\hat{\\mathbf{A}} = \\tilde{\\mathbf{D}}^{-1/2}\\tilde{\\mathbf{A}}\\tilde{\\mathbf{D}}^{-1/2}$
from a raw edge list, implement the GCN layer as ten lines of dense PyTorch, verify the
lecture's hand-simulation numerically, train a two-layer GCN on **Cora** against the
structure-blind MLP baseline, and finish by proving your layer computes exactly what
PyG's `GCNConv` computes. The method is the graph convolutional network of
[Kipf & Welling, 2017](https://arxiv.org/abs/1609.02907) — semi-supervised node
classification by learned, normalized neighborhood averaging — and the ablation you run
in §8 is SGC ([Wu et al., 2019](https://arxiv.org/abs/1902.07153)), which asks how much
of the GCN's win survives with the "neural network" deleted.

### Goals
1. Build and *verify* $\\hat{\\mathbf{A}}$ — assertions before training, always.
2. Implement the GCN layer from the lecture's GCN-layer definition and check it on paper-sized examples.
3. Reproduce the two numbers from lecture: GCN ≈ 0.80, MLP ≈ 0.60 test accuracy on Cora — and *watch* the hidden representations organize while training.
4. Implement sparse propagation and **measure** the lecture's §2 cost claims (dense vs sparse).
5. Run the SGC ablation: how much of the GCN's win is the fixed low-pass filter?
6. Show from-scratch == library: your layer vs `GCNConv`, `allclose`.
"""),

    md("""## 0 · Setup

One pinned install; on Colab we never reinstall `torch` (the preinstalled build is
supported by PyG 2.8). The `SMOKE` flag lets the course CI run this notebook with
reduced epochs — leave it untouched.
"""),

    code("""import os, sys, random

SMOKE = os.environ.get("SMOKE", "") == "1"     # CI mode: fewer epochs, softer thresholds
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    %pip install -q torch_geometric==2.8.0.post1

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 41
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} · device {device} · SMOKE={SMOKE}")"""),

    md("""## 1 · Cora, hands on

Cora: 2,708 papers, 5,429 citations, 1,433-word vocabulary, 7 topics, and — the
semi-supervised twist — only 140 training labels. `Planetoid` hands us the standard
splits used by every paper since 2016.

**Predict before you run the next cell** (write both guesses down): out of 2,708 papers,
how many cite or are cited by *nobody* (degree 0)? And what is the *maximum* degree —
tens? hundreds? Your intuition about real graph degree distributions is being calibrated
here; check yourself against the printout.
"""),

    code("""from torch_geometric.datasets import Planetoid

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]

print(data)
print(f"nodes {data.num_nodes} · edges {data.num_edges} · features {dataset.num_features} "
      f"· classes {dataset.num_classes}")
print(f"labeled for training: {int(data.train_mask.sum())} "
      f"({100 * data.train_mask.sum() / data.num_nodes:.1f}%) · test: {int(data.test_mask.sum())}")

deg = torch.bincount(data.edge_index[0], minlength=data.num_nodes)
print(f"degree — mean {deg.float().mean():.2f}, max {int(deg.max())}, isolated {(deg == 0).sum().item()}")"""),

    md("""## 2 · Build the propagation operator $\\hat{\\mathbf{A}}$  *(exercise 1)*

From the lecture: $\\tilde{\\mathbf{A}} = \\mathbf{A} + \\mathbf{I}$,
$\\tilde{\\mathbf{D}} = \\mathrm{diag}(\\tilde{\\mathbf{A}}\\mathbf{1})$, and
$\\hat{\\mathbf{A}} = \\tilde{\\mathbf{D}}^{-1/2}\\tilde{\\mathbf{A}}\\tilde{\\mathbf{D}}^{-1/2}$.

**What to do:** in the next cell, implement `normalize_adjacency(A)` so that it returns
$\\hat{\\mathbf{A}}$ computed from a dense adjacency matrix: add self-loops, compute the
degrees of the self-looped graph, and scale rows *and* columns by the inverse square
root of the degree. The check below the function runs it on a 3-node path graph whose
$\\hat{\\mathbf{A}}$ you can (and should) verify by hand — the habit from Pitfall 3.
When the cell prints "exercise 1 ✓" your operator matches the hand computation.
"""),

    todo("""def to_dense_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    \"\"\"Dense symmetric adjacency from a PyG edge_index (already contains both directions).\"\"\"
    A = torch.zeros(num_nodes, num_nodes)
    A[edge_index[0], edge_index[1]] = 1.0
    return A


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    \"\"\"Return A_hat = D̃^{-1/2} (A + I) D̃^{-1/2}.

    Steps: add self-loops; compute the degree vector of the result; build the
    inverse-sqrt degree; scale rows AND columns (keep everything dense).
    \"\"\"
    ### BEGIN SOLUTION
    A_tilde = A + torch.eye(A.shape[0])
    d = A_tilde.sum(dim=1)
    d_inv_sqrt = d.pow(-0.5)
    return d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]
    ### END SOLUTION


# ── check on the 3-node path 0—1—2 (hand-verifiable) ─────────────────────────
path = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
A_hat_path = normalize_adjacency(to_dense_adj(path, 3))
expected = torch.tensor([
    [1/2,            1/6**0.5, 0.0],
    [1/6**0.5,       1/3,      1/6**0.5],
    [0.0,            1/6**0.5, 1/2],
])
assert torch.allclose(A_hat_path, expected, atol=1e-6), (
    f"A_hat mismatch — check: (1) self-loops added BEFORE computing degrees? "
    f"(2) scaled by d^-1/2 on rows AND columns?\\ngot:\\n{A_hat_path}"
)
assert torch.allclose(A_hat_path, A_hat_path.T), "A_hat must stay symmetric"
print("exercise 1 ✓ — A_hat matches the hand computation")""",
         stub="""def to_dense_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    \"\"\"Dense symmetric adjacency from a PyG edge_index (already contains both directions).\"\"\"
    A = torch.zeros(num_nodes, num_nodes)
    A[edge_index[0], edge_index[1]] = 1.0
    return A


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    \"\"\"Return A_hat = D̃^{-1/2} (A + I) D̃^{-1/2}.

    Steps: add self-loops; compute the degree vector of the result; build the
    inverse-sqrt degree; scale rows AND columns (keep everything dense).
    \"\"\"
    # TODO: implement (≈ 4 lines). Delete the raise when done.
    raise NotImplementedError("implement normalize_adjacency")


# ── check on the 3-node path 0—1—2 (hand-verifiable) ─────────────────────────
path = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
A_hat_path = normalize_adjacency(to_dense_adj(path, 3))
expected = torch.tensor([
    [1/2,            1/6**0.5, 0.0],
    [1/6**0.5,       1/3,      1/6**0.5],
    [0.0,            1/6**0.5, 1/2],
])
assert torch.allclose(A_hat_path, expected, atol=1e-6), (
    f"A_hat mismatch — check: (1) self-loops added BEFORE computing degrees? "
    f"(2) scaled by d^-1/2 on rows AND columns?\\ngot:\\n{A_hat_path}"
)
assert torch.allclose(A_hat_path, A_hat_path.T), "A_hat must stay symmetric"
print("exercise 1 ✓ — A_hat matches the hand computation")"""),

    md("""## 3 · Verify the lecture's hand-simulation

Nothing to implement here — read the next cell, then run it. Before trusting code with
2,708 nodes, we make it reproduce the 4-node example you can check with a pencil (the
lecture's hand-simulation example): mean-aggregate neighbors, add your own value, apply
ReLU. The expected result is $(3,\\; 1,\\; 3.5,\\; 0)$ — including node D getting
clipped to zero.
"""),

    code("""ex_edges = torch.tensor([[0, 1, 0, 2, 1, 2, 1, 3], [1, 0, 2, 0, 2, 1, 3, 1]])  # AB,AC,BC,BD
A_ex = to_dense_adj(ex_edges, 4)
x_ex = torch.tensor([2.0, -1.0, 3.0, 1.0])

D_inv = torch.diag(1.0 / A_ex.sum(dim=1))
h1 = F.relu(D_inv @ A_ex @ x_ex + x_ex)          # mean of neighbors + own value, ReLU

print("h(1) =", h1.tolist())
assert torch.allclose(h1, torch.tensor([3.0, 1.0, 3.5, 0.0])), "does not match the lecture table"
print("matches the lecture table ✓ (note D: erased by the ReLU)")"""),

    md("""## 4 · The GCN layer and model  *(exercise 2)*

This section turns the lecture's GCN-layer definition into code: one layer computes
`A_hat @ (X @ W)` — the propagation is fixed, and only the channel mix $\\mathbf{W}$
is learned.

**What to do:** in the cell after the algorithm below, implement `GCNLayer.forward` so
that it returns `A_hat @ self.W(X)` — mix the channels first, then propagate (that
order is cheaper). The two-layer `GCN` model around it is provided. The check verifies
that with $\\mathbf{W} = \\mathbf{I}$ your layer equals plain propagation; when the
cell prints "exercise 2 ✓" it is correct.
"""),

    md("""**Algorithm · Two-layer GCN forward pass.** The algorithm below (mirroring
the lecture's algorithm float exactly) specifies exactly what your implementation in
the next cell must do; follow it step by step:

**Input:** features $\\mathbf{X} \\in \\mathbb{R}^{n \\times d}$; sparse
$\\hat{\\mathbf{A}}$; weights $\\mathbf{W}_1$, $\\mathbf{W}_2$; dropout rate $p$.
**Output:** logits $\\mathbf{Z} \\in \\mathbb{R}^{n \\times C}$.

1. $\\mathbf{H} \\leftarrow \\mathbf{X}\\mathbf{W}_1$ — mix channels first (cheaper)
2. $\\mathbf{H} \\leftarrow \\hat{\\mathbf{A}}\\,\\mathbf{H}$ — propagate one hop
3. $\\mathbf{H} \\leftarrow \\mathrm{ReLU}(\\mathbf{H})$
4. $\\mathbf{H} \\leftarrow \\mathrm{Dropout}(\\mathbf{H}, p)$ — training only
5. $\\mathbf{Z} \\leftarrow \\hat{\\mathbf{A}}\\,(\\mathbf{H}\\mathbf{W}_2)$ — second layer
6. **return** $\\mathbf{Z}$ — softmax lives inside the loss

Steps 1–2 are `GCNLayer.forward` (one call); steps 1–6 are `GCN.forward`.
"""),

    todo("""class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        \"\"\"One GCN propagation: A_hat @ (X W). Mix channels first — it is cheaper
        (n×d @ d×d' then n×n @ n×d', instead of n×n @ n×d first).\"\"\"
        ### BEGIN SOLUTION
        return A_hat @ self.W(X)
        ### END SOLUTION


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, p_drop=0.5):
        super().__init__()
        self.l1, self.l2 = GCNLayer(in_dim, hidden), GCNLayer(hidden, out_dim)
        self.p = p_drop

    def forward(self, X, A_hat):
        H = F.relu(self.l1(X, A_hat))
        H = F.dropout(H, p=self.p, training=self.training)
        return self.l2(H, A_hat)                       # logits; softmax lives in the loss


# shape + value check: with W = I, one layer must equal plain propagation
layer = GCNLayer(3, 3)
with torch.no_grad():
    layer.W.weight.copy_(torch.eye(3))
probe = torch.randn(3, 3)
assert torch.allclose(layer(probe, A_hat_path), A_hat_path @ probe, atol=1e-6)
print("exercise 2 ✓ — layer computes A_hat @ X W")""",
         stub="""class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, X: torch.Tensor, A_hat: torch.Tensor) -> torch.Tensor:
        \"\"\"One GCN propagation: A_hat @ (X W). Mix channels first — it is cheaper
        (n×d @ d×d' then n×n @ n×d', instead of n×n @ n×d first).\"\"\"
        # TODO: one line. Delete the raise when done.
        raise NotImplementedError("implement GCNLayer.forward")


class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, p_drop=0.5):
        super().__init__()
        self.l1, self.l2 = GCNLayer(in_dim, hidden), GCNLayer(hidden, out_dim)
        self.p = p_drop

    def forward(self, X, A_hat):
        H = F.relu(self.l1(X, A_hat))
        H = F.dropout(H, p=self.p, training=self.training)
        return self.l2(H, A_hat)                       # logits; softmax lives in the loss


# shape + value check: with W = I, one layer must equal plain propagation
layer = GCNLayer(3, 3)
with torch.no_grad():
    layer.W.weight.copy_(torch.eye(3))
probe = torch.randn(3, 3)
assert torch.allclose(layer(probe, A_hat_path), A_hat_path @ probe, atol=1e-6)
print("exercise 2 ✓ — layer computes A_hat @ X W")"""),

    md("""## 5 · Train on Cora — and measure what nothing-but-features buys first  *(exercise 3)*

We now train two models under one protocol. The MLP sees only the bag-of-words matrix;
the GCN sees the same features **plus** propagation. The professional habit from
Pitfall 5: the GNN number is meaningless without the baseline next to it.

**What to do:** in the next cell, implement `accuracy(logits, y, mask)` so that it
returns the fraction of correct predictions (argmax over classes) among the nodes
selected by `mask`. The asserts check it on a tiny hand-computable probe; "exercise 3 ✓"
confirms it. The cell after that (provided) trains both models with your `accuracy` —
run it and compare the two printed test accuracies.
"""),

    todo("""def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    \"\"\"Fraction of correct predictions among the nodes selected by `mask`.\"\"\"
    ### BEGIN SOLUTION
    pred = logits.argmax(dim=-1)
    return (pred[mask] == y[mask]).float().mean().item()
    ### END SOLUTION


_probe_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
_probe_y = torch.tensor([0, 1, 1])
_probe_mask = torch.tensor([True, True, False])
assert abs(accuracy(_probe_logits, _probe_y, _probe_mask) - 1.0) < 1e-6
assert abs(accuracy(_probe_logits, _probe_y, torch.tensor([True] * 3)) - 2 / 3) < 1e-6
print("exercise 3 ✓ — masked accuracy")""",
         stub="""def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    \"\"\"Fraction of correct predictions among the nodes selected by `mask`.\"\"\"
    # TODO: argmax over classes, compare to y under the mask (≈ 2 lines).
    raise NotImplementedError("implement accuracy")


_probe_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
_probe_y = torch.tensor([0, 1, 1])
_probe_mask = torch.tensor([True, True, False])
assert abs(accuracy(_probe_logits, _probe_y, _probe_mask) - 1.0) < 1e-6
assert abs(accuracy(_probe_logits, _probe_y, torch.tensor([True] * 3)) - 2 / 3) < 1e-6
print("exercise 3 ✓ — masked accuracy")"""),

    code("""EPOCHS = 30 if SMOKE else 150

A_hat = normalize_adjacency(to_dense_adj(data.edge_index, data.num_nodes)).to(device)
X, y = data.x.to(device), data.y.to(device)
train_m, val_m, test_m = (m.to(device) for m in (data.train_mask, data.val_mask, data.test_mask))


def train(model, forward):
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_test = 0.0, 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train(); opt.zero_grad()
        loss = F.cross_entropy(forward(model)[train_m], y[train_m])
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits = forward(model)
            va, ta = accuracy(logits, y, val_m), accuracy(logits, y, test_m)
        if va > best_val:
            best_val, best_test = va, ta
        if epoch % 25 == 0 or epoch == EPOCHS:
            print(f"  epoch {epoch:3d} · loss {loss:.3f} · val {va:.3f} · test {ta:.3f}")
    return best_test


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Dropout(0.5), nn.Linear(hidden, out_dim))

    def forward(self, X):
        return self.net(X)


print("MLP (features only):")
mlp_acc = train(MLP(dataset.num_features, 64, dataset.num_classes).to(device), lambda m: m(X))
print("GCN (features + propagation):")
gcn_acc = train(GCN(dataset.num_features, 16, dataset.num_classes).to(device), lambda m: m(X, A_hat))

print(f"\\nMLP {mlp_acc:.3f}  vs  GCN {gcn_acc:.3f}  → structure is worth "
      f"{100 * (gcn_acc - mlp_acc):.1f} accuracy points on Cora")
floor = 0.55 if SMOKE else 0.75
assert gcn_acc >= floor, f"GCN test accuracy {gcn_acc:.3f} below the {floor} floor — check A_hat and the layer"
assert gcn_acc > mlp_acc, "the GCN should beat the structure-blind baseline here"
print("training checks ✓")"""),

    md("""## 6 · Watch the embeddings organize

Numbers hide what training actually does to the representation. Project the GCN's hidden
layer to 2-D (PCA — no learned projection, so what you see is really there) before and
after training: seven diffuse clouds become seven clusters, arranged so that *citation
neighborhoods share regions*. The MLP's hidden layer, trained the same way, separates the
classes too — but pull up any node's neighbors and they scatter, because nothing tied
them together.
"""),

    code("""import matplotlib.pyplot as plt


def hidden_pca(model, X, A_hat):
    model.eval()
    with torch.no_grad():
        H = F.relu(model.l1(X, A_hat))                     # hidden layer, n × 16
        H = H - H.mean(dim=0, keepdim=True)
        U_, S_, V_ = torch.pca_lowrank(H, q=2)
        return (H @ V_[:, :2]).cpu()


untrained = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
before = hidden_pca(untrained, X, A_hat)

trained = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
_ = train(trained, lambda m: m(X, A_hat))
after = hidden_pca(trained, X, A_hat)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
for ax, Z, name in [(axes[0], before, "before training"), (axes[1], after, "after training")]:
    ax.scatter(Z[:, 0], Z[:, 1], c=data.y, s=4, cmap="tab10", alpha=0.7)
    ax.set_title(f"GCN hidden layer, {name}")
    ax.set_xticks([]); ax.set_yticks([])
plt.show()
print("Look for: 7 clusters after training, with graph-neighborhoods sharing regions.")"""),

    md("""## 7 · Sparse $\\hat{\\mathbf{A}}$, and the price of dense  *(exercise 4)*

The lecture's §2 claims sparse propagation is $O(m)$ and dense is $O(n^2)$. Claims are
cheap, so you will measure this one.

**What to do:** in the next cell, implement `normalize_adjacency_sparse(edge_index,
num_nodes)` so that it builds $\\hat{\\mathbf{A}}$ **directly from the edge list** as a
`torch.sparse_coo_tensor` — no $n \\times n$ dense matrix is allowed anywhere in the
construction (the docstring lists the steps). The asserts compare it against your dense
builder on the path graph and on Cora; "exercise 4 ✓" confirms the match. The timing
cell after it (provided) then measures dense versus sparse propagation — run it and
read the printed comparison.
"""),

    todo("""def normalize_adjacency_sparse(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    \"\"\"Sparse A_hat = D̃^{-1/2} (A + I) D̃^{-1/2}, built straight from edge_index.

    Steps: append self-loop pairs (u,u) to the edge list; compute degrees of the
    self-looped graph with torch.bincount on the row index; per-edge value
    1/sqrt(deg[row] * deg[col]); assemble torch.sparse_coo_tensor and coalesce().
    \"\"\"
    ### BEGIN SOLUTION
    loops = torch.arange(num_nodes)
    idx = torch.cat([edge_index, torch.stack([loops, loops])], dim=1)
    deg = torch.bincount(idx[0], minlength=num_nodes).float()
    vals = (deg[idx[0]] * deg[idx[1]]).pow(-0.5)
    return torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes)).coalesce()
    ### END SOLUTION


# equivalence with the dense builder — on the path graph AND on Cora
assert torch.allclose(normalize_adjacency_sparse(path, 3).to_dense(), A_hat_path, atol=1e-6)
A_hat_sparse = normalize_adjacency_sparse(data.edge_index, data.num_nodes).to(device)
assert torch.allclose(A_hat_sparse.to_dense(), A_hat, atol=1e-5), (
    "sparse != dense on Cora — check: coalesce() called? degrees via bincount on the "
    "SELF-LOOPED row index (minlength=num_nodes)?"
)
print("exercise 4 ✓ — sparse construction matches the dense one")""",
         stub="""def normalize_adjacency_sparse(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    \"\"\"Sparse A_hat = D̃^{-1/2} (A + I) D̃^{-1/2}, built straight from edge_index.

    Steps: append self-loop pairs (u,u) to the edge list; compute degrees of the
    self-looped graph with torch.bincount on the row index; per-edge value
    1/sqrt(deg[row] * deg[col]); assemble torch.sparse_coo_tensor and coalesce().
    \"\"\"
    # TODO: implement (≈ 5 lines). No dense n×n tensor allowed anywhere here.
    raise NotImplementedError("implement normalize_adjacency_sparse")


# equivalence with the dense builder — on the path graph AND on Cora
assert torch.allclose(normalize_adjacency_sparse(path, 3).to_dense(), A_hat_path, atol=1e-6)
A_hat_sparse = normalize_adjacency_sparse(data.edge_index, data.num_nodes).to(device)
assert torch.allclose(A_hat_sparse.to_dense(), A_hat, atol=1e-5), (
    "sparse != dense on Cora — check: coalesce() called? degrees via bincount on the "
    "SELF-LOOPED row index (minlength=num_nodes)?"
)
print("exercise 4 ✓ — sparse construction matches the dense one")"""),

    code("""import time

H_probe = torch.randn(data.num_nodes, 16, device=device)
reps = 5 if SMOKE else 30

def bench(op):
    op()                                   # warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        op()
    return (time.perf_counter() - t0) / reps * 1e3

t_dense = bench(lambda: A_hat @ H_probe)
t_sparse = bench(lambda: torch.sparse.mm(A_hat_sparse, H_probe))
nnz = A_hat_sparse._nnz()
print(f"propagation on Cora, {reps} reps: dense {t_dense:.2f} ms  vs  sparse {t_sparse:.2f} ms")
print(f"dense touches n² = {data.num_nodes**2:,} entries; sparse touches nnz = {nnz:,} "
      f"({100 * nnz / data.num_nodes**2:.2f}% of them)")
print("On Cora both are fast — the point is the SCALING: n² vs m. Re-read lecture Q7 "
      "for what happens at n = 2.4M, where the dense operator would need ~23 TB.")"""),

    md("""## 8 · The SGC ablation: how much was the filter?

The lecture's honest remark: on homophilous benchmarks, the fixed low-pass filter does
most of the GCN's work. **Predict before you run:** within how many accuracy points of
your GCN will plain logistic regression on the precomputed features land — 15? 5? 1?
Write down your number before running the next cell. SGC
([Wu et al., 2019](https://arxiv.org/abs/1902.07153)) deletes every nonlinearity, so a
2-layer GCN collapses into logistic regression on the *precomputed* features
$\\hat{\\mathbf{A}}^2\\mathbf{X}$ — no message passing at training time at all.

There is nothing to implement in this section — the next cell is provided. The
algorithm below (mirroring the lecture's float; here $K = 2$) specifies exactly what
that cell computes; read them side by side:

**Input:** $\\mathbf{X}$; sparse $\\hat{\\mathbf{A}}$; depth $K$; labeled set
$\\mathcal{V}_{\\text{train}}$. **Output:** predictions for all nodes.

1. $\\bar{\\mathbf{X}} \\leftarrow \\mathbf{X}$
2. **for** $k = 1, \\dots, K$ **do**
3. &nbsp;&nbsp;&nbsp;&nbsp;$\\bar{\\mathbf{X}} \\leftarrow \\hat{\\mathbf{A}}\\,\\bar{\\mathbf{X}}$ — sparse multiply, once, offline
4. **end for**
5. fit logistic regression $\\boldsymbol{\\Theta}$ on the labeled rows of $\\bar{\\mathbf{X}}$
6. **return** $\\mathrm{softmax}(\\bar{\\mathbf{X}}\\boldsymbol{\\Theta})$

The code below is exactly this: line 1 of the cell is steps 1–4 ($K = 2$ unrolled), and
the `nn.Linear` trained with cross-entropy *is* step 5 — multinomial logistic regression.
"""),

    code("""X_sgc = A_hat_sparse @ (A_hat_sparse @ X)         # precompute once — this IS the "GNN"

sgc = nn.Linear(dataset.num_features, dataset.num_classes).to(device)
print("SGC (logistic regression on Â²X):")
sgc_acc = train(sgc, lambda m: m(X_sgc))

print(f"\\nMLP {mlp_acc:.3f}   SGC {sgc_acc:.3f}   GCN {gcn_acc:.3f}")
gap_structure = sgc_acc - mlp_acc
gap_nonlinear = gcn_acc - sgc_acc
print(f"the fixed filter bought {100 * gap_structure:.1f} pts; "
      f"the nonlinear network on top bought {100 * gap_nonlinear:.1f} pts")
floor = 0.55 if SMOKE else 0.72
assert sgc_acc >= floor, f"SGC accuracy {sgc_acc:.3f} below {floor} — check the precomputation"
print("SGC check ✓ — now answer reflection R3 below")"""),

    md("""## 9 · From scratch == the library

Last step of the ladder: hand your weights to PyG's `GCNConv` and demand the same
numbers. If this assertion passes, you have *implemented* the operator the entire
ecosystem ships — everything PyG adds is engineering (sparse kernels, caching,
mini-batching), not different math.
"""),

    code("""from torch_geometric.nn import GCNConv

d_in, d_out = 8, 5
X_small = torch.randn(3, d_in)

ours = GCNLayer(d_in, d_out)
theirs = GCNConv(d_in, d_out, bias=False)         # normalize=True + self-loops = our A_hat
with torch.no_grad():
    theirs.lin.weight.copy_(ours.W.weight)

out_ours = ours(X_small, A_hat_path)
out_theirs = theirs(X_small, path)

assert torch.allclose(out_ours, out_theirs, atol=1e-5), (
    f"max diff {(out_ours - out_theirs).abs().max():.2e} — check: GCNConv built with "
    f"bias=False? weights copied under no_grad BEFORE the forward pass?"
)
print("from-scratch == GCNConv ✓ (max diff",
      f"{(out_ours - out_theirs).abs().max().item():.2e})")"""),

    md("""## 10 · Stretch (optional, ungraded)

1. **The depth curve.** Generalize `GCN` to `n_layers ∈ {1,2,3,4,6}` and plot test
   accuracy vs depth (5 seeds each, error bars). You are reproducing Pitfall 2 — and the
   lecture's low-pass proposition predicts the shape before you run it.
2. **Change the graph, keep the code.** Swap `Planetoid(name="Cora")` for `"PubMed"`
   (19,717 nodes). Does the GCN-over-MLP gap grow or shrink? Time the sparse vs dense
   propagation again — dense should now visibly hurt.

## 11 · Reflection (answer in this cell, 2–4 sentences each)

**R1.** Your GCN used 2,568 unlabeled nodes at training time without ever seeing their
labels. Point to the exact line of code through which they influence the loss.

**R2.** The MLP baseline transfers to a brand-new paper with no citations; your GCN, as
written, does not handle it gracefully. What exactly breaks, and which Week 7 idea fixes it?

**R3.** Interpret your §8 numbers: if SGC lands close to the GCN, what — concretely —
did the ReLU and the trained first layer contribute on Cora, and what kind of dataset
would you expect to *widen* the GCN–SGC gap? (Hint: the lecture's heterophily remark.)

*(your answers here)*

## What to submit

Moodle expects one `.ipynb` with: all four exercise checks printing ✓, the training
cell's assertions passing, the PCA figure rendered, the timing measurements printed,
the SGC check ✓, the `GCNConv` equivalence ✓, and all three reflection answers filled
in. Grading: assertions 70% · reflections 30%. Run *Runtime → Restart and run all*
before submitting — a notebook that doesn't execute top-to-bottom scores 0.

**AI policy reminder** (course honor code): AI assistants are allowed for this lab *with
disclosure* — add a line here naming any tools you used and for what. You must be able to
explain any line of your submission on request; undeclared use or inability to explain is
a violation.
"""),
]


def build(student: bool) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    })
    for kind, text, stub in CELLS:
        if kind == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(text))
            continue
        if stub is None:
            nb.cells.append(nbf.v4.new_code_cell(text))
            continue
        if student:
            cell = nbf.v4.new_code_cell(stub)
            cell.metadata["tags"] = ["student-todo"]
        else:
            cell = nbf.v4.new_code_cell(text)
        nb.cells.append(cell)
    return nb


def main() -> None:
    student = build(student=True)
    solution = build(student=False)
    # sanity: no solution markers may survive into the student build
    for cell in student.cells:
        assert "BEGIN SOLUTION" not in cell.source, "solution leak into student notebook"
    STUDENT_OUT.parent.mkdir(parents=True, exist_ok=True)
    SOLUTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(student, STUDENT_OUT)
    nbf.write(solution, SOLUTION_OUT)
    print("wrote", STUDENT_OUT)
    print("wrote", SOLUTION_OUT)


if __name__ == "__main__":
    main()
