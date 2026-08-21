#!/usr/bin/env python
"""Generate Lab 7 (the GNN design space: SAGE/GAT/GIN by hand and in PyG,
plus the full ablation grid with seeds and error bars) — student + solution.

    python scripts/labgen/make_lab07.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "labs" / "lab07_design.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "labs" / "lab07_design.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(text, stub): return ("code", text, stub)


CELLS = [
    md("""# Lab 7 · The design space — SAGE, GAT, GIN, and an honest ablation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/labs/lab07_design.ipynb)

**Week 7 · [lecture](https://lukmanovr.github.io/dkr/lectures/07-gnn-design.html) · ≈ 30 min of compute (free Colab; a GPU runtime speeds the grid but is not required)**

You verify the lecture's hand computations against real layers, build one layer from
PyG's `MessagePassing` template so the framework stops being magic, reproduce the
lecture's measured ablation grid — 16 configurations, 3 seeds, error bars — and end
by scaling two contenders to ogbn-arxiv, where Cora's lessons start to bend.

### Goals
1. Hand-compute SAGE, GIN, and GAT updates and assert them against the formulas.
2. Implement a SAGE layer from `MessagePassing` and match it to the hand math.
3. Demonstrate the aggregator-confusion propositions in code, including the real-valued asterisk.
4. Run the ablation grid with the lecture's protocol and write a claims paragraph that cites its cells.
5. Measure what changes at 60× Cora on ogbn-arxiv.
"""),

    md("""## 0 · Setup"""),

    code("""import os, sys, random, statistics, itertools, time

SMOKE = os.environ.get("SMOKE", "") == "1"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    %pip install -q torch_geometric==2.8.0.post1 ogb==1.3.6

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GINConv, MessagePassing

SEED = 7
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} · device {DEV} — environment OK")"""),

    md("""## 1 · The lecture's arithmetic, asserted  *(exercise 1)*

Three updates, computed the way the lecture computed them. If you did the reading,
each function is a transcription; the asserts hold you to the exact numbers.
"""),

    todo("""def sage_update(h_self, H_nbrs, w_self, w_nbr):
    \"\"\"Scalar SAGE-mean update, identity activation:
    w_self * h_self + w_nbr * mean(H_nbrs).\"\"\"
    ### BEGIN SOLUTION
    return w_self * h_self + w_nbr * (sum(H_nbrs) / len(H_nbrs))
    ### END SOLUTION


def gin_update(h_self, H_nbrs, eps=0.0):
    \"\"\"Scalar GIN update with identity MLP: (1+eps)*h_self + sum(H_nbrs).\"\"\"
    ### BEGIN SOLUTION
    return (1 + eps) * h_self + sum(H_nbrs)
    ### END SOLUTION


def gat_alphas(scores):
    \"\"\"Softmax over a list of raw attention scores.\"\"\"
    ### BEGIN SOLUTION
    e = [pow(2.718281828459045, s) for s in scores]
    z = sum(e)
    return [x / z for x in e]
    ### END SOLUTION


# — SAGE: the lecture's node C (h=2, neighbors {1,3,4,0}, W=[1, 1/2])
assert abs(sage_update(2, [1, 3, 4, 0], 1.0, 0.5) - 3.0) < 1e-9, (
    "mean(1,3,4,0)=2; 1·2 + ½·2 = 3 — the lecture's first hand computation"
)
assert abs(sage_update(2, [1, 3, 4], 1.0, 0.5) - 10 / 3) < 1e-9, (
    "drop F: mean(1,3,4)=8/3; 2 + 4/3 = 10/3 — the neighborhood moved, the self-term did not"
)

# — GIN: the lecture's Q2(b): h=(1,1) scalarized per-dim → self + sum
assert abs(gin_update(1, [2, 0]) - 3.0) < 1e-9 and abs(gin_update(1, [0, 2]) - 3.0) < 1e-9, (
    "GIN with ε=0, identity MLP: self + sum, per coordinate → (3,3) on Q2's toy"
)

# — GAT: the lecture's worked head — scores {3, 3, 1} over {self, u1, u2}
alphas = gat_alphas([3, 3, 1])
assert abs(sum(alphas) - 1.0) < 1e-9, "attention weights must sum to 1 (softmax)"
assert abs(alphas[0] - alphas[1]) < 1e-9, "equal scores → equal weights"
assert alphas[2] < 0.07 and alphas[0] > 0.46, (
    f"softmax over (3,3,1) ≈ (0.47, 0.47, 0.06); got {[round(a, 3) for a in alphas]} — "
    f"the content-mismatched neighbor is nearly muted"
)
print("exercise 1 ✓ — SAGE separates, GIN counts, GAT mutes: the lecture's numbers, verified")""",
         stub="""def sage_update(h_self, H_nbrs, w_self, w_nbr):
    \"\"\"Scalar SAGE-mean update, identity activation:
    w_self * h_self + w_nbr * mean(H_nbrs).\"\"\"
    # TODO: one line.
    raise NotImplementedError


def gin_update(h_self, H_nbrs, eps=0.0):
    \"\"\"Scalar GIN update with identity MLP: (1+eps)*h_self + sum(H_nbrs).\"\"\"
    # TODO: one line.
    raise NotImplementedError


def gat_alphas(scores):
    \"\"\"Softmax over a list of raw attention scores.\"\"\"
    # TODO: exponentiate, normalize. 3 lines.
    raise NotImplementedError


# — SAGE: the lecture's node C (h=2, neighbors {1,3,4,0}, W=[1, 1/2])
assert abs(sage_update(2, [1, 3, 4, 0], 1.0, 0.5) - 3.0) < 1e-9, (
    "mean(1,3,4,0)=2; 1·2 + ½·2 = 3 — the lecture's first hand computation"
)
assert abs(sage_update(2, [1, 3, 4], 1.0, 0.5) - 10 / 3) < 1e-9, (
    "drop F: mean(1,3,4)=8/3; 2 + 4/3 = 10/3 — the neighborhood moved, the self-term did not"
)

# — GIN: the lecture's Q2(b): h=(1,1) scalarized per-dim → self + sum
assert abs(gin_update(1, [2, 0]) - 3.0) < 1e-9 and abs(gin_update(1, [0, 2]) - 3.0) < 1e-9, (
    "GIN with ε=0, identity MLP: self + sum, per coordinate → (3,3) on Q2's toy"
)

# — GAT: the lecture's worked head — scores {3, 3, 1} over {self, u1, u2}
alphas = gat_alphas([3, 3, 1])
assert abs(sum(alphas) - 1.0) < 1e-9, "attention weights must sum to 1 (softmax)"
assert abs(alphas[0] - alphas[1]) < 1e-9, "equal scores → equal weights"
assert alphas[2] < 0.07 and alphas[0] > 0.46, (
    f"softmax over (3,3,1) ≈ (0.47, 0.47, 0.06); got {[round(a, 3) for a in alphas]} — "
    f"the content-mismatched neighbor is nearly muted"
)
print("exercise 1 ✓ — SAGE separates, GIN counts, GAT mutes: the lecture's numbers, verified")"""),

    md("""## 2 · One layer from the template  *(exercise 2 — skill: MessagePassing)*

The lecture's claim: every architecture is `message()` + an `aggr` + `update()`.
Prove it to yourself by building SAGE-mean from the base class and matching it, to
six decimals, against the same computation done with dense matrices.
"""),

    todo("""class MiniSAGE(MessagePassing):
    \"\"\"SAGE-mean, from the template: h'_v = W_self h_v + W_nbr · mean_{u∈N(v)} h_u.
    (No activation — we test the linear algebra.)\"\"\"
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="mean")
        self.lin_self = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.lin_nbr = torch.nn.Linear(in_dim, out_dim, bias=False)

    ### BEGIN SOLUTION
    def forward(self, x, edge_index):
        agg = self.propagate(edge_index, x=x)
        return self.lin_self(x) + self.lin_nbr(agg)

    def message(self, x_j):
        return x_j
    ### END SOLUTION


# a 4-node path 0-1-2-3, 2-dim features; both edge directions
ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [1.0, 1.0]])
layer = MiniSAGE(2, 2)
with torch.no_grad():
    layer.lin_self.weight.copy_(torch.eye(2))
    layer.lin_nbr.weight.copy_(0.5 * torch.eye(2))

out = layer(X, ei)
# dense reference: neighbor means computed by hand from the path structure
mean_nbrs = torch.tensor([[0.0, 1.0],            # N(0)={1}
                          [1.5, 1.0],            # N(1)={0,2}
                          [0.5, 1.0],            # N(2)={1,3}
                          [2.0, 2.0]])           # N(3)={2}
expected = X + 0.5 * mean_nbrs
assert torch.allclose(out, expected, atol=1e-6), (
    f"MiniSAGE disagrees with the dense computation.\\ngot:\\n{out}\\nexpected:\\n{expected}\\n"
    f"Checks: message() should return x_j (the SENDER); propagate handles the scatter; "
    f"forward combines lin_self(x) with lin_nbr(aggregate)."
)
print("exercise 2 ✓ — the template is now something you have built, not something you import")""",
         stub="""class MiniSAGE(MessagePassing):
    \"\"\"SAGE-mean, from the template: h'_v = W_self h_v + W_nbr · mean_{u∈N(v)} h_u.
    (No activation — we test the linear algebra.)\"\"\"
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr="mean")
        self.lin_self = torch.nn.Linear(in_dim, out_dim, bias=False)
        self.lin_nbr = torch.nn.Linear(in_dim, out_dim, bias=False)

    # TODO: forward(x, edge_index): agg = self.propagate(edge_index, x=x);
    #       return lin_self(x) + lin_nbr(agg).
    #       message(x_j): return x_j  — the sender's features, untouched.
    raise NotImplementedError("implement forward() and message()")


# a 4-node path 0-1-2-3, 2-dim features; both edge directions
ei = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
X = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0], [1.0, 1.0]])
layer = MiniSAGE(2, 2)
with torch.no_grad():
    layer.lin_self.weight.copy_(torch.eye(2))
    layer.lin_nbr.weight.copy_(0.5 * torch.eye(2))

out = layer(X, ei)
# dense reference: neighbor means computed by hand from the path structure
mean_nbrs = torch.tensor([[0.0, 1.0],            # N(0)={1}
                          [1.5, 1.0],            # N(1)={0,2}
                          [0.5, 1.0],            # N(2)={1,3}
                          [2.0, 2.0]])           # N(3)={2}
expected = X + 0.5 * mean_nbrs
assert torch.allclose(out, expected, atol=1e-6), (
    f"MiniSAGE disagrees with the dense computation.\\ngot:\\n{out}\\nexpected:\\n{expected}\\n"
    f"Checks: message() should return x_j (the SENDER); propagate handles the scatter; "
    f"forward combines lin_self(x) with lin_nbr(aggregate)."
)
print("exercise 2 ✓ — the template is now something you have built, not something you import")"""),

    md("""## 3 · The aggregator propositions, in code  *(exercise 3)*

The lecture's @prp-agg with tensors instead of chalk — including the real-valued
asterisk that the widget's standing challenge hinted at.
"""),

    todo("""def confusion_check(A: torch.Tensor, B: torch.Tensor) -> dict:
    \"\"\"A, B: (n_i, d) neighbor feature stacks. Return which aggregators are
    CONFUSED (produce identical outputs): {'mean': bool, 'max': bool, 'sum': bool}.\"\"\"
    ### BEGIN SOLUTION
    return {
        "mean": torch.allclose(A.mean(0), B.mean(0)),
        "max": torch.allclose(A.max(0).values, B.max(0).values),
        "sum": torch.allclose(A.sum(0), B.sum(0)),
    }
    ### END SOLUTION


t = torch.tensor([1.0, 0.0]); gld = torch.tensor([0.0, 1.0])

case1 = confusion_check(torch.stack([t]), torch.stack([t, t]))
assert case1 == {"mean": True, "max": True, "sum": False}, (
    f"{{t}} vs {{t,t}}: mean and max confused, sum separates — got {case1}"
)
case3 = confusion_check(torch.stack([t, t, gld]), torch.stack([t, gld, gld]))
assert case3 == {"mean": False, "max": True, "sum": False}, (
    f"{{t,t,g}} vs {{t,g,g}}: only max is confused — got {case3}"
)

# the asterisk: with REAL-VALUED scalars, even sum collides —
scalar_case = confusion_check(torch.tensor([[1.0], [3.0]]), torch.tensor([[2.0], [2.0]]))
assert scalar_case["sum"] and scalar_case["mean"], (
    "{1,3} vs {2,2}: sum 4 = 4 and mean 2 = 2 — sum's injectivity is a ONE-HOT/countable "
    "story; with raw reals it needs a learned embedding first (that is why GIN has an MLP)"
)
assert not scalar_case["max"], "max separates {1,3} from {2,2} (3 ≠ 2) — no aggregator dominates"
print("exercise 3 ✓ — sum wins on one-hots, needs its MLP on reals; the propositions hold in float32")""",
         stub="""def confusion_check(A: torch.Tensor, B: torch.Tensor) -> dict:
    \"\"\"A, B: (n_i, d) neighbor feature stacks. Return which aggregators are
    CONFUSED (produce identical outputs): {'mean': bool, 'max': bool, 'sum': bool}.\"\"\"
    # TODO: three torch.allclose comparisons. 5 lines.
    raise NotImplementedError


t = torch.tensor([1.0, 0.0]); gld = torch.tensor([0.0, 1.0])

case1 = confusion_check(torch.stack([t]), torch.stack([t, t]))
assert case1 == {"mean": True, "max": True, "sum": False}, (
    f"{{t}} vs {{t,t}}: mean and max confused, sum separates — got {case1}"
)
case3 = confusion_check(torch.stack([t, t, gld]), torch.stack([t, gld, gld]))
assert case3 == {"mean": False, "max": True, "sum": False}, (
    f"{{t,t,g}} vs {{t,g,g}}: only max is confused — got {case3}"
)

# the asterisk: with REAL-VALUED scalars, even sum collides —
scalar_case = confusion_check(torch.tensor([[1.0], [3.0]]), torch.tensor([[2.0], [2.0]]))
assert scalar_case["sum"] and scalar_case["mean"], (
    "{1,3} vs {2,2}: sum 4 = 4 and mean 2 = 2 — sum's injectivity is a ONE-HOT/countable "
    "story; with raw reals it needs a learned embedding first (that is why GIN has an MLP)"
)
assert not scalar_case["max"], "max separates {1,3} from {2,2} (3 ≠ 2) — no aggregator dominates"
print("exercise 3 ✓ — sum wins on one-hots, needs its MLP on reals; the propositions hold in float32")"""),

    md("""## 4 · The ablation grid  *(exercise 4 — the lecture's Figure 4, reproduced)*

The training harness is provided and is exactly the lecture's protocol: hidden 64,
dropout 0.5, Adam(0.01, wd 5e-4), early stopping on validation (patience 30), test
accuracy at the best validation epoch. Your part is the statistics: turn raw runs
into the mean ± std table that claims are made of.

Under SMOKE the grid shrinks (2 architectures × 2 seeds, few epochs); run the full
16 × 3 grid before submitting — it is ≈ 8 minutes on CPU, less on a GPU runtime.
""" ),

    todo("""data = Planetoid(root="data/Planetoid", name="Cora")[0].to(DEV)

CONVS = {
    "GCN": lambda i, o: GCNConv(i, o),
    "SAGE": lambda i, o: SAGEConv(i, o),
    "GAT": lambda i, o: GATConv(i, o, heads=4, concat=False),
    "GIN": lambda i, o: GINConv(torch.nn.Sequential(
        torch.nn.Linear(i, o), torch.nn.ReLU(), torch.nn.Linear(o, o))),
}


class Net(torch.nn.Module):
    def __init__(self, arch, depth, residual, hidden=64):
        super().__init__()
        self.residual = residual
        dims = [data.num_features] + [hidden] * depth
        self.convs = torch.nn.ModuleList(
            [CONVS[arch](dims[i], dims[i + 1]) for i in range(depth)])
        self.out = torch.nn.Linear(hidden, 7)

    def forward(self, x, ei):
        for conv in self.convs:
            h = F.dropout(F.relu(conv(x, ei)), 0.5, self.training)
            x = h + x if (self.residual and h.shape == x.shape) else h
        return self.out(x)


def run(arch, depth, residual, seed, max_epochs=None):
    torch.manual_seed(seed)
    net = Net(arch, depth, residual).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_test, patience = 0.0, 0.0, 0
    for ep in range(max_epochs or (40 if SMOKE else 200)):
        net.train(); opt.zero_grad()
        out = net(data.x, data.edge_index)
        F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(data.x, data.edge_index).argmax(1)
            val = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        if val > best_val:
            best_val, best_test, patience = val, test, 0
        else:
            patience += 1
            if patience >= 30:
                break
    return best_test


def summarize(runs: dict) -> dict:
    \"\"\"runs: {(arch, depth, residual): [acc, acc, ...]} with accs in [0,1].
    Return {config: {'mean': float, 'std': float}} in PERCENT, rounded to 1 dp.
    std is the sample std (statistics.stdev); a single run gets std 0.0.\"\"\"
    ### BEGIN SOLUTION
    out = {}
    for cfg, accs in runs.items():
        out[cfg] = {
            "mean": round(100 * statistics.mean(accs), 1),
            "std": round(100 * statistics.stdev(accs), 1) if len(accs) > 1 else 0.0,
        }
    return out
    ### END SOLUTION


toy = summarize({("GCN", 2, False): [0.80, 0.81, 0.82], ("GAT", 4, False): [0.20, 0.30]})
assert toy[("GCN", 2, False)] == {"mean": 81.0, "std": 1.0}, (
    f"mean of (80,81,82)=81.0, sample std=1.0 — got {toy[('GCN', 2, False)]}"
)
assert toy[("GAT", 4, False)]["std"] == 7.1, "sample std of (20,30) = 7.07 → 7.1"

ARCHS = ["GCN", "GAT"] if SMOKE else ["GCN", "SAGE", "GAT", "GIN"]
SEEDS = [0, 1] if SMOKE else [0, 1, 2]
grid = {}
t0 = time.time()
for arch, depth, residual in itertools.product(ARCHS, [2, 4], [False, True]):
    grid[(arch, depth, residual)] = [run(arch, depth, residual, s) for s in SEEDS]
    m = statistics.mean(grid[(arch, depth, residual)])
    print(f"  {arch:>4} · {depth} layers · {'res  ' if residual else 'plain'} → "
          f"{100 * m:5.1f}%   [{time.time() - t0:4.0f}s]")
table = summarize(grid)

gat4_plain = table[("GAT", 4, False)]["mean"]
gat4_res = table[("GAT", 4, True)]["mean"]
print(f"\\nGAT depth-4: {gat4_plain}% plain → {gat4_res}% with residuals")
assert gat4_res - gat4_plain > (10 if SMOKE else 20), (
    "the residual rescue should be dramatic at depth 4 — if not, check that the "
    "residual branch adds h + x only when shapes match (i.e., from layer 2 on)"
)
print("exercise 4 ✓ — you have reproduced the lecture's headline finding with your own error bars")""",
         stub="""data = Planetoid(root="data/Planetoid", name="Cora")[0].to(DEV)

CONVS = {
    "GCN": lambda i, o: GCNConv(i, o),
    "SAGE": lambda i, o: SAGEConv(i, o),
    "GAT": lambda i, o: GATConv(i, o, heads=4, concat=False),
    "GIN": lambda i, o: GINConv(torch.nn.Sequential(
        torch.nn.Linear(i, o), torch.nn.ReLU(), torch.nn.Linear(o, o))),
}


class Net(torch.nn.Module):
    def __init__(self, arch, depth, residual, hidden=64):
        super().__init__()
        self.residual = residual
        dims = [data.num_features] + [hidden] * depth
        self.convs = torch.nn.ModuleList(
            [CONVS[arch](dims[i], dims[i + 1]) for i in range(depth)])
        self.out = torch.nn.Linear(hidden, 7)

    def forward(self, x, ei):
        for conv in self.convs:
            h = F.dropout(F.relu(conv(x, ei)), 0.5, self.training)
            x = h + x if (self.residual and h.shape == x.shape) else h
        return self.out(x)


def run(arch, depth, residual, seed, max_epochs=None):
    torch.manual_seed(seed)
    net = Net(arch, depth, residual).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_test, patience = 0.0, 0.0, 0
    for ep in range(max_epochs or (40 if SMOKE else 200)):
        net.train(); opt.zero_grad()
        out = net(data.x, data.edge_index)
        F.cross_entropy(out[data.train_mask], data.y[data.train_mask]).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(data.x, data.edge_index).argmax(1)
            val = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()
            test = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
        if val > best_val:
            best_val, best_test, patience = val, test, 0
        else:
            patience += 1
            if patience >= 30:
                break
    return best_test


def summarize(runs: dict) -> dict:
    \"\"\"runs: {(arch, depth, residual): [acc, acc, ...]} with accs in [0,1].
    Return {config: {'mean': float, 'std': float}} in PERCENT, rounded to 1 dp.
    std is the sample std (statistics.stdev); a single run gets std 0.0.\"\"\"
    # TODO: statistics.mean / statistics.stdev, ×100, round(…, 1). ~7 lines.
    raise NotImplementedError


toy = summarize({("GCN", 2, False): [0.80, 0.81, 0.82], ("GAT", 4, False): [0.20, 0.30]})
assert toy[("GCN", 2, False)] == {"mean": 81.0, "std": 1.0}, (
    f"mean of (80,81,82)=81.0, sample std=1.0 — got {toy[('GCN', 2, False)]}"
)
assert toy[("GAT", 4, False)]["std"] == 7.1, "sample std of (20,30) = 7.07 → 7.1"

ARCHS = ["GCN", "GAT"] if SMOKE else ["GCN", "SAGE", "GAT", "GIN"]
SEEDS = [0, 1] if SMOKE else [0, 1, 2]
grid = {}
t0 = time.time()
for arch, depth, residual in itertools.product(ARCHS, [2, 4], [False, True]):
    grid[(arch, depth, residual)] = [run(arch, depth, residual, s) for s in SEEDS]
    m = statistics.mean(grid[(arch, depth, residual)])
    print(f"  {arch:>4} · {depth} layers · {'res  ' if residual else 'plain'} → "
          f"{100 * m:5.1f}%   [{time.time() - t0:4.0f}s]")
table = summarize(grid)

gat4_plain = table[("GAT", 4, False)]["mean"]
gat4_res = table[("GAT", 4, True)]["mean"]
print(f"\\nGAT depth-4: {gat4_plain}% plain → {gat4_res}% with residuals")
assert gat4_res - gat4_plain > (10 if SMOKE else 20), (
    "the residual rescue should be dramatic at depth 4 — if not, check that the "
    "residual branch adds h + x only when shapes match (i.e., from layer 2 on)"
)
print("exercise 4 ✓ — you have reproduced the lecture's headline finding with your own error bars")"""),

    md("""### Your claims paragraph *(graded — write it in this cell)*

Using ONLY your table above, write 2–4 claims in the lecture's three-move format:
**sentence → cells → scope**. At least one claim must be a *negative* finding
("indistinguishable under this protocol"). Claims that cite no cells score zero.

*(your claims here)*
"""),

    md("""## 5 · What changes at 60× Cora  *(exercise 5 — ogbn-arxiv)*

ogbn-arxiv: 169,343 papers, 1.16M citations, a **time-based split** (train on
pre-2017 papers, validate on 2017, test on 2018+ — the honest split discipline from
Week 1, institutionalized). The download is ~80 MB on first run.

**Predict before you run:** Cora's grid said GCN ≥ SAGE (within noise). Does that
hold at 60× the size, under a time split, with 40 classes?
"""),

    todo("""from ogb.nodeproppred import PygNodePropPredDataset

# ogb 1.3.6 predates torch>=2.6's weights_only load default; the cache is OGB's own file
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
arxiv = PygNodePropPredDataset("ogbn-arxiv", root="data/ogb")
torch.load = _load
split = arxiv.get_idx_split()
adata = arxiv[0]
adata.edge_index = torch.cat(
    [adata.edge_index, adata.edge_index.flip(0)], dim=1)   # arxiv ships directed; symmetrize
adata.y = adata.y.squeeze()

if SMOKE:
    keep = torch.zeros(adata.num_nodes, dtype=torch.bool)
    keep[:30000] = True
    adata = adata.subgraph(keep)
    for k in split:
        split[k] = split[k][split[k] < 30000]
adata = adata.to(DEV)
tr_idx, va_idx, te_idx = split["train"].to(DEV), split["valid"].to(DEV), split["test"].to(DEV)
print(f"arxiv: {adata.num_nodes:,} nodes · {adata.edge_index.shape[1]:,} directed edges · 40 classes")


class ArxivNet(torch.nn.Module):
    def __init__(self, arch, hidden=128, depth=2):
        super().__init__()
        dims = [adata.num_features] + [hidden] * depth
        self.convs = torch.nn.ModuleList(
            [CONVS[arch](dims[i], dims[i + 1]) for i in range(depth)])
        self.norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(hidden) for _ in range(depth)])
        self.out = torch.nn.Linear(hidden, 40)

    def forward(self, x, ei):
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(F.relu(norm(conv(x, ei))), 0.3, self.training)
        return self.out(x)


def run_arxiv(arch, seed):
    \"\"\"Train full-batch on arxiv; return (val_acc, test_acc) at best val epoch.\"\"\"
    ### BEGIN SOLUTION
    torch.manual_seed(seed)
    net = ArxivNet(arch).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    best_val, best_test = 0.0, 0.0
    for ep in range(15 if SMOKE else 60):
        net.train(); opt.zero_grad()
        out = net(adata.x, adata.edge_index)
        F.cross_entropy(out[tr_idx], adata.y[tr_idx]).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(adata.x, adata.edge_index).argmax(1)
            val = (pred[va_idx] == adata.y[va_idx]).float().mean().item()
            test = (pred[te_idx] == adata.y[te_idx]).float().mean().item()
        if val > best_val:
            best_val, best_test = val, test
    return best_val, best_test
    ### END SOLUTION


results_ax = {}
for arch in ["GCN", "SAGE"]:
    v, t = run_arxiv(arch, seed=0)
    results_ax[arch] = (v, t)
    print(f"  {arch}: val {100 * v:.1f}% · test {100 * t:.1f}%")

floor = 0.35 if SMOKE else 0.60
for arch, (v, t) in results_ax.items():
    assert t > floor, (
        f"{arch} test accuracy {100 * t:.1f}% under the floor — check that you symmetrized "
        f"edges, squeezed y, and evaluated at the best-VALIDATION epoch (not the last)"
    )
print("exercise 5 ✓ — same models, different world: bigger graph, honest time split, tighter error bars")
print("note the val-vs-test gap: the future is harder than the past — that IS the time split working")""",
         stub="""from ogb.nodeproppred import PygNodePropPredDataset

# ogb 1.3.6 predates torch>=2.6's weights_only load default; the cache is OGB's own file
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
arxiv = PygNodePropPredDataset("ogbn-arxiv", root="data/ogb")
torch.load = _load
split = arxiv.get_idx_split()
adata = arxiv[0]
adata.edge_index = torch.cat(
    [adata.edge_index, adata.edge_index.flip(0)], dim=1)   # arxiv ships directed; symmetrize
adata.y = adata.y.squeeze()

if SMOKE:
    keep = torch.zeros(adata.num_nodes, dtype=torch.bool)
    keep[:30000] = True
    adata = adata.subgraph(keep)
    for k in split:
        split[k] = split[k][split[k] < 30000]
adata = adata.to(DEV)
tr_idx, va_idx, te_idx = split["train"].to(DEV), split["valid"].to(DEV), split["test"].to(DEV)
print(f"arxiv: {adata.num_nodes:,} nodes · {adata.edge_index.shape[1]:,} directed edges · 40 classes")


class ArxivNet(torch.nn.Module):
    def __init__(self, arch, hidden=128, depth=2):
        super().__init__()
        dims = [adata.num_features] + [hidden] * depth
        self.convs = torch.nn.ModuleList(
            [CONVS[arch](dims[i], dims[i + 1]) for i in range(depth)])
        self.norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(hidden) for _ in range(depth)])
        self.out = torch.nn.Linear(hidden, 40)

    def forward(self, x, ei):
        for conv, norm in zip(self.convs, self.norms):
            x = F.dropout(F.relu(norm(conv(x, ei))), 0.3, self.training)
        return self.out(x)


def run_arxiv(arch, seed):
    \"\"\"Train full-batch on arxiv; return (val_acc, test_acc) at best val epoch.\"\"\"
    # TODO: mirror exercise 4's run(): seed, ArxivNet(arch), Adam(0.01),
    # (15 if SMOKE else 60) epochs, cross-entropy on tr_idx, track best-val
    # test accuracy. ~18 lines.
    raise NotImplementedError


results_ax = {}
for arch in ["GCN", "SAGE"]:
    v, t = run_arxiv(arch, seed=0)
    results_ax[arch] = (v, t)
    print(f"  {arch}: val {100 * v:.1f}% · test {100 * t:.1f}%")

floor = 0.35 if SMOKE else 0.60
for arch, (v, t) in results_ax.items():
    assert t > floor, (
        f"{arch} test accuracy {100 * t:.1f}% under the floor — check that you symmetrized "
        f"edges, squeezed y, and evaluated at the best-VALIDATION epoch (not the last)"
    )
print("exercise 5 ✓ — same models, different world: bigger graph, honest time split, tighter error bars")
print("note the val-vs-test gap: the future is harder than the past — that IS the time split working")"""),

    md("""## 6 · Stretch (optional, ungraded)

1. **The normalization axis.** Extend exercise 4's grid with ±BatchNorm at depth 4.
   Does it substitute for residuals, complement them, or neither? Cite cells.
2. **DropEdge vs dropout.** Add DropEdge (`torch_geometric.utils.dropout_edge`) to
   the depth-4 GCN and ablate it against feature dropout — separately and together.
3. **Jumping knowledge.** Concatenate all layer outputs before the classifier in the
   depth-4 models. Which architecture does JK help most, and does that match the
   oversmoothing story?
4. **GATv2.** Swap `GATConv` → `GATv2Conv` across the grid. On homophilous Cora, does
   dynamic attention earn its keep? (Predict first, from the lecture's §3.)

## 7 · Reflection (answer in this cell, 2–4 sentences each)

**R1.** Your grid's best cell and the lecture's differ by fractions of a point despite
identical protocols. Name two legitimate sources of that difference and say why
neither undermines the shared conclusions.

**R2.** The GAT-at-depth-4 crash has a mechanism story (§5 of the lecture) and a
number (your table). Write the two-sentence version you would tell a teammate who
wants to "just make it deeper."

**R3.** On arxiv, the val→test drop is larger than anything Cora showed you. What
does the time split measure that Cora's random split cannot, and which
production-deployment fact does it rehearse?

*(your answers here)*

## What to submit

One executed notebook on Moodle: all five exercise checks ✓, the **full** 16×3 grid
(not the SMOKE grid), your claims paragraph, and the three reflections. Grading:
assertions 60% · claims paragraph 15% · reflections 25%. Run *Runtime → Restart and
run all* before submitting.

**AI policy reminder** (course honor code): AI assistants are allowed for this lab
*with disclosure* — add a line here naming any tools you used and for what. You must
be able to explain any line of your submission on request; undeclared use or
inability to explain is a violation.
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
