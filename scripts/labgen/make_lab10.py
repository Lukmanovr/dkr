#!/usr/bin/env python
"""Generate Lab 10 (heterogeneous graphs: hand R-GCN updates and the parameter
bill, DBLP typed node classification, inverse channels, and the encoder-vs-
lookup showdown on FB15k-237) — student + solution notebooks.

    python scripts/labgen/make_lab10.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "labs" / "lab10_hetero.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "labs" / "lab10_hetero.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(solution, stub): return ("code", solution, stub)


CELLS = [
    md("""# Lab 10 · Typed message passing: R-GCN by hand, on DBLP, and in a fair showdown

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/labs/lab10_hetero.ipynb)

**Week 10 · [lecture](https://lukmanovr.github.io/dkr/lectures/10-hetero-rgcn.html) · ≈ 15 min of compute — a GPU runtime is recommended for exercise 4**

You hand-compute the lecture's R-GCN updates and the parameter bill; train a typed
classifier on the real DBLP graph and beat the feature-only baseline; build the
inverse-channel construction that every R-GCN implementation silently depends on;
and reproduce the encoder-vs-lookup showdown on FB15k-237 with Lab 4's exact
protocol — landing within tolerance of the lecture's table.

The method under test is R-GCN — relational message passing with one weight per
typed channel, introduced by [Schlichtkrull et al., 2018](https://arxiv.org/abs/1703.06103)
— and the skepticism you will apply to it is the benchmark discipline of the HGB
study, [Lv et al., 2021](https://arxiv.org/abs/2112.14936): typed machinery is a
claim, and claims get measured against matched baselines.

### Goals
1. Compute typed updates and parameter counts by hand, asserted to the lecture.
2. Train a heterogeneous GNN on DBLP and measure what the typed graph is worth.
3. Implement the inverse-channel edge construction, and prove it is not optional.
4. Run the showdown: shallow DistMult vs R-GCN+DistMult under an identical
   training budget and evaluation protocol.
"""),

    md("""## 0 · Setup"""),

    code("""import os, sys, statistics

SMOKE = os.environ.get("SMOKE", "") == "1"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    %pip install -q torch_geometric==2.8.0.post1

import torch
import torch.nn.functional as F
from torch_geometric.datasets import DBLP, FB15k_237
from torch_geometric.nn import HeteroConv, SAGEConv, Linear, RGCNConv

SEED = 10
torch.manual_seed(SEED)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} · device {DEV} — environment OK")
if DEV.type != "cuda":
    print("(no GPU: exercise 4's encoder will be slow but correct — consider Runtime → Change runtime type)")"""),

    md("""## 1 · Typed updates and the parameter bill  *(exercise 1)*

**What you will implement**, in the next code cell:

1. `rgcn_update(h_self, channels, w_self=1.0)` — the scalar R-GCN update for a
   single node. `channels` is a list of `(w_r, [h_u, ...])` pairs; return
   `w_self * h_self` plus, for each channel, `w_r` times the mean of that
   channel's neighbor values.
2. `params_full(R, d)` — the per-layer parameter count with one full d×d weight
   matrix per relation direction plus the self-loop, i.e. (2R+1) matrices.
3. `params_basis(R, d, B)` — the per-layer parameter count under basis
   decomposition: B basis matrices of d², 2R·B mixing coefficients, and the
   undecomposed self-loop d².

**How you will know it worked:** the asserts in the same cell compare your
outputs to the lecture's worked examples (5.25 for node P1, 3.0 for A2, 2.0 for
the untyped collapse, 4,750,000 and 324,220 for the parameter counts). When
they all pass, the cell prints "exercise 1 ✓".

The algorithm below specifies exactly what your `rgcn_update` implementation
must do; follow it step by step:

> **One R-GCN layer** — **Input:** features $h_u$ for all nodes; relations with
> inverse channels; weights $w_\\text{self}, \\{w_r\\}$. **Output:** updated $h'_v$.
>
> 1. **for** each node $v$ **do**
> 2. &nbsp;&nbsp;&nbsp;&nbsp;$m \\leftarrow w_\\text{self} \\cdot h_v$  (the self-channel)
> 3. &nbsp;&nbsp;&nbsp;&nbsp;**for** each relation $r$ with $N_r(v) \\neq \\varnothing$ **do**
> 4. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$m \\leftarrow m + w_r \\cdot \\text{mean}\\{h_u : u \\in N_r(v)\\}$
> 5. &nbsp;&nbsp;&nbsp;&nbsp;**end for**
> 6. &nbsp;&nbsp;&nbsp;&nbsp;$h'_v \\leftarrow m$  (σ = identity in this scalar version)
> 7. **end for**
>
One ordering detail matters: take the mean WITHIN each channel first, and apply
the channel's weight second. This ordering is what makes typing meaningful —
three co-authors cannot outvote one citation — and it is exactly what the
5.25-vs-2.00 assert pair checks.
"""),

    todo("""def rgcn_update(h_self, channels, w_self=1.0):
    \"\"\"Scalar R-GCN update: w_self*h_self + sum over channels of
    w_r * mean(neighbor features). `channels` = list of (w_r, [h_u, ...]).\"\"\"
    ### BEGIN SOLUTION
    out = w_self * h_self
    for w_r, nbrs in channels:
        out += w_r * sum(nbrs) / len(nbrs)
    return out
    ### END SOLUTION


def params_full(R, d):
    \"\"\"Per-layer parameters, full per-relation weights: (2R+1) matrices of d².\"\"\"
    ### BEGIN SOLUTION
    return (2 * R + 1) * d * d
    ### END SOLUTION


def params_basis(R, d, B):
    \"\"\"Per-layer parameters with basis decomposition: B bases of d², 2R·B
    mixing coefficients, plus the (undecomposed) self-loop d².\"\"\"
    ### BEGIN SOLUTION
    return B * d * d + 2 * R * B + d * d
    ### END SOLUTION


# — the lecture's ledger at P1: self, writes{A1=1,A2=2}, cites{P2=3}, pub⁻¹{V=2}
p1 = rgcn_update(1.0, [(0.5, [1, 2]), (1.0, [3]), (0.25, [2])])
assert abs(p1 - 5.25) < 1e-12, f"P1: 1 + ½·1.5 + 1·3 + ¼·2 = 5.25 — got {p1}"
# — the self-check node A2: one inverse-writes channel from {P1=1, P2=3}, w=½
a2 = rgcn_update(2.0, [(0.5, [1, 3])])
assert abs(a2 - 3.0) < 1e-12, f"A2: 2 + ½·2 = 3 — got {a2}"
# — untyped collapse: ONE channel, shared w=½, plain mean over all four
unt = rgcn_update(1.0, [(0.5, [1, 2, 3, 2])])
assert abs(unt - 2.0) < 1e-12, f"untyped: 1 + ½·2 = 2 — got {unt}"

# — the bill at FB15k-237 scale
assert params_full(237, 100) == 4_750_000, "full: 475 matrices × 10,000"
assert params_basis(237, 100, 30) == 324_220, "basis-30: 300,000 + 14,220 + 10,000"
assert round(params_full(237, 100) / params_basis(237, 100, 30), 1) == 14.7
# — and at a modest schema, where the treatment is not worth the disease
assert params_full(24, 200) == 1_960_000 and params_basis(24, 200, 30) == 1_241_440
print("exercise 1 ✓ — typed 5.25 vs untyped 2.00; the 4.75M bill and its 14.7× discount")""",
         stub="""def rgcn_update(h_self, channels, w_self=1.0):
    \"\"\"Scalar R-GCN update: w_self*h_self + sum over channels of
    w_r * mean(neighbor features). `channels` = list of (w_r, [h_u, ...]).\"\"\"
    # TODO: ~4 lines. Mean WITHIN each channel, then weight, then sum.
    raise NotImplementedError


def params_full(R, d):
    \"\"\"Per-layer parameters, full per-relation weights: (2R+1) matrices of d².\"\"\"
    # TODO: one line. Both directions plus the self-loop.
    raise NotImplementedError


def params_basis(R, d, B):
    \"\"\"Per-layer parameters with basis decomposition: B bases of d², 2R·B
    mixing coefficients, plus the (undecomposed) self-loop d².\"\"\"
    # TODO: one line, three terms.
    raise NotImplementedError


# — the lecture's ledger at P1: self, writes{A1=1,A2=2}, cites{P2=3}, pub⁻¹{V=2}
p1 = rgcn_update(1.0, [(0.5, [1, 2]), (1.0, [3]), (0.25, [2])])
assert abs(p1 - 5.25) < 1e-12, f"P1: 1 + ½·1.5 + 1·3 + ¼·2 = 5.25 — got {p1}"
# — the self-check node A2: one inverse-writes channel from {P1=1, P2=3}, w=½
a2 = rgcn_update(2.0, [(0.5, [1, 3])])
assert abs(a2 - 3.0) < 1e-12, f"A2: 2 + ½·2 = 3 — got {a2}"
# — untyped collapse: ONE channel, shared w=½, plain mean over all four
unt = rgcn_update(1.0, [(0.5, [1, 2, 3, 2])])
assert abs(unt - 2.0) < 1e-12, f"untyped: 1 + ½·2 = 2 — got {unt}"

# — the bill at FB15k-237 scale
assert params_full(237, 100) == 4_750_000, "full: 475 matrices × 10,000"
assert params_basis(237, 100, 30) == 324_220, "basis-30: 300,000 + 14,220 + 10,000"
assert round(params_full(237, 100) / params_basis(237, 100, 30), 1) == 14.7
# — and at a modest schema, where the treatment is not worth the disease
assert params_full(24, 200) == 1_960_000 and params_basis(24, 200, 30) == 1_241_440
print("exercise 1 ✓ — typed 5.25 vs untyped 2.00; the 4.75M bill and its 14.7× discount")"""),

    md("""## 2 · DBLP: what the typed graph is worth  *(exercise 2)*

The lecture measured this gap over three seeds: a feature-only MLP reaches
79.0 ± 0.5% test accuracy on DBLP author classification, and a typed GNN
reaches 84.0 ± 0.7%. Here you reproduce that gap with one seed.

**What you will implement:** the `forward(self, x_dict, edge_index_dict)`
method of the `HeteroGNN` class, two cells below (the next cell loads DBLP and
trains the provided MLP baseline first). For each of the two `HeteroConv`
layers, apply the layer, then apply ReLU followed by dropout(0.5) to every node
type's tensor; finally return `self.head(x_dict["author"])`. Everything else —
the MLP baseline, both training loops — is provided.

**How you will know it worked:** the exercise cell trains your model and
asserts that its test accuracy beats 81% and beats the MLP by more than 2
points; when both hold, it prints "exercise 2 ✓".
"""),

    code("""dblp = DBLP(root="data/DBLP")[0]
dblp["conference"].x = torch.ones(dblp["conference"].num_nodes, 1)   # featureless type
y = dblp["author"].y
masks = {s: dblp["author"][f"{s}_mask"] for s in ("train", "val", "test")}
print(dblp)
print("author classes:", int(y.max()) + 1, "· train/val/test:",
      [int(m.sum()) for m in masks.values()])


def run_mlp(seed=0, epochs=None):
    torch.manual_seed(seed)
    mlp = torch.nn.Sequential(torch.nn.Linear(334, 64), torch.nn.ReLU(),
                              torch.nn.Dropout(0.5), torch.nn.Linear(64, 4))
    opt = torch.optim.Adam(mlp.parameters(), lr=0.01, weight_decay=5e-4)
    X = dblp["author"].x
    best_val, best_test = 0, 0
    for ep in range(epochs or (40 if SMOKE else 200)):
        mlp.train(); opt.zero_grad()
        F.cross_entropy(mlp(X)[masks["train"]], y[masks["train"]]).backward()
        opt.step()
        mlp.eval()
        with torch.no_grad():
            pred = mlp(X).argmax(1)
            va = (pred[masks["val"]] == y[masks["val"]]).float().mean().item()
            te = (pred[masks["test"]] == y[masks["test"]]).float().mean().item()
        if va > best_val:
            best_val, best_test = va, te
    return best_test


mlp_acc = run_mlp()
print(f"feature-only MLP: {100 * mlp_acc:.1f}%  — the floor any graph model must clear")"""),

    todo("""class HeteroGNN(torch.nn.Module):
    \"\"\"Two typed layers (one SAGEConv per edge type, outputs SUMMED per node
    type), then a linear head on the author embeddings.\"\"\"
    def __init__(self, hidden=64):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            HeteroConv({et: SAGEConv((-1, -1), hidden) for et in dblp.edge_types},
                       aggr="sum")
            for _ in range(2)])
        self.head = Linear(hidden, 4)

    ### BEGIN SOLUTION
    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {k: F.dropout(F.relu(v), 0.5, self.training)
                      for k, v in x_dict.items()}
        return self.head(x_dict["author"])
    ### END SOLUTION


def run_hetero(seed=0, epochs=None):
    torch.manual_seed(seed)
    net = HeteroGNN()
    net(dblp.x_dict, dblp.edge_index_dict)          # lazy shape init
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_test = 0, 0
    for ep in range(epochs or (40 if SMOKE else 200)):
        net.train(); opt.zero_grad()
        out = net(dblp.x_dict, dblp.edge_index_dict)
        F.cross_entropy(out[masks["train"]], y[masks["train"]]).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(dblp.x_dict, dblp.edge_index_dict).argmax(1)
            va = (pred[masks["val"]] == y[masks["val"]]).float().mean().item()
            te = (pred[masks["test"]] == y[masks["test"]]).float().mean().item()
        if va > best_val:
            best_val, best_test = va, te
    return best_test


gnn_acc = run_hetero()
print(f"typed 2-layer GNN: {100 * gnn_acc:.1f}%  (MLP was {100 * mlp_acc:.1f}%)")

floor = 0.70 if SMOKE else 0.81
assert gnn_acc > floor, (
    f"typed GNN {100 * gnn_acc:.1f}% under the floor — checks: relu+dropout "
    f"INSIDE the loop over layers; return the head on x_dict['author'] only"
)
if not SMOKE:
    assert gnn_acc > mlp_acc + 0.02, (
        f"the typed graph should be worth >2 points over features alone "
        f"(lecture: +5.0 over three seeds) — got {100 * (gnn_acc - mlp_acc):.1f}"
    )
print("exercise 2 ✓ — the graph pays: co-authorship signal the MLP cannot see")""",
         stub="""class HeteroGNN(torch.nn.Module):
    \"\"\"Two typed layers (one SAGEConv per edge type, outputs SUMMED per node
    type), then a linear head on the author embeddings.\"\"\"
    def __init__(self, hidden=64):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            HeteroConv({et: SAGEConv((-1, -1), hidden) for et in dblp.edge_types},
                       aggr="sum")
            for _ in range(2)])
        self.head = Linear(hidden, 4)

    # TODO: forward(x_dict, edge_index_dict): for each conv, apply it, then
    # relu + dropout(0.5) on EVERY type's tensor; finally return
    # self.head(x_dict["author"]). ~6 lines.
    raise NotImplementedError("implement forward()")


def run_hetero(seed=0, epochs=None):
    torch.manual_seed(seed)
    net = HeteroGNN()
    net(dblp.x_dict, dblp.edge_index_dict)          # lazy shape init
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_test = 0, 0
    for ep in range(epochs or (40 if SMOKE else 200)):
        net.train(); opt.zero_grad()
        out = net(dblp.x_dict, dblp.edge_index_dict)
        F.cross_entropy(out[masks["train"]], y[masks["train"]]).backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            pred = net(dblp.x_dict, dblp.edge_index_dict).argmax(1)
            va = (pred[masks["val"]] == y[masks["val"]]).float().mean().item()
            te = (pred[masks["test"]] == y[masks["test"]]).float().mean().item()
        if va > best_val:
            best_val, best_test = va, te
    return best_test


gnn_acc = run_hetero()
print(f"typed 2-layer GNN: {100 * gnn_acc:.1f}%  (MLP was {100 * mlp_acc:.1f}%)")

floor = 0.70 if SMOKE else 0.81
assert gnn_acc > floor, (
    f"typed GNN {100 * gnn_acc:.1f}% under the floor — checks: relu+dropout "
    f"INSIDE the loop over layers; return the head on x_dict['author'] only"
)
if not SMOKE:
    assert gnn_acc > mlp_acc + 0.02, (
        f"the typed graph should be worth >2 points over features alone "
        f"(lecture: +5.0 over three seeds) — got {100 * (gnn_acc - mlp_acc):.1f}"
    )
print("exercise 2 ✓ — the graph pays: co-authorship signal the MLP cannot see")"""),

    md("""## 3 · Inverse channels  *(exercise 3)*

`RGCNConv` propagates messages only along the edges you give it, in the
direction you give them. Every practical R-GCN therefore adds an *inverse*
copy of each edge, so that information can also flow backwards. If you forget
this step, any entity that only ever appears as an edge target never receives
a message at all — a silent bug that no error message will report.

**What you will implement:** `add_inverse(edge_index, edge_type,
num_relations)` in the next cell. It must return a pair
`(edge_index2, edge_type2)` in which every original edge also appears flipped
(target→source), with the flipped copy's relation type offset by
`num_relations` so forward and inverse channels get separate weights. This is
about three lines using `torch.cat` and `.flip(0)`.

**How you will know it worked:** the asserts in the same cell check the shape
and content of your construction, then count FB15k-237's pure-sink entities
and verify that with your inverse edges every one of them now receives
messages. When they all pass, the cell prints "exercise 3 ✓".
"""),

    todo("""fb_train = FB15k_237("data/FB15k237", split="train")[0]
N_ENT = fb_train.num_nodes
N_REL = int(fb_train.edge_type.max()) + 1
print(f"FB15k-237: {N_ENT} entities · {N_REL} relations · {fb_train.num_edges} train triples")


def add_inverse(edge_index, edge_type, num_relations):
    \"\"\"Return (edge_index2, edge_type2) with every edge also present flipped,
    the flipped copy's type offset by num_relations.\"\"\"
    ### BEGIN SOLUTION
    ei2 = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    et2 = torch.cat([edge_type, edge_type + num_relations])
    return ei2, et2
    ### END SOLUTION


ei2, et2 = add_inverse(fb_train.edge_index, fb_train.edge_type, N_REL)
E = fb_train.num_edges
assert ei2.shape[1] == 2 * E and et2.shape[0] == 2 * E, "every edge appears twice"
assert (ei2[:, E:] == fb_train.edge_index.flip(0)).all(), "second half must be the flip"
assert (et2[:E] == fb_train.edge_type).all() and (et2[E:] == fb_train.edge_type + N_REL).all(), (
    "forward types 0..R-1 untouched; inverse types offset to R..2R-1"
)
assert int(et2.max()) == 2 * N_REL - 1, "the model must be built with num_relations=2R"

# the diode, demonstrated: pure sinks receive nothing without inverse channels
out_deg = torch.zeros(N_ENT).index_add_(0, fb_train.edge_index[0], torch.ones(E))
in_deg = torch.zeros(N_ENT).index_add_(0, fb_train.edge_index[1], torch.ones(E))
sinks = ((out_deg == 0) & (in_deg > 0)).sum().item()
assert sinks > 500, "FB15k-237 has hundreds of pure-sink entities (724 in this split)"
deg2 = torch.zeros(N_ENT).index_add_(0, ei2[1], torch.ones(2 * E))
assert (deg2[(out_deg == 0) & (in_deg > 0)] > 0).all(), (
    "with inverse channels every former sink now RECEIVES messages"
)
print(f"exercise 3 ✓ — {sinks:,} pure sinks (plus 1,126 pure sources, mute in reverse) "
      f"would sit in an information diode without your three lines")""",
         stub="""fb_train = FB15k_237("data/FB15k237", split="train")[0]
N_ENT = fb_train.num_nodes
N_REL = int(fb_train.edge_type.max()) + 1
print(f"FB15k-237: {N_ENT} entities · {N_REL} relations · {fb_train.num_edges} train triples")


def add_inverse(edge_index, edge_type, num_relations):
    \"\"\"Return (edge_index2, edge_type2) with every edge also present flipped,
    the flipped copy's type offset by num_relations.\"\"\"
    # TODO: 3 lines — torch.cat with .flip(0) on the edge index; offset types.
    raise NotImplementedError


ei2, et2 = add_inverse(fb_train.edge_index, fb_train.edge_type, N_REL)
E = fb_train.num_edges
assert ei2.shape[1] == 2 * E and et2.shape[0] == 2 * E, "every edge appears twice"
assert (ei2[:, E:] == fb_train.edge_index.flip(0)).all(), "second half must be the flip"
assert (et2[:E] == fb_train.edge_type).all() and (et2[E:] == fb_train.edge_type + N_REL).all(), (
    "forward types 0..R-1 untouched; inverse types offset to R..2R-1"
)
assert int(et2.max()) == 2 * N_REL - 1, "the model must be built with num_relations=2R"

# the diode, demonstrated: pure sinks receive nothing without inverse channels
out_deg = torch.zeros(N_ENT).index_add_(0, fb_train.edge_index[0], torch.ones(E))
in_deg = torch.zeros(N_ENT).index_add_(0, fb_train.edge_index[1], torch.ones(E))
sinks = ((out_deg == 0) & (in_deg > 0)).sum().item()
assert sinks > 500, "FB15k-237 has hundreds of pure-sink entities (724 in this split)"
deg2 = torch.zeros(N_ENT).index_add_(0, ei2[1], torch.ones(2 * E))
assert (deg2[(out_deg == 0) & (in_deg > 0)] > 0).all(), (
    "with inverse channels every former sink now RECEIVES messages"
)
print(f"exercise 3 ✓ — {sinks:,} pure sinks (plus 1,126 pure sources, mute in reverse) "
      f"would sit in an information diode without your three lines")"""),

    md("""## 4 · The showdown: lookup vs encoder  *(exercise 4)*

You will compare two KG-completion pipelines on FB15k-237 under identical
training budgets: shallow DistMult (embedding lookup) versus R-GCN encoder +
DistMult. The training code for both pipelines is provided in the next cell —
it is Lab 4's recipe with the encoder swapped in.

**What you will implement:** `evaluate(E, R, n_facts=None, seed=123)` in the
cell after the training code. Given entity embeddings `E` and relation
embeddings `R`, it must sample test facts with the seeded generator, score all
candidate entities in BOTH directions (tail prediction and head prediction)
with DistMult, set the scores of known-true answers other than the target to
−10⁹ (the "filtered" protocol), and return the pair `(mrr, hits10)`.

**How you will know it worked:** the exercise cell trains both pipelines,
calls your `evaluate` on each, and asserts that both MRRs land in the expected
bands (lecture values: 0.179 for the lookup, 0.126 for the encoder) and that
the lookup wins at this budget. When all three asserts pass, the cell prints
"exercise 4 ✓".

Under SMOKE the models barely train (1 epoch, 200 eval facts); run the full
version before submitting (5 epochs each, 1,000 facts — about 5 minutes with
a GPU).

Both pipelines follow the encoder–decoder template of
[Schlichtkrull et al., 2018](https://arxiv.org/abs/1703.06103) under Lab 4's
protocol. The algorithm below specifies exactly what the whole pipeline —
including your `evaluate` — must do; follow it step by step:

> **Encoder-agnostic KG completion** — **Input:** train triples $T$; entity
> table $E$; relation table $R$; encoder (lookup: $Z = E$; R-GCN: one typed
> pass over ALL edges); margin $\\gamma$; epochs. **Output:** trained $E, R$;
> filtered MRR / Hits@10.
>
> 1. **for** each epoch, **for** each batch $B \\subset T$ **do**
> 2. &nbsp;&nbsp;&nbsp;&nbsp;$Z \\leftarrow \\text{enc}(E)$
> 3. &nbsp;&nbsp;&nbsp;&nbsp;$\\tilde B \\leftarrow$ corrupt one side of each triple uniformly
> 4. &nbsp;&nbsp;&nbsp;&nbsp;$\\ell \\leftarrow \\text{mean ReLU}(\\gamma - s(B) + s(\\tilde B))$, $s(h,r,t) = \\langle Z_h, R_r, Z_t \\rangle$
> 5. &nbsp;&nbsp;&nbsp;&nbsp;gradient step on $\\ell$
> 6. **end for**
> 7. **for** each sampled test fact, BOTH directions: score all candidates,
>    set known-true answers (except the target) to $-10^9$, record the rank
> 8. **return** mean reciprocal rank; fraction of ranks $\\le 10$
>
Lines 1–6 are the provided trainers; lines 7–8 are your `evaluate`. Line 2
carries the entire cost asymmetry between the pipelines: the lookup's encoder
is a free table lookup, while the R-GCN's encoder is a full-graph propagation
on every gradient step.

**Predict before you run:** the lecture's table says which pipeline wins at
this budget. Write down your prediction — which pipeline wins, by roughly how
much MRR — before running the next cell.
"""),

    code("""fb_val = FB15k_237("data/FB15k237", split="val")[0]
fb_test = FB15k_237("data/FB15k237", split="test")[0]
tt = lambda d: torch.stack([d.edge_index[0], d.edge_type, d.edge_index[1]], 1)
train_T, val_T, test_T = tt(fb_train), tt(fb_val), tt(fb_test)
ALL = torch.cat([train_T, val_T, test_T])
known_t, known_h = {}, {}
for h, r, t in ALL.tolist():
    known_t.setdefault((h, r), set()).add(t)
    known_h.setdefault((r, t), set()).add(h)


def corrupt(triples, n_ent, g):
    C = triples.clone()
    side = torch.randint(2, (len(C),), generator=g)
    ent = torch.randint(n_ent, (len(C),), generator=g)
    C[side == 0, 0] = ent[side == 0]
    C[side == 1, 2] = ent[side == 1]
    return C


def margin_loss(pos, neg, gamma=1.0):
    return F.relu(gamma - pos + neg).mean()


EPOCHS = 1 if SMOKE else 5


def train_lookup_distmult(d=100, lr=0.01, batch=2048):
    g = torch.Generator().manual_seed(0)
    E = torch.nn.Parameter(torch.empty(N_ENT, d).uniform_(-0.06, 0.06, generator=g))
    R = torch.nn.Parameter(torch.empty(N_REL, d).uniform_(-0.06, 0.06, generator=g))
    opt = torch.optim.Adam([E, R], lr=lr)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(train_T), generator=g)
        for i in range(0, len(train_T), batch):
            pos = train_T[perm[i:i + batch]]
            neg = corrupt(pos, N_ENT, g)
            sc = lambda T: (E[T[:, 0]] * R[T[:, 1]] * E[T[:, 2]]).sum(1)
            loss = margin_loss(sc(pos), sc(neg))
            opt.zero_grad(); loss.backward(); opt.step()
    return E.detach(), R.detach()


class RGCNEncoder(torch.nn.Module):
    def __init__(self, d=100, bases=30):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = torch.nn.Parameter(torch.empty(N_ENT, d).uniform_(-0.06, 0.06, generator=g))
        self.conv = RGCNConv(d, d, num_relations=2 * N_REL, num_bases=bases)
        self.rel = torch.nn.Parameter(torch.empty(N_REL, d).uniform_(-0.06, 0.06, generator=g))

    def encode(self, ei, et):
        return self.emb + F.relu(self.conv(self.emb, ei, et))


def train_rgcn_distmult(batch=8192, lr=0.01):
    ei_dev, et_dev = ei2.to(DEV), et2.to(DEV)      # exercise 3's construction!
    model = RGCNEncoder().to(DEV)
    T = train_T.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(T), generator=g).to(DEV)
        for i in range(0, len(T), batch):
            pos = T[perm[i:i + batch]]
            neg = corrupt(pos.cpu(), N_ENT, g).to(DEV)
            Z = model.encode(ei_dev, et_dev)
            sc = lambda X: (Z[X[:, 0]] * model.rel[X[:, 1]] * Z[X[:, 2]]).sum(1)
            loss = margin_loss(sc(pos), sc(neg))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return model.encode(ei_dev, et_dev).cpu(), model.rel.detach().cpu()"""),

    todo("""def evaluate(E, R, n_facts=None, seed=123):
    \"\"\"Filtered MRR and Hits@10 over a sample of test facts, BOTH directions,
    DistMult scoring. Filter known true answers (except the target) by setting
    their scores to -1e9 before ranking. Return (mrr, hits10).\"\"\"
    ### BEGIN SOLUTION
    g_eval = torch.Generator().manual_seed(seed)
    n = n_facts or (200 if SMOKE else 1000)
    sample = test_T[torch.randperm(len(test_T), generator=g_eval)[:n]]
    ranks = []
    with torch.no_grad():
        for h, r, t in sample.tolist():
            s = (E[h] * R[r] * E).sum(1)
            s[list(known_t[(h, r)] - {t})] = -1e9
            ranks.append(int((s > s[t]).sum()) + 1)
            s = (E * R[r] * E[t]).sum(1)
            s[list(known_h[(r, t)] - {h})] = -1e9
            ranks.append(int((s > s[h]).sum()) + 1)
    rk = torch.tensor(ranks, dtype=torch.float)
    return (1 / rk).mean().item(), (rk <= 10).float().mean().item()
    ### END SOLUTION


print("training the lookup pipeline…")
E_lk, R_lk = train_lookup_distmult()
mrr_lk, h10_lk = evaluate(E_lk, R_lk)
print(f"  lookup + DistMult:  MRR {mrr_lk:.3f} · Hits@10 {h10_lk:.3f}")

print("training the encoder pipeline (the slow one — count the seconds)…")
Z_enc, R_enc = train_rgcn_distmult()
mrr_enc, h10_enc = evaluate(Z_enc, R_enc)
print(f"  R-GCN + DistMult:   MRR {mrr_enc:.3f} · Hits@10 {h10_enc:.3f}")

if not SMOKE:
    assert 0.13 < mrr_lk < 0.23, (
        f"lookup MRR {mrr_lk:.3f} outside the expected band (lecture at this "
        f"budget: 0.179) — check the filter excludes the target itself"
    )
    assert 0.08 < mrr_enc < 0.17, (
        f"encoder MRR {mrr_enc:.3f} outside the expected band (lecture: 0.126)"
    )
    assert mrr_lk > mrr_enc, (
        "at this matched budget the LOOKUP should win — if your encoder wins, "
        "something differs from the protocol (which would be interesting, but "
        "check the obvious first: same epochs? same eval sample?)"
    )
print("exercise 4 ✓ — the lecture's uncomfortable table, reproduced by your own referee")""",
         stub="""def evaluate(E, R, n_facts=None, seed=123):
    \"\"\"Filtered MRR and Hits@10 over a sample of test facts, BOTH directions,
    DistMult scoring. Filter known true answers (except the target) by setting
    their scores to -1e9 before ranking. Return (mrr, hits10).\"\"\"
    # TODO (~14 lines): sample (200 if SMOKE else 1000) test facts with the
    # seeded generator; for each, score all tails (E[h]*R[r]*E).sum(1), mask
    # known_t[(h,r)] minus the target, rank; same for heads with known_h;
    # return mean reciprocal rank and Hits@10.
    raise NotImplementedError


print("training the lookup pipeline…")
E_lk, R_lk = train_lookup_distmult()
mrr_lk, h10_lk = evaluate(E_lk, R_lk)
print(f"  lookup + DistMult:  MRR {mrr_lk:.3f} · Hits@10 {h10_lk:.3f}")

print("training the encoder pipeline (the slow one — count the seconds)…")
Z_enc, R_enc = train_rgcn_distmult()
mrr_enc, h10_enc = evaluate(Z_enc, R_enc)
print(f"  R-GCN + DistMult:   MRR {mrr_enc:.3f} · Hits@10 {h10_enc:.3f}")

if not SMOKE:
    assert 0.13 < mrr_lk < 0.23, (
        f"lookup MRR {mrr_lk:.3f} outside the expected band (lecture at this "
        f"budget: 0.179) — check the filter excludes the target itself"
    )
    assert 0.08 < mrr_enc < 0.17, (
        f"encoder MRR {mrr_enc:.3f} outside the expected band (lecture: 0.126)"
    )
    assert mrr_lk > mrr_enc, (
        "at this matched budget the LOOKUP should win — if your encoder wins, "
        "something differs from the protocol (which would be interesting, but "
        "check the obvious first: same epochs? same eval sample?)"
    )
print("exercise 4 ✓ — the lecture's uncomfortable table, reproduced by your own referee")"""),

    md("""### Your claims paragraph *(graded — write it in this cell)*

Write three claims in THIS cell, replacing the placeholder below. Each claim
is 1–2 sentences following the lecture's format: state the claim, quote the
specific numbers from your own cell outputs that support it, and state its
scope (what settings the claim covers). The three claims must be:

1. What the typed graph bought on DBLP (cite your MLP and GNN accuracies from
   exercise 2).
2. The showdown's verdict, including the wall-clock time you observed for each
   pipeline (cite your MRR/Hits@10 numbers from exercise 4).
3. What these results do **not** license you to conclude (consider inductive
   settings, node features, and tuning budgets).

*(your claims here)*

## 5 · Stretch (optional, ungraded)

1. **The untyped control.** Merge DBLP's edge types into one relation (project
   all features to a shared dimension first) and train the same depth. How
   much of the +5 was *types* rather than *graph*?
2. **Basis inspection.** Retrain the encoder with `num_bases=10` and pull
   `model.conv.comp` (the coefficient vectors $a_r$). Which relations load
   the same bases? Do the clusters make semantic sense?
3. **The inductive experiment.** Hold out 5% of entities (not triples) from
   training; compare pipelines on links involving them. The lookup forfeits —
   by how much does the encoder win the game the lookup cannot play?
4. **Metapath probe.** On DBLP, compute the author–paper–author adjacency by
   sparse matrix product and feed its row sums (co-author counts) to the MLP.
   How much of the GNN's gap does one hand-built metapath feature close?

## 6 · Reflection (answer in this cell, 2–4 sentences each)

**R1.** Write the encoder's job description in your own words: on what kind of
task, data, and deployment would you bet on R-GCN over the lookup, and what
measurement would settle the bet?

**R2.** Exercise 3 counted thousands of pure-sink entities. Explain what a
missing inverse channel does to *their* embeddings specifically, and why the
resulting failure would be hard to notice from aggregate MRR alone.

**R3.** Your Lab 4 TransE beat this lab's 5-epoch DistMult under the same
protocol. Using Week 4's geometry vocabulary, give one hypothesis for why —
and design (do not run) the two-line experiment that would test it.

*(your answers here)*

## What to submit

One executed notebook on Moodle: all four exercise checks ✓ run **full** (not
SMOKE), the claims paragraph, and the three reflections. *Runtime → Restart
and run all* first. Grading: assertions 60% · claims 15% · reflections 25%.

**AI policy reminder** (course honor code): AI assistants are allowed for this
lab *with disclosure* — add a line here naming any tools you used and for
what. You must be able to explain any line of your submission on request;
undeclared use or inability to explain is a violation.
"""),
]


def build(student: bool) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    })
    for kind, text, extra in CELLS:
        if kind == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(text))
        elif extra is None:
            nb.cells.append(nbf.v4.new_code_cell(text))
        else:
            if student:
                cell = nbf.v4.new_code_cell(extra)
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
