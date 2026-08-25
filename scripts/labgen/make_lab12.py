#!/usr/bin/env python
"""Generate Lab 12 (link prediction with proper splits: heuristic floor, VGAE,
SEAL with the leak demonstrated, and GraphRNN-S judged by statistics) —
student + solution notebooks.

    python scripts/labgen/make_lab12.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "labs" / "lab12_linkgen.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "labs" / "lab12_linkgen.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(solution, stub): return ("code", solution, stub)


SPLIT_CELL_SOL = '''def build_split(edge_index, seed=0):
    """Split undirected unique edges into train/val/test positives (85/5/10)
    with a seeded shuffle. Returns dict with train_pos, val_pos, test_pos
    (2,n) tensors and msg (the symmetrized message graph = train edges)."""
    ### BEGIN SOLUTION
    eu = edge_index[:, edge_index[0] < edge_index[1]]
    E = eu.shape[1]
    perm = torch.randperm(E, generator=torch.Generator().manual_seed(seed))
    n_test, n_val = int(0.1 * E), int(0.05 * E)
    test = eu[:, perm[:n_test]]
    val = eu[:, perm[n_test:n_test + n_val]]
    train = eu[:, perm[n_test + n_val:]]
    msg = torch.cat([train, train.flip(0)], dim=1)
    return {"train_pos": train, "val_pos": val, "test_pos": test, "msg": msg}
    ### END SOLUTION


S = build_split(data.edge_index)
train_pos, val_pos, test_pos, msg = S["train_pos"], S["val_pos"], S["test_pos"], S["msg"]

# the auditor — twelve lines that would have caught this week's 0.363
key = lambda P: set((min(a, b), max(a, b)) for a, b in P.t().tolist())
kt, kv, ktr = key(test_pos), key(val_pos), key(train_pos)
km = key(msg)
assert len(kt & ktr) == 0 and len(kt & kv) == 0 and len(kv & ktr) == 0, "buckets must be disjoint"
assert len(kt & km) == 0 and len(kv & km) == 0, "test/val edges must NOT be in the message graph"
assert km == ktr, "the message graph is exactly the train positives (symmetrized)"
assert (len(ktr), len(kv), len(kt)) == (4488, 263, 527), (
    f"seed-0 sizes must match the lecture (4488/263/527) — got {(len(ktr), len(kv), len(kt))}"
)
print("exercise 1 \\u2713 — four buckets, one job each, auditor satisfied")'''

SPLIT_CELL_STUB = SPLIT_CELL_SOL.replace('''    ### BEGIN SOLUTION
    eu = edge_index[:, edge_index[0] < edge_index[1]]
    E = eu.shape[1]
    perm = torch.randperm(E, generator=torch.Generator().manual_seed(seed))
    n_test, n_val = int(0.1 * E), int(0.05 * E)
    test = eu[:, perm[:n_test]]
    val = eu[:, perm[n_test:n_test + n_val]]
    train = eu[:, perm[n_test + n_val:]]
    msg = torch.cat([train, train.flip(0)], dim=1)
    return {"train_pos": train, "val_pos": val, "test_pos": test, "msg": msg}
    ### END SOLUTION''',
'''    # TODO (~8 lines): unique edges (src < dst); seeded randperm; slice 10% /
    # 5% / rest; message graph = train edges + their flips.
    raise NotImplementedError''')


CELLS = [
    md("""# Lab 12 · Link prediction, leak-proof — and graphs from nothing

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/labs/lab12_linkgen.ipynb)

**Week 12 · [lecture](https://lukmanovr.github.io/dkr/lectures/12-link-generation.html) · ≈ 8 min of compute (CPU-friendly; GPU shaves it further)**

You build the four-bucket split and its auditor; reproduce the heuristic floor
to the third decimal; implement VGAE's variational core; implement SEAL's
labeling twice — honestly and leakily — and watch the leak score below a coin;
and train a graph generator judged by your own statistics. The three learned
methods are VGAE ([Kipf & Welling, 2016](https://arxiv.org/abs/1611.07308)),
SEAL ([Zhang & Chen, 2018](https://arxiv.org/abs/1802.09691)), and GraphRNN
([You et al., 2018](https://arxiv.org/abs/1802.08773)) — each in its budget,
CPU-honest form.

### Goals
1. Build a proper LP split and the auditor that certifies it.
2. Reproduce CN 0.724 / AA 0.725 on the shared seeds.
3. Implement reparameterization + KL; train VGAE past 0.88.
4. Implement SEAL labels; assert leak-free > 0.85 AND leaky < 0.5.
5. Train GraphRNN-S; grade samples by TV distance against an ER baseline.
"""),

    md("""## 0 · Setup"""),

    code("""import os, sys, math, time

SMOKE = os.environ.get("SMOKE", "") == "1"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    %pip install -q torch_geometric==2.8.0.post1

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GINConv, global_add_pool
from torch_geometric.data import Data as PygData
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score

torch.manual_seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = Planetoid(root="data/Planetoid", name="Cora")[0]
N = data.num_nodes
print(f"Cora: {N} nodes · device {DEV}")"""),

    md("""## 1 · The split and the auditor  *(exercise 1)*

One function, four buckets — then twelve lines of asserts that would have
caught this week's 0.363 before any GPU spun up.
"""),

    todo(SPLIT_CELL_SOL, SPLIT_CELL_STUB),

    md("""## 2 · The heuristic floor  *(exercise 2 — determinism is a feature)*

Same split, same negatives, same seeds as the lecture: your numbers must
MATCH, not resemble.
"""),

    todo("""edge_set = key(data.edge_index[:, data.edge_index[0] < data.edge_index[1]])


def sample_negs(n, gen):
    out = []
    while len(out) < n:
        a = int(torch.randint(N, (1,), generator=gen))
        b = int(torch.randint(N, (1,), generator=gen))
        if a == b:
            continue
        k = (min(a, b), max(a, b))
        if k not in edge_set:
            out.append(k)
    return torch.tensor(out).t()


test_neg = sample_negs(len(kt), torch.Generator().manual_seed(1))

adj = [set() for _ in range(N)]
for a, b in train_pos.t().tolist():
    adj[a].add(b); adj[b].add(a)


def cn_score(pairs):
    \"\"\"Common-neighbor counts for each (a, b) column, from the MESSAGE graph.\"\"\"
    ### BEGIN SOLUTION
    return np.array([len(adj[a] & adj[b]) for a, b in pairs.t().tolist()], dtype=float)
    ### END SOLUTION


def aa_score(pairs):
    \"\"\"Adamic-Adar: sum over common neighbors c of 1/log(deg(c)), deg > 1.\"\"\"
    ### BEGIN SOLUTION
    out = []
    for a, b in pairs.t().tolist():
        s = 0.0
        for c in adj[a] & adj[b]:
            d = len(adj[c])
            if d > 1:
                s += 1.0 / math.log(d)
        out.append(s)
    return np.array(out)
    ### END SOLUTION


def auc(ps, ns):
    y = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
    return roc_auc_score(y, np.concatenate([ps, ns]))


cn_auc = round(auc(cn_score(test_pos), cn_score(test_neg)), 3)
aa_auc = round(auc(aa_score(test_pos), aa_score(test_neg)), 3)
print(f"CN {cn_auc} · AA {aa_auc}")
assert cn_auc == 0.724 and aa_auc == 0.725, (
    f"shared seeds must give the lecture's exact floor (0.724/0.725) — got "
    f"{cn_auc}/{aa_auc}. Heuristics read the MESSAGE graph only."
)
print("exercise 2 \\u2713 — the floor, reproduced to the third decimal")""",
         stub="""edge_set = key(data.edge_index[:, data.edge_index[0] < data.edge_index[1]])


def sample_negs(n, gen):
    out = []
    while len(out) < n:
        a = int(torch.randint(N, (1,), generator=gen))
        b = int(torch.randint(N, (1,), generator=gen))
        if a == b:
            continue
        k = (min(a, b), max(a, b))
        if k not in edge_set:
            out.append(k)
    return torch.tensor(out).t()


test_neg = sample_negs(len(kt), torch.Generator().manual_seed(1))

adj = [set() for _ in range(N)]
for a, b in train_pos.t().tolist():
    adj[a].add(b); adj[b].add(a)


def cn_score(pairs):
    \"\"\"Common-neighbor counts for each (a, b) column, from the MESSAGE graph.\"\"\"
    # TODO: one line with set intersection.
    raise NotImplementedError


def aa_score(pairs):
    \"\"\"Adamic-Adar: sum over common neighbors c of 1/log(deg(c)), deg > 1.\"\"\"
    # TODO: ~7 lines.
    raise NotImplementedError


def auc(ps, ns):
    y = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
    return roc_auc_score(y, np.concatenate([ps, ns]))


cn_auc = round(auc(cn_score(test_pos), cn_score(test_neg)), 3)
aa_auc = round(auc(aa_score(test_pos), aa_score(test_neg)), 3)
print(f"CN {cn_auc} · AA {aa_auc}")
assert cn_auc == 0.724 and aa_auc == 0.725, (
    f"shared seeds must give the lecture's exact floor (0.724/0.725) — got "
    f"{cn_auc}/{aa_auc}. Heuristics read the MESSAGE graph only."
)
print("exercise 2 \\u2713 — the floor, reproduced to the third decimal")"""),

    md("""## 3 · VGAE's variational core  *(exercise 3)*

The encoder is provided; you supply the two pieces that make it variational:
the reparameterized sample and the closed-form KL
([Kipf & Welling, 2016](https://arxiv.org/abs/1611.07308)). Code against the
spec, not the vibe — the harness below runs this algorithm; **you implement
steps 2 and 5**:

**Algorithm · one VGAE training step**
**Input:** features `X`; message graph; supervision positives `P`. **Output:** one gradient step on the ELBO.

1. `mu, logvar` ← GCN(`X`, message graph) — two heads over a shared layer
2. **reparameterize:** `eps` ~ N(0, I); `z` ← `mu + sigma * eps` with `sigma = exp(logvar / 2)`  *(your `reparameterize`)*
3. sample |`P`| negative non-edges, positives excluded
4. `recon` ← BCE of `z_a·z_b`: positives → 1, negatives → 0
5. **KL:** mean over nodes of `-1/2 * sum_j (1 + logvar - mu^2 - exp(logvar))`, scaled by `1/N`  *(your `kl_term`)*
6. backpropagate `recon + KL`; optimizer step — at evaluation, score with `mu` only (no sampling)
"""),

    todo("""class Encoder(torch.nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.c1 = GCNConv(data.num_features, 64)
        self.cmu = GCNConv(64, d)
        self.clog = GCNConv(64, d)

    def forward(self, x, ei):
        h = F.relu(self.c1(x, ei))
        return self.cmu(h, ei), self.clog(h, ei)


def reparameterize(mu, logvar):
    \"\"\"z = mu + sigma * eps with eps ~ N(0, I); sigma = exp(logvar / 2).\"\"\"
    ### BEGIN SOLUTION
    return mu + torch.randn_like(mu) * (0.5 * logvar).exp()
    ### END SOLUTION


def kl_term(mu, logvar):
    \"\"\"Mean over nodes of -1/2 * sum_j (1 + logvar - mu^2 - exp(logvar)).\"\"\"
    ### BEGIN SOLUTION
    return (-0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(1)).mean()
    ### END SOLUTION


mu0 = torch.zeros(4, 2); lv0 = torch.zeros(4, 2)
assert abs(kl_term(mu0, lv0).item()) < 1e-6, "KL of N(0,1) against N(0,1) is 0"
assert kl_term(mu0 + 2, lv0).item() > 3.9, "shifted means must be penalized (2 per dim)"
z1 = reparameterize(mu0, lv0 - 100)
assert torch.allclose(z1, mu0, atol=1e-3), "tiny variance -> sample collapses to mu"

torch.manual_seed(0)
enc = Encoder().to(DEV)
opt = torch.optim.Adam(enc.parameters(), lr=0.01)
x, ei = data.x.to(DEV), msg.to(DEV)
tp = train_pos.to(DEV)
gen = torch.Generator().manual_seed(2)
for ep in range(40 if SMOKE else 200):
    enc.train(); opt.zero_grad()
    mu, logv = enc(x, ei)
    z = reparameterize(mu, logv)
    neg = sample_negs(tp.shape[1], gen).to(DEV)
    pl = (z[tp[0]] * z[tp[1]]).sum(1)
    nl = (z[neg[0]] * z[neg[1]]).sum(1)
    recon = F.binary_cross_entropy_with_logits(pl, torch.ones_like(pl)) + \\
            F.binary_cross_entropy_with_logits(nl, torch.zeros_like(nl))
    (recon + kl_term(mu, logv) / N).backward()
    opt.step()
enc.eval()
with torch.no_grad():
    mu, _ = enc(x, ei)
Z = mu.cpu()
sc = lambda P: (Z[P[0]] * Z[P[1]]).sum(1).numpy()
vgae_auc = auc(sc(test_pos), sc(test_neg))
print(f"VGAE test AUC: {vgae_auc:.3f}")
assert vgae_auc > (0.75 if SMOKE else 0.88), (
    f"VGAE {vgae_auc:.3f} under the floor — check reparameterize returns "
    f"mu + sigma*eps (not a fresh normal) and the KL sign"
)
print("exercise 3 \\u2713 — eighteen points over the floor, variational plumbing verified")""",
         stub="""class Encoder(torch.nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.c1 = GCNConv(data.num_features, 64)
        self.cmu = GCNConv(64, d)
        self.clog = GCNConv(64, d)

    def forward(self, x, ei):
        h = F.relu(self.c1(x, ei))
        return self.cmu(h, ei), self.clog(h, ei)


def reparameterize(mu, logvar):
    \"\"\"z = mu + sigma * eps with eps ~ N(0, I); sigma = exp(logvar / 2).\"\"\"
    # TODO: one line with torch.randn_like.
    raise NotImplementedError


def kl_term(mu, logvar):
    \"\"\"Mean over nodes of -1/2 * sum_j (1 + logvar - mu^2 - exp(logvar)).\"\"\"
    # TODO: one line.
    raise NotImplementedError


mu0 = torch.zeros(4, 2); lv0 = torch.zeros(4, 2)
assert abs(kl_term(mu0, lv0).item()) < 1e-6, "KL of N(0,1) against N(0,1) is 0"
assert kl_term(mu0 + 2, lv0).item() > 3.9, "shifted means must be penalized (2 per dim)"
z1 = reparameterize(mu0, lv0 - 100)
assert torch.allclose(z1, mu0, atol=1e-3), "tiny variance -> sample collapses to mu"

torch.manual_seed(0)
enc = Encoder().to(DEV)
opt = torch.optim.Adam(enc.parameters(), lr=0.01)
x, ei = data.x.to(DEV), msg.to(DEV)
tp = train_pos.to(DEV)
gen = torch.Generator().manual_seed(2)
for ep in range(40 if SMOKE else 200):
    enc.train(); opt.zero_grad()
    mu, logv = enc(x, ei)
    z = reparameterize(mu, logv)
    neg = sample_negs(tp.shape[1], gen).to(DEV)
    pl = (z[tp[0]] * z[tp[1]]).sum(1)
    nl = (z[neg[0]] * z[neg[1]]).sum(1)
    recon = F.binary_cross_entropy_with_logits(pl, torch.ones_like(pl)) + \\
            F.binary_cross_entropy_with_logits(nl, torch.zeros_like(nl))
    (recon + kl_term(mu, logv) / N).backward()
    opt.step()
enc.eval()
with torch.no_grad():
    mu, _ = enc(x, ei)
Z = mu.cpu()
sc = lambda P: (Z[P[0]] * Z[P[1]]).sum(1).numpy()
vgae_auc = auc(sc(test_pos), sc(test_neg))
print(f"VGAE test AUC: {vgae_auc:.3f}")
assert vgae_auc > (0.75 if SMOKE else 0.88), (
    f"VGAE {vgae_auc:.3f} under the floor — check reparameterize returns "
    f"mu + sigma*eps (not a fresh normal) and the KL sign"
)
print("exercise 3 \\u2713 — eighteen points over the floor, variational plumbing verified")"""),

    md("""## 4 · SEAL's labels — honest and leaky  *(exercise 4 — own the cautionary tale)*

You implement the labeling function with a `leak` switch. The harness trains
BOTH variants: your leak-free labels must clear 0.85, and the leaky ones must
score **below 0.5** — this week's 0.363, reproduced as a passing test.

**Algorithm · SEAL labeling** ([Zhang & Chen, 2018](https://arxiv.org/abs/1802.09691)), for a candidate pair `(a, b)`:
**Input:** message-graph adjacency sets `adj`; candidate `(a, b)`. **Output:** a labeled subgraph a GNN can score.

1. `adj_a` ← `adj[a] - {b}`; `adj_b` ← `adj[b] - {a}` — remove the candidate edge from the adjacency sets
2. extract the enclosing subgraph: `{a, b}` plus their neighbors, minus the edge `(a, b)` itself
3. **for** each node `v`: `d_a, d_b` ∈ {0 = the endpoint itself, 1 = adjacent, 2 = else}, read from `adj_a` / `adj_b`
4. feature `x_v` ← one-hot of `(d_a, d_b)` over the 9 combinations
5. a small GIN classifies the labeled subgraph, pooled to one logit

`leak=True` means exactly "skip step 1": the labels read the *raw* adjacency,
the candidate edge leaves its `(0,1)` fingerprint on training positives only,
and the sabotage assert below fails as a violated spec line, not a mystery.
"""),

    todo("""def node_labels(nodes, a, b, leak=False):
    \"\"\"For each node v: label (da, db) with dx = 0 if v==x, 1 if v adjacent
    to x, else 2 — as a one-hot over the 9 combinations. leak=False must
    compute adjacency WITH THE TARGET EDGE (a,b) REMOVED; leak=True uses the
    raw adjacency (the bug).\"\"\"
    ### BEGIN SOLUTION
    adj_a = adj[a] if leak else (adj[a] - {b})
    adj_b = adj[b] if leak else (adj[b] - {a})
    feats = []
    for v in nodes:
        da = 0 if v == a else (1 if v in adj_a else 2)
        db = 0 if v == b else (1 if v in adj_b else 2)
        f = torch.zeros(9)
        f[da * 3 + db] = 1.0
        feats.append(f)
    return torch.stack(feats)
    ### END SOLUTION


# toy check: for a TRAIN positive (edge in adj), the endpoint labels differ
a0, b0 = train_pos[0, 0].item(), train_pos[1, 0].item()
Lh = node_labels([a0, b0], a0, b0, leak=False)
Ll = node_labels([a0, b0], a0, b0, leak=True)
assert Lh[0].argmax().item() == 0 * 3 + 2, "leak-free: a reads (0,2) — the edge is gone"
assert Ll[0].argmax().item() == 0 * 3 + 1, "leaky: a reads (0,1) — the edge left its fingerprint"


def enclosing(a, b, leak):
    adj_a = adj[a] - {b}; adj_b = adj[b] - {a}
    nodes = list(({a, b} | adj_a | adj_b))[:50]
    if a not in nodes: nodes.append(a)
    if b not in nodes: nodes.append(b)
    idx = {v: i for i, v in enumerate(nodes)}
    ns = set(nodes)
    edges = [(idx[v], idx[u]) for v in nodes for u in adj[v]
             if u in ns and v < u and (v, u) != (min(a, b), max(a, b))]
    ei = (torch.tensor(edges).t() if edges else torch.zeros(2, 0, dtype=torch.long))
    ei = torch.cat([ei, ei.flip(0)], dim=1) if ei.numel() else ei
    return PygData(x=node_labels(nodes, a, b, leak=leak), edge_index=ei)


def run_seal(leak):
    g3 = torch.Generator().manual_seed(3)
    ntr = 500 if SMOKE else 2000
    trp = train_pos[:, torch.randperm(train_pos.shape[1], generator=g3)[:ntr]]
    trn = sample_negs(ntr, g3)
    def ds(pos, neg):
        out = []
        for a, b in pos.t().tolist():
            d = enclosing(a, b, leak); d.y = torch.tensor([1.0]); out.append(d)
        for a, b in neg.t().tolist():
            d = enclosing(a, b, leak); d.y = torch.tensor([0.0]); out.append(d)
        return out
    tr_ds, te_ds = ds(trp, trn), ds(test_pos, test_neg)

    class Net(torch.nn.Module):
        def __init__(self, h=32):
            super().__init__()
            self.g1 = GINConv(torch.nn.Sequential(torch.nn.Linear(9, h), torch.nn.ReLU(), torch.nn.Linear(h, h)))
            self.g2 = GINConv(torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, h)))
            self.out = torch.nn.Linear(h, 1)

        def forward(self, d):
            h = F.relu(self.g1(d.x, d.edge_index))
            h = F.relu(self.g2(h, d.edge_index))
            return self.out(global_add_pool(h, d.batch)).squeeze(-1)

    torch.manual_seed(0)
    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    for ep in range(3 if SMOKE else 8):
        for batch in DataLoader(tr_ds, batch_size=64, shuffle=True):
            batch = batch.to(DEV)
            loss = F.binary_cross_entropy_with_logits(net(batch), batch.y)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    scores = []
    with torch.no_grad():
        for batch in DataLoader(te_ds, batch_size=128):
            scores.append(net(batch.to(DEV)).cpu().numpy())
    scores = np.concatenate(scores)
    return auc(scores[:len(kt)], scores[len(kt):])


seal_clean = run_seal(leak=False)
seal_leaky = run_seal(leak=True)
print(f"SEAL leak-free: {seal_clean:.3f} · SEAL leaky: {seal_leaky:.3f}")
assert seal_clean > (0.75 if SMOKE else 0.85), f"clean SEAL {seal_clean:.3f} under floor"
assert seal_leaky < (0.55 if SMOKE else 0.50), (
    f"the LEAKY variant should score below a coin ({seal_leaky:.3f}) — the model "
    f"learns the fingerprint on train positives and votes backwards on test"
)
print("exercise 4 \\u2713 — you have now personally caused, observed, and fixed the 0.363")""",
         stub="""def node_labels(nodes, a, b, leak=False):
    \"\"\"For each node v: label (da, db) with dx = 0 if v==x, 1 if v adjacent
    to x, else 2 — as a one-hot over the 9 combinations. leak=False must
    compute adjacency WITH THE TARGET EDGE (a,b) REMOVED; leak=True uses the
    raw adjacency (the bug).\"\"\"
    # TODO (~10 lines): adj_a = adj[a] - {b} unless leak; likewise adj_b;
    # per node, one-hot of (da*3 + db).
    raise NotImplementedError


# toy check: for a TRAIN positive (edge in adj), the endpoint labels differ
a0, b0 = train_pos[0, 0].item(), train_pos[1, 0].item()
Lh = node_labels([a0, b0], a0, b0, leak=False)
Ll = node_labels([a0, b0], a0, b0, leak=True)
assert Lh[0].argmax().item() == 0 * 3 + 2, "leak-free: a reads (0,2) — the edge is gone"
assert Ll[0].argmax().item() == 0 * 3 + 1, "leaky: a reads (0,1) — the edge left its fingerprint"


def enclosing(a, b, leak):
    adj_a = adj[a] - {b}; adj_b = adj[b] - {a}
    nodes = list(({a, b} | adj_a | adj_b))[:50]
    if a not in nodes: nodes.append(a)
    if b not in nodes: nodes.append(b)
    idx = {v: i for i, v in enumerate(nodes)}
    ns = set(nodes)
    edges = [(idx[v], idx[u]) for v in nodes for u in adj[v]
             if u in ns and v < u and (v, u) != (min(a, b), max(a, b))]
    ei = (torch.tensor(edges).t() if edges else torch.zeros(2, 0, dtype=torch.long))
    ei = torch.cat([ei, ei.flip(0)], dim=1) if ei.numel() else ei
    return PygData(x=node_labels(nodes, a, b, leak=leak), edge_index=ei)


def run_seal(leak):
    g3 = torch.Generator().manual_seed(3)
    ntr = 500 if SMOKE else 2000
    trp = train_pos[:, torch.randperm(train_pos.shape[1], generator=g3)[:ntr]]
    trn = sample_negs(ntr, g3)
    def ds(pos, neg):
        out = []
        for a, b in pos.t().tolist():
            d = enclosing(a, b, leak); d.y = torch.tensor([1.0]); out.append(d)
        for a, b in neg.t().tolist():
            d = enclosing(a, b, leak); d.y = torch.tensor([0.0]); out.append(d)
        return out
    tr_ds, te_ds = ds(trp, trn), ds(test_pos, test_neg)

    class Net(torch.nn.Module):
        def __init__(self, h=32):
            super().__init__()
            self.g1 = GINConv(torch.nn.Sequential(torch.nn.Linear(9, h), torch.nn.ReLU(), torch.nn.Linear(h, h)))
            self.g2 = GINConv(torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, h)))
            self.out = torch.nn.Linear(h, 1)

        def forward(self, d):
            h = F.relu(self.g1(d.x, d.edge_index))
            h = F.relu(self.g2(h, d.edge_index))
            return self.out(global_add_pool(h, d.batch)).squeeze(-1)

    torch.manual_seed(0)
    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=0.005)
    for ep in range(3 if SMOKE else 8):
        for batch in DataLoader(tr_ds, batch_size=64, shuffle=True):
            batch = batch.to(DEV)
            loss = F.binary_cross_entropy_with_logits(net(batch), batch.y)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    scores = []
    with torch.no_grad():
        for batch in DataLoader(te_ds, batch_size=128):
            scores.append(net(batch.to(DEV)).cpu().numpy())
    scores = np.concatenate(scores)
    return auc(scores[:len(kt)], scores[len(kt):])


seal_clean = run_seal(leak=False)
seal_leaky = run_seal(leak=True)
print(f"SEAL leak-free: {seal_clean:.3f} · SEAL leaky: {seal_leaky:.3f}")
assert seal_clean > (0.75 if SMOKE else 0.85), f"clean SEAL {seal_clean:.3f} under floor"
assert seal_leaky < (0.55 if SMOKE else 0.50), (
    f"the LEAKY variant should score below a coin ({seal_leaky:.3f}) — the model "
    f"learns the fingerprint on train positives and votes backwards on test"
)
print("exercise 4 \\u2713 — you have now personally caused, observed, and fixed the 0.363")"""),

    md("""## 5 · GraphRNN-S, judged by statistics  *(exercise 5)*

The model and training are provided; your parts are the BFS sequencing (the
masterstroke) and the TV metric (the judge).
"""),

    todo("""rng = np.random.default_rng(0)


def community_graph(gen):
    n1, n2 = int(gen.integers(6, 10)), int(gen.integers(6, 10))
    g = nx.random_partition_graph([n1, n2], 0.7, 0.05, seed=int(gen.integers(1 << 30)))
    return nx.convert_node_labels_to_integers(g)


train_graphs = [community_graph(rng) for _ in range(60 if SMOKE else 200)]
test_graphs = [community_graph(rng) for _ in range(40)]
M = 12


def bfs_sequence(g):
    \"\"\"BFS-order the nodes from node 0; return rows S_i (i=1..n-1) of width M
    where S_i[k] = 1 iff node i connects to node i-1-k in the ordering.\"\"\"
    ### BEGIN SOLUTION
    order = list(nx.bfs_tree(g, source=0).nodes())
    order += [v for v in g if v not in order]
    pos = {v: i for i, v in enumerate(order)}
    rows = []
    for i in range(1, g.number_of_nodes()):
        row = torch.zeros(M)
        for u in g[order[i]]:
            j = pos[u]
            if j < i and i - 1 - j < M:
                row[i - 1 - j] = 1.0
        rows.append(row)
    return torch.stack(rows) if rows else torch.zeros(0, M)
    ### END SOLUTION


s0 = bfs_sequence(nx.path_graph(5))
assert s0.shape == (4, M) and (s0.sum(1) == 1).all(), (
    "on a path, every new BFS node connects to exactly one predecessor"
)
assert (s0[:, 0] == 1).all(), "on a path the predecessor is always the previous node (slot 0)"


def tv_distance(vals1, vals2, bins, lo, hi):
    \"\"\"Total-variation distance between the two samples' histograms.\"\"\"
    ### BEGIN SOLUTION
    def h(vals):
        hh, _ = np.histogram(vals, bins=bins, range=(lo, hi))
        return hh / max(hh.sum(), 1)
    return float(np.abs(h(vals1) - h(vals2)).sum() / 2)
    ### END SOLUTION


assert abs(tv_distance([1, 1, 2], [1, 1, 2], 4, 0, 4)) < 1e-9, "identical samples: TV 0"
assert abs(tv_distance([0, 0], [3, 3], 4, 0, 4) - 1.0) < 1e-9, "disjoint samples: TV 1"


class GraphRNNS(torch.nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.gru = torch.nn.GRU(M, h, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(h, 64), torch.nn.ReLU(), torch.nn.Linear(64, M))


torch.manual_seed(0)
grnn = GraphRNNS().to(DEV)
opt = torch.optim.Adam(grnn.parameters(), lr=0.003)
seqs = [bfs_sequence(g) for g in train_graphs]
for ep in range(10 if SMOKE else 60):
    for i in rng.permutation(len(seqs)):
        s = seqs[i]
        if len(s) < 2:
            continue
        inp = torch.cat([torch.ones(1, M), s[:-1]]).unsqueeze(0).to(DEV)
        out, _ = grnn.gru(inp)
        loss = F.binary_cross_entropy_with_logits(grnn.head(out), s.unsqueeze(0).to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()


def generate(gen_torch, n_nodes):
    grnn.eval()
    with torch.no_grad():
        rows, h, x = [], None, torch.ones(1, 1, M).to(DEV)
        for i in range(1, n_nodes):
            out, h = grnn.gru(x, h)
            p = torch.sigmoid(grnn.head(out))[0, 0]
            row = (torch.rand(M, generator=gen_torch).to(DEV) < p).float()
            rows.append(row.cpu())
            x = row.unsqueeze(0).unsqueeze(0).to(DEV)
    g = nx.Graph(); g.add_nodes_from(range(n_nodes))
    for i, row in enumerate(rows, start=1):
        for k in range(M):
            j = i - 1 - k
            if j >= 0 and row[k] > 0:
                g.add_edge(i, j)
    return g


gt = torch.Generator().manual_seed(5)
gen_graphs = [generate(gt, int(rng.integers(12, 20))) for _ in range(10 if SMOKE else 40)]
er_graphs = [nx.gnp_random_graph(int(rng.integers(12, 20)), 0.35,
                                 seed=int(rng.integers(1 << 30))) for _ in range(10 if SMOKE else 40)]

clu = lambda gs: [c for g in gs for c in nx.clustering(g).values()]
tv_model = tv_distance(clu(gen_graphs), clu(test_graphs), 10, 0, 1)
tv_er = tv_distance(clu(er_graphs), clu(test_graphs), 10, 0, 1)
print(f"clustering TV — model {tv_model:.3f} · ER {tv_er:.3f}")
assert tv_model < tv_er - (0.05 if SMOKE else 0.2), (
    f"the trained generator must beat ER on clustering TV by a clear margin "
    f"(lecture: 0.154 vs 0.596) — got {tv_model:.3f} vs {tv_er:.3f}"
)
print("exercise 5 \\u2713 — the generator learned what independence cannot fake")""",
         stub="""rng = np.random.default_rng(0)


def community_graph(gen):
    n1, n2 = int(gen.integers(6, 10)), int(gen.integers(6, 10))
    g = nx.random_partition_graph([n1, n2], 0.7, 0.05, seed=int(gen.integers(1 << 30)))
    return nx.convert_node_labels_to_integers(g)


train_graphs = [community_graph(rng) for _ in range(60 if SMOKE else 200)]
test_graphs = [community_graph(rng) for _ in range(40)]
M = 12


def bfs_sequence(g):
    \"\"\"BFS-order the nodes from node 0; return rows S_i (i=1..n-1) of width M
    where S_i[k] = 1 iff node i connects to node i-1-k in the ordering.\"\"\"
    # TODO (~10 lines): nx.bfs_tree ordering (append any unreached nodes);
    # for each node i, mark its edges to predecessors within the M-window.
    raise NotImplementedError


s0 = bfs_sequence(nx.path_graph(5))
assert s0.shape == (4, M) and (s0.sum(1) == 1).all(), (
    "on a path, every new BFS node connects to exactly one predecessor"
)
assert (s0[:, 0] == 1).all(), "on a path the predecessor is always the previous node (slot 0)"


def tv_distance(vals1, vals2, bins, lo, hi):
    \"\"\"Total-variation distance between the two samples' histograms.\"\"\"
    # TODO: histogram both (normalized), half the absolute difference sum.
    raise NotImplementedError


assert abs(tv_distance([1, 1, 2], [1, 1, 2], 4, 0, 4)) < 1e-9, "identical samples: TV 0"
assert abs(tv_distance([0, 0], [3, 3], 4, 0, 4) - 1.0) < 1e-9, "disjoint samples: TV 1"


class GraphRNNS(torch.nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.gru = torch.nn.GRU(M, h, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(h, 64), torch.nn.ReLU(), torch.nn.Linear(64, M))


torch.manual_seed(0)
grnn = GraphRNNS().to(DEV)
opt = torch.optim.Adam(grnn.parameters(), lr=0.003)
seqs = [bfs_sequence(g) for g in train_graphs]
for ep in range(10 if SMOKE else 60):
    for i in rng.permutation(len(seqs)):
        s = seqs[i]
        if len(s) < 2:
            continue
        inp = torch.cat([torch.ones(1, M), s[:-1]]).unsqueeze(0).to(DEV)
        out, _ = grnn.gru(inp)
        loss = F.binary_cross_entropy_with_logits(grnn.head(out), s.unsqueeze(0).to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()


def generate(gen_torch, n_nodes):
    grnn.eval()
    with torch.no_grad():
        rows, h, x = [], None, torch.ones(1, 1, M).to(DEV)
        for i in range(1, n_nodes):
            out, h = grnn.gru(x, h)
            p = torch.sigmoid(grnn.head(out))[0, 0]
            row = (torch.rand(M, generator=gen_torch).to(DEV) < p).float()
            rows.append(row.cpu())
            x = row.unsqueeze(0).unsqueeze(0).to(DEV)
    g = nx.Graph(); g.add_nodes_from(range(n_nodes))
    for i, row in enumerate(rows, start=1):
        for k in range(M):
            j = i - 1 - k
            if j >= 0 and row[k] > 0:
                g.add_edge(i, j)
    return g


gt = torch.Generator().manual_seed(5)
gen_graphs = [generate(gt, int(rng.integers(12, 20))) for _ in range(10 if SMOKE else 40)]
er_graphs = [nx.gnp_random_graph(int(rng.integers(12, 20)), 0.35,
                                 seed=int(rng.integers(1 << 30))) for _ in range(10 if SMOKE else 40)]

clu = lambda gs: [c for g in gs for c in nx.clustering(g).values()]
tv_model = tv_distance(clu(gen_graphs), clu(test_graphs), 10, 0, 1)
tv_er = tv_distance(clu(er_graphs), clu(test_graphs), 10, 0, 1)
print(f"clustering TV — model {tv_model:.3f} · ER {tv_er:.3f}")
assert tv_model < tv_er - (0.05 if SMOKE else 0.2), (
    f"the trained generator must beat ER on clustering TV by a clear margin "
    f"(lecture: 0.154 vs 0.596) — got {tv_model:.3f} vs {tv_er:.3f}"
)
print("exercise 5 \\u2713 — the generator learned what independence cannot fake")"""),

    md("""### Your claims paragraph *(graded — write it in this cell)*

Three claims, cells cited: one about the floor and what it says about Cora's
link signal; one about the leak (your two SEAL numbers, and the mechanism in
one sentence); one about the generator (your TV margin, and what it does NOT
prove — Week 9's evaluator blind spot).

*(your claims here)*

## 6 · Stretch (optional, ungraded)

1. **Hits@50.** Evaluate every method with Hits@50 over the merged candidate
   set. Does the method ranking change from AUC's?
2. **Degree-matched negatives.** Resample test negatives to match positive
   endpoint degrees; re-run the ladder. Who falls furthest?
3. **The MLP decoder.** Replace VGAE's inner product with an MLP over
   [z_a ‖ z_b ‖ z_a∘z_b]. Worth how much AUC?
4. **Orbit statistics.** Add a 4-node motif count to the generator's judging
   set. Does ER get further exposed?

## 7 · Reflection (answer in this cell, 2–4 sentences each)

**R1.** Your leaky SEAL scored below 0.5. Explain why *sub-random* test
performance is stronger evidence of leakage than *random* performance would
be, and name the three tripwires from the lecture.

**R2.** Week 4 built filtered ranking for KG completion; this week built the
four-bucket split. Name what each protects against, and the shared principle.

**R3.** Your generator beat ER on clustering TV. Construct (describe, don't
code) two clearly different graph distributions your TV-on-clustering judge
could NOT tell apart, and connect to Week 9.

*(your answers here)*

## What to submit

One executed notebook on Moodle, run **full** (not SMOKE): five checks ✓,
claims, reflections. *Restart and run all* first. Grading: assertions 60% ·
claims 15% · reflections 25%.

**AI policy reminder** (course honor code): AI assistants are allowed for
this lab *with disclosure* — add a line here naming any tools you used and
for what. You must be able to explain any line of your submission on
request; undeclared use or inability to explain is a violation.
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
