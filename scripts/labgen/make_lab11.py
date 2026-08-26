#!/usr/bin/env python
"""Generate Lab 11 (scaling: the explosion counted, a fanout sampler from CSR
arrays, SGC derived and raced, and the measured accuracy/time/memory table) —
student + solution notebooks.

    python scripts/labgen/make_lab11.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "labs" / "lab11_scaling.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "labs" / "lab11_scaling.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(solution, stub): return ("code", solution, stub)


CELLS = [
    md("""# Lab 11 · Scaling — count the explosion, build the sampler, race SGC

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/labs/lab11_scaling.ipynb)

**Week 11 · [lecture](https://lukmanovr.github.io/dkr/lectures/11-scaling.html) · ≈ 10 min of compute (GPU helps; everything also runs on free-tier CPU)**

Full-batch GNN training holds the whole graph in GPU memory at once, and on
real graphs it stops working: this lab is about the two escapes production
systems actually run. **Neighbor sampling**, introduced with GraphSAGE
([Hamilton et al., 2017](https://arxiv.org/abs/1706.02216)), bounds each
node's computation tree by drawing at most *f* neighbors per hop, so memory
scales with the batch instead of the graph. **Precompute-and-decouple**,
distilled to its purest form in SGC
([Wu et al., 2019](https://arxiv.org/abs/1902.07153)), deletes the
nonlinearity so that all propagation collapses into one offline sparse
product, leaving a logistic regression to train. You will measure the
neighborhood explosion on ogbn-arxiv and match the lecture's numbers on the
same seeds; build the CSR fanout sampler with your own hands; derive SGC and
beat the epoch-time column; and assemble the measured accuracy/time table
that decides deployments.

### Goals
1. Count L-hop neighborhoods on a real graph and reproduce 18 / 4,577 / 22,663.
2. Implement one block of fanout sampling: bounded, deduplicated, deterministic.
3. Implement SGC's precompute in sparse ops and race it against full-batch GCN.
4. Assemble the trade-off table and write claims that cite its cells.
"""),

    md("""## 0 · Setup"""),

    code("""import os, sys, time, statistics

SMOKE = os.environ.get("SMOKE", "") == "1"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    %pip install -q torch_geometric==2.8.0.post1 ogb==1.3.6

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ogb 1.3.6 predates torch>=2.6's weights_only default; the cache is OGB's own file
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
from ogb.nodeproppred import PygNodePropPredDataset
ds = PygNodePropPredDataset("ogbn-arxiv", root="data/ogb")
torch.load = _load

split = ds.get_idx_split()
data = ds[0]
data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)  # symmetrize
data.y = data.y.squeeze()
N, E = data.num_nodes, data.edge_index.shape[1]
tr_idx, va_idx, te_idx = split["train"], split["valid"], split["test"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"arxiv: {N:,} nodes · {E:,} directed edges · device {DEV}")"""),

    md("""## 1 · The neighborhood explosion, counted  *(exercise 1)*

The next cell builds a CSR (compressed sparse row) view of the graph — it is
provided; study it, because exercises 1 and 2 both read neighbors from it.
Node `v`'s neighbors are the slice `dst_sorted[ptr[v] : ptr[v+1]]`.

**What you will implement:** `khop_size(v, L)` in the cell after the CSR cell.
It must return the number of DISTINCT nodes within `L` hops of node `v`,
counting `v` itself — a breadth-first search over the CSR slices, tracking a
`seen` set and expanding only the new frontier at each hop.

**How you will know it worked:** the cell averages your counts over 200 seed
nodes and asserts they equal the lecture's numbers exactly (18 nodes at 1 hop,
4,577 at 2 hops, 22,663 at 3 hops — same seeds, same counts). When the asserts
pass, the cell prints "exercise 1 ✓".
"""),

    code("""# CSR: node v's neighbors are dst_sorted[ptr[v] : ptr[v+1]]
perm = torch.argsort(data.edge_index[0])
dst_sorted = data.edge_index[1][perm]
deg = torch.zeros(N, dtype=torch.long).index_add_(0, data.edge_index[0],
                                                  torch.ones(E, dtype=torch.long))
ptr = torch.cat([torch.zeros(1, dtype=torch.long), deg.cumsum(0)])
print(f"avg degree {deg.float().mean():.1f} · max degree {deg.max()} · median {deg.float().median():.0f}")
print("(mean 3.5x the median: the heavy tail the lecture blamed — verified)")"""),

    todo("""def khop_size(v, L):
    \"\"\"Number of distinct nodes within L hops of v (v itself included).\"\"\"
    ### BEGIN SOLUTION
    frontier, seen = {v}, {v}
    for _ in range(L):
        nxt = set()
        for u in frontier:
            nxt.update(dst_sorted[ptr[u]:ptr[u + 1]].tolist())
        frontier = nxt - seen
        seen |= nxt
    return len(seen)
    ### END SOLUTION


g = torch.Generator().manual_seed(0)
seeds = torch.randperm(N, generator=g)[:20 if SMOKE else 200]
sizes = {L: round(sum(khop_size(int(s), L) for s in seeds) / len(seeds)) for L in (1, 2, 3)}
print(f"avg L-hop sizes over {len(seeds)} seeds:", sizes)

if not SMOKE:
    assert sizes[1] == 18 and sizes[2] == 4577 and sizes[3] == 22663, (
        f"same seeds as the lecture must give the same counts (18/4577/22663) — got "
        f"{sizes}. Checks: count DISTINCT nodes (a set, not a sum); include v itself; "
        f"expand only the NEW frontier each hop."
    )
    assert sizes[3] / N > 0.12, "three hops reach ≥12% of the whole graph"
else:
    assert sizes[1] < sizes[2] < sizes[3], "sizes must explode with L"
print("exercise 1 ✓ — the explosion is not a metaphor; you just counted it")""",
         stub="""def khop_size(v, L):
    \"\"\"Number of distinct nodes within L hops of v (v itself included).\"\"\"
    # TODO (~8 lines): BFS over the CSR slices; track a `seen` set and a
    # frontier of NEW nodes per hop; return len(seen).
    raise NotImplementedError


g = torch.Generator().manual_seed(0)
seeds = torch.randperm(N, generator=g)[:20 if SMOKE else 200]
sizes = {L: round(sum(khop_size(int(s), L) for s in seeds) / len(seeds)) for L in (1, 2, 3)}
print(f"avg L-hop sizes over {len(seeds)} seeds:", sizes)

if not SMOKE:
    assert sizes[1] == 18 and sizes[2] == 4577 and sizes[3] == 22663, (
        f"same seeds as the lecture must give the same counts (18/4577/22663) — got "
        f"{sizes}. Checks: count DISTINCT nodes (a set, not a sum); include v itself; "
        f"expand only the NEW frontier each hop."
    )
    assert sizes[3] / N > 0.12, "three hops reach ≥12% of the whole graph"
else:
    assert sizes[1] < sizes[2] < sizes[3], "sizes must explode with L"
print("exercise 1 ✓ — the explosion is not a metaphor; you just counted it")"""),

    md("""## 2 · One block of fanout sampling  *(exercise 2)*

**What you will implement:** `sample_block(batch_nodes, fanout, gen)` in the
next cell. Given a batch of target nodes, it must draw at most `fanout`
neighbors for each target, collect the drawn edges using batch-local indices,
and return three things: `uniq` (the deduplicated union of batch nodes and
drawn neighbors), `edges` (a (2, m) tensor of [source position in `uniq`;
target position in the batch]), and `pos` (the batch nodes' positions in
`uniq`, so that `uniq[pos] == batch_nodes`). Every production neighbor
sampler is this function with better engineering.

**How you will know it worked:** the asserts in the same cell check the remap
contract (`uniq[pos] == batch`), the fanout bound, determinism under a fixed
seed, and that an uncapped fanout loads strictly more nodes. When they all
pass, the cell prints "exercise 2 ✓".

The algorithm below (the lecture's Algorithm 1) specifies exactly what your
implementation must do; follow it step by step.

**Input:** CSR arrays `ptr`, `dst_sorted`; batch targets B; fanout f; seeded RNG.
**Output:** deduplicated union U; local edge list E_B; positions of B in U.

1. S ← ∅ — the drawn (source, target-position) pairs
2. **for** each target v_i in B **do**
3. &nbsp;&nbsp;&nbsp;&nbsp;N ← `dst_sorted[ptr[v_i] : ptr[v_i+1]]`
4. &nbsp;&nbsp;&nbsp;&nbsp;**if** |N| > f **then** draw f uniform samples from N **else** keep all of N
5. &nbsp;&nbsp;&nbsp;&nbsp;add (u, i) to S for every drawn neighbor u
6. **end for**
7. U ← unique(B ∪ {u : (u, i) ∈ S}), with inverse map π (global ID → local index)
8. E_B ← {(π(u), i) : (u, i) ∈ S};  pos ← π(B)
9. **return** U, E_B, pos

Three practical hints. Step 7 is the remap from global node IDs to local
indices — it is the step everyone gets wrong once, and it is what the first
assert checks (`uniq[pos] == batch`); `torch.unique(..., return_inverse=True)`
computes the map π for you. Draw samples with `torch.randint(...,
generator=gen)`; sampling with replacement is fine — it keeps the code simple
on purpose. Finally, use only the generator `gen` for randomness, because the
determinism assert reruns your function with the same seed and requires an
identical sample.
"""),

    todo("""def sample_block(batch_nodes, fanout, gen):
    \"\"\"One sampling layer. Returns (uniq, edges, pos):
      uniq  — 1-D tensor: deduplicated union of batch nodes and drawn neighbors
      edges — (2, m): [source-position-in-uniq; target-position-in-BATCH]
      pos   — batch nodes' positions in uniq (so uniq[pos] == batch_nodes)\"\"\"
    ### BEGIN SOLUTION
    srcs, dsts = [], []
    for i, v in enumerate(batch_nodes.tolist()):
        lo, hi = ptr[v].item(), ptr[v + 1].item()
        n = hi - lo
        if n == 0:
            continue
        nb = dst_sorted[lo:hi] if n <= fanout else dst_sorted[lo + torch.randint(n, (fanout,), generator=gen)]
        srcs.append(nb)
        dsts.append(torch.full((len(nb),), i, dtype=torch.long))
    src = torch.cat(srcs) if srcs else torch.zeros(0, dtype=torch.long)
    dst = torch.cat(dsts) if dsts else torch.zeros(0, dtype=torch.long)
    uniq, inv = torch.unique(torch.cat([batch_nodes, src]), return_inverse=True)
    return uniq, torch.stack([inv[len(batch_nodes):], dst]), inv[:len(batch_nodes)]
    ### END SOLUTION


gen = torch.Generator().manual_seed(7)
b = tr_idx[:256]
uniq, edges, pos = sample_block(b, 10, gen)
assert (uniq[pos] == b).all(), "uniq[pos] must recover the batch (the remap contract)"
assert edges.shape[0] == 2 and edges[1].max() < len(b), "edge targets are BATCH positions"
counts = torch.bincount(edges[1], minlength=len(b))
assert counts.max() <= 10, f"fanout bound violated: a node drew {counts.max()} neighbors"
gen2 = torch.Generator().manual_seed(7)
uniq2, edges2, _ = sample_block(b, 10, gen2)
assert (uniq2 == uniq).all() and (edges2 == edges).all(), "same seed, same sample — determinism"
big = sample_block(b, 10**9, torch.Generator().manual_seed(0))[0]
assert len(big) > len(uniq), "no cap should load strictly more than fanout 10"
print(f"exercise 2 ✓ — batch 256 loads {len(uniq)} nodes at fanout 10 vs {len(big)} uncapped")""",
         stub="""def sample_block(batch_nodes, fanout, gen):
    \"\"\"One sampling layer. Returns (uniq, edges, pos):
      uniq  — 1-D tensor: deduplicated union of batch nodes and drawn neighbors
      edges — (2, m): [source-position-in-uniq; target-position-in-BATCH]
      pos   — batch nodes' positions in uniq (so uniq[pos] == batch_nodes)\"\"\"
    # TODO (~12 lines): per target, slice CSR, draw ≤ fanout with
    # torch.randint(..., generator=gen); collect (neighbor, target-position)
    # pairs; torch.unique with return_inverse for the dedup + remap.
    raise NotImplementedError


gen = torch.Generator().manual_seed(7)
b = tr_idx[:256]
uniq, edges, pos = sample_block(b, 10, gen)
assert (uniq[pos] == b).all(), "uniq[pos] must recover the batch (the remap contract)"
assert edges.shape[0] == 2 and edges[1].max() < len(b), "edge targets are BATCH positions"
counts = torch.bincount(edges[1], minlength=len(b))
assert counts.max() <= 10, f"fanout bound violated: a node drew {counts.max()} neighbors"
gen2 = torch.Generator().manual_seed(7)
uniq2, edges2, _ = sample_block(b, 10, gen2)
assert (uniq2 == uniq).all() and (edges2 == edges).all(), "same seed, same sample — determinism"
big = sample_block(b, 10**9, torch.Generator().manual_seed(0))[0]
assert len(big) > len(uniq), "no cap should load strictly more than fanout 10"
print(f"exercise 2 ✓ — batch 256 loads {len(uniq)} nodes at fanout 10 vs {len(big)} uncapped")"""),

    md("""### Training with your sampler *(provided — nothing to implement here)*

The next cell is provided: it stacks two calls to your `sample_block` to train
a 2-layer sampled GraphSAGE, with the scatter-mean aggregation written out by
hand so you can see it. Just run it. Under SMOKE it trains on a slice of the
training set; run the full version before submitting. If its assert fails,
the usual cause is block wiring — the inner block must sample neighbors of
the OUTER block's union, not of the original batch.
"""),

    code("""W1s = torch.nn.Linear(128, 128); W1n = torch.nn.Linear(128, 128)
W2s = torch.nn.Linear(128, 40); W2n = torch.nn.Linear(128, 40)
params = list(W1s.parameters()) + list(W1n.parameters()) + list(W2s.parameters()) + list(W2n.parameters())
for p in [W1s, W1n, W2s, W2n]:
    p.to(DEV)
torch.manual_seed(0)
opt = torch.optim.Adam(params, lr=0.01)
gen = torch.Generator().manual_seed(0)


def scatter_mean(x_src, edges, n_out, dim=128):
    src, dst = edges
    agg = torch.zeros(n_out, x_src.shape[1], device=DEV).index_add_(0, dst.to(DEV), x_src[src.to(DEV)])
    cnt = torch.zeros(n_out, device=DEV).index_add_(0, dst.to(DEV),
        torch.ones(edges.shape[1], device=DEV)).clamp(min=1)
    return agg / cnt.unsqueeze(1)


train_pool = tr_idx[:8192] if SMOKE else tr_idx
t0 = time.time()
for ep in range(1 if SMOKE else 2):
    order = train_pool[torch.randperm(len(train_pool), generator=gen)]
    for i in range(0, len(order), 1024):
        bt = order[i:i + 1024]
        u1, e1, p1 = sample_block(bt, 10, gen)
        u2, e2, p2 = sample_block(u1, 10, gen)
        xs = data.x[u2].to(DEV)
        h1 = F.relu(W1s(xs[p2.to(DEV)]) + W1n(scatter_mean(xs, e2, len(u1))))
        out = W2s(h1[p1.to(DEV)]) + W2n(scatter_mean(h1, e1, len(bt)))
        loss = F.cross_entropy(out, data.y[bt].to(DEV))
        opt.zero_grad(); loss.backward(); opt.step()
sample_time = time.time() - t0
# full-neighborhood inference (no sampling at eval — the lecture's rule)
with torch.no_grad():
    x = data.x.to(DEV); ei = data.edge_index.to(DEV)
    agg = torch.zeros(N, 128, device=DEV).index_add_(0, ei[1], x[ei[0]])
    cnt = torch.zeros(N, device=DEV).index_add_(0, ei[1], torch.ones(E, device=DEV)).clamp(min=1)
    h = F.relu(W1s(x) + W1n(agg / cnt.unsqueeze(1)))
    agg2 = torch.zeros(N, 40, device=DEV).index_add_(0, ei[1], W2n(h)[ei[0]])
    out = W2s(h) + agg2 / cnt.unsqueeze(1)
    sage_acc = (out[te_idx.to(DEV)].argmax(1).cpu() == data.y[te_idx]).float().mean().item()
print(f"sampled SAGE: test {sage_acc:.3f} · {sample_time:.0f}s trained")
assert sage_acc > (0.25 if SMOKE else 0.52), (
    f"sampled SAGE {sage_acc:.3f} under the floor — check the block wiring "
    f"(inner block samples neighbors of the OUTER union)"
)
print("sampler training ✓ — bounded memory, and you felt every second the pipeline costs")"""),

    md("""## 3 · SGC: precompute once, then race  *(exercise 3)*

SGC starts from the lecture's algebraic collapse: with the nonlinearity
removed, a 2-layer GCN $\\hat A (\\hat A X W_0) W_1$ equals $(\\hat A^2 X) W$
— so all graph propagation can be computed once, offline, and training reduces
to a logistic regression on the propagated features.

**What you will implement:** `sgc_features(K)` in the next cell. It must
return $\\hat A^K X$, where $\\hat A = \\tilde D^{-1/2}(A+I)\\tilde D^{-1/2}$,
computed with sparse operations on CPU (build the sparse normalized adjacency
once, then multiply K times; handle the self-loop as a separate
`selfweight * X` term as the docstring describes). Everything after that —
the logistic-regression training loop and the timed full-batch GCN it races —
is provided in the same cell.

**How you will know it worked:** the asserts check that K=0 returns the
features unchanged, that K=2 changes them, that SGC's test accuracy clears
0.60, and that your SGC epoch is at least 5× cheaper than a full-batch GCN
epoch (if it is not, graph work is leaking into your training loop). When all
of these pass, the cell prints "exercise 3 ✓".

The algorithm below specifies exactly what the full SGC pipeline must do;
steps 1–3 are your `sgc_features`, steps 4–7 are the provided classifier loop.

**Algorithm — SGC: propagate once, then train a linear model**
*([Wu et al., 2019](https://arxiv.org/abs/1902.07153))*

**Input:** adjacency A, features X, depth K, labels on the train split.
**Output:** classifier weights W.

1. Â ← D̃^{-1/2} (A + I) D̃^{-1/2}  — sparse, built once
2. **for** k = 1 … K **do** X ← Â X (sparse–dense multiply) **end for** — offline, once
3. cache X_K ← X; discard the graph
4. **for** each epoch **do**
5. &nbsp;&nbsp;&nbsp;&nbsp;gradient step on cross-entropy of softmax(X_K[train] · W)
6. **end for**
7. **return** W  — inference is X_K · W: no edges consulted

Step 2 costs O(K·E·d) and is paid exactly once, before training starts; after
it, the graph is never consulted again. That one-time cost is why the assert
demands your training epochs be at least 5× cheaper than a full-batch GCN's.
"""),

    todo("""def sgc_features(K):
    \"\"\"Return Â^K X where Â = D̃^{-1/2}(A+I)D̃^{-1/2}, via sparse ops on CPU.
    Handle the self-loop as A-part + selfweight*X (see the degree vector).\"\"\"
    ### BEGIN SOLUTION
    src, dst = data.edge_index
    d2 = torch.zeros(N).index_add_(0, src, torch.ones(E)) + 1
    vals = (d2[src] * d2[dst]).rsqrt()
    A = torch.sparse_coo_tensor(data.edge_index, vals, (N, N)).coalesce()
    selfw = 1.0 / d2
    X = data.x.clone()
    for _ in range(K):
        X = torch.sparse.mm(A, X) + selfw.unsqueeze(1) * X
    return X
    ### END SOLUTION


t0 = time.time()
X2 = sgc_features(2)
pre_time = time.time() - t0
assert X2.shape == data.x.shape and not torch.allclose(X2, data.x), "K=2 must change the features"
assert torch.allclose(sgc_features(0), data.x), "K=0 is the identity"
print(f"precompute: {pre_time:.1f}s, once — the graph's entire role in training, prepaid")

torch.manual_seed(0)
Wc = torch.nn.Linear(128, 40).to(DEV)
optc = torch.optim.Adam(Wc.parameters(), lr=0.01)
Xtr, ytr = X2[tr_idx].to(DEV), data.y[tr_idx].to(DEV)
t0 = time.time()
EPS = 50 if SMOKE else 300
for ep in range(EPS):
    optc.zero_grad()
    F.cross_entropy(Wc(Xtr), ytr).backward()
    optc.step()
sgc_per_ep = (time.time() - t0) / EPS
with torch.no_grad():
    sgc_acc = (Wc(X2.to(DEV)).argmax(1).cpu()[te_idx] == data.y[te_idx]).float().mean().item()
print(f"SGC: test {sgc_acc:.3f} · {1000 * sgc_per_ep:.1f} ms/epoch")
assert sgc_acc > (0.35 if SMOKE else 0.60), f"SGC accuracy {sgc_acc:.3f} under the floor"

# the race: a few epochs of full-batch GCN, timed
class GCN2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = GCNConv(128, 128), GCNConv(128, 40)

    def forward(self, x, ei):
        return self.c2(F.relu(self.c1(x, ei)), ei)


torch.manual_seed(0)
netg = GCN2().to(DEV)
optg = torch.optim.Adam(netg.parameters(), lr=0.01)
dx, dei, dy = data.x.to(DEV), data.edge_index.to(DEV), data.y.to(DEV)
t0 = time.time()
GEPS = 2 if SMOKE else 5
for ep in range(GEPS):
    optg.zero_grad()
    F.cross_entropy(netg(dx, dei)[tr_idx.to(DEV)], dy[tr_idx.to(DEV)]).backward()
    optg.step()
gcn_per_ep = (time.time() - t0) / GEPS
print(f"full-batch GCN: {gcn_per_ep:.2f} s/epoch — vs SGC's {1000 * sgc_per_ep:.1f} ms")
assert sgc_per_ep < gcn_per_ep / 5, (
    "SGC's epoch must be at least 5x cheaper than a full-batch GCN epoch — "
    "if not, your precompute is leaking into the training loop"
)
print("exercise 3 ✓ — same ballpark accuracy, epochs measured in milliseconds")""",
         stub="""def sgc_features(K):
    \"\"\"Return Â^K X where Â = D̃^{-1/2}(A+I)D̃^{-1/2}, via sparse ops on CPU.
    Handle the self-loop as A-part + selfweight*X (see the degree vector).\"\"\"
    # TODO (~8 lines): degree+1 vector; edge values (d[src]*d[dst])^-1/2 in a
    # torch.sparse_coo_tensor; loop K times: X = sparse.mm(A, X) + (1/d)·X.
    raise NotImplementedError


t0 = time.time()
X2 = sgc_features(2)
pre_time = time.time() - t0
assert X2.shape == data.x.shape and not torch.allclose(X2, data.x), "K=2 must change the features"
assert torch.allclose(sgc_features(0), data.x), "K=0 is the identity"
print(f"precompute: {pre_time:.1f}s, once — the graph's entire role in training, prepaid")

torch.manual_seed(0)
Wc = torch.nn.Linear(128, 40).to(DEV)
optc = torch.optim.Adam(Wc.parameters(), lr=0.01)
Xtr, ytr = X2[tr_idx].to(DEV), data.y[tr_idx].to(DEV)
t0 = time.time()
EPS = 50 if SMOKE else 300
for ep in range(EPS):
    optc.zero_grad()
    F.cross_entropy(Wc(Xtr), ytr).backward()
    optc.step()
sgc_per_ep = (time.time() - t0) / EPS
with torch.no_grad():
    sgc_acc = (Wc(X2.to(DEV)).argmax(1).cpu()[te_idx] == data.y[te_idx]).float().mean().item()
print(f"SGC: test {sgc_acc:.3f} · {1000 * sgc_per_ep:.1f} ms/epoch")
assert sgc_acc > (0.35 if SMOKE else 0.60), f"SGC accuracy {sgc_acc:.3f} under the floor"

# the race: a few epochs of full-batch GCN, timed
class GCN2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1, self.c2 = GCNConv(128, 128), GCNConv(128, 40)

    def forward(self, x, ei):
        return self.c2(F.relu(self.c1(x, ei)), ei)


torch.manual_seed(0)
netg = GCN2().to(DEV)
optg = torch.optim.Adam(netg.parameters(), lr=0.01)
dx, dei, dy = data.x.to(DEV), data.edge_index.to(DEV), data.y.to(DEV)
t0 = time.time()
GEPS = 2 if SMOKE else 5
for ep in range(GEPS):
    optg.zero_grad()
    F.cross_entropy(netg(dx, dei)[tr_idx.to(DEV)], dy[tr_idx.to(DEV)]).backward()
    optg.step()
gcn_per_ep = (time.time() - t0) / GEPS
print(f"full-batch GCN: {gcn_per_ep:.2f} s/epoch — vs SGC's {1000 * sgc_per_ep:.1f} ms")
assert sgc_per_ep < gcn_per_ep / 5, (
    "SGC's epoch must be at least 5x cheaper than a full-batch GCN epoch — "
    "if not, your precompute is leaking into the training loop"
)
print("exercise 3 ✓ — same ballpark accuracy, epochs measured in milliseconds")"""),

    md("""## 4 · The trade-off table  *(exercise 4)*

Nothing to implement here: the next cell collects the numbers you measured in
exercises 2 and 3 (plus the lecture's full-batch GCN and Cluster-GCN rows for
reference) into one accuracy/time/memory table. Run it, then write your claims
paragraph in the cell after it, citing this table.
"""),

    code("""table = {
    "sampled SAGE (yours)": {"acc": round(sage_acc, 3), "note": "memory batch-bounded"},
    "SGC (yours)": {"acc": round(sgc_acc, 3), "ms_per_ep": round(1000 * sgc_per_ep, 2),
                     "note": f"precompute {pre_time:.1f}s once"},
    "full-batch GCN (lecture)": {"acc": 0.665, "s_per_ep": 0.10, "peak_mb": 2821},
    "Cluster-GCN random (lecture)": {"acc": 0.628, "s_per_ep": 0.51, "peak_mb": 31},
}
for k, v in table.items():
    print(f"  {k:<30} {v}")
if not SMOKE:
    assert table["SGC (yours)"]["acc"] > 0.60 and table["sampled SAGE (yours)"]["acc"] > 0.52
print("the table is yours — now make it argue")"""),

    md("""### Your claims paragraph *(graded — write it in this cell)*

Write three claims in THIS cell, replacing the placeholder below. Each claim
is 1–2 sentences in the lecture's format: state the claim, cite the specific
numbers (and the table cells or exercise outputs they come from) that support
it, and state its scope. The three claims must be:

1. What the neighborhood explosion you counted in exercise 1 implies for how
   batches must be designed.
2. A comparison of YOUR SGC row against the lecture's full-batch GCN row —
   how much accuracy you paid and how much time you bought.
3. What this table does NOT establish — be specific (for example,
   implementation quality and task-dependence both limit its scope).

*(your claims here)*

## 5 · Stretch (optional, ungraded)

1. **The fanout dial.** Rerun your sampler at fanouts 2, 5, 25: plot accuracy
   and time. Where does the marginal neighbor stop paying?
2. **SIGN-lite.** Concatenate Â X and Â²X and rerun the classifier — does the
   two-operator view beat SGC's one?
3. **Poor-man's Cluster-GCN.** Random 40-way partition, intra-cluster GCN
   steps — the lecture's "Cluster-GCN: partition, then train" algorithm
   with the uniform-random partitioner
   ([Chiang et al., 2019](https://arxiv.org/abs/1905.07953)). Verify the
   1/k edge-retention formula and reproduce ≈0.628.
4. **The friendship paradox, measured.** Compare the mean degree of nodes vs
   the mean degree of *neighbors* on arxiv. Explain the explosion's 9× gap.

## 6 · Reflection (answer in this cell, 2–4 sentences each)

**R1.** The products extrapolation: ogbn-products has 2.4M nodes and 124M
edges. Using the lecture's audit lines, estimate full-batch training memory
at width 128 and state which of YOUR two implementations survives, with the
arithmetic shown.

**R2.** Your sampler trained at a fraction of full-batch memory but each
epoch cost more wall-clock. Name the exact bottleneck (be more specific than
"sampling is slow") and the two standard engineering remedies.

**R3.** A teammate proposes 4-layer sampled SAGE at fanout 20 "for more
context." Compute the per-target bound and give the two-sentence reply the
lecture equips you with.

*(your answers here)*

## What to submit

One executed notebook on Moodle, run **full** (not SMOKE): four checks ✓,
the claims paragraph, three reflections. *Restart and run all* first.
Grading: assertions 60% · claims 15% · reflections 25%.

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
