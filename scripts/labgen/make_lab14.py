#!/usr/bin/env python
"""Generate Lab 14 (the temporal wall and its auditor, LightGCN propagation
with hand asserts on the widget graph, BPR from the definition, the measured
ladder, and the protocol-inflation experiment) — student + solution notebooks.

    python scripts/labgen/make_lab14.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "labs" / "lab14_recsys.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "labs" / "lab14_recsys.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(solution, stub): return ("code", solution, stub)


SPLIT_SOL = '''def make_temporal_split(raw):
    """Global temporal 80/10/10: sort by timestamp, cut at the 80% and 90%
    marks, then drop val/test rows whose user OR item never appears in train
    (cold start — excluded and counted, not silently kept).
    Returns (train, val, test) integer arrays of rows (user, item, rating, ts)."""
    ### BEGIN SOLUTION
    d = raw[np.argsort(raw[:, 3], kind="stable")]
    n = len(d)
    tr = d[: int(0.8 * n)]
    va = d[int(0.8 * n): int(0.9 * n)]
    te = d[int(0.9 * n):]
    seen_u, seen_i = set(tr[:, 0].tolist()), set(tr[:, 1].tolist())
    keep = lambda X: X[[u in seen_u and i in seen_i for u, i in zip(X[:, 0], X[:, 1])]]
    return tr, keep(va), keep(te)
    ### END SOLUTION


tr, va, te = make_temporal_split(raw)

# the auditor — deployment's contract, checkable by machine
assert tr[:, 3].max() <= va[:, 3].min() <= va[:, 3].max() <= te[:, 3].min(), (
    "the wall must be a wall: every train timestamp precedes every val "
    "timestamp precedes every test timestamp"
)
assert (len(tr), len(va), len(te)) == (80000, 1519, 1344), (
    f"sizes must match the lecture (80000/1519/1344 after cold-start "
    f"filtering) — got {(len(tr), len(va), len(te))}"
)
n_test_users = len(np.unique(te[:, 0]))
assert n_test_users == 66, (
    f"exactly 66 of 943 users survive the wall (the lecture's honest, thin "
    f"population) — got {n_test_users}"
)
import datetime
wall = datetime.datetime.fromtimestamp(int(tr[:, 3].max()), datetime.timezone.utc)
print(f"the wall falls on {wall.date()} — train before, test after, no exceptions")
print(f"surviving test users: {n_test_users} of 943 — thin, awkward, honest")
print("exercise 1 \\u2713 — a protocol you can assert beats one you can only defend")'''

SPLIT_STUB = SPLIT_SOL.replace('''    ### BEGIN SOLUTION
    d = raw[np.argsort(raw[:, 3], kind="stable")]
    n = len(d)
    tr = d[: int(0.8 * n)]
    va = d[int(0.8 * n): int(0.9 * n)]
    te = d[int(0.9 * n):]
    seen_u, seen_i = set(tr[:, 0].tolist()), set(tr[:, 1].tolist())
    keep = lambda X: X[[u in seen_u and i in seen_i for u, i in zip(X[:, 0], X[:, 1])]]
    return tr, keep(va), keep(te)
    ### END SOLUTION''',
'''    # TODO (~7 lines): argsort by column 3 (timestamp, stable); slice 80/10/10;
    # build the train-seen user and item sets; filter val and test to rows
    # whose user AND item are train-seen; return the three arrays.
    raise NotImplementedError''')


PROP_SOL = '''def propagate(P0, Q0, K):
    """LightGCN propagation: K rounds of e <- A_hat e over the bipartite
    graph, followed by the layer-MEAN readout. Returns (P_final, Q_final,
    layers) where layers[k] = (P_k, Q_k) — kept so the asserts (and you)
    can watch the signal spread layer by layer."""
    ### BEGIN SOLUTION
    layers = [(P0, Q0)]
    P, Q = P0, Q0
    for _ in range(K):
        P, Q = torch.sparse.mm(A_ui, Q), torch.sparse.mm(A_iu, P)
        layers.append((P, Q))
    Pf = torch.stack([p for p, _ in layers]).mean(0)
    Qf = torch.stack([q for _, q in layers]).mean(0)
    return Pf, Qf, layers
    ### END SOLUTION


# hand-check on the LECTURE WIDGET's toy graph before touching real data:
# 3 users, 4 items, edges u1-i1, u1-i2, u2-i2, u2-i3, u3-i3, u3-i4
toy_edges = np.array([[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3]])
A_ui, A_iu = build_norm_adj(toy_edges, 3, 4)
P0 = torch.tensor([[1.0], [0.0], [0.0]])
Q0 = torch.zeros(4, 1)
Pf, Qf, L = propagate(P0, Q0, 4)

u2_at_2 = float(L[2][0][1, 0])
u3_at_2 = float(L[2][0][2, 0])
u3_at_4 = float(L[4][0][2, 0])
assert abs(u2_at_2 - 0.25) < 1e-5, (
    f"after 2 steps u2 must hold exactly 0.250 of u1's signal (1/sqrt(4) in, "
    f"1/sqrt(4) out through shared item i2) — got {u2_at_2:.4f}"
)
assert abs(u3_at_2) < 1e-7, "u3 shares no item with u1 — it CANNOT hear at K=2"
assert abs(u3_at_4 - 0.0625) < 1e-5, (
    f"u3 first hears at K=4 with exactly 0.0625 — got {u3_at_4:.4f}"
)
assert abs(float(Pf[1, 0]) - (0 + 0 + 0.25 + 0 + 0.3125) / 5) < 1e-5, (
    "the readout is the MEAN over layers 0..K, the k=0 term included"
)
print("exercise 2 \\u2713 — the widget's numbers, reproduced by your propagation")'''

PROP_STUB = PROP_SOL.replace('''    ### BEGIN SOLUTION
    layers = [(P0, Q0)]
    P, Q = P0, Q0
    for _ in range(K):
        P, Q = torch.sparse.mm(A_ui, Q), torch.sparse.mm(A_iu, P)
        layers.append((P, Q))
    Pf = torch.stack([p for p, _ in layers]).mean(0)
    Qf = torch.stack([q for _, q in layers]).mean(0)
    return Pf, Qf, layers
    ### END SOLUTION''',
'''    # TODO (~7 lines): start layers = [(P0, Q0)]; K times, update
    # P <- A_ui @ Q and Q <- A_iu @ P **simultaneously** (compute both from
    # the previous pair!), appending each; final = mean over the K+1 layers.
    raise NotImplementedError''')


BPR_SOL = '''def bpr_loss(pu, qi, qj):
    """Bayesian Personalized Ranking: -log sigmoid(score_pos - score_neg),
    averaged. pu, qi, qj: (B, d) tensors — user, observed item, sampled item."""
    ### BEGIN SOLUTION
    x = (pu * (qi - qj)).sum(1)
    return -torch.nn.functional.logsigmoid(x).mean()
    ### END SOLUTION


# unit asserts a hand can verify
one = torch.ones(1, 2)
assert abs(float(bpr_loss(one, one * 3, one * 0)) -
           float(-torch.nn.functional.logsigmoid(torch.tensor(6.0)))) < 1e-6, \\
    "a confidently correct ordering must give a near-zero loss (here -log sig(6))"
assert abs(float(bpr_loss(one, one, one)) - 0.6931) < 1e-3, (
    "when positive and negative tie, the loss is exactly log 2 = 0.6931 — "
    "the model 'knows nothing' point, worth recognizing on sight"
)
assert float(bpr_loss(one, one * 0, one * 3)) > 3.0, \\
    "a confidently WRONG ordering must be punished hard"
print("exercise 3a \\u2713 — the loss that only asserts what the data supports")'''

BPR_STUB = BPR_SOL.replace('''    ### BEGIN SOLUTION
    x = (pu * (qi - qj)).sum(1)
    return -torch.nn.functional.logsigmoid(x).mean()
    ### END SOLUTION''',
'''    # TODO (2 lines): the pairwise score difference is (pu * (qi - qj)).sum(1);
    # return -logsigmoid(difference).mean().
    raise NotImplementedError''')


CELLS = [
    md("""# Lab 14 · The temporal wall, LightGCN, and the honest ladder

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/labs/lab14_recsys.ipynb)

**Week 14 · [lecture](https://lukmanovr.github.io/dkr/lectures/14-production.html) · ≈ 10 min of compute (CPU-friendly; GPU shaves the training)**

You build deployment's contract (the temporal split) and its one-line
auditor; reproduce the lecture widget's propagation masses exactly;
implement BPR from the definition; train the ladder's end rungs into
measured bands with the popularity baseline watching; and then run the
week's signature experiment — the same model on a random split — to
measure the protocol inflation with your own hands.

The model is LightGCN ([He et al., 2020](https://arxiv.org/abs/2002.02126)),
a GCN stripped to degree-normalized propagation plus a layer-mean readout —
the ablation-honest recommender that deleted its way to the top. The loss is
Bayesian Personalized Ranking ([Rendle et al., 2009](https://arxiv.org/abs/1205.2618)),
which only ever asserts the comparison the data supports: the observed item
over a sampled unobserved one.

### Goals
1. Temporal split with machine-checkable wall; sizes 80000/1519/1344;
   66 surviving test users.
2. `propagate` hitting the widget's masses: u2 = 0.250 at K=2,
   u3 = 0.0625 at K=4.
3. `bpr_loss` with the log-2 tie point; K=0 and K=3 trained into bands;
   popularity computed alongside.
4. The inflation experiment: your own random-vs-temporal ratio > 1.4.
"""),

    md("""## 0 · Setup

MovieLens-100k: 943 users, 1,682 movies, 100,000 timestamped ratings —
seven months of 1997–98. All interactions are treated as implicit
positives (the lecture's protocol)."""),

    code("""import os, sys, time

SMOKE = os.environ.get("SMOKE", "") == "1"

import numpy as np
import torch

torch.manual_seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists("data/ml-100k/u.data"):
    import io, zipfile, urllib.request
    os.makedirs("data", exist_ok=True)
    buf = urllib.request.urlopen(
        "https://files.grouplens.org/datasets/movielens/ml-100k.zip").read()
    zipfile.ZipFile(io.BytesIO(buf)).extractall("data")

raw = np.loadtxt("data/ml-100k/u.data", dtype=np.int64)
raw[:, 0] -= 1
raw[:, 1] -= 1
N_U, N_I = int(raw[:, 0].max()) + 1, int(raw[:, 1].max()) + 1
print(f"{len(raw)} interactions \\u00b7 {N_U} users \\u00b7 {N_I} items "
      f"\\u00b7 device {DEV} \\u00b7 SMOKE={SMOKE}")"""),

    md("""## 1 · The temporal wall and its auditor  *(exercise 1)*

A random split asks "can you interpolate a known past?"; deployment asks
"can you extrapolate into an unknown future?". The global temporal split is
the only protocol here that matches the deployed question — and its virtue
is that its no-leakage contract can be checked with one assert: every train
timestamp precedes every test timestamp.

**What you will implement:** `make_temporal_split(raw)` in the next cell. It
must sort the interactions by timestamp (a stable sort), cut them into
train/val/test at the 80% and 90% marks, and then drop any val or test row
whose user OR item never appears in train (cold-start rows are excluded and
counted, never silently kept). It returns the three integer arrays of
(user, item, rating, timestamp) rows.

**How you will know it worked:** the auditor asserts in the same cell check
that the wall holds (all train timestamps ≤ all val timestamps ≤ all test
timestamps) and that the sizes match the lecture exactly — 80000/1519/1344
rows and 66 surviving test users. When they pass, the cell prints
"exercise 1 ✓".

**Predict before you run:** 943 users have interactions. Roughly how many
will still be *evaluable* after an 80% time wall (they need train history
AND post-wall interactions on train-seen items)? Write down your number
before running the next cell."""),

    todo(SPLIT_SOL, SPLIT_STUB),

    md("""## 2 · LightGCN's propagation  *(exercise 2)*

LightGCN is a GCN with the feature transforms, nonlinearity, and self-loops
all deleted: what remains is degree-normalized propagation plus a mean over
layers. Before trusting your implementation with 100,000 interactions, you
will make it reproduce the lecture widget's hand-computed numbers on a
7-node toy graph — ground truth a human can verify.

**What you will implement:** `propagate(P0, Q0, K)` two cells below (the
next cell provides `build_norm_adj`, the normalized-adjacency builder; run
it first). Given user and item embedding tables, `propagate` must run K
rounds of bipartite propagation with the sparse operators `A_ui` and `A_iu`,
then return the layer-MEAN readout `(P_final, Q_final)` together with the
list of per-layer pairs, so the asserts can watch the signal spread layer by
layer.

**How you will know it worked:** the asserts trace one unit of signal placed
on user u1 through the toy graph and require the exact hand values — u2
holds 0.250 of it at K=2, u3 hears nothing until K=4 and then holds exactly
0.0625 — plus the layer-mean readout including the k=0 term. When they pass,
the cell prints "exercise 2 ✓".

The algorithm below specifies exactly what your `propagate` implementation
must do (it is the inner loop of the lecture's "LightGCN with BPR"
algorithm); follow it step by step:

**Input:** tables $P^{(0)}$ (users), $Q^{(0)}$ (items); sparse operators
`A_ui`, `A_iu`; depth $K$.
**Output:** final $P, Q$; the list of per-layer pairs.

1. layers $\\leftarrow [(P^{(0)}, Q^{(0)})]$
2. **for** $k = 1 \\dots K$ **do**
3. &nbsp;&nbsp;&nbsp;&nbsp;$P^{(k)} \\leftarrow$ `A_ui` $\\cdot\\, Q^{(k-1)}$ and $Q^{(k)} \\leftarrow$ `A_iu` $\\cdot\\, P^{(k-1)}$ —
   **both from the previous pair** (the trap: never feed the just-updated $Q^{(k)}$ into $P^{(k)}$)
4. &nbsp;&nbsp;&nbsp;&nbsp;append $(P^{(k)}, Q^{(k)})$ to layers
5. **end for**
6. **return** the MEAN over all $K{+}1$ layers ($k{=}0$ included) — and the layers"""),

    code("""def build_norm_adj(edges, n_u, n_i):
    \"\"\"Symmetric-normalized bipartite propagation matrices (sparse torch):
    A_ui maps item-vectors to users, A_iu maps user-vectors to items,
    both weighted 1/sqrt(d_u * d_i).\"\"\"
    ui = torch.tensor(np.asarray(edges).T[:2])
    du = torch.zeros(n_u); du.index_add_(0, ui[0], torch.ones(ui.shape[1]))
    di = torch.zeros(n_i); di.index_add_(0, ui[1], torch.ones(ui.shape[1]))
    w = 1.0 / (du[ui[0]].clamp(min=1).sqrt() * di[ui[1]].clamp(min=1).sqrt())
    A_ui = torch.sparse_coo_tensor(ui, w, (n_u, n_i)).coalesce()
    A_iu = torch.sparse_coo_tensor(ui.flip(0), w, (n_i, n_u)).coalesce()
    return A_ui, A_iu


print("builder ready — now make it move signal")"""),

    todo(PROP_SOL, PROP_STUB),

    md("""## 3 · BPR and the ladder  *(exercise 3)*

**What you will implement:** `bpr_loss(pu, qi, qj)` in the next cell — the
Bayesian Personalized Ranking loss. Given batched embeddings for a user, an
observed (positive) item, and a sampled (negative) item, compute the score
difference `(pu * (qi - qj)).sum(1)` and return the mean of
`-log sigmoid(difference)`. It is about two lines.

**How you will know it worked:** the asserts check three hand-verifiable
values — a confidently correct ordering gives a near-zero loss, a
confidently wrong one is punished hard, and when positive and negative tie
the loss is exactly log 2 ≈ 0.6931 (the "model knows nothing" point, worth
recognizing on sight). When they pass, the cell prints "exercise 3a ✓".

After that, the cell following it is provided: a training harness with the
BPR loop, early stopping on validation Recall@20, and a ranking evaluator
that masks train items. It runs YOUR `propagate` and YOUR `bpr_loss` at K=0
and at K=3, and also computes the one-line popularity baseline, because the
lecture's rule is: always run the baseline.

The algorithm below is the outer loop of the lecture's "LightGCN with BPR"
algorithm; the provided `train_lightgcn` runs it around your `bpr_loss`:

**Input:** train pairs $T$; tables $P^{(0)}, Q^{(0)}$; depth $K$; L2
strength $\\lambda$.
**Output:** trained tables; best-validation scores.

1. **for** each batch of observed pairs $(u, i) \\subset T$ **do**
2. &nbsp;&nbsp;&nbsp;&nbsp;sample one unobserved item $j$ per pair
3. &nbsp;&nbsp;&nbsp;&nbsp;$P, Q \\leftarrow$ `propagate`$(P^{(0)}, Q^{(0)}, K)$
4. &nbsp;&nbsp;&nbsp;&nbsp;$\\ell \\leftarrow -\\mathrm{mean}\\,\\log\\sigma\\big(P_u \\cdot (Q_i - Q_j)\\big) + \\lambda\\lVert\\Theta\\rVert^2$ — your `bpr_loss`, plus L2
5. &nbsp;&nbsp;&nbsp;&nbsp;gradient step — the only parameters are $P^{(0)}, Q^{(0)}$
6. **end for**
7. every 20 epochs: validation Recall@20; keep the best checkpoint; stop early"""),

    todo(BPR_SOL, BPR_STUB),

    code("""def eval_ranking(scores, tr_arr, te_arr, K=20):
    \"\"\"Mask train items, rank everything else, average Recall@K over users
    with test positives (the lecture's protocol; NDCG in the stretch).\"\"\"
    s = scores.clone()
    s[tr_arr[:, 0], tr_arr[:, 1]] = -1e9
    topk = torch.topk(s, K, dim=1).indices.cpu().numpy()
    pos = {}
    for u, i in te_arr[:, :2]:
        pos.setdefault(int(u), set()).add(int(i))
    recs = []
    for u, P in pos.items():
        hits = sum(1 for it in topk[u] if it in P)
        recs.append(hits / min(len(P), K))
    return float(np.mean(recs))


def train_lightgcn(tr_arr, va_arr, K_layers, name=""):
    global A_ui, A_iu
    torch.manual_seed(0)
    A_ui, A_iu = build_norm_adj(tr_arr[:, :2], N_U, N_I)
    A_ui, A_iu = A_ui.to(DEV), A_iu.to(DEV)
    P = torch.nn.Parameter(torch.randn(N_U, 64, device=DEV) * 0.1)
    Q = torch.nn.Parameter(torch.randn(N_I, 64, device=DEV) * 0.1)
    opt = torch.optim.Adam([P, Q], lr=1e-2)
    tru = torch.tensor(tr_arr[:, 0], device=DEV)
    tri = torch.tensor(tr_arr[:, 1], device=DEV)
    gen = torch.Generator().manual_seed(0)
    epochs = 40 if SMOKE else 300
    best_va, best_scores, patience = -1, None, 0
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(tru), generator=gen).to(DEV)
        for k in range(0, len(perm), 8192):
            idx = perm[k: k + 8192]
            u, ipos = tru[idx], tri[idx]
            ineg = torch.randint(0, N_I, (len(idx),), generator=gen).to(DEV)
            Pf, Qf, _ = propagate(P, Q, K_layers)
            loss = bpr_loss(Pf[u], Qf[ipos], Qf[ineg]) + 1e-4 * (
                P[u].pow(2).sum() + Q[ipos].pow(2).sum() + Q[ineg].pow(2).sum()
            ) / len(idx)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                Pf, Qf, _ = propagate(P, Q, K_layers)
                r = eval_ranking((Pf @ Qf.T).cpu(), tr_arr, va_arr)
            if r > best_va:
                best_va, patience = r, 0
                best_scores = (Pf @ Qf.T).cpu()
            else:
                patience += 1
                if patience >= 3:
                    break
    print(f"{name}: best val R@20 {best_va:.4f}  [{time.time() - t0:.0f}s]")
    return best_scores


# popularity: the one-line baseline the lecture insists on
pop = torch.zeros(N_I)
pop.index_add_(0, torch.tensor(tr[:, 1]), torch.ones(len(tr)))
pop_r = eval_ranking(pop.unsqueeze(0).repeat(N_U, 1), tr, te)
print(f"popularity R@20 (temporal): {pop_r:.4f} — remember this number")"""),

    md("""**Predict before you run:** the lecture's full-budget ladder put K=0 at
0.156 — *below* popularity — and K=3 at 0.195. Will your run (same seed,
same protocol) land in those neighborhoods? Which side of popularity will
YOUR K=0 fall on? Write down your prediction before running the next
cell."""),

    code("""sc0 = train_lightgcn(tr, va, 0, "LightGCN K=0 (= BPR-MF)")
r0 = eval_ranking(sc0, tr, te)
sc3 = train_lightgcn(tr, va, 3, "LightGCN K=3")
r3 = eval_ranking(sc3, tr, te)
print(f"\\ntemporal test R@20:  popularity {pop_r:.4f} \\u00b7 K=0 {r0:.4f} "
      f"\\u00b7 K=3 {r3:.4f}")

if SMOKE:
    assert r0 > 0.02 and r3 > 0.02, "SMOKE sanity: both models must beat noise"
else:
    assert 0.10 < r0 < 0.20, (
        f"K=0 must land in the measured band (reference 0.156) — got {r0:.4f}"
    )
    assert 0.15 < r3 < 0.24, (
        f"K=3 must land in the measured band (reference 0.195) — got {r3:.4f}"
    )
    assert r3 > r0 + 0.015, (
        f"the graph must pay: K=3 clearly above K=0 (reference gap 0.039) — "
        f"got {r3:.4f} vs {r0:.4f}"
    )
    assert pop_r > r0, (
        "the humbling is part of the lesson: on this split, popularity beats "
        "pure MF — if your K=0 beat popularity, check the split's wall"
    )
print("exercise 3 \\u2713 — the ladder, on the honest split, with the baseline watching")"""),

    md("""## 4 · The inflation experiment  *(exercise 4)*

This exercise measures how much a leaky protocol inflates a result. Nothing
to implement: the next cell (provided) trains the SAME K=3 model with the
SAME budget on a RANDOM 80/10/10 split of the SAME data, then divides the
random-split Recall@20 by your temporal-split Recall@20 from exercise 3.
The lecture measured an inflation of ×1.77; the assert demands your ratio
exceed 1.4, a margin wide enough that hardware and seed noise cannot rescue
a leaky protocol. When it passes, the cell prints "exercise 4 ✓".

**Predict before you run:** write down your expected inflation ratio before
running the next cell."""),

    code("""rng = np.random.default_rng(0)
perm = rng.permutation(len(raw))
d = raw[perm]
tr_r = d[: int(0.8 * len(d))]
va_r = d[int(0.8 * len(d)): int(0.9 * len(d))]
te_r = d[int(0.9 * len(d)):]
seen_u, seen_i = set(tr_r[:, 0].tolist()), set(tr_r[:, 1].tolist())
keep = lambda X: X[[u in seen_u and i in seen_i for u, i in zip(X[:, 0], X[:, 1])]]
va_r, te_r = keep(va_r), keep(te_r)
print(f"random split: {len(tr_r)}/{len(va_r)}/{len(te_r)} \\u00b7 "
      f"{len(np.unique(te_r[:, 0]))} test users (almost everyone — no wall)")

sc3r = train_lightgcn(tr_r, va_r, 3, "LightGCN K=3 (random split)")
r3r = eval_ranking(sc3r, tr_r, te_r)
inflation = r3r / r3
print(f"\\nrandom-split test R@20: {r3r:.4f}")
print(f"INFLATION: {r3r:.4f} / {r3:.4f} = \\u00d7{inflation:.2f}  (lecture: \\u00d71.77)")

assert r3r > r3, "the random split must flatter the model"
if not SMOKE:
    assert inflation > 1.4, (
        f"the protocol inflation must be unmistakable (lecture \\u00d71.77; we "
        f"demand > 1.4) — got \\u00d7{inflation:.2f}; if lower, check that your "
        f"temporal split's wall actually holds"
    )
print("exercise 4 \\u2713 — same model, same data, same budget: the number "
      "nearly doubled. You will never un-know this.")"""),

    md("""### Your claims paragraph *(graded — write it in this cell)*

Write three claims in THIS cell, replacing the placeholder below. Each claim
is 1–2 sentences that state the claim, cite the specific numbers from your
own cell outputs that support it, and state its scope. The three claims must
be:

1. **Popularity:** cite your popularity and K=0 numbers from exercise 3, and
   explain what mechanism lets a sorted list beat trained personalization on
   a temporal split.
2. **Co-consumption:** cite your exercise-2 propagation masses (who hears
   whom at which K), and explain how the ladder's K=0 → K=3 gain is that
   same mechanism operating at scale.
3. **Protocol:** cite your measured inflation ratio from exercise 4, and say
   which of your two K=3 numbers deployment would actually deliver.

*(your claims here)*

## 5 · Stretch (optional, ungraded)

1. **Leave-last-out.** Implement the third protocol; place its number
   between the other two (lecture: 0.210) and explain its residual leak.
2. **NDCG's disagreement.** Add NDCG@20 to the evaluator; check whether
   the interior of YOUR ladder reorders (the lecture's did).
3. **The K ablation.** Fill in K=1 and K=2; is your ladder monotone in
   recall like the lecture's?
4. **A mini-memory.** For each user keep a GRU state over their event
   sequence and use it to re-rank the top 50: does recency help on the
   temporal split?
5. **The sampler audit.** Write `neighbors_before(u, t, n)` and the
   assert from the lecture's pitfall list; plant a peeking bug and watch
   the assert catch it.

## 6 · Reflection (answer in this cell, 2–4 sentences each)

**R1.** Your exercise-1 auditor is one assert; Week 12's link-prediction
auditor was twelve. What structural property of the temporal split makes
its no-leakage guarantee so much cheaper to check, and what does that
suggest about choosing protocols in general?

**R2.** Only 66 users were evaluable, and the lecture called the resulting
numbers "thin, awkward, honest." A colleague proposes fixing the thinness
by evaluating on the random split's 943 users instead. Write the two-
sentence reply that names exactly what would be traded away.

**R3.** LightGCN deleted parameters and improved; Week 10's lookup matched
its encoder; Week 13's attention bought nothing at matched budget. State
the general principle these three measurements share, and one situation
where you'd bet the OPPOSITE direction (more capacity paying).

*(your answers here)*

## What to submit

One executed notebook on Moodle, run **full** (not SMOKE): four checks ✓,
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
