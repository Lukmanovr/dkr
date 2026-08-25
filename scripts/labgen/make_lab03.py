#!/usr/bin/env python
"""Generate Lab 3 (shallow embeddings: walks -> skip-gram -> SGNS -> Cora,
plus honest link prediction) — student and solution notebooks.

    python scripts/labgen/make_lab03.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "labs" / "lab03_embeddings.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "labs" / "lab03_embeddings.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(text, stub): return ("code", text, stub)


CELLS = [
    md("""# Lab 3 · DeepWalk from scratch — walks, skip-gram, and honest evaluation

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/labs/lab03_embeddings.ipynb)

**Week 3 · [lecture](https://lukmanovr.github.io/dkr/lectures/03-embeddings.html) · ≈ 25 min of compute (free Colab or CPU — no GPU needed)**

You will build the entire pipeline the lecture derived — walks → windows → SGNS — with
no embedding library anywhere, train it on the karate club and then on Cora, and end
with two honest measurements: does *learned position* beat Lab 2's community one-hots,
and exactly how many AUC points does the classic link-prediction leak buy?

This is representation learning at its most transparent: instead of designing node
features, you let an objective *learn* them from random-walk co-occurrence. The recipe
is DeepWalk's ([Perozzi et al., 2014](https://arxiv.org/abs/1403.6652)) with the
negative-sampling objective it borrowed from word2vec
([Mikolov et al., 2013](https://arxiv.org/abs/1310.4546)); node2vec's biased walker
([Grover & Leskovec, 2016](https://arxiv.org/abs/1607.00653)) waits in the stretch
goals.

### Goals
1. Implement random walks and skip-gram pair extraction, verified against hand counts.
2. Implement the SGNS loss in PyTorch and train real embeddings (two tables, k negatives).
3. Scale your own code to Cora and beat the 62% structure baseline from Lab 2.
4. Run link prediction the leaky way and the clean way, and put a number on the difference.
"""),

    md("""## 0 · Setup  *(the contract — pinned, SMOKE-aware)*"""),

    code("""import os, sys, random
from collections import Counter

SMOKE = os.environ.get("SMOKE", "") == "1"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    %pip install -q torch_geometric==2.8.0.post1

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SEED = 41
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"torch {torch.__version__} · networkx {nx.__version__} — environment OK")"""),

    md("""## 1 · Walks  *(exercise 1 — skill: the corpus generator)*

The "sentences". Exercises 1 and 2 together implement the lecture's Algorithm 1 — code
against this spec, not a vibe:

> **Algorithm · Walk corpus → skip-gram pairs** (DeepWalk)
>
> **Input:** graph $G$; walks per node $\\gamma$; walk length $K$; window $w$; seed $s$.
> **Output:** list $\\mathcal{D}$ of (center, context) pairs.
>
> 1. rng ← Random($s$); $W$ ← empty list; $\\mathcal{D}$ ← empty list
> 2. **for** round $= 1, \\ldots, \\gamma$: **for** each start node $v$ in **sorted** order:
> 3. &nbsp;&nbsp;&nbsp;&nbsp;walk ← $(v)$; **while** $|\\text{walk}| < K$ and last(walk) has neighbors: append a neighbor chosen uniformly by rng
> 4. &nbsp;&nbsp;&nbsp;&nbsp;append walk to $W$
> 5. **for** each walk in $W$, each position $i$, each $j \\neq i$ with $|i-j| \\le w$: add $(\\text{walk}_i, \\text{walk}_j)$ to $\\mathcal{D}$
> 6. **return** $\\mathcal{D}$ — exactly $|V| \\cdot \\gamma \\cdot (2wK - w(w{+}1))$ pairs when no walk ends early

Exercise 1 is steps 1–4 ($\\gamma$ = `num_walks`, $K$ = `length`); exercise 2 is step 5.
The determinism contract (one `random.Random(seed)` driving everything, nodes iterated
in sorted order) is part of the exercise, because reproducible corpora are what make
every later assert possible.
"""),

    todo("""CAST = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (2, 5), (4, 5)])  # A..F = 0..5


def random_walks(G: nx.Graph, num_walks: int, length: int, seed: int) -> list:
    \"\"\"num_walks walks of `length` nodes from EVERY node of G (sorted order),
    driven by one random.Random(seed). Returns a list of lists of nodes.
    A walk stuck at an isolated node ends early.\"\"\"
    ### BEGIN SOLUTION
    rng = random.Random(seed)
    walks = []
    for _ in range(num_walks):
        for s in sorted(G):
            walk = [s]
            while len(walk) < length:
                nbrs = list(G[walk[-1]])
                if not nbrs:
                    break
                walk.append(rng.choice(nbrs))
            walks.append(walk)
    return walks
    ### END SOLUTION


walks = random_walks(CAST, num_walks=4, length=8, seed=7)
assert len(walks) == 24, f"got {len(walks)} walks, expected 4 rounds × 6 start nodes = 24"
assert all(len(w) == 8 for w in walks), "every cast-graph walk should reach full length (no dead ends here)"
assert sorted(w[0] for w in walks) == sorted(list(range(6)) * 4), (
    "each node must start exactly num_walks walks — iterate rounds outer, sorted(G) inner"
)
for w in walks:
    for a, b in zip(w, w[1:]):
        assert CAST.has_edge(a, b), f"walk step {a}->{b} is not an edge"
assert walks == random_walks(CAST, 4, 8, seed=7), "same seed must reproduce the identical corpus"
print("exercise 1 ✓ — deterministic corpus:", len(walks), "walks; first walk:",
      " → ".join("ABCDEF"[v] for v in walks[0]))""",
         stub="""CAST = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (2, 5), (4, 5)])  # A..F = 0..5


def random_walks(G: nx.Graph, num_walks: int, length: int, seed: int) -> list:
    \"\"\"num_walks walks of `length` nodes from EVERY node of G (sorted order),
    driven by one random.Random(seed). Returns a list of lists of nodes.
    A walk stuck at an isolated node ends early.\"\"\"
    # TODO: rng = random.Random(seed); outer loop over rounds, inner over sorted(G);
    # extend with rng.choice of the current node's neighbors. ~10 lines.
    raise NotImplementedError("implement the random-walk corpus generator")


walks = random_walks(CAST, num_walks=4, length=8, seed=7)
assert len(walks) == 24, f"got {len(walks)} walks, expected 4 rounds × 6 start nodes = 24"
assert all(len(w) == 8 for w in walks), "every cast-graph walk should reach full length (no dead ends here)"
assert sorted(w[0] for w in walks) == sorted(list(range(6)) * 4), (
    "each node must start exactly num_walks walks — iterate rounds outer, sorted(G) inner"
)
for w in walks:
    for a, b in zip(w, w[1:]):
        assert CAST.has_edge(a, b), f"walk step {a}->{b} is not an edge"
assert walks == random_walks(CAST, 4, 8, seed=7), "same seed must reproduce the identical corpus"
print("exercise 1 ✓ — deterministic corpus:", len(walks), "walks; first walk:",
      " → ".join("ABCDEF"[v] for v in walks[0]))"""),

    md("""## 2 · Windows  *(exercise 2 — skill: pairs, checked against the lecture)*

Slide the window, emit (center, context) pairs. The first assert is the lecture's Q2
example — if you did that self-check, you already know the answer by hand.
"""),

    todo("""def skipgram_pairs(walks: list, window: int) -> list:
    \"\"\"All (center, context) pairs with 1 <= |i - j| <= window, walk by walk,
    in reading order (i ascending, then j ascending).\"\"\"
    ### BEGIN SOLUTION
    pairs = []
    for w in walks:
        for i, c in enumerate(w):
            for j in range(max(0, i - window), min(len(w), i + window + 1)):
                if j != i:
                    pairs.append((c, w[j]))
    return pairs
    ### END SOLUTION


q2 = skipgram_pairs([[1, 2, 4, 5]], window=1)          # the lecture's walk (B, C, E, F)
assert q2 == [(1, 2), (2, 1), (2, 4), (4, 2), (4, 5), (5, 4)], (
    f"walk (B,C,E,F), w=1 must give the lecture's six pairs in reading order; got {q2}. "
    f"Check: BOTH directions of each adjacency in the walk, no self-pairs, no wrap-around."
)
n_pairs = len(skipgram_pairs(walks, window=2))
expected = 24 * (2 * 2 * 8 - 2 * 3)                    # per walk: 2wK − w(w+1)
assert n_pairs == expected, (
    f"{n_pairs} pairs from the cast corpus, expected exactly {expected} "
    f"(per length-8 walk with w=2: 2·2·8 − 2·3 = 26). Off-by-one in the window edges?"
)
print(f"exercise 2 ✓ — {n_pairs} pairs from 24 walks; the formula 2wK − w(w+1) checks out")""",
         stub="""def skipgram_pairs(walks: list, window: int) -> list:
    \"\"\"All (center, context) pairs with 1 <= |i - j| <= window, walk by walk,
    in reading order (i ascending, then j ascending).\"\"\"
    # TODO: three loops — walks, centers, window positions (clipped to the walk).
    raise NotImplementedError("emit the skip-gram pairs")


q2 = skipgram_pairs([[1, 2, 4, 5]], window=1)          # the lecture's walk (B, C, E, F)
assert q2 == [(1, 2), (2, 1), (2, 4), (4, 2), (4, 5), (5, 4)], (
    f"walk (B,C,E,F), w=1 must give the lecture's six pairs in reading order; got {q2}. "
    f"Check: BOTH directions of each adjacency in the walk, no self-pairs, no wrap-around."
)
n_pairs = len(skipgram_pairs(walks, window=2))
expected = 24 * (2 * 2 * 8 - 2 * 3)                    # per walk: 2wK − w(w+1)
assert n_pairs == expected, (
    f"{n_pairs} pairs from the cast corpus, expected exactly {expected} "
    f"(per length-8 walk with w=2: 2·2·8 − 2·3 = 26). Off-by-one in the window edges?"
)
print(f"exercise 2 ✓ — {n_pairs} pairs from 24 walks; the formula 2wK − w(w+1) checks out")"""),

    md("""## 3 · The SGNS loss  *(exercise 3 — skill: the lecture's formula, in torch)*

Two tables (center and context, as word2vec really has), $k$ negatives per pair, and
the loss you derived:
$-\\left[\\log\\sigma(\\mathbf{z}_u\\!\\cdot\\!\\mathbf{z}'_v) + \\sum_k \\log\\sigma(-\\mathbf{z}_u\\!\\cdot\\!\\mathbf{z}'_{n_k})\\right]$.
Your loss plus one optimizer step is the lecture's Algorithm 2 — autograd computes the
pulls and pushes the spec writes out by hand:

> **Algorithm · One SGNS update** ([Mikolov et al., 2013](https://arxiv.org/abs/1310.4546))
>
> **Input:** pair $(u, v) \\in \\mathcal{D}$; center table $\\mathbf{Z}$, context table
> $\\mathbf{Z}'$; negatives $k$; noise distribution $P_n$ (uniform in this lab);
> learning rate $\\eta$. **Output:** updated rows of $\\mathbf{Z}$, $\\mathbf{Z}'$.
>
> 1. draw fake contexts $n_1, \\ldots, n_k \\sim P_n$
> 2. $g \\leftarrow 1 - \\sigma(\\mathbf{z}_u \\cdot \\mathbf{z}'_v)$ — pull strength
> 3. $\\Delta \\leftarrow g\\,\\mathbf{z}'_v$; then $\\mathbf{z}'_v \\leftarrow \\mathbf{z}'_v + \\eta g \\mathbf{z}_u$
> 4. **for** $i = 1, \\ldots, k$: $h \\leftarrow \\sigma(\\mathbf{z}_u \\cdot \\mathbf{z}'_{n_i})$ — push strength;
>    $\\Delta \\leftarrow \\Delta - h\\,\\mathbf{z}'_{n_i}$; then $\\mathbf{z}'_{n_i} \\leftarrow \\mathbf{z}'_{n_i} - \\eta h \\mathbf{z}_u$
> 5. $\\mathbf{z}_u \\leftarrow \\mathbf{z}_u + \\eta \\Delta$ — exactly $k + 2$ rows touched

Use `F.logsigmoid` — the numerically stable form; `torch.log(torch.sigmoid(...))`
overflows to `-inf` for confident wrong scores, and the assert checks you didn't.
"""),

    todo("""def sgns_loss(z_u, z_v, z_neg):
    \"\"\"z_u, z_v: (B, d) center/context vectors. z_neg: (B, k, d) negative context
    vectors. Return the mean SGNS loss (a scalar to MINIMIZE).\"\"\"
    ### BEGIN SOLUTION
    pos = F.logsigmoid((z_u * z_v).sum(dim=1))
    neg = F.logsigmoid(-(z_neg * z_u.unsqueeze(1)).sum(dim=2)).sum(dim=1)
    return -(pos + neg).mean()
    ### END SOLUTION


def train_sgns(pairs, n, d=16, k=5, epochs=3, batch=1024, lr=0.05, seed=0):
    \"\"\"Provided: batching + optimizer around YOUR loss. Returns the center table.\"\"\"
    g = torch.Generator().manual_seed(seed)
    center = torch.nn.Embedding(n, d)
    context = torch.nn.Embedding(n, d)
    torch.nn.init.normal_(center.weight, std=0.1, generator=g)
    torch.nn.init.normal_(context.weight, std=0.1, generator=g)
    opt = torch.optim.Adam(list(center.parameters()) + list(context.parameters()), lr=lr)
    P = torch.tensor(pairs, dtype=torch.long)
    epoch_losses = []
    for ep in range(epochs):
        perm = torch.randperm(len(P), generator=g)
        total = 0.0
        for i in range(0, len(P), batch):
            idx = perm[i:i + batch]
            u, v = P[idx, 0], P[idx, 1]
            negs = torch.randint(0, n, (len(idx), k), generator=g)
            loss = sgns_loss(center(u), context(v), context(negs))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        epoch_losses.append(total / len(P))
    return center.weight.detach(), epoch_losses


# sanity on a tiny batch: finite, positive, and stable for extreme scores
zu = torch.tensor([[10.0, 0.0], [-10.0, 0.0]])
zv = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
zn = torch.zeros(2, 3, 2)
val = sgns_loss(zu, zv, zn)
assert torch.isfinite(val), (
    "loss is not finite on extreme scores — use F.logsigmoid, not log(sigmoid(...))"
)
assert val.item() > 0, "SGNS loss to MINIMIZE must be positive (it is a negated log-likelihood)"

KARATE = nx.karate_club_graph()
k_pairs = skipgram_pairs(random_walks(KARATE, num_walks=10, length=12, seed=3), window=3)
Z, losses = train_sgns(k_pairs, n=34, d=16, epochs=3, seed=0)
assert losses[-1] < losses[0] - 0.1, (
    f"loss went {losses[0]:.3f} → {losses[-1]:.3f}; it should drop clearly. If it "
    f"EXPLODES, check the sign (return the NEGATED mean log-likelihood); if it stalls "
    f"at ln 2 ≈ 0.693 per term, your negatives may be getting the positive's sign."
)

truth = np.array([1 if KARATE.nodes[v]["club"] == "Mr. Hi" else 0 for v in sorted(KARATE)])
Znp = Z.numpy()
cen0, cen1 = Znp[truth == 1].mean(0), Znp[truth == 0].mean(0)
d0 = ((Znp - cen0) ** 2).sum(1); d1 = ((Znp - cen1) ** 2).sum(1)
sep = int(((d0 < d1) == (truth == 1)).sum())
assert sep >= 30, (
    f"nearest-centroid separation {sep}/34 — trained embeddings should recover the "
    f"factions almost perfectly. If ~17 (chance), the loss's pull/push signs are flipped."
)
print(f"exercise 3 ✓ — loss {losses[0]:.3f} → {losses[-1]:.3f}, faction separation {sep}/34")""",
         stub="""def sgns_loss(z_u, z_v, z_neg):
    \"\"\"z_u, z_v: (B, d) center/context vectors. z_neg: (B, k, d) negative context
    vectors. Return the mean SGNS loss (a scalar to MINIMIZE).\"\"\"
    # TODO: positive term logsigmoid(sum z_u*z_v); negative term logsigmoid of MINUS
    # the (B, k) dot products, summed over k; return the negated mean. 3 lines.
    raise NotImplementedError("implement the SGNS loss")


def train_sgns(pairs, n, d=16, k=5, epochs=3, batch=1024, lr=0.05, seed=0):
    \"\"\"Provided: batching + optimizer around YOUR loss. Returns the center table.\"\"\"
    g = torch.Generator().manual_seed(seed)
    center = torch.nn.Embedding(n, d)
    context = torch.nn.Embedding(n, d)
    torch.nn.init.normal_(center.weight, std=0.1, generator=g)
    torch.nn.init.normal_(context.weight, std=0.1, generator=g)
    opt = torch.optim.Adam(list(center.parameters()) + list(context.parameters()), lr=lr)
    P = torch.tensor(pairs, dtype=torch.long)
    epoch_losses = []
    for ep in range(epochs):
        perm = torch.randperm(len(P), generator=g)
        total = 0.0
        for i in range(0, len(P), batch):
            idx = perm[i:i + batch]
            u, v = P[idx, 0], P[idx, 1]
            negs = torch.randint(0, n, (len(idx), k), generator=g)
            loss = sgns_loss(center(u), context(v), context(negs))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        epoch_losses.append(total / len(P))
    return center.weight.detach(), epoch_losses


# sanity on a tiny batch: finite, positive, and stable for extreme scores
zu = torch.tensor([[10.0, 0.0], [-10.0, 0.0]])
zv = torch.tensor([[10.0, 0.0], [10.0, 0.0]])
zn = torch.zeros(2, 3, 2)
val = sgns_loss(zu, zv, zn)
assert torch.isfinite(val), (
    "loss is not finite on extreme scores — use F.logsigmoid, not log(sigmoid(...))"
)
assert val.item() > 0, "SGNS loss to MINIMIZE must be positive (it is a negated log-likelihood)"

KARATE = nx.karate_club_graph()
k_pairs = skipgram_pairs(random_walks(KARATE, num_walks=10, length=12, seed=3), window=3)
Z, losses = train_sgns(k_pairs, n=34, d=16, epochs=3, seed=0)
assert losses[-1] < losses[0] - 0.1, (
    f"loss went {losses[0]:.3f} → {losses[-1]:.3f}; it should drop clearly. If it "
    f"EXPLODES, check the sign (return the NEGATED mean log-likelihood); if it stalls "
    f"at ln 2 ≈ 0.693 per term, your negatives may be getting the positive's sign."
)

truth = np.array([1 if KARATE.nodes[v]["club"] == "Mr. Hi" else 0 for v in sorted(KARATE)])
Znp = Z.numpy()
cen0, cen1 = Znp[truth == 1].mean(0), Znp[truth == 0].mean(0)
d0 = ((Znp - cen0) ** 2).sum(1); d1 = ((Znp - cen1) ** 2).sum(1)
sep = int(((d0 < d1) == (truth == 1)).sum())
assert sep >= 30, (
    f"nearest-centroid separation {sep}/34 — trained embeddings should recover the "
    f"factions almost perfectly. If ~17 (chance), the loss's pull/push signs are flipped."
)
print(f"exercise 3 ✓ — loss {losses[0]:.3f} → {losses[-1]:.3f}, faction separation {sep}/34")"""),

    md("""### See it *(provided — the lecture's Figure 3, from your own training run)*"""),

    code("""from sklearn.decomposition import PCA

xy = PCA(n_components=2, random_state=0).fit_transform(Znp)
plt.figure(figsize=(6.5, 4.5))
for cls, color, label in [(1, "#d9a62e", "Mr. Hi's faction"), (0, "#0f8377", "the Officer's")]:
    m = truth == cls
    plt.scatter(xy[m, 0], xy[m, 1], c=color, s=60, label=label, alpha=0.85)
for b in (2, 8):
    plt.scatter(*xy[b], s=150, facecolors="none", edgecolors="#cf4a30", linewidths=2)
plt.legend(); plt.title("your SGNS embeddings (d=16 → PCA) — rings: boundary members 2, 8")
plt.axis("off"); plt.show()
print("compare with the lecture's widget: same algorithm, same graph, same geometry")"""),

    md("""## 4 · Scale it to Cora  *(exercise 4 — skill: your pipeline, for real)*

No new algorithms — wire YOUR three functions together at Cora scale. The count assert
is exact because the pair formula is exact: for walks of length 20 with window 5, each
walk emits $2wK − w(w{+}1) = 170$ pairs.

**Predict before you run:** Lab 2's best structure-only number was 61.7% (statistics +
community one-hots). Learned 64-d walk embeddings, same classifier — higher or lower,
and by how much?
"""),

    todo("""from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]
Gc = to_networkx(data, to_undirected=True)
NUM_WALKS, LENGTH, WINDOW = (2, 10, 3) if SMOKE else (5, 20, 5)


def cora_pairs():
    \"\"\"Build the Cora skip-gram corpus with YOUR random_walks + skipgram_pairs,
    using NUM_WALKS, LENGTH, WINDOW and seed=1.\"\"\"
    ### BEGIN SOLUTION
    return skipgram_pairs(random_walks(Gc, NUM_WALKS, LENGTH, seed=1), WINDOW)
    ### END SOLUTION


pairs_c = cora_pairs()
per_walk = 2 * WINDOW * LENGTH - WINDOW * (WINDOW + 1)
expected = Gc.number_of_nodes() * NUM_WALKS * per_walk
assert len(pairs_c) == expected, (
    f"{len(pairs_c):,} pairs, expected exactly {expected:,} "
    f"(= n × num_walks × [2wK − w(w+1)]) — are you using seed=1 and the module-level "
    f"NUM_WALKS/LENGTH/WINDOW? (Cora has no dead-end nodes, so walks never end early.)"
)
print(f"corpus: {len(pairs_c):,} pairs — training (a couple of minutes on CPU)...")

Zc, losses_c = train_sgns(pairs_c, n=Gc.number_of_nodes(), d=64, lr=0.025,
                          epochs=2 if SMOKE else 8, batch=4096, seed=0)
assert losses_c[-1] < losses_c[0], "Cora SGNS loss must decrease"

y = data.y.numpy()
train_m, test_m = data.train_mask.numpy(), data.test_mask.numpy()
# L2-normalize before probing: SGNS makes a vector's NORM track walk frequency
# (how often the node was a center) while its DIRECTION carries position — the
# signal we want. Without this line the probe loses ~4 points to degree noise.
Zn = Zc.numpy()
Zn = Zn / (np.linalg.norm(Zn, axis=1, keepdims=True) + 1e-9)
clf = LogisticRegression(max_iter=3000).fit(Zn[train_m], y[train_m])
acc = clf.score(Zn[test_m], y[test_m])
print(f"walk embeddings + logistic regression: {acc:.1%}")
floor = 0.45 if SMOKE else 0.65
assert acc > floor, (
    f"accuracy {acc:.1%} under the {floor:.0%} bar — with the given hyperparameters "
    f"this lands around 68% (SMOKE: lower, the corpus is tiny). Check that rows of Z "
    f"are indexed by node id (they are, if your walker used the node ids as tokens)."
)
print(f"exercise 4 ✓ — the ladder so far: statistics 18.5% → +communities 61.7% → learned position {acc:.1%}")
print("no hand-designed features anywhere — the vectors were learned from walks alone")""",
         stub="""from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]
Gc = to_networkx(data, to_undirected=True)
NUM_WALKS, LENGTH, WINDOW = (2, 10, 3) if SMOKE else (5, 20, 5)


def cora_pairs():
    \"\"\"Build the Cora skip-gram corpus with YOUR random_walks + skipgram_pairs,
    using NUM_WALKS, LENGTH, WINDOW and seed=1.\"\"\"
    # TODO: one line — compose your two functions from exercises 1 and 2.
    raise NotImplementedError("build the Cora corpus")


pairs_c = cora_pairs()
per_walk = 2 * WINDOW * LENGTH - WINDOW * (WINDOW + 1)
expected = Gc.number_of_nodes() * NUM_WALKS * per_walk
assert len(pairs_c) == expected, (
    f"{len(pairs_c):,} pairs, expected exactly {expected:,} "
    f"(= n × num_walks × [2wK − w(w+1)]) — are you using seed=1 and the module-level "
    f"NUM_WALKS/LENGTH/WINDOW? (Cora has no dead-end nodes, so walks never end early.)"
)
print(f"corpus: {len(pairs_c):,} pairs — training (a couple of minutes on CPU)...")

Zc, losses_c = train_sgns(pairs_c, n=Gc.number_of_nodes(), d=64, lr=0.025,
                          epochs=2 if SMOKE else 8, batch=4096, seed=0)
assert losses_c[-1] < losses_c[0], "Cora SGNS loss must decrease"

y = data.y.numpy()
train_m, test_m = data.train_mask.numpy(), data.test_mask.numpy()
# L2-normalize before probing: SGNS makes a vector's NORM track walk frequency
# (how often the node was a center) while its DIRECTION carries position — the
# signal we want. Without this line the probe loses ~4 points to degree noise.
Zn = Zc.numpy()
Zn = Zn / (np.linalg.norm(Zn, axis=1, keepdims=True) + 1e-9)
clf = LogisticRegression(max_iter=3000).fit(Zn[train_m], y[train_m])
acc = clf.score(Zn[test_m], y[test_m])
print(f"walk embeddings + logistic regression: {acc:.1%}")
floor = 0.45 if SMOKE else 0.65
assert acc > floor, (
    f"accuracy {acc:.1%} under the {floor:.0%} bar — with the given hyperparameters "
    f"this lands around 68% (SMOKE: lower, the corpus is tiny). Check that rows of Z "
    f"are indexed by node id (they are, if your walker used the node ids as tokens)."
)
print(f"exercise 4 ✓ — the ladder so far: statistics 18.5% → +communities 61.7% → learned position {acc:.1%}")
print("no hand-designed features anywhere — the vectors were learned from walks alone")"""),

    md("""## 5 · Link prediction: the leak, measured  *(exercise 5 — skill: honest AUC)*

The lecture's Pitfall 1, quantified on your own embeddings. We hold out 10% of Cora's
edges plus an equal number of sampled non-edges. Two contestants score the same test
pairs by dot product:

- **leaky** — the exercise-4 embeddings, whose walks strolled across the test edges;
- **clean** — embeddings retrained by the same recipe on the graph *with test edges
  removed*.

**Predict before you run:** which AUC is higher, and by roughly how much?
"""),

    todo("""split_rng = random.Random(7)
all_edges = sorted(Gc.edges)
test_edges = split_rng.sample(all_edges, k=len(all_edges) // 10)
G_train = Gc.copy()
G_train.remove_edges_from(test_edges)

neg_pairs = []
n_nodes = Gc.number_of_nodes()
while len(neg_pairs) < len(test_edges):
    u, v = split_rng.randrange(n_nodes), split_rng.randrange(n_nodes)
    if u != v and not Gc.has_edge(u, v):
        neg_pairs.append((u, v))

test_pairs = test_edges + neg_pairs
test_labels = np.array([1] * len(test_edges) + [0] * len(neg_pairs))


def dot_auc(Z: np.ndarray, pairs: list, labels: np.ndarray) -> float:
    \"\"\"AUC of the dot-product scores Z[u]·Z[v] against the pair labels.\"\"\"
    ### BEGIN SOLUTION
    scores = np.array([Z[u] @ Z[v] for u, v in pairs])
    return roc_auc_score(labels, scores)
    ### END SOLUTION


auc_leaky = dot_auc(Zn, test_pairs, test_labels)

pairs_clean = skipgram_pairs(random_walks(G_train, NUM_WALKS, LENGTH, seed=1), WINDOW)
Z_clean, _ = train_sgns(pairs_clean, n=n_nodes, d=64, lr=0.025,
                        epochs=2 if SMOKE else 8, batch=4096, seed=0)
Zcl = Z_clean.numpy()
Zcl = Zcl / (np.linalg.norm(Zcl, axis=1, keepdims=True) + 1e-9)
auc_clean = dot_auc(Zcl, test_pairs, test_labels)

print(f"leaky AUC (walks saw the test edges): {auc_leaky:.3f}")
print(f"clean AUC (they never did):          {auc_clean:.3f}")
print(f"the leak is worth {auc_leaky - auc_clean:+.3f} AUC — for free, and for nothing")
assert auc_leaky > auc_clean + 0.02, (
    "the leaky run should be clearly inflated — if not, check that dot_auc scores "
    "with Z[u] @ Z[v] and that G_train really had the test edges removed"
)
assert auc_clean > 0.55, (
    f"clean AUC {auc_clean:.3f} barely beats coin-flipping — embeddings should still "
    f"rank held-out edges well above random non-edges"
)
print("exercise 5 ✓ — never let your walks see the test set")""",
         stub="""split_rng = random.Random(7)
all_edges = sorted(Gc.edges)
test_edges = split_rng.sample(all_edges, k=len(all_edges) // 10)
G_train = Gc.copy()
G_train.remove_edges_from(test_edges)

neg_pairs = []
n_nodes = Gc.number_of_nodes()
while len(neg_pairs) < len(test_edges):
    u, v = split_rng.randrange(n_nodes), split_rng.randrange(n_nodes)
    if u != v and not Gc.has_edge(u, v):
        neg_pairs.append((u, v))

test_pairs = test_edges + neg_pairs
test_labels = np.array([1] * len(test_edges) + [0] * len(neg_pairs))


def dot_auc(Z: np.ndarray, pairs: list, labels: np.ndarray) -> float:
    \"\"\"AUC of the dot-product scores Z[u]·Z[v] against the pair labels.\"\"\"
    # TODO: score every pair with a dot product, then roc_auc_score. 2 lines.
    raise NotImplementedError("score pairs and compute AUC")


auc_leaky = dot_auc(Zn, test_pairs, test_labels)

pairs_clean = skipgram_pairs(random_walks(G_train, NUM_WALKS, LENGTH, seed=1), WINDOW)
Z_clean, _ = train_sgns(pairs_clean, n=n_nodes, d=64, lr=0.025,
                        epochs=2 if SMOKE else 8, batch=4096, seed=0)
Zcl = Z_clean.numpy()
Zcl = Zcl / (np.linalg.norm(Zcl, axis=1, keepdims=True) + 1e-9)
auc_clean = dot_auc(Zcl, test_pairs, test_labels)

print(f"leaky AUC (walks saw the test edges): {auc_leaky:.3f}")
print(f"clean AUC (they never did):          {auc_clean:.3f}")
print(f"the leak is worth {auc_leaky - auc_clean:+.3f} AUC — for free, and for nothing")
assert auc_leaky > auc_clean + 0.02, (
    "the leaky run should be clearly inflated — if not, check that dot_auc scores "
    "with Z[u] @ Z[v] and that G_train really had the test edges removed"
)
assert auc_clean > 0.55, (
    f"clean AUC {auc_clean:.3f} barely beats coin-flipping — embeddings should still "
    f"rank held-out edges well above random non-edges"
)
print("exercise 5 ✓ — never let your walks see the test set")"""),

    md("""### The t-SNE ritual *(provided — enjoy it, then read Pitfall 4 again)*"""),

    code("""if SMOKE:
    print("SMOKE: skipping t-SNE (slow); the PCA in section 3 covers the smoke path")
else:
    from sklearn.manifold import TSNE
    xy_c = TSNE(n_components=2, random_state=0, perplexity=30).fit_transform(Zn)
    plt.figure(figsize=(7, 5))
    plt.scatter(xy_c[:, 0], xy_c[:, 1], c=y, cmap="Dark2", s=8, alpha=0.8)
    plt.title("Cora walk embeddings, t-SNE — colors = paper topics (never seen in training)")
    plt.axis("off"); plt.show()
    print("pretty — and admissible as EVIDENCE of nothing. Your accuracy and AUC")
    print("numbers above are the conclusions; this plot is the debugging aid.")"""),

    md("""## 6 · Stretch (optional, ungraded)

1. **node2vec bias.** Add p and q to your walker (you need one step of memory). Re-run
   exercise 4 with (p=4, q=0.25) and (p=0.25, q=4) — which regime helps Cora topic
   classification, and does that match the lecture's homophily story?
2. **Hadamard edge features.** For link prediction, replace the raw dot product with a
   logistic regression on the elementwise product z_u ⊙ z_v (node2vec's Table 1
   recipe). How many AUC points does the learned scorer add over the raw dot?
3. **Pitfall 2, verified.** Train two runs with different seeds and compare: (a) raw
   coordinate correlation, (b) the two runs' pairwise-distance matrices. Which one is
   stable? (That is why coordinates are not results.)

## 7 · Reflection (answer in this cell, 2–4 sentences each)

**R1.** word2vec keeps two tables, and your `train_sgns` returns only the center one.
What would go wrong (or just get strange) if positives and negatives were scored
against the *center* table itself — why is a separate context table the cleaner design?

**R2.** Explain your exercise-5 gap mechanically: through which exact quantity did the
held-out edges reach the leaky model's scores? ("The walks saw them" — yes, but name
the object: which counts changed, and what did SGNS do with them?)

**R3.** Your embeddings hit ~68% on Cora using zero features; Week 6's GCN reaches
~81% using features. Design a two-cell experiment that would tell you how much of that
gap is *features* versus *architecture* — what would you train, and what result would
support which conclusion?

*(your answers here)*

## What to submit

One executed notebook on Moodle: all five exercise checks ✓, both scatter plots
rendered, and the three reflection answers. Grading: assertions 70% · reflections 30%.
Run *Runtime → Restart and run all* before submitting.

**AI policy reminder** (course honor code): AI assistants are allowed for this lab *with
disclosure* — add a line here naming any tools you used and for what. You must be able
to explain any line of your submission on request; undeclared use or inability to
explain is a violation.
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
