#!/usr/bin/env python
"""Generate Lab 2 (classical graph ML: centralities, PageRank, communities, WL,
and the feature-based baseline) — student and solution notebooks.

    python scripts/labgen/make_lab02.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "labs" / "lab02_classical.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "labs" / "lab02_classical.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(text, stub): return ("code", text, stub)


CELLS = [
    md("""# Lab 2 · Structure as features — centralities, PageRank, communities, WL

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/labs/lab02_classical.ipynb)

**Week 2 · [lecture](https://lukmanovr.github.io/dkr/lectures/02-classical-graph-ml.html) · ≈ 25 min of compute (free Colab or CPU — no GPU needed)**

Everything the lecture computed by hand, you now compute at scale — and check against
the hand results first, real libraries second. The methods here are the permanent
baselines of graph ML: PageRank is the fixed point that founded a search engine
[[Brin & Page, 1998](https://doi.org/10.1016/S0169-7552(98)00110-X)], and the 1968
Weisfeiler–Leman relabeling became the decade-defining graph kernel
[[Shervashidze et al., 2011](https://jmlr.org/papers/v12/shervashidze11a.html)] —
every learned model in this course is measured against them. The lab ends with the
measurement the lecture promised: a structure-only classifier on Cora, the baseline
number the rest of the course has to beat.

### Goals
1. Implement PageRank power iteration from scratch and match `networkx` to 6 decimals.
2. Count triangles with $\\mathbf{A}^3$ and reproduce clustering coefficients exactly.
3. Implement modularity from its definition, verify the lecture's pencil values, then run Louvain and spectral clustering on the karate club and score both against 1977 reality.
4. Implement one round of Weisfeiler–Leman refinement and watch it fingerprint graphs — and fail on the pair it must fail on.
5. Build the structure-features + logistic-regression baselines for nodes (Cora) and graphs (PROTEINS).
"""),

    md("""## 0 · Setup  *(the contract from Lab 1 — pinned, SMOKE-aware)*"""),

    code("""import os, sys, random
from collections import Counter

SMOKE = os.environ.get("SMOKE", "") == "1"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    %pip install -q torch_geometric==2.8.0.post1

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"torch {torch.__version__} · networkx {nx.__version__} — environment OK")"""),

    md("""## 1 · Five importances, one club  *(read and investigate before you write)*

The karate club again — but now we rank it. There is nothing to implement in this
section; read the code, then run it. Context for your predictions: Mr. Hi is node 0,
the Officer is node 33, and they have degrees 16 and 17.

Write down your answers to these two questions before running the next cell:

1. Who wins **betweenness** — the instructor or the president?
2. Does the **same** member win eigenvector centrality?
"""),

    code("""G = nx.karate_club_graph()

measures = {
    "degree":      dict(G.degree),
    "closeness":   nx.closeness_centrality(G),
    "betweenness": nx.betweenness_centrality(G),
    "eigenvector": nx.eigenvector_centrality(G, max_iter=1000),
    "pagerank":    nx.pagerank(G, alpha=0.85, weight=None),   # the club has edge weights; the lecture world is unweighted
}
for name, scores in measures.items():
    top3 = sorted(scores, key=scores.get, reverse=True)[:3]
    print(f"{name:>12}: top-3 = {top3}   (champion's score {scores[top3[0]]:.4f})")

print()
print("So: the Officer (33) out-degrees Mr. Hi (0) — but Mr. Hi carries more")
print("shortest-path traffic (betweenness), while eigenvector sides with the Officer,")
print("whose neighbors are themselves heavyweights. One club, different champions —")
print("the lecture's kite story, on real data.")"""),

    md("""## 2 · PageRank from scratch  *(exercise 1 — skill: power iteration)*

**What to do:** in the next cell, implement `pagerank(G, beta, iters)` so that it
returns a dict `{node: rank}` whose values sum to 1, computed by the damped update from
the lecture, $\\mathbf{r} \\leftarrow \\beta M\\mathbf{r} + (1-\\beta)\\tfrac{1}{n}\\mathbf{1}$,
directly on an (undirected) `networkx` graph. At each step every node keeps
$(1-\\beta)/n$ base mass and receives $\\beta \\cdot r_u / d_u$ from each neighbor $u$.

The algorithm below (the lecture's Algorithm 1, restated) specifies exactly what your
implementation in the next cell must do; follow it step by step.

> **Algorithm · PageRank by power iteration** ([Brin & Page, 1998](https://doi.org/10.1016/S0169-7552(98)00110-X))
>
> **Input:** neighbor lists $N(\\cdot)$ (each undirected edge counts both ways, so $|N(u)| = d_u$); damping $\\beta$; iteration count $K$. **Output:** rank vector $\\mathbf{r}$ with $\\sum_u r_u = 1$.
>
> 1. $n \\leftarrow |V|$; &nbsp; $r_u \\leftarrow 1/n$ for every node $u$
> 2. **for** $k = 1 \\ldots K$ **do**
> 3. &nbsp;&nbsp;&nbsp;&nbsp;$r'_v \\leftarrow (1-\\beta)/n$ for every node $v$ &nbsp;&nbsp;*(teleport mass)*
> 4. &nbsp;&nbsp;&nbsp;&nbsp;**for** each node $u$ **do** $s \\leftarrow \\beta \\, r_u / |N(u)|$; add $s$ to $r'_v$ for every $v \\in N(u)$
> 5. &nbsp;&nbsp;&nbsp;&nbsp;$r \\leftarrow r'$
> 6. **end for**
> 7. **return** $r$

The asserts below the function check your ranks against `networkx` (which runs the same
iteration) and the tolerance is deliberately strict: your values must match to
**six decimals**. When the cell prints "exercise 1 ✓" your implementation is correct.
If you are close but not equal, read the assertion messages — each one names the usual
bug behind that symptom.
"""),

    todo("""def pagerank(G: nx.Graph, beta: float = 0.85, iters: int = 200) -> dict:
    \"\"\"PageRank by power iteration. Returns {node: rank}, summing to 1.
    Treat each undirected edge as two directed links (u→v and v→u).\"\"\"
    ### BEGIN SOLUTION
    n = G.number_of_nodes()
    r = {v: 1.0 / n for v in G}
    for _ in range(iters):
        nxt = {v: (1.0 - beta) / n for v in G}
        for u in G:
            share = beta * r[u] / G.degree[u]
            for v in G[u]:
                nxt[v] += share
        r = nxt
    return r
    ### END SOLUTION


pr = pagerank(G)
assert isinstance(pr, dict) and len(pr) == 34, "return one rank per node, as a dict"
assert abs(sum(pr.values()) - 1.0) < 1e-9, (
    f"ranks sum to {sum(pr.values()):.6f}, not 1 — mass is leaking. Check that every "
    f"node's outgoing share is beta*r[u]/degree(u) and the base term is (1-beta)/n."
)
reference = nx.pagerank(G, alpha=0.85, tol=1e-12, max_iter=500, weight=None)
worst = max(abs(pr[v] - reference[v]) for v in G)
assert worst < 1e-6, (
    f"max deviation from networkx is {worst:.2e} (> 1e-6). If it is ~1e-3, you are "
    f"iterating too few times; if the SHAPE of the ranking is wrong, check that you "
    f"divide by the SENDER's degree, not the receiver's."
)
print(f"exercise 1 ✓ — max deviation from networkx: {worst:.2e}")
print(f"top-3: {sorted(pr, key=pr.get, reverse=True)[:3]} — the two leaders, then the bridge node 32")""",
         stub="""def pagerank(G: nx.Graph, beta: float = 0.85, iters: int = 200) -> dict:
    \"\"\"PageRank by power iteration. Returns {node: rank}, summing to 1.
    Treat each undirected edge as two directed links (u→v and v→u).\"\"\"
    # TODO: start uniform; each iteration, give every node (1-beta)/n, then move
    # beta * r[u] / degree(u) from every u to each of its neighbors. ~8 lines.
    raise NotImplementedError("implement the damped power iteration")


pr = pagerank(G)
assert isinstance(pr, dict) and len(pr) == 34, "return one rank per node, as a dict"
assert abs(sum(pr.values()) - 1.0) < 1e-9, (
    f"ranks sum to {sum(pr.values()):.6f}, not 1 — mass is leaking. Check that every "
    f"node's outgoing share is beta*r[u]/degree(u) and the base term is (1-beta)/n."
)
reference = nx.pagerank(G, alpha=0.85, tol=1e-12, max_iter=500, weight=None)
worst = max(abs(pr[v] - reference[v]) for v in G)
assert worst < 1e-6, (
    f"max deviation from networkx is {worst:.2e} (> 1e-6). If it is ~1e-3, you are "
    f"iterating too few times; if the SHAPE of the ranking is wrong, check that you "
    f"divide by the SENDER's degree, not the receiver's."
)
print(f"exercise 1 ✓ — max deviation from networkx: {worst:.2e}")
print(f"top-3: {sorted(pr, key=pr.get, reverse=True)[:3]} — the two leaders, then the bridge node 32")"""),

    md("""## 3 · Triangles by matrix power  *(exercise 2 — skill: A³ and clustering)*

The lecture proved $(\\mathbf{A}^3)[u,u] = 2\\,\\times$ the number of triangles at $u$.
Now you use that fact.

**What to do:** in the next cell, implement `clustering_from_A(G)` so that it returns a
dict mapping every node to its local clustering coefficient, computed from a dense
adjacency matrix via the diagonal of $\\mathbf{A}^3$ (nodes with degree < 2 get 0 by
convention). You must place the factor of 2 correctly yourself. The assert compares
every value against `nx.clustering` exactly; when the cell prints "exercise 2 ✓" your
implementation is correct.
"""),

    todo("""def clustering_from_A(G: nx.Graph) -> dict:
    \"\"\"Local clustering coefficient per node via the diagonal of A³.
    Nodes with degree < 2 get 0 by convention.\"\"\"
    ### BEGIN SOLUTION
    nodes = list(G)
    A = nx.to_numpy_array(G, nodelist=nodes, weight=None)   # 0/1 matrix — the club has edge weights
    tri = np.diag(A @ A @ A) / 2.0
    deg = A.sum(axis=1)
    out = {}
    for i, v in enumerate(nodes):
        possible = deg[i] * (deg[i] - 1) / 2.0
        out[v] = tri[i] / possible if possible > 0 else 0.0
    return out
    ### END SOLUTION


cc = clustering_from_A(G)
ref = nx.clustering(G)
assert all(abs(cc[v] - ref[v]) < 1e-9 for v in G), (
    "mismatch with nx.clustering — the two usual bugs: (1) forgetting that the "
    "diagonal of A³ counts each triangle TWICE (both orientations), (2) dividing by "
    "d² instead of d(d-1)/2 possible pairs, (3) forgetting weight=None — the "
    "club's edges carry weights and to_numpy_array uses them unless told not to."
)
print("exercise 2 ✓ — A³ reproduces nx.clustering exactly")
print(f"most clustered members: {sorted(cc, key=cc.get, reverse=True)[:3]} "
      f"(tight friend-triangles at the club's edge, not its hubs)")""",
         stub="""def clustering_from_A(G: nx.Graph) -> dict:
    \"\"\"Local clustering coefficient per node via the diagonal of A³.
    Nodes with degree < 2 get 0 by convention.\"\"\"
    # TODO: A = nx.to_numpy_array(G, weight=None); triangles at i sit on the diagonal of A@A@A
    # (divided by ...?); divide by the number of neighbor PAIRS. ~7 lines.
    raise NotImplementedError("clustering coefficients via A³")


cc = clustering_from_A(G)
ref = nx.clustering(G)
assert all(abs(cc[v] - ref[v]) < 1e-9 for v in G), (
    "mismatch with nx.clustering — the two usual bugs: (1) forgetting that the "
    "diagonal of A³ counts each triangle TWICE (both orientations), (2) dividing by "
    "d² instead of d(d-1)/2 possible pairs, (3) forgetting weight=None — the "
    "club's edges carry weights and to_numpy_array uses them unless told not to."
)
print("exercise 2 ✓ — A³ reproduces nx.clustering exactly")
print(f"most clustered members: {sorted(cc, key=cc.get, reverse=True)[:3]} "
      f"(tight friend-triangles at the club's edge, not its hubs)")"""),

    md("""## 4 · Modularity, from the definition  *(exercise 3 — skill: Q and its null model)*

**What to do:** in the next cell, implement `modularity(G, communities)` so that it
returns $Q = \\sum_c \\left[ e_c/m - (D_c/2m)^2 \\right]$ for a partition given as an
iterable of node-sets. The asserts check it three ways: against the two pencil values
from the lecture's cast graph (the exact fractions $6/49$ and $10/49$), and against
`networkx` on an arbitrary partition. When the cell prints "exercise 3 ✓" all three
checks have passed.

**Predict before you run:** the lecture's worked example moved hub $C$ across the cut
without changing the cut size, and $Q$ *rose*. Write down, in one sentence, why that
happened — before running the next cell and letting the assertions check your memory.
"""),

    todo("""CAST = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (2, 5), (4, 5)])  # A..F = 0..5


def modularity(G: nx.Graph, communities) -> float:
    \"\"\"Q for a partition given as an iterable of node-sets.\"\"\"
    ### BEGIN SOLUTION
    m = G.number_of_edges()
    q = 0.0
    for c in communities:
        c = set(c)
        e_c = sum(1 for u, v in G.edges if u in c and v in c)
        d_c = sum(G.degree[v] for v in c)
        q += e_c / m - (d_c / (2 * m)) ** 2
    return q
    ### END SOLUTION


q1 = modularity(CAST, [{0, 1, 2, 3}, {4, 5}])
q2 = modularity(CAST, [{0, 1, 3}, {2, 4, 5}])
assert abs(q1 - 6 / 49) < 1e-12, (
    f"Q(ABCD|EF) = {q1:.6f}, expected 6/49 ≈ 0.122449 — check: internal edges are "
    f"AB,AC,AD,BC and EF (5 of 7); degree sums 10 and 4."
)
assert abs(q2 - 10 / 49) < 1e-12, (
    f"Q(ABD|CEF) = {q2:.6f}, expected 10/49 ≈ 0.204082 — same 5 internal edges, but "
    f"degree sums 6 and 8: the null model term is what changed."
)
random_part = [{0, 4, 5}, {1, 2, 3}]
assert abs(modularity(CAST, random_part)
           - nx.community.modularity(CAST, random_part)) < 1e-12, (
    "your Q disagrees with networkx on an arbitrary partition — likely counting "
    "internal edges twice (iterate G.edges, not ordered pairs)"
)
print(f"exercise 3 ✓ — Q(ABCD|EF) = {q1:.6f}, Q(ABD|CEF) = {q2:.6f}: moving the hub out helped")""",
         stub="""CAST = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (2, 5), (4, 5)])  # A..F = 0..5


def modularity(G: nx.Graph, communities) -> float:
    \"\"\"Q for a partition given as an iterable of node-sets.\"\"\"
    # TODO: for each community, count internal edges e_c and sum of degrees D_c;
    # add e_c/m − (D_c/2m)². ~7 lines.
    raise NotImplementedError("modularity from the definition")


q1 = modularity(CAST, [{0, 1, 2, 3}, {4, 5}])
q2 = modularity(CAST, [{0, 1, 3}, {2, 4, 5}])
assert abs(q1 - 6 / 49) < 1e-12, (
    f"Q(ABCD|EF) = {q1:.6f}, expected 6/49 ≈ 0.122449 — check: internal edges are "
    f"AB,AC,AD,BC and EF (5 of 7); degree sums 10 and 4."
)
assert abs(q2 - 10 / 49) < 1e-12, (
    f"Q(ABD|CEF) = {q2:.6f}, expected 10/49 ≈ 0.204082 — same 5 internal edges, but "
    f"degree sums 6 and 8: the null model term is what changed."
)
random_part = [{0, 4, 5}, {1, 2, 3}]
assert abs(modularity(CAST, random_part)
           - nx.community.modularity(CAST, random_part)) < 1e-12, (
    "your Q disagrees with networkx on an arbitrary partition — likely counting "
    "internal edges twice (iterate G.edges, not ordered pairs)"
)
print(f"exercise 3 ✓ — Q(ABCD|EF) = {q1:.6f}, Q(ABD|CEF) = {q2:.6f}: moving the hub out helped")"""),

    md("""### Louvain on the club — algorithm meets 1977

Now the real algorithm. There is nothing to implement here — the next cell is provided;
read it and run it. You implemented Louvain's objective ($Q$); the library performs the
greedy moves and aggregation. The cell scores the communities Louvain finds against the
real 1977 split using the adjusted Rand index, where 0 means roughly random agreement
and 1 means identical partitions.
"""),

    code("""louvain = nx.community.louvain_communities(G, seed=SEED, weight=None)
q_louvain = modularity(G, louvain)
truth = [1 if G.nodes[v]["club"] == "Mr. Hi" else 0 for v in G]
found = [next(i for i, c in enumerate(louvain) if v in c) for v in G]
ari = adjusted_rand_score(truth, found)

print(f"Louvain found {len(louvain)} communities, sizes {sorted(map(len, louvain), reverse=True)}")
print(f"Q = {q_louvain:.3f} · adjusted Rand vs the real split = {ari:.2f}")
assert q_louvain > 0.35, "Louvain should comfortably exceed Q = 0.35 on the club"

pos = nx.spring_layout(G, seed=7)
palette = ["#d9603b", "#0f8377", "#7c5cd6", "#d9a62e", "#d1567e", "#199473"]
node_colors = [palette[found[v] % 6] for v in G]
plt.figure(figsize=(7, 4.5))
nx.draw_networkx_edges(G, pos, alpha=0.25)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=120)
plt.title(f"Louvain communities (Q = {q_louvain:.3f}) — finer than the 2-way split, and honestly so")
plt.axis("off"); plt.show()
print("Louvain sees sub-factions inside each camp — 1977's binary choice was the")
print("politics; the friendship structure itself has more texture. Both are real.")"""),

    md("""## 5 · One eigenvector against reality  *(exercise 4 — skill: spectral bisection)*

Now you reproduce the lecture's Fiedler figure yourself.

**What to do:** in the next cell, implement `fiedler_split(G)` so that it builds the
Laplacian of the club as a numpy array, takes the eigenvector of the
**second-smallest** eigenvalue, splits the members by the sign of their entry, and
returns the tuple `(agreement_count, predicted_sides_dict)` — where `agreement_count`
counts how many of the 34 members land on their true side, scored under the better of
the two sign alignments (the eigenvector's global sign is arbitrary). The lecture told
you the answer is 32; the assertion holds you to it, and "exercise 4 ✓" confirms it.
"""),

    todo("""def fiedler_split(G: nx.Graph):
    \"\"\"Return (agreement_count, predicted_sides_dict) for the sign-split of the
    Fiedler vector vs the real 'club' attribute. Count agreement under the better
    of the two sign alignments (the eigenvector's global sign is arbitrary).\"\"\"
    ### BEGIN SOLUTION
    nodes = list(G)
    A = nx.to_numpy_array(G, nodelist=nodes, weight=None)   # 0/1 matrix — the club has edge weights
    L = np.diag(A.sum(axis=1)) - A
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1]                      # columns sorted by eigenvalue
    pred = {v: int(fiedler[i] > 0) for i, v in enumerate(nodes)}
    true = {v: int(G.nodes[v]["club"] == "Mr. Hi") for v in nodes}
    agree = sum(pred[v] == true[v] for v in nodes)
    agree = max(agree, len(nodes) - agree)
    return agree, pred
    ### END SOLUTION


agree, pred = fiedler_split(G)
assert agree == 32, (
    f"agreement = {agree}/34, expected 32 — if you got 2, flip: the eigenvector's "
    f"sign is arbitrary, score the better alignment; if ~17, you probably took the "
    f"eigenvector of the SMALLEST eigenvalue (the constant one — check eigh's ordering)."
)
print(f"exercise 4 ✓ — one eigenvector of L: {agree}/34 correct, no training, no labels")
print("the two misses are boundary members 2 and 8 — exactly where a smooth signal crosses zero")""",
         stub="""def fiedler_split(G: nx.Graph):
    \"\"\"Return (agreement_count, predicted_sides_dict) for the sign-split of the
    Fiedler vector vs the real 'club' attribute. Count agreement under the better
    of the two sign alignments (the eigenvector's global sign is arbitrary).\"\"\"
    # TODO: L = D − A as numpy (to_numpy_array with weight=None!); np.linalg.eigh(L); take eigvecs[:, 1]; split by
    # sign; compare with the 'club' attribute both ways. ~10 lines.
    raise NotImplementedError("spectral bisection of the karate club")


agree, pred = fiedler_split(G)
assert agree == 32, (
    f"agreement = {agree}/34, expected 32 — if you got 2, flip: the eigenvector's "
    f"sign is arbitrary, score the better alignment; if ~17, you probably took the "
    f"eigenvector of the SMALLEST eigenvalue (the constant one — check eigh's ordering)."
)
print(f"exercise 4 ✓ — one eigenvector of L: {agree}/34 correct, no training, no labels")
print("the two misses are boundary members 2 and 8 — exactly where a smooth signal crosses zero")"""),

    md("""## 6 · The 1968 trick  *(exercise 5 — skill: WL color refinement)*

**What to do:** in the next cell, implement `wl_refine(G, colors, table, round_id)` so
that it performs one refinement round and returns the new `{node: color_id}` dict.
Every node's new color is determined by the pair *(its own color, the sorted multiset
of its neighbors' colors)*. The `table` dict is the "hash": it assigns a fresh integer
id to each signature it has never seen, and it is **shared across graphs**, so two
graphs processed with the same table get comparable colors.

The algorithm below (the lecture's Algorithm 2, restated from
[Weisfeiler & Leman, 1968](https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf))
specifies exactly what your implementation in the next cell must do; follow it step by
step.

> **Algorithm · WL color refinement**
>
> **Input:** graph $G$; rounds $h$; signature table $T$ **shared across graphs**. **Output:** histogram of the colors from all rounds — the WL fingerprint.
>
> 1. $c(v) \\leftarrow c_0$ for every node (one shared initial color; the code uses $-1$)
> 2. **for** round $i = 1 \\ldots h$ **do**
> 3. &nbsp;&nbsp;&nbsp;&nbsp;**for** each node $v$ **do** $sig \\leftarrow (i,\\ c(v),\\ \\text{sorted tuple of neighbor colors})$
> 4. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $sig \\notin T$ **then** $T[sig] \\leftarrow$ fresh id; &nbsp;then $c'(v) \\leftarrow T[sig]$
> 5. &nbsp;&nbsp;&nbsp;&nbsp;$c \\leftarrow c'$ &nbsp;&nbsp;*(colors only split, never merge)*
> 6. **end for**
> 7. **return** the counter of colors from rounds $0 \\ldots h$

Your `wl_refine` implements one round (lines 3–4); the provided `wl_histogram` wraps it
into the full loop. The asserts then reproduce the two verdicts from the lecture: the
path–star pair must be distinguished after one round, and the two-triangles–hexagon
pair must never be distinguished at all. When the cell prints "exercise 5 ✓" both
verdicts hold.
"""),

    todo("""def wl_refine(G: nx.Graph, colors: dict, table: dict, round_id: int) -> dict:
    \"\"\"One WL round: return the new {node: color_id} dict, allocating ids via table.\"\"\"
    ### BEGIN SOLUTION
    new = {}
    for v in G:
        sig = (round_id, colors[v], tuple(sorted(colors[u] for u in G[v])))
        if sig not in table:
            table[sig] = len(table)
        new[v] = table[sig]
    return new
    ### END SOLUTION


def wl_histogram(G: nx.Graph, rounds: int, table: dict) -> Counter:
    \"\"\"Counter over colors from ALL rounds (incl. round 0) — the WL feature vector.\"\"\"
    colors = {v: -1 for v in G}          # shared initial color (id -1 by convention)
    hist = Counter(colors.values())
    for r in range(rounds):
        colors = wl_refine(G, colors, table, r)
        hist.update(colors.values())
    return hist


table = {}
cast_hist = wl_histogram(CAST, rounds=2, table=table)
final_counts = sorted(Counter(
    wl_refine(CAST, wl_refine(CAST, {v: -1 for v in CAST}, table, 0), table, 1).values()
).values())
assert final_counts == [1, 1, 1, 1, 2], (
    f"cast graph after 2 rounds: class sizes {final_counts}, expected [1, 1, 1, 1, 2] "
    f"(A, B, C, D each alone; E and F forever alike — the lecture's Figure). Check that "
    f"the signature includes the node's OWN color, and that neighbor colors are sorted."
)

path = nx.path_graph(4)
star = nx.star_graph(3)
t = {}
assert wl_histogram(path, 1, t) != wl_histogram(star, 1, t), (
    "path vs star must differ after ONE round — their degree histograms differ"
)
tri2 = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
hexg = nx.cycle_graph(6)
t = {}
assert wl_histogram(tri2, 5, t) == wl_histogram(hexg, 5, t), (
    "two triangles vs hexagon must be IDENTICAL at every round — every node is "
    "'degree-2 with degree-2 neighbors' forever. If they differ, your table is "
    "probably not shared between the two graphs."
)
print("exercise 5 ✓ — WL distinguishes path/star in 1 round and never separates the rings")
print("hold the rings result until Week 9: GNNs inherit exactly this blindness")""",
         stub="""def wl_refine(G: nx.Graph, colors: dict, table: dict, round_id: int) -> dict:
    \"\"\"One WL round: return the new {node: color_id} dict, allocating ids via table.\"\"\"
    # TODO: for each node, build the signature (round_id, own color, sorted tuple of
    # neighbor colors); look it up in table (adding a fresh id if new). ~7 lines.
    raise NotImplementedError("one round of WL color refinement")


def wl_histogram(G: nx.Graph, rounds: int, table: dict) -> Counter:
    \"\"\"Counter over colors from ALL rounds (incl. round 0) — the WL feature vector.\"\"\"
    colors = {v: -1 for v in G}          # shared initial color (id -1 by convention)
    hist = Counter(colors.values())
    for r in range(rounds):
        colors = wl_refine(G, colors, table, r)
        hist.update(colors.values())
    return hist


table = {}
cast_hist = wl_histogram(CAST, rounds=2, table=table)
final_counts = sorted(Counter(
    wl_refine(CAST, wl_refine(CAST, {v: -1 for v in CAST}, table, 0), table, 1).values()
).values())
assert final_counts == [1, 1, 1, 1, 2], (
    f"cast graph after 2 rounds: class sizes {final_counts}, expected [1, 1, 1, 1, 2] "
    f"(A, B, C, D each alone; E and F forever alike — the lecture's Figure). Check that "
    f"the signature includes the node's OWN color, and that neighbor colors are sorted."
)

path = nx.path_graph(4)
star = nx.star_graph(3)
t = {}
assert wl_histogram(path, 1, t) != wl_histogram(star, 1, t), (
    "path vs star must differ after ONE round — their degree histograms differ"
)
tri2 = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))
hexg = nx.cycle_graph(6)
t = {}
assert wl_histogram(tri2, 5, t) == wl_histogram(hexg, 5, t), (
    "two triangles vs hexagon must be IDENTICAL at every round — every node is "
    "'degree-2 with degree-2 neighbors' forever. If they differ, your table is "
    "probably not shared between the two graphs."
)
print("exercise 5 ✓ — WL distinguishes path/star in 1 round and never separates the rings")
print("hold the rings result until Week 9: GNNs inherit exactly this blindness")"""),

    md("""## 7 · The measurement: what structure alone predicts  *(exercise 6)*

The lecture promised a measurement; now you produce it — in two stages, and the first
stage is designed to *disappoint you in an instructive way*.

**Stage 1 — what to do:** in the next cell, implement `structure_features(G)` so that
it returns an `(n, 7)` numpy matrix with one row per node (in node order 0..n−1) and
these seven columns: degree, clustering coefficient, triangle count, PageRank, core
number, average neighbor degree, and max neighbor degree. The provided code below the
function then trains logistic regression on the standard 140-node split; the asserts
check the matrix shape and that the resulting accuracy lands in the expected band, and
"exercise 6 ✓" is printed by the Stage 2 cell once both stages have run.

**Predict before you run:** papers have 7 topics, so chance is ≈ 14%. Write down your
prediction before running the next cell — does structure-statistics-only land near 20%,
40%, or 60%?
""" ),

    todo("""from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]
Gc = to_networkx(data, to_undirected=True)
print(f"Cora: {Gc.number_of_nodes():,} papers · {Gc.number_of_edges():,} citations")


def structure_features(G: nx.Graph) -> np.ndarray:
    \"\"\"(n, 7) matrix: degree, clustering, triangles, pagerank, core number,
    average neighbor degree, max neighbor degree — for nodes 0..n-1 in order.\"\"\"
    ### BEGIN SOLUTION
    deg = dict(G.degree)
    cc = nx.clustering(G)
    tri = nx.triangles(G)
    prk = nx.pagerank(G, alpha=0.85)
    core = nx.core_number(G)
    and_ = nx.average_neighbor_degree(G)
    mnd = {v: max((deg[u] for u in G[v]), default=0) for v in G}
    feats = [deg, cc, tri, prk, core, and_, mnd]
    return np.array([[f[v] for f in feats] for v in sorted(G)])
    ### END SOLUTION


X = structure_features(Gc)
assert X.shape == (Gc.number_of_nodes(), 7), (
    f"feature matrix is {X.shape}, expected (n, 7) — one row per node in node order"
)
assert not np.isnan(X).any(), "NaNs in features — isolated nodes need the default=0 guards"

y = data.y.numpy()
train, test = data.train_mask.numpy(), data.test_mask.numpy()
mu, sd = X[train].mean(0), X[train].std(0) + 1e-9      # standardize on train stats only
Xz = (X - mu) / sd
clf = LogisticRegression(max_iter=2000).fit(Xz[train], y[train])
acc_stats = clf.score(Xz[test], y[test])

print(f"structural statistics only: {acc_stats:.1%}   (chance for 7 classes ≈ 14.3%)")
assert 0.14 < acc_stats < 0.40, (
    f"got {acc_stats:.1%} — expected weak-but-above-chance (≈15–40%). Below chance: "
    f"check that feature rows are in node order 0..n-1 (sorted(G)), matching data.y."
)
print()
print("Barely above chance — and that LOW number is the finding: centralities and")
print("clustering describe how IMPORTANT or TIGHT a node is, and a paper's topic has")
print("almost nothing to do with its importance. Statistics are topic-blind.")""",
         stub="""from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_networkx

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]
Gc = to_networkx(data, to_undirected=True)
print(f"Cora: {Gc.number_of_nodes():,} papers · {Gc.number_of_edges():,} citations")


def structure_features(G: nx.Graph) -> np.ndarray:
    \"\"\"(n, 7) matrix: degree, clustering, triangles, pagerank, core number,
    average neighbor degree, max neighbor degree — for nodes 0..n-1 in order.\"\"\"
    # TODO: seven dicts (nx has a function for the first six; the seventh is a
    # comprehension), then assemble rows in sorted(G) node order. ~9 lines.
    raise NotImplementedError("assemble the structural feature matrix")


X = structure_features(Gc)
assert X.shape == (Gc.number_of_nodes(), 7), (
    f"feature matrix is {X.shape}, expected (n, 7) — one row per node in node order"
)
assert not np.isnan(X).any(), "NaNs in features — isolated nodes need the default=0 guards"

y = data.y.numpy()
train, test = data.train_mask.numpy(), data.test_mask.numpy()
mu, sd = X[train].mean(0), X[train].std(0) + 1e-9      # standardize on train stats only
Xz = (X - mu) / sd
clf = LogisticRegression(max_iter=2000).fit(Xz[train], y[train])
acc_stats = clf.score(Xz[test], y[test])

print(f"structural statistics only: {acc_stats:.1%}   (chance for 7 classes ≈ 14.3%)")
assert 0.14 < acc_stats < 0.40, (
    f"got {acc_stats:.1%} — expected weak-but-above-chance (≈15–40%). Below chance: "
    f"check that feature rows are in node order 0..n-1 (sorted(G)), matching data.y."
)
print()
print("Barely above chance — and that LOW number is the finding: centralities and")
print("clustering describe how IMPORTANT or TIGHT a node is, and a paper's topic has")
print("almost nothing to do with its importance. Statistics are topic-blind.")"""),

    md("""### Stage 2 — add *where* you are, not just *what you look like*

Topic does live in the graph — via homophily: papers cite their own field, so a paper's
**position** (which citation cluster it sits in) is informative even though its
statistics are not. There is nothing to implement here: the next cell is provided —
read it and run it. It adds Louvain communities as one-hot features, re-trains the same
classifier, and prints "exercise 6 ✓" with the semester's structure-only baseline.
"""),

    code("""comms = nx.community.louvain_communities(Gc, seed=SEED)
cid = {v: i for i, c in enumerate(comms) for v in c}
C = np.zeros((len(y), len(comms)))
for v in sorted(Gc):
    C[v, cid[v]] = 1.0
X_pos = np.hstack([Xz, C])

clf2 = LogisticRegression(max_iter=3000).fit(X_pos[train], y[train])
acc_pos = clf2.score(X_pos[test], y[test])
print(f"statistics + {len(comms)} community one-hots: {acc_pos:.1%}   (statistics alone: {acc_stats:.1%})")
assert acc_pos > acc_stats + 0.15, (
    "community features should add >15 points — check that the one-hot row order is "
    "sorted(Gc), matching the label vector"
)
print()
print(f"exercise 6 ✓ — the semester's baseline: {acc_pos:.1%} from structure alone")
print("The jump says WHERE you sit predicts topic; HOW CENTRAL you are does not.")
print("Week 3 replaces the crude one-hot 'where' with learned continuous coordinates,")
print("and Week 6's GCN — which also reads the papers' words — reaches ~81%.")"""),

    md("""## 8 · WL features classify real molecules  *(provided — read, run, and note the year)*

Nothing to implement here — read the next cell, then run it. It points your
`wl_histogram` from exercise 5 at PROTEINS: 1,113 real protein graphs, with the task
"is this an enzyme?". This is the
[2011 state of the art](https://jmlr.org/papers/v12/shervashidze11a.html) — built
from a 1968 algorithm — and it remains a strong baseline today. (Under CI SMOKE we subsample; the
dataset server is external, so this cell degrades gracefully if it is unreachable.)
"""),

    code("""try:
    from torch_geometric.datasets import TUDataset
    proteins = TUDataset(root="data/TUDataset", name="PROTEINS")
    graphs = [to_networkx(g, to_undirected=True) for g in proteins]
    labels = np.array([int(g.y) for g in proteins])
    if SMOKE:
        # the dataset is sorted by class — subsample randomly, not by prefix
        keep = np.random.RandomState(SEED).permutation(len(graphs))[:400]
        graphs, labels = [graphs[i] for i in keep], labels[keep]

    shared = {}
    hists = [wl_histogram(g, rounds=3, table=shared) for g in graphs]
    n_colors = len(shared) + 1
    F = np.zeros((len(graphs), n_colors))
    for i, h in enumerate(hists):
        for color, count in h.items():
            F[i, color + 1] = count          # color -1 (round 0) lands in column 0

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(graphs))
    cut = int(0.8 * len(graphs))
    tr, te = idx[:cut], idx[cut:]
    wl_clf = LogisticRegression(max_iter=5000).fit(F[tr], labels[tr])
    wl_acc = wl_clf.score(F[te], labels[te])
    wl_majority = np.mean(labels[te] == np.bincount(labels[tr]).argmax())
    print(f"WL features ({F.shape[1]} colors) + logistic regression: {wl_acc:.1%}"
          f"   (majority: {wl_majority:.1%})")
    assert wl_acc > wl_majority, "WL features should beat the majority class"
    print("a 1968 pencil algorithm, competitive on 2020s benchmarks — Week 9 explains why")
except Exception as e:                        # noqa: BLE001 — external dataset host
    if SMOKE:
        print(f"PROTEINS unavailable in CI ({type(e).__name__}) — skipped gracefully")
    else:
        raise"""),

    md("""## 9 · Stretch (optional, ungraded)

1. **Personalized PageRank as a recommender.** Modify your `pagerank` so the teleport
   mass all lands on one member $q$ instead of spreading uniformly. For $q = 0$
   (Mr. Hi), which non-neighbors get the highest personalized rank — and would you
   "recommend them as friends"?
2. **Break modularity.** Build a ring of 24 triangle-cliques connected in a cycle
   (`nx.ring_of_cliques(24, 3)`) and check whether maximizing modularity keeps the
   triangles separate or merges neighbors — the resolution limit, live.
3. **Feature importance.** For the Cora classifier, inspect `clf.coef_` — which
   structural features carry the signal, and is Pitfall 5 (everything is degree in
   disguise) visible in the correlations?

## 10 · Reflection (answer in this cell, 2–4 sentences each)

**R1.** Your PageRank matched networkx to 1e-6 — but the *rankings* already agreed
after far fewer iterations. When does the difference between "scores converged" and
"ranking converged" matter in a real system?

**R2.** Louvain found more than two communities, yet its Q beat the true 2-way split's.
What does that say about "the" community structure of a network — and about trusting
any single partition?

**R3.** The Cora baseline used zero information from the papers' text. Predict: will
adding the bag-of-words features to the SAME logistic regression close most of the gap
to the GCN's ~81%, or not — and what would each outcome tell you?

*(your answers here)*

## What to submit

One executed notebook on Moodle: all six exercise checks ✓, the Louvain figure
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
