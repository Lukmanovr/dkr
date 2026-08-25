#!/usr/bin/env python
"""Generate Lab 9 (expressiveness: WL from scratch, the ceiling verified on an
untrained GIN, augmentations on trial, CSL graphs, oversquashing measured) —
student + solution notebooks. Entirely deterministic: numpy + networkx, no
training, no seeds to blame.

    python scripts/labgen/make_lab09.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "labs" / "lab09_expressiveness.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "labs" / "lab09_expressiveness.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(solution, stub): return ("code", solution, stub)


CELLS = [
    md("""# Lab 9 · The ceiling — WL, GIN, and the two walls, all verified

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/labs/lab09_expressiveness.ipynb)

**Week 9 · [lecture](https://lukmanovr.github.io/dkr/lectures/09-expressiveness.html) · ≈ 2 min of compute — every claim this week is a computation, not a training run**

You implement WL refinement from scratch and drive it over the lecture's blind
pairs; verify the ceiling theorem on an *untrained* GIN (no weights needed — that
is the theorem's point); put feature augmentations through the two-question trial;
build the CSL stress-test family; and measure oversquashing on the barbell to the
same three decimals as the lecture. The two papers this lab reproduces:
[Xu et al., 2019 — *How Powerful are Graph Neural Networks?*](https://arxiv.org/abs/1810.00826)
(the ceiling and GIN) and
[Alon & Yahav, 2021 — *On the Bottleneck of Graph Neural Networks*](https://arxiv.org/abs/2006.05205)
(oversquashing).

### Goals
1. Implement WL color refinement and reproduce every verdict from the lecture.
2. Show identical WL colors force identical GNN embeddings — with random weights.
3. Run the augmentation trial: separate different graphs, spare the same graph.
4. Verify the CSL pair is 4-regular, WL-blind, non-isomorphic — then crack it.
5. Measure the barbell's 290× sensitivity crush and its one-edge, 25× rescue.
"""),

    md("""## 0 · Setup"""),

    code("""import os, sys
from collections import Counter

SMOKE = os.environ.get("SMOKE", "") == "1"   # everything here is fast; SMOKE changes nothing
import numpy as np
import networkx as nx

print("numpy", np.__version__, "· networkx", nx.__version__, "— no torch this week, on purpose")

# the lecture's graph zoo, as edge lists
TRI2 = nx.disjoint_union(nx.cycle_graph(3), nx.cycle_graph(3))   # two triangles
HEX = nx.cycle_graph(6)                                          # one hexagon
DECALIN = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0),
                    (0, 6), (6, 7), (7, 8), (8, 9), (9, 1)])
BICYCLO = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
                    (5, 6), (6, 7), (7, 8), (8, 9), (9, 5), (0, 5)])
P4P3 = nx.disjoint_union(nx.path_graph(4), nx.path_graph(3))
P5P2 = nx.disjoint_union(nx.path_graph(5), nx.path_graph(2))"""),

    md("""## 1 · WL refinement, from scratch  *(exercise 1 — the measuring stick)*

The definition from the lecture, as ~10 lines of Python. The one subtlety the
asserts will catch if you miss it: the color table must be **shared across both
graphs** — a fresh color means "a signature never seen in either graph".

> **Algorithm · WL comparison with a shared color table**
>
> **Input:** graphs G1, G2; rounds T. &nbsp; **Output:** per-round histogram pair.
>
> 1. c(v) ← 0 for every node of *both* graphs
> 2. **for** round t = 1 … T **do**
> 3. &nbsp;&nbsp;&nbsp;&nbsp;for each node v: sig(v) ← (c(v), sorted multiset of neighbor colors)
> 4. &nbsp;&nbsp;&nbsp;&nbsp;build ONE table over both graphs' signatures: new sig → fresh color
> 5. &nbsp;&nbsp;&nbsp;&nbsp;c(v) ← table[sig(v)] for every node of both graphs
> 6. &nbsp;&nbsp;&nbsp;&nbsp;h1, h2 ← color counts of G1, G2
> 7. **end for**
> 8. **return** (h1, h2) — equal dicts at every round ⇒ WL abstains
>
> Step 4 is the subtlety: one table spanning both graphs means *same signature,
> same color, everywhere* — six nodes of color 3 vs six of color 4 is a
> difference, not a match. A per-graph table silently erases it.
"""),

    todo("""def wl_histograms(G1, G2, rounds):
    \"\"\"Run `rounds` rounds of WL refinement on G1 and G2 with a SHARED color
    table. Return (hist1, hist2): each a dict color -> class size at the
    final round. Colors are shared, so equal dicts mean WL sees no difference.\"\"\"
    ### BEGIN SOLUTION
    c1 = {v: 0 for v in G1}
    c2 = {v: 0 for v in G2}
    for _ in range(rounds):
        sig1 = {v: (c1[v], tuple(sorted(c1[u] for u in G1[v]))) for v in G1}
        sig2 = {v: (c2[v], tuple(sorted(c2[u] for u in G2[v]))) for v in G2}
        table = {s: i for i, s in enumerate(sorted(set(sig1.values()) | set(sig2.values())))}
        c1 = {v: table[sig1[v]] for v in G1}
        c2 = {v: table[sig2[v]] for v in G2}
    return (dict(Counter(c1.values())), dict(Counter(c2.values())))
    ### END SOLUTION


def wl_blind(G1, G2, rounds=12):
    \"\"\"True if the histograms match at EVERY round up to `rounds`.\"\"\"
    return all(wl_histograms(G1, G2, r) [0] == wl_histograms(G1, G2, r)[1]
               for r in range(1, rounds + 1))


# — the founding blind pair
assert wl_blind(TRI2, HEX), (
    "two triangles vs one hexagon: every node always sees {my color, my color} — "
    "the histograms must match at every round. If they split, your color table "
    "is probably not shared across the two graphs."
)
# — the molecules
h1, h2 = wl_histograms(DECALIN, BICYCLO, 10)
assert h1 == h2 and sorted(h1.values()) == [2, 4, 4], (
    f"decalin vs bicyclopentyl must stabilize at the SAME three classes with "
    f"sizes 2, 4, 4 — got {h1} vs {h2}"
)
assert not nx.is_isomorphic(DECALIN, BICYCLO), "and yet they are different molecules"
# — the pair WL catches: same degree sequence, split at round 2
r1 = wl_histograms(P4P3, P5P2, 1)
r2 = wl_histograms(P4P3, P5P2, 2)
assert r1[0] == r1[1] and sorted(r1[0].values()) == [3, 4], (
    f"round 1 = degree partition, same two classes of sizes 3 and 4 both sides — got {r1}"
)
assert r2[0] != r2[1], (
    "round 2 must split them: P3's middle sees two LEAVES — no interior node of "
    "P5 ever does. If round 2 matches, check that signatures include the node's "
    "own current color, not just the neighbor multiset."
)
print("exercise 1 ✓ — blind where the lecture said blind, sharp exactly at round 2")""",
         stub="""def wl_histograms(G1, G2, rounds):
    \"\"\"Run `rounds` rounds of WL refinement on G1 and G2 with a SHARED color
    table. Return (hist1, hist2): each a dict color -> class size at the
    final round. Colors are shared, so equal dicts mean WL sees no difference.\"\"\"
    # TODO (~10 lines): per round, signature = (own color, sorted neighbor
    # colors); build ONE table over both graphs' signatures; recolor; return
    # Counter dicts. Compare COLORS, not just class sizes — six nodes of color
    # 3 and six of color 4 are a difference, not a match.
    raise NotImplementedError


def wl_blind(G1, G2, rounds=12):
    \"\"\"True if the histograms match at EVERY round up to `rounds`.\"\"\"
    return all(wl_histograms(G1, G2, r) [0] == wl_histograms(G1, G2, r)[1]
               for r in range(1, rounds + 1))


# — the founding blind pair
assert wl_blind(TRI2, HEX), (
    "two triangles vs one hexagon: every node always sees {my color, my color} — "
    "the histograms must match at every round. If they split, your color table "
    "is probably not shared across the two graphs."
)
# — the molecules
h1, h2 = wl_histograms(DECALIN, BICYCLO, 10)
assert h1 == h2 and sorted(h1.values()) == [2, 4, 4], (
    f"decalin vs bicyclopentyl must stabilize at the SAME three classes with "
    f"sizes 2, 4, 4 — got {h1} vs {h2}"
)
assert not nx.is_isomorphic(DECALIN, BICYCLO), "and yet they are different molecules"
# — the pair WL catches: same degree sequence, split at round 2
r1 = wl_histograms(P4P3, P5P2, 1)
r2 = wl_histograms(P4P3, P5P2, 2)
assert r1[0] == r1[1] and sorted(r1[0].values()) == [3, 4], (
    f"round 1 = degree partition, same two classes of sizes 3 and 4 both sides — got {r1}"
)
assert r2[0] != r2[1], (
    "round 2 must split them: P3's middle sees two LEAVES — no interior node of "
    "P5 ever does. If round 2 matches, check that signatures include the node's "
    "own current color, not just the neighbor multiset."
)
print("exercise 1 ✓ — blind where the lecture said blind, sharp exactly at round 2")"""),

    md("""## 2 · The ceiling, verified on an UNTRAINED GIN  *(exercise 2)*

The theorem says: same WL histograms ⇒ same pooled embedding, *for every setting
of the weights*. So we do not train — we draw random weights once and check both
directions. Implement one GIN layer in numpy; the MLP weights are provided.
"""),

    todo("""rng = np.random.default_rng(9)
W1, W2 = rng.standard_normal((8, 16)), rng.standard_normal((16, 8))
EPS = 0.1


def gin_layer(X, A):
    \"\"\"One GIN layer: MLP((1+EPS) * X + A @ X), with MLP(z) = tanh(z W1) W2.
    X: (n, 8) features; A: (n, n) dense adjacency (no self-loops).\"\"\"
    ### BEGIN SOLUTION
    Z = (1 + EPS) * X + A @ X
    return np.tanh(Z @ W1) @ W2
    ### END SOLUTION


def gin_readout(G, layers=3):
    \"\"\"Uniform features -> `layers` GIN layers -> SUM-pool to one vector.\"\"\"
    ### BEGIN SOLUTION
    A = nx.to_numpy_array(G, weight=None)
    X = np.ones((len(A), 8))
    for _ in range(layers):
        X = gin_layer(X, A)
    return X.sum(axis=0)
    ### END SOLUTION


# direction 1: WL-blind pair -> identical readouts, with RANDOM weights
d_blind = np.abs(gin_readout(TRI2) - gin_readout(HEX)).max()
assert d_blind < 1e-9, (
    f"the blind pair must give IDENTICAL readouts (got max diff {d_blind:.2e}) — "
    f"the theorem holds for every weight setting, so if it fails, the bug is in "
    f"the layer: uniform start features + equal degrees must stay equal."
)
d_mol = np.abs(gin_readout(DECALIN) - gin_readout(BICYCLO)).max()
assert d_mol < 1e-9, f"the molecules too: identical readouts required, got {d_mol:.2e}"

# direction 2: a WL-splittable pair should split (random weights, almost surely)
d_split = np.abs(gin_readout(P4P3) - gin_readout(P5P2)).max()
assert d_split > 1e-6, (
    f"P4∪P3 vs P5∪P2 are WL-distinguishable, so a random GIN should give "
    f"different readouts — got max diff {d_split:.2e}. Check the (1+EPS) self term."
)
print(f"exercise 2 ✓ — blind pairs: 0 exactly; splittable pair differs by {d_split:.3f}")
print("   no training happened. The ceiling is a property of the ARCHITECTURE.")""",
         stub="""rng = np.random.default_rng(9)
W1, W2 = rng.standard_normal((8, 16)), rng.standard_normal((16, 8))
EPS = 0.1


def gin_layer(X, A):
    \"\"\"One GIN layer: MLP((1+EPS) * X + A @ X), with MLP(z) = tanh(z W1) W2.
    X: (n, 8) features; A: (n, n) dense adjacency (no self-loops).\"\"\"
    # TODO: 2 lines.
    raise NotImplementedError


def gin_readout(G, layers=3):
    \"\"\"Uniform features -> `layers` GIN layers -> SUM-pool to one vector.\"\"\"
    # TODO: adjacency with weight=None, ones features (n, 8), stack layers,
    # sum over nodes. ~5 lines.
    raise NotImplementedError


# direction 1: WL-blind pair -> identical readouts, with RANDOM weights
d_blind = np.abs(gin_readout(TRI2) - gin_readout(HEX)).max()
assert d_blind < 1e-9, (
    f"the blind pair must give IDENTICAL readouts (got max diff {d_blind:.2e}) — "
    f"the theorem holds for every weight setting, so if it fails, the bug is in "
    f"the layer: uniform start features + equal degrees must stay equal."
)
d_mol = np.abs(gin_readout(DECALIN) - gin_readout(BICYCLO)).max()
assert d_mol < 1e-9, f"the molecules too: identical readouts required, got {d_mol:.2e}"

# direction 2: a WL-splittable pair should split (random weights, almost surely)
d_split = np.abs(gin_readout(P4P3) - gin_readout(P5P2)).max()
assert d_split > 1e-6, (
    f"P4∪P3 vs P5∪P2 are WL-distinguishable, so a random GIN should give "
    f"different readouts — got max diff {d_split:.2e}. Check the (1+EPS) self term."
)
print(f"exercise 2 ✓ — blind pairs: 0 exactly; splittable pair differs by {d_split:.3f}")
print("   no training happened. The ceiling is a property of the ARCHITECTURE.")"""),

    md("""## 3 · Augmentations on trial  *(exercise 3 — the honesty test)*

An augmentation must pass two tests at once: **separate** genuinely different
graphs, and **spare** two relabeled copies of the same graph. Implement per-node
triangle counts; the trial harness (provided) then runs your feature and two
rivals through both tests.
"""),

    todo("""HEX_RELABELED = nx.relabel_nodes(HEX, {v: (5 * v + 2) % 6 for v in HEX})


def triangle_counts(G):
    \"\"\"Per-node triangle counts as a dict {node: count}.
    Hint: count adjacent neighbor pairs, or use diag(A³)/2.\"\"\"
    ### BEGIN SOLUTION
    A = nx.to_numpy_array(G, weight=None)
    t = np.diag(np.linalg.matrix_power(A, 3)) / 2
    return {v: int(t[i]) for i, v in enumerate(G)}
    ### END SOLUTION


def wl_blind_with_features(G1, G2, feat_fn, rounds=8):
    \"\"\"WL with initial colors = feat_fn(G); True if histograms match at every round.\"\"\"
    f1, f2 = feat_fn(G1), feat_fn(G2)
    table = {v: i for i, v in enumerate(sorted(set(f1.values()) | set(f2.values())))}
    c1 = {v: table[f1[v]] for v in G1}
    c2 = {v: table[f2[v]] for v in G2}
    if Counter(c1.values()) != Counter(c2.values()):
        return False                       # the features alone already separate
    for _ in range(rounds):
        s1 = {v: (c1[v], tuple(sorted(c1[u] for u in G1[v]))) for v in G1}
        s2 = {v: (c2[v], tuple(sorted(c2[u] for u in G2[v]))) for v in G2}
        tab = {s: i for i, s in enumerate(sorted(set(s1.values()) | set(s2.values())))}
        c1 = {v: tab[s1[v]] for v in G1}
        c2 = {v: tab[s2[v]] for v in G2}
        if Counter(c1.values()) != Counter(c2.values()):
            return False                   # colors, not just sizes: shared table
    return True


degree_feat = lambda G: dict(G.degree())
rid = np.random.default_rng(0)
random_feat = lambda G: {v: int(rid.integers(1 << 30)) for v in G}

# trial 1: separate the DIFFERENT pair (two triangles vs hexagon)?
assert wl_blind_with_features(TRI2, HEX, degree_feat), "degree: all 2s — still blind"
assert not wl_blind_with_features(TRI2, HEX, triangle_counts), (
    "triangle counts must separate them: every triangle node touches 1 triangle, "
    "every hexagon node 0 — round 0 already differs"
)
assert not wl_blind_with_features(TRI2, HEX, random_feat), "random IDs separate everything…"
# trial 2: spare the SAME graph (hexagon vs its relabeling)?
assert wl_blind_with_features(HEX, HEX_RELABELED, triangle_counts), (
    "triangle counts are structural: a relabeled copy gets the same counts — "
    "the histograms must match (0 triangles everywhere)"
)
assert not wl_blind_with_features(HEX, HEX_RELABELED, random_feat), (
    "…including two copies of the SAME graph: the false alarm, demonstrated"
)
print("exercise 3 ✓ — structural features: power without lies; random IDs: power with them")""",
         stub="""HEX_RELABELED = nx.relabel_nodes(HEX, {v: (5 * v + 2) % 6 for v in HEX})


def triangle_counts(G):
    \"\"\"Per-node triangle counts as a dict {node: count}.
    Hint: count adjacent neighbor pairs, or use diag(A³)/2.\"\"\"
    # TODO: ~3 lines.
    raise NotImplementedError


def wl_blind_with_features(G1, G2, feat_fn, rounds=8):
    \"\"\"WL with initial colors = feat_fn(G); True if histograms match at every round.\"\"\"
    f1, f2 = feat_fn(G1), feat_fn(G2)
    table = {v: i for i, v in enumerate(sorted(set(f1.values()) | set(f2.values())))}
    c1 = {v: table[f1[v]] for v in G1}
    c2 = {v: table[f2[v]] for v in G2}
    if Counter(c1.values()) != Counter(c2.values()):
        return False                       # the features alone already separate
    for _ in range(rounds):
        s1 = {v: (c1[v], tuple(sorted(c1[u] for u in G1[v]))) for v in G1}
        s2 = {v: (c2[v], tuple(sorted(c2[u] for u in G2[v]))) for v in G2}
        tab = {s: i for i, s in enumerate(sorted(set(s1.values()) | set(s2.values())))}
        c1 = {v: tab[s1[v]] for v in G1}
        c2 = {v: tab[s2[v]] for v in G2}
        if Counter(c1.values()) != Counter(c2.values()):
            return False                   # colors, not just sizes: shared table
    return True


degree_feat = lambda G: dict(G.degree())
rid = np.random.default_rng(0)
random_feat = lambda G: {v: int(rid.integers(1 << 30)) for v in G}

# trial 1: separate the DIFFERENT pair (two triangles vs hexagon)?
assert wl_blind_with_features(TRI2, HEX, degree_feat), "degree: all 2s — still blind"
assert not wl_blind_with_features(TRI2, HEX, triangle_counts), (
    "triangle counts must separate them: every triangle node touches 1 triangle, "
    "every hexagon node 0 — round 0 already differs"
)
assert not wl_blind_with_features(TRI2, HEX, random_feat), "random IDs separate everything…"
# trial 2: spare the SAME graph (hexagon vs its relabeling)?
assert wl_blind_with_features(HEX, HEX_RELABELED, triangle_counts), (
    "triangle counts are structural: a relabeled copy gets the same counts — "
    "the histograms must match (0 triangles everywhere)"
)
assert not wl_blind_with_features(HEX, HEX_RELABELED, random_feat), (
    "…including two copies of the SAME graph: the false alarm, demonstrated"
)
print("exercise 3 ✓ — structural features: power without lies; random IDs: power with them")"""),

    md("""## 4 · The CSL stress test  *(exercise 4 — cracking a 4-regular safe)*

$\\mathrm{CSL}(11, s)$: 11 nodes in a ring, edges to distance 1 and distance
$s$. Build the family, verify the lecture's three claims, then crack the pair
with the random-walk return probabilities you can also compute by hand
($6/64$ vs $0$ at three steps — count the triangles).

One term the asserts use: CSL graphs are **vertex-transitive** — some symmetry
of the graph maps any node to any other, so every node "looks the same" and WL
can never split the node set.

> **Algorithm · Random-walk return features (RWPE)**
>
> **Input:** graph G with adjacency A; steps kmax. &nbsp;
> **Output:** feature vector r(v) ∈ R^kmax per node.
>
> 1. M ← D⁻¹A &nbsp;&nbsp;*(the random-walk operator, Week 2)*
> 2. P ← I
> 3. **for** k = 1 … kmax **do**
> 4. &nbsp;&nbsp;&nbsp;&nbsp;P ← P·M
> 5. &nbsp;&nbsp;&nbsp;&nbsp;r_k(v) ← P[v, v] for every v &nbsp;&nbsp;*(prob. a k-step walk returns)*
> 6. **end for**
> 7. **return** r(v) = (r_1(v), …, r_kmax(v)) — isomorphic copies agree
"""),

    todo("""def csl(n, s):
    g = nx.Graph()
    for v in range(n):
        g.add_edge(v, (v + 1) % n)
        g.add_edge(v, (v + s) % n)
    return g


def rw_return_probs(G, kmax=4):
    \"\"\"(kmax, n) array: entry [k-1, i] = probability a k-step uniform random
    walk from node i ends back at node i.\"\"\"
    ### BEGIN SOLUTION
    A = nx.to_numpy_array(G, weight=None)
    M = A / A.sum(1, keepdims=True)
    out, P = [], np.eye(len(A))
    for _ in range(kmax):
        P = P @ M
        out.append(np.diag(P).copy())
    return np.array(out)
    ### END SOLUTION


C2, C3s = csl(11, 2), csl(11, 3)
assert all(d == 4 for _, d in C2.degree()) and all(d == 4 for _, d in C3s.degree()), "both 4-regular"
assert wl_blind(C2, C3s), "both vertex-transitive: WL never splits either — blind forever"
assert not nx.is_isomorphic(C2, C3s), "and yet non-isomorphic — the safe is real"

r2, r3 = rw_return_probs(C2), rw_return_probs(C3s)
assert abs(r2[2, 0] - 6 / 64) < 1e-12, (
    f"CSL(11,2): 3 triangles through each node × 2 orientations = 6 closed "
    f"3-walks out of 4³ — return prob 6/64, got {r2[2, 0]:.4f}"
)
assert r3[2, 0] == 0.0, "CSL(11,3): {±1, ±3} cannot sum to 0 in three steps — no triangles"
assert np.allclose(r2, r2[:, :1]) and np.allclose(r3, r3[:, :1]), (
    "vertex-transitive: every node has the SAME return signature within a graph"
)

# the crack: RW returns as initial features -> WL separates at round 0
rw_feat = lambda G: {v: tuple(np.round(rw_return_probs(G)[:, i], 6))
                     for i, v in enumerate(G)}
assert not wl_blind_with_features(C2, C3s, rw_feat), "RWPE separates the WL-blind pair"
assert wl_blind_with_features(C2, csl(11, 2), rw_feat), (
    "…and honestly: a fresh copy of CSL(11,2) matches the original"
)
print(f"exercise 4 ✓ — return probs {r2[2, 0]:.4f} vs {r3[2, 0]:.4f}: the safe is open, honestly")""",
         stub="""def csl(n, s):
    g = nx.Graph()
    for v in range(n):
        g.add_edge(v, (v + 1) % n)
        g.add_edge(v, (v + s) % n)
    return g


def rw_return_probs(G, kmax=4):
    \"\"\"(kmax, n) array: entry [k-1, i] = probability a k-step uniform random
    walk from node i ends back at node i.\"\"\"
    # TODO: row-normalize the adjacency, take powers, read the diagonal. ~7 lines.
    raise NotImplementedError


C2, C3s = csl(11, 2), csl(11, 3)
assert all(d == 4 for _, d in C2.degree()) and all(d == 4 for _, d in C3s.degree()), "both 4-regular"
assert wl_blind(C2, C3s), "both vertex-transitive: WL never splits either — blind forever"
assert not nx.is_isomorphic(C2, C3s), "and yet non-isomorphic — the safe is real"

r2, r3 = rw_return_probs(C2), rw_return_probs(C3s)
assert abs(r2[2, 0] - 6 / 64) < 1e-12, (
    f"CSL(11,2): 3 triangles through each node × 2 orientations = 6 closed "
    f"3-walks out of 4³ — return prob 6/64, got {r2[2, 0]:.4f}"
)
assert r3[2, 0] == 0.0, "CSL(11,3): {±1, ±3} cannot sum to 0 in three steps — no triangles"
assert np.allclose(r2, r2[:, :1]) and np.allclose(r3, r3[:, :1]), (
    "vertex-transitive: every node has the SAME return signature within a graph"
)

# the crack: RW returns as initial features -> WL separates at round 0
rw_feat = lambda G: {v: tuple(np.round(rw_return_probs(G)[:, i], 6))
                     for i, v in enumerate(G)}
assert not wl_blind_with_features(C2, C3s, rw_feat), "RWPE separates the WL-blind pair"
assert wl_blind_with_features(C2, csl(11, 2), rw_feat), (
    "…and honestly: a fresh copy of CSL(11,2) matches the original"
)
print(f"exercise 4 ✓ — return probs {r2[2, 0]:.4f} vs {r3[2, 0]:.4f}: the safe is open, honestly")"""),

    md("""## 5 · Oversquashing, to the decimal  *(exercise 5 — the second wall)*

The barbell: two 5-cliques joined by a 2-node path. Influence after $K$ layers is
the entry $(\\hat A^K)_{uv}$ — build it (HW2's `build_ahat`, new graph) and
reproduce the lecture's 290× crush and 25× one-edge rescue.
"""),

    todo("""BARBELL = nx.barbell_graph(5, 2)      # nodes 0-4 clique · 5,6 path · 7-11 clique


def sensitivity(G, K):
    \"\"\"(n, n) matrix of K-step influences: the K-th power of
    D̃^{-1/2}(A+I)D̃^{-1/2}. Unweighted adjacency!\"\"\"
    ### BEGIN SOLUTION
    A = nx.to_numpy_array(G, weight=None)
    At = A + np.eye(len(A))
    d = At.sum(1)
    Dm = np.diag(d ** -0.5)
    return np.linalg.matrix_power(Dm @ At @ Dm, K)
    ### END SOLUTION


S4, S5 = sensitivity(BARBELL, 4), sensitivity(BARBELL, 5)

# unreachable is NOT squashed: node 11 is 5 hops out — exactly zero at K=4
assert S4[0, 11] == 0.0, (
    "at K=4, node 11 (distance 5) is outside the receptive field: influence must "
    "be EXACTLY zero — a radius problem, not a bottleneck problem"
)
near, far = S5[0, 4], S5[0, 11]
ratio = near / far
assert 0.17 < near < 0.19 and 250 < ratio < 330, (
    f"K=5: same-clique influence ≈ 0.179, across-bridge ≈ 6e-4, ratio ≈ 290 — "
    f"got near {near:.4f}, ratio {ratio:.0f}. Check the self-loop and weight=None."
)

RESCUED = BARBELL.copy()
RESCUED.add_edge(1, 10)               # ONE edge across
rescue = sensitivity(RESCUED, 5)[0, 11] / far
assert rescue > 20, (
    f"one added edge must lift far-clique influence >20× (got {rescue:.0f}×) — "
    f"the bottleneck, not the distance, was the problem"
)
print(f"exercise 5 ✓ — near {near:.3f} · far {far:.1e} · ratio {ratio:.0f}× · rescue {rescue:.0f}×")
print("   the lecture's figure, reproduced from your own matrix powers")""",
         stub="""BARBELL = nx.barbell_graph(5, 2)      # nodes 0-4 clique · 5,6 path · 7-11 clique


def sensitivity(G, K):
    \"\"\"(n, n) matrix of K-step influences: the K-th power of
    D̃^{-1/2}(A+I)D̃^{-1/2}. Unweighted adjacency!\"\"\"
    # TODO: HW2's build_ahat + np.linalg.matrix_power. ~6 lines.
    raise NotImplementedError


S4, S5 = sensitivity(BARBELL, 4), sensitivity(BARBELL, 5)

# unreachable is NOT squashed: node 11 is 5 hops out — exactly zero at K=4
assert S4[0, 11] == 0.0, (
    "at K=4, node 11 (distance 5) is outside the receptive field: influence must "
    "be EXACTLY zero — a radius problem, not a bottleneck problem"
)
near, far = S5[0, 4], S5[0, 11]
ratio = near / far
assert 0.17 < near < 0.19 and 250 < ratio < 330, (
    f"K=5: same-clique influence ≈ 0.179, across-bridge ≈ 6e-4, ratio ≈ 290 — "
    f"got near {near:.4f}, ratio {ratio:.0f}. Check the self-loop and weight=None."
)

RESCUED = BARBELL.copy()
RESCUED.add_edge(1, 10)               # ONE edge across
rescue = sensitivity(RESCUED, 5)[0, 11] / far
assert rescue > 20, (
    f"one added edge must lift far-clique influence >20× (got {rescue:.0f}×) — "
    f"the bottleneck, not the distance, was the problem"
)
print(f"exercise 5 ✓ — near {near:.3f} · far {far:.1e} · ratio {ratio:.0f}× · rescue {rescue:.0f}×")
print("   the lecture's figure, reproduced from your own matrix powers")"""),

    md("""## 6 · Stretch (optional, ungraded)

1. **Break your own WL.** Find (or construct) a pair of *connected* 3-regular
   graphs on 10 nodes that your `wl_blind` declares blind. Verify non-isomorphism
   with `nx.is_isomorphic`. (Hint: the Petersen graph has a famous twin.)
2. **Subgraph-GNN preview.** For the blind pair, delete each node in turn and
   compare the *multisets* of `gin_readout` values over the resulting bags.
   Does node deletion crack the pair the way the lecture claimed?
3. **Laplacian PE, with its wrinkle.** Separate the CSL pair using the second
   Laplacian eigenvector as a feature — then flip its sign on one copy and watch
   the honesty test fail. Repair with `abs()` and re-run both trials.
4. **Curvature spotting.** Compute effective resistance (`nx.resistance_distance`)
   between the barbell's bridgeheads and within a clique — *effective
   resistance*: view every edge as a 1Ω resistor; the resistance between two
   nodes is low when many parallel paths connect them, high across a lone
   bridge. How does the ratio compare with your sensitivity ratio?

## 7 · Reflection (answer in this cell, 2–4 sentences each)

**R1.** Exercise 2 verified the ceiling *without training*. Explain to a
skeptical teammate why "we never trained it" makes the demonstration stronger,
not weaker.

**R2.** Random IDs passed the separation test and failed the honesty test. Name
a real deployment scenario where that failure would surface as a silent
production bug, and the test you would add to CI to catch it.

**R3.** Your barbell numbers say influence across the bridge is ~6×10⁻⁴ at
K=5. A colleague proposes "just use K=10." Predict qualitatively what happens
to (a) the far-clique influence and (b) the within-clique representations, and
what the right fix is instead.

*(your answers here)*

## What to submit

One executed notebook on Moodle: all five exercise checks ✓ and the three
reflections. *Runtime → Restart and run all* must pass top to bottom (it takes
about two minutes — determinism is cheap). Grading: assertions 70% ·
reflections 30%.

**AI policy reminder** (course honor code): AI assistants are allowed for this
lab *with disclosure* — add a line here naming any tools you used and for what.
You must be able to explain any line of your submission on request; undeclared
use or inability to explain is a violation.
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
