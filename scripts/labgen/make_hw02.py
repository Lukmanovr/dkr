#!/usr/bin/env python
"""Generate Homework 2 (Weeks 5-7: reasoning, GCN, the design space) — student
and solution notebooks. Part A: derivations (markdown answers). Part B:
deterministic implementation exercises with assertion autograders. Part C: an
ablation-extension investigation on the provided Week-7 harness.

    python scripts/labgen/make_hw02.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "homeworks" / "hw02_deep_models.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "homeworks" / "hw02_deep_models.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(solution, stub): return ("code", solution, stub)
def answer(question, solution): return ("markdown-answer", question, solution)


CELLS = [
    md("""# Homework 2 · Deep models — reasoning geometry, spectra, and the design space

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/homeworks/hw02_deep_models.ipynb)

**Covers Weeks 5–7 · out Oct 15 · due Thursday, Oct 29, 23:59 (Moodle) · ≈ 5–10 hours**

Three parts. **A — Derivations** (30%): written proofs in the markdown cells (LaTeX
or a legible photographed page). **B — Implementation** (60%, autograded): three
*deterministic* exercises — no training, no seeds to blame — where you build the
week's theory and the asserts check it held. **C — Investigation** (10%): extend the
Week-7 ablation grid by one axis of your choosing, on the provided harness.

Rules: individual work; AI assistants allowed *with disclosure* per the honor code
(state tool + purpose in the final cell); you must be able to explain any line and
any proof step on request. Rubric, late policy, FAQ: see the
[HW2 page](https://lukmanovr.github.io/dkr/homeworks/hw2.html).
"""),

    md("""## 0 · Setup"""),

    code("""import os, sys, random, math, statistics, itertools

SMOKE = os.environ.get("SMOKE", "") == "1"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    %pip install -q torch_geometric==2.8.0.post1

import numpy as np
import networkx as nx

SEED = 27
random.seed(SEED); np.random.seed(SEED)
print("numpy", np.__version__, "· networkx", nx.__version__, "— environment OK")
print("(torch is imported only in Part C — Parts A and B are numpy-only on purpose)")"""),

    md("""## Part A · Derivations (30 pts)

Answer in the markdown cell after each question. State what you use; every proof
here fits in a few lines if you use the right Week 5–7 fact.

### A1 (8 pts) · Boxes close, points don't

Let a *box* in $\\mathbb R^d$ be $B = \\{x : \\ell_i \\le x_i \\le u_i\\}$.
**(a)** Prove that the intersection of two boxes is a box (possibly empty), giving
the formula for the new $\\ell, u$. **(b)** Show with a concrete 1-dimensional
example that no single *point* embedding of a conjunctive query can represent an
answer **set** of two entities faithfully — state what nearest-neighbor retrieval
from a point must get wrong. **(c)** In one sentence: why does (a) + (b) make boxes
the right container for conjunctive queries?
"""),

    answer("""*(your answer for A1 here)*""",
           """**A1 solution.**
**(a)** $B_1 \\cap B_2 = \\{x : \\max(\\ell^1_i, \\ell^2_i) \\le x_i \\le
\\min(u^1_i, u^2_i)\\}$ — the constraints combine per coordinate, so the
intersection is again a coordinate-wise interval product, i.e. a box; it is empty
iff $\\max(\\ell_i) > \\min(u_i)$ in some dimension. $\\square$
**(b)** Entities at $x = 0$ and $x = 2$ are both answers. A point query $q$ ranks
by distance: to score both answers *equally best* it must sit at $q = 1$ — where
the non-answer entity at $x = 1$ (if one exists) is ranked strictly better than
both true answers. A point can encode one location, not a *region*; some entity
ordering is always wrong when the answer set is not a singleton. The box
$[0, 2]$ contains exactly the two answers.
**(c)** Conjunction is intersection of answer sets; boxes represent sets and are
closed under exactly that operator — representation follows the operators the
problem requires (the Week-5 design principle).""",),

    md("""### A2 (10 pts) · The spectrum does the smoothing

For the 4-cycle $C_4$, form $\\hat A = \\tilde D^{-1/2}(A + I)\\tilde D^{-1/2}$.
**(a)** Show $\\hat A = \\tfrac13(A + I)$ here, and compute all four eigenvalues.
(Hint: $A + I$ is circulant with first row $(1, 1, 0, 1)$; its eigenvalues are
$1 + \\omega^k + \\omega^{3k}$ for $\\omega = i$, $k = 0..3$ — or just diagonalize
by hand.) **(b)** Give the dominant eigenvector and the limit of $\\hat A^K x$ as
$K \\to \\infty$. **(c)** Compute the smallest $K$ for which *every* non-dominant
component has shrunk by a factor of at least 100.
"""),

    answer("""*(your answer for A2 here)*""",
           """**A2 solution.**
**(a)** Every node of $C_4$ has degree 2, so $\\tilde d = 3$ uniformly and
$\\hat A = \\tfrac13(A+I)$. Circulant eigenvalues of $(1,1,0,1)$:
$k{=}0: 3$, $k{=}1: 1 + i - i = 1$, $k{=}2: 1 - 1 - 1 = -1$, $k{=}3: 1$.
Dividing by 3: $\\lambda = \\{1, \\tfrac13, \\tfrac13, -\\tfrac13\\}$.
**(b)** Dominant eigenvector $\\propto \\tilde D^{1/2}\\mathbf 1 \\propto
\\mathbf 1$ (regular graph). $\\hat A^K x \\to (\\bar x)\\mathbf 1$ where $\\bar
x$ is the mean of $x$ — total collapse to a constant vector.
**(c)** All non-dominant magnitudes are $\\tfrac13$: need $(1/3)^K \\le 1/100$,
i.e. $K \\ge \\log 100 / \\log 3 \\approx 4.19$ → $\\mathbf{K = 5}$ layers.
Five plain propagations and $C_4$'s features are constant to 1% — the concrete
version of "oversmoothing is a property of the operator, not of training".""",),

    md("""### A3 (6 pts) · Static attention, on two receivers

Two receivers $v_1, v_2$ and two senders $u_1, u_2$ with distinct features. You
want an attention pattern where $v_1$ prefers $u_1$ and $v_2$ prefers $u_2$.
Using the decomposition $e_{vu} = \\text{LeakyReLU}(\\underbrace{\\mathbf
a_1^\\top \\mathbf W h_v}_{\\text{receiver term}} + \\underbrace{\\mathbf
a_2^\\top \\mathbf W h_u}_{\\text{sender term}})$, prove no GAT parameters
achieve it, and say in one line why GATv2's $e_{vu} = \\mathbf a^\\top
\\text{LeakyReLU}(\\mathbf W [h_v \\Vert h_u])$ escapes the argument.
"""),

    answer("""*(your answer for A3 here)*""",
           """**A3 solution.** Fix any parameters and let $s_j = \\mathbf a_2^\\top
\\mathbf W h_{u_j}$ be the sender scores. For receiver $v$, $e_{vu_j} =
\\text{LeakyReLU}(c_v + s_j)$ with $c_v$ the receiver term — a *constant* within
$v$'s row. LeakyReLU is strictly increasing, so $e_{vu_1} > e_{vu_2} \\iff s_1 >
s_2$: the comparison does not involve $c_v$ at all. Hence *every* receiver ranks
the senders identically — $v_1$ preferring $u_1$ while $v_2$ prefers $u_2$ is
unsatisfiable. $\\square$ GATv2 applies the nonlinearity *before* the shared
$\\mathbf a$, so receiver and sender features mix inside a non-additive function;
the score is no longer (monotone function of) receiver-constant + sender-constant
and the separation argument dies.""",),

    md("""### A4 (6 pts) · Choosing your invariance

**(a)** Prove: mean aggregation is invariant to replacing a neighborhood multiset
$S$ by $k$ copies of itself ($S^{(k)}$, each element repeated $k$ times), and max
aggregation is invariant to *any* change of multiplicities. **(b)** Give one task
where this invariance is *fatal* and one where it is *exactly what you want* —
one sentence each.
"""),

    answer("""*(your answer for A4 here)*""",
           """**A4 solution.**
**(a)** Mean: $\\frac{1}{k|S|}\\sum_{x \\in S} k\\,x = \\frac{1}{|S|}\\sum_{x
\\in S} x$ — the $k$ cancels. Max: $\\max$ depends only on the *support* (which
values occur), so any multiplicity change with the same support leaves it fixed.
$\\square$
**(b)** Fatal: counting tasks — e.g. molecule property prediction where the
*number* of identical substituents matters ({t} vs {t,t} is invisible to both),
or degree-sensitive node roles. Exactly right: population-mix tasks — e.g.
classifying a node by the *proportions* of neighbor types regardless of how many
neighbors were sampled (mean), or detecting whether at least one anomalous
neighbor type is present at any count (max). Full marks for any pair with the
correct mechanism named.""",),

    md("""## Part B · Implementation (60 pts, autograded)

Deterministic on purpose: every assert either always passes or always fails with
your code — nothing to reroll. The visible asserts are most of the grade; hidden
variants run the same checks with different numbers.

### B1 (20 pts) · Query2box-lite — reasoning as geometry

The toy KG below places 10 biomedical entities in 2-D by hand. Relations act on
*boxes*: translate the center, then widen. Implement the three box operations;
the asserts then answer a real two-hop conjunctive query with them:
*"proteins targeted by drugs that treat Inflammation AND Pain."*
""" ),

    todo("""ENT = {
    "Aspirin": (1.0, 1.0), "Ibuprofen": (1.0, 3.0), "Naproxen": (1.0, 5.0),
    "Paracetamol": (4.0, 4.0),
    "COX1": (7.0, 1.0), "COX2": (7.0, 3.0), "TRPV1": (7.0, 5.0),
    "Inflammation": (11.0, 2.0), "Pain": (11.0, 4.0), "Fever": (11.0, 6.0),
}
# relation = (translation vector, widening per dimension), applied to a box
REL = {
    "treated_by": ((-10.0, 0.0), (0.5, 1.5)),   # disease -> drugs that treat it
    "targets":    ((6.0, 0.0),  (0.5, 0.5)),    # drug -> proteins it binds
}


def point_box(p):
    \"\"\"A degenerate box: center p, half-size (0, 0).\"\"\"
    return (np.array(p, float), np.zeros(2))


def translate(box, rel):
    \"\"\"Apply a relation: center += translation; half-size += widening.
    Return (center, half) as np arrays.\"\"\"
    ### BEGIN SOLUTION
    (c, h), (t, w) = box, rel
    return (c + np.asarray(t, float), h + np.asarray(w, float))
    ### END SOLUTION


def intersect(box1, box2):
    \"\"\"Intersection of two boxes: per-dimension [max(lo), min(hi)].
    Return (center, half); if empty in any dimension, return None.\"\"\"
    ### BEGIN SOLUTION
    lo = np.maximum(box1[0] - box1[1], box2[0] - box2[1])
    hi = np.minimum(box1[0] + box1[1], box2[0] + box2[1])
    if (lo > hi).any():
        return None
    return ((lo + hi) / 2, (hi - lo) / 2)
    ### END SOLUTION


def inside(box, p):
    \"\"\"Is point p inside the (closed) box?\"\"\"
    ### BEGIN SOLUTION
    c, h = box
    return bool((np.abs(np.asarray(p, float) - c) <= h + 1e-9).all())
    ### END SOLUTION


def answers(box):
    return {e for e, p in ENT.items() if inside(box, p)}


# — hop 1a: drugs that treat Inflammation
b_inf = translate(point_box(ENT["Inflammation"]), REL["treated_by"])
assert answers(b_inf) == {"Aspirin", "Ibuprofen"}, (
    f"treated_by(Inflammation) should be a box at center (1,2), half (0.5,1.5) "
    f"containing Aspirin and Ibuprofen — got {answers(b_inf)}"
)
# — hop 1b: drugs that treat Pain
b_pain = translate(point_box(ENT["Pain"]), REL["treated_by"])
assert answers(b_pain) == {"Ibuprofen", "Naproxen"}, f"got {answers(b_pain)}"

# — conjunction: treats BOTH  →  intersection of the two boxes
b_both = intersect(b_inf, b_pain)
assert b_both is not None and answers(b_both) == {"Ibuprofen"}, (
    f"the intersection box should contain exactly Ibuprofen — got "
    f"{None if b_both is None else answers(b_both)}. Check: intersection uses "
    f"max of lows and min of highs, per dimension."
)

# — hop 2: proteins targeted by those drugs
b_prot = translate(b_both, REL["targets"])
assert answers(b_prot) == {"COX2"}, (
    f"targets(Ibuprofen-box) should contain exactly COX2 — got {answers(b_prot)}"
)

# — and an empty intersection must be recognized, not faked
assert intersect(point_box((0, 0)), point_box((5, 5))) is None, (
    "two disjoint degenerate boxes must intersect to None (empty), not a box"
)
print("B1 ✓ — a 2-hop conjunctive query answered by translate → intersect → translate")""",
         stub="""ENT = {
    "Aspirin": (1.0, 1.0), "Ibuprofen": (1.0, 3.0), "Naproxen": (1.0, 5.0),
    "Paracetamol": (4.0, 4.0),
    "COX1": (7.0, 1.0), "COX2": (7.0, 3.0), "TRPV1": (7.0, 5.0),
    "Inflammation": (11.0, 2.0), "Pain": (11.0, 4.0), "Fever": (11.0, 6.0),
}
# relation = (translation vector, widening per dimension), applied to a box
REL = {
    "treated_by": ((-10.0, 0.0), (0.5, 1.5)),   # disease -> drugs that treat it
    "targets":    ((6.0, 0.0),  (0.5, 0.5)),    # drug -> proteins it binds
}


def point_box(p):
    \"\"\"A degenerate box: center p, half-size (0, 0).\"\"\"
    return (np.array(p, float), np.zeros(2))


def translate(box, rel):
    \"\"\"Apply a relation: center += translation; half-size += widening.
    Return (center, half) as np arrays.\"\"\"
    # TODO: 2 lines.
    raise NotImplementedError


def intersect(box1, box2):
    \"\"\"Intersection of two boxes: per-dimension [max(lo), min(hi)].
    Return (center, half); if empty in any dimension, return None.\"\"\"
    # TODO: compute lo/hi corners, check emptiness, convert back. ~5 lines.
    raise NotImplementedError


def inside(box, p):
    \"\"\"Is point p inside the (closed) box?\"\"\"
    # TODO: 2 lines (use a small tolerance, e.g. 1e-9).
    raise NotImplementedError


def answers(box):
    return {e for e, p in ENT.items() if inside(box, p)}


# — hop 1a: drugs that treat Inflammation
b_inf = translate(point_box(ENT["Inflammation"]), REL["treated_by"])
assert answers(b_inf) == {"Aspirin", "Ibuprofen"}, (
    f"treated_by(Inflammation) should be a box at center (1,2), half (0.5,1.5) "
    f"containing Aspirin and Ibuprofen — got {answers(b_inf)}"
)
# — hop 1b: drugs that treat Pain
b_pain = translate(point_box(ENT["Pain"]), REL["treated_by"])
assert answers(b_pain) == {"Ibuprofen", "Naproxen"}, f"got {answers(b_pain)}"

# — conjunction: treats BOTH  →  intersection of the two boxes
b_both = intersect(b_inf, b_pain)
assert b_both is not None and answers(b_both) == {"Ibuprofen"}, (
    f"the intersection box should contain exactly Ibuprofen — got "
    f"{None if b_both is None else answers(b_both)}. Check: intersection uses "
    f"max of lows and min of highs, per dimension."
)

# — hop 2: proteins targeted by those drugs
b_prot = translate(b_both, REL["targets"])
assert answers(b_prot) == {"COX2"}, (
    f"targets(Ibuprofen-box) should contain exactly COX2 — got {answers(b_prot)}"
)

# — and an empty intersection must be recognized, not faked
assert intersect(point_box((0, 0)), point_box((5, 5))) is None, (
    "two disjoint degenerate boxes must intersect to None (empty), not a box"
)
print("B1 ✓ — a 2-hop conjunctive query answered by translate → intersect → translate")"""),

    md("""### B2 (20 pts) · Oversmoothing, predicted then measured

Your A2 proof says the contraction rate of plain propagation is the second
eigenvalue. Here you *measure* it on the karate club and check the theory to two
decimal places. Implement `build_ahat` and `contraction_rate`; the asserts
compare your measured rate against the spectrum numpy computes.
"""),

    todo("""KG = nx.karate_club_graph()   # careful: its edges carry weights — ignore them


def build_ahat(G):
    \"\"\"Return the dense n×n numpy matrix D̃^{-1/2} (A + I) D̃^{-1/2},
    with A the UNWEIGHTED adjacency matrix (weight=None!).\"\"\"
    ### BEGIN SOLUTION
    A = nx.to_numpy_array(G, weight=None)
    At = A + np.eye(len(A))
    d = At.sum(1)
    Dm = np.diag(d ** -0.5)
    return Dm @ At @ Dm
    ### END SOLUTION


def contraction_rate(Ahat, x0, K=40):
    \"\"\"Propagate x_{k+1} = Ahat @ x_k from x0. Let x* be the (eventual) limit
    component: the projection of x0 onto the dominant eigenvector. Return the
    ratio ||x_{K+1} - x*|| / ||x_K - x*|| — the per-step shrink factor of
    everything that is NOT the dominant direction.\"\"\"
    ### BEGIN SOLUTION
    lam, V = np.linalg.eigh(Ahat)
    v1 = V[:, np.argmax(lam)]
    xstar = (v1 @ x0) * v1
    x = x0.copy()
    for _ in range(K):
        x = Ahat @ x
    return float(np.linalg.norm(Ahat @ x - xstar) / np.linalg.norm(x - xstar))
    ### END SOLUTION


Ahat = build_ahat(KG)
assert Ahat.shape == (34, 34) and np.allclose(Ahat, Ahat.T), "Â must be 34×34 and symmetric"
assert abs(Ahat[0, 0] - 1 / 17) < 1e-9, (
    "Â[0,0] should be 1/(deg+1) = 1/17 for node 0 (degree 16). If not: either "
    "you forgot the self-loop, or nx.to_numpy_array picked up edge WEIGHTS — "
    "pass weight=None (the karate club's edges carry weights!)"
)

lam = np.linalg.eigvalsh(Ahat)
lam_sorted = np.sort(np.abs(lam))[::-1]
assert abs(lam_sorted[0] - 1.0) < 1e-9, "largest eigenvalue of Â is exactly 1"
lam2 = lam_sorted[1]

x0 = np.random.default_rng(0).standard_normal(34)
rate = contraction_rate(Ahat, x0)
assert abs(rate - lam2) < 0.02, (
    f"measured contraction {rate:.4f} vs |λ₂| = {lam2:.4f} — after 40 steps the "
    f"slowest-dying component dominates, so these must agree to ~2 decimals. "
    f"Check that x* projects onto the eigenvector of the LARGEST eigenvalue."
)
print(f"B2 ✓ — theory |λ₂| = {lam2:.4f}, measured rate = {rate:.4f}: the spectrum IS the smoothing speed")""",
         stub="""KG = nx.karate_club_graph()   # careful: its edges carry weights — ignore them


def build_ahat(G):
    \"\"\"Return the dense n×n numpy matrix D̃^{-1/2} (A + I) D̃^{-1/2},
    with A the UNWEIGHTED adjacency matrix (weight=None!).\"\"\"
    # TODO: adjacency (weight=None!), add I, degree vector, sandwich. ~5 lines.
    raise NotImplementedError


def contraction_rate(Ahat, x0, K=40):
    \"\"\"Propagate x_{k+1} = Ahat @ x_k from x0. Let x* be the (eventual) limit
    component: the projection of x0 onto the dominant eigenvector. Return the
    ratio ||x_{K+1} - x*|| / ||x_K - x*|| — the per-step shrink factor of
    everything that is NOT the dominant direction.\"\"\"
    # TODO: eigh for the dominant eigenvector, project x0 for x*, propagate K
    # steps, return the ratio. ~7 lines.
    raise NotImplementedError


Ahat = build_ahat(KG)
assert Ahat.shape == (34, 34) and np.allclose(Ahat, Ahat.T), "Â must be 34×34 and symmetric"
assert abs(Ahat[0, 0] - 1 / 17) < 1e-9, (
    "Â[0,0] should be 1/(deg+1) = 1/17 for node 0 (degree 16). If not: either "
    "you forgot the self-loop, or nx.to_numpy_array picked up edge WEIGHTS — "
    "pass weight=None (the karate club's edges carry weights!)"
)

lam = np.linalg.eigvalsh(Ahat)
lam_sorted = np.sort(np.abs(lam))[::-1]
assert abs(lam_sorted[0] - 1.0) < 1e-9, "largest eigenvalue of Â is exactly 1"
lam2 = lam_sorted[1]

x0 = np.random.default_rng(0).standard_normal(34)
rate = contraction_rate(Ahat, x0)
assert abs(rate - lam2) < 0.02, (
    f"measured contraction {rate:.4f} vs |λ₂| = {lam2:.4f} — after 40 steps the "
    f"slowest-dying component dominates, so these must agree to ~2 decimals. "
    f"Check that x* projects onto the eigenvector of the LARGEST eigenvalue."
)
print(f"B2 ✓ — theory |λ₂| = {lam2:.4f}, measured rate = {rate:.4f}: the spectrum IS the smoothing speed")"""),

    md("""### B3 (20 pts) · Static attention, caught by experiment

Your A3 proof says every GAT receiver ranks senders identically. Implement both
scoring functions; the asserts then (i) verify the theorem across 100 random
parameter draws — the argmax column is constant *every single time* — and (ii)
exhibit GATv2 parameters (provided) under which each receiver prefers a
*different* sender, the pattern your proof shows GAT can never produce.

Setup: 5 receivers and 5 senders, one-hot features $h_i = e_i \\in \\mathbb R^5$.
"""),

    todo("""def leaky(z, s=0.2):
    return np.where(z > 0, z, s * z)


def gat_score(a, W, hv, hu):
    \"\"\"GAT (2018): LeakyReLU( a · [W hv ‖ W hu] ).  a: (2m,), W: (m, d).\"\"\"
    ### BEGIN SOLUTION
    z = np.concatenate([W @ hv, W @ hu])
    return float(leaky(a @ z))
    ### END SOLUTION


def gatv2_score(a, W, hv, hu):
    \"\"\"GATv2 (2022): a · LeakyReLU( W [hv ‖ hu] ).  a: (m,), W: (m, 2d).\"\"\"
    ### BEGIN SOLUTION
    z = W @ np.concatenate([hv, hu])
    return float(a @ leaky(z))
    ### END SOLUTION


H = np.eye(5)                      # one-hot features for 5 nodes
rng = np.random.default_rng(SEED)

# (i) the theorem, empirically: 100 random GATs, argmax constant across receivers
constant_cols = 0
for _ in range(100):
    a, W = rng.standard_normal(8), rng.standard_normal((4, 5))
    S = np.array([[gat_score(a, W, H[v], H[u]) for u in range(5)] for v in range(5)])
    if len(set(S.argmax(axis=1))) == 1:
        constant_cols += 1
assert constant_cols == 100, (
    f"static attention means EVERY random GAT gives all receivers the same "
    f"favorite sender — got a constant argmax in only {constant_cols}/100 draws. "
    f"Check the concatenation order [W hv ‖ W hu] and that LeakyReLU wraps the sum."
)

# (ii) GATv2 with W = [I | -I], a = -1: score is -Σ leaky(hv - hu),
# maximized when hu = hv — each receiver prefers ITSELF as sender
W2 = np.concatenate([np.eye(5), -np.eye(5)], axis=1)   # (5, 10)
a2 = -np.ones(5)
S2 = np.array([[gatv2_score(a2, W2, H[v], H[u]) for u in range(5)] for v in range(5)])
assert (S2.argmax(axis=1) == np.arange(5)).all(), (
    f"with W=[I|-I], a=-1, receiver v's best sender must be v itself "
    f"(diagonal argmax) — got {S2.argmax(axis=1)}. Check the nonlinearity is "
    f"applied BEFORE the dot with a."
)
print("B3 ✓ — 100/100 GATs are static; one line of GATv2 parameters is not")""",
         stub="""def leaky(z, s=0.2):
    return np.where(z > 0, z, s * z)


def gat_score(a, W, hv, hu):
    \"\"\"GAT (2018): LeakyReLU( a · [W hv ‖ W hu] ).  a: (2m,), W: (m, d).\"\"\"
    # TODO: 2 lines. Transform both, concatenate, dot with a, THEN LeakyReLU.
    raise NotImplementedError


def gatv2_score(a, W, hv, hu):
    \"\"\"GATv2 (2022): a · LeakyReLU( W [hv ‖ hu] ).  a: (m,), W: (m, 2d).\"\"\"
    # TODO: 2 lines. Concatenate raw features, transform, LeakyReLU, THEN dot.
    raise NotImplementedError


H = np.eye(5)                      # one-hot features for 5 nodes
rng = np.random.default_rng(SEED)

# (i) the theorem, empirically: 100 random GATs, argmax constant across receivers
constant_cols = 0
for _ in range(100):
    a, W = rng.standard_normal(8), rng.standard_normal((4, 5))
    S = np.array([[gat_score(a, W, H[v], H[u]) for u in range(5)] for v in range(5)])
    if len(set(S.argmax(axis=1))) == 1:
        constant_cols += 1
assert constant_cols == 100, (
    f"static attention means EVERY random GAT gives all receivers the same "
    f"favorite sender — got a constant argmax in only {constant_cols}/100 draws. "
    f"Check the concatenation order [W hv ‖ W hu] and that LeakyReLU wraps the sum."
)

# (ii) GATv2 with W = [I | -I], a = -1: score is -Σ leaky(hv - hu),
# maximized when hu = hv — each receiver prefers ITSELF as sender
W2 = np.concatenate([np.eye(5), -np.eye(5)], axis=1)   # (5, 10)
a2 = -np.ones(5)
S2 = np.array([[gatv2_score(a2, W2, H[v], H[u]) for u in range(5)] for v in range(5)])
assert (S2.argmax(axis=1) == np.arange(5)).all(), (
    f"with W=[I|-I], a=-1, receiver v's best sender must be v itself "
    f"(diagonal argmax) — got {S2.argmax(axis=1)}. Check the nonlinearity is "
    f"applied BEFORE the dot with a."
)
print("B3 ✓ — 100/100 GATs are static; one line of GATv2 parameters is not")"""),

    md("""## Part C · Investigation (10 pts) — one new axis for the ablation grid

Lab 7's grid measured architecture × depth × residuals. Pick **one** new axis and
measure its effect with the harness below, holding everything else at the Lab-7
protocol. Choose from: **BatchNorm** (± between layers), **DropEdge** (edge
dropout ± , see `torch_geometric.utils.dropout_edge`), **width** (16 vs 64),
**GATv2 vs GAT**, or **jumping knowledge** (concat all layer outputs). Rules of
evidence, same as the lecture: change one factor; ≥2 seeds per cell; report
mean ± std; every claim cites its cells; one *negative* finding is worth as much
as a positive one.

The harness below is Lab 7's, verbatim, plus a `variant` hook where your axis
plugs in. The demo run (GCN, depth 4, ± your variant) is the shape your report
should take.
"""),

    code("""import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv, GINConv

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = Planetoid(root="data/Planetoid", name="Cora")[0].to(DEV)

CONVS = {
    "GCN": lambda i, o: GCNConv(i, o),
    "SAGE": lambda i, o: SAGEConv(i, o),
    "GAT": lambda i, o: GATConv(i, o, heads=4, concat=False),
    "GATv2": lambda i, o: GATv2Conv(i, o, heads=4, concat=False),
    "GIN": lambda i, o: GINConv(torch.nn.Sequential(
        torch.nn.Linear(i, o), torch.nn.ReLU(), torch.nn.Linear(o, o))),
}


class Net(torch.nn.Module):
    def __init__(self, arch, depth, residual, hidden=64, batchnorm=False, jk=False):
        super().__init__()
        self.residual, self.jk = residual, jk
        dims = [data.num_features] + [hidden] * depth
        self.convs = torch.nn.ModuleList(
            [CONVS[arch](dims[i], dims[i + 1]) for i in range(depth)])
        self.norms = torch.nn.ModuleList(
            [torch.nn.BatchNorm1d(hidden) if batchnorm else torch.nn.Identity()
             for _ in range(depth)])
        self.out = torch.nn.Linear(hidden * (depth if jk else 1), 7)

    def forward(self, x, ei):
        layer_outs = []
        for conv, norm in zip(self.convs, self.norms):
            h = F.dropout(F.relu(norm(conv(x, ei))), 0.5, self.training)
            x = h + x if (self.residual and h.shape == x.shape) else h
            layer_outs.append(x)
        return self.out(torch.cat(layer_outs, dim=1) if self.jk else x)


def run(arch, depth, residual, seed, drop_edge=0.0, **net_kw):
    from torch_geometric.utils import dropout_edge
    torch.manual_seed(seed)
    net = Net(arch, depth, residual, **net_kw).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_test, patience = 0.0, 0.0, 0
    for ep in range(40 if SMOKE else 200):
        net.train(); opt.zero_grad()
        ei = dropout_edge(data.edge_index, p=drop_edge)[0] if drop_edge else data.edge_index
        out = net(data.x, ei)
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


def cell(label, accs):
    m = 100 * statistics.mean(accs)
    s = 100 * (statistics.stdev(accs) if len(accs) > 1 else 0.0)
    print(f"  {label:<28} {m:5.1f} ± {s:3.1f}   (n={len(accs)})")
    return m, s


# demo of the reporting shape (this demo is NOT your investigation — pick an
# axis, then mirror these lines for it):
seeds = [0] if SMOKE else [0, 1]
print("demo axis — jumping knowledge on depth-4 GCN:")
cell("GCN·4·plain", [run("GCN", 4, False, s) for s in seeds])
cell("GCN·4·plain + JK", [run("GCN", 4, False, s, jk=True) for s in seeds])
print("(your investigation: 2–4 cells like these, ≥2 seeds each, full runs — not SMOKE)")"""),

    answer("""### Your Part C report *(graded — write it here)*

**Axis chosen:** …

**Hypothesis (before running):** one sentence — which cells you expect to differ
and *why*, naming a mechanism from Weeks 6–7.

**Cells (mean ± std, ≥2 seeds each):** paste your `cell(...)` lines.

**Claims (2–3 sentences, each citing cells):** …

**Scope (one sentence):** what these numbers do *not* license you to conclude.
""",
           """### Part C — model answer (structure; your numbers will differ)

**Axis chosen:** BatchNorm at depth 4.

**Hypothesis:** BatchNorm re-separates features after each propagation, so it
should recover part of the depth-4 crash *without* residuals — mechanism:
oversmoothing shrinks feature variance along non-dominant directions (A2/B2),
and BatchNorm rescales exactly that variance.

**Cells:** e.g.
`GAT·4·plain 26.2 ± 9.8` · `GAT·4·plain+BN 74.x ± 1.x` ·
`GAT·4·res 80.5 ± 1.0` · `GAT·4·res+BN 79.x ± 1.x`.

**Claims:** (1) BatchNorm alone rescues most of the depth-4 GAT crash
(26 → ~74, cells 1–2), consistent with the variance mechanism. (2) On top of
residuals it adds nothing outside noise (cells 3–4) — the two repairs are
substitutes here, not complements. Negative findings phrased exactly like this
earn full credit.

**Scope:** Cora, public split, this protocol, 2 seeds; no claim about other
datasets, depths beyond 4, or interaction with DropEdge.

*Grading: axis correctly isolated (3), ≥2 seeds + mean±std (2), claims cite
cells (3), honest scope (2).*""",),

    md("""## Submission

One executed notebook on Moodle: Part A answers written, all three B checks ✓
(run *Runtime → Restart and run all* first), Part C run **full** (not SMOKE) with
the report filled in.

**AI policy** (honor code): AI assistants are allowed *with disclosure* — state
below which tools you used and for what. You must be able to explain any line
and any proof step on request; undeclared use or inability to explain is a
violation.

*Disclosure: …*
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
        elif kind == "markdown-answer":
            nb.cells.append(nbf.v4.new_markdown_cell(text if student else extra))
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
        assert "solution." not in cell.source.lower()[:30], "answer leak"
    STUDENT_OUT.parent.mkdir(parents=True, exist_ok=True)
    SOLUTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(student, STUDENT_OUT)
    nbf.write(solution, SOLUTION_OUT)
    print("wrote", STUDENT_OUT)
    print("wrote", SOLUTION_OUT)


if __name__ == "__main__":
    main()
