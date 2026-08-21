#!/usr/bin/env python
"""Generate Homework 1 (Weeks 1-4: foundations) — student and solution notebooks.
Part A: derivations (answered in markdown). Part B: implementation with
assertion autograders. Part C: a small open investigation.

    python scripts/labgen/make_hw01.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "homeworks" / "hw01_foundations.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "homeworks" / "hw01_foundations.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(text, stub): return ("code", text, stub)
def answer(text, solution): return ("markdown-answer", text, solution)


CELLS = [
    md("""# Homework 1 · Foundations — representations, classical ML, embeddings, KG geometry

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/homeworks/hw01_foundations.ipynb)

**Covers Weeks 1–4 · out Sep 17 · due Thursday, Oct 1, 23:59 (Moodle) · ≈ 5–10 hours**

Three parts. **A — Derivations** (30%): written proofs, answered in the markdown cells
(LaTeX welcome; a photographed hand-written page pasted as an image is also fine if
legible). **B — Implementation** (60%): code against assertion autograders — the
asserts you can see are the majority of the grade; hidden variants of the same checks
run on our side. **C — Investigation** (10%): one small experiment, honestly reported.

Rules: individual work; AI assistants allowed *with disclosure* per the honor code
(state tool + purpose in the final cell); you must be able to explain any line and any
proof step on request. Late policy and rubric details: see the
[HW1 page](https://lukmanovr.github.io/dkr/homeworks/hw1.html).
"""),

    md("""## 0 · Setup"""),

    code("""import os, sys, random, math
from collections import Counter

SMOKE = os.environ.get("SMOKE", "") == "1"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    %pip install -q torch_geometric==2.8.0.post1

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx

SEED = 17
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("environment OK")"""),

    md("""---
# Part A · Derivations (30 points)

Write your answers **in the indicated markdown cells**. State what you are proving,
justify each step, and mark the end of a proof. Partial credit follows the structure
of the proof, so structure your proof.
"""),

    md("""## A1 · Relabeling and the spectrum (10 pts)

Let $G$ be a graph with adjacency matrix $\\mathbf{A}$, and let $\\pi$ be a permutation
of its nodes with permutation matrix $\\mathbf{P}$ (i.e. $P_{\\pi(i), i} = 1$).

**(a)** (3 pts) Show that the adjacency matrix of the relabeled graph is
$\\mathbf{A}' = \\mathbf{P}\\mathbf{A}\\mathbf{P}^\\top$.

**(b)** (3 pts) Conclude that the *multiset of eigenvalues* of $\\mathbf{A}$ is
invariant under relabeling — a legitimate graph feature in the Week-1 sense.

**(c)** (4 pts) The converse fails: exhibit two **non-isomorphic** graphs on 5 nodes
with identical adjacency spectra, and verify the non-isomorphism with a one-line
structural argument (no eigenvalue computation needed for this part — you will verify
the spectra numerically in B1).
"""),

    answer("""**A1 — your answer:**

*(a)* …

*(b)* …

*(c)* my two graphs are: …
""", """**A1 — solution.**

*(a)* Entry $(\\mathbf{P}\\mathbf{A}\\mathbf{P}^\\top)_{\\pi(i)\\pi(j)}
= \\sum_{k,l} P_{\\pi(i)k} A_{kl} P_{\\pi(j)l} = A_{ij}$, since $P_{\\pi(i)k} = 1$ iff
$k = i$. So the $(\\pi(i), \\pi(j))$ entry of $\\mathbf{A}'$ equals the $(i,j)$ entry
of $\\mathbf{A}$ — precisely the adjacency of the relabeled graph. $\\square$

*(b)* $\\mathbf{P}$ is orthogonal ($\\mathbf{P}^\\top = \\mathbf{P}^{-1}$), so
$\\mathbf{A}' = \\mathbf{P}\\mathbf{A}\\mathbf{P}^{-1}$ is similar to $\\mathbf{A}$;
similar matrices have identical characteristic polynomials, hence identical eigenvalue
multisets. $\\square$

*(c)* The classic cospectral pair: the star $K_{1,4}$ and the disjoint union
$C_4 \\cup K_1$. Both have spectrum $\\{2, 0, 0, 0, -2\\}$. Non-isomorphic in one
line: one is connected, the other is not (or: max degree 4 vs 2). The moral is
Week 9's in miniature: spectra — like WL fingerprints — are sound but not complete
invariants.
"""),

    md("""## A2 · PageRank needs its teleport (8 pts)

Consider the two-page web $X \\to Y$, $Y \\to X$, and pure link-following
($\\beta = 1$), i.e. $\\mathbf{r}^{(k+1)} = M \\mathbf{r}^{(k)}$.

**(a)** (3 pts) Starting from $\\mathbf{r}^{(0)} = (1, 0)$, compute the trajectory and
show that power iteration never converges.

**(b)** (3 pts) With damping $\\beta \\in (0,1)$ and uniform teleport, compute the
(unique) fixed point and show the oscillation decays at rate $\\beta$ — connect to the
contraction proof from the Week 2 lecture.

**(c)** (2 pts) In one sentence: what structural property of this tiny web caused (a),
and why does it not contradict the Week 2 existence proposition?
""" ),

    answer("""**A2 — your answer:**

*(a)* …

*(b)* …

*(c)* …
""", """**A2 — solution.**

*(a)* $M = \\begin{pmatrix} 0 & 1 \\\\ 1 & 0 \\end{pmatrix}$, so the iteration swaps
coordinates: $(1,0) \\to (0,1) \\to (1,0) \\to \\cdots$ — a 2-cycle, never convergent.

*(b)* Fixed point: by symmetry $\\mathbf{r}^* = (\\tfrac12, \\tfrac12)$ (check:
$\\beta M \\mathbf{r}^* + \\frac{1-\\beta}{2}\\mathbf{1} = \\mathbf{r}^*$). Error
dynamics: $\\mathbf{e}^{(k+1)} = \\beta M \\mathbf{e}^{(k)}$, and
$\\|M\\mathbf{e}\\|_1 = \\|\\mathbf{e}\\|_1$, so
$\\|\\mathbf{e}^{(k)}\\|_1 = \\beta^k \\|\\mathbf{e}^{(0)}\\|_1 \\to 0$: the
oscillation still flips sign each step but shrinks geometrically at rate $\\beta$ —
exactly the contraction constant from the lecture's Banach argument.

*(c)* The web is bipartite/periodic (every cycle has even length), so mass sloshes
with period 2; the Week 2 proposition assumed $\\beta < 1$ precisely to break
periodicity, so no contradiction — the theorem's hypothesis is doing real work.
"""),

    md("""## A3 · What SGNS converges to (6 pts)

Fix one pair of nodes and let $s = \\mathbf{z}_u^\\top \\mathbf{z}'_v$ be its score.
Suppose the training stream shows this pair as a **positive** $p$ times, and as a
**sampled negative** $q$ times. The pair's total contribution to the SGNS objective is
$$
J(s) = p \\log \\sigma(s) + q \\log \\sigma(-s).
$$

**(a)** (4 pts) Maximize $J$ over $s$: show the optimum satisfies
$\\sigma(s^*) = \\frac{p}{p+q}$, i.e. $s^* = \\log(p/q)$.

**(b)** (2 pts) For DeepWalk with $k$ negatives drawn from the unigram distribution,
$q \\approx k \\cdot \\#(u) \\#(v) / |\\mathcal{D}|$. Substitute and recognize the
resulting $s^*$ — which Week 3 proposition have you just derived the core of?
"""),

    answer("""**A3 — your answer:**

*(a)* …

*(b)* …
""", """**A3 — solution.**

*(a)* $J'(s) = p\\,\\sigma(-s) - q\\,\\sigma(s)$ (using $\\frac{d}{ds}\\log\\sigma(s)
= \\sigma(-s)$ and $\\frac{d}{ds}\\log\\sigma(-s) = -\\sigma(s)$). Setting to zero:
$p\\,(1 - \\sigma(s)) = q\\,\\sigma(s) \\Rightarrow \\sigma(s^*) = p/(p+q)$, hence
$s^* = \\log(p/q)$. $J'' < 0$ everywhere, so it is the maximum. $\\square$

*(b)* $s^* = \\log \\frac{p}{k\\,\\#(u)\\#(v)/|\\mathcal{D}|}
= \\log \\frac{\\#(u,v)\\,|\\mathcal{D}|}{\\#(u)\\,\\#(v)} - \\log k
= \\mathrm{PMI}(u,v) - \\log k$ — the fixed point in the Week 3
Levy–Goldberg/NetMF proposition: SGNS implicitly factorizes the shifted PMI matrix.
You verify this numerically in B2.
"""),

    md("""## A4 · The DistMult column, proved (6 pts)

Prove the DistMult column of the Week 4 cheat sheet. Throughout,
$f(h,r,t) = \\sum_i h_i r_i t_i$ with all vectors real.

**(a)** (1 pt) **Symmetry** (the ✓): show $f(h,r,t) = f(t,r,h)$ for all parameters.

**(b)** (2 pts) **Antisymmetry** (the ✗): conclude that no DistMult parameterization
can score $(h,r,t)$ strictly above $(t,r,h)$ — state precisely what "cannot represent
antisymmetry" means and prove it.

**(c)** (2 pts) **Inversion** (the ✗): suppose relations $r_1, r_2$ satisfy
$f(a, r_2, b) = f(b, r_1, a)$ for **all** entity vectors $a, b$. Show this forces
$\\mathbf{r}_2 = \\mathbf{r}_1$, and explain why that degenerates inversion into
symmetry rather than representing it.

**(d)** (1 pt) **1-to-N** (the ✓): construct explicit $d = 2$ vectors — a head $h$,
one relation $r$, and two *distinct* tails $t_1 \\neq t_2$ — such that
$f(h,r,t_1) = f(h,r,t_2) > 0$. One line of arithmetic suffices.
"""),

    answer("""**A4 — your answer:**

*(a)* …

*(b)* …

*(c)* …

*(d)* …
""", """**A4 — solution.**

*(a)* $\\sum_i h_i r_i t_i = \\sum_i t_i r_i h_i$: multiplication commutes. $\\square$

*(b)* "Represent antisymmetry" means: there exist parameters and entities with
$f(h,r,t)$ high while $f(t,r,h)$ is low (any strict separation). By (a) the two values
are *equal* for every parameter choice — no separation of any size exists. $\\square$

*(c)* $f(a,r_2,b) = f(b,r_1,a) = f(a,r_1,b)$ for all $a,b$ (using (a) on the right
side). So $\\sum_i a_i (r_{2,i} - r_{1,i}) b_i = 0$ for all $a, b$; choosing
$a = b = e_i$ (the $i$-th basis vector) gives $r_{2,i} = r_{1,i}$ for every $i$.
Inversion collapses to "$r_2$ *is* $r_1$", which by (a) is symmetric — a genuine
directed pair like located_in / contains cannot be expressed, only merged. $\\square$

*(d)* $h = (1, 1)$, $r = (1, 1)$, $t_1 = (2, 0)$, $t_2 = (0, 2)$:
$f = 2$ for both, and $t_1 \\neq t_2$. Distinct tails, equal maximal enthusiasm —
the similarity-scorer's escape from the crush that @A4-adjacent functional models
(TransE, RotatE) provably cannot make. $\\square$
"""),

    md("""---
# Part B · Implementation (60 points) — autograded

The asserts below are the visible majority of the grade; hidden variants of the same
checks (different graphs, different toy vectors) run server-side. Write clean,
deterministic code — every function takes an explicit seed where randomness exists.
"""),

    md("""## B1 · Spectra as fingerprints — and their limit (20 pts)

Implement the spectral fingerprint and use it to *find* the cospectral pair from A1(c)
among a set of candidate graphs, then confirm the pair is non-isomorphic by a
structural invariant the spectrum misses.
"""),

    todo("""def spectrum(G: nx.Graph) -> tuple:
    \"\"\"The adjacency spectrum as a tuple of floats rounded to 6 decimals,
    sorted descending. (Rounding makes spectra comparable as dict keys.)\"\"\"
    ### BEGIN SOLUTION
    A = nx.to_numpy_array(G, weight=None)
    ev = np.linalg.eigvalsh(A)
    return tuple(round(float(x), 6) for x in sorted(ev, reverse=True))
    ### END SOLUTION


def find_cospectral_pair(graphs: dict) -> tuple:
    \"\"\"graphs: {name: nx.Graph}. Return the (name_a, name_b) pair (sorted) with
    identical spectra but non-isomorphic graphs (different degree multisets or
    different component counts count as proof here).\"\"\"
    ### BEGIN SOLUTION
    by_spec = {}
    for name, G in graphs.items():
        by_spec.setdefault(spectrum(G), []).append(name)
    for names in by_spec.values():
        if len(names) >= 2:
            a, b = sorted(names)[:2]
            return (a, b)
    return None
    ### END SOLUTION


CANDIDATES = {
    "path5": nx.path_graph(5),
    "star": nx.star_graph(4),                       # K_{1,4}
    "c4_plus_iso": nx.disjoint_union(nx.cycle_graph(4), nx.empty_graph(1)),
    "cycle5": nx.cycle_graph(5),
    "complete4_plus_iso": nx.disjoint_union(nx.complete_graph(4), nx.empty_graph(1)),
}

s_star = spectrum(CANDIDATES["star"])
assert len(s_star) == 5 and abs(s_star[0] - 2.0) < 1e-9, (
    f"spectrum of K_1,4 should start at 2.0 (largest eigenvalue = sqrt(4)); got {s_star[:2]} — "
    f"use eigvalsh on the unweighted 0/1 adjacency and sort descending"
)
pair = find_cospectral_pair(CANDIDATES)
assert pair == ("c4_plus_iso", "star"), (
    f"the cospectral pair is A1(c)'s: K_1,4 and C4 ∪ K1 — got {pair}. If you found "
    f"nothing, check the rounding; if a different pair, your spectrum is off."
)
Ga, Gb = CANDIDATES[pair[0]], CANDIDATES[pair[1]]
assert sorted(d for _, d in Ga.degree) != sorted(d for _, d in Gb.degree), (
    "confirm non-isomorphism structurally: the two graphs have different degree multisets"
)
print(f"B1 ✓ — {pair[0]} and {pair[1]} share spectrum {s_star} yet are not isomorphic")
print("the spectrum, like WL, is a sound but incomplete fingerprint (Week 9 will echo this)")""",
         stub="""def spectrum(G: nx.Graph) -> tuple:
    \"\"\"The adjacency spectrum as a tuple of floats rounded to 6 decimals,
    sorted descending. (Rounding makes spectra comparable as dict keys.)\"\"\"
    # TODO: nx.to_numpy_array(..., weight=None) → np.linalg.eigvalsh → round + sort.
    raise NotImplementedError


def find_cospectral_pair(graphs: dict) -> tuple:
    \"\"\"graphs: {name: nx.Graph}. Return the (name_a, name_b) pair (sorted) with
    identical spectra but non-isomorphic graphs (different degree multisets or
    different component counts count as proof here).\"\"\"
    # TODO: group names by spectrum; return the sorted pair sharing one.
    raise NotImplementedError


CANDIDATES = {
    "path5": nx.path_graph(5),
    "star": nx.star_graph(4),                       # K_{1,4}
    "c4_plus_iso": nx.disjoint_union(nx.cycle_graph(4), nx.empty_graph(1)),
    "cycle5": nx.cycle_graph(5),
    "complete4_plus_iso": nx.disjoint_union(nx.complete_graph(4), nx.empty_graph(1)),
}

s_star = spectrum(CANDIDATES["star"])
assert len(s_star) == 5 and abs(s_star[0] - 2.0) < 1e-9, (
    f"spectrum of K_1,4 should start at 2.0 (largest eigenvalue = sqrt(4)); got {s_star[:2]} — "
    f"use eigvalsh on the unweighted 0/1 adjacency and sort descending"
)
pair = find_cospectral_pair(CANDIDATES)
assert pair == ("c4_plus_iso", "star"), (
    f"the cospectral pair is A1(c)'s: K_1,4 and C4 ∪ K1 — got {pair}. If you found "
    f"nothing, check the rounding; if a different pair, your spectrum is off."
)
Ga, Gb = CANDIDATES[pair[0]], CANDIDATES[pair[1]]
assert sorted(d for _, d in Ga.degree) != sorted(d for _, d in Gb.degree), (
    "confirm non-isomorphism structurally: the two graphs have different degree multisets"
)
print(f"B1 ✓ — {pair[0]} and {pair[1]} share spectrum {s_star} yet are not isomorphic")
print("the spectrum, like WL, is a sound but incomplete fingerprint (Week 9 will echo this)")"""),

    md("""## B2 · Verify A3 empirically: SGNS finds the shifted PMI (20 pts)

Train a tiny SGNS on the cast graph (code adapted from Lab 3, provided) and test the
theory: learned scores $\\mathbf{z}_u^\\top \\mathbf{z}'_v$ should correlate strongly
with $\\mathrm{PMI}(u,v) - \\log k$ across pairs. Your part: compute the empirical
shifted-PMI matrix from the same walk corpus the trainer used.
"""),

    todo("""CAST = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (2, 5), (4, 5)])


def walks_and_pairs(G, num_walks=200, length=10, window=2, seed=3):
    rng = random.Random(seed)
    pairs = []
    for _ in range(num_walks):
        for s in sorted(G):
            w = [s]
            for _ in range(length - 1):
                w.append(rng.choice(list(G[w[-1]])))
            for i, c in enumerate(w):
                for j in range(max(0, i - window), min(len(w), i + window + 1)):
                    if j != i:
                        pairs.append((c, w[j]))
    return pairs


def shifted_pmi(pairs, n, k):
    \"\"\"Return the (n, n) matrix  M[u,v] = log( #(u,v) * |D| / (#u * #v) ) - log k,
    where #(u,v) counts pair occurrences, #u counts u as a CENTER, and #v counts v
    as a CONTEXT. Cells with #(u,v) = 0 get -inf (np.NINF is fine).\"\"\"
    ### BEGIN SOLUTION
    D = len(pairs)
    cnt = Counter(pairs)
    cu = Counter(u for u, _ in pairs)
    cv = Counter(v for _, v in pairs)
    M = np.full((n, n), -np.inf)
    for (u, v), c in cnt.items():
        M[u, v] = math.log(c * D / (cu[u] * cv[v])) - math.log(k)
    return M
    ### END SOLUTION


K_NEG = 5
pairs = walks_and_pairs(CAST)
M = shifted_pmi(pairs, 6, K_NEG)
assert M.shape == (6, 6) and np.isneginf(M[0, 0]) is not None, "shape (6,6); unseen pairs -inf"
assert np.isfinite(M[0, 1]) and np.isfinite(M[1, 0]), "adjacent pairs certainly co-occur"
row_check = math.log(Counter(pairs)[(0, 1)] * len(pairs)
                     / (Counter(u for u, _ in pairs)[0] * Counter(v for _, v in pairs)[1])) - math.log(K_NEG)
assert abs(M[0, 1] - row_check) < 1e-12, "M[0,1] must match the formula exactly — check numerator/denominator roles"

# provided trainer (two tables, k=5 negatives drawn from the CONTEXT unigram
# distribution — exactly the noise A3(b) assumes; with uniform noise the fixed
# point shifts and the correlation tops out around 0.88. Precision matters.)
def train_tiny_sgns(pairs, n, d=8, k=K_NEG, epochs=40, lr=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    cen, ctx = torch.nn.Embedding(n, d), torch.nn.Embedding(n, d)
    torch.nn.init.normal_(cen.weight, std=0.1, generator=g)
    torch.nn.init.normal_(ctx.weight, std=0.1, generator=g)
    opt = torch.optim.Adam(list(cen.parameters()) + list(ctx.parameters()), lr=lr)
    cv_counts = Counter(v for _, v in pairs)
    noise = torch.tensor([cv_counts[i] for i in range(n)], dtype=torch.float)
    noise = noise / noise.sum()
    P = torch.tensor(pairs)
    for _ in range(epochs):
        perm = torch.randperm(len(P), generator=g)
        for i in range(0, len(P), 512):
            idx = perm[i:i + 512]
            u, v = P[idx, 0], P[idx, 1]
            negs = torch.multinomial(noise, len(idx) * k, replacement=True,
                                     generator=g).view(len(idx), k)
            zu, zv, zn = cen(u), ctx(v), ctx(negs)
            loss = -(F.logsigmoid((zu * zv).sum(1))
                     + F.logsigmoid(-(zn * zu.unsqueeze(1)).sum(2)).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return cen.weight.detach() @ ctx.weight.detach().T


S = train_tiny_sgns(pairs, 6).numpy()
mask = np.isfinite(M)
corr = np.corrcoef(S[mask], M[mask])[0, 1]
print(f"correlation between learned scores and shifted PMI: {corr:.3f}")
assert corr > 0.9, (
    f"correlation {corr:.3f} — with d=8 on a 6-node graph, SGNS should track the "
    f"shifted PMI closely (>0.9). Check #u = center counts and #v = CONTEXT counts, "
    f"not both centers."
)
print("B2 ✓ — A3's fixed point is real: the 'neural' method is a matrix factorization in disguise")""",
         stub="""CAST = nx.Graph([(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (2, 5), (4, 5)])


def walks_and_pairs(G, num_walks=200, length=10, window=2, seed=3):
    rng = random.Random(seed)
    pairs = []
    for _ in range(num_walks):
        for s in sorted(G):
            w = [s]
            for _ in range(length - 1):
                w.append(rng.choice(list(G[w[-1]])))
            for i, c in enumerate(w):
                for j in range(max(0, i - window), min(len(w), i + window + 1)):
                    if j != i:
                        pairs.append((c, w[j]))
    return pairs


def shifted_pmi(pairs, n, k):
    \"\"\"Return the (n, n) matrix  M[u,v] = log( #(u,v) * |D| / (#u * #v) ) - log k,
    where #(u,v) counts pair occurrences, #u counts u as a CENTER, and #v counts v
    as a CONTEXT. Cells with #(u,v) = 0 get -inf (np.NINF is fine).\"\"\"
    # TODO: three Counters (pairs, centers, contexts) and the formula. ~9 lines.
    raise NotImplementedError


K_NEG = 5
pairs = walks_and_pairs(CAST)
M = shifted_pmi(pairs, 6, K_NEG)
assert M.shape == (6, 6) and np.isneginf(M[0, 0]) is not None, "shape (6,6); unseen pairs -inf"
assert np.isfinite(M[0, 1]) and np.isfinite(M[1, 0]), "adjacent pairs certainly co-occur"
row_check = math.log(Counter(pairs)[(0, 1)] * len(pairs)
                     / (Counter(u for u, _ in pairs)[0] * Counter(v for _, v in pairs)[1])) - math.log(K_NEG)
assert abs(M[0, 1] - row_check) < 1e-12, "M[0,1] must match the formula exactly — check numerator/denominator roles"

# provided trainer (two tables, k=5 negatives drawn from the CONTEXT unigram
# distribution — exactly the noise A3(b) assumes; with uniform noise the fixed
# point shifts and the correlation tops out around 0.88. Precision matters.)
def train_tiny_sgns(pairs, n, d=8, k=K_NEG, epochs=40, lr=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    cen, ctx = torch.nn.Embedding(n, d), torch.nn.Embedding(n, d)
    torch.nn.init.normal_(cen.weight, std=0.1, generator=g)
    torch.nn.init.normal_(ctx.weight, std=0.1, generator=g)
    opt = torch.optim.Adam(list(cen.parameters()) + list(ctx.parameters()), lr=lr)
    cv_counts = Counter(v for _, v in pairs)
    noise = torch.tensor([cv_counts[i] for i in range(n)], dtype=torch.float)
    noise = noise / noise.sum()
    P = torch.tensor(pairs)
    for _ in range(epochs):
        perm = torch.randperm(len(P), generator=g)
        for i in range(0, len(P), 512):
            idx = perm[i:i + 512]
            u, v = P[idx, 0], P[idx, 1]
            negs = torch.multinomial(noise, len(idx) * k, replacement=True,
                                     generator=g).view(len(idx), k)
            zu, zv, zn = cen(u), ctx(v), ctx(negs)
            loss = -(F.logsigmoid((zu * zv).sum(1))
                     + F.logsigmoid(-(zn * zu.unsqueeze(1)).sum(2)).sum(1)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return cen.weight.detach() @ ctx.weight.detach().T


S = train_tiny_sgns(pairs, 6).numpy()
mask = np.isfinite(M)
corr = np.corrcoef(S[mask], M[mask])[0, 1]
print(f"correlation between learned scores and shifted PMI: {corr:.3f}")
assert corr > 0.9, (
    f"correlation {corr:.3f} — with d=8 on a 6-node graph, SGNS should track the "
    f"shifted PMI closely (>0.9). Check #u = center counts and #v = CONTEXT counts, "
    f"not both centers."
)
print("B2 ✓ — A3's fixed point is real: the 'neural' method is a matrix factorization in disguise")"""),

    md("""## B3 · A4's consequence, measured (20 pts)

Implement DistMult, train it on the Week-4 fragment, and demonstrate the theorem with
numbers: the model assigns *identical* scores to every fact and its reversal — so it
cannot rank *(Kazan, capital_of, Tatarstan)* above *(Tatarstan, capital_of, Kazan)*,
no matter how long you train.
"""),

    todo("""ENT = ["IU", "KFU", "Innopolis", "Kazan", "Tatarstan", "Russia", "Volga",
       "university", "city", "region"]
REL = ["located_in", "instance_of", "capital_of", "part_of", "flows_through"]
TRIPLES = [(0, 1, 7), (0, 0, 2), (2, 0, 4), (3, 2, 4), (3, 0, 4), (4, 3, 5),
           (1, 1, 7), (1, 0, 3), (2, 1, 8), (3, 1, 8), (4, 1, 9), (6, 4, 4)]


def distmult_score(E: torch.Tensor, R: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
    \"\"\"E: (n_ent, d) entity table. R: (n_rel, d). triples: (B, 3) long tensor of
    (h, r, t) rows. Return the (B,) DistMult scores Σ_i h_i r_i t_i.\"\"\"
    ### BEGIN SOLUTION
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    return (E[h] * R[r] * E[t]).sum(dim=1)
    ### END SOLUTION


# hand-check: the lecture's toy — h=(2,1), r=(1,3), t=(1,2) → 8, both directions
E_toy = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
R_toy = torch.tensor([[1.0, 3.0]])
fwd = distmult_score(E_toy, R_toy, torch.tensor([[0, 0, 1]]))
bwd = distmult_score(E_toy, R_toy, torch.tensor([[1, 0, 0]]))
assert abs(fwd.item() - 8.0) < 1e-6, f"lecture toy scores 8, got {fwd.item()}"
assert abs(fwd.item() - bwd.item()) < 1e-6, "A4(a) in tensor form: swap must not change the score"

# provided: train with corruption + logistic loss
g = torch.Generator().manual_seed(1)
E_w = torch.nn.Parameter(0.2 * torch.randn(10, 8, generator=g))
R_w = torch.nn.Parameter(0.2 * torch.randn(5, 8, generator=g))
opt = torch.optim.Adam([E_w, R_w], lr=0.05)
T = torch.tensor(TRIPLES)
for step in range(1500):
    corrupt = T.clone()
    side = torch.randint(0, 2, (len(T),), generator=g)
    rand_e = torch.randint(0, 10, (len(T),), generator=g)
    corrupt[side == 0, 0] = rand_e[side == 0]
    corrupt[side == 1, 2] = rand_e[side == 1]
    pos = distmult_score(E_w, R_w, T)
    neg = distmult_score(E_w, R_w, corrupt)
    loss = -(F.logsigmoid(pos).mean() + F.logsigmoid(-neg).mean())
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    T_rev = T[:, [2, 1, 0]]
    gap = (distmult_score(E_w, R_w, T) - distmult_score(E_w, R_w, T_rev)).abs().max().item()
    pos_mean = distmult_score(E_w, R_w, T).mean().item()
print(f"after training: mean true-fact score {pos_mean:.2f}, max |forward − reversed| = {gap:.2e}")
assert pos_mean > 0.5, "training should make true facts score clearly positive"
assert gap < 1e-5, (
    "A4's theorem, violated?! forward and reversed scores must be numerically identical "
    "— if not, your score is not symmetric in (h, t); re-check the formula"
)
print("B3 ✓ — 1,500 optimizer steps, and the model still cannot tell capital_of from its reverse.")
print("Theory did not just predict the failure — it guaranteed it. Choose models by pattern audit.")""",
         stub="""ENT = ["IU", "KFU", "Innopolis", "Kazan", "Tatarstan", "Russia", "Volga",
       "university", "city", "region"]
REL = ["located_in", "instance_of", "capital_of", "part_of", "flows_through"]
TRIPLES = [(0, 1, 7), (0, 0, 2), (2, 0, 4), (3, 2, 4), (3, 0, 4), (4, 3, 5),
           (1, 1, 7), (1, 0, 3), (2, 1, 8), (3, 1, 8), (4, 1, 9), (6, 4, 4)]


def distmult_score(E: torch.Tensor, R: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
    \"\"\"E: (n_ent, d) entity table. R: (n_rel, d). triples: (B, 3) long tensor of
    (h, r, t) rows. Return the (B,) DistMult scores Σ_i h_i r_i t_i.\"\"\"
    # TODO: index the tables, multiply elementwise, sum over d. One line.
    raise NotImplementedError


# hand-check: the lecture's toy — h=(2,1), r=(1,3), t=(1,2) → 8, both directions
E_toy = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
R_toy = torch.tensor([[1.0, 3.0]])
fwd = distmult_score(E_toy, R_toy, torch.tensor([[0, 0, 1]]))
bwd = distmult_score(E_toy, R_toy, torch.tensor([[1, 0, 0]]))
assert abs(fwd.item() - 8.0) < 1e-6, f"lecture toy scores 8, got {fwd.item()}"
assert abs(fwd.item() - bwd.item()) < 1e-6, "A4(a) in tensor form: swap must not change the score"

# provided: train with corruption + logistic loss
g = torch.Generator().manual_seed(1)
E_w = torch.nn.Parameter(0.2 * torch.randn(10, 8, generator=g))
R_w = torch.nn.Parameter(0.2 * torch.randn(5, 8, generator=g))
opt = torch.optim.Adam([E_w, R_w], lr=0.05)
T = torch.tensor(TRIPLES)
for step in range(1500):
    corrupt = T.clone()
    side = torch.randint(0, 2, (len(T),), generator=g)
    rand_e = torch.randint(0, 10, (len(T),), generator=g)
    corrupt[side == 0, 0] = rand_e[side == 0]
    corrupt[side == 1, 2] = rand_e[side == 1]
    pos = distmult_score(E_w, R_w, T)
    neg = distmult_score(E_w, R_w, corrupt)
    loss = -(F.logsigmoid(pos).mean() + F.logsigmoid(-neg).mean())
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    T_rev = T[:, [2, 1, 0]]
    gap = (distmult_score(E_w, R_w, T) - distmult_score(E_w, R_w, T_rev)).abs().max().item()
    pos_mean = distmult_score(E_w, R_w, T).mean().item()
print(f"after training: mean true-fact score {pos_mean:.2f}, max |forward − reversed| = {gap:.2e}")
assert pos_mean > 0.5, "training should make true facts score clearly positive"
assert gap < 1e-5, (
    "A4's theorem, violated?! forward and reversed scores must be numerically identical "
    "— if not, your score is not symmetric in (h, t); re-check the formula"
)
print("B3 ✓ — 1,500 optimizer steps, and the model still cannot tell capital_of from its reverse.")
print("Theory did not just predict the failure — it guaranteed it. Choose models by pattern audit.")"""),

    md("""---
# Part C · Open investigation (10 points)

Choose **one** hyperparameter from Labs 2–4 — for example: DeepWalk's window $w$, the
embedding dimension $d$, TransE's margin $\\gamma$, the self-adversarial temperature
$\\alpha$, or Louvain's resolution. Then:

1. **Hypothesis** (2 pts): one sentence predicting the effect and *why* (mechanism,
   not vibes).
2. **Experiment** (4 pts): at most three configurations, seeds fixed, one metric,
   reusing lab code. Put the runs in the code cell below.
3. **Conclusion** (4 pts): ≤ 10 lines — what happened, whether the hypothesis
   survived, and one thing that surprised you. A small table or plot is welcome.

Honest null results score full points; cherry-picking scores zero.
"""),

    code("""# C — your experiment here (keep it under ~2 minutes of compute)
"""),

    answer("""**C — hypothesis, results, conclusion:**

*Hypothesis:* …

*Results:* …

*Conclusion:* …
""", """**C — grading notes (for TAs).**

Full credit requires: a mechanistic hypothesis (e.g. "larger $w$ blurs communities
into one another because distant nodes enter the context multiset, so Cora accuracy
should peak at moderate $w$"), a controlled comparison (same seeds, one variable), and
a conclusion that engages with the actual numbers. Typical strong choices:
$w \\in \\{1, 5, 10\\}$ on Cora (accuracy usually peaks mid-range); $\\gamma \\in
\\{0.5, 1, 2\\}$ for TransE (too-large margins slow convergence and distort geometry);
$d \\in \\{4, 16, 64\\}$ (diminishing returns + Week 3 Q4(b) overfitting discussion).
Null results reported honestly get full credit; sweeps without a hypothesis cap at
6/10.
"""),

    md("""---
## Submission

One executed notebook on Moodle by **Thursday, October 1, 23:59**: Part A answers in
their markdown cells, all three B autograders ✓, Part C's experiment and write-up.
Run *Runtime → Restart and run all* before submitting.

**Honor code / AI disclosure** — state below any AI tools used and for what; you must
be able to explain any line and any proof step on request:

*(your disclosure here)*
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
            nb.cells.append(nbf.v4.new_markdown_cell(extra if not student else text))
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
        assert "solution." not in cell.source.lower()[:60] or cell.cell_type == "code", "answer leak"
    STUDENT_OUT.parent.mkdir(parents=True, exist_ok=True)
    SOLUTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(student, STUDENT_OUT)
    nbf.write(solution, SOLUTION_OUT)
    print("wrote", STUDENT_OUT)
    print("wrote", SOLUTION_OUT)


if __name__ == "__main__":
    main()
