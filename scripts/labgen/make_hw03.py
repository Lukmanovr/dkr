#!/usr/bin/env python
"""Generate Homework 3 (Weeks 9-12: expressiveness, typed message passing,
scaling audits, link prediction discipline) — student + solution notebooks.
Part B is fully deterministic.

    python scripts/labgen/make_hw03.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "homeworks" / "hw03_frontiers.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "homeworks" / "hw03_frontiers.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(solution, stub): return ("code", solution, stub)
def answer(question, solution): return ("markdown-answer", question, solution)


CELLS = [
    md("""# Homework 3 · Frontiers — ceilings, types, scale, and honest edges

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/homeworks/hw03_frontiers.ipynb)

**Covers Weeks 9–12 · out Nov 12 · due Thursday, Nov 26, 23:59 (Moodle) · ≈ 5–10 hours**

Three parts. **A — Derivations** (30%): four proofs in markdown cells.
**B — Implementation** (60%, autograded, fully deterministic): a capacity
auditor, a split validator that must catch four planted leaks, and the ranking
metrics module. **C — Investigation** (10%): the negatives experiment.

Rules: individual work; AI assistants allowed *with disclosure* per the honor
code; you must be able to explain any line and any proof step on request.
Rubric, late policy, FAQ: the [HW3 page](https://lukmanovr.github.io/dkr/homeworks/hw3.html).
"""),

    md("""This closing assignment is about ceilings and honesty: what message passing can
never distinguish, what typed and scaled variants actually cost, and what a link
prediction score is worth once the evaluation stops leaking. You will prove the
Weisfeiler–Leman ceiling on a concrete pair of graphs — the bound that makes
expressiveness a design constraint rather than a benchmark race
([Xu et al., 2019](https://arxiv.org/abs/1810.00826)) — audit parameter and memory
budgets the way Weeks 10–11 audited them, and build the three tools (capacity
auditor, split validator, ranking metrics) that separate defensible edge-prediction
numbers from the inflated kind catalogued by
[Sun et al., 2020](https://arxiv.org/abs/1911.03903). Everything in Part B is
deterministic on purpose: the subject is engineering discipline, not the seed
lottery.
"""),

    md("""## 0 · Setup"""),

    code("""import os, sys, math
SMOKE = os.environ.get("SMOKE", "") == "1"
import numpy as np
import networkx as nx
print("numpy", np.__version__, "· networkx", nx.__version__, "— Part B is torch-free by design")"""),

    md("""## Part A · Derivations (30 pts)

### A1 (8 pts) · Two graphs no GNN can separate

$G_1 = K_{3,3}$ (complete bipartite, 6 nodes) and $G_2$ = the triangular
prism (two triangles joined by a perfect matching). **(a)** Verify both are
3-regular and non-isomorphic (one sentence each). **(b)** Prove no
message-passing GNN with uniform features distinguishes them (Week 9's
theorem — state the chain precisely). **(c)** Give the cheapest structural
feature that separates them, with its value on each graph.
"""),

    answer("""*(your answer for A1 here)*""",
           """**A1 solution.** **(a)** Every $K_{3,3}$ node has degree 3 (all of the
other side); every prism node has degree 3 (two triangle mates + its match).
Non-isomorphic: the prism contains triangles, $K_{3,3}$ is bipartite and has
none. **(b)** Both are 3-regular with uniform features, so 1-WL colors every
node identically in both graphs at every round (round 1: everyone sees
{c,c,c}; induction holds) — identical histograms forever; by the ceiling
theorem (trees determine embeddings; MP-GNN ≤ 1-WL), every message-passing
GNN yields identical node-embedding multisets and identical pooled readouts.
$\\square$ **(c)** Per-node triangle count: 1 on every prism node
($\\mathrm{diag}(A^3)/2 = 1$), 0 on every $K_{3,3}$ node — one scalar
feature, and round 0 already separates. (Odd-cycle/bipartiteness checks work
too.)""",),

    md("""### A2 (8 pts) · The typed parameter audit

A production schema has $R = 58$ relations, planned hidden size $d = 64$,
and a skewed relation histogram: the busiest relation has 1.1M triples, the
median has 4,100, the bottom ten average 240. **(a)** Compute the per-layer
parameter count for full per-relation R-GCN weights (both directions +
self-loop). **(b)** Compute it for basis decomposition with $B = 16$.
**(c)** For the bottom-ten relations, compare parameters-per-training-triple
under both schemes and make the recommendation, with the one caveat the
basis scheme carries.
"""),

    answer("""*(your answer for A2 here)*""",
           """**A2 solution.** **(a)** $(2 \\cdot 58 + 1) \\cdot 64^2 = 117 \\cdot
4096 = 479{,}232$. **(b)** $16 \\cdot 4096 + 2 \\cdot 58 \\cdot 16 + 4096 =
65{,}536 + 1{,}856 + 4{,}096 = 71{,}488$ — a 6.7× reduction. **(c)** Full:
each direction of a bottom relation owns $4{,}096$ parameters for ~240
triples — 17 parameters per example, unspecified-by-data territory. Basis:
those relations own only 16 mixing coefficients each (~0.07 per example);
capacity lives in shared bases fed by the busy relations. Recommendation:
basis decomposition, clearly. Caveat: sharing forces rare relations to
express themselves through directions shaped by common ones — if a rare
relation is semantically *unlike* everything else, the basis prior actively
hurts it, and a block/full hybrid for outlier relations is the practical
compromise.""",),

    md("""### A3 (7 pts) · SGC in three layers, and its failure mode

**(a)** Collapse the 3-layer GCN
$\\hat A\\,\\sigma(\\hat A\\,\\sigma(\\hat A X W_0) W_1) W_2$ into SGC form,
stating the exact substitution. **(b)** State the compute cost of the
precompute in $N, E, d, K$. **(c)** Describe one concrete task where the
deleted nonlinearities were load-bearing, and the observable symptom that
would tell you so from a bake-off table.
"""),

    answer("""*(your answer for A3 here)*""",
           """**A3 solution.** **(a)** Set every $\\sigma = \\mathrm{id}$; matrix
association gives $\\hat A^3 X (W_0 W_1 W_2) = (\\hat A^3 X) W$ — one
precomputed diffusion, one learned matrix. **(b)** $K$ sparse products,
each $O(E \\cdot d)$: total $O(K E d)$, once, streaming, CPU-friendly.
**(c)** Any task needing nonlinear interaction of multi-hop information —
e.g. heterophilous node classification ("my label differs from my
neighbors'"), or detecting *combinations* (label = A-neighbor AND
B-neighbor present at 2 hops). Symptom in a bake-off: SGC lands far below
GCN at equal $K$ (rather than the usual near-tie), and the gap *grows* with
$K$ — linear smoothing is actively destroying the signal the nonlinear
model preserves.""",),

    md("""### A4 (7 pts) · The symmetric decoder, formally

**(a)** Prove: for the task "given that exactly one of $(a \\to b)$,
$(b \\to a)$ exists, predict which," any scorer of the form
$s(a, b) = f(z_a^\\top z_b)$ with $f$ monotone achieves exactly AUC 0.5.
**(b)** Name two decoders from this course that break the symmetry and say
*mechanically* where each breaks it. **(c)** Explain in one sentence why the
same issue does NOT arise for SEAL.
"""),

    answer("""*(your answer for A4 here)*""",
           """**A4 solution.** **(a)** $z_a^\\top z_b = z_b^\\top z_a$, so $s(a,b) =
s(b,a)$ for every pair: the two candidate directions always tie, every
comparison is decided by the tie-break coin, AUC $= 0.5$ exactly.
$\\square$ **(b)** DistMult-with-roles / any two-embedding scheme
($z^{src}_a{}^\\top \\mathrm{diag}(r)\\, z^{tgt}_b$): symmetry breaks
because source and target use different embeddings; ComplEx: the Hermitian
product $\\mathrm{Re}(\\langle z_a, r, \\bar z_b\\rangle)$ is asymmetric in
$(a,b)$ through conjugation; (TransE's $-\\|z_a + r - z_b\\|$ also
qualifies — the translation direction distinguishes head from tail.)
**(c)** SEAL classifies the labeled subgraph, and the labels $(d_a, d_b)$
are ordered — swapping the roles of $a$ and $b$ transposes every label, so
the classifier sees different inputs for the two directions.""",),

    md("""## Part B · Implementation (60 pts, autograded, deterministic)

### B1 (20 pts) · The capacity auditor

This exercise turns Week 11's capacity-audit method into three reusable
functions.

**What you will implement**, in the exercise cell:

1. `memory_bill_mb(N, E, d, layers)` — the approximate full-batch float32
   memory bill in MB: features + retained activations + per-edge messages.
2. `fanout_bound(f, L)` — the worst-case number of nodes a sampled batch
   touches per target: $1 + f + f^2 + \\cdots + f^L$.
3. `verdict(N, E, d, layers, gpu_mb)` — return the string `"fits"` if the
   bill is under 80% of the GPU budget, otherwise `"OOM"`.

**How you will know it worked:** the asserts run your audit on the lecture's
ogbn-arxiv and ogbn-products numbers and require the same bills and the same
fits/OOM verdicts the lecture reached. When they pass, the cell prints
"B1 ✓".
"""),

    md("""#### The spec for B1

Week 11's audit, as a reusable function. The full-batch bill has three float32
lines — features, retained activations, per-edge messages — and the sampled
alternative is priced by the geometric fanout bound that GraphSAGE's sampling
contract introduced ([Hamilton et al., 2017](https://arxiv.org/abs/1706.02216)).
The algorithm below specifies exactly what your three functions must compute;
follow it step by step.

**Algorithm 1 · The capacity audit**

**Input:** node count $N$; directed edge count $E$; hidden width $d$; layer count $L$; GPU budget $M$ (MB); fanout $f$ for the sampled alternative. **Output:** memory bill in MB; a fits/OOM verdict; worst-case nodes touched per target.

1. $\\text{bill} \\leftarrow \\bigl(N d \\,+\\, L\\, N d \\,+\\, L\\, E d\\bigr) \\times 4 \\ \\text{bytes} \\;/\\; 2^{20}$
2. **if** $\\text{bill} < 0.8\\, M$ **then** verdict $\\leftarrow$ "fits" **else** verdict $\\leftarrow$ "OOM"
3. $\\text{bound} \\leftarrow 1 + f + f^2 + \\cdots + f^L$
4. **return** bill, verdict, bound

Step 1 is the lecture's three-line bill — features, activations, messages — and the
edge-message term dominates because messages are per-edge, not per-node. Step 2's
20% headroom pays the bookkeeping tax of gradients and optimizer state. Step 3 is
the worst case a sampled batch can touch per target node. The asserts run your
audit on the lecture's ogbn-arxiv and ogbn-products numbers and expect the same
verdicts the lecture reached.
"""),

    todo("""def memory_bill_mb(N, E, d, layers):
    \"\"\"Approximate full-batch float32 residency in MB:
    features (N*d*4) + retained activations (layers * N*d*4)
    + edge messages (layers * E*d*4). Return a float (MB).\"\"\"
    ### BEGIN SOLUTION
    b = N * d * 4 + layers * N * d * 4 + layers * E * d * 4
    return b / 2**20
    ### END SOLUTION


def fanout_bound(f, L):
    \"\"\"Worst-case nodes touched per target: 1 + f + f^2 + ... + f^L.\"\"\"
    ### BEGIN SOLUTION
    return sum(f ** k for k in range(L + 1))
    ### END SOLUTION


def verdict(N, E, d, layers, gpu_mb):
    \"\"\"'fits' if the bill is under 80% of gpu_mb, else 'OOM'.\"\"\"
    ### BEGIN SOLUTION
    return "fits" if memory_bill_mb(N, E, d, layers) < 0.8 * gpu_mb else "OOM"
    ### END SOLUTION


# arxiv, the lecture's numbers
arxiv = memory_bill_mb(169_343, 2_332_486, 128, 2)
assert 2500 < arxiv < 2900, f"arxiv 2-layer bill should land ~2.7 GB (got {arxiv:.0f} MB)"
# products, the wall
prods = memory_bill_mb(2_449_029, 123_718_280 * 2, 128, 2)
assert prods > 200_000, "products should bill hundreds of GB"
assert verdict(169_343, 2_332_486, 128, 2, 8192) == "fits"
assert verdict(2_449_029, 123_718_280 * 2, 128, 2, 8192) == "OOM"
assert fanout_bound(10, 2) == 111 and fanout_bound(10, 3) == 1111
assert fanout_bound(25, 2) == 651, "the GraphSAGE-default bound"
print("B1 \\u2713 — the audit that replaces the OOM-and-guess loop")""",
         stub="""def memory_bill_mb(N, E, d, layers):
    \"\"\"Approximate full-batch float32 residency in MB:
    features (N*d*4) + retained activations (layers * N*d*4)
    + edge messages (layers * E*d*4). Return a float (MB).\"\"\"
    # TODO: one expression / 2**20.
    raise NotImplementedError


def fanout_bound(f, L):
    \"\"\"Worst-case nodes touched per target: 1 + f + f^2 + ... + f^L.\"\"\"
    # TODO: one line.
    raise NotImplementedError


def verdict(N, E, d, layers, gpu_mb):
    \"\"\"'fits' if the bill is under 80% of gpu_mb, else 'OOM'.\"\"\"
    # TODO: one line.
    raise NotImplementedError


# arxiv, the lecture's numbers
arxiv = memory_bill_mb(169_343, 2_332_486, 128, 2)
assert 2500 < arxiv < 2900, f"arxiv 2-layer bill should land ~2.7 GB (got {arxiv:.0f} MB)"
# products, the wall
prods = memory_bill_mb(2_449_029, 123_718_280 * 2, 128, 2)
assert prods > 200_000, "products should bill hundreds of GB"
assert verdict(169_343, 2_332_486, 128, 2, 8192) == "fits"
assert verdict(2_449_029, 123_718_280 * 2, 128, 2, 8192) == "OOM"
assert fanout_bound(10, 2) == 111 and fanout_bound(10, 3) == 1111
assert fanout_bound(25, 2) == 651, "the GraphSAGE-default bound"
print("B1 \\u2713 — the audit that replaces the OOM-and-guess loop")"""),

    md("""### B2 (20 pts) · The split validator that catches four leaks

**What you will implement:** `check_split(message, train_pos, test_pos,
test_neg)` in the exercise cell. Each argument is a set of
`frozenset({u, v})` edge pairs. The function must check for three kinds of
leak, in a fixed order, and return the FIRST violation it finds as a string —
`"test_in_message"`, `"overlap"`, or `"neg_is_positive"` — or `"ok"` if the
split is clean.

**How you will know it worked:** the asserts hand your validator one clean
split (which must return `"ok"`) and four sabotaged ones (each of which must
be named correctly, including a split where one bad negative hides among good
ones). When all five pass, the cell prints "B2 ✓".
"""),

    md("""#### The spec for B2

Week 12's four-bucket split gives every edge exactly one job, and every leak in
that lecture violated exactly one bucket's card. Modern benchmarks such as OGB
([Hu et al., 2020](https://arxiv.org/abs/2005.00687)) institutionalized this
discipline by shipping the splits themselves; here you write the auditor, with the
checks in a fixed order so a multi-fault split reports its *first* violation
deterministically. The algorithm below specifies exactly what your `check_split`
must do; follow it step by step.

**Algorithm 2 · The split audit**

**Input:** edge sets — message $M$, train positives $P_{tr}$, test positives $P_{te}$, test negatives $N_{te}$ (each a set of unordered pairs). **Output:** the first violation's name, or "ok".

1. **if** $P_{te} \\cap M \\ne \\emptyset$ **then return** "test_in_message"
2. **if** $P_{tr} \\cap P_{te} \\ne \\emptyset$ **then return** "overlap"
3. **if** $N_{te} \\cap (P_{tr} \\cup P_{te}) \\ne \\emptyset$ **then return** "neg_is_positive"
4. **return** "ok"

Check 1 is the leak Week 3 measured at +17 AUC — the test answer visible in the
message graph; check 2 is grading on the training set; check 3 poisons the answer
key itself. The asserts hand your auditor one clean split and four sabotaged ones:
it must pass the first and name each sabotage.
"""),

    todo("""def check_split(message, train_pos, test_pos, test_neg):
    \"\"\"Each argument: a set of frozenset({u, v}) pairs. Return the FIRST
    violation found, checked in this order:
      1. any test positive inside the message set  -> "test_in_message"
      2. any pair in two positive buckets          -> "overlap"
      3. any test negative that is a positive      -> "neg_is_positive"
      otherwise                                    -> "ok"  \"\"\"
    ### BEGIN SOLUTION
    if test_pos & message:
        return "test_in_message"
    if train_pos & test_pos:
        return "overlap"
    if test_neg & (train_pos | test_pos):
        return "neg_is_positive"
    return "ok"
    ### END SOLUTION


P = lambda *pairs: {frozenset(p) for p in pairs}
msg = P((0, 1), (1, 2), (2, 3))
trp = P((0, 1), (1, 2), (2, 3))
tep = P((0, 3), (1, 3))
ten = P((0, 2))
assert check_split(msg, trp, tep, ten) == "ok", "the clean split must pass"
assert check_split(msg | P((0, 3)), trp, tep, ten) == "test_in_message", (
    "sabotage 1: a test edge hiding in the message graph (Week 3's +17 AUC)"
)
assert check_split(msg, trp | P((0, 3)), tep, ten) == "overlap", (
    "sabotage 2: one pair in two buckets"
)
assert check_split(msg, trp, tep, P((1, 3))) == "neg_is_positive", (
    "sabotage 3: a 'negative' that is a test positive (poisoned answer key)"
)
assert check_split(msg, trp, tep, P((0, 2), (1, 3))) == "neg_is_positive", (
    "sabotage 4: one bad negative among good ones must still be caught"
)
print("B2 \\u2713 — twelve lines that would have caught the 0.363")""",
         stub="""def check_split(message, train_pos, test_pos, test_neg):
    \"\"\"Each argument: a set of frozenset({u, v}) pairs. Return the FIRST
    violation found, checked in this order:
      1. any test positive inside the message set  -> "test_in_message"
      2. any pair in two positive buckets          -> "overlap"
      3. any test negative that is a positive      -> "neg_is_positive"
      otherwise                                    -> "ok"  \"\"\"
    # TODO: three set intersections, in order.
    raise NotImplementedError


P = lambda *pairs: {frozenset(p) for p in pairs}
msg = P((0, 1), (1, 2), (2, 3))
trp = P((0, 1), (1, 2), (2, 3))
tep = P((0, 3), (1, 3))
ten = P((0, 2))
assert check_split(msg, trp, tep, ten) == "ok", "the clean split must pass"
assert check_split(msg | P((0, 3)), trp, tep, ten) == "test_in_message", (
    "sabotage 1: a test edge hiding in the message graph (Week 3's +17 AUC)"
)
assert check_split(msg, trp | P((0, 3)), tep, ten) == "overlap", (
    "sabotage 2: one pair in two buckets"
)
assert check_split(msg, trp, tep, P((1, 3))) == "neg_is_positive", (
    "sabotage 3: a 'negative' that is a test positive (poisoned answer key)"
)
assert check_split(msg, trp, tep, P((0, 2), (1, 3))) == "neg_is_positive", (
    "sabotage 4: one bad negative among good ones must still be caught"
)
print("B2 \\u2713 — twelve lines that would have caught the 0.363")"""),

    md("""### B3 (20 pts) · The metrics module — and the disagreement, reproduced

**What you will implement**, in the exercise cell:

1. `auc_pairwise(pos_scores, neg_scores)` — the probability that a random
   positive outscores a random negative, with ties counting 1/2.
2. `hits_at_k(pos_scores, neg_scores, k)` — the fraction of positives ranked
   within the top k of the MERGED list (rank = 1 + number of strictly higher
   scores in the merged list).
3. `mrr(pos_scores, neg_scores)` — the mean over positives of 1/rank, where
   rank is computed against the NEGATIVES only.

**How you will know it worked:** the asserts first unit-test each metric on
tiny hand-checkable score lists, then reproduce the lecture's model-A/model-B
disagreement: model A wins AUC while scoring zero Hits@10, and model B does
the reverse. When everything passes, the cell prints "B3 ✓".
"""),

    md("""#### The spec for B3

Ranking metrics in the shared-candidate-pool form used here. The protocol descends
from the raw/filtered ranking evaluation of
[Bordes et al., 2013](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html) —
Week 4's evaluation procedure — simplified: one shared pool of negatives, and no
filtering step because the pool is constructed clean. The algorithm below
specifies exactly what your three functions must compute; follow it step by
step.

**Algorithm 3 · Pool-ranking metrics**

**Input:** positive scores $p_1, \\ldots, p_m$; negative scores $n_1, \\ldots, n_q$; cutoff $K$. **Output:** AUC, Hits@K, MRR.

1. $\\text{AUC} \\leftarrow \\frac{1}{mq}\\sum_{i,j} \\bigl(\\,[p_i > n_j] + \\tfrac12\\,[p_i = n_j]\\,\\bigr)$
2. **for** each positive $p_i$: $\\text{rank}_i \\leftarrow 1 + \\#\\{s \\text{ in the merged list} : s > p_i\\}$
3. $\\text{Hits@}K \\leftarrow \\frac{1}{m}\\,\\#\\{i : \\text{rank}_i \\le K\\}$
4. **for** each positive $p_i$: $r_i \\leftarrow 1 + \\#\\{j : n_j > p_i\\}$ — negatives only
5. $\\text{MRR} \\leftarrow \\frac{1}{m}\\sum_i 1/r_i$
6. **return** AUC, Hits@K, MRR

Mind the two different ranks: Hits@K ranks each positive against the *merged* list
(step 2) while MRR ranks against the *negatives only* (step 4) — the convention
difference the docstrings spell out, and the source of most silent bugs here. The
asserts finish by reproducing the lecture's disagreement: a model can win AUC while
losing Hits@10, and which metric you report decides which model ships.
"""),

    todo("""def auc_pairwise(pos_scores, neg_scores):
    \"\"\"P(random positive outscores random negative), ties count 1/2.\"\"\"
    ### BEGIN SOLUTION
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos_scores) * len(neg_scores))
    ### END SOLUTION


def hits_at_k(pos_scores, neg_scores, k):
    \"\"\"Fraction of positives ranked within the top k of the MERGED list
    (rank = 1 + number of strictly higher scores in the merged list).\"\"\"
    ### BEGIN SOLUTION
    allsc = list(pos_scores) + list(neg_scores)
    hits = 0
    for p in pos_scores:
        rank = 1 + sum(1 for s in allsc if s > p)
        if rank <= k:
            hits += 1
    return hits / len(pos_scores)
    ### END SOLUTION


def mrr(pos_scores, neg_scores):
    \"\"\"Mean over positives of 1/rank against the NEGATIVES only
    (rank = 1 + number of negatives strictly above).\"\"\"
    ### BEGIN SOLUTION
    return sum(1.0 / (1 + sum(1 for n in neg_scores if n > p)) for p in pos_scores) / len(pos_scores)
    ### END SOLUTION


assert auc_pairwise([2, 3], [0, 1]) == 1.0 and auc_pairwise([1], [1]) == 0.5
assert hits_at_k([5, 0], [4, 3, 2, 1], 1) == 0.5, "one positive tops the merged list"
assert abs(mrr([5, 0], [4, 3, 2, 1]) - (1 + 1 / 5) / 2) < 1e-9

# the lecture's disagreement, in miniature: 10 true among 1000 candidates
rng = np.random.default_rng(0)
neg = list(rng.normal(0, 1, 990))
top_neg = sorted(neg, reverse=True)
# model A: every positive just below the top 30 negatives — great AUC, zero Hits@10
posA = [top_neg[29] - 1e-6] * 10
# model B: five positives at the very top, five at the bottom
posB = [top_neg[0] + 1] * 5 + [min(neg) - 1] * 5
aucA, aucB = auc_pairwise(posA, neg), auc_pairwise(posB, neg)
hA, hB = hits_at_k(posA, neg, 10), hits_at_k(posB, neg, 10)
print(f"model A: AUC {aucA:.3f} · Hits@10 {hA:.2f}   |   model B: AUC {aucB:.3f} · Hits@10 {hB:.2f}")
assert aucA > aucB and hB > hA, (
    "A must win AUC while B wins Hits@10 — the disagreement that decides products"
)
print("B3 \\u2713 — metrics implemented, and their disagreement is now yours to cite")""",
         stub="""def auc_pairwise(pos_scores, neg_scores):
    \"\"\"P(random positive outscores random negative), ties count 1/2.\"\"\"
    # TODO: double loop is fine at this size.
    raise NotImplementedError


def hits_at_k(pos_scores, neg_scores, k):
    \"\"\"Fraction of positives ranked within the top k of the MERGED list
    (rank = 1 + number of strictly higher scores in the merged list).\"\"\"
    # TODO: ~6 lines.
    raise NotImplementedError


def mrr(pos_scores, neg_scores):
    \"\"\"Mean over positives of 1/rank against the NEGATIVES only
    (rank = 1 + number of negatives strictly above).\"\"\"
    # TODO: ~3 lines.
    raise NotImplementedError


assert auc_pairwise([2, 3], [0, 1]) == 1.0 and auc_pairwise([1], [1]) == 0.5
assert hits_at_k([5, 0], [4, 3, 2, 1], 1) == 0.5, "one positive tops the merged list"
assert abs(mrr([5, 0], [4, 3, 2, 1]) - (1 + 1 / 5) / 2) < 1e-9

# the lecture's disagreement, in miniature: 10 true among 1000 candidates
rng = np.random.default_rng(0)
neg = list(rng.normal(0, 1, 990))
top_neg = sorted(neg, reverse=True)
# model A: every positive just below the top 30 negatives — great AUC, zero Hits@10
posA = [top_neg[29] - 1e-6] * 10
# model B: five positives at the very top, five at the bottom
posB = [top_neg[0] + 1] * 5 + [min(neg) - 1] * 5
aucA, aucB = auc_pairwise(posA, neg), auc_pairwise(posB, neg)
hA, hB = hits_at_k(posA, neg, 10), hits_at_k(posB, neg, 10)
print(f"model A: AUC {aucA:.3f} · Hits@10 {hA:.2f}   |   model B: AUC {aucB:.3f} · Hits@10 {hB:.2f}")
assert aucA > aucB and hB > hA, (
    "A must win AUC while B wins Hits@10 — the disagreement that decides products"
)
print("B3 \\u2713 — metrics implemented, and their disagreement is now yours to cite")"""),

    md("""## Part C · Investigation (10 pts) — how hard are your negatives?

The harness below evaluates Adamic–Adar on the karate club under two negative
samplers: uniform non-edges vs degree-matched non-edges (negatives whose
endpoint degrees mimic the positives'). Run it, then write the report.
"""),

    code("""G = nx.karate_club_graph()
Nn = 34
rng = np.random.default_rng(0)
edges = [frozenset(e) for e in G.edges()]
held = list(rng.choice(len(edges), 15, replace=False))
test_p = [tuple(edges[i]) for i in held]
msg_adj = {v: set(G[v]) for v in G}
for a, b in test_p:
    msg_adj[a].discard(b); msg_adj[b].discard(a)
non_edges = [(u, v) for u in range(Nn) for v in range(u + 1, Nn)
             if not G.has_edge(u, v)]


def aa(a, b):
    return sum(1 / math.log(len(msg_adj[c])) for c in msg_adj[a] & msg_adj[b]
               if len(msg_adj[c]) > 1)


def eval_under(negs):
    ps = [aa(*p) for p in test_p]
    ns = [aa(*n) for n in negs]
    return auc_pairwise(ps, ns)


uniform_negs = [non_edges[i] for i in rng.choice(len(non_edges), 15, replace=False)]
deg = dict(G.degree())
pos_degs = sorted(deg[a] + deg[b] for a, b in test_p)
scored = sorted(non_edges, key=lambda e: abs(deg[e[0]] + deg[e[1]] - float(np.median(pos_degs))))
matched_negs = scored[:15]
u_auc, m_auc = eval_under(uniform_negs), eval_under(matched_negs)
print(f"AA under uniform negatives:        AUC {u_auc:.3f}")
print(f"AA under degree-matched negatives: AUC {m_auc:.3f}")
print(f"hardening cost: {u_auc - m_auc:+.3f}")"""),

    answer("""### Your Part C report *(graded — write it here)*

**Numbers:** paste the two AUCs and the gap.

**Mechanism (2–3 sentences):** why does matching degrees make negatives
harder *for this particular heuristic*?

**Claims (2 sentences, numbers cited):** what does the gap say about
uniform-negative evaluations in general?

**Scope (1 sentence):** what these 15-edge karate numbers do not license.
""",
           """### Part C — model answer (structure; your exact numbers may differ slightly)

**Numbers:** uniform ≈ 0.9x, degree-matched ≈ 0.7x — a drop of roughly
0.1–0.2 AUC from hardening alone, no model change.

**Mechanism:** Adamic–Adar correlates strongly with endpoint degree — high-
degree endpoints simply have more chances at common neighbors — so uniform
negatives (mostly low-degree pairs, scoring ~0) are trivially separable,
while degree-matched negatives neutralize the degree shortcut and force the
heuristic to win on genuine shared-neighborhood structure.

**Claims:** (1) A fraction of the headline AUC under uniform negatives is
attributable to degree information alone (the gap measures it directly).
(2) Papers comparing methods under uniform negatives may be ranking degree
exploitation rather than link understanding — rankings can compress or flip
under hardening.

**Scope:** 15 test edges on a 34-node graph — the *phenomenon* is standard,
these particular magnitudes are anecdote; rerun at Cora scale before citing
figures.

*Grading: mechanism correct and specific (4) · claims cite numbers (4) ·
scope honest (2).*""",),

    md("""## Submission

One executed notebook on Moodle: Part A written, all three B checks ✓
(*Restart and run all* first — Part B is deterministic, so SMOKE and full
agree by construction), Part C run and reported.

**AI policy** (honor code): disclosure of tools + purpose below; you must be
able to explain any line and any proof step on request.

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
    STUDENT_OUT.parent.mkdir(parents=True, exist_ok=True)
    SOLUTION_OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(student, STUDENT_OUT)
    nbf.write(solution, SOLUTION_OUT)
    print("wrote", STUDENT_OUT)
    print("wrote", SOLUTION_OUT)


if __name__ == "__main__":
    main()
