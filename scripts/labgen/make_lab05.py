#!/usr/bin/env python
"""Generate Lab 5 (Query2box-lite multi-hop reasoning + mini-GraphRAG) —
student and solution notebooks.

    python scripts/labgen/make_lab05.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "labs" / "lab05_reasoning.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "labs" / "lab05_reasoning.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(text, stub): return ("code", text, stub)


CELLS = [
    md("""# Lab 5 · Questions as geometry — Query2box-lite and a mini-GraphRAG

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/labs/lab05_reasoning.ipynb)

**Week 5 · [lecture](https://lukmanovr.github.io/dkr/lectures/05-kg-reasoning.html) · ≈ 25 min of compute (free Colab or CPU — no GPU needed)**

Two halves, matching the lecture. First you build box embeddings — distance,
projection, intersection — train them on real FB15k-237 multi-hop queries, and produce
the **easy/hard answer split** that separates reasoning from lookup. Then you build
both retrieval pipelines of the GraphRAG bake-off and referee them with the coverage
metric.

### Goals
1. Implement the box distance, projection, and intersection operators against hand asserts.
2. Compose 2-hop query boxes and train them on FB15k-237.
3. Compute the easy/hard answer split (checked on the lecture's toy KG) and measure the gap.
4. Build vector and graph retrieval over a small corpus and score both with fact coverage.
"""),

    md("""## 0 · Setup"""),

    code("""import os, sys, random, math
from collections import defaultdict, Counter

SMOKE = os.environ.get("SMOKE", "") == "1"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    %pip install -q torch_geometric==2.8.0.post1

import numpy as np
import torch
import torch.nn.functional as F

SEED = 5
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"torch {torch.__version__} — environment OK")"""),

    md("""## 1 · The box distance  *(exercise 1 — skill: inside vs outside)*

Query2box scores an entity $\\mathbf{v}$ against a box $(\\mathbf{c}, \\mathbf{o})$
with two L1 terms:
$$
d(\\mathbf{v}; \\mathbf{c}, \\mathbf{o}) = \\underbrace{\\bigl\\|\\max(|\\mathbf{v}-\\mathbf{c}| - \\mathbf{o},\\, 0)\\bigr\\|_1}_{\\text{outside}}
\\; + \\; \\alpha \\underbrace{\\bigl\\|\\min(|\\mathbf{v}-\\mathbf{c}|,\\, \\mathbf{o})\\bigr\\|_1}_{\\text{inside}}
$$
with $\\alpha < 1$: being outside is expensive, position *within* the box is cheap but
not free (so the model can still rank inside answers).
"""),

    todo("""ALPHA = 0.2


def box_dist(v: torch.Tensor, c: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    \"\"\"v: (..., d) points. c, o: (..., d) box center and (nonneg) offset,
    broadcastable against v. Returns (...,) distances.\"\"\"
    ### BEGIN SOLUTION
    delta = (v - c).abs()
    outside = (delta - o).clamp(min=0).sum(dim=-1)
    inside = torch.minimum(delta, o).sum(dim=-1)
    return outside + ALPHA * inside
    ### END SOLUTION


c1 = torch.tensor([0.0]); o1 = torch.tensor([1.0])
assert abs(box_dist(torch.tensor([0.0]), c1, o1).item() - 0.0) < 1e-6, "at the center: distance 0"
assert abs(box_dist(torch.tensor([0.5]), c1, o1).item() - 0.1) < 1e-6, (
    "inside at |v−c|=0.5: only the inside term fires → 0.2 × 0.5 = 0.1"
)
assert abs(box_dist(torch.tensor([2.0]), c1, o1).item() - 1.2) < 1e-6, (
    "outside at |v−c|=2, o=1: outside 1.0 + inside 0.2·min(2,1)=0.2 → 1.2"
)
b = box_dist(torch.tensor([[0.5, 0.0], [2.0, 0.0]]), torch.zeros(2), torch.ones(2))
assert b.shape == (2,), "must broadcast over a batch of points"
print("exercise 1 ✓ — the box knows its inside from its outside")""",
         stub="""ALPHA = 0.2


def box_dist(v: torch.Tensor, c: torch.Tensor, o: torch.Tensor) -> torch.Tensor:
    \"\"\"v: (..., d) points. c, o: (..., d) box center and (nonneg) offset,
    broadcastable against v. Returns (...,) distances.\"\"\"
    # TODO: delta = |v−c|; outside = relu(delta − o) summed; inside = min(delta, o)
    # summed; return outside + ALPHA * inside. 4 lines.
    raise NotImplementedError


c1 = torch.tensor([0.0]); o1 = torch.tensor([1.0])
assert abs(box_dist(torch.tensor([0.0]), c1, o1).item() - 0.0) < 1e-6, "at the center: distance 0"
assert abs(box_dist(torch.tensor([0.5]), c1, o1).item() - 0.1) < 1e-6, (
    "inside at |v−c|=0.5: only the inside term fires → 0.2 × 0.5 = 0.1"
)
assert abs(box_dist(torch.tensor([2.0]), c1, o1).item() - 1.2) < 1e-6, (
    "outside at |v−c|=2, o=1: outside 1.0 + inside 0.2·min(2,1)=0.2 → 1.2"
)
b = box_dist(torch.tensor([[0.5, 0.0], [2.0, 0.0]]), torch.zeros(2), torch.ones(2))
assert b.shape == (2,), "must broadcast over a batch of points"
print("exercise 1 ✓ — the box knows its inside from its outside")"""),

    md("""## 2 · Projection and intersection  *(exercise 2 — skill: the two operators)*

Projection translates the center and **grows** the offset (following a relation can
only widen an answer set); intersection averages the centers and **shrinks** the
offset to the elementwise minimum. (The paper adds attention and a learned shrink —
this is the honest "lite" version, and it works.)
"""),

    todo("""def project(c, o, rc, ro):
    \"\"\"Apply relation (rc, ro) to box (c, o): translate center, grow offset.
    Offsets must stay nonnegative: grow by |ro|.\"\"\"
    ### BEGIN SOLUTION
    return c + rc, o + ro.abs()
    ### END SOLUTION


def intersect(cs, os_):
    \"\"\"cs, os_: (k, d) stacked centers/offsets of k boxes.
    Return (d,) center = mean of centers, (d,) offset = elementwise min.\"\"\"
    ### BEGIN SOLUTION
    return cs.mean(dim=0), os_.min(dim=0).values
    ### END SOLUTION


c, o = project(torch.tensor([1.0, 1.0]), torch.tensor([0.5, 0.5]),
               torch.tensor([2.0, -1.0]), torch.tensor([-0.5, 1.0]))
assert torch.allclose(c, torch.tensor([3.0, 0.0])) and torch.allclose(o, torch.tensor([1.0, 1.5])), (
    "projection: center 1+2=3, 1−1=0; offset 0.5+|−0.5|=1.0, 0.5+1=1.5 (offsets grow by |ro|)"
)
ci, oi = intersect(torch.tensor([[1.0], [2.0]]), torch.tensor([[1.0], [1.0]]))
assert abs(ci.item() - 1.5) < 1e-6 and abs(oi.item() - 1.0) < 1e-6, (
    "intersection of [0,2] and [1,3]: lite center = 1.5 (mean), offset = min = 1.0"
)
print("exercise 2 ✓ — project grows, intersect shrinks")""",
         stub="""def project(c, o, rc, ro):
    \"\"\"Apply relation (rc, ro) to box (c, o): translate center, grow offset.
    Offsets must stay nonnegative: grow by |ro|.\"\"\"
    # TODO: one line.
    raise NotImplementedError


def intersect(cs, os_):
    \"\"\"cs, os_: (k, d) stacked centers/offsets of k boxes.
    Return (d,) center = mean of centers, (d,) offset = elementwise min.\"\"\"
    # TODO: one line.
    raise NotImplementedError


c, o = project(torch.tensor([1.0, 1.0]), torch.tensor([0.5, 0.5]),
               torch.tensor([2.0, -1.0]), torch.tensor([-0.5, 1.0]))
assert torch.allclose(c, torch.tensor([3.0, 0.0])) and torch.allclose(o, torch.tensor([1.0, 1.5])), (
    "projection: center 1+2=3, 1−1=0; offset 0.5+|−0.5|=1.0, 0.5+1=1.5 (offsets grow by |ro|)"
)
ci, oi = intersect(torch.tensor([[1.0], [2.0]]), torch.tensor([[1.0], [1.0]]))
assert abs(ci.item() - 1.5) < 1e-6 and abs(oi.item() - 1.0) < 1e-6, (
    "intersection of [0,2] and [1,3]: lite center = 1.5 (mean), offset = min = 1.0"
)
print("exercise 2 ✓ — project grows, intersect shrinks")"""),

    md("""## 3 · Easy vs hard answers  *(exercise 3 — checked on the lecture's toy KG)*

Before training anything: the evaluation discipline. For a 2-hop query, an answer is
**easy** if the train graph alone reaches it, **hard** if reaching it needs a held-out
edge. Your function is checked against the lecture's biomedical toy — where you
already know the split by heart.
"""),

    todo("""def two_hop_answers(adj: dict, anchor, r1, r2) -> set:
    \"\"\"adj: {(head, rel): set(tails)}. The query is anchor —r1→ mid —r2→ answer.
    Return the set of all reachable answers.\"\"\"
    ### BEGIN SOLUTION
    out = set()
    for mid in adj.get((anchor, r1), set()):
        out |= adj.get((mid, r2), set())
    return out
    ### END SOLUTION


def easy_hard_split(train_adj: dict, full_adj: dict, anchor, r1, r2) -> tuple:
    \"\"\"Return (easy, hard): easy = answers reachable in the TRAIN graph;
    hard = answers reachable in the FULL graph but not in train.\"\"\"
    ### BEGIN SOLUTION
    easy = two_hop_answers(train_adj, anchor, r1, r2)
    return easy, two_hop_answers(full_adj, anchor, r1, r2) - easy
    ### END SOLUTION


# the lecture's toy: assoc⁻¹ then targets⁻¹ from Inflammation
train_adj = defaultdict(set)
full_adj = defaultdict(set)
STATED = [("Inflammation", "assoc_rev", "COX-2"), ("COX-2", "targets_rev", "Aspirin"),
          ("COX-2", "targets_rev", "Ibuprofen")]
for h, r, t in STATED:
    train_adj[(h, r)].add(t); full_adj[(h, r)].add(t)
full_adj[("COX-2", "targets_rev")].add("Naproxen")     # the never-stated edge

easy, hard = easy_hard_split(train_adj, full_adj, "Inflammation", "assoc_rev", "targets_rev")
assert easy == {"Aspirin", "Ibuprofen"}, f"easy answers must be the traversal set, got {easy}"
assert hard == {"Naproxen"}, f"hard answers = full-graph reachable minus easy, got {hard}"
print("exercise 3 ✓ — the split that separates reasoning from recall, verified on the hero query")""",
         stub="""def two_hop_answers(adj: dict, anchor, r1, r2) -> set:
    \"\"\"adj: {(head, rel): set(tails)}. The query is anchor —r1→ mid —r2→ answer.
    Return the set of all reachable answers.\"\"\"
    # TODO: union of adj[(mid, r2)] over mids in adj[(anchor, r1)]. 4 lines.
    raise NotImplementedError


def easy_hard_split(train_adj: dict, full_adj: dict, anchor, r1, r2) -> tuple:
    \"\"\"Return (easy, hard): easy = answers reachable in the TRAIN graph;
    hard = answers reachable in the FULL graph but not in train.\"\"\"
    # TODO: two calls to two_hop_answers and one set difference.
    raise NotImplementedError


# the lecture's toy: assoc⁻¹ then targets⁻¹ from Inflammation
train_adj = defaultdict(set)
full_adj = defaultdict(set)
STATED = [("Inflammation", "assoc_rev", "COX-2"), ("COX-2", "targets_rev", "Aspirin"),
          ("COX-2", "targets_rev", "Ibuprofen")]
for h, r, t in STATED:
    train_adj[(h, r)].add(t); full_adj[(h, r)].add(t)
full_adj[("COX-2", "targets_rev")].add("Naproxen")     # the never-stated edge

easy, hard = easy_hard_split(train_adj, full_adj, "Inflammation", "assoc_rev", "targets_rev")
assert easy == {"Aspirin", "Ibuprofen"}, f"easy answers must be the traversal set, got {easy}"
assert hard == {"Naproxen"}, f"hard answers = full-graph reachable minus easy, got {hard}"
print("exercise 3 ✓ — the split that separates reasoning from recall, verified on the hero query")"""),

    md("""## 4 · Train on real queries, measure the gap  *(provided + exercise 4)*

FB15k-237 again (cached from Lab 4 if you ran it in the same runtime). We sample
2-hop paths, train boxes on the **train graph only**, and evaluate Hits@10 on easy
versus hard answers of held-out queries. Your part: composing the 2-hop query box
from your own operators.
"""),

    todo("""from torch_geometric.datasets import FB15k_237

splits = {}
for split in ["train", "val", "test"]:
    d = FB15k_237(root="data/FB15k237", split=split)[0]
    splits[split] = torch.stack([d.edge_index[0], d.edge_type, d.edge_index[1]], dim=1)
N_ENT = int(FB15k_237(root="data/FB15k237", split="train")[0].num_nodes)
N_REL = int(max(s[:, 1].max() for s in splits.values())) + 1
train_T = splits["train"]

train_adj_fb = defaultdict(set)
full_adj_fb = defaultdict(set)
for name, T in splits.items():
    for h, r, t in T.tolist():
        full_adj_fb[(h, r)].add(t)
        if name == "train":
            train_adj_fb[(h, r)].add(t)

D = 32
g = torch.Generator().manual_seed(0)
E = torch.nn.Parameter(0.3 * torch.randn(N_ENT, D, generator=g))
RC = torch.nn.Parameter(0.3 * torch.randn(N_REL, D, generator=g))
RO = torch.nn.Parameter(0.1 * torch.rand(N_REL, D, generator=g))


def query_box_2p(anchor_ids: torch.Tensor, r1_ids: torch.Tensor, r2_ids: torch.Tensor):
    \"\"\"Batch of 2-hop queries → (B, d) centers and offsets, using YOUR project():
    start from the anchor entity as a zero-offset box, project by r1, then by r2.\"\"\"
    ### BEGIN SOLUTION
    c, o = E[anchor_ids], torch.zeros_like(E[anchor_ids])
    c, o = project(c, o, RC[r1_ids], RO[r1_ids])
    c, o = project(c, o, RC[r2_ids], RO[r2_ids])
    return c, o
    ### END SOLUTION


ct, ot = query_box_2p(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
assert ct.shape == (1, D) and ot.shape == (1, D) and (ot >= 0).all(), (
    "compose anchor → project(r1) → project(r2); offsets must be nonnegative"
)

# — sample training paths (from the train graph) and test queries (with hard answers)
def sample_paths(adj, k, seed):
    rng = random.Random(seed)
    keys = [kk for kk in adj if adj[kk]]
    out_rels = defaultdict(list)                 # head -> relations leaving it
    for (h, r) in keys:
        out_rels[h].append(r)
    out = []
    while len(out) < k:
        (a, r1) = rng.choice(keys)
        mid = rng.choice(sorted(adj[(a, r1)]))
        if not out_rels[mid]:
            continue
        r2 = rng.choice(out_rels[mid])
        t = rng.choice(sorted(adj[(mid, r2)]))
        out.append((a, r1, r2, t))
    return out

N_TRAIN_Q = 1500 if SMOKE else 6000
print("sampling paths...")
train_q = sample_paths(train_adj_fb, N_TRAIN_Q, seed=1)

test_q = []
rng_t = random.Random(2)
tries = 0
while len(test_q) < (60 if SMOKE else 200) and tries < 400000:
    tries += 1
    a, r1, r2, _ = train_q[rng_t.randrange(len(train_q))]
    ez, hd = easy_hard_split(train_adj_fb, full_adj_fb, a, r1, r2)
    if hd and ez:
        test_q.append((a, r1, r2, ez, hd))
print(f"{len(train_q)} training paths · {len(test_q)} test queries with both easy AND hard answers")

opt = torch.optim.Adam([E, RC, RO], lr=0.01)
GAMMA, K_NEG = 6.0, 16

# phase A — 1-hop pretraining on ALL train triples: every triple is a tiny query
# (anchor —r→ ?). Without this, 237 relations must learn their geometry from a few
# thousand path samples, and they don't. (Found the hard way; keep the phase.)
print("phase A: 1-hop pretraining...")
for ep in range(1 if SMOKE else 3):
    perm = torch.randperm(len(train_T), generator=g)
    for i in range(0, len(train_T), 1024):
        idx = perm[i:i + 1024]
        h, r, t = train_T[idx, 0], train_T[idx, 1], train_T[idx, 2]
        c, o = project(E[h], torch.zeros_like(E[h]), RC[r], RO[r])
        d_pos = box_dist(E[t], c, o)
        negs = torch.randint(0, N_ENT, (len(idx), K_NEG), generator=g)
        d_neg = box_dist(E[negs], c.unsqueeze(1), o.unsqueeze(1))
        loss = (-F.logsigmoid(GAMMA - d_pos) - F.logsigmoid(d_neg - GAMMA).mean(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()

# phase B — 2-hop fine-tuning on the sampled paths
print("phase B: 2-hop fine-tuning...")
TQ = torch.tensor([[a, r1, r2, t] for a, r1, r2, t in train_q])
for ep in range(2 if SMOKE else 5):
    perm = torch.randperm(len(TQ), generator=g)
    total = 0.0
    for i in range(0, len(TQ), 512):
        idx = perm[i:i + 512]
        a, r1, r2, t = TQ[idx, 0], TQ[idx, 1], TQ[idx, 2], TQ[idx, 3]
        c, o = query_box_2p(a, r1, r2)
        d_pos = box_dist(E[t], c, o)
        negs = torch.randint(0, N_ENT, (len(idx), K_NEG), generator=g)
        d_neg = box_dist(E[negs], c.unsqueeze(1), o.unsqueeze(1))
        loss = (-F.logsigmoid(GAMMA - d_pos) - F.logsigmoid(d_neg - GAMMA).mean(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(idx)
    print(f"  epoch {ep + 1}: loss {total / len(TQ):.4f}")

# — evaluation: Hits@10 on easy vs hard answers (each filtered against the rest)
with torch.no_grad():
    easy_hits, hard_hits = [], []
    for a, r1, r2, ez, hd in test_q:
        c, o = query_box_2p(torch.tensor([a]), torch.tensor([r1]), torch.tensor([r2]))
        dists = box_dist(E, c, o)                     # (N_ENT,)
        all_ans = ez | hd
        for kind, answers, sink in [("easy", ez, easy_hits), ("hard", hd, hard_hits)]:
            for ans in answers:
                dd = dists.clone()
                for other in all_ans:
                    if other != ans:
                        dd[other] = float("inf")
                rank = int((dd < dd[ans]).sum().item()) + 1
                sink.append(rank <= 10)
easy_h = float(np.mean(easy_hits)); hard_h = float(np.mean(hard_hits))
rand_h = 10 / N_ENT
print(f"Hits@10 — easy answers: {easy_h:.3f} · hard answers: {hard_h:.3f} · random: {rand_h:.4f}")
if not SMOKE:                      # the ordering is a statistical claim — the tiny
    assert easy_h > hard_h, (      # SMOKE sample is a plumbing check, not statistics
        "easy answers must outscore hard ones — the model has literally seen the paths to them"
    )
assert hard_h > (3 if SMOKE else 10) * rand_h, (
    f"hard Hits@10 {hard_h:.3f} should still clear random ({rand_h:.4f}) by a wide margin — "
    f"the box generalizes; if it doesn't, check that project() feeds YOUR trained RC/RO"
)
print(f"exercise 4 ✓ — the gap is the lecture's §3 in one line: report hard, or you are reporting a database")""",
         stub="""from torch_geometric.datasets import FB15k_237

splits = {}
for split in ["train", "val", "test"]:
    d = FB15k_237(root="data/FB15k237", split=split)[0]
    splits[split] = torch.stack([d.edge_index[0], d.edge_type, d.edge_index[1]], dim=1)
N_ENT = int(FB15k_237(root="data/FB15k237", split="train")[0].num_nodes)
N_REL = int(max(s[:, 1].max() for s in splits.values())) + 1
train_T = splits["train"]

train_adj_fb = defaultdict(set)
full_adj_fb = defaultdict(set)
for name, T in splits.items():
    for h, r, t in T.tolist():
        full_adj_fb[(h, r)].add(t)
        if name == "train":
            train_adj_fb[(h, r)].add(t)

D = 32
g = torch.Generator().manual_seed(0)
E = torch.nn.Parameter(0.3 * torch.randn(N_ENT, D, generator=g))
RC = torch.nn.Parameter(0.3 * torch.randn(N_REL, D, generator=g))
RO = torch.nn.Parameter(0.1 * torch.rand(N_REL, D, generator=g))


def query_box_2p(anchor_ids: torch.Tensor, r1_ids: torch.Tensor, r2_ids: torch.Tensor):
    \"\"\"Batch of 2-hop queries → (B, d) centers and offsets, using YOUR project():
    start from the anchor entity as a zero-offset box, project by r1, then by r2.\"\"\"
    # TODO: c = E[anchor], o = zeros; project twice. 4 lines.
    raise NotImplementedError


ct, ot = query_box_2p(torch.tensor([0]), torch.tensor([0]), torch.tensor([1]))
assert ct.shape == (1, D) and ot.shape == (1, D) and (ot >= 0).all(), (
    "compose anchor → project(r1) → project(r2); offsets must be nonnegative"
)

# — sample training paths (from the train graph) and test queries (with hard answers)
def sample_paths(adj, k, seed):
    rng = random.Random(seed)
    keys = [kk for kk in adj if adj[kk]]
    out_rels = defaultdict(list)                 # head -> relations leaving it
    for (h, r) in keys:
        out_rels[h].append(r)
    out = []
    while len(out) < k:
        (a, r1) = rng.choice(keys)
        mid = rng.choice(sorted(adj[(a, r1)]))
        if not out_rels[mid]:
            continue
        r2 = rng.choice(out_rels[mid])
        t = rng.choice(sorted(adj[(mid, r2)]))
        out.append((a, r1, r2, t))
    return out

N_TRAIN_Q = 1500 if SMOKE else 6000
print("sampling paths...")
train_q = sample_paths(train_adj_fb, N_TRAIN_Q, seed=1)

test_q = []
rng_t = random.Random(2)
tries = 0
while len(test_q) < (60 if SMOKE else 200) and tries < 400000:
    tries += 1
    a, r1, r2, _ = train_q[rng_t.randrange(len(train_q))]
    ez, hd = easy_hard_split(train_adj_fb, full_adj_fb, a, r1, r2)
    if hd and ez:
        test_q.append((a, r1, r2, ez, hd))
print(f"{len(train_q)} training paths · {len(test_q)} test queries with both easy AND hard answers")

opt = torch.optim.Adam([E, RC, RO], lr=0.01)
GAMMA, K_NEG = 6.0, 16

# phase A — 1-hop pretraining on ALL train triples: every triple is a tiny query
# (anchor —r→ ?). Without this, 237 relations must learn their geometry from a few
# thousand path samples, and they don't. (Found the hard way; keep the phase.)
print("phase A: 1-hop pretraining...")
for ep in range(1 if SMOKE else 3):
    perm = torch.randperm(len(train_T), generator=g)
    for i in range(0, len(train_T), 1024):
        idx = perm[i:i + 1024]
        h, r, t = train_T[idx, 0], train_T[idx, 1], train_T[idx, 2]
        c, o = project(E[h], torch.zeros_like(E[h]), RC[r], RO[r])
        d_pos = box_dist(E[t], c, o)
        negs = torch.randint(0, N_ENT, (len(idx), K_NEG), generator=g)
        d_neg = box_dist(E[negs], c.unsqueeze(1), o.unsqueeze(1))
        loss = (-F.logsigmoid(GAMMA - d_pos) - F.logsigmoid(d_neg - GAMMA).mean(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()

# phase B — 2-hop fine-tuning on the sampled paths
print("phase B: 2-hop fine-tuning...")
TQ = torch.tensor([[a, r1, r2, t] for a, r1, r2, t in train_q])
for ep in range(2 if SMOKE else 5):
    perm = torch.randperm(len(TQ), generator=g)
    total = 0.0
    for i in range(0, len(TQ), 512):
        idx = perm[i:i + 512]
        a, r1, r2, t = TQ[idx, 0], TQ[idx, 1], TQ[idx, 2], TQ[idx, 3]
        c, o = query_box_2p(a, r1, r2)
        d_pos = box_dist(E[t], c, o)
        negs = torch.randint(0, N_ENT, (len(idx), K_NEG), generator=g)
        d_neg = box_dist(E[negs], c.unsqueeze(1), o.unsqueeze(1))
        loss = (-F.logsigmoid(GAMMA - d_pos) - F.logsigmoid(d_neg - GAMMA).mean(1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item() * len(idx)
    print(f"  epoch {ep + 1}: loss {total / len(TQ):.4f}")

# — evaluation: Hits@10 on easy vs hard answers (each filtered against the rest)
with torch.no_grad():
    easy_hits, hard_hits = [], []
    for a, r1, r2, ez, hd in test_q:
        c, o = query_box_2p(torch.tensor([a]), torch.tensor([r1]), torch.tensor([r2]))
        dists = box_dist(E, c, o)                     # (N_ENT,)
        all_ans = ez | hd
        for kind, answers, sink in [("easy", ez, easy_hits), ("hard", hd, hard_hits)]:
            for ans in answers:
                dd = dists.clone()
                for other in all_ans:
                    if other != ans:
                        dd[other] = float("inf")
                rank = int((dd < dd[ans]).sum().item()) + 1
                sink.append(rank <= 10)
easy_h = float(np.mean(easy_hits)); hard_h = float(np.mean(hard_hits))
rand_h = 10 / N_ENT
print(f"Hits@10 — easy answers: {easy_h:.3f} · hard answers: {hard_h:.3f} · random: {rand_h:.4f}")
if not SMOKE:                      # the ordering is a statistical claim — the tiny
    assert easy_h > hard_h, (      # SMOKE sample is a plumbing check, not statistics
        "easy answers must outscore hard ones — the model has literally seen the paths to them"
    )
assert hard_h > (3 if SMOKE else 10) * rand_h, (
    f"hard Hits@10 {hard_h:.3f} should still clear random ({rand_h:.4f}) by a wide margin — "
    f"the box generalizes; if it doesn't, check that project() feeds YOUR trained RC/RO"
)
print(f"exercise 4 ✓ — the gap is the lecture's §3 in one line: report hard, or you are reporting a database")"""),

    md("""## 5 · Mini-GraphRAG  *(exercise 5 — the bake-off, refereed by coverage)*

Six documents, a KG extracted from them (with provenance), three questions with gold
supporting facts. The vector retriever is provided (real cosine over bag-of-words);
you build the graph retriever: link entities by string match, expand k hops, collect
facts. The referee is the coverage metric from the lecture's Figure 3.
"""),

    todo("""DOCS = [
    "Innopolis University is a young IT university located in the city of Innopolis.",
    "Innopolis is a town in the Republic of Tatarstan, founded in 2012.",
    "Kazan Federal University is one of the oldest universities in Russia, located in Kazan.",
    "Kazan is the capital of Tatarstan and a major university city.",
    "The Volga river flows through the Republic of Tatarstan.",
    "Innopolis University hosts an annual robotics olympiad for school students.",
]
KG = [  # (head, relation, tail, source_doc)
    ("Innopolis University", "located_in", "Innopolis", 0),
    ("Innopolis", "located_in", "Tatarstan", 1),
    ("KFU", "located_in", "Kazan", 2),
    ("Kazan", "capital_of", "Tatarstan", 3),
    ("Volga", "flows_through", "Tatarstan", 4),
    ("Innopolis University", "hosts", "robotics olympiad", 5),
]
QUESTIONS = [
    {"q": "Which region is Innopolis University located in?", "link": ["Innopolis University"], "hops": 2, "gold": [0, 1]},
    {"q": "In which city is Kazan Federal University located?", "link": ["KFU"], "hops": 1, "gold": [2]},
    {"q": "Do local school students get any outreach events?", "link": [], "hops": 1, "gold": [5]},
]
STOP = {"is", "a", "the", "in", "of", "for", "which", "do", "any", "get", "and",
        "one", "an", "on", "to", "it", "was", "are", "through"}


def bag(text):
    counts = Counter(w for w in text.lower().replace("?", "").replace(".", "").replace(",", "").split()
                     if w not in STOP)
    return counts


def cosine(a, b):
    dot = sum(a[w] * b[w] for w in a if w in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb or 1)


def vector_facts(question, k=2):
    \"\"\"Provided: top-k docs by cosine; the retrieved facts are those sourced there.\"\"\"
    qb = bag(question)
    top = sorted(range(len(DOCS)), key=lambda i: -cosine(qb, bag(DOCS[i])))[:k]
    return {fi for fi, (_, _, _, d) in enumerate(KG) if d in top}


def graph_facts(entities: list, hops: int) -> set:
    \"\"\"YOUR retriever: start from the linked entities; for each hop, collect every
    fact whose head OR tail is in the frontier, and add both endpoints to it.
    Return the set of fact indices.\"\"\"
    ### BEGIN SOLUTION
    frontier = set(entities)
    facts = set()
    for _ in range(hops):
        grow = set(frontier)
        for fi, (h, _, t, _) in enumerate(KG):
            if h in frontier or t in frontier:
                facts.add(fi); grow.add(h); grow.add(t)
        frontier = grow
    return facts
    ### END SOLUTION


def coverage(retrieved: set, gold: list) -> float:
    return sum(1 for fi in gold if fi in retrieved) / len(gold)


results = []
for Q in QUESTIONS:
    v = coverage(vector_facts(Q["q"]), Q["gold"])
    gr = coverage(graph_facts(Q["link"], Q["hops"]), Q["gold"]) if Q["link"] else 0.0
    results.append((Q["q"][:44], v, gr))
    print(f"{Q['q'][:52]:<54} vector {v:.2f} · graph {gr:.2f}")

assert results[0][1] < 1.0 and results[0][2] == 1.0, (
    "the 2-hop question: top-2 cosine misses a hop (the Tatarstan doc reads like the "
    "Kazan docs), while 2-hop expansion from IU must cover both gold facts"
)
assert results[1][2] == 1.0, "the 1-hop question: graph retrieval covers its single fact"
assert results[2][1] == 1.0 and results[2][2] == 0.0, (
    "the fuzzy question has no linkable entity: graph retrieval comes back empty, "
    "similarity finds the olympiad doc — this is vector RAG's home turf"
)
print("exercise 5 ✓ — structure wins the hops, similarity wins the fuzz; production routes between them")""",
         stub="""DOCS = [
    "Innopolis University is a young IT university located in the city of Innopolis.",
    "Innopolis is a town in the Republic of Tatarstan, founded in 2012.",
    "Kazan Federal University is one of the oldest universities in Russia, located in Kazan.",
    "Kazan is the capital of Tatarstan and a major university city.",
    "The Volga river flows through the Republic of Tatarstan.",
    "Innopolis University hosts an annual robotics olympiad for school students.",
]
KG = [  # (head, relation, tail, source_doc)
    ("Innopolis University", "located_in", "Innopolis", 0),
    ("Innopolis", "located_in", "Tatarstan", 1),
    ("KFU", "located_in", "Kazan", 2),
    ("Kazan", "capital_of", "Tatarstan", 3),
    ("Volga", "flows_through", "Tatarstan", 4),
    ("Innopolis University", "hosts", "robotics olympiad", 5),
]
QUESTIONS = [
    {"q": "Which region is Innopolis University located in?", "link": ["Innopolis University"], "hops": 2, "gold": [0, 1]},
    {"q": "In which city is Kazan Federal University located?", "link": ["KFU"], "hops": 1, "gold": [2]},
    {"q": "Do local school students get any outreach events?", "link": [], "hops": 1, "gold": [5]},
]
STOP = {"is", "a", "the", "in", "of", "for", "which", "do", "any", "get", "and",
        "one", "an", "on", "to", "it", "was", "are", "through"}


def bag(text):
    counts = Counter(w for w in text.lower().replace("?", "").replace(".", "").replace(",", "").split()
                     if w not in STOP)
    return counts


def cosine(a, b):
    dot = sum(a[w] * b[w] for w in a if w in b)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb or 1)


def vector_facts(question, k=2):
    \"\"\"Provided: top-k docs by cosine; the retrieved facts are those sourced there.\"\"\"
    qb = bag(question)
    top = sorted(range(len(DOCS)), key=lambda i: -cosine(qb, bag(DOCS[i])))[:k]
    return {fi for fi, (_, _, _, d) in enumerate(KG) if d in top}


def graph_facts(entities: list, hops: int) -> set:
    \"\"\"YOUR retriever: start from the linked entities; for each hop, collect every
    fact whose head OR tail is in the frontier, and add both endpoints to it.
    Return the set of fact indices.\"\"\"
    # TODO: frontier = set(entities); per hop, sweep KG, collect touching facts,
    # grow the frontier with their endpoints. ~9 lines.
    raise NotImplementedError


def coverage(retrieved: set, gold: list) -> float:
    return sum(1 for fi in gold if fi in retrieved) / len(gold)


results = []
for Q in QUESTIONS:
    v = coverage(vector_facts(Q["q"]), Q["gold"])
    gr = coverage(graph_facts(Q["link"], Q["hops"]), Q["gold"]) if Q["link"] else 0.0
    results.append((Q["q"][:44], v, gr))
    print(f"{Q['q'][:52]:<54} vector {v:.2f} · graph {gr:.2f}")

assert results[0][1] < 1.0 and results[0][2] == 1.0, (
    "the 2-hop question: top-2 cosine misses a hop (the Tatarstan doc reads like the "
    "Kazan docs), while 2-hop expansion from IU must cover both gold facts"
)
assert results[1][2] == 1.0, "the 1-hop question: graph retrieval covers its single fact"
assert results[2][1] == 1.0 and results[2][2] == 0.0, (
    "the fuzzy question has no linkable entity: graph retrieval comes back empty, "
    "similarity finds the olympiad doc — this is vector RAG's home turf"
)
print("exercise 5 ✓ — structure wins the hops, similarity wins the fuzz; production routes between them")"""),

    md("""## 6 · Stretch (optional, ungraded)

1. **Intersections for real.** Sample 2i queries (two 1-hop branches meeting) from
   FB15k-237, answer them with your `intersect`, and compare hard-answer Hits@10
   against answering each branch separately and set-intersecting the top-100s.
2. **The gap curve.** Re-run exercise 4 at D ∈ {8, 32, 128}: does more capacity close
   the easy/hard gap, or mostly inflate the easy side? (This is project topic C4's
   seed.)
3. **Break the linker.** Add a question that refers to IU as "the young university on
   the Volga side" and fix `graph_facts` to survive it (alias table? substring
   heuristics? embedding match?). Measure what your fix breaks.

## 7 · Reflection (answer in this cell, 2–4 sentences each)

**R1.** Your exercise-4 gap: state the easy and hard numbers, then explain in your own
words what a leaderboard that ignores this split is actually ranking.

**R2.** In the bake-off, the vector retriever failed the 2-hop question because of a
*lookalike* document. Real corpora are full of lookalikes. What property of graph
retrieval makes it immune to this specific failure — and what does it pay for that
immunity (see the fuzzy question)?

**R3.** Lab 4's TransE also "generalized" to unseen facts, and today's boxes
generalize to hard answers. What is the *additional* thing query embedding buys that
one-hop completion does not — and what did @prp-closure say it still cannot buy?

*(your answers here)*

## What to submit

One executed notebook on Moodle: all five exercise checks ✓ and the three reflection
answers. Grading: assertions 70% · reflections 30%. Run *Runtime → Restart and run
all* before submitting.

**AI policy reminder** (course honor code): AI assistants are allowed for this lab
*with disclosure* — add a line here naming any tools you used and for what. You must
be able to explain any line of your submission on request; undeclared use or
inability to explain is a violation.
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
