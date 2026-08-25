#!/usr/bin/env python
"""Generate Lab 4 (TransE and RotatE from scratch on FB15k-237, filtered
evaluation, and the symmetric-relation showdown) — student + solution notebooks.

    python scripts/labgen/make_lab04.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "labs" / "lab04_kge.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "labs" / "lab04_kge.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(text, stub): return ("code", text, stub)


CELLS = [
    md("""# Lab 4 · Facts as geometry — TransE and RotatE on FB15k-237

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/labs/lab04_kge.ipynb)

**Week 4 · [lecture](https://lukmanovr.github.io/dkr/lectures/04-knowledge-graphs.html) · ≈ 25 min of compute (free Colab or CPU — no GPU needed)**

Knowledge-graph embedding turns facts into geometry: entities become vectors, a
relation becomes a geometric operation, and the truth of an unstated fact becomes a
score you can rank. It is the workhorse of KG completion — the only approach that
scales to the structured missingness of real knowledge graphs. In this lab you
implement the two bookend models of the modern family from scratch (~100 lines
each): **TransE**, the translation model that started the field
([Bordes et al., 2013](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)),
and **RotatE**, its strongest direct descendant
([Sun et al., 2019](https://arxiv.org/abs/1902.10197)). You train them on the honest
benchmark, evaluate with the filtered protocol you can now compute by hand — and
close with the lecture's signature experiment: catching
[the TransE-symmetry proposition from the lecture](https://lukmanovr.github.io/dkr/lectures/04-knowledge-graphs.html#prp-transe-sym)
red-handed on a synthetic symmetric relation while RotatE parks its phases at π.

### Goals
1. Implement the TransE score, corruption sampling, and the margin loss — each against hand-checkable asserts.
2. Train TransE on FB15k-237 and run **filtered** MRR / Hits@10 on held-out facts.
3. Implement the RotatE score in complex tensors.
4. The showdown: train both models on symmetric + antisymmetric synthetic relations and measure exactly what the theory predicts.
"""),

    md("""## 0 · Setup"""),

    code("""import os, sys, random, math
SMOKE = os.environ.get("SMOKE", "") == "1"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    %pip install -q torch_geometric==2.8.0.post1

import numpy as np
import torch
import torch.nn.functional as F

SEED = 4
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"torch {torch.__version__} — environment OK")"""),

    md("""## 1 · The honest benchmark  *(provided — read the printout like a datasheet)*

FB15k-237: Freebase facts, with the leaky inverse relations already removed
([Toutanova & Chen, 2015](https://doi.org/10.18653/v1/W15-4007) — this week's
optional reading, and Pitfall 3). The download is a
few tens of MB from an external host; if it fails transiently, re-run the cell.
"""),

    code("""from torch_geometric.datasets import FB15k_237

splits = {}
for split in ["train", "val", "test"]:
    d = FB15k_237(root="data/FB15k237", split=split)[0]
    splits[split] = torch.stack([d.edge_index[0], d.edge_type, d.edge_index[1]], dim=1)
N_ENT = int(FB15k_237(root="data/FB15k237", split="train")[0].num_nodes)
N_REL = int(max(s[:, 1].max() for s in splits.values())) + 1

train_T, valid_T, test_T = splits["train"], splits["val"], splits["test"]
if SMOKE:
    g = torch.Generator().manual_seed(0)
    train_T = train_T[torch.randperm(len(train_T), generator=g)[:60000]]
print(f"entities {N_ENT:,} · relations {N_REL} · train {len(train_T):,} · "
      f"valid {len(valid_T):,} · test {len(test_T):,}")
print("every row is one fact: (head, relation, tail) as integer ids")"""),

    md("""## 2 · The TransE score  *(exercise 1 — skill: the decoder, vectorized)*

$f(h,r,t) = -\\|\\mathbf{h} + \\mathbf{r} - \\mathbf{t}\\|_2$, batched. The assert is
the lecture's scorers-figure toy — you already know both numbers.
"""),

    todo("""def transe_score(E: torch.Tensor, R: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
    \"\"\"E: (n_ent, d) entity embeddings. R: (n_rel, d). triples: (B, 3) rows of
    (h, r, t) ids. Return (B,) scores  −‖E[h] + R[r] − E[t]‖₂  (higher = truer).\"\"\"
    ### BEGIN SOLUTION
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    return -(E[h] + R[r] - E[t]).norm(dim=1)
    ### END SOLUTION


E_toy = torch.tensor([[1.0, 1.0], [3.0, 2.0], [3.0, 1.0]])   # h, t, decoy
R_toy = torch.tensor([[2.0, 1.0]])
s_true = transe_score(E_toy, R_toy, torch.tensor([[0, 0, 1]]))
s_dec = transe_score(E_toy, R_toy, torch.tensor([[0, 0, 2]]))
assert abs(s_true.item() - 0.0) < 1e-6, f"the lecture's exact hit scores 0, got {s_true.item():.3f}"
assert abs(s_dec.item() + 1.0) < 1e-6, f"the decoy at distance 1 scores −1, got {s_dec.item():.3f}"
batch = transe_score(E_toy, R_toy, torch.tensor([[0, 0, 1], [0, 0, 2]]))
assert batch.shape == (2,), "must be vectorized over the batch dimension"
print("exercise 1 ✓ — the translation decoder, hand-verified")""",
         stub="""def transe_score(E: torch.Tensor, R: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
    \"\"\"E: (n_ent, d) entity embeddings. R: (n_rel, d). triples: (B, 3) rows of
    (h, r, t) ids. Return (B,) scores  −‖E[h] + R[r] − E[t]‖₂  (higher = truer).\"\"\"
    # TODO: index, add, subtract, .norm(dim=1), negate. One line.
    raise NotImplementedError


E_toy = torch.tensor([[1.0, 1.0], [3.0, 2.0], [3.0, 1.0]])   # h, t, decoy
R_toy = torch.tensor([[2.0, 1.0]])
s_true = transe_score(E_toy, R_toy, torch.tensor([[0, 0, 1]]))
s_dec = transe_score(E_toy, R_toy, torch.tensor([[0, 0, 2]]))
assert abs(s_true.item() - 0.0) < 1e-6, f"the lecture's exact hit scores 0, got {s_true.item():.3f}"
assert abs(s_dec.item() + 1.0) < 1e-6, f"the decoy at distance 1 scores −1, got {s_dec.item():.3f}"
batch = transe_score(E_toy, R_toy, torch.tensor([[0, 0, 1], [0, 0, 2]]))
assert batch.shape == (2,), "must be vectorized over the batch dimension"
print("exercise 1 ✓ — the translation decoder, hand-verified")"""),

    md("""## 3 · Corruption and the margin  *(exercise 2 — skill: training under OWA)*

Two pieces: the corruption sampler (replace head **or** tail, uniformly), and the
margin ranking loss
$\\mathcal{L} = \\text{mean}\\,\\max(0, \\gamma - f_{\\text{pos}} + f_{\\text{neg}})$.

You are coding against a spec — the lecture's training-epoch algorithm
([Bordes et al., 2013](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)):

> **Algorithm · TransE training epoch**
>
> **Input:** triples T, entity table E, relation table R, scorer f, margin γ, learning rate η. **Output:** updated E, R.
>
> 1. **e** ← **e** / ‖**e**‖ for every entity row of E  *(project to the unit sphere — before any gradient step)*
> 2. **for** each minibatch B ⊂ T (shuffled) **do**
> 3. &nbsp;&nbsp;&nbsp;&nbsp;**for** each (h, r, t) ∈ B **do**
> 4. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(h′, r, t′) ← replace h **or** t (coin flip) by a uniform random entity
> 5. &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ℓ ← max(0, γ − f(h,r,t) + f(h′,r,t′))
> 6. &nbsp;&nbsp;&nbsp;&nbsp;**end for**
> 7. &nbsp;&nbsp;&nbsp;&nbsp;one SGD/Adam step on the mean of ℓ over B
> 8. **end for**
> 9. **return** E, R

Your `corrupt` is line 4 and your `margin_loss` is line 5; the provided
`train_transe` below wires them into the full loop, renormalization order included.
"""),

    todo("""def corrupt(triples: torch.Tensor, n_ent: int, g: torch.Generator) -> torch.Tensor:
    \"\"\"Return a copy where each row has its head OR tail (50/50) replaced by a
    uniformly random entity. Use torch.randint(..., generator=g) twice and a
    boolean side mask — no Python loops.\"\"\"
    ### BEGIN SOLUTION
    out = triples.clone()
    side = torch.randint(0, 2, (len(triples),), generator=g)
    rand_e = torch.randint(0, n_ent, (len(triples),), generator=g)
    out[side == 0, 0] = rand_e[side == 0]
    out[side == 1, 2] = rand_e[side == 1]
    return out
    ### END SOLUTION


def margin_loss(pos: torch.Tensor, neg: torch.Tensor, gamma: float) -> torch.Tensor:
    \"\"\"mean( relu( gamma − pos + neg ) ) over the batch.\"\"\"
    ### BEGIN SOLUTION
    return F.relu(gamma - pos + neg).mean()
    ### END SOLUTION


g0 = torch.Generator().manual_seed(0)
T_demo = torch.tensor([[0, 0, 1]] * 4000)
C = corrupt(T_demo, n_ent=1000, g=g0)
assert C.shape == T_demo.shape and not C.equal(T_demo), "corrupt must change rows"
head_changed = (C[:, 0] != 0)
tail_changed = (C[:, 2] != 1)
assert not (head_changed & tail_changed).any(), "corrupt ONE slot per row, never both"
frac_head = head_changed.float().mean().item()
assert 0.42 < frac_head < 0.58, (
    f"head-corruption fraction {frac_head:.2f} — should be ≈0.5 (a 50/50 side mask)"
)
assert (C[:, 1] == 0).all(), "the relation is never corrupted"

lo = margin_loss(torch.tensor([-1.0]), torch.tensor([-3.0]), gamma=1.0)
hi = margin_loss(torch.tensor([-1.0]), torch.tensor([-1.5]), gamma=1.0)
assert abs(lo.item() - 0.0) < 1e-6, "pos=−1, neg=−3, γ=1: margin satisfied → loss 0"
assert abs(hi.item() - 0.5) < 1e-6, "pos=−1, neg=−1.5, γ=1: relu(1−(−1)+(−1.5)) = 0.5"
print("exercise 2 ✓ — corruption is honest, the margin does what it says")""",
         stub="""def corrupt(triples: torch.Tensor, n_ent: int, g: torch.Generator) -> torch.Tensor:
    \"\"\"Return a copy where each row has its head OR tail (50/50) replaced by a
    uniformly random entity. Use torch.randint(..., generator=g) twice and a
    boolean side mask — no Python loops.\"\"\"
    # TODO: clone; draw a 0/1 side per row and a random entity per row; write the
    # entity into column 0 where side==0, column 2 where side==1. ~6 lines.
    raise NotImplementedError


def margin_loss(pos: torch.Tensor, neg: torch.Tensor, gamma: float) -> torch.Tensor:
    \"\"\"mean( relu( gamma − pos + neg ) ) over the batch.\"\"\"
    # TODO: one line with F.relu.
    raise NotImplementedError


g0 = torch.Generator().manual_seed(0)
T_demo = torch.tensor([[0, 0, 1]] * 4000)
C = corrupt(T_demo, n_ent=1000, g=g0)
assert C.shape == T_demo.shape and not C.equal(T_demo), "corrupt must change rows"
head_changed = (C[:, 0] != 0)
tail_changed = (C[:, 2] != 1)
assert not (head_changed & tail_changed).any(), "corrupt ONE slot per row, never both"
frac_head = head_changed.float().mean().item()
assert 0.42 < frac_head < 0.58, (
    f"head-corruption fraction {frac_head:.2f} — should be ≈0.5 (a 50/50 side mask)"
)
assert (C[:, 1] == 0).all(), "the relation is never corrupted"

lo = margin_loss(torch.tensor([-1.0]), torch.tensor([-3.0]), gamma=1.0)
hi = margin_loss(torch.tensor([-1.0]), torch.tensor([-1.5]), gamma=1.0)
assert abs(lo.item() - 0.0) < 1e-6, "pos=−1, neg=−3, γ=1: margin satisfied → loss 0"
assert abs(hi.item() - 0.5) < 1e-6, "pos=−1, neg=−1.5, γ=1: relu(1−(−1)+(−1.5)) = 0.5"
print("exercise 2 ✓ — corruption is honest, the margin does what it says")"""),

    md("""### Train TransE on the real thing *(provided — your three functions, at scale)*"""),

    code("""def train_transe(triples, n_ent, n_rel, d=100, gamma=1.0, lr=0.01,
                 epochs=3, batch=2048, seed=0):
    g = torch.Generator().manual_seed(seed)
    E = torch.nn.Parameter(torch.empty(n_ent, d).uniform_(-6 / d ** 0.5, 6 / d ** 0.5, generator=g))
    R = torch.nn.Parameter(torch.empty(n_rel, d).uniform_(-6 / d ** 0.5, 6 / d ** 0.5, generator=g))
    opt = torch.optim.Adam([E, R], lr=lr)
    for ep in range(epochs):
        with torch.no_grad():                       # paper-faithful: entities on the unit sphere
            E.data = F.normalize(E.data, dim=1)
        perm = torch.randperm(len(triples), generator=g)
        total = 0.0
        for i in range(0, len(triples), batch):
            idx = perm[i:i + batch]
            pos_t = triples[idx]
            neg_t = corrupt(pos_t, n_ent, g)
            loss = margin_loss(transe_score(E, R, pos_t), transe_score(E, R, neg_t), gamma)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(idx)
        print(f"  epoch {ep + 1}: mean margin loss {total / len(triples):.4f}")
    return E.detach(), R.detach()


EPOCHS = 2 if SMOKE else 5
print("training TransE (a few minutes on CPU)...")
E_t, R_t = train_transe(train_T, N_ENT, N_REL, epochs=EPOCHS)
print("done")"""),

    md("""## 4 · Filtered evaluation  *(exercise 3 — skill: the protocol, exactly)*

The spec, mirroring the lecture's evaluation algorithm (the protocol of
[Bordes et al., 2013](https://proceedings.neurips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html)):

> **Algorithm · Filtered ranking evaluation**
>
> **Input:** test triples S, scorer f, known-true indexes K(h,r)→{t} and K′(r,t)→{h} built from train ∪ valid ∪ test. **Output:** filtered MRR, Hits@K.
>
> 1. ranks ← ()
> 2. **for** each (h, r, t) ∈ S **do**
> 3. &nbsp;&nbsp;&nbsp;&nbsp;s_e ← f(h, r, e) for every entity e  *(tail vacancy)*
> 4. &nbsp;&nbsp;&nbsp;&nbsp;s_e ← −∞ for e ∈ K(h,r) \\\\ {t}  *(filter known truths — never the target)*
> 5. &nbsp;&nbsp;&nbsp;&nbsp;append 1 + #{e : s_e > s_t} to ranks
> 6. &nbsp;&nbsp;&nbsp;&nbsp;repeat lines 3–5 for the head vacancy, using K′
> 7. **end for**
> 8. MRR ← mean(1/ranks);  Hits@K ← mean[rank ≤ K]
> 9. **return** MRR, Hits@K

Implement the single-query filtered rank — lines 4–5. The toy assert is the
lecture's Q5: raw rank 4 with two known-true competitors above → filtered rank 2.
Then the real thing (provided plumbing runs lines 2–9 around your function): MRR and
Hits@10 over a held-out sample, both directions, filtered. Note line 5 counts
*strictly greater* scores — ties are never resolved in the model's favor.
"""),

    todo("""def filtered_rank(scores: torch.Tensor, target: int, known: set) -> int:
    \"\"\"scores: (n_ent,) model scores for every candidate in the vacant slot.
    target: the true entity id. known: ids of OTHER entities that also form true
    triples in this slot (train/valid/test) — to be excluded from the ranking.
    Returns the 1-based rank of target among the remaining candidates.\"\"\"
    ### BEGIN SOLUTION
    s = scores.clone()
    for e in known:
        if e != target:
            s[e] = -float("inf")
    return int((s > s[target]).sum().item()) + 1
    ### END SOLUTION


# the lecture's Q5, as tensors: target scores 4th raw; two known-trues above it
toy = torch.tensor([9.0, 8.0, 7.0, 5.0, 3.0, 1.0])   # candidate scores, ids 0..5
raw = filtered_rank(toy, target=3, known=set())
filt = filtered_rank(toy, target=3, known={0, 2})
assert raw == 4, f"raw rank of id 3 must be 4, got {raw}"
assert filt == 2, f"filtering out the two known-trues above → rank 2, got {filt} (Q5!)"
assert filtered_rank(toy, target=3, known={3, 0, 2}) == 2, "target itself in `known` must NOT be excluded"

# the real protocol (provided plumbing around YOUR function)
from collections import defaultdict
known_tails = defaultdict(set)
known_heads = defaultdict(set)
for split in [train_T, valid_T, test_T]:
    for h, r, t in split.tolist():
        known_tails[(h, r)].add(t)
        known_heads[(r, t)].add(h)

g_eval = torch.Generator().manual_seed(7)
sample = test_T[torch.randperm(len(test_T), generator=g_eval)[:400 if SMOKE else 2000]]
ranks = []
with torch.no_grad():
    for h, r, t in sample.tolist():
        s_tail = -(E_t[h] + R_t[r] - E_t).norm(dim=1)          # score every tail
        ranks.append(filtered_rank(s_tail, t, known_tails[(h, r)]))
        s_head = -(E_t + R_t[r] - E_t[t]).norm(dim=1)          # score every head
        ranks.append(filtered_rank(s_head, h, known_heads[(r, t)]))
ranks = np.array(ranks)
mrr = (1 / ranks).mean()
hits10 = (ranks <= 10).mean()
print(f"filtered MRR {mrr:.3f} · Hits@10 {hits10:.3f}  ({len(sample)} test facts, both directions)")
floor_mrr, floor_h = (0.05, 0.12) if SMOKE else (0.12, 0.25)
assert mrr > floor_mrr and hits10 > floor_h, (
    f"MRR {mrr:.3f} / Hits@10 {hits10:.3f} under the floor — with the given recipe "
    f"TransE lands well above this. Check that `known` uses the (h,r)/(r,t) key and "
    f"that you excluded known competitors but never the target."
)
print("exercise 3 ✓ — for calibration: well-tuned TransE reaches ≈0.29 MRR; we trained minutes, not hours")""",
         stub="""def filtered_rank(scores: torch.Tensor, target: int, known: set) -> int:
    \"\"\"scores: (n_ent,) model scores for every candidate in the vacant slot.
    target: the true entity id. known: ids of OTHER entities that also form true
    triples in this slot (train/valid/test) — to be excluded from the ranking.
    Returns the 1-based rank of target among the remaining candidates.\"\"\"
    # TODO: set known ids (except the target!) to −inf, then rank = 1 + #{scores
    # strictly greater than the target's}. ~5 lines.
    raise NotImplementedError


# the lecture's Q5, as tensors: target scores 4th raw; two known-trues above it
toy = torch.tensor([9.0, 8.0, 7.0, 5.0, 3.0, 1.0])   # candidate scores, ids 0..5
raw = filtered_rank(toy, target=3, known=set())
filt = filtered_rank(toy, target=3, known={0, 2})
assert raw == 4, f"raw rank of id 3 must be 4, got {raw}"
assert filt == 2, f"filtering out the two known-trues above → rank 2, got {filt} (Q5!)"
assert filtered_rank(toy, target=3, known={3, 0, 2}) == 2, "target itself in `known` must NOT be excluded"

# the real protocol (provided plumbing around YOUR function)
from collections import defaultdict
known_tails = defaultdict(set)
known_heads = defaultdict(set)
for split in [train_T, valid_T, test_T]:
    for h, r, t in split.tolist():
        known_tails[(h, r)].add(t)
        known_heads[(r, t)].add(h)

g_eval = torch.Generator().manual_seed(7)
sample = test_T[torch.randperm(len(test_T), generator=g_eval)[:400 if SMOKE else 2000]]
ranks = []
with torch.no_grad():
    for h, r, t in sample.tolist():
        s_tail = -(E_t[h] + R_t[r] - E_t).norm(dim=1)          # score every tail
        ranks.append(filtered_rank(s_tail, t, known_tails[(h, r)]))
        s_head = -(E_t + R_t[r] - E_t[t]).norm(dim=1)          # score every head
        ranks.append(filtered_rank(s_head, h, known_heads[(r, t)]))
ranks = np.array(ranks)
mrr = (1 / ranks).mean()
hits10 = (ranks <= 10).mean()
print(f"filtered MRR {mrr:.3f} · Hits@10 {hits10:.3f}  ({len(sample)} test facts, both directions)")
floor_mrr, floor_h = (0.05, 0.12) if SMOKE else (0.12, 0.25)
assert mrr > floor_mrr and hits10 > floor_h, (
    f"MRR {mrr:.3f} / Hits@10 {hits10:.3f} under the floor — with the given recipe "
    f"TransE lands well above this. Check that `known` uses the (h,r)/(r,t) key and "
    f"that you excluded known competitors but never the target."
)
print("exercise 3 ✓ — for calibration: well-tuned TransE reaches ≈0.29 MRR; we trained minutes, not hours")"""),

    md("""## 5 · The RotatE score  *(exercise 4 — skill: rotations, in tensors)*

Entities are complex vectors; relations are pure phases. Store entities as complex
tensors and relations as real phase angles $\\theta$, with
$\\mathbf{r} = e^{i\\theta}$ built by `torch.polar`. The assert is the lecture's
quarter-turn: $h$ at 45°, rotate by 90°, land exactly on $t$ at 135°.
"""),

    todo("""def rotate_score(E: torch.Tensor, PHASE: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
    \"\"\"E: (n_ent, d) COMPLEX entity embeddings. PHASE: (n_rel, d) real angles.
    Return (B,) scores  −‖E[h] ∘ e^{iθ_r} − E[t]‖  (norm over complex coords).\"\"\"
    ### BEGIN SOLUTION
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    rot = torch.polar(torch.ones_like(PHASE[r]), PHASE[r])
    return -(E[h] * rot - E[t]).abs().pow(2).sum(dim=1).sqrt()
    ### END SOLUTION


E_c = torch.stack([
    torch.polar(torch.ones(1), torch.tensor([math.pi / 4]))[0:1].flatten(),      # h at 45°
    torch.polar(torch.ones(1), torch.tensor([3 * math.pi / 4]))[0:1].flatten(),  # t at 135°
    torch.polar(torch.ones(1), torch.tensor([0.0]))[0:1].flatten(),              # decoy at 0°
])
PH = torch.tensor([[math.pi / 2]])                                               # rotate 90°
s_hit = rotate_score(E_c, PH, torch.tensor([[0, 0, 1]]))
s_miss = rotate_score(E_c, PH, torch.tensor([[0, 0, 2]]))
assert abs(s_hit.item()) < 1e-6, f"45° + 90° lands exactly on 135°: score 0, got {s_hit.item():.4f}"
assert s_miss.item() < -0.5, "the decoy at 0° must score clearly worse"
print("exercise 4 ✓ — the rotation decoder, hand-verified on the lecture's quarter-turn")""",
         stub="""def rotate_score(E: torch.Tensor, PHASE: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
    \"\"\"E: (n_ent, d) COMPLEX entity embeddings. PHASE: (n_rel, d) real angles.
    Return (B,) scores  −‖E[h] ∘ e^{iθ_r} − E[t]‖  (norm over complex coords).\"\"\"
    # TODO: rot = torch.polar(ones, PHASE[r]); difference E[h]*rot − E[t];
    # complex norm = .abs()² summed then sqrt; negate. ~3 lines.
    raise NotImplementedError


E_c = torch.stack([
    torch.polar(torch.ones(1), torch.tensor([math.pi / 4]))[0:1].flatten(),      # h at 45°
    torch.polar(torch.ones(1), torch.tensor([3 * math.pi / 4]))[0:1].flatten(),  # t at 135°
    torch.polar(torch.ones(1), torch.tensor([0.0]))[0:1].flatten(),              # decoy at 0°
])
PH = torch.tensor([[math.pi / 2]])                                               # rotate 90°
s_hit = rotate_score(E_c, PH, torch.tensor([[0, 0, 1]]))
s_miss = rotate_score(E_c, PH, torch.tensor([[0, 0, 2]]))
assert abs(s_hit.item()) < 1e-6, f"45° + 90° lands exactly on 135°: score 0, got {s_hit.item():.4f}"
assert s_miss.item() < -0.5, "the decoy at 0° must score clearly worse"
print("exercise 4 ✓ — the rotation decoder, hand-verified on the lecture's quarter-turn")"""),

    md("""## 6 · The showdown: theory, caught red-handed  *(exercise 5)*

A synthetic KG with two relations on 12 entities: **married** (four symmetric pairs,
both directions stated) and **manages** (a strict chain — antisymmetric). We train
TransE and RotatE side by side (provided).
[The TransE-symmetry proposition from the lecture](https://lukmanovr.github.io/dkr/lectures/04-knowledge-graphs.html#prp-transe-sym)
predicts TransE must drive
$\\|\\mathbf{r}_{married}\\|$ toward zero while keeping $\\|\\mathbf{r}_{manages}\\|$
healthy; RotatE's Table-1 fix predicts the *married* phases park near 0 or π. Your
part: extract the evidence.
"""),

    todo("""MARRIED, MANAGES = 0, 1
pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
syn = [(a, MARRIED, b) for a, b in pairs] + [(b, MARRIED, a) for a, b in pairs]
syn += [(i, MANAGES, i + 1) for i in range(8, 11)] + [(0, MANAGES, 8), (2, MANAGES, 9)]
SYN = torch.tensor(syn)
N_SYN = 12

# — provided: train both models on the same facts
g = torch.Generator().manual_seed(2)
E_s = torch.nn.Parameter(0.5 * torch.randn(N_SYN, 8, generator=g))
R_s = torch.nn.Parameter(0.5 * torch.randn(2, 8, generator=g))
opt = torch.optim.Adam([E_s, R_s], lr=0.02)
for step in range(8000):
    with torch.no_grad():                 # entities on the unit sphere, per the paper —
        E_s.data = F.normalize(E_s.data, dim=1)   # unbounded space lets the margin go
    neg = corrupt(SYN, N_SYN, g)                  # slack before the crush shows
    loss = margin_loss(transe_score(E_s, R_s, SYN), transe_score(E_s, R_s, neg), 2.0)
    opt.zero_grad(); loss.backward(); opt.step()

E_re = torch.nn.Parameter(0.5 * torch.randn(N_SYN, 4, generator=g))
E_im = torch.nn.Parameter(0.5 * torch.randn(N_SYN, 4, generator=g))
PH_s = torch.nn.Parameter(2 * math.pi * torch.rand(2, 4, generator=g))
opt2 = torch.optim.Adam([E_re, E_im, PH_s], lr=0.02)
for step in range(3000):
    E_cplx = torch.complex(E_re, E_im)
    neg = corrupt(SYN, N_SYN, g)
    loss = margin_loss(rotate_score(E_cplx, PH_s, SYN), rotate_score(E_cplx, PH_s, neg), 1.0)
    opt2.zero_grad(); loss.backward(); opt2.step()


def transe_evidence(R: torch.Tensor) -> float:
    \"\"\"Return ‖r_married‖ / ‖r_manages‖ — the theory says this ratio collapses.\"\"\"
    ### BEGIN SOLUTION
    return (R[MARRIED].norm() / R[MANAGES].norm()).item()
    ### END SOLUTION


def rotate_evidence(PHASE: torch.Tensor) -> float:
    \"\"\"Return mean |sin θ| over the married relation's phases — near 0 iff every
    phase sits near 0 or π, i.e. the rotation is (per coordinate) a half-turn or
    the identity: exactly the symmetric solutions of Table 1.\"\"\"
    ### BEGIN SOLUTION
    return PHASE[MARRIED].sin().abs().mean().item()
    ### END SOLUTION


ratio = transe_evidence(R_s.detach())
phase_dev = rotate_evidence(PH_s.detach())
manage_dev = PH_s.detach()[MANAGES].sin().abs().mean().item()
print(f"TransE:  ‖r_married‖ / ‖r_manages‖ = {ratio:.3f}   (theory: → 0)")
print(f"RotatE:  mean|sin θ_married| = {phase_dev:.3f} vs mean|sin θ_manages| = {manage_dev:.3f}")
assert ratio < 0.4, (
    f"ratio {ratio:.3f} — TransE should have crushed the married vector "
    f"(@prp-transe-sym: symmetry forces r → 0). More steps or check the loss wiring."
)
assert phase_dev < 0.3, (
    f"married phases deviate from {{0, π}} by {phase_dev:.3f} on average — RotatE "
    f"should place symmetric relations at half-turns/identity (RotatE paper, Fig. 3)."
)
assert manage_dev > phase_dev + 0.1, (
    "the antisymmetric chain relation should keep its phases clearly AWAY from {0, π}"
)
print("exercise 5 ✓ — the cheat sheet is not advice, it is prophecy")""",
         stub="""MARRIED, MANAGES = 0, 1
pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
syn = [(a, MARRIED, b) for a, b in pairs] + [(b, MARRIED, a) for a, b in pairs]
syn += [(i, MANAGES, i + 1) for i in range(8, 11)] + [(0, MANAGES, 8), (2, MANAGES, 9)]
SYN = torch.tensor(syn)
N_SYN = 12

# — provided: train both models on the same facts
g = torch.Generator().manual_seed(2)
E_s = torch.nn.Parameter(0.5 * torch.randn(N_SYN, 8, generator=g))
R_s = torch.nn.Parameter(0.5 * torch.randn(2, 8, generator=g))
opt = torch.optim.Adam([E_s, R_s], lr=0.02)
for step in range(8000):
    with torch.no_grad():                 # entities on the unit sphere, per the paper —
        E_s.data = F.normalize(E_s.data, dim=1)   # unbounded space lets the margin go
    neg = corrupt(SYN, N_SYN, g)                  # slack before the crush shows
    loss = margin_loss(transe_score(E_s, R_s, SYN), transe_score(E_s, R_s, neg), 2.0)
    opt.zero_grad(); loss.backward(); opt.step()

E_re = torch.nn.Parameter(0.5 * torch.randn(N_SYN, 4, generator=g))
E_im = torch.nn.Parameter(0.5 * torch.randn(N_SYN, 4, generator=g))
PH_s = torch.nn.Parameter(2 * math.pi * torch.rand(2, 4, generator=g))
opt2 = torch.optim.Adam([E_re, E_im, PH_s], lr=0.02)
for step in range(3000):
    E_cplx = torch.complex(E_re, E_im)
    neg = corrupt(SYN, N_SYN, g)
    loss = margin_loss(rotate_score(E_cplx, PH_s, SYN), rotate_score(E_cplx, PH_s, neg), 1.0)
    opt2.zero_grad(); loss.backward(); opt2.step()


def transe_evidence(R: torch.Tensor) -> float:
    \"\"\"Return ‖r_married‖ / ‖r_manages‖ — the theory says this ratio collapses.\"\"\"
    # TODO: one line of norms.
    raise NotImplementedError


def rotate_evidence(PHASE: torch.Tensor) -> float:
    \"\"\"Return mean |sin θ| over the married relation's phases — near 0 iff every
    phase sits near 0 or π, i.e. the rotation is (per coordinate) a half-turn or
    the identity: exactly the symmetric solutions of Table 1.\"\"\"
    # TODO: one line: sin, abs, mean.
    raise NotImplementedError


ratio = transe_evidence(R_s.detach())
phase_dev = rotate_evidence(PH_s.detach())
manage_dev = PH_s.detach()[MANAGES].sin().abs().mean().item()
print(f"TransE:  ‖r_married‖ / ‖r_manages‖ = {ratio:.3f}   (theory: → 0)")
print(f"RotatE:  mean|sin θ_married| = {phase_dev:.3f} vs mean|sin θ_manages| = {manage_dev:.3f}")
assert ratio < 0.4, (
    f"ratio {ratio:.3f} — TransE should have crushed the married vector "
    f"(@prp-transe-sym: symmetry forces r → 0). More steps or check the loss wiring."
)
assert phase_dev < 0.3, (
    f"married phases deviate from {{0, π}} by {phase_dev:.3f} on average — RotatE "
    f"should place symmetric relations at half-turns/identity (RotatE paper, Fig. 3)."
)
assert manage_dev > phase_dev + 0.1, (
    "the antisymmetric chain relation should keep its phases clearly AWAY from {0, π}"
)
print("exercise 5 ✓ — the cheat sheet is not advice, it is prophecy")"""),

    md("""## 7 · Stretch (optional, ungraded)

1. **RotatE at scale.** Swap `rotate_score` into the FB15k-237 training loop (keep
   entities complex; you will need a phase parameterization and a modulus constraint
   or none — discuss which). Does it beat your TransE MRR at equal budget?
2. **Self-adversarial.** Replace uniform corruption with score-weighted negatives
   (temperature α = 1) and measure the MRR difference at fixed epochs.
3. **Pattern audit.** Find the most *symmetric* relation in FB15k-237 empirically
   (highest fraction of reversed pairs also present) and check whether your TransE
   gave it a smaller-than-average ‖r‖ — the showdown, in the wild.

## 8 · Reflection (answer in this cell, 2–4 sentences each)

**R1.** Your exercise-3 numbers came from minutes of training, and well-tuned TransE
roughly doubles them. For *which claims* in this lab does that gap matter, and for
which does it not? (Think: theory demos vs. leaderboard comparisons.)

**R2.** The corruption sampler will occasionally manufacture a TRUE fact as a
negative (the lecture's ⚠). Estimate roughly how often on FB15k-237 tail-corruption
(hint: average number of true tails per (h, r) is ≈ 272k/⟨distinct (h,r)⟩ — a couple),
and explain why the damage stays bounded.

**R3.** The showdown proves the cheat sheet with 12 entities. Name one reason the
same diagnosis is harder to read off a real 15k-entity model, and one measurement you
would trust anyway (stretch 3 is a hint).

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
