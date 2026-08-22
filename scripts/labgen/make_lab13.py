#!/usr/bin/env python
"""Generate Lab 13 (RWSE from the definition with exact hand asserts, the CSL
crack, the honest ZINC ladder at a student budget, and the GPS block with the
padding-mask tripwire) — student + solution notebooks.

    python scripts/labgen/make_lab13.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "labs" / "lab13_transformers.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "labs" / "lab13_transformers.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(solution, stub): return ("code", solution, stub)


RWSE_SOL = '''def rwse_diag(A, K):
    """Random-walk structural encoding from the definition: for each node i,
    the vector [(M^1)_ii, (M^2)_ii, ..., (M^K)_ii] with M = D^{-1} A
    (row-normalized adjacency). A: (n, n) dense array. Returns (n, K)."""
    ### BEGIN SOLUTION
    A = np.asarray(A, dtype=float)
    deg = A.sum(1).clip(min=1)
    M = A / deg[:, None]
    P = np.eye(len(A))
    cols = []
    for _ in range(K):
        P = P @ M
        cols.append(np.diag(P).copy())
    return np.stack(cols, axis=1)
    ### END SOLUTION


# the lecture's triangle-with-tail: triangle 0-1-2, tail edge 2-3
A_tt = np.zeros((4, 4))
for a, b in [(0, 1), (1, 2), (2, 0), (2, 3)]:
    A_tt[a, b] = A_tt[b, a] = 1.0
R_tt = rwse_diag(A_tt, 4)

assert R_tt.shape == (4, 4), f"expected (4, 4), got {R_tt.shape}"
assert np.abs(R_tt[:, 0]).max() < 1e-12, (
    "k=1 returns must all be 0 — no self-loops means no 1-step return"
)
assert np.allclose(R_tt[:, 1], [5 / 12, 5 / 12, 2 / 3, 1 / 3], atol=1e-9), (
    f"k=2 must be [5/12, 5/12, 2/3, 1/3] (lecture hand table) — got {np.round(R_tt[:, 1], 4)}"
    " — did you row-normalize (divide each row by ITS degree)?"
)
assert np.allclose(R_tt[:3, 2], 1 / 6, atol=1e-9), (
    f"k=3 for the three triangle members must be exactly 1/6 — got {np.round(R_tt[:3, 2], 4)}"
)
assert abs(R_tt[3, 2]) < 1e-12, (
    f"the tail node's k=3 return must be EXACTLY zero (a 3-step return needs a "
    f"3-cycle, and node 3 has none) — got {R_tt[3, 2]:.2e}"
)
assert np.allclose(R_tt[:, 3], [0.2986, 0.2986, 0.5278, 0.2222], atol=5e-4), (
    f"k=4 must match the lecture table [0.2986, 0.2986, 0.5278, 0.2222] — got {np.round(R_tt[:, 3], 4)}"
)
assert abs(R_tt[3, 3] - 2 / 9) < 1e-9, (
    "the tail's k=4 return is exactly 2/9 — the hub's own 2-step rate, rented "
    "(lecture derivation)"
)
print("exercise 1 \\u2713 — your RWSE matches the hand table, zero included")'''

RWSE_STUB = RWSE_SOL.replace('''    ### BEGIN SOLUTION
    A = np.asarray(A, dtype=float)
    deg = A.sum(1).clip(min=1)
    M = A / deg[:, None]
    P = np.eye(len(A))
    cols = []
    for _ in range(K):
        P = P @ M
        cols.append(np.diag(P).copy())
    return np.stack(cols, axis=1)
    ### END SOLUTION''',
'''    # TODO (~7 lines): row-normalize A by its degrees (clip degrees at 1 to
    # avoid dividing by zero on isolated nodes); repeatedly multiply a running
    # power P by M, collecting np.diag(P) after each multiply; stack the K
    # diagonals into columns of an (n, K) array.
    raise NotImplementedError''')


SEP_SOL = '''def first_separating_k(Ra, Rb, tol=1e-8):
    """Smallest k (1-based) at which two graphs' RWSE fingerprint MULTISETS
    differ (sort each column before comparing — node order must not matter);
    None if no k separates them."""
    ### BEGIN SOLUTION
    for k in range(Ra.shape[1]):
        a, b = np.sort(Ra[:, k]), np.sort(Rb[:, k])
        if not np.allclose(a, b, atol=tol):
            return k + 1
    return None
    ### END SOLUTION


R2 = rwse_diag(nx.to_numpy_array(csl2, nodelist=range(11)), 8)
R3 = rwse_diag(nx.to_numpy_array(csl3, nodelist=range(11)), 8)

# vertex-transitivity: within each graph, every node has the SAME fingerprint
assert np.allclose(R2, R2[0], atol=1e-9) and np.allclose(R3, R3[0], atol=1e-9), (
    "every node of a vertex-transitive graph must have an identical fingerprint"
)
# the crack, on your numbers: 6 closed 3-walks over 4^3 vs none at all
assert abs(R2[0, 2] - 6 / 64) < 1e-9, (
    f"CSL(11,2) k=3 return must be 6/64 = 0.09375 (three triangles per node, "
    f"two directions each, over 4^3 walks) — got {R2[0, 2]:.5f}"
)
assert abs(R3[0, 2]) < 1e-12, (
    f"CSL(11,3) k=3 return must be EXACTLY zero (no steps from {{\\u00b11, \\u00b13}} "
    f"sum to 0 mod 11 in three moves) — got {R3[0, 2]:.2e}"
)
k_sep = first_separating_k(R2, R3)
assert k_sep == 3, (
    f"the graphs 1-WL cannot separate must split at k=3 — your separator says {k_sep}"
)
print(f"WL said identical; RWSE separates at k={k_sep}: "
      f"{R2[0, 2]:.5f} vs {R3[0, 2]:.5f}")

# the threshold probe: classify the triangle-with-tail by ONE entry
members = {i for i in range(4) if R_tt[i, 2] > 0.08}
assert members == {0, 1, 2}, (
    f"thresholding the k=3 entry at 0.08 must pick out exactly the triangle "
    f"members — got {sorted(members)}"
)
print("exercise 2 \\u2713 — the 1-WL ceiling, walked over with a feature")'''

SEP_STUB = SEP_SOL.replace('''    ### BEGIN SOLUTION
    for k in range(Ra.shape[1]):
        a, b = np.sort(Ra[:, k]), np.sort(Rb[:, k])
        if not np.allclose(a, b, atol=tol):
            return k + 1
    return None
    ### END SOLUTION''',
'''    # TODO (~5 lines): for each column k, sort both graphs' columns and
    # compare with np.allclose at tol; return k + 1 (1-based) at the first
    # mismatch; return None if the loop completes.
    raise NotImplementedError''')


ATTACH_SOL = '''def attach_rwse(ds, K=16):
    """For each PyG molecule: dense adjacency from edge_index, rwse_diag,
    stored as d.rwse — a float32 tensor of shape (num_nodes, K)."""
    out = []
    for d in ds:
        ### BEGIN SOLUTION
        A = np.zeros((d.num_nodes, d.num_nodes))
        A[d.edge_index[0].numpy(), d.edge_index[1].numpy()] = 1.0
        d.rwse = torch.tensor(rwse_diag(A, K), dtype=torch.float32)
        ### END SOLUTION
        out.append(d)
    return out


t0 = time.time()
tr_l, va_l, te_l = attach_rwse(tr_list), attach_rwse(va_list), attach_rwse(te_list)
print(f"RWSE attached to {len(tr_l) + len(va_l) + len(te_l)} molecules "
      f"in {time.time() - t0:.0f}s — offline, once, cacheable forever (Week 11)")

m = tr_l[0]
assert hasattr(m, "rwse") and m.rwse.shape == (m.num_nodes, 16), "wrong shape"
assert m.rwse.dtype == torch.float32, "store float32 — it feeds a Linear layer"
assert m.rwse[:, 0].abs().max() < 1e-6, (
    "molecules have no self-loops, so every k=1 return must be 0 — check that "
    "you built A from edge_index, not from A + I"
)
assert 0.0 <= m.rwse.min() and m.rwse.max() <= 1.0, "returns are probabilities"
print("exercise 3a \\u2713 — the encoding is wired into the pipeline")'''

ATTACH_STUB = ATTACH_SOL.replace('''        ### BEGIN SOLUTION
        A = np.zeros((d.num_nodes, d.num_nodes))
        A[d.edge_index[0].numpy(), d.edge_index[1].numpy()] = 1.0
        d.rwse = torch.tensor(rwse_diag(A, K), dtype=torch.float32)
        ### END SOLUTION''',
'''        # TODO (~3 lines): zeros (n, n); set A[src, dst] = 1 from
        # d.edge_index (it already stores both directions); d.rwse =
        # torch.tensor(rwse_diag(A, K), dtype=torch.float32).
        raise NotImplementedError''')


GLOBAL_SOL = '''def global_channel(block, x, batch, sabotage_mask=False):
    """The GPS global channel: densify the batch, attend with the padding
    mask, re-flatten to (num_nodes, h). With sabotage_mask=True, attention
    sees the padding — the bug, kept on a switch so the asserts can price it."""
    ### BEGIN SOLUTION
    dense, mask = to_dense_batch(x, batch)
    kpm = None if sabotage_mask else ~mask
    att, _ = block.mha(dense, dense, dense, key_padding_mask=kpm)
    return att[mask]
    ### END SOLUTION


# --- the tripwire pair, on an untrained net (correctness needs no training) ---
torch.manual_seed(1)
probe = Net(attn=True, rwse=True).to(DEV).eval()
small = te_l[0]
big = max(te_l[:50], key=lambda d: d.num_nodes)
print(f"probe molecules: {small.num_nodes} atoms alone vs batched next to "
      f"{big.num_nodes} atoms ({big.num_nodes - small.num_nodes} phantom rows of padding)")
with torch.no_grad():
    lone = next(iter(DataLoader([small], batch_size=1))).to(DEV)
    pair = next(iter(DataLoader([small, big], batch_size=2))).to(DEV)
    r_lone = probe.node_reps(lone)
    r_pair = probe.node_reps(pair)[pair.batch == 0]
    r_sab = probe.node_reps(pair, sabotage_mask=True)[pair.batch == 0]

diff_ok = (r_lone - r_pair).abs().max().item()
diff_sab = (r_lone - r_sab).abs().max().item()
print(f"masked: {diff_ok:.2e} · mask sabotaged: {diff_sab:.2e}")
assert diff_ok < 1e-3, (
    f"adding phantom padding nodes must NOT move a real node's output — got "
    f"max diff {diff_ok:.2e}. Is key_padding_mask=~mask (True marks PADDING, "
    f"so the mask from to_dense_batch must be inverted)?"
)
assert diff_sab > 1e-2, (
    f"with the mask sabotaged, outputs MUST move (attention now reads phantom "
    f"atoms) — got {diff_sab:.2e}; your sabotage branch may still be masking"
)
print("exercise 4 \\u2713 — the genre's quietest bug, priced: "
      f"{diff_sab:.2f} of silent damage, caught by a two-line assert")'''

GLOBAL_STUB = GLOBAL_SOL.replace('''    ### BEGIN SOLUTION
    dense, mask = to_dense_batch(x, batch)
    kpm = None if sabotage_mask else ~mask
    att, _ = block.mha(dense, dense, dense, key_padding_mask=kpm)
    return att[mask]
    ### END SOLUTION''',
'''    # TODO (~4 lines): to_dense_batch(x, batch) -> (dense (B, n_max, h),
    # mask (B, n_max) with True on REAL nodes); key_padding_mask wants True on
    # PADDING, so pass ~mask (or None when sabotaging); run
    # block.mha(dense, dense, dense, key_padding_mask=...); return att[mask].
    raise NotImplementedError''')


CELLS = [
    md("""# Lab 13 · The graph re-enters as features

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/labs/lab13_transformers.ipynb)

**Week 13 · [lecture](https://lukmanovr.github.io/dkr/lectures/13-transformers.html) · ≈ 10 min of compute (CPU-friendly; GPU shaves the ZINC runs)**

You implement RWSE from the definition and hit the lecture's hand table
exactly — the tail node's zero included; you separate the CSL pair that 1-WL
provably cannot; you wire the encoding into a real molecule pipeline and run
the ladder at your own budget; and you build the GPS global channel, where a
pair of asserts turns the padding-mask bug into a number.

### Goals
1. `rwse_diag` matching the triangle-with-tail hand table at $10^{-9}$.
2. Separate CSL(11,2) from CSL(11,3) at $k=3$: 0.09375 vs exactly 0.
3. Wire RWSE into ZINC; train MPNN-only and MPNN+RWSE under one protocol.
4. Implement `global_channel`; pass the phantom-padding invariance assert
   AND the sabotage assert.
"""),

    md("""## 0 · Setup

ZINC-subset (the community's 12k-molecule slice): constrained-solubility
regression, graded by MAE. Same data, same protocol shape as the lecture's
table — only the budget differs, deliberately."""),

    code("""import os, sys, time

SMOKE = os.environ.get("SMOKE", "") == "1"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    %pip install -q torch_geometric==2.8.0.post1

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.utils import to_dense_batch

torch.manual_seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tr_list = list(ZINC("data/ZINC", subset=True, split="train"))
va_list = list(ZINC("data/ZINC", subset=True, split="val"))
te_list = list(ZINC("data/ZINC", subset=True, split="test"))
print(f"ZINC subset: {len(tr_list)}/{len(va_list)}/{len(te_list)} molecules "
      f"\\u00b7 device {DEV} \\u00b7 SMOKE={SMOKE}")"""),

    md("""## 1 · RWSE from the definition  *(exercise 1 · build the fingerprint)*

The definition is three lines of math: $M = D^{-1}A$, take powers, read the
diagonal. Your implementation must reproduce the lecture's hand table on the
triangle-with-tail **exactly** — this is Checklist-G ground truth a human can
hold, and it is what makes every later use of `rwse_diag` trustworthy.

**Predict before you run:** what must entry $R[3, k{=}3]$ (the tail node's
3-step return) be, and what graph fact forces it? Commit before executing."""),

    todo(RWSE_SOL, RWSE_STUB),

    md("""## 2 · The CSL crack  *(exercise 2 · separate the inseparable)*

Week 9's stress pair: $\\mathrm{CSL}(11,2)$ and $\\mathrm{CSL}(11,3)$, both
4-regular, both vertex-transitive, provably identical to 1-WL and therefore
to every MPNN. First watch WL fail — the refinement below converges to one
color per graph, the same color histogram both sides. Then crack the pair
with the third entry of your own encoding."""),

    code("""def wl_colors(G, rounds=4):
    col = {v: 0 for v in G}
    for _ in range(rounds):
        sig = {v: (col[v], tuple(sorted(col[u] for u in G[v]))) for v in G}
        table = {}
        col = {v: table.setdefault(sig[v], len(table)) for v in G}
    return col


csl2 = nx.circulant_graph(11, [1, 2])
csl3 = nx.circulant_graph(11, [1, 3])
c2, c3 = wl_colors(csl2), wl_colors(csl3)
assert len(set(c2.values())) == 1 and len(set(c3.values())) == 1, \\
    "WL must assign every node of each CSL graph the same color"
print("1-WL verdict: one color each, identical histograms — 'the same graph'.")
print("Now overrule it with a feature the architecture cannot compute:")"""),

    todo(SEP_SOL, SEP_STUB),

    md("""## 3 · The ladder, at your budget  *(exercise 3 · wire it in, then measure)*

Two runs under one protocol — MPNN-only, then MPNN+RWSE — at a budget a lab
can afford (12 epochs full mode, a 2-epoch slice under SMOKE). The lecture's
100-epoch table is provided below as constants; your claims paragraph will
compare against it.

First, the wiring: your `rwse_diag`, applied to every molecule, offline
(Week 11's precompute discipline)."""),

    todo(ATTACH_SOL, ATTACH_STUB),

    md("""The model and harness are given — your effort goes to the ideas, not the
training loop (the local channel is a GINE convolution so bond types are
read; the attention channel and `node_reps` exist for exercise 4; the
`global_channel` function it calls is *yours to write there*)."""),

    code("""class Block(torch.nn.Module):
    \"\"\"One GPS-ish layer: local GINE + (optional) global attention, summed
    into a shared residual stream, LayerNormed — the lecture's block.\"\"\"
    def __init__(self, h, attn):
        super().__init__()
        self.attn = attn
        self.conv = GINEConv(torch.nn.Sequential(
            torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, h)))
        if attn:
            self.mha = torch.nn.MultiheadAttention(h, 4, batch_first=True)
            self.ln = torch.nn.LayerNorm(h)
        self.ffn = torch.nn.Sequential(torch.nn.Linear(h, 2 * h), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * h, h))
        self.ln2 = torch.nn.LayerNorm(h)

    def forward(self, x, ei, ea, batch, sabotage_mask=False):
        local = self.conv(x, ei, ea)
        if self.attn:
            glob = global_channel(self, x, batch, sabotage_mask)  # exercise 4
            x = self.ln(x + local + glob)
        else:
            x = x + local
        return self.ln2(x + self.ffn(x))


class Net(torch.nn.Module):
    def __init__(self, h=64, layers=4, attn=False, rwse=False):
        super().__init__()
        self.rwse = rwse
        self.emb = torch.nn.Embedding(28, h)     # 28 atom types
        self.eemb = torch.nn.Embedding(4, h)     # 4 bond types
        if rwse:
            self.rw_lin = torch.nn.Linear(16, h)  # RWSE enters HERE — summed in
        self.blocks = torch.nn.ModuleList([Block(h, attn) for _ in range(layers)])
        self.head = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.ReLU(),
                                        torch.nn.Linear(h, 1))

    def node_reps(self, d, sabotage_mask=False):
        x = self.emb(d.x.squeeze(-1))
        if self.rwse:
            x = x + self.rw_lin(d.rwse)
        ea = self.eemb(d.edge_attr)
        for b in self.blocks:
            x = b(x, d.edge_index, ea, d.batch, sabotage_mask)
        return x

    def forward(self, d, sabotage_mask=False):
        x = self.node_reps(d, sabotage_mask)
        return self.head(global_add_pool(x, d.batch)).squeeze(-1)


def run(attn, rwse, epochs, train_set, name=""):
    torch.manual_seed(0)
    net = Net(attn=attn, rwse=rwse).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tl = DataLoader(train_set, batch_size=128, shuffle=True)
    t0 = time.time()
    best_va, best_te = 1e9, 1e9
    for ep in range(epochs):
        net.train()
        for batch in tl:
            batch = batch.to(DEV)
            loss = F.l1_loss(net(batch), batch.y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        net.eval()
        maes = []
        for split in (va_l, te_l):
            errs = []
            with torch.no_grad():
                for b in DataLoader(split, batch_size=256):
                    b = b.to(DEV)
                    errs.append((net(b) - b.y).abs().cpu())
            maes.append(torch.cat(errs).mean().item())
        if maes[0] < best_va:
            best_va, best_te = maes[0], maes[1]
    dt = time.time() - t0
    print(f"{name}: val {best_va:.3f} \\u00b7 TEST MAE {best_te:.3f}  [{dt:.0f}s]")
    return round(best_te, 3), round(dt)


# the lecture's 100-epoch table (measured, same protocol shape) — for claims
LECTURE_100EP = {"MPNN": (0.349, 188), "MPNN+RWSE": (0.296, 194),
                 "GPS-lite": (0.303, 529)}
TUNED_REF = 0.070  # literature, ~10x schedules + polish — a DIFFERENT budget
print("harness ready")"""),

    md("""**Predict before you run:** the lecture's 100-epoch gap was 0.053 in
RWSE's favor. At 12 epochs, will the gap be bigger, smaller, or gone —
and why? (Think about what a cosine schedule has and hasn't done by
epoch 12.)"""),

    code("""EPOCHS = 2 if SMOKE else 12
TRAIN = tr_l[:1500] if SMOKE else tr_l

mpnn_te, mpnn_s = run(False, False, EPOCHS, TRAIN, "MPNN-only")
rwse_te, rwse_s = run(False, True, EPOCHS, TRAIN, "MPNN + RWSE")

CEIL = 2.0 if SMOKE else 0.60
for name, v in [("MPNN", mpnn_te), ("MPNN+RWSE", rwse_te)]:
    assert 0.15 < v < CEIL, (
        f"{name} test MAE {v} outside the sanity band (reference machine: "
        f"0.488 / 0.477 at 12 epochs; SMOKE ~1.2-1.5) — check the wiring"
    )

print(f"\\nyour {EPOCHS}-epoch ladder:  MPNN {mpnn_te} \\u00b7 +RWSE {rwse_te}")
print(f"lecture's 100-epoch table: {LECTURE_100EP}  (tuned lit ref \\u2248 {TUNED_REF})")
print("exercise 3 \\u2713 — measured, not asserted into an ordering: at 12 "
      "epochs the two rungs are near-tied (reference: 0.488 vs 0.477). The "
      "0.053 gap needed the FULL budget to emerge — budget moves everything, "
      "including ablations. Say this in your claims paragraph.")"""),

    md("""## 4 · The GPS block and its tripwire  *(exercise 4 · build the global channel)*

The global channel is four lines — densify, mask, attend, re-flatten — and
one of those lines is the genre's quietest bug. `to_dense_batch` pads every
molecule to the batch's largest; attention will happily read the phantom
atoms unless `key_padding_mask` says otherwise; the model still trains,
merely worse, and nothing crashes.

The harness below prices the bug: it runs one molecule alone, then batched
next to a bigger molecule (phantom padding rows appear), and demands your
real outputs not move; then it flips `sabotage_mask=True` and demands they
DO move.

**Predict before you run:** roughly how large will the sabotage diff be,
given outputs are LayerNormed (so O(1) scale)?"""),

    todo(GLOBAL_SOL, GLOBAL_STUB),

    md("""With the channel correct, train the full hybrid briefly — enough to watch
the loss fall, not enough to claim a ranking (that discipline is the
lecture's Figure 4)."""),

    code("""GPS_EPOCHS = 2 if SMOKE else 3
GPS_TRAIN = tr_l[:800] if SMOKE else tr_l[:4000]

torch.manual_seed(0)
gps = Net(attn=True, rwse=True).to(DEV)
n_params = sum(p.numel() for p in gps.parameters())
n_mpnn = sum(p.numel() for p in Net(attn=False, rwse=True).parameters())
print(f"parameters: GPS {n_params:,} vs MPNN+RWSE {n_mpnn:,} — the attention "
      f"channel is not free (Week 11's audit reflex)")

opt = torch.optim.Adam(gps.parameters(), lr=1e-3)
tl = DataLoader(GPS_TRAIN, batch_size=128, shuffle=True)
losses = []
for ep in range(GPS_EPOCHS):
    gps.train()
    tot, nb = 0.0, 0
    for batch in tl:
        batch = batch.to(DEV)
        loss = F.l1_loss(gps(batch), batch.y)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    losses.append(round(tot / nb, 3))
    print(f"epoch {ep}: train L1 {losses[-1]}")

assert losses[-1] < losses[0], (
    f"the hybrid's training loss must fall across epochs — got {losses} "
    f"(reference: [2.03, 0.92, 0.67] on 4k molecules)"
)
print("exercise 4b \\u2713 — the hybrid trains; ranking it fairly would need "
      "the full protocol, which is exactly the lecture's point")"""),

    md("""### Your claims paragraph *(graded — write it in this cell)*

Three claims, cells cited: one about the hand-verified encoding (the exact
zero, and what quantity the $k{=}3$ entry computes that message passing
provably cannot); one about the CSL crack (what 1-WL said, what your $k=3$
column said, and why that is Week 9's Route 2 industrialized); one about
budgets (your 12-epoch ladder vs the provided 100-epoch table — including
the honest sentence about what attention did not buy at matched budget, and
what your near-tie at 12 epochs says about comparing at ONE budget).

*(your claims here)*

## 5 · Stretch (optional, ungraded)

1. **LapPE and the sign flip.** Compute the Laplacian's first nontrivial
   eigenvector for one molecule; feed it as a feature with and without a
   random sign flip at test time. How much does the output move?
2. **The $K$ ablation.** Rerun the RWSE rung with $K = 4$ and $K = 32$.
   Where does the marginal entry stop paying?
3. **A mini distance bias.** Add a learned scalar per shortest-path
   distance to the attention logits (Graphormer's door two) on ZINC's
   small graphs. Worth anything at this budget?
4. **Your own budget.** Rerun the ladder at 50 epochs. Does the RWSE gap
   reopen? Where does GPS land?

## 6 · Reflection (answer in this cell, 2–4 sentences each)

**R1.** The tail node's $k{=}3$ return is exactly zero. State the graph
fact that forces this, and explain why no MPNN on degree-uniform features
can compute this entry about its own node — which Week 9 theorem applies?

**R2.** Your sabotaged mask moved outputs by ~0.6 yet the sabotaged model
would still train to a plausible-looking MAE. Describe how this bug would
present in a real project (what would you observe? what wouldn't you?),
and write the one CI test you would now always include.

**R3.** At 12 epochs your two rungs nearly tied; at 100 the lecture's gap
is 0.053. A paper ablates its encoding at ONE budget and reports "no
effect." Using this week's five questions, what is the minimal extra
experiment you would demand before believing it?

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
