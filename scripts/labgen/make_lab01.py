#!/usr/bin/env python
"""Generate Lab 1 (onboarding: environment, karate club, representations) —
student and solution notebooks. Same single-source pattern as make_lab06.py.

    python scripts/labgen/make_lab01.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
STUDENT_OUT = ROOT / "labs" / "lab01_graphs.ipynb"
SOLUTION_OUT = ROOT.parent / "dkr-private" / "solutions" / "labs" / "lab01_graphs.ipynb"


def md(text): return ("markdown", text, None)
def code(text): return ("code", text, None)
def todo(text, stub): return ("code", text, stub)


CELLS = [
    md("""# Lab 1 · Hello, graphs — environment, karate club, and three representations

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lukmanovr/dkr/blob/main/labs/lab01_graphs.ipynb)

**Week 1 · [lecture](https://lukmanovr.github.io/dkr/lectures/01-why-graphs.html) · ≈ 15 min of compute (free Colab or CPU — no GPU needed today)**

This is the onboarding lab: you set up the exact environment used all semester, meet the
real Zachary karate club — the 34-member network from
[Zachary, 1977](https://doi.org/10.1086/jar.33.4.3629752) that opened the lecture —
build the course's cast graph as a PyTorch Geometric
`edge_index` **by hand**, verify the lecture's hand computations numerically, and take a
first look at Cora and at a real Wikidata fragment.

The representations built here are the substrate of every method in this course: the
survey of [Hamilton et al., 2017](https://arxiv.org/abs/1709.05584) frames all of
graph ML as encoders and decoders over exactly these structures, and every one of them
starts life as an `edge_index`.

### Goals
1. A working, pinned Colab/CPU environment (the one every later lab assumes).
2. Build and *verify* an `edge_index` from scratch — the representation the whole course runs on.
3. Compute degrees, walk counts via $\\mathbf{A}^2$, and the Laplacian in torch, matching the lecture's pencil results.
4. See one real dataset (Cora) and one real knowledge-graph fragment (Wikidata) up close.
"""),

    md("""## 0 · Setup  *(the contract for all fifteen labs)*

One pinned install; on Colab we never reinstall `torch`. The `SMOKE` flag lets course CI
run this notebook headlessly — leave it untouched.
"""),

    code("""import os, sys, random

SMOKE = os.environ.get("SMOKE", "") == "1"
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    %pip install -q torch_geometric==2.8.0.post1

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import torch_geometric

SEED = 41
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print(f"torch {torch.__version__} · torch_geometric {torch_geometric.__version__} · networkx {nx.__version__}")
print("environment OK — this exact stack is what Weeks 2–15 assume")"""),

    md("""## 1 · The 1977 dataset, live

The karate club from lecture ships with networkx — 34 members, and the `club` attribute
records which side each member actually joined after the split.

**Predict before you run** (lecture check): how many *edges* should networkx report?
And which two members will have the highest degree? Write down your predictions before
running the next cell.
"""),

    code("""G = nx.karate_club_graph()
print(f"nodes {G.number_of_nodes()} · edges {G.number_of_edges()}")

deg_sorted = sorted(G.degree, key=lambda kv: -kv[1])[:4]
print("highest degrees:", deg_sorted)
print("node 0 is Mr. Hi, node 33 is the Officer — the two hubs the factions formed around")

colors = ["#d9a62e" if G.nodes[v]["club"] == "Mr. Hi" else "#0f8377" for v in G]
pos = nx.spring_layout(G, seed=7)
plt.figure(figsize=(7, 4.5))
nx.draw_networkx_edges(G, pos, alpha=0.3)
nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=[60 + 25 * G.degree[v] for v in G])
plt.title("Zachary's karate club — colored by the side each member joined")
plt.axis("off"); plt.show()"""),

    md("""## 2 · Build the cast graph by hand  *(exercise 1 — skill: edge list → edge_index)*

The lecture's cast graph has nodes A–F, numbered 0–5, and the seven undirected edges
AB, AC, AD, BC, CE, CF, EF. PyG stores an *undirected* graph as a **directed edge list
containing both directions** — a `(2, 2m)` integer tensor called `edge_index`.

**What to do:** in the next cell, implement `cast_edge_index()` so that it returns the
cast graph as a `(2, 14)` LongTensor containing both directions of the seven edges.
Type the seven edges in by hand — do not compute or import them. The order of the
columns is up to you, but every edge must appear in both directions. The asserts below
the function check the shape, the edge set, and the both-directions property; when the
cell prints "exercise 1 ✓" your implementation is correct.

The algorithm below (Algorithm 1 from the lecture) specifies exactly what your
implementation in the next cell must do; follow it step by step. This exercise
implements lines 1–5; exercise 2 in Section 3 implements lines 6–7.

> **Algorithm 1 · Edge list → COO `edge_index`, with degrees**
>
> **Input:** undirected edge set E = {(u₁,v₁), …, (u_m,v_m)}; node count n
> **Output:** `edge_index` ∈ ℤ^(2×2m); degree vector d ∈ ℤⁿ
>
> 1. P ← empty list
> 2. **for** each edge (u, v) ∈ E **do**
> 3. &nbsp;&nbsp;&nbsp;&nbsp;append (u, v) to P; append (v, u) to P *(both directions)*
> 4. **end for**
> 5. `edge_index` ← transpose(P) — shape (2, 2m)
> 6. d ← zeros(n)
> 7. **for** each entry u in row 0 of `edge_index` **do** d[u] ← d[u] + 1
> 8. **return** `edge_index`, d — sanity: Σᵤ dᵤ = 2m (handshake lemma)
"""),

    todo("""def cast_edge_index() -> torch.Tensor:
    \"\"\"Return the cast graph as a (2, 14) LongTensor containing BOTH directions
    of the 7 undirected edges: AB, AC, AD, BC, CE, CF, EF (A=0 … F=5).\"\"\"
    ### BEGIN SOLUTION
    und = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (2, 5), (4, 5)]
    pairs = und + [(b, a) for a, b in und]
    return torch.tensor(pairs, dtype=torch.long).t()
    ### END SOLUTION


ei = cast_edge_index()
assert ei.shape == (2, 14), (
    f"edge_index shape is {tuple(ei.shape)}, expected (2, 14) — did you include BOTH "
    f"directions of each of the 7 undirected edges, and transpose to 2 rows?"
)
expected_pairs = {(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (2, 5), (4, 5)}
got_pairs = {tuple(sorted(p)) for p in ei.t().tolist()}
assert got_pairs == expected_pairs, (
    f"edge set mismatch — extra/missing undirected pairs: "
    f"{got_pairs ^ expected_pairs}. Check the edge list against the lecture's cast graph."
)
directed = set(map(tuple, ei.t().tolist()))
assert all((b, a) in directed for a, b in directed), (
    "every edge must appear in both directions — (u,v) AND (v,u)"
)
print("exercise 1 ✓ — the cast graph lives in a tensor now")""",
         stub="""def cast_edge_index() -> torch.Tensor:
    \"\"\"Return the cast graph as a (2, 14) LongTensor containing BOTH directions
    of the 7 undirected edges: AB, AC, AD, BC, CE, CF, EF (A=0 … F=5).\"\"\"
    # TODO: list the 7 pairs, add the reversed pairs, build the tensor, transpose.
    raise NotImplementedError("build the cast graph's edge_index")


ei = cast_edge_index()
assert ei.shape == (2, 14), (
    f"edge_index shape is {tuple(ei.shape)}, expected (2, 14) — did you include BOTH "
    f"directions of each of the 7 undirected edges, and transpose to 2 rows?"
)
expected_pairs = {(0, 1), (0, 2), (0, 3), (1, 2), (2, 4), (2, 5), (4, 5)}
got_pairs = {tuple(sorted(p)) for p in ei.t().tolist()}
assert got_pairs == expected_pairs, (
    f"edge set mismatch — extra/missing undirected pairs: "
    f"{got_pairs ^ expected_pairs}. Check the edge list against the lecture's cast graph."
)
directed = set(map(tuple, ei.t().tolist()))
assert all((b, a) in directed for a, b in directed), (
    "every edge must appear in both directions — (u,v) AND (v,u)"
)
print("exercise 1 ✓ — the cast graph lives in a tensor now")"""),

    md("""## 3 · Degrees from the edge list  *(exercise 2 — skill: structure numbers without a matrix)*

**What to do:** in the next cell, implement `degrees(edge_index, num_nodes)` so that it
returns a length-`num_nodes` integer tensor whose entry `d[u]` is the degree of node `u`.
Compute it directly from `edge_index` and do not build an adjacency matrix — working
without the matrix is the sparse habit from Pitfall 1 in the lecture. You are
implementing lines 6–7 of Algorithm 1 above; torch collapses that loop into a single
`bincount` call. The assert compares your output against the hand-computed degrees
`[3, 2, 4, 1, 2, 2]`; when the cell prints "exercise 2 ✓" your implementation is correct.
"""),

    todo("""def degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    \"\"\"Degree of every node, computed from the (both-directions) edge_index.
    Hint: how many times does each node appear in row 0?\"\"\"
    ### BEGIN SOLUTION
    return torch.bincount(edge_index[0], minlength=num_nodes)
    ### END SOLUTION


d = degrees(ei, 6)
assert d.tolist() == [3, 2, 4, 1, 2, 2], (
    f"degrees {d.tolist()} — expected [3, 2, 4, 1, 2, 2] for A..F. Check: with both "
    f"directions stored, counting row-0 occurrences counts each undirected edge once per endpoint."
)
print("exercise 2 ✓ — degrees:", dict(zip("ABCDEF", d.tolist())))""",
         stub="""def degrees(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    \"\"\"Degree of every node, computed from the (both-directions) edge_index.
    Hint: how many times does each node appear in row 0?\"\"\"
    # TODO: one line with torch.bincount.
    raise NotImplementedError("compute degrees from edge_index")


d = degrees(ei, 6)
assert d.tolist() == [3, 2, 4, 1, 2, 2], (
    f"degrees {d.tolist()} — expected [3, 2, 4, 1, 2, 2] for A..F. Check: with both "
    f"directions stored, counting row-0 occurrences counts each undirected edge once per endpoint."
)
print("exercise 2 ✓ — degrees:", dict(zip("ABCDEF", d.tolist())))"""),

    md("""## 4 · The lecture's pencil math, verified  *(exercise 3 — skill: A², Laplacian)*

The lecture proved $(\\mathbf{A}^k)[u,v]$ counts length-$k$ walks and that
$\\mathbf{x}^\\top\\mathbf{L}\\mathbf{x}$ counts cut edges for indicator signals. Now you
confirm every claim numerically, on the same cast graph, against the same numbers
computed in the lecture examples and self-checks.

**What to do:** in the next cell, `dense_adjacency` is written for you; implement
`laplacian(A)` so that it returns $\\mathbf{L} = \\mathbf{D} - \\mathbf{A}$, where
$\\mathbf{D}$ is the diagonal matrix of row sums of $\\mathbf{A}$. The asserts then
verify the walk counts in $\\mathbf{A}^2$ and the cut value
$\\mathbf{x}^\\top\\mathbf{L}\\mathbf{x} = 2$ against the lecture's pencil computations;
when the cell prints "exercise 3 ✓" every check has passed.
"""),

    todo("""def dense_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    \"\"\"Dense A (fine at n=6; a firing offense at n=10⁶ — see the lecture's cost widget).\"\"\"
    A = torch.zeros(num_nodes, num_nodes)
    A[edge_index[0], edge_index[1]] = 1.0
    return A


def laplacian(A: torch.Tensor) -> torch.Tensor:
    \"\"\"L = D − A.\"\"\"
    ### BEGIN SOLUTION
    return torch.diag(A.sum(dim=1)) - A
    ### END SOLUTION


A = dense_adjacency(ei, 6)
A2 = A @ A
assert A2[0, 0].item() == 3, "A²[A,A] should equal deg(A)=3 — the out-and-back walks"
assert A2[0, 2].item() == 1, "A²[A,C] should be 1 — the single walk A→B→C (common neighbor B)"
assert A2[1, 5].item() == 1, "A²[B,F] should be 1 — self-check Q2's walk B→C→F"

L = laplacian(A)
assert torch.allclose(L.sum(dim=1), torch.zeros(6)), (
    "every Laplacian row must sum to 0 — check L = D − A with D the diagonal of row sums"
)
x = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
cut = (x @ L @ x).item()
assert cut == 2.0, (
    f"xᵀLx = {cut}, expected 2.0 — the two cut edges CE and CF of the "
    f"{{A,B,C,D}} vs {{E,F}} partition (self-check Q4)"
)
print(f"exercise 3 ✓ — A² and L agree with every pencil computation; cut size = {cut:.0f}")""",
         stub="""def dense_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    \"\"\"Dense A (fine at n=6; a firing offense at n=10⁶ — see the lecture's cost widget).\"\"\"
    A = torch.zeros(num_nodes, num_nodes)
    A[edge_index[0], edge_index[1]] = 1.0
    return A


def laplacian(A: torch.Tensor) -> torch.Tensor:
    \"\"\"L = D − A.\"\"\"
    # TODO: one line — the degree matrix is torch.diag of the row sums.
    raise NotImplementedError("build the Laplacian")


A = dense_adjacency(ei, 6)
A2 = A @ A
assert A2[0, 0].item() == 3, "A²[A,A] should equal deg(A)=3 — the out-and-back walks"
assert A2[0, 2].item() == 1, "A²[A,C] should be 1 — the single walk A→B→C (common neighbor B)"
assert A2[1, 5].item() == 1, "A²[B,F] should be 1 — self-check Q2's walk B→C→F"

L = laplacian(A)
assert torch.allclose(L.sum(dim=1), torch.zeros(6)), (
    "every Laplacian row must sum to 0 — check L = D − A with D the diagonal of row sums"
)
x = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
cut = (x @ L @ x).item()
assert cut == 2.0, (
    f"xᵀLx = {cut}, expected 2.0 — the two cut edges CE and CF of the "
    f"{{A,B,C,D}} vs {{E,F}} partition (self-check Q4)"
)
print(f"exercise 3 ✓ — A² and L agree with every pencil computation; cut size = {cut:.0f}")"""),

    md("""## 5 · First contact with Cora

Cora returns as the star of Week 6. Today, just meet it — and settle a storage question
from the lecture.

**Predict before you run:** would Cora's *dense* adjacency matrix fit in free Colab's
~12.7 GB of RAM? And ogbn-products' (n = 2.4 M)? Write down your two yes/no answers
before running the next cell.
"""),

    code("""from torch_geometric.datasets import Planetoid

dataset = Planetoid(root="data/Planetoid", name="Cora")
data = dataset[0]
print(data)

n = data.num_nodes
dense_bytes = n * n * 4
print(f"Cora: n = {n:,} → dense A = {dense_bytes/1e6:.0f} MB — fits easily. Surprised?")
np_ = 2_449_029
print(f"ogbn-products: n = {np_:,} → dense A = {np_**2*4/1e12:.0f} TB — does not exist.")
print("moral: 'sparse or nothing' is a statement about SCALE, and scale arrives fast (Week 11)")"""),

    md("""## 6 · A real knowledge-graph fragment  *(exercise 4 — skill: typed edges are just more bookkeeping)*

The next cell contains twelve real facts from Wikidata about this course's own
neighborhood, written as (head, relation, tail) triples — the heterogeneous shape from
the lecture's Figure 3, and the object Weeks 4–5 are devoted to. There is nothing to
download; the fragment is pasted directly into the cell.

**What to do:** in the next cell, implement `kg_stats(triples)` so that it returns the
tuple `(num_entities, num_relation_types, num_facts)`. Entities are everything that
appears as a head *or* a tail; relation types are the distinct middle elements. The
asserts check for 10 entities, 5 relation types, and 12 facts; when the cell prints
"exercise 4 ✓" your counts are correct.
"""),

    todo("""TRIPLES = [
    ("Innopolis University", "instance_of", "university"),
    ("Innopolis University", "located_in", "Innopolis"),
    ("Innopolis", "located_in", "Tatarstan"),
    ("Kazan", "capital_of", "Tatarstan"),
    ("Kazan", "located_in", "Tatarstan"),
    ("Tatarstan", "part_of", "Russia"),
    ("Kazan Federal University", "instance_of", "university"),
    ("Kazan Federal University", "located_in", "Kazan"),
    ("Innopolis", "instance_of", "city"),
    ("Kazan", "instance_of", "city"),
    ("Tatarstan", "instance_of", "region"),
    ("Volga", "flows_through", "Tatarstan"),
]


def kg_stats(triples) -> tuple[int, int, int]:
    \"\"\"Return (num_entities, num_relation_types, num_facts). Entities are everything
    appearing as a head OR a tail.\"\"\"
    ### BEGIN SOLUTION
    entities = {h for h, _, _ in triples} | {t for _, _, t in triples}
    relations = {r for _, r, _ in triples}
    return len(entities), len(relations), len(triples)
    ### END SOLUTION


ne, nr, nf = kg_stats(TRIPLES)
assert nf == 12, "num_facts is just len(triples)"
assert nr == 5, (
    f"found {nr} relation types, expected 5 — collect the MIDDLE element of each triple into a set"
)
assert ne == 10, (
    f"found {ne} entities, expected 10 — the union of heads and tails "
    f"(note 'university', 'city', 'region' are entities here too)"
)
print(f"exercise 4 ✓ — {ne} entities · {nr} relation types · {nf} facts")
print("this IS a graph: typed nodes, typed edges. Week 4 teaches machines to guess the missing facts.")""",
         stub="""TRIPLES = [
    ("Innopolis University", "instance_of", "university"),
    ("Innopolis University", "located_in", "Innopolis"),
    ("Innopolis", "located_in", "Tatarstan"),
    ("Kazan", "capital_of", "Tatarstan"),
    ("Kazan", "located_in", "Tatarstan"),
    ("Tatarstan", "part_of", "Russia"),
    ("Kazan Federal University", "instance_of", "university"),
    ("Kazan Federal University", "located_in", "Kazan"),
    ("Innopolis", "instance_of", "city"),
    ("Kazan", "instance_of", "city"),
    ("Tatarstan", "instance_of", "region"),
    ("Volga", "flows_through", "Tatarstan"),
]


def kg_stats(triples) -> tuple[int, int, int]:
    \"\"\"Return (num_entities, num_relation_types, num_facts). Entities are everything
    appearing as a head OR a tail.\"\"\"
    # TODO: two set comprehensions and a len (≈ 3 lines).
    raise NotImplementedError("count entities, relation types, facts")


ne, nr, nf = kg_stats(TRIPLES)
assert nf == 12, "num_facts is just len(triples)"
assert nr == 5, (
    f"found {nr} relation types, expected 5 — collect the MIDDLE element of each triple into a set"
)
assert ne == 10, (
    f"found {ne} entities, expected 10 — the union of heads and tails "
    f"(note 'university', 'city', 'region' are entities here too)"
)
print(f"exercise 4 ✓ — {ne} entities · {nr} relation types · {nf} facts")
print("this IS a graph: typed nodes, typed edges. Week 4 teaches machines to guess the missing facts.")"""),

    md("""## 7 · Stretch (optional, ungraded)

1. **The prediction, reproduced.** Compute the karate club's Laplacian in numpy, take
   the eigenvector for the second-smallest eigenvalue, and split members by its sign.
   How many of the 34 does this one vector classify correctly? (You are two weeks early
   to spectral clustering — enjoy the sneak preview.)
2. **Your own graph.** Build `edge_index` for any system you know — your friend group,
   a metro map — and report n, m, max degree, and one thing the structure told you.

## 8 · Reflection (answer in this cell, 2–4 sentences each)

**R1.** PyG stores each undirected edge twice. Name one operation from this lab that
this redundancy makes a one-liner, and one cost it incurs.

**R2.** Cora's dense matrix fits in RAM comfortably — so when exactly is "sparse or
nothing" *actually* binding? Use the lecture's cost widget to give an approximate n.

**R3.** Pick a system from your daily life, and specify it as a graph: nodes, edges,
type (directed/weighted/bipartite/heterogeneous), and the prediction level of one useful
question about it.

*(your answers here)*

## What to submit

One executed notebook on Moodle: all four exercise checks ✓, the karate-club figure
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
