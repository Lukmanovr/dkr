# DKR Course Design Document

**Data and Knowledge Representation — Innopolis University, Fall 2026**
Version 1.0 (Gate 1 review draft) · 2026-08-20 · prepared for instructor approval

This document is the single source of truth for the course build. Everything downstream —
lecture pages, labs, homeworks, exams, project, website — implements what is specified here.
Sections marked **[DECISION]** need explicit instructor sign-off; everything else is
proposed as final unless objected to.

---

## 1. Course thesis and positioning

**Thesis.** Relational structure is the dominant untapped signal in industrial data. A 2026
ML engineer meets it as knowledge graphs grounding LLM systems, as recommender bipartite
graphs, as fraud/transaction networks, as molecules, and — most often and least glamorously —
as the relational databases every company already owns. This course teaches students to
*represent* such data (graphs, heterogeneous graphs, knowledge graphs), to *learn* from it
(shallow embeddings → GNNs → graph transformers), and to *reason* over it (KG embeddings,
multi-hop queries, GraphRAG), with enough theory to read current NeurIPS/ICLR papers and
enough engineering to ship on real infrastructure.

**Positioning against the two authorities:**

- *Hamilton, Graph Representation Learning* (2020) is the theoretical spine: we adopt its
  notation, its encoder–decoder framing (which unifies shallow embeddings, KG embeddings,
  and GNNs — a framing we exploit in three separate weeks), and its theorem statements.
- *Stanford CS224W (Fall 2025)* is the topic-selection reference. We follow its modern turn
  (graph transformers, LLM+GNN, relational deep learning) but deliberately keep three things
  Stanford dropped: classical graph statistics/centralities, PageRank, and community
  detection. Our students take this as their *first and only* graphs course; the classical
  toolkit is interview-relevant, exam-tested here for years, and the conceptual on-ramp for
  message passing (propagation before learned propagation).

**Differences from the brief's §6 starting table** (all changes, with reasons):

| Change | Reason |
|---|---|
| PageRank + community detection (modularity/Louvain) folded into Wk2 | Legacy exam topics worth keeping; PageRank's power iteration is the natural bridge from centralities to random-walk embeddings in Wk3 |
| Label propagation folded into Wk3 (contrast) and Wk6 (motivation) | "Propagation without learning" is the cleanest motivation for GCN; Correct&Smooth kept as Wk3 optional reading |
| Mapper/TDA and anonymous-walk kernels dropped | Taught last year but off-thesis; the 90-minute budget goes to KG reasoning and transformers, which students will use professionally |
| Wk5 gains an explicit "KGs in the LLM era / GraphRAG" segment | 2026 reality: KG skills are exercised mostly inside LLM systems; CS224W F2025 confirms the direction (3 LLM-adjacent lectures) |
| Wk14 gains a Relational Deep Learning (RelBench) segment | CS224W F2025 devotes two lectures to RDL; "your SQL database is a temporal heterogeneous graph" is the single most career-relevant idea in the modern syllabus |
| Wk13 lab uses ZINC molecules | Molecular property prediction is where graph transformers demonstrably win; doubles as the course's chemistry application |

## 2. Final syllabus with per-week learning objectives

Format: 15 Thursdays, 2026-08-27 → 2026-12-03. Each content week = one 90-min lecture +
one lab. Objectives are measurable (derive/implement/compare/hand-simulate); each maps to
assessment in §5.

### Week 1 · Aug 27 — Why graphs? Representing relational data
Running example: Zachary's karate club + a Wikidata ego-network around *Innopolis University*.
A student can:
1. model a real system as a graph, choosing and justifying type (directed/weighted/bipartite/heterogeneous/multigraph);
2. construct and interconvert adjacency matrix, adjacency list, edge list, and incidence matrix, and state their space/time trade-offs on sparse graphs;
3. define degree, degree matrix, and the unnormalized/normalized Laplacians, and compute them by hand on ≤6-node graphs;
4. classify problems as node-, edge-, or graph-level tasks and name one industrial instance of each;
5. explain why node ordering is arbitrary and what that demands of any learning model (permutation invariance/equivariance, informally).

**Lab 1 (onboarding):** Colab + PyG environment, pinned installs; load Cora and a Wikidata slice; compute degree stats and plot degree distributions; NetworkX↔PyG conversion; first visualization.

### Week 2 · Sep 3 — Classical graph ML: structure as features
Running example: airport network (OpenFlights) + PROTEINS graphs.
A student can:
1. compute degree/eigenvector/closeness/betweenness centralities and clustering coefficient by hand on small graphs, and select the right centrality for a stated question;
2. derive PageRank as a random-surfer fixed point, run 2–3 power iterations by hand, and explain damping/teleport and Personalized PageRank;
3. explain graphlet features and the WL relabeling procedure, and compute one WL iteration by hand;
4. define modularity and step through one round of Louvain's local-moving phase;
5. build a feature-based node/graph classifier and articulate why hand-crafted features hit a ceiling.

**Lab 2:** feature engineering + classical classifiers (logistic regression/GBDT) for node and graph classification; Louvain/Leiden communities on a real network; compare against random baselines.

### Week 3 · Sep 10 — Shallow node embeddings
Running example: co-purchase network (Amazon photo) link prediction.
A student can:
1. state the encoder–decoder framework (encoder, decoder, similarity, loss) and place any embedding method in it;
2. derive the DeepWalk/node2vec objective from random-walk co-occurrence through the softmax to the negative-sampling approximation, defining every symbol (fixing last year's broken midterm formula);
3. explain node2vec's p/q parameters as BFS/DFS interpolation and predict their effect on a given graph;
4. connect random-walk embeddings to matrix factorization (NetMF result, statement only) and use it to reason about what these methods can/cannot capture;
5. enumerate the four limitations of shallow embeddings (no sharing, transductive, no features, O(|V|d) params) — the setup for GNNs;
6. contrast learned embeddings with label propagation on the same task.

**Lab 3:** DeepWalk from scratch (walks → skip-gram with negative sampling in PyTorch) → node2vec via PyG; link-prediction evaluation done right (edge splits); t-SNE/UMAP embedding visualizations.

### Week 4 · Sep 17 — Knowledge graphs I: representation and embeddings
Running example: Wikidata triples about universities, people, cities; FB15k-237.
A student can:
1. represent facts as (head, relation, tail) triples; explain RDF, ontologies/schemas, and open- vs closed-world assumptions;
2. define the TransE, DistMult, ComplEx, and RotatE scoring functions and compute each on toy embeddings by hand;
3. prove which relation patterns (symmetry, antisymmetry, inversion, composition, 1-to-N) each model can and cannot represent — the model-selection cheat-sheet;
4. derive the margin-based and self-adversarial negative-sampling losses and explain corruption sampling;
5. run filtered ranking evaluation and compute MRR/Hits@K by hand on a toy example; explain why FB15k had to become FB15k-237 (inverse-relation leakage).

**Lab 4:** TransE and RotatE from scratch (plain PyTorch, ~100 lines each) on FB15k-237; filtered evaluation; visualize relation embeddings; break TransE on a symmetric relation and watch RotatE fix it.

### Week 5 · Sep 24 — Knowledge graphs II: reasoning, and KGs in the LLM era
Running example: multi-hop questions over a movie/biomedical KG ("drugs that target proteins associated with disease X").
A student can:
1. formalize conjunctive/path queries over incomplete KGs and explain why answering them ≠ chained lookup;
2. explain query embedding: Query2box's box semantics (projection/intersection) and why boxes, not points; BetaE's handling of negation (statement level);
3. evaluate KG completion/reasoning systems without fooling themselves (entity leakage, filtered metrics, hard negatives);
4. describe the GraphRAG pattern (KG-structured retrieval feeding an LLM) and argue when it beats vanilla vector RAG;
5. sketch LLM-assisted KG construction and its failure modes (precision, canonicalization).

**Lab 5:** multi-hop query answering with box embeddings on an FB15k-237 subset; mini-GraphRAG: build a small KG from documents, retrieve subgraphs to answer multi-hop questions, compare against vector-only retrieval.
**Project:** topic list published; team formation opens.

### Week 6 · Oct 1 — GNNs I: message passing and the GCN ★ pilot week
Running example: Cora citation network, one node's computation tree followed all lecture.
A student can:
1. write the general message-passing framework (message/aggregate/update) and instantiate it for the basic GNN and GCN;
2. derive GCN from first principles twice: spatially (normalized neighborhood averaging; why D̃^-1/2ÃD̃^-1/2) and spectrally (graph Fourier transform → ChebNet → first-order truncation), and state what the spectral view buys;
3. prove permutation equivariance of a message-passing layer and permutation invariance of sum-pooling readout;
4. hand-simulate 1–2 GCN layers on a ≤6-node graph with 2-d features (exam-style);
5. implement GCN in <30 lines with sparse matrix operations and explain each line.

**Lab 6:** GCN from scratch (dense, then sparse) reproducing Kipf & Welling's Cora numbers, then the same in PyG; visualize learned representations layer by layer.

### Week 7 · Oct 8 — GNNs II: the design space
Running example: ogbn-arxiv (what changes when the graph is 100× Cora).
A student can:
1. define GraphSAGE (sampling + agg variants), GAT (attention coefficients, multi-head), and GIN (sum aggregation + MLP), and compute each update by hand on toy graphs;
2. compare mean/max/sum aggregators: which multisets they confuse and which tasks that dooms (leads into Wk9 theory);
3. justify architectural choices — skip/residual connections, jumping knowledge, Batch/Layer/PairNorm, DropEdge — by the failure mode each addresses;
4. read a GNN architecture description in a paper and reimplement it from the equations;
5. design and run a controlled ablation (one factor at a time, seeds, error bars).

**Lab 7:** implement GraphSAGE, GAT, GIN in PyG; ablation study on Cora + ogbn-arxiv subset (aggregator × depth × normalization), with a results table and a claim defended by evidence.
**HW1 due (start of Wk6 covered separately — see §5).**

### Week 8 · Oct 15 — MIDTERM
Written, 90 min, full lecture slot, covers Wks 1–7. Review page + practice set published in Wk7.
**Lab 8:** midterm debrief (common errors, worked solutions of hardest problems) + project kickoff: teams finalized, proposal template walkthrough.

### Week 9 · Oct 22 — GNN theory: what GNNs can and cannot see
Running example: two molecules WL cannot tell apart; circulant-skip-link (CSL) graphs.
A student can:
1. execute WL color refinement to stability by hand and use it to decide (non-)isomorphism candidates;
2. state and sketch the proof of GIN ≡ WL expressiveness (injective sum aggregation; why mean/max lose);
3. exhibit graph pairs message passing cannot distinguish and explain *why* (identical computation trees);
4. define oversmoothing (feature convergence with depth) and oversquashing (exponential receptive field through bounded cuts) and match each to its mitigations (depth control, residuals/normalization; rewiring, global attention);
5. explain what k-WL/higher-order GNNs buy and what they cost.

**Lab 9:** implement WL refinement; build WL-indistinguishable graph pairs and verify GIN/GCN behavior on them; measure oversmoothing empirically (feature similarity vs depth); reproduce a mini expressiveness experiment from the GIN paper.

### Week 10 · Oct 29 — Heterogeneous and relational GNNs
Running example: academic graph (author–paper–venue) + FB15k-237 with an RGCN encoder.
A student can:
1. extend message passing to typed nodes/edges (per-relation transforms) and hand-simulate one RGCN layer on a small typed graph;
2. explain RGCN's parameter explosion and derive the basis/block-diagonal decompositions with parameter counts;
3. describe attention-based heterogeneous models (HAN metapaths, HGT) at architecture level and choose between them for a stated schema;
4. assemble encoder–decoder KG completion (RGCN encoder + DistMult decoder) and compare it with Wk4's shallow embeddings — same decoder, learned encoder;
5. map a relational database schema to a heterogeneous graph (the RDL bridge, previewing Wk14).

**Lab 10:** RGCN on a heterogeneous academic dataset (node classification) + RGCN-encoder link prediction on FB15k-237; compare with Lab 4's TransE numbers honestly (same splits, same metrics).

### Week 11 · Nov 5 — Scaling GNNs to real graphs
Running example: ogbn-products (2.4M nodes — what actually breaks).
A student can:
1. compute full-batch GNN memory requirements and identify the neighborhood-explosion problem quantitatively;
2. explain and contrast neighbor sampling (GraphSAGE), Cluster-GCN, and GraphSAINT: what each samples, bias/variance trade-offs, when each wins;
3. derive SGC from GCN (remove nonlinearities → precompute propagation) and state when the simplification is harmless — and how SIGN generalizes it;
4. train on a graph that does not fit in GPU memory using mini-batching, and profile time/memory;
5. sketch a production GNN serving story (precomputed embeddings vs on-demand inference).

**Lab 11:** ogbn-arxiv full-graph vs NeighborLoader vs Cluster-GCN vs SGC: accuracy/time/memory table; scale one config to ogbn-products (CPU-friendly SGC path so free Colab survives).

### Week 12 · Nov 12 — Link prediction and graph generation
Running example: collaboration-graph link prediction; generating community graphs.
A student can:
1. set up link prediction without leakage: message vs supervision edges, temporal vs random splits, negative sampling for evaluation, and choose sane metrics (Hits@K, MRR vs AUC);
2. derive the VGAE objective (inner-product decoder, ELBO) and implement it;
3. explain SEAL's enclosing-subgraph + labeling-trick idea and why it beats node-embedding dot products for structural roles;
4. describe autoregressive generation (GraphRNN's node/edge ordering) and diffusion-based generation (DiGress, high level), and pick evaluation metrics for generated graphs (degree/clustering/orbit MMD);
5. articulate where generation matters (molecules, synthetic data, anonymization).

**Lab 12:** VGAE and SEAL on link prediction with proper splits; train a small GraphRNN on community graphs and evaluate samples against the training distribution.
**HW3 released.**

### Week 13 · Nov 19 — Beyond message passing: graph transformers
Running example: ZINC molecular property prediction; a long-range dependency task where MPNNs fail.
A student can:
1. explain why locality limits message passing (Wk9 callbacks: oversquashing, expressiveness) and what full attention changes;
2. define Laplacian positional encodings and random-walk structural encodings, compute RWSE by hand on small graphs, and explain the eigenvector sign-ambiguity problem;
3. describe Graphormer's three encodings (centrality, spatial, edge) and how attention bias injects structure;
4. explain the GraphGPS recipe (parallel MPNN + attention + PE/SE) and when the hybrid beats either extreme;
5. choose between MPNN and graph transformer for a stated task/scale (attention's O(n²) reality).

**Lab 13:** GPS-style model on ZINC-subset: MPNN-only vs +attention vs +RWSE ablation; visualize what attention attends to on molecules.

### Week 14 · Nov 26 — Graphs in production: temporal, recommenders, relational DL, GraphRAG
Running example: a day in the life of graph ML at a marketplace company (recsys + fraud + analytics DB).
A student can:
1. model temporal graphs (snapshots vs event streams) and explain TGN's memory/message/update components at architecture level;
2. derive LightGCN from GCN (drop transforms/nonlinearity on bipartite graphs) and implement its BPR training; tell the PinSage web-scale story;
3. evaluate recommenders properly (Recall@K/NDCG@K, temporal splits, popularity bias);
4. explain Relational Deep Learning: a relational database as a temporal heterogeneous graph, RelBench tasks, why this may make manual feature engineering on tables obsolete;
5. place GraphRAG and fraud detection in this landscape and describe the graph ML job market map (recsys, integrity/fraud, drug discovery, KG platform, ML infra).

**Lab 14:** LightGCN on MovieLens-100k with proper temporal evaluation (stretch: TGN on a small temporal benchmark).
**Project midway feedback returned this week.**

### Week 15 · Dec 3 — FINAL EXAM + project presentations
Written final, cumulative with post-midterm emphasis, in the lecture slot.
**Lab 15:** project presentations (8 min/team + questions); course retrospective.

## 3. Lecture template (the 11 mandatory components, Quarto realization)

Every lecture `.qmd` renders to (a) the lecture page and (b) a revealjs deck from the same
source (slide content marked with Quarto profiles/conditional content, so page and deck
cannot diverge). Page order, implementing brief §4.2 exactly:

1. **Header block** — week number, title, date, "≈90 min" badge, lab link (adversarial-ML template header style).
2. **Learning objectives** — 3–6 measurable items (callout, `objectives` class) — from §2 verbatim.
3. **Motivation** — running example introduced with a figure or opening widget *before any formalism* (Distill rule).
4. **Recap box** — collapsible callout linking to the specific prior sections needed, with one-line reminders.
5. **Main content** — numbered environments via Quarto crossref (`Definition 6.1`, `Theorem 6.2`, `Example 6.3`), complete derivations (every "it can be shown" is a CI-failing lint), `.math-block`-style plain-English caption under every display formula (ported convention), ≥4 interactive widgets placed at the concept they explain, static figures with numbered captions.
6. **Common pitfalls** — dedicated section, red-accented callouts, each pitfall = misconception + counterexample.
7. **In this week's lab** — 3–5 sentence bridge naming exactly which objects from the lecture get implemented.
8. **Required reading (2 papers)** — each with 2–4-sentence "what to look for" annotation; **Also see**: GRL book chapter + CS224W lecture; **Optional** (2–4 papers).
9. **Check your understanding** — 5–8 questions, mix conceptual/derivation, answers in collapsible blocks.
10. ~~Teaching script / speaker notes~~ — **removed by instructor decision (2026-08-22)**: no per-lecture timing artifact is produced (brief §4.2.10 deviation, approved). The page's predict-before-reveal blocks and widget Try-this prompts carry the in-class engagement duty; the instructor teaches from the projected page freely.
11. ~~Slides link~~ — **removed by instructor decision (2026-08-21)**: no slide decks in this course. Brief §4.2.11 deviation, approved. The lecture page is the single classroom artifact; a print stylesheet covers the offline/PDF role; the deck's engagement beats live on as the page's predict-before-reveal blocks and the teaching script.

House style ported from the adversarial-ML template: serif body (Libre Baskerville),
Playfair display headings, Source Sans UI, JetBrains Mono code, terracotta `#e07a5f` accent
palette with the template's blues/teal/yellow; `.margin-note`-equivalent asides; keyword
introductions in accent color; references grouped by category. New (improving on both
references): light/dark theme pair with toggle, full-text search, collapsible proofs and
answers, numbered theorem environments, `repo-actions` edit/source links, stable slugs
(`lectures/06-gcn.qmd` numbering fixed at authoring time and never renumbered).

## 4. Lab template (Colab notebooks)

Fixed cell skeleton for all 15 labs:

1. **Title + badge cell** — "Open in Colab" badge (`colab.research.google.com/github/lukmanovr/dkr/blob/main/labs/labNN_*.ipynb`), week link, estimated time (≤30 min compute), "what you'll build".
2. **Goals** — 3–5 bullets mirroring lecture objectives implemented here.
3. **Setup** — single pinned install cell (exact versions from §8 ecosystem research), `SMOKE = os.environ.get("SMOKE")` flag that shrinks epochs/datasets for CI, seed fixing, GPU check with CPU fallback.
4. **Guided part** — completed, explained code interleaved with short prose ("read this, run this, notice that") and at least one in-notebook visualization (embeddings, attention weights, message flow).
5. **Exercises** — `# TODO` cells with docstring specs and `assert`-based checks students run themselves (shape checks first, value/behavior checks after, so partial progress is diagnosable).
6. **Stretch** — 1–2 open exercises for strong students, no assertions.
7. **Reflection** — 2–3 questions answered in a Markdown cell.
8. **What to submit** — exact deliverables + how graded (assertions passed + reflection quality).

Solutions: identical notebooks with TODOs filled, in `dkr-private/solutions/labs/`.
CI in the public repo runs `scripts/check_no_solutions.py` (greps for solution markers) on
every push, and executes student notebooks with `SMOKE=1` — student notebooks must run
end-to-end *up to* the first TODO assertion, solutions must run fully green.

## 5. Assessment design

### 5.1 Weights (locked at Gate 0)
Labs 15% (best 12 of 14 × 1.25%) · Homeworks 30% (3 × 10%) · Midterm 15% · Project 20% ·
Final 20%. Cutoffs A/B/C/D = 90/75/60.

### 5.2 Calendar

| Item | Out | Due | Covers |
|---|---|---|---|
| HW1 | after Wk4 (Sep 17) | Wk6 (Oct 1) | Wks 1–4: representations, classical ML, embeddings, KG embeddings |
| Midterm | — | Wk8 (Oct 15) | Wks 1–7 |
| HW2 | after Wk8 (Oct 15) | Wk10 (Oct 29) | Wks 5–7 + midterm remediation: KG reasoning, GNN I–II |
| HW3 | after Wk12 (Nov 12) | Wk14 (Nov 26) | Wks 9–12: theory, hetero, scaling, link prediction |
| Project proposal | Wk5 (topics out) | Wk9 (Oct 22) | — |
| Project midway | — | Wk12 (Nov 12) | — |
| Project report + talk | — | Wk15 (Dec 3) | — |
| Final | — | Wk15 (Dec 3) | cumulative, post-midterm emphasis |

No HW4: three homeworks + project + 14 labs is a full load for a semester with two written
exams. **[DECISION — confirm]**

### 5.3 Assessment map (objective → where tested)
Every §2 objective is tested at least twice (once formative — lab/HW, once summative — exam/project):

| Objective family | Lab | HW | Exam |
|---|---|---|---|
| Representations, Laplacian by hand (Wk1) | L1 | HW1-P1 | Mid Q1–3 |
| Centralities, PageRank iterations, WL relabeling, modularity (Wk2) | L2 | HW1-P2 | Mid Q4–7 |
| Embedding objectives, negative sampling derivation, p/q (Wk3) | L3 | HW1-P3 | Mid Q8–11 |
| KG scoring functions, relation patterns, MRR/Hits@K (Wk4) | L4 | HW1-P4 | Mid Q12–15 |
| Query embeddings, GraphRAG (Wk5) | L5 | HW2-P1 | Final Q4–5 |
| Message passing, GCN derivation + hand-simulation, equivariance (Wk6) | L6 | HW2-P2 | Mid Q16–19 + Final Q1–3 |
| SAGE/GAT/GIN updates, aggregators, ablation method (Wk7) | L7 | HW2-P3 | Mid Q18–19 + Final Q6–8 |
| WL/GIN expressiveness, oversmoothing/oversquashing (Wk9) | L9 | HW3-P1 | Final Q9–12 |
| RGCN decompositions, hetero MP, RDL bridge (Wk10) | L10 | HW3-P2 | Final Q13–14 |
| Sampling methods, SGC derivation, memory math (Wk11) | L11 | HW3-P3 | Final Q15–16 |
| Link-pred splits, VGAE, SEAL, generation (Wk12) | L12 | HW3-P4 | Final Q17–18 |
| PE/SE, Graphormer/GPS (Wk13) | L13 | — | Final Q19–20 |
| Temporal, LightGCN, RelBench (Wk14) | L14 | — | Final Q19–20 |
| End-to-end system building | — | — | Project |

(Exact exam question counts are indicative; exam blueprints in Phase E fix them.)

### 5.4 Exams (meaningful improvement over 2025)
Keep: exam.cls format, cover rules, hand-simulation problems, MCQ+open mix, final's
cheat-sheet+calculator policy. Improve: point-weighted questions (not uniform 2pts),
a printed grading table, every problem with a written solution and rubric in the private
bank, fixed LaTeX/typos, fresh small graphs for every hand-simulation (never reuse 2025
instances), problem-bank tagging by objective so future years can resample, and a public
sample exam per exam with full solutions. Midterm: 90 min, ~6 problems ≈ 30 pts.
Final: 90 min (per 2025 practice — the "100 minutes" on last year's cover contradicted its
own instructions), ~7 problems, cumulative, ≥40% post-midterm content.

### 5.5 Homework design
Each HW ≈ 5–10 h: one derivation part (Markdown/LaTeX answers in the notebook; e.g. HW1:
prove which relation patterns DistMult can represent; HW2: derive GCN as first-order
ChebNet truncation; HW3: prove sum-aggregation injectivity on bounded multisets) + one
implementation part with assertion-based autograder cells + one small open investigation
(pick a hyperparameter, form a hypothesis, test it, write 10 lines). Rubrics published with
each HW; autograder covers ~60% of points.

### 5.6 Project
Teams of 2–3. Deliverables: proposal (1 page: task, dataset, method, evaluation plan, risk),
midway check (repo + preliminary results + blocker list, 15-min TA/instructor meeting),
final report (6 pages, NeurIPS style), code repo (reproducible: README + pinned env +
one-command run), 8-min talk. Rubric (100 pts): problem formulation & motivation 15,
method correctness 25, experimental rigor (baselines! ablations! error bars!) 25,
reproducibility 15, report clarity 10, presentation 10. Originality rules: any external
code must be cited per the honor code; the *contribution* (what you did beyond the
tutorial) must be explicit in the report; AI-use statement mandatory.
15+ suggested topics with datasets ship in `project/topics.qmd` (drafted in Phase E;
spanning OGB node/link/graph tasks, KG completion on Wikidata subsets, GraphRAG systems,
recommenders on MovieLens/Amazon, fraud on Elliptic, molecules on ZINC/QM9, temporal on TGB,
RelBench tasks, reproducibility studies of course papers).

## 6. Website information architecture

```
Home            hero: course pitch, next-deadline strip, quick links (this week's lecture/lab)
Syllabus        week-by-week table: date · lecture (link) · lab (Colab badge) · HW/project events · readings
Lectures        index + 15 lecture pages (each with slides link + speaker notes)
Labs            index + 15 lab pages (Colab badge, goals, what-to-submit) — notebooks live in repo
Homeworks       3 HW pages: notebook badge + rubric + FAQ per HW
Project         overview · milestones · rubric · topic list · past-projects placeholder
Exams           midterm & final info · topic checklists · sample exams (PDF + solutions) · rules
Policies        grading · honor code & AI policy · late policy
Resources       reading list (all annotated papers by week) · notation reference · FAQ · datasets
Staff           instructor + TA placeholders, office hours placeholders
```

Two-click rule: Home → Syllabus row → any artifact; Home → Lectures/Labs → item.
Search covers everything; footer links repo + "report an issue".

## 7. Interactive visualization inventory (69 widgets)

Per-lecture: 4 **core** (ship with the week's PR; a week is not "done" without them) +
1 **stretch** (ship if the week lands on schedule; tracked honestly in status reports).
All: self-contained JS (one file, one HTML include), vendored d3 v7, no network calls,
keyboard-accessible (tab focus + arrow keys on steppers/sliders), static SVG fallback
rendered when JS is unavailable, shared palette/util module `assets/d3/_dkr.js`.

| Wk | Core widgets (4) | Stretch |
|---|---|---|
| 1 | graph-builder → live adjacency/degree/Laplacian matrices · representation cost comparator (matrix/list/edge-list vs \|V\|,\|E\|) · graph-type morpher (one dataset shown as directed/weighted/bipartite/hetero) · task explorer (node/edge/graph highlighting on citation net) | degree-distribution explorer with real datasets, log-log toggle |
| 2 | centrality playground (metric picker + editable graph) · PageRank power-iteration stepper (damping slider, teleport viz) · WL relabeling stepper (1 iteration, hash table shown) · Louvain move-by-move (modularity gain readout) | triangle/clustering-coefficient highlighter |
| 3 | biased random-walk stepper (p/q sliders, BFS↔DFS gradient) · walk→skip-gram training-pair conveyor · live 2-D embedding trainer on karate club (loss curve, epoch stepper) · negative-sampling anatomy (k slider, sampled negatives flash) | co-occurrence matrix vs Z·Zᵀ heatmap comparator |
| 4 | TransE 2-D geometry (drag entities, translation arrows, score readout) · RotatE complex-plane rotator · relation-pattern capability matrix (click pattern × model → geometric demo/counterexample) · filtered-ranking calculator (corrupt triple, watch rank/MRR/Hits@K) | corruption sampler bias demo |
| 5 | query-graph composer (1p/2p/3p/2i over toy KG, answer set live) · Query2box box algebra (drag/resize boxes, intersection shading) · GraphRAG pipeline stepper (question→subgraph→context→answer) · KG-incompleteness demo (hide edges, watch multi-hop answers break) | BetaE beta-distribution embedding explorer |
| 6 | message-passing stepper (per-node vectors, layer stepper, computation-tree expander) · spectral playground (edit graph → Laplacian eigenvectors as colorings; polynomial filter response) · normalization comparer (A vs D⁻¹A vs D̃^-1/2ÃD̃^-1/2 feature evolution) · permutation-equivariance demo (shuffle ordering; GNN outputs track, MLP breaks) | oversmoothing depth slider (teaser for Wk9) |
| 7 | aggregator arena (build multisets that fool mean/max/sum) · GAT attention visualizer (edge widths = α, edit features, multi-head tabs) · GraphSAGE fanout sampler (sampled tree vs full tree, cost counter) · residual signal tracer (depth slider, with/without skips) | design-space heatmap (agg × depth × norm → precomputed accuracy) |
| 9 | WL refinement lab (editable pair incl. CSL/regular pairs WL can't split; stability detector) · multiset injectivity explorer (sum recovers, mean/max collide — interactive proof) · computation-tree twins (two nodes, identical trees, identical embeddings) · oversquashing bottleneck (tree/barbell: receptive-field growth vs one-edge cut) | k-WL (2-WL) on small graphs |
| 10 | typed message-passing stepper (per-relation colors/weights) · RGCN parameter counter (relations slider × basis slider → param count, quality note) · metapath builder (author–paper–venue; metapath → reachable set) · DB-schema→hetero-graph animator (tables/FKs → typed graph; RDL bridge) | HAN semantic-attention inspector |
| 11 | neighborhood-explosion calculator (fanout/depth sliders → nodes touched, memory bar vs GPU sizes) · Cluster-GCN partitioner (clusters colored, cut edges dimmed, batch = cluster) · sampler comparer (node/edge/RW GraphSAINT samplers, subgraph stats) · SGC collapse demo (K-step propagation precompute → linear model; accuracy vs K) | full-batch vs sampled gradient variance visual |
| 12 | edge-split visualizer (message/supervision/valid/test edges; leakage alarm when misused) · VGAE latent explorer (2-D latents ↔ decoded edge-probability heatmap) · SEAL enclosing-subgraph extractor (click edge → subgraph + DRNL labels) · GraphRNN generation stepper (sequential node/edge decisions, adjacency growing) | graph-diffusion denoise scrubber |
| 13 | attention-vs-adjacency matrix comparer (same molecule: sparse MP vs full attention) · Laplacian-PE gallery (ring/grid/molecule eigenvector colorings; sign-flip toggle) · RWSE fingerprinter (per-node return-probability curves as structural signatures) · Graphormer attention-bias anatomy (one attention score decomposed: content + centrality + spatial + edge) | long-range failure demo (task MPNN can't solve, transformer can) |
| 14 | temporal event player (edge stream → TGN memory updates per event) · LightGCN propagation on bipartite graph (layer-combination slider) · Recall@K/NDCG@K explorer (K slider on toy recommender) · relational-DB→temporal-hetero-graph explorer (RelBench-style schema, time-split visual) | fraud-motif spotter (suspicious subgraph patterns) |

Total: 56 core + 13 stretch = 69 specified; ≥4 per content lecture guaranteed by core set.
Midterm week ships an interactive topic-map review widget (not counted).

Survey-adopted additions folded into the table above (see §10): the Wk6 spectral
playground doubles as Penn's frequency-response widget (coefficients → spectral response +
diffusion side by side); Wk6 gains two more stretch candidates — the untrained-GCN
karate-club demo (L65) and a propagation→convolution slider (TUM's label-prop→GCN arc);
the GDL "blueprint table" (architecture = domain + symmetry) recurs as a static-plus-hover
panel from Wk6 onward; Wk9's stretch may instead ship the adversarial edge-flip robustness
demo (TUM), tying into the instructor's adversarial-ML course.

## 8. Toolchain decisions

| Decision | Choice | Justification |
|---|---|---|
| Site generator | **Quarto** (website project + revealjs profiles) | Same-source pages+slides (hard requirement §4.2.11); crossref theorem numbering; built-in search, light/dark themes, collapsible callouts; the SDS 632 reference is Quarto — we replicate the experience and fix its gaps (no dark mode, no collapsible proofs, external notebooks) |
| Math | KaTeX (MathJax fallback via Quarto default `html-math-method: katex`) | Fast, matches template; server-side where possible |
| Graph library | **PyTorch Geometric** (all 15 labs) | 2026 industry/academic default, CS224W-aligned, OGB/RelBench native; one library end-to-end per brief; DGL rejected (ecosystem momentum) |
| Widgets | Vanilla d3 v7, vendored; one JS + one HTML include per widget | Offline requirement; template's proven pattern; no build step |
| Notebook CI | papermill headless execution, `SMOKE=1` env contract, CPU | Free, deterministic; full-fidelity runs done manually per release |
| Slides | Quarto revealjs profile per lecture | Same source as page |
| Exams | LaTeX exam.cls (continuity with 2025) in private repo; PDFs of samples in public | Instructor's existing toolchain |
| Deploy | GitHub Actions → GitHub Pages (`quarto render` + link check + notebook smoke on PRs) | Brief requirement |
| Link check | lychee (site) in CI | Catches rot early |
| Widget QA | Playwright smoke: loads, no console errors, controls respond | Cheap regression net for 69 widgets |

## 9. Reading list (per-week, fetch-verified)

Full annotated list (2 required + 2–4 optional per content week, with "what to look for"
annotations): **[docs/reading-list.md](reading-list.md)**. Every citation was verified by
fetching the URL — twice, by independent agents (selection pass + adversarial verification
pass); one venue correction found and applied (Wk12 diffusion survey → IEEE TKDE 2024).
Required pairs at a glance:

| Wk | Required paper 1 (original method) | Required paper 2 (follow-up/critique/survey) |
|---|---|---|
| 1 | Battaglia et al., *Relational inductive biases…* (arXiv 2018) | Hamilton, Ying, Leskovec, *Representation Learning on Graphs* (IEEE DEB 2017) |
| 2 | Shervashidze et al., *WL Graph Kernels* (JMLR 2011) | Blondel et al., *Fast unfolding of communities* (Louvain, JSTAT 2008) |
| 3 | Perozzi et al., *DeepWalk* (KDD 2014) | Grover & Leskovec, *node2vec* (KDD 2016) |
| 4 | Bordes et al., *TransE* (NeurIPS 2013) | Sun et al., *RotatE* (ICLR 2019) |
| 5 | Ren, Hu, Leskovec, *Query2box* (ICLR 2020) | Edge et al., *GraphRAG* (arXiv 2024, Microsoft) |
| 6 | Kipf & Welling, *GCN* (ICLR 2017) | Gilmer et al., *Neural Message Passing* (ICML 2017) |
| 7 | Hamilton et al., *GraphSAGE* (NeurIPS 2017) | Veličković et al., *GAT* (ICLR 2018) |
| 9 | Xu et al., *How Powerful are GNNs?* (GIN, ICLR 2019) | Alon & Yahav, *Bottleneck/oversquashing* (ICLR 2021) |
| 10 | Schlichtkrull et al., *RGCN* (ESWC 2018) | Hu et al., *Heterogeneous Graph Transformer* (WWW 2020) |
| 11 | Chiang et al., *Cluster-GCN* (KDD 2019) | Zeng et al., *GraphSAINT* (ICLR 2020) |
| 12 | Zhang & Chen, *SEAL* (NeurIPS 2018) | You et al., *GraphRNN* (ICML 2018) |
| 13 | Ying et al., *Graphormer* (NeurIPS 2021) | Rampášek et al., *GraphGPS* (NeurIPS 2022) |
| 14 | Rossi et al., *TGN* (arXiv 2020) | He et al., *LightGCN* (SIGIR 2020) |

Optional papers include the modern anchors the syllabus references: ULTRA/KG foundation
models (ICLR 2024, Wks 5+10), RelBench (NeurIPS 2024, Wk14), Temporal Graph Benchmark
(NeurIPS 2023, Wk14), DiGress (ICLR 2023, Wk12), SignNet (ICLR 2023, Wk13), GATv2
(ICLR 2022, Wk7), benchmark-hygiene critiques (Toutanova & Chen 2015; Sun et al. ACL 2020;
Lv et al. KDD 2021), and applications (Google Maps ETA, PinSage).

## 10. What we learned from other courses (survey of 6, and what we adopt)

**Penn ESE 5140 (Ribeiro)** — one coherent derivation arc (convolution → graph filter →
GNN → equivariance → stability), full written script per lecture, labs that verify theory.
*Adopt:* the polynomial-filter derivation as Wk6's spectral path (filter $p_w(\mathbf{L})$
before GCN); a frequency-response widget (edit filter coefficients → spectral response +
node-domain diffusion, Wk6 core widget); the MovieLens "model ladder" (linear → MLP →
graph filter → 1-layer → 2-layer GNN) as a Wk14 lab exercise; full speaker scripts (our
speaker notes are already scripts — validated).

**Geometric Deep Learning / Cambridge L65 (Bronstein, Veličković, Liò)** — everything
derived from one symmetry blueprint; practicals mix proofs and code with per-exercise mark
budgets; "derive what wasn't shown in lecture" as explicit practical design principle.
*Adopt:* the Erlangen-programme narrative as Wk1's closing hook ("what is the one true
architecture?" ← "what is the one true geometry?"); the blueprint table
(architecture = domain + symmetry group) as a recurring widget from Wk6 on; the
untrained-GCN-already-separates-karate-club demo (Wk6 widget: inductive bias made visceral);
the GATv2 dictionary-lookup exercise (prove static attention fails → fix in 10 lines) in
HW2; "specify ψ, φ, ρ to recover GCN/GAT/MPNN" as an exam-bank problem family; labs mixing
proof exercises with code exercises, each with a published point value.

**UvA DL notebooks (Lippe)** — the gold standard for notebook engineering.
*Adopt:* the 4-node hand-check ritual — every new layer is first run on the same tiny
fixed graph with identity weights and students reproduce one node's update by hand
(this also aligns labs with our hand-simulation exam style); dense→sparse derivation path
(dense matmul → why O(n²) fails → PyG edge_index → block-diagonal batching, shown by
printing a real Batch object, Lab 6/7); the MLP-baseline discipline (every task lab trains
the no-graph baseline and quantifies what structure buys); a `layer_by_name` registry +
one shared training harness across all labs; assertion `unittests` cells; pretrained
checkpoints auto-downloaded for expensive models; optional 10-minute viva as the honor-code
enforcement instrument for suspected undeclared AI use (matches our AI policy's
"explain any line on request").

**McGill COMP 766 (Hamilton)** — the course the GRL book came from.
*Adopt:* the encoder-decoder unification table (GRL Table 3.1) as an interactive widget
spanning Wks 3–4 (shallow embeddings and KG embeddings as rows of one table — the
course's conceptual spine); KGs placed immediately after node embeddings (validates our
Wk3→Wk4 order against CS224W's late placement); classical-baselines-before-deep-methods
discipline (labs must beat the classical baseline from Wk2).

**TU Munich MLGS (Günnemann)** — probabilistic spine, robustness unit, scope discipline.
*Adopt:* the label-propagation→GCN arc ("GCN = learned, parameterized diffusion" — our
Wk3→Wk6 bridge, now with an explicit propagation→convolution slider widget in Wk6);
a GNN-robustness stretch segment (adversarial edge-flip widget as Wk9 stretch; ties into
the instructor's adversarial-ML course — cross-course synergy); their "companion course"
pointer pattern for material we exclude.

**CS224W Fall 2025 (baseline)** — its 3-HW/6-colab/staged-project assessment structure
matches ours; its LLM-era tail (KG foundation models, LLM+GNN, RDL) is threaded through
our Wks 5/10/14 instead of standing as separate lectures our 90-min budget can't afford.

## 11. GRL book mapping and notation

Complete chapter→week mapping with equation-level imports lives in the research record;
the operative decisions:

- **Coverage:** GRL supports Wks 1–4 and 6–12 directly (Ch1–9). Wk4 maps 1:1 onto Ch4.
  **Zero book coverage:** Wk5 (query embeddings, GraphRAG), Wk13 (graph transformers, PEs),
  Wk14 (temporal) — these weeks source from papers (§9) and still import the book's
  notation as lingua franca.
- **Two-pass assignments (deliberate):** Ch7.1 (spectral) is read twice — Wk6 assigns
  7.1.1–7.1.4 (GFT → polynomial filters → GCN derivation), Wk9 re-reads the low-pass-filter
  box for oversmoothing. Oversmoothing likewise appears with the influence-score lens
  (Ch5.3, Thm 3) in Wk7 and the spectral lens in Wk9. Ch2.2 (neighborhood overlap) is
  assigned in Wk3 (random-walk similarity → embeddings bridge) and re-assigned in Wk12 as
  link-prediction baselines. GIN's formula (Ch7.3, Eq 7.63) is introduced in Wk7; its
  WL-optimality proof (Thms 4–5) belongs to Wk9.
- **The book's best thread, kept explicit:** spectral clustering (2.3) → Laplacian
  eigenmaps (3.2) → DeepWalk as implicit matrix factorization of a Laplacian-related
  matrix (3.3.1) — taught as one arc across Wks 2–3.
- **Gaps the book named for us** (each already covered by §9 papers): betweenness/closeness
  formulas, modularity/Louvain, global PageRank, oversquashing, SEAL/labeling trick,
  Cluster-GCN/GraphSAINT, HAN/HGT, graph transformers, diffusion generation, temporal.
- **Notation:** [docs/notation.md](notation.md) codifies the book's symbol table plus our
  fixes: row-stochastic $\mathbf{P} = \mathbf{D}^{-1}\mathbf{A}$ (the book mixes
  conventions), $\mathbf{X} \in \mathbb{R}^{n \times m_0}$ rows-per-node, permutation
  matrices subscripted $\mathbf{P}_\pi$, KG scores oriented higher-is-better, and six house
  rules (captioned formulas, bold/calligraphic typing, layer superscripts). Citations to
  the book go by section number (stable across draft/published editions).

## 12. Ecosystem facts and risk register (verified Aug 2026)

**Environment contract for all 15 labs:**
- Colab 2026.07 runtime: Python 3.12.13, **torch 2.11.0 preinstalled — never reinstall
  torch**; free tier = T4-when-available, ~12.7 GB RAM, ~90-min idle disconnect.
- Install cell: `pip install torch_geometric==2.8.0.post1` (pure-Python, <1 min; PyG 2.8
  supports torch 2.9–2.12). **No pyg-lib/torch-scatter/compiled extras anywhere.** Pin
  every other package (`ogb==1.3.6` + compatible numpy, `pykeen`, `relbench` demo-tier).
- Every lab completes CPU-only in ≤10–15 min (small hidden dims, SMOKE-aware epochs);
  ≤30 min with GPU headroom. Kaggle Notebooks documented as the fallback platform.
- Expect one Colab runtime bump mid-semester (→ torch 2.12): all notebooks re-run that
  week (calendar reminder in handover doc).

**Dataset policy** (per-dataset reliability audited):

| Lab datasets | Source | Policy |
|---|---|---|
| Cora/PubMed, MUTAG/PROTEINS, FB15k-237, WN18RR, ZINC-subset, CoDEx, JODIE wikipedia | third-party GitHub raws, personal domains, a Dropbox link (ZINC = highest rot risk) | **Vendor all raw files in course-owned storage** (GitHub release + HF dataset mirror, CDN-backed vs classroom burst load); notebooks pre-place files under the PyG `root` so the course mirror is tried first, upstream second |
| ogbn-arxiv, ogbl-collab | snap.stanford.edu via frozen `ogb==1.3.6` | Graded-lab tier; mirror processed copies (ODC-BY allows); smoke-test at semester start |
| ogbn-products | snap.stanford.edu | **Demo-only** — processing OOMs free Colab (12.7 GB); scalability lab grades on ogbn-arxiv |
| MovieLens-100k | files.grouplens.org | **License forbids redistribution** — no public mirror; stable upstream URL + private class Drive fallback; citation required in lab text |
| Temporal (Wk14 stretch) | PyG `JODIEDataset` (stable snap URL) | Chosen over py-tgb (TGB's storage already migrated once, breaking old links) |

**KG tooling:** TransE/RotatE implemented from scratch in Lab 4 (~50 lines each), checked
against `torch_geometric.nn.kge` reference implementations (zero extra installs); PyKEEN
shown as the production tool in one demo cell; RelBench (v2, Jan 2026) demo-tier with the
small `rel-f1` database.

**Risk register** (top items; full mitigations inline above):
1. Dataset URL rot mid-semester → vendored mirrors, tried first (highest-likelihood risk; fully mitigated).
2. Colab runtime bump breaking pins → torch-agnostic install contract, re-test week, runtime-version fallback selector documented for students.
3. GPU quota exhaustion → CPU-first lab design, Kaggle fallback.
4. OGB package staleness (frozen Apr 2023) vs numpy/Python drift → pinned numpy, semester-start smoke test, mirrored processed files.
5. Widget volume (69) slipping the schedule → core-4/stretch tiering; a week ships when its core 4 pass Playwright QA; stretch tracked honestly in status reports.
6. Solo-instructor grading load → autograded assertions cover ~60% of HW points; lab grading is submission-checklist-based; viva reserved for flagged cases.

## 13. Build plan and quality gates

Per approved plan: Gate 2 = Quarto skeleton + templates + complete Week 6 pilot (lecture,
4 core widgets + stretch, slides, speaker notes, lab, solutions) deployed to Pages —
instructor reviews depth/tone/visuals before mass production. Then weeks 1–15 in order,
one PR per week; status report to instructor after weeks 5/10/15 with an honest
weak-spot list. Then assessments/policies, then final QA (CI: site build, all notebooks
SMOKE-executed, links, widgets; manual: 2-click nav, dark mode, mobile, notation sweep)
and a handover doc.

Definition of done per week: lecture page passes component lint (template components
present, ≥4 widgets each with a Try-this experiment, ≥2 annotated required papers, 5–8
self-checks with answers, teaching script ≈90 min, ≥2 predict-before-reveal blocks,
figures conform to docs/figure-style.md); the Quality Bar below walked and satisfied;
lab runs green in SMOKE mode and ≤30 min real mode on free Colab (spot-verified);
solutions pass all assertions; no TODO/TBD strings; every citation resolves.

## 14. The Quality Bar (standing, instructor-approved 2026-08-21)

Three checklists govern every page, figure, and widget; the full annotated versions live
in the approved plan of 2026-08-21 and are summarized here. **The target: every lecture
on par with or better than the Distill.pub GNN articles.**

**A · What makes a lecture great** — (1–4) narrative spine: a driving question with
stakes, one running example carrying the lecture, tension→resolution section openers, an
honest limitations ending; (5–9) Mayer load discipline: coherence, signaling,
segmenting (≤5 paragraphs between visual/interactive beats), spatial contiguity
(explanations next to their objects), recap-as-pretraining; (10–12) dual coding: every
core object in ≥2 linked color-coded representations, one visual vocabulary
(figure-style.md), animation only for state/causality; (13–17) active learning:
predict-before-reveal blocks, self-checks with feedback, a pencil-and-paper "you compute
it", ≥4 manipulable widgets with suggested experiments, details-on-demand (hover
glossary, collapsibles); (18–20) rigor with scaffolding: complete derivations, notation
discipline, honest failures; (21–24) memory & transfer: cross-week callbacks, takeaway
anchors, pitfalls-as-inoculation, a production touchpoint; (25) conversational
second-person voice.

*Measured 2026-08-21 (Lighthouse, localhost, simulated slow-4G): Accessibility 100 ·
Best-practices 100 · Performance 55.* The performance score is dominated by ~2 MB of
render-blocking framework CSS/JS (KaTeX + Bootstrap + fonts) under simulated 4G and by
the dev server sending no cache headers (GitHub Pages does). **Perf backlog, honest:**
vendor + subset KaTeX with `font-display: swap`; prune unused Bootstrap via a trimmed
SCSS build; re-measure on the deployed Pages site where caching applies. Widgets already
lazy-init and all scripts defer, so interactivity cost is near zero (TBT 0 ms, CLS 0.07).

**B · What makes a modern website great** — Core Web Vitals (LCP ≤ 2.5 s, INP ≤ 200 ms,
CLS ≤ 0.1) with lazy widget init and preconnected swap fonts; 60–75-ch measure and a
type scale; print stylesheet; token-driven AA-contrast theming; ≤2-click navigation with
scroll-spy ToC, search, reading-time and dates; visible focus states, 150–250 ms
micro-transitions, reduced-motion respect, keyboard operability, skip link; aria +
text fallbacks on every widget/figure, no color-only encoding; touch-usable at 360 px;
OG cards, favicon, useful 404, license footer; exactly one signature delight moment.

**C · What great design is made of** — visible hierarchy; a consistent grid; proximity
does the grouping; whitespace as material; a modular rhythm; role-based color with a
sparingly-spent accent; functional contrast; stable identity colors (the cast);
purposeful four-face type pairing; one recognizable figure grammar; finished details;
restraint in effects; motion that communicates; a distinctive single identity across
the whole site.

**F · What makes a figure great** (Rougier et al. *Ten Simple Rules*, Tufte, Gestalt) —
one message per figure, stated by the caption; audience-fit to the course-so-far;
self-sufficient under the screenshot test; data-ink discipline; honest (schematics
labeled; **data-derived content generated by script, never hand-placed**); Gestalt
layout with no collisions, orphan annotations, or dead-vs-cramped zones; text measured
not estimated (containers sized to text, ≥11 px rendered); color roles never doubled
in one figure, AA in both themes, never color-only encoding; the shared grammar of
`docs/figure-style.md`; proven rendered in light AND dark and legible at 360 px;
teaching captions; progressive-sibling discipline. **Enforced by the Figure Protocol
F1–F5** (figure-style.md): author → render both themes via `scripts/figshot.mjs` →
geometry lint via `scripts/figlint.mjs` (CI-wired) → mandatory visual inspection of the
screenshots → per-figure sign-off in the commit.

**G · What makes a lab coding exercise great** (PRIMM, Use-Modify-Create, worked-example
effect, autograder-feedback evidence) — read/run/investigate before write; predict
before run; a Use→Modify→Create arc per lab; one named concept per exercise; faded
scaffolding with precise contracts; immediate diagnosable feedback (assertion ladders
shape→value→behavior whose messages say what to check); verify on a hand-computable
case before real data; at least one measurement of a lecture claim; boilerplate
provided so effort goes to decisions; visible success at every section end; reflection
prompts as self-explanation, rubric-linked; runs top-to-bottom on free Colab/CPU with
Restart-and-run-all and the SMOKE contract; integrity-by-design (predict/explain
moments tied to this notebook's outputs + the AI-disclosure restated at submission).

**[DECISION] items for Gate 1 sign-off:** §1 syllabus deltas (esp. dropping Mapper/TDA);
§5.2 no-HW4; §5.4 final at 90 min; §5.6 project milestone dates; license proposal
(CC BY-NC-SA 4.0 content / MIT code); everything else in this document.
