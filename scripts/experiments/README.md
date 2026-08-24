# Measurement scripts — the provenance of every number in the course

Every quantitative claim in the lectures, figures, widgets, and lab assert
bands was **measured before it was written**, by the scripts in this
directory. They are committed so that any claim can be re-derived (or
falsified) on another machine, and so that the figure generators'
docstrings point at code that actually exists.

Run any of them from the **repository root** (they resolve dataset caches
relative to `data/`):

```bash
KMP_DUPLICATE_LIB_OK=TRUE PYTHONIOENCODING=utf-8 python scripts/experiments/w14_experiment.py
```

Each writes a `*_results.json` next to itself; those JSONs are committed
too, so the baked numbers can be diffed against a fresh run.

| Script | Produces | Numbers it anchors |
|---|---|---|
| `w10_experiment3.py` | `w10_results3.json` | Week 10's encoder showdown — DBLP MLP 79.0 → typed 84.0; FB15k-237 TransE 0.178 / DistMult 0.179 / R-GCN+DistMult 0.126 (the lookup wins at 1/22 the compute). **The ±0.06-init protocol is the one consistent series — earlier rounds used a different init and are not comparable.** |
| `w11_experiment.py` | `w11_results.json` | Week 11's scaling table — arxiv neighborhood explosion 18/4,577/22,663; full-batch 0.665 / 2,821 MB; sampled SAGE 0.605 / 64 MB; random-partition cluster 0.628 on 2.5% of edges; SGC K=2 0.638 / 1.5 ms per epoch |
| `w12_experiment.py` | `w12_results.json`, `w12_graphs.json`* | Week 12's link-prediction ladder on the shared Cora split (4,488/263/527) — CN 0.724, AA 0.725, VGAE 0.903, SEAL 0.897, **leaky SEAL 0.363**; GraphRNN-S clustering TV 0.154 vs ER 0.596 |
| `w13_experiment.py` | `w13_results.json` | Week 13's matched-budget ZINC ladder — MPNN 0.349, +RWSE 0.296, GPS-lite 0.303 (100 epochs each) + the hand RWSE table for the triangle-with-tail |
| `w13_lab_budget.py` | `w13_lab_budget.json` | Lab 13's assert bands — 12-epoch MPNN 0.488 / +RWSE 0.477, SMOKE floors, and the padding-mask probe (masked 7e-7 vs sabotaged 0.61) |
| `w14_experiment.py` | `w14_results.json` | Week 14's protocol table — global-temporal popularity 0.1655, LightGCN K=0…3 at 0.1562/0.1749/0.1851/0.1953, and the random-split 0.3462 (**×1.77 inflation**) |
| `w14_experiment2.py` | appends to `w14_results.json` | The leave-last-out row — popularity 0.1266, LightGCN K=3 0.2096 |
| `ablation_results.json` | (data only) | Week 7's 16-config × 3-seed Cora ablation baked into `w7-ablation.js` — GCN·2 81.0±0.6, GAT·4 plain 26.2±9.8 → 80.5±1.0 with residuals, GIN·4 44.2±19.5 |

\* `w12_graphs.json` holds the sampled generator/ER/train graphs baked into
`assets/d3/w12-gen-data.js`.

## Reproducing

Datasets are **not** committed (see `.gitignore`); each script downloads
into `data/` on first run through the same loaders the labs use. Budget
roughly: Planetoid/Cora and MovieLens-100k seconds, ZINC a minute,
ogbn-arxiv ~90 MB (slow on some networks — it is worth keeping the cache).

Hardware and dates are recorded in each script's docstring and in the
figure generators that consume its numbers. Re-running on different
hardware should reproduce the *orderings and the verdicts* — the third
decimal may move, which is exactly why the lab asserts are bands rather
than equalities.
