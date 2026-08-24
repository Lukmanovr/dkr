"""Week 14 addendum: per-user leave-last-out split (the literature's usual
compromise) for popularity and LightGCN K=3 — third row of the protocol
table alongside global-temporal and random."""
import json
import os

import numpy as np
import torch

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "w14_experiment.py")).read().split('results = {}')[0])

# leave-last-out: per user, last interaction -> test, 2nd-last -> val
order = np.lexsort((raw[:, 3], raw[:, 0]))  # by user, then ts
d = raw[order]
tr_rows, va_rows, te_rows = [], [], []
for u in np.unique(d[:, 0]):
    rows = d[d[:, 0] == u]
    if len(rows) < 3:
        tr_rows.append(rows)
        continue
    tr_rows.append(rows[:-2])
    va_rows.append(rows[-2:-1])
    te_rows.append(rows[-1:])
tr = np.concatenate(tr_rows)
va = np.concatenate(va_rows)
te = np.concatenate(te_rows)
tr_items = set(tr[:, 1].tolist())
keep = lambda X: X[[i in tr_items for i in X[:, 1]]]
va, te = keep(va), keep(te)
print(f"leave-last-out: train {len(tr)} va {len(va)} te {len(te)}", flush=True)

sp_dir = os.path.dirname(os.path.abspath(__file__))
results = json.load(open(os.path.join(sp_dir, "w14_results.json")))

pop = torch.zeros(N_I)
pop.index_add_(0, torch.tensor(tr[:, 1]), torch.ones(len(tr)))
r, nd, nu = eval_ranking(pop.unsqueeze(0).repeat(N_U, 1), tr, te)
results["pop_llo"] = {"recall20": r, "ndcg20": nd, "users": nu}
print(f"popularity (LLO): R@20 {r} NDCG@20 {nd} ({nu} users)", flush=True)

sc, dt = train_lightgcn(tr, va, 3, name="LightGCN K=3 (leave-last-out)")
r, nd, nu = eval_ranking(sc, tr, te)
results["lgcn3_llo"] = {"recall20": r, "ndcg20": nd, "secs": dt, "users": nu}
print(f"  test R@20 {r} NDCG@20 {nd}", flush=True)

json.dump(results, open(os.path.join(sp_dir, "w14_results.json"), "w"), indent=1)
print("saved", flush=True)
