"""Week 14 measured ladder on MovieLens-100k, GLOBAL TEMPORAL split:
popularity -> BPR-MF (= LightGCN K=0) -> LightGCN K in {1,2,3};
plus the same LightGCN K=3 under a RANDOM split to measure the inflation
a leaky protocol buys. Metrics: Recall@20, NDCG@20 over test users."""
import json
import os
import time

import numpy as np
import torch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng = np.random.default_rng(0)

raw = np.loadtxt("data/ml-100k/u.data", dtype=np.int64)  # user item rating ts
print(f"interactions {len(raw)} · users {len(np.unique(raw[:, 0]))} · items {len(np.unique(raw[:, 1]))}")

# 0-index
raw[:, 0] -= 1
raw[:, 1] -= 1
N_U, N_I = int(raw[:, 0].max()) + 1, int(raw[:, 1].max()) + 1


def make_split(kind):
    if kind == "temporal":
        order = np.argsort(raw[:, 3], kind="stable")
    else:
        order = rng.permutation(len(raw))
    d = raw[order]
    n = len(d)
    tr, va, te = d[: int(0.8 * n)], d[int(0.8 * n): int(0.9 * n)], d[int(0.9 * n):]
    tr_seen_u, tr_seen_i = set(tr[:, 0].tolist()), set(tr[:, 1].tolist())
    keep = lambda X: X[[u in tr_seen_u and i in tr_seen_i for u, i in zip(X[:, 0], X[:, 1])]]
    va, te = keep(va), keep(te)
    return tr, va, te


def eval_ranking(scores, tr, te, K=20):
    """scores: (N_U, N_I) tensor. Mask train items, rank, Recall@K / NDCG@K
    averaged over users WITH test positives."""
    s = scores.clone()
    s[tr[:, 0], tr[:, 1]] = -1e9
    users = np.unique(te[:, 0])
    rec, ndcg = [], []
    topk = torch.topk(s, K, dim=1).indices.cpu().numpy()
    pos_by_u = {}
    for u, i in te[:, :2]:
        pos_by_u.setdefault(int(u), set()).add(int(i))
    for u in users:
        pos = pos_by_u[int(u)]
        hits = [1 if it in pos else 0 for it in topk[u]]
        rec.append(sum(hits) / min(len(pos), K))
        dcg = sum(h / np.log2(r + 2) for r, h in enumerate(hits))
        idcg = sum(1 / np.log2(r + 2) for r in range(min(len(pos), K)))
        ndcg.append(dcg / idcg)
    return round(float(np.mean(rec)), 4), round(float(np.mean(ndcg)), 4), len(users)


def norm_adj(tr):
    """Symmetric-normalized bipartite propagation matrix, sparse torch."""
    ui = torch.tensor(tr[:, :2].T)
    du = torch.zeros(N_U); du.index_add_(0, ui[0], torch.ones(ui.shape[1]))
    di = torch.zeros(N_I); di.index_add_(0, ui[1], torch.ones(ui.shape[1]))
    w = 1.0 / (du[ui[0]].clamp(min=1).sqrt() * di[ui[1]].clamp(min=1).sqrt())
    A_ui = torch.sparse_coo_tensor(ui, w, (N_U, N_I)).coalesce().to(DEV)
    A_iu = torch.sparse_coo_tensor(ui.flip(0), w, (N_I, N_U)).coalesce().to(DEV)
    return A_ui, A_iu


def train_lightgcn(tr, va, K_layers, epochs=300, d=64, lr=1e-2, batch=8192, name=""):
    torch.manual_seed(0)
    P = torch.nn.Parameter(torch.randn(N_U, d, device=DEV) * 0.1)
    Q = torch.nn.Parameter(torch.randn(N_I, d, device=DEV) * 0.1)
    opt = torch.optim.Adam([P, Q], lr=lr)
    A_ui, A_iu = norm_adj(tr)
    tru = torch.tensor(tr[:, 0], device=DEV)
    tri = torch.tensor(tr[:, 1], device=DEV)
    gen = torch.Generator(device="cpu").manual_seed(0)

    def propagate(P0, Q0):
        us, is_ = [P0], [Q0]
        pu, qi = P0, Q0
        for _ in range(K_layers):
            pu, qi = torch.sparse.mm(A_ui, qi), torch.sparse.mm(A_iu, pu)
            us.append(pu); is_.append(qi)
        return torch.stack(us).mean(0), torch.stack(is_).mean(0)

    t0 = time.time()
    best_va, best_state, patience = -1, None, 0
    for ep in range(epochs):
        perm = torch.randperm(len(tru), generator=gen).to(DEV)
        for k in range(0, len(perm), batch):
            idx = perm[k: k + batch]
            u, ipos = tru[idx], tri[idx]
            ineg = torch.randint(0, N_I, (len(idx),), generator=gen).to(DEV)
            Pk, Qk = propagate(P, Q)
            x = (Pk[u] * (Qk[ipos] - Qk[ineg])).sum(1)
            loss = -torch.nn.functional.logsigmoid(x).mean() \
                + 1e-4 * (P[u].pow(2).sum() + Q[ipos].pow(2).sum() + Q[ineg].pow(2).sum()) / len(idx)
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 20 == 0:
            with torch.no_grad():
                Pk, Qk = propagate(P, Q)
                r, nd, _ = eval_ranking((Pk @ Qk.T).cpu(), tr, va)
            if r > best_va:
                best_va, best_state, patience = r, (Pk.cpu().clone(), Qk.cpu().clone()), 0
            else:
                patience += 1
                if patience >= 3:
                    break
    Pk, Qk = best_state
    dt = round(time.time() - t0)
    print(f"{name}: val R@20 {best_va} [{dt}s]", flush=True)
    return (Pk @ Qk.T), dt


results = {}
for kind in ("temporal", "random"):
    tr, va, te = make_split(kind)
    print(f"--- {kind}: train {len(tr)} va {len(va)} te {len(te)}", flush=True)
    if kind == "temporal":
        # popularity
        cnt = torch.zeros(N_U, N_I)
        pop = torch.zeros(N_I)
        pop.index_add_(0, torch.tensor(tr[:, 1]), torch.ones(len(tr)))
        scores = pop.unsqueeze(0).repeat(N_U, 1)
        r, nd, nu = eval_ranking(scores, tr, te)
        results["pop_temporal"] = {"recall20": r, "ndcg20": nd, "users": nu}
        print(f"popularity: R@20 {r} NDCG@20 {nd} ({nu} users)", flush=True)
        for K in (0, 1, 2, 3):
            sc, dt = train_lightgcn(tr, va, K, name=f"LightGCN K={K} ({kind})")
            r, nd, nu = eval_ranking(sc, tr, te)
            results[f"lgcn{K}_temporal"] = {"recall20": r, "ndcg20": nd, "secs": dt}
            print(f"  test R@20 {r} NDCG@20 {nd}", flush=True)
    else:
        sc, dt = train_lightgcn(tr, va, 3, name=f"LightGCN K=3 ({kind})")
        r, nd, nu = eval_ranking(sc, tr, te)
        results["lgcn3_random"] = {"recall20": r, "ndcg20": nd, "secs": dt}
        print(f"  test R@20 {r} NDCG@20 {nd}", flush=True)

sp = os.path.dirname(os.path.abspath(__file__))
json.dump(results, open(os.path.join(sp, "w14_results.json"), "w"), indent=1)
print("saved", flush=True)
