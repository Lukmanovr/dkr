"""Definitive consistent series: init ±0.06 everywhere (the lab's code),
DistMult and RGCN+DistMult at 5 and 15 epochs; TransE (Lab-4 recipe) at 5.
Eval: 2000 test facts, both directions, filtered (canonical numbers) and the
same models on 1000 facts (lab-band check)."""
import json
import os
import time

import torch
import torch.nn.functional as F
from torch_geometric.datasets import FB15k_237
from torch_geometric.nn import RGCNConv

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train = FB15k_237("data/FB15k237", split="train")[0]
val = FB15k_237("data/FB15k237", split="val")[0]
test = FB15k_237("data/FB15k237", split="test")[0]
N_ENT, N_REL = train.num_nodes, int(train.edge_type.max()) + 1
tt = lambda d: torch.stack([d.edge_index[0], d.edge_type, d.edge_index[1]], 1)
train_T, val_T, test_T = tt(train), tt(val), tt(test)
ALL = torch.cat([train_T, val_T, test_T])
known_t, known_h = {}, {}
for h, r, t in ALL.tolist():
    known_t.setdefault((h, r), set()).add(t)
    known_h.setdefault((r, t), set()).add(h)


def corrupt(triples, n_ent, g):
    C = triples.clone()
    side = torch.randint(2, (len(C),), generator=g)
    ent = torch.randint(n_ent, (len(C),), generator=g)
    C[side == 0, 0] = ent[side == 0]
    C[side == 1, 2] = ent[side == 1]
    return C


def margin_loss(pos, neg, gamma=1.0):
    return F.relu(gamma - pos + neg).mean()


def evaluate(sc_t, sc_h, n, name):
    g_eval = torch.Generator().manual_seed(123)
    sample = test_T[torch.randperm(len(test_T), generator=g_eval)[:n]]
    ranks = []
    with torch.no_grad():
        for h, r, t in sample.tolist():
            s = sc_t(h, r)
            s[list(known_t[(h, r)] - {t})] = -1e9
            ranks.append(int((s > s[t]).sum()) + 1)
            s = sc_h(r, t)
            s[list(known_h[(r, t)] - {h})] = -1e9
            ranks.append(int((s > s[h]).sum()) + 1)
    rk = torch.tensor(ranks, dtype=torch.float)
    out = {"MRR": round((1 / rk).mean().item(), 3),
           "H10": round((rk <= 10).float().mean().item(), 3), "n": n}
    print(name, out, flush=True)
    return out


def train_distmult(epochs, d=100, lr=0.01, batch=2048):
    g = torch.Generator().manual_seed(0)
    E = torch.nn.Parameter(torch.empty(N_ENT, d).uniform_(-0.06, 0.06, generator=g))
    R = torch.nn.Parameter(torch.empty(N_REL, d).uniform_(-0.06, 0.06, generator=g))
    opt = torch.optim.Adam([E, R], lr=lr)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(train_T), generator=g)
        for i in range(0, len(train_T), batch):
            pos = train_T[perm[i:i + batch]]
            neg = corrupt(pos, N_ENT, g)
            sc = lambda T: (E[T[:, 0]] * R[T[:, 1]] * E[T[:, 2]]).sum(1)
            loss = margin_loss(sc(pos), sc(neg))
            opt.zero_grad(); loss.backward(); opt.step()
    return E.detach(), R.detach(), round(time.time() - t0)


def train_transe(epochs=5, d=100, lr=0.01, batch=2048):
    g = torch.Generator().manual_seed(0)
    E = torch.nn.Parameter(torch.empty(N_ENT, d).uniform_(-0.6, 0.6, generator=g))
    R = torch.nn.Parameter(torch.empty(N_REL, d).uniform_(-0.6, 0.6, generator=g))
    opt = torch.optim.Adam([E, R], lr=lr)
    t0 = time.time()
    for ep in range(epochs):
        with torch.no_grad():
            E.data = F.normalize(E.data, dim=1)
        perm = torch.randperm(len(train_T), generator=g)
        for i in range(0, len(train_T), batch):
            pos = train_T[perm[i:i + batch]]
            neg = corrupt(pos, N_ENT, g)
            sc = lambda T: -(E[T[:, 0]] + R[T[:, 1]] - E[T[:, 2]]).norm(dim=1)
            loss = margin_loss(sc(pos), sc(neg))
            opt.zero_grad(); loss.backward(); opt.step()
    return E.detach(), R.detach(), round(time.time() - t0)


class Enc(torch.nn.Module):
    def __init__(self, d=100, bases=30):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = torch.nn.Parameter(torch.empty(N_ENT, d).uniform_(-0.06, 0.06, generator=g))
        self.conv = RGCNConv(d, d, num_relations=2 * N_REL, num_bases=bases)
        self.rel = torch.nn.Parameter(torch.empty(N_REL, d).uniform_(-0.06, 0.06, generator=g))

    def encode(self, ei, et):
        return self.emb + F.relu(self.conv(self.emb, ei, et))


def train_rgcn(epochs, batch=8192, lr=0.01):
    ei = torch.cat([train.edge_index, train.edge_index.flip(0)], 1).to(DEV)
    et = torch.cat([train.edge_type, train.edge_type + N_REL]).to(DEV)
    model = Enc().to(DEV)
    T = train_T.to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(T), generator=g).to(DEV)
        for i in range(0, len(T), batch):
            pos = T[perm[i:i + batch]]
            neg = corrupt(pos.cpu(), N_ENT, g).to(DEV)
            Z = model.encode(ei, et)
            sc = lambda X: (Z[X[:, 0]] * model.rel[X[:, 1]] * Z[X[:, 2]]).sum(1)
            loss = margin_loss(sc(pos), sc(neg))
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return model.encode(ei, et).cpu(), model.rel.detach().cpu(), round(time.time() - t0)


results = {}
for ep in (5, 15):
    E, R, secs = train_distmult(ep)
    r2000 = evaluate(lambda h, r: (E[h] * R[r] * E).sum(1),
                     lambda r, t: (E * R[r] * E[t]).sum(1), 2000, f"DistMult {ep}ep ({secs}s)")
    r1000 = evaluate(lambda h, r: (E[h] * R[r] * E).sum(1),
                     lambda r, t: (E * R[r] * E[t]).sum(1), 1000, f"  (1000-fact)")
    results[f"DistMult_{ep}ep"] = {"secs": secs, "e2000": r2000, "e1000": r1000}

E, R, secs = train_transe()
results["TransE_5ep"] = {"secs": secs, "e2000": evaluate(
    lambda h, r: -(E[h] + R[r] - E).norm(dim=1),
    lambda r, t: -(E + R[r] - E[t]).norm(dim=1), 2000, f"TransE 5ep ({secs}s)")}

for ep in (5, 15):
    Z, RR, secs = train_rgcn(ep)
    r2000 = evaluate(lambda h, r: (Z[h] * RR[r] * Z).sum(1),
                     lambda r, t: (Z * RR[r] * Z[t]).sum(1), 2000, f"RGCN {ep}ep ({secs}s)")
    r1000 = evaluate(lambda h, r: (Z[h] * RR[r] * Z).sum(1),
                     lambda r, t: (Z * RR[r] * Z[t]).sum(1), 1000, f"  (1000-fact)")
    results[f"RGCN_{ep}ep"] = {"secs": secs, "e2000": r2000, "e1000": r1000}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w10_results3.json")
json.dump(results, open(out, "w"), indent=1)
print("saved", out, flush=True)
