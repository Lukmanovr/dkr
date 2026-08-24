"""Week 12 measurements.
Part A (Cora link prediction, ONE proper split shared by all methods):
  common-neighbors and Adamic-Adar heuristics, VGAE, SEAL-lite — AUC ladder.
Part B (generation): GraphRNN-S trained on 2-community graphs; degree and
clustering MMD of samples vs held-out graphs, against an ER baseline.
Saves w12_results.json + w12_graphs.json (sample edge lists for the widget).
"""
import json
import math
import os
import time

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
rng = np.random.default_rng(0)

# ═══════════════ Part A: link prediction on Cora ═══════════════════════════
data = Planetoid(root="data/Planetoid", name="Cora")[0]
N = data.num_nodes
# undirected unique edges
eu = data.edge_index[:, data.edge_index[0] < data.edge_index[1]]
E = eu.shape[1]
perm = torch.randperm(E, generator=torch.Generator().manual_seed(0))
n_test, n_val = int(0.1 * E), int(0.05 * E)
test_pos = eu[:, perm[:n_test]]
val_pos = eu[:, perm[n_test:n_test + n_val]]
train_pos = eu[:, perm[n_test + n_val:]]
# message graph: TRAIN edges only, symmetrized
msg = torch.cat([train_pos, train_pos.flip(0)], dim=1)
print(f"Cora LP split: {train_pos.shape[1]} train / {n_val} val / {n_test} test positives", flush=True)

edge_set = set(map(tuple, eu.t().tolist()))


def sample_negs(n, gen):
    out = []
    while len(out) < n:
        a = int(torch.randint(N, (1,), generator=gen))
        b = int(torch.randint(N, (1,), generator=gen))
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        if key not in edge_set:
            out.append(key)
    return torch.tensor(out).t()


gneg = torch.Generator().manual_seed(1)
test_neg = sample_negs(n_test, gneg)
val_neg = sample_negs(n_val, gneg)


def auc(pos_scores, neg_scores):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    s = np.concatenate([pos_scores, neg_scores])
    return roc_auc_score(y, s)


# adjacency sets from MESSAGE graph only (no test leakage)
adj = [set() for _ in range(N)]
for a, b in train_pos.t().tolist():
    adj[a].add(b); adj[b].add(a)


def cn_score(pairs):
    return np.array([len(adj[a] & adj[b]) for a, b in pairs.t().tolist()], dtype=float)


def aa_score(pairs):
    out = []
    for a, b in pairs.t().tolist():
        s = 0.0
        for c in adj[a] & adj[b]:
            d = len(adj[c])
            if d > 1:
                s += 1.0 / math.log(d)
        out.append(s)
    return np.array(out)


results = {}
results["CN"] = round(auc(cn_score(test_pos), cn_score(test_neg)), 3)
results["AA"] = round(auc(aa_score(test_pos), aa_score(test_neg)), 3)
print("heuristics:", {k: results[k] for k in ("CN", "AA")}, flush=True)


# ── VGAE ────────────────────────────────────────────────────────────────────
class VGAE(torch.nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.c1 = GCNConv(data.num_features, 64)
        self.cmu = GCNConv(64, d)
        self.clog = GCNConv(64, d)

    def encode(self, x, ei):
        h = F.relu(self.c1(x, ei))
        return self.cmu(h, ei), self.clog(h, ei)


def train_vgae(epochs=200):
    torch.manual_seed(0)
    m = VGAE().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=0.01)
    x, ei = data.x.to(DEV), msg.to(DEV)
    tp = train_pos.to(DEV)
    gen = torch.Generator().manual_seed(2)
    for ep in range(epochs):
        m.train(); opt.zero_grad()
        mu, logv = m.encode(x, ei)
        z = mu + torch.randn_like(mu) * (0.5 * logv).exp()
        neg = sample_negs(tp.shape[1], gen).to(DEV)
        pos_logit = (z[tp[0]] * z[tp[1]]).sum(1)
        neg_logit = (z[neg[0]] * z[neg[1]]).sum(1)
        recon = F.binary_cross_entropy_with_logits(pos_logit, torch.ones_like(pos_logit)) + \
                F.binary_cross_entropy_with_logits(neg_logit, torch.zeros_like(neg_logit))
        kl = -0.5 / N * (1 + logv - mu ** 2 - logv.exp()).sum(1).mean()
        (recon + kl).backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        mu, _ = m.encode(x, ei)
    return mu.cpu()


t0 = time.time()
Z = train_vgae()
sc = lambda P: (Z[P[0]] * Z[P[1]]).sum(1).numpy()
results["VGAE"] = round(auc(sc(test_pos), sc(test_neg)), 3)
print(f"VGAE: {results['VGAE']} [{time.time() - t0:.0f}s]", flush=True)

# ── SEAL-lite ───────────────────────────────────────────────────────────────
# 1-hop enclosing subgraphs, DRNL-ish labels, small GIN-style net.
from torch_geometric.data import Data as PygData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool


def enclosing(a, b):
    # the target edge must vanish EVERYWHERE: subgraph, and the distance labels
    adj_a = adj[a] - {b}
    adj_b = adj[b] - {a}
    nodes = {a, b} | adj_a | adj_b
    nodes = list(nodes)[:50]
    if a not in nodes: nodes.append(a)
    if b not in nodes: nodes.append(b)
    idx = {v: i for i, v in enumerate(nodes)}
    edges = []
    ns = set(nodes)
    for v in nodes:
        for u in adj[v]:
            if u in ns and v < u:
                if (v, u) != (min(a, b), max(a, b)):   # remove the target edge!
                    edges.append((idx[v], idx[u]))
    # DRNL-lite: label = (min(d_a, 2), min(d_b, 2)) one-hot, target edge excluded
    feats = []
    for v in nodes:
        da = 0 if v == a else (1 if v in adj_a else 2)
        db = 0 if v == b else (1 if v in adj_b else 2)
        f = torch.zeros(9)
        f[da * 3 + db] = 1.0
        feats.append(f)
    if edges:
        ei = torch.tensor(edges).t()
        ei = torch.cat([ei, ei.flip(0)], dim=1)
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)
    return PygData(x=torch.stack(feats), edge_index=ei)


def build_dataset(pos, neg, label):
    out = []
    for a, b in pos.t().tolist():
        d = enclosing(a, b); d.y = torch.tensor([1.0]); out.append(d)
    for a, b in neg.t().tolist():
        d = enclosing(a, b); d.y = torch.tensor([0.0]); out.append(d)
    return out


gen3 = torch.Generator().manual_seed(3)
train_sub_pos = train_pos[:, torch.randperm(train_pos.shape[1], generator=gen3)[:2000]]
train_sub_neg = sample_negs(2000, gen3)
print("extracting SEAL subgraphs...", flush=True)
t0 = time.time()
train_ds = build_dataset(train_sub_pos, train_sub_neg, None)
test_ds = build_dataset(test_pos, test_neg, None)
print(f"  extracted {len(train_ds)}+{len(test_ds)} in {time.time() - t0:.0f}s", flush=True)


class SealNet(torch.nn.Module):
    def __init__(self, h=32):
        super().__init__()
        mlp1 = torch.nn.Sequential(torch.nn.Linear(9, h), torch.nn.ReLU(), torch.nn.Linear(h, h))
        mlp2 = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.ReLU(), torch.nn.Linear(h, h))
        self.g1, self.g2 = GINConv(mlp1), GINConv(mlp2)
        self.out = torch.nn.Linear(h, 1)

    def forward(self, d):
        h = F.relu(self.g1(d.x, d.edge_index))
        h = F.relu(self.g2(h, d.edge_index))
        return self.out(global_add_pool(h, d.batch)).squeeze(-1)


torch.manual_seed(0)
net = SealNet().to(DEV)
opt = torch.optim.Adam(net.parameters(), lr=0.005)
loader = DataLoader(train_ds, batch_size=64, shuffle=True)
t0 = time.time()
for ep in range(8):
    for batch in loader:
        batch = batch.to(DEV)
        loss = F.binary_cross_entropy_with_logits(net(batch), batch.y)
        opt.zero_grad(); loss.backward(); opt.step()
net.eval()
scores = []
with torch.no_grad():
    for batch in DataLoader(test_ds, batch_size=128):
        scores.append(net(batch.to(DEV)).cpu().numpy())
scores = np.concatenate(scores)
results["SEAL"] = round(auc(scores[:n_test], scores[n_test:]), 3)
print(f"SEAL-lite: {results['SEAL']} [{time.time() - t0:.0f}s]", flush=True)

# ═══════════════ Part B: GraphRNN-S on community graphs ════════════════════
def community_graph(gen):
    n1, n2 = int(gen.integers(6, 10)), int(gen.integers(6, 10))
    g = nx.random_partition_graph([n1, n2], 0.7, 0.05, seed=int(gen.integers(1 << 30)))
    return nx.convert_node_labels_to_integers(g)


train_graphs = [community_graph(rng) for _ in range(200)]
test_graphs = [community_graph(rng) for _ in range(40)]
NMAX = max(g.number_of_nodes() for g in train_graphs + test_graphs)
M = 12   # BFS-band width


def bfs_sequence(g):
    order = list(nx.bfs_tree(g, source=0).nodes())
    rest = [v for v in g if v not in order]
    order += rest
    pos = {v: i for i, v in enumerate(order)}
    rows = []
    for i in range(1, g.number_of_nodes()):
        row = torch.zeros(M)
        for u in g[order[i]]:
            j = pos[u]
            if j < i and i - 1 - j < M:
                row[i - 1 - j] = 1.0
        rows.append(row)
    return torch.stack(rows) if rows else torch.zeros(0, M)


class GraphRNNS(torch.nn.Module):
    def __init__(self, h=64):
        super().__init__()
        self.gru = torch.nn.GRU(M, h, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Linear(h, 64), torch.nn.ReLU(), torch.nn.Linear(64, M))

    def forward(self, seq):
        out, _ = self.gru(seq)
        return self.head(out)


torch.manual_seed(0)
grnn = GraphRNNS().to(DEV)
opt = torch.optim.Adam(grnn.parameters(), lr=0.003)
seqs = [bfs_sequence(g) for g in train_graphs]
t0 = time.time()
for ep in range(60):
    tot = 0.0
    order = rng.permutation(len(seqs))
    for i in order:
        s = seqs[i]
        if len(s) < 2:
            continue
        inp = torch.cat([torch.ones(1, M), s[:-1]]).unsqueeze(0).to(DEV)   # SOS row
        tgt = s.unsqueeze(0).to(DEV)
        logit = grnn(inp)
        loss = F.binary_cross_entropy_with_logits(logit, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
print(f"GraphRNN-S trained [{time.time() - t0:.0f}s], final loss {tot / len(seqs):.3f}", flush=True)


def generate(gen_torch, n_nodes):
    grnn.eval()
    with torch.no_grad():
        rows = []
        h = None
        x = torch.ones(1, 1, M).to(DEV)
        for i in range(1, n_nodes):
            out, h = grnn.gru(x, h)
            p = torch.sigmoid(grnn.head(out))[0, 0]
            row = (torch.rand(M, generator=gen_torch).to(DEV) < p).float()
            rows.append(row.cpu())
            x = row.unsqueeze(0).unsqueeze(0).to(DEV)
    g = nx.Graph()
    g.add_nodes_from(range(n_nodes))
    for i, row in enumerate(rows, start=1):
        for k in range(M):
            j = i - 1 - k
            if j >= 0 and row[k] > 0:
                g.add_edge(i, j)
    return g


gt = torch.Generator().manual_seed(5)
gen_graphs = [generate(gt, int(rng.integers(12, 20))) for _ in range(40)]
er_graphs = [nx.gnp_random_graph(int(rng.integers(12, 20)), 0.35,
                                 seed=int(rng.integers(1 << 30))) for _ in range(40)]


def mmd_hist(gs1, gs2, stat):
    def hist(gs):
        vals = []
        for g in gs:
            vals.extend(stat(g))
        h, _ = np.histogram(vals, bins=np.arange(0, 1.05, 0.05) if max(vals or [1]) <= 1
                            else np.arange(0, max(vals) + 2))
        return h / max(h.sum(), 1)
    h1, h2 = hist(gs1), hist(gs2)
    n = max(len(h1), len(h2))
    h1 = np.pad(h1, (0, n - len(h1)))
    h2 = np.pad(h2, (0, n - len(h2)))
    return float(np.abs(h1 - h2).sum() / 2)   # total variation as a simple MMD proxy


deg_stat = lambda g: [d for _, d in g.degree()]
clu_stat = lambda g: list(nx.clustering(g).values())

gen_res = {
    "deg_TV_model": round(mmd_hist(gen_graphs, test_graphs, deg_stat), 3),
    "deg_TV_ER": round(mmd_hist(er_graphs, test_graphs, deg_stat), 3),
    "clu_TV_model": round(mmd_hist(gen_graphs, test_graphs, clu_stat), 3),
    "clu_TV_ER": round(mmd_hist(er_graphs, test_graphs, clu_stat), 3),
}
results["generation"] = gen_res
print("generation:", gen_res, flush=True)

sp = os.path.dirname(os.path.abspath(__file__))
json.dump(results, open(os.path.join(sp, "w12_results.json"), "w"), indent=1)
# export sample graphs for the widget (8 each)
export = {
    "train": [list(map(list, g.edges())) for g in train_graphs[:8]],
    "model": [list(map(list, g.edges())) for g in gen_graphs[:8]],
    "er": [list(map(list, g.edges())) for g in er_graphs[:8]],
}
json.dump(export, open(os.path.join(sp, "w12_graphs.json"), "w"))
print("saved results + widget graphs", flush=True)
