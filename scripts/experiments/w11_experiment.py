"""Week 11 measured scaling table on ogbn-arxiv: full-batch GCN vs hand-rolled
neighbor-sampled SAGE vs poor-man's Cluster-GCN vs SGC. Accuracy / epoch time /
peak GPU memory. Plus the neighborhood-explosion counts."""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

torch.serialization.add_safe_globals = torch.serialization.add_safe_globals  # noqa
_load = torch.load
torch.load = lambda *a, **k: _load(*a, **{**k, "weights_only": False})
from ogb.nodeproppred import PygNodePropPredDataset  # noqa: E402
ds = PygNodePropPredDataset("ogbn-arxiv", root="data/ogb")
torch.load = _load
split = ds.get_idx_split()
data = ds[0]
data.edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
data.y = data.y.squeeze()
N, E = data.num_nodes, data.edge_index.shape[1]
print(f"arxiv: {N:,} nodes, {E:,} directed edges", flush=True)
DEV = torch.device("cuda")

# ── CSR adjacency for hand sampling / hop counting ─────────────────────────
perm = torch.argsort(data.edge_index[0])
dst_sorted = data.edge_index[1][perm]
deg = torch.zeros(N, dtype=torch.long).index_add_(0, data.edge_index[0], torch.ones(E, dtype=torch.long))
ptr = torch.cat([torch.zeros(1, dtype=torch.long), deg.cumsum(0)])
avg_deg = deg.float().mean().item()
print(f"avg degree {avg_deg:.1f}", flush=True)

# neighborhood explosion: avg unique nodes within L hops, over 200 seeds
g = torch.Generator().manual_seed(0)
seeds = torch.randperm(N, generator=g)[:200]
hops_avg = []
for L in (1, 2, 3):
    tot = 0
    for s in seeds.tolist():
        frontier = {s}
        seen = {s}
        for _ in range(L):
            nxt = set()
            for v in frontier:
                nxt.update(dst_sorted[ptr[v]:ptr[v + 1]].tolist())
            frontier = nxt - seen
            seen |= nxt
        tot += len(seen)
    hops_avg.append(round(tot / len(seeds)))
print("avg L-hop neighborhood sizes:", hops_avg, flush=True)

tr_idx, va_idx, te_idx = split["train"], split["valid"], split["test"]
results = {"explosion": {"avg_deg": round(avg_deg, 1), "hops": hops_avg}}


def accuracy(logits, idx):
    return (logits[idx].argmax(1) == data.y[idx]).float().mean().item()


# ── 1. full-batch GCN (2 layers, hidden 128) ───────────────────────────────
def run_fullbatch(epochs=100):
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    d = data.to(DEV)
    tr = tr_idx.to(DEV)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = GCNConv(128, 128)
            self.c2 = GCNConv(128, 40)

        def forward(self, x, ei):
            return self.c2(F.relu(self.c1(x, ei)), ei)

    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    t0 = time.time()
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        out = net(d.x, d.edge_index)
        F.cross_entropy(out[tr], d.y.to(DEV)[tr]).backward()
        opt.step()
    per_ep = (time.time() - t0) / epochs
    mem = torch.cuda.max_memory_allocated() / 2 ** 20   # TRAINING peak
    net.eval()
    with torch.no_grad():
        logits = net(d.x, d.edge_index).cpu()
    data.to("cpu")
    return accuracy(logits, te_idx), per_ep, mem


acc, t, m = run_fullbatch()
results["fullbatch_gcn"] = {"test": round(acc, 3), "s_per_epoch": round(t, 2), "peak_mb": round(m)}
print("full-batch GCN:", results["fullbatch_gcn"], flush=True)

# ── 2. hand-rolled neighbor-sampled SAGE (2 layers, fanout 10/10) ──────────
def sample_block(batch_nodes, fanout, gen):
    """One layer of neighbor sampling: returns (unique input nodes, edge list
    into the batch positions)."""
    srcs, dsts = [], []
    for i, v in enumerate(batch_nodes.tolist()):
        lo, hi = ptr[v].item(), ptr[v + 1].item()
        n = hi - lo
        if n == 0:
            continue
        if n <= fanout:
            nb = dst_sorted[lo:hi]
        else:
            sel = torch.randint(n, (fanout,), generator=gen)
            nb = dst_sorted[lo + sel]
        srcs.append(nb)
        dsts.append(torch.full((len(nb),), i, dtype=torch.long))
    src = torch.cat(srcs) if srcs else torch.zeros(0, dtype=torch.long)
    dst = torch.cat(dsts) if dsts else torch.zeros(0, dtype=torch.long)
    uniq, inv = torch.unique(torch.cat([batch_nodes, src]), return_inverse=True)
    # positions: first len(batch) entries are the batch itself
    return uniq, torch.stack([inv[len(batch_nodes):], dst]), inv[:len(batch_nodes)]


def run_sampled(epochs=5, batch=1024, fanout=10):
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = SAGEConv(128, 128)
            self.c2 = SAGEConv(128, 40)

    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    t0 = time.time()
    for ep in range(epochs):
        order = tr_idx[torch.randperm(len(tr_idx), generator=gen)]
        for i in range(0, len(order), batch):
            b = order[i:i + batch]
            # two-layer sampling: layer-2 needs neighbors of neighbors
            u1, e1, pos1 = sample_block(b, fanout, gen)          # inner layer targets = b
            u2, e2, pos2 = sample_block(u1, fanout, gen)         # outer layer targets = u1
            # manual SAGE-mean aggregation with scatter ops
            xs = data.x[u2].to(DEV)
            src2, dst2 = e2[0].to(DEV), e2[1].to(DEV)
            agg = torch.zeros(len(u1), 128, device=DEV).index_add_(0, dst2, xs[src2])
            cnt = torch.zeros(len(u1), device=DEV).index_add_(0, dst2, torch.ones(len(src2), device=DEV)).clamp(min=1)
            h1 = F.relu(net.c1.lin_l(xs[pos2.to(DEV)]) + net.c1.lin_r(agg / cnt.unsqueeze(1)))
            src1, dst1 = e1[0].to(DEV), e1[1].to(DEV)
            agg1 = torch.zeros(len(b), 128, device=DEV).index_add_(0, dst1, h1[src1])
            cnt1 = torch.zeros(len(b), device=DEV).index_add_(0, dst1, torch.ones(len(src1), device=DEV)).clamp(min=1)
            out = net.c2.lin_l(h1[pos1.to(DEV)]) + net.c2.lin_r(agg1 / cnt1.unsqueeze(1))
            loss = F.cross_entropy(out, data.y[b].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    per_ep = (time.time() - t0) / epochs
    mem = torch.cuda.max_memory_allocated() / 2 ** 20   # TRAINING peak
    # full-graph inference on GPU (cheap for arxiv)
    net.eval()
    with torch.no_grad():
        x = data.x.to(DEV)
        ei = data.edge_index.to(DEV)
        h = F.relu(net.c1(x, ei))
        logits = net.c2(h, ei).cpu()
    return accuracy(logits, te_idx), per_ep, mem


acc, t, m = run_sampled()
results["sampled_sage"] = {"test": round(acc, 3), "s_per_epoch": round(t, 2), "peak_mb": round(m)}
print("sampled SAGE:", results["sampled_sage"], flush=True)

# ── 3. poor-man's Cluster-GCN: random balanced partition, intra-cluster ────
def run_cluster(epochs=15, n_parts=40):
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    part = torch.randint(n_parts, (N,), generator=gen)
    # precompute per-cluster node lists and intra-cluster subgraph edges
    clusters = []
    src, dst = data.edge_index
    same = part[src] == part[dst]
    for c in range(n_parts):
        nodes = (part == c).nonzero(as_tuple=True)[0]
        mask = same & (part[src] == c)
        es, ed = src[mask], dst[mask]
        remap = torch.full((N,), -1, dtype=torch.long)
        remap[nodes] = torch.arange(len(nodes))
        clusters.append((nodes, torch.stack([remap[es], remap[ed]])))

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = GCNConv(128, 128)
            self.c2 = GCNConv(128, 40)

        def forward(self, x, ei):
            return self.c2(F.relu(self.c1(x, ei)), ei)

    net = Net().to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    tr_mask = torch.zeros(N, dtype=torch.bool)
    tr_mask[tr_idx] = True
    t0 = time.time()
    for ep in range(epochs):
        for nodes, ei in clusters:
            x = data.x[nodes].to(DEV)
            out = net(x, ei.to(DEV))
            m2 = tr_mask[nodes]
            if m2.sum() == 0:
                continue
            loss = F.cross_entropy(out[m2.to(DEV)], data.y[nodes][m2].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
    per_ep = (time.time() - t0) / epochs
    mem = torch.cuda.max_memory_allocated() / 2 ** 20   # TRAINING peak
    net.eval()
    with torch.no_grad():
        logits = net(data.x.to(DEV), data.edge_index.to(DEV)).cpu()
    frac_kept = same.float().mean().item()
    return accuracy(logits, te_idx), per_ep, mem, frac_kept


acc, t, m, kept = run_cluster()
results["cluster_gcn"] = {"test": round(acc, 3), "s_per_epoch": round(t, 2), "peak_mb": round(m),
                          "edges_kept": round(kept, 3)}
print("cluster GCN:", results["cluster_gcn"], flush=True)

# ── 4. SGC: precompute A^2 X on CPU, logistic regression ───────────────────
def run_sgc(K=2, epochs=300):
    t0 = time.time()
    # symmetric normalized adjacency with self loops, sparse CPU
    src, dst = data.edge_index
    deg2 = torch.zeros(N).index_add_(0, src, torch.ones(E)) + 1
    vals = (deg2[src] * deg2[dst]).rsqrt()
    A = torch.sparse_coo_tensor(data.edge_index, vals, (N, N)).coalesce()
    selfw = 1.0 / deg2
    X = data.x.clone()
    for _ in range(K):
        X = torch.sparse.mm(A, X) + selfw.unsqueeze(1) * X
    pre = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(0)
    W = torch.nn.Linear(128, 40).to(DEV)
    opt = torch.optim.Adam(W.parameters(), lr=0.01)
    Xtr = X[tr_idx].to(DEV)
    ytr = data.y[tr_idx].to(DEV)
    t0 = time.time()
    for ep in range(epochs):
        opt.zero_grad()
        F.cross_entropy(W(Xtr), ytr).backward()
        opt.step()
    per_ep = (time.time() - t0) / epochs
    with torch.no_grad():
        logits = W(X.to(DEV)).cpu()
    mem = torch.cuda.max_memory_allocated() / 2 ** 20
    return accuracy(logits, te_idx), pre, per_ep, mem


acc, pre, t, m = run_sgc()
results["sgc"] = {"test": round(acc, 3), "precompute_s": round(pre, 1),
                  "s_per_epoch": round(t, 4), "peak_mb": round(m)}
print("SGC:", results["sgc"], flush=True)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w11_results.json")
json.dump(results, open(out, "w"), indent=1)
print("saved", out, flush=True)
