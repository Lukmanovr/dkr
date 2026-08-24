"""Week 13 measured ladder on ZINC-subset: MPNN-only vs MPNN+RWSE vs
GPS-lite (parallel MPNN + global attention + RWSE). Test MAE each, same
budget. Also: RWSE hand-verification numbers on the triangle-with-tail."""
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.utils import to_dense_batch

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tr = ZINC("data/ZINC", subset=True, split="train")
va = ZINC("data/ZINC", subset=True, split="val")
te = ZINC("data/ZINC", subset=True, split="test")
print(f"ZINC subset: {len(tr)}/{len(va)}/{len(te)} molecules", flush=True)

K_RWSE = 16


def add_rwse(ds):
    out = []
    for d in ds:
        A = torch.zeros(d.num_nodes, d.num_nodes)
        A[d.edge_index[0], d.edge_index[1]] = 1
        deg = A.sum(1).clamp(min=1)
        M = A / deg.unsqueeze(1)
        P = torch.eye(d.num_nodes)
        rw = []
        for _ in range(K_RWSE):
            P = P @ M
            rw.append(P.diagonal().clone())
        d.rwse = torch.stack(rw, dim=1)
        out.append(d)
    return out


t0 = time.time()
tr_l, va_l, te_l = add_rwse(list(tr)), add_rwse(list(va)), add_rwse(list(te))
print(f"RWSE precomputed in {time.time() - t0:.0f}s", flush=True)


class Block(torch.nn.Module):
    """One GPS-ish layer: local GINE + optional global attention, summed."""
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

    def forward(self, x, ei, ea, batch):
        local = self.conv(x, ei, ea)
        if self.attn:
            dense, mask = to_dense_batch(x, batch)
            att, _ = self.mha(dense, dense, dense, key_padding_mask=~mask)
            glob = att[mask]
            x = self.ln(x + local + glob)
        else:
            x = x + local
        return self.ln2(x + self.ffn(x))


class Net(torch.nn.Module):
    def __init__(self, h=64, layers=4, attn=False, rwse=False):
        super().__init__()
        self.rwse = rwse
        self.emb = torch.nn.Embedding(28, h)
        self.eemb = torch.nn.Embedding(4, h)
        if rwse:
            self.rw_lin = torch.nn.Linear(K_RWSE, h)
        self.blocks = torch.nn.ModuleList([Block(h, attn) for _ in range(layers)])
        self.head = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.ReLU(),
                                        torch.nn.Linear(h, 1))

    def forward(self, d):
        x = self.emb(d.x.squeeze(-1))
        if self.rwse:
            x = x + self.rw_lin(d.rwse)
        ea = self.eemb(d.edge_attr)
        for b in self.blocks:
            x = b(x, d.edge_index, ea, d.batch)
        return self.head(global_add_pool(x, d.batch)).squeeze(-1)


def run(attn, rwse, epochs=100, name=""):
    torch.manual_seed(0)
    net = Net(attn=attn, rwse=rwse).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tl = DataLoader(tr_l, batch_size=128, shuffle=True)
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
                for batch in DataLoader(split, batch_size=256):
                    batch = batch.to(DEV)
                    errs.append((net(batch) - batch.y).abs().cpu())
            maes.append(torch.cat(errs).mean().item())
        if maes[0] < best_va:
            best_va, best_te = maes[0], maes[1]
    print(f"{name}: val {best_va:.3f} · TEST MAE {best_te:.3f}  [{time.time() - t0:.0f}s]", flush=True)
    return round(best_te, 3), round(time.time() - t0)


results = {}
results["MPNN"] = run(False, False, name="MPNN-only")
results["MPNN_RWSE"] = run(False, True, name="MPNN + RWSE")
results["GPS"] = run(True, True, name="GPS-lite (MPNN ∥ attention + RWSE)")

# RWSE hand numbers: triangle (0,1,2) with tail 2-3
import networkx as nx
G = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3)])
A = nx.to_numpy_array(G)
M = A / A.sum(1, keepdims=True)
P = np.eye(4)
hand = {}
for k in range(1, 5):
    P = P @ M
    hand[k] = [round(float(P[i, i]), 4) for i in range(4)]
print("triangle+tail RWSE diag:", hand, flush=True)
results["hand_rwse"] = hand

sp = os.path.dirname(os.path.abspath(__file__))
json.dump(results, open(os.path.join(sp, "w13_results.json"), "w"), indent=1)
print("saved", flush=True)
