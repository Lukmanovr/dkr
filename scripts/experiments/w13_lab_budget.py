"""Lab-13 budget measurement: 12-epoch MPNN / MPNN+RWSE floors, SMOKE floors
(1500-molecule slice, 2 epochs), GPS 3-epoch loss drop, mask-invariance and
mask-sabotage diffs. Sets honest assert bands for make_lab13.py."""
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
rwse_secs = round(time.time() - t0)
print(f"RWSE precompute {rwse_secs}s", flush=True)


class Block(torch.nn.Module):
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
            dense, mask = to_dense_batch(x, batch)
            kpm = None if sabotage_mask else ~mask
            att, _ = self.mha(dense, dense, dense, key_padding_mask=kpm)
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

    def forward(self, d, sabotage_mask=False):
        x = self.emb(d.x.squeeze(-1))
        if self.rwse:
            x = x + self.rw_lin(d.rwse)
        ea = self.eemb(d.edge_attr)
        for b in self.blocks:
            x = b(x, d.edge_index, ea, d.batch, sabotage_mask)
        return self.head(global_add_pool(x, d.batch)).squeeze(-1)

    def node_reps(self, d, sabotage_mask=False):
        x = self.emb(d.x.squeeze(-1))
        if self.rwse:
            x = x + self.rw_lin(d.rwse)
        ea = self.eemb(d.edge_attr)
        for b in self.blocks:
            x = b(x, d.edge_index, ea, d.batch, sabotage_mask)
        return x


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
                for batch in DataLoader(split, batch_size=256):
                    batch = batch.to(DEV)
                    errs.append((net(batch) - batch.y).abs().cpu())
            maes.append(torch.cat(errs).mean().item())
        if maes[0] < best_va:
            best_va, best_te = maes[0], maes[1]
    dt = round(time.time() - t0)
    print(f"{name}: val {best_va:.3f} test {best_te:.3f} [{dt}s]", flush=True)
    return round(best_te, 3), dt


res = {"rwse_secs": rwse_secs}
res["mpnn12"] = run(False, False, 12, tr_l, "MPNN 12ep")
res["rwse12"] = run(False, True, 12, tr_l, "MPNN+RWSE 12ep")
res["mpnn_smoke"] = run(False, False, 2, tr_l[:1500], "MPNN smoke")
res["rwse_smoke"] = run(False, True, 2, tr_l[:1500], "RWSE smoke")

# GPS 3-epoch loss drop + mask checks
torch.manual_seed(0)
gps = Net(attn=True, rwse=True).to(DEV)
opt = torch.optim.Adam(gps.parameters(), lr=1e-3)
tl = DataLoader(tr_l[:4000], batch_size=128, shuffle=True)
losses = []
t0 = time.time()
for ep in range(3):
    gps.train()
    tot, nb = 0.0, 0
    for batch in tl:
        batch = batch.to(DEV)
        loss = F.l1_loss(gps(batch), batch.y)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    losses.append(round(tot / nb, 3))
res["gps3_losses"] = losses
res["gps3_secs"] = round(time.time() - t0)
print("GPS 3ep losses", losses, res["gps3_secs"], "s", flush=True)

# mask invariance: small graph alone vs batched with a larger graph
gps.eval()
small, big = te_l[0], max(te_l[:50], key=lambda d: d.num_nodes)
print("small n:", small.num_nodes, "big n:", big.num_nodes)
lone = DataLoader([small], batch_size=1)
pair = DataLoader([small, big], batch_size=2)
with torch.no_grad():
    b1 = next(iter(lone)).to(DEV)
    r1 = gps.node_reps(b1)
    b2 = next(iter(pair)).to(DEV)
    r2 = gps.node_reps(b2)[b2.batch == 0]
    diff_ok = (r1 - r2).abs().max().item()
    r2s = gps.node_reps(b2, sabotage_mask=True)[b2.batch == 0]
    diff_sab = (r1 - r2s).abs().max().item()
res["mask_diff_ok"] = diff_ok
res["mask_diff_sabotage"] = diff_sab
print(f"mask ok diff {diff_ok:.2e} · sabotage diff {diff_sab:.2e}", flush=True)

sp = os.path.dirname(os.path.abspath(__file__))
json.dump(res, open(os.path.join(sp, "w13_lab_budget.json"), "w"), indent=1)
print("saved", flush=True)
