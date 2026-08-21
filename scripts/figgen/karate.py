#!/usr/bin/env python
"""Generate assets/figures/fig-w1-hero.html — the real Zachary karate club (1977),
colored by the actual post-split memberships, with the one node Zachary's model
mispredicted marked. Layout: deterministic spring embedding computed here.

Protocol F1: data-derived figure → generated, never hand-placed.
Run: python scripts/figgen/karate.py
"""
import math
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "assets" / "figures" / "fig-w1-hero.html"

# Zachary (1977), standard 0-indexed 78-edge list (as shipped by networkx).
EDGES = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),(0,12),(0,13),
         (0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),(1,17),(1,19),(1,21),(1,30),
         (2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),(2,32),(3,7),(3,12),(3,13),(4,6),
         (4,10),(5,6),(5,10),(5,16),(6,16),(8,30),(8,32),(8,33),(9,33),(13,33),(14,32),
         (14,33),(15,32),(15,33),(18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),
         (23,25),(23,27),(23,29),(23,32),(23,33),(24,25),(24,27),(24,31),(25,31),(26,29),
         (26,33),(27,33),(28,31),(28,33),(29,32),(29,33),(30,32),(30,33),(31,32),(31,33),
         (32,33)]
N = 34
# Actual club joined after the split (networkx 'club' attribute).
MR_HI = {0,1,2,3,4,5,6,7,8,10,11,12,13,16,17,19,21}
MISPREDICTED = 8  # the single node Zachary's min-cut model got wrong (33/34 correct)

C_HI, C_OFF = "#d9a62e", "#0f8377"   # gold = Mr. Hi's faction, teal = the Officer's

# ── deterministic spring layout ──────────────────────────────────────────────
rng = random.Random(7)
pos = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(N)]
adj = set(EDGES) | {(b, a) for a, b in EDGES}
for it in range(600):
    t = 0.08 * (1 - it / 600)
    disp = [[0.0, 0.0] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            dx, dy = pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]
            d2 = dx * dx + dy * dy + 1e-6
            f = 0.02 / d2                      # repulsion
            disp[i][0] += f * dx; disp[i][1] += f * dy
    for a, b in EDGES:
        dx, dy = pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]
        d = math.sqrt(dx * dx + dy * dy) + 1e-6
        f = 0.9 * d                            # attraction
        disp[a][0] -= f * dx / d * 0.05; disp[a][1] -= f * dy / d * 0.05
        disp[b][0] += f * dx / d * 0.05; disp[b][1] += f * dy / d * 0.05
    for i in range(N):
        m = math.sqrt(disp[i][0] ** 2 + disp[i][1] ** 2) + 1e-9
        pos[i][0] += disp[i][0] / m * min(m, t)
        pos[i][1] += disp[i][1] / m * min(m, t)

xs, ys = [p[0] for p in pos], [p[1] for p in pos]
def sx(x): return 60 + (x - min(xs)) / (max(xs) - min(xs)) * 560
def sy(y): return 58 + (y - min(ys)) / (max(ys) - min(ys)) * 260
P = [(round(sx(x), 1), round(sy(y), 1)) for x, y in pos]

edges_svg = "\n    ".join(
    f'<line x1="{P[a][0]}" y1="{P[a][1]}" x2="{P[b][0]}" y2="{P[b][1]}"/>' for a, b in EDGES)

nodes_svg = []
for i in range(N):
    x, y = P[i]
    c = C_HI if i in MR_HI else C_OFF
    r = 13 if i in (0, 33) else 8              # the two leaders, larger
    ring = ' stroke="#cf4a30" stroke-width="3"' if i == MISPREDICTED else ""
    nodes_svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}"{ring}/>')
nodes = "\n    ".join(nodes_svg)
PLACED = []  # centers+halfwidths of labels already placed, so labels repel each other


def place_label(anchor_node, text, size=12.5):
    """Choose the offset that keeps the label's box farthest from all nodes AND
    from labels placed before it."""
    ax, ay = P[anchor_node]
    w = len(text) * size * 0.52
    best, best_score = None, -1.0
    for k in range(16):
        ang = 2 * math.pi * k / 16
        for rad in (36, 50, 66, 84):
            cx, cy = ax + rad * math.cos(ang), ay + rad * math.sin(ang)
            if not (20 + w / 2 <= cx <= 740 - w / 2 and 24 <= cy <= 336):
                continue
            score = min(
                math.hypot(cx + dx - px, cy - py)
                for (px, py) in P
                for dx in (-w / 2, 0, w / 2))
            for (ox, oy, ow) in PLACED:
                gap_x = max(0.0, abs(cx - ox) - (w + ow) / 2)
                gap_y = abs(cy - oy)
                score = min(score, math.hypot(gap_x, gap_y))
            if score > best_score:
                best_score, best = score, (cx, cy)
    PLACED.append((best[0], best[1], w))
    return best

hix, hiy = place_label(0, "Mr. Hi (instructor)")
ofx, ofy = place_label(33, "the Officer (president)")
mx, my = place_label(MISPREDICTED, "the one miss")
lx0, ly0 = P[0]
lx33, ly33 = P[33]
lxm, lym = P[MISPREDICTED]

svg = f"""```{{=html}}
<figure class="dkr-fig">
<svg viewBox="0 0 760 400" role="img" aria-label="The real Zachary karate club network of 1977, colored by which side each member actually joined after the split; one ringed node is the only member the graph model mispredicted">
  <!-- generated by scripts/figgen/karate.py from the real 78-edge dataset -->
  <g stroke="var(--dkr-muted, #8e8e9a)" stroke-width="1" opacity="0.45">
    {edges_svg}
  </g>
  <g>
    {nodes}
  </g>
  <g stroke-width="1.2" stroke-dasharray="2,3" opacity="0.85" fill="none">
    <line x1="{hix}" y1="{hiy}" x2="{lx0}" y2="{ly0}" stroke="#b98a1e"/>
    <line x1="{ofx}" y1="{ofy}" x2="{lx33}" y2="{ly33}" stroke="var(--dkr-blue-dark, #0a5f56)"/>
    <line x1="{mx}" y1="{my}" x2="{lxm}" y2="{lym}" stroke="#cf4a30"/>
  </g>
  <g font-family="'Source Sans 3', sans-serif" font-size="12.5" font-weight="700" text-anchor="middle">
    <text x="{hix}" y="{hiy}" fill="#b98a1e">Mr. Hi (instructor)</text>
    <text x="{ofx}" y="{ofy}" fill="var(--dkr-blue-dark, #0a5f56)">the Officer (president)</text>
    <text x="{mx}" y="{my}" fill="#cf4a30" font-weight="600">the one miss</text>
  </g>
  <g font-family="'Source Sans 3', sans-serif">
    <rect x="180" y="352" width="400" height="30" rx="15" fill="var(--dkr-green, #199473)"/>
    <text x="380" y="372" text-anchor="middle" font-size="14" font-weight="700" fill="var(--dkr-bg, #fff)">structure alone predicted 33 of 34 allegiances — in 1977</text>
  </g>
  <g font-family="'Source Sans 3', sans-serif" font-size="12" fill="var(--dkr-muted, #8e8e9a)">
    <circle cx="80" cy="30" r="7" fill="{C_HI}"/><text x="93" y="34">joined Mr. Hi's new club</text>
    <circle cx="280" cy="30" r="7" fill="{C_OFF}"/><text x="293" y="34">stayed with the Officer</text>
    <text x="640" y="34" text-anchor="end">friendship = edge · 34 members · 78 ties</text>
  </g>
</svg>
</figure>
```
"""
OUT.write_text(svg, encoding="utf-8")
print("wrote", OUT)
