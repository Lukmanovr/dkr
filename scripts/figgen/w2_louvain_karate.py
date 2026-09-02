#!/usr/bin/env python
"""Generate assets/figures/fig-w2-louvain-karate.html — Louvain on Zachary's karate
club versus the real 1977 split (lecture 2, §4.2, the resolution lesson).

Teaching point: at the default resolution Louvain does NOT return the two 1977
factions. It returns FOUR communities whose modularity is HIGHER than the
two-faction split's, and the two factions are (almost) unions of those four.
Every number on the figure is measured here, never asserted; the generator
refuses to run if the measured facts stop matching the figure's wording.

Reproducibility
  * networkx.community.louvain_communities(G, seed=0, weight=None) — SEED below; the
    networkx version actually used is written into the SVG comment (measured with 3.6.1).
    weight=None is deliberate: since networkx 3.x karate_club_graph() carries Zachary's
    interaction counts as edge weights, and Louvain/modularity would silently use them.
    The lecture's club is the 78 unweighted ties (m = 78, binary A), and Lab 2 runs this
    exact call with weight=None, so the figure measures that graph. (For the record the
    weighted defaults also give four communities and the same lone exception, member 8,
    with Q = 0.444 vs 0.391.)
  * Lab 2 runs the identical call with seed 0 (2026-09-02) and then, as a seed-sensitivity
    check, with its general SEED = 42 (= LAB_SEED here). Louvain is
    seed-sensitive on a 34-node graph: measured, seed 42 stops at THREE communities
    (Q = 0.385, still above the two-faction split), while seeds 0, 8, 26, 28, ... give
    this figure's four. The generator measures the lab-seed result and writes it into
    the caption, so the reader is told exactly what the lab will print.
  * The 34 node positions are PARSED out of fig-w2-fiedler.html (the lecture's
    earlier karate figure) so the club looks identical across the lecture's
    figures; no new layout is invented here.

Run: python scripts/figgen/w2_louvain_karate.py
"""
import math
import re
from pathlib import Path

import networkx as nx

from karate_common import EDGES as K_EDGES, N as K_N, MR_HI, fiedler_vector

ROOT = Path(__file__).resolve().parents[2]
FIGDIR = ROOT / "assets" / "figures"
SRC = FIGDIR / "fig-w2-fiedler.html"
OUT = FIGDIR / "fig-w2-louvain-karate.html"
SEED = 0
LAB_SEED = 42          # scripts/labgen/make_lab02.py: louvain_communities(G, seed=SEED, weight=None)
SANS = "'Source Sans 3', sans-serif"
MONO = "'JetBrains Mono', monospace"
W, H = 760, 540

# ═══════════════ 1. positions: the exact coordinates of the Fiedler figure ═══════
txt = SRC.read_text(encoding="utf-8")
edge_group = txt.split("</g>")[0]          # the first <g> of that figure is its 78-edge structure group
edge_lines = re.findall(r'<line x1="([\d.]+)" y1="([\d.]+)" x2="([\d.]+)" y2="([\d.]+)"', edge_group)
assert len(edge_lines) == len(K_EDGES) == 78, f"expected 78 edge lines in {SRC.name}, found {len(edge_lines)}"
POS = {}
for (a, b), (x1, y1, x2, y2) in zip(K_EDGES, edge_lines):   # w2_figs.py writes edges in K_EDGES order
    for node, pt in ((a, (float(x1), float(y1))), (b, (float(x2), float(y2)))):
        assert POS.setdefault(node, pt) == pt, f"member {node} has two different coordinates in {SRC.name}"
assert len(POS) == K_N
node_circles = {(float(x), float(y))
                for x, y in re.findall(r'<circle cx="([\d.]+)" cy="([\d.]+)" r="(?:10|12|15)"', txt)}
assert node_circles == set(POS.values()), "parsed positions do not match the Fiedler figure's node circles"

# ═══════════════ 2. Louvain versus the 1977 split (all measured) ═════════════════
G = nx.karate_club_graph()
assert sorted(tuple(sorted(e)) for e in G.edges()) == sorted(K_EDGES), "networkx's karate edge list drifted"
FACTION = {i: G.nodes[i]["club"] for i in G}                   # "Mr. Hi" | "Officer"
assert {i for i in G if FACTION[i] == "Mr. Hi"} == MR_HI
M = G.number_of_edges()
DEG = dict(G.degree())

assert G.size(weight="weight") != M, "networkx no longer weights the club; the weight=None notes are stale"
COMMS = [sorted(c) for c in nx.community.louvain_communities(G, seed=SEED, weight=None)]
TWO = [sorted(MR_HI), sorted(set(G) - MR_HI)]
LAB_COMMS = [sorted(c) for c in nx.community.louvain_communities(G, seed=LAB_SEED, weight=None)]
LAB_K = len(LAB_COMMS)
LAB_Q = nx.community.modularity(G, LAB_COMMS, weight=None)


def modularity_by_hand(parts):
    """Q = sum_c [ e_c/m - (D_c/2m)^2 ] — the community form of the lecture's
    @eq-w02-modularity, evaluated from the raw edge list as an independent check
    on networkx's modularity()."""
    q = 0.0
    for c in parts:
        cs = set(c)
        e_c = sum(1 for a, b in K_EDGES if a in cs and b in cs)
        D_c = sum(DEG[i] for i in c)
        q += e_c / M - (D_c / (2 * M)) ** 2
    return q


Q_LOUVAIN = nx.community.modularity(G, COMMS, weight=None)
Q_TWO = nx.community.modularity(G, TWO, weight=None)
assert abs(Q_LOUVAIN - modularity_by_hand(COMMS)) < 1e-12
assert abs(Q_TWO - modularity_by_hand(TWO)) < 1e-12
assert Q_LOUVAIN > Q_TWO, "the teaching point needs Q(Louvain) > Q(two factions)"
K = len(COMMS)
# the caption's sentence about the lab assumes: a different, coarser-than-four but
# finer-than-two local optimum that still beats the two-faction split
assert 2 < LAB_K < K and LAB_Q > Q_TWO and abs(LAB_Q - Q_LOUVAIN) > 1e-9, \
    f"Lab 2's seed {LAB_SEED} now gives {LAB_K} communities with Q = {LAB_Q:.3f}; re-word the caption"


def majority(c):
    hi = sum(1 for i in c if i in MR_HI)
    return "Mr. Hi" if 2 * hi > len(c) else "Officer"


# "the factions are unions of the communities" fails exactly at the members whose
# own 1977 side differs from their Louvain community's majority side
VIOLATORS = sorted(i for c in COMMS for i in c if FACTION[i] != majority(c))
MIXED = [c for c in COMMS if len({FACTION[i] for i in c}) > 1]
MEMBERS_IN_MIXED = sum(len(c) for c in MIXED)
PURE_MEMBERS = K_N - MEMBERS_IN_MIXED

# the figure's wording ("four", "member 8 ... joined Mr. Hi") is tied to these facts
assert K == 4, f"Louvain (seed {SEED}) returned {K} communities; the figure's wording assumes four"
assert VIOLATORS == [8] and FACTION[8] == "Mr. Hi", \
    f"violators changed to {VIOLATORS}; re-word the callout and caption before regenerating"
assert len(MIXED) == 1
V = VIOLATORS[0]
MIXED_SIZE = len(MIXED[0])

# cross-reference for the caption: member 8 is also one of the Fiedler figure's misses
fv = fiedler_vector()
sign = 1 if fv[0] > 0 else -1
fiedler_hi = {i for i in range(K_N) if fv[i] * sign > 0}
fiedler_wrong = {i for i in range(K_N) if (i in fiedler_hi) != (i in MR_HI)}
if len(fiedler_wrong) > K_N / 2:
    fiedler_wrong = set(range(K_N)) - fiedler_wrong
assert V in fiedler_wrong, "caption says member 8 is also a Fiedler miss; that stopped being true"

# ═══════════════ 3. colour roles (computed, not hand-assigned) ═══════════════════
# The larger community on each 1977 side is that side's "core", the smaller its
# "satellite". Warm hues (accent, yellow) for Mr. Hi's side, cool (teal, purple)
# for the Officer's, so the two-ness survives even before the reader finds the rings.
TOKENS = {
    "accent": "var(--dkr-accent, #d9603b)",
    "yellow": "var(--dkr-yellow, #d9a62e)",
    "blue": "var(--dkr-blue, #0f8377)",
    "purple": "var(--dkr-purple, #7c5cd6)",
}
SIDE = {s: sorted([c for c in COMMS if majority(c) == s], key=len, reverse=True)
        for s in ("Mr. Hi", "Officer")}
assert len(SIDE["Mr. Hi"]) == 2 and len(SIDE["Officer"]) == 2, "colour roles assume two communities per side"
COMM_INFO = [  # (members, colour token key, majority side)
    (SIDE["Mr. Hi"][0], "accent", "Mr. Hi"),
    (SIDE["Mr. Hi"][1], "yellow", "Mr. Hi"),
    (SIDE["Officer"][0], "blue", "Officer"),
    (SIDE["Officer"][1], "purple", "Officer"),
]
COLOR_OF = {i: TOKENS[tok] for members, tok, _ in COMM_INFO for i in members}
assert len(COLOR_OF) == K_N

RING = "var(--dkr-text, #1c1c21)"           # ring = joined Mr. Hi's new club in 1977
RING_W = 2.2
RED = "var(--dkr-red, #cf4a30)"
MUTED = "var(--dkr-muted, #6e6e7a)"
TEXT = "var(--dkr-text, #1c1c21)"
R_NODE, R_LEADER = 10, 15                   # Lecture 1 radii; the two leaders (0 = Mr. Hi, 33 = the Officer)


def node_r(i):
    return R_LEADER if i in (0, 33) else R_NODE


def outer_r(i):
    return node_r(i) + (RING_W / 2 if i in MR_HI else 0)


# ═══════════════ 4. annotation placement, self-checked against the layout ═══════
def text_w(s, size, mono=False):
    return len(s) * size * (0.60 if mono else 0.50)


def assert_clear(box, pad=4, ignore=()):
    """A label box must stay >= pad px from every node's outer rim."""
    x0, y0, x1, y1 = box
    for i, (x, y) in POS.items():
        if i in ignore:
            continue
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        gap = math.hypot(dx, dy) - outer_r(i)
        assert gap >= pad, f"label box {box} is {gap:.1f}px from member {i} (need {pad})"


def seg_point_dist(ax, ay, bx, by, px, py):
    vx, vy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def best_leader(y_start, x_lo, x_hi, target, end_gap):
    """Pick the leader start on a horizontal line that keeps the dotted leader
    farthest from every other node; returns (sx, sy, ex, ey, clearance)."""
    tx, ty = POS[target]
    best = None
    for xs in range(int(x_lo), int(x_hi) + 1, 4):
        dx, dy = tx - xs, ty - y_start
        d = math.hypot(dx, dy)
        ex, ey = tx - dx / d * end_gap, ty - dy / d * end_gap
        clearance = min(seg_point_dist(xs, y_start, ex, ey, *POS[i]) - outer_r(i)
                        for i in POS if i != target)
        if best is None or clearance > best[-1]:
            best = (xs, y_start, ex, ey, clearance)
    assert best[-1] >= 3, f"no leader route clears the nodes (best clearance {best[-1]:.1f}px)"
    return best


parts = []

# edges: identical style to the Fiedler figure
parts.append('  <g stroke="var(--dkr-muted, #8e8e9a)" stroke-width="1.3" opacity="0.5">')
for a, b in K_EDGES:
    parts.append(f'    <line x1="{POS[a][0]}" y1="{POS[a][1]}" x2="{POS[b][0]}" y2="{POS[b][1]}"/>')
parts.append("  </g>")

# the exception's halo sits behind the discs so a neighbouring disc may overlap it
HALO_R = 15
vx, vy = POS[V]
parts.append(f'  <circle cx="{vx}" cy="{vy}" r="{HALO_R}" fill="none" stroke="{RED}" stroke-width="1.8" stroke-dasharray="3,2.5" opacity="0.9"/>')

# members: fill = Louvain community, ring = joined Mr. Hi's new club in 1977
for i in range(K_N):
    x, y = POS[i]
    ring = f' stroke="{RING}" stroke-width="{RING_W}"' if i in MR_HI else ""
    parts.append(f'  <circle cx="{x}" cy="{y}" r="{node_r(i)}" fill="{COLOR_OF[i]}" opacity="0.9"{ring}/>')

# one label per community, placed by search: on a ring around the community's
# centroid, the spot with the most clearance from every member and from labels
# already placed, inside the canvas and clear of the legend rows and the strip
LABEL_SIZE = 13
placed_boxes = []


def box_gap(box):
    x0, y0, x1, y1 = box
    g = 1e9
    for (ax0, ay0, ax1, ay1) in placed_boxes:
        gx = max(ax0 - x1, 0.0, x0 - ax1)
        gy = max(ay0 - y1, 0.0, y0 - ay1)
        g = min(g, math.hypot(gx, gy))
    return g


def node_gap(box, pad=0.0):
    x0, y0, x1, y1 = box
    g = 1e9
    for i, (x, y) in POS.items():
        dx = max(x0 - x, 0.0, x - x1)
        dy = max(y0 - y, 0.0, y - y1)
        g = min(g, math.hypot(dx, dy) - outer_r(i))
    return g - pad


for members, tok, side in COMM_INFO:
    label = f"community of {len(members)}"
    w = text_w(label, LABEL_SIZE)
    cx = sum(POS[i][0] for i in members) / len(members)
    cy = sum(POS[i][1] for i in members) / len(members)
    best = None
    for rad in (70, 90, 110, 130, 150):
        for k in range(24):
            ang = 2 * math.pi * k / 24
            x = cx + rad * math.cos(ang) - w / 2 + 11      # text start (swatch sits 11px left)
            y = cy + rad * math.sin(ang)
            if x - 17 < 12 or x + w > W - 12 or y < 70 or y > H - 96:
                continue
            box = (x - 17, y - 10.5, x + w, y + 3)
            score = min(node_gap(box), box_gap(box), rad * 0.15)   # prefer close-in, clear spots
            if best is None or score > best[0]:
                best = (score, x, y, box)
    assert best is not None and best[0] >= 4, f"no clear spot for the label of {tok} ({best})"
    _, x, y, box = best
    placed_boxes.append(box)
    sx, sy = x - 11, y - 4.5
    ring = f' stroke="{RING}" stroke-width="2"' if side == "Mr. Hi" else ""
    parts.append(f'  <circle cx="{sx:.1f}" cy="{sy:.1f}" r="6" fill="{TOKENS[tok]}" opacity="0.9"{ring}/>')
    parts.append(f'  <text x="{x:.1f}" y="{y:.1f}" font-family="{SANS}" font-size="{LABEL_SIZE}" font-weight="600" fill="{TEXT}">{label}</text>')

# the exception: red dashed halo (drawn earlier, behind the discs) plus its member
# number on the clearest side of the node, and two explanatory lines placed by
# search in free space — no leader has to cross the cloud
vx, vy = POS[V]
best_tag = None
for k in range(12):
    ang = 2 * math.pi * k / 12
    tx, ty = vx + 28 * math.cos(ang), vy + 28 * math.sin(ang)
    box = (tx - 6, ty - 10, tx + 6, ty + 3)
    g = min(node_gap(box) if True else 0, 99)
    # ignore the exception itself when scoring
    g = min(math.hypot(max(box[0] - x, 0.0, x - box[2]), max(box[1] - y, 0.0, y - box[3])) - outer_r(i)
            for i, (x, y) in POS.items() if i != V)
    if best_tag is None or g > best_tag[0]:
        best_tag = (g, tx, ty)
_, tx, ty = best_tag
parts.append(f'  <text x="{tx:.0f}" y="{ty + 4:.0f}" text-anchor="middle" font-family="{SANS}" font-size="13" font-weight="700" fill="{RED}" stroke="var(--dkr-bg, #fbfbfa)" stroke-width="3.5" paint-order="stroke">{V}</text>')

line1 = f"member {V} — Louvain's one disagreement with 1977"
line2 = f"filed in the community of {MIXED_SIZE}, yet he joined Mr. Hi's new club"
w1, w2 = text_w(line1, 13.5), text_w(line2, 12.5)
wmax = max(w1, w2)
best_call = None
for CX in range(int(wmax / 2) + 12, int(W - wmax / 2 - 12), 8):
    for CY1 in (432, 444, 80, 98, 116, 134, 152):
        CY2 = CY1 + 17
        b1 = (CX - w1 / 2, CY1 - 11, CX + w1 / 2, CY1 + 3)
        b2 = (CX - w2 / 2, CY2 - 10, CX + w2 / 2, CY2 + 3)
        gap = min(node_gap(b1), node_gap(b2), box_gap(b1), box_gap(b2))
        if best_call is None or gap > best_call[0]:
            best_call = (gap, CX, CY1, CY2)
assert best_call is not None and best_call[0] >= 6, f"no clear position for the exception note ({best_call})"
_, CX, CY1, CY2 = best_call
placed_boxes.append((CX - wmax / 2, CY1 - 11, CX + wmax / 2, CY2 + 3))
parts.append(f'  <text x="{CX}" y="{CY1}" text-anchor="middle" font-family="{SANS}" font-size="13.5" font-weight="700" fill="{RED}">{line1}</text>')
parts.append(f'  <text x="{CX}" y="{CY2}" text-anchor="middle" font-family="{SANS}" font-size="12.5" fill="{MUTED}">{line2}</text>')
clr = best_call[0]
sx = CX

# legend: two centred rows above the graph
row1 = f"fill = Louvain community (seed {SEED})   ·   ring = which side the member took in 1977:"
parts.append(f'  <text x="380" y="24" text-anchor="middle" font-family="{SANS}" font-size="12.5" fill="{MUTED}">{row1}</text>')
k1, k2 = "joined Mr. Hi's new club", "stayed with the Officer"
wk1, wk2 = text_w(k1, 12.5), text_w(k2, 12.5)
gap = 34
total = 16 + wk1 + gap + 16 + wk2
lx = 380 - total / 2
parts.append(f'  <g font-family="{SANS}" font-size="12.5" fill="{MUTED}">')
parts.append(f'    <circle cx="{lx + 6:.1f}" cy="42" r="6" fill="{MUTED}" opacity="0.55" stroke="{RING}" stroke-width="2"/>')
parts.append(f'    <text x="{lx + 16:.1f}" y="46">{k1}</text>')
lx2 = lx + 16 + wk1 + gap
parts.append(f'    <circle cx="{lx2 + 6:.1f}" cy="42" r="6" fill="{MUTED}" opacity="0.55"/>')
parts.append(f'    <text x="{lx2 + 16:.1f}" y="46">{k2}</text>')
parts.append("  </g>")

# stat strip (monospace readout) and the claim band
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
stat = f"Louvain (seed {SEED}): {K} communities, Q = {Q_LOUVAIN:.3f} · the 1977 two-faction split: Q = {Q_TWO:.3f}"
parts.append(f'  <text x="380" y="{H - 62}" text-anchor="middle" font-family="{MONO}" font-size="13" fill="{TEXT}">{stat}</text>')
claim = f"the optimum of Q is finer than the sociology — modularity sees {WORDS[K]} groups, the club saw two"
band_w = round(text_w(claim, 13.5) + 56)
parts.append(f'''  <g font-family="{SANS}">
    <rect x="{380 - band_w / 2:.0f}" y="{H - 44}" width="{band_w}" height="30" rx="15" fill="var(--dkr-green, #199473)"/>
    <text x="380" y="{H - 24}" text-anchor="middle" font-size="13.5" font-weight="700" fill="var(--dkr-bg, #fff)">{claim}</text>
  </g>''')

sizes = ", ".join(str(len(m)) for m, _, _ in COMM_INFO[:-1]) + f" and {len(COMM_INFO[-1][0])}"
aria = (f"Zachary's karate club in the same layout as the Fiedler figure, each of the 34 members filled by the "
        f"community Louvain finds at seed {SEED}: {WORDS[K]} communities of {sizes} members with modularity "
        f"{Q_LOUVAIN:.3f}, against the real 1977 split's modularity {Q_TWO:.3f}; members who joined Mr. Hi's club are "
        f"ringed, and the two factions are unions of the four communities except for member {V}")

hi_core, hi_sat = len(SIDE["Mr. Hi"][0]), len(SIDE["Mr. Hi"][1])
off_core, off_sat = len(SIDE["Officer"][0]), len(SIDE["Officer"][1])
caption = (
    f"Louvain on the karate club, drawn exactly as Lecture 1 draws it and as the Fiedler figure above: fill is the community it "
    f"returns at the default resolution (seed {SEED}), rings mark the members who joined Mr. Hi in 1977. It finds not "
    f"two groups but {WORDS[K]} — a core of {hi_core} and a satellite of {hi_sat} on Mr. Hi's side, a core of "
    f"{off_core} and a satellite of {off_sat} on the Officer's — and that finer partition scores Q = {Q_LOUVAIN:.3f}, "
    f"above the Q = {Q_TWO:.3f} of the real split; the two factions are exact unions of the four communities except "
    f"for member {V}, whom Louvain files with the Officer's core although he followed Mr. Hi (the same boundary member "
    f"the Fiedler vector misplaces). This is not a bug but the resolution lesson: Q is maximized at whatever scale its "
    f"null model makes cheapest, and on {M} unweighted ties two tight sub-groups per faction beat the coarser "
    f"sociology, so the resolution parameter, not more optimization, is what would move the answer.")

svg = (f"```{{=html}}\n<figure class=\"dkr-fig\">\n"
       f"<svg viewBox=\"0 0 {W} {H}\" role=\"img\" aria-label=\"{aria}\">\n"
       f"  <!-- generated by scripts/figgen/w2_louvain_karate.py · networkx {nx.__version__} · "
       f"louvain_communities(seed={SEED}, weight=None) on the {M} unweighted ties (Lab 2's call; its seed "
       f"{LAB_SEED} gives {LAB_K} communities, Q = {LAB_Q:.3f}) · positions parsed from fig-w2-fiedler.html -->\n"
       + "\n".join(parts) + "\n</svg>\n"
       f"<figcaption class=\"fig-caption\">{caption}</figcaption>\n</figure>\n```\n")
OUT.write_text(svg, encoding="utf-8")

print("wrote", OUT)
print(f"networkx {nx.__version__} · louvain_communities(seed={SEED}, weight=None)")
print(f"Lab 2's seed {LAB_SEED}: {LAB_K} communities, Q = {LAB_Q:.6f}, sizes {sorted(map(len, LAB_COMMS), reverse=True)}")
for members, tok, side in COMM_INFO:
    print(f"  {tok:6s} ({side:7s}) {len(members):2d} members: {members}")
print(f"Q(Louvain) = {Q_LOUVAIN:.6f}   Q(two clubs) = {Q_TWO:.6f}")
print(f"violators (own 1977 side != community majority): {VIOLATORS} -> {len(VIOLATORS)} member(s)")
print(f"members in a mixed community: {MEMBERS_IN_MIXED}; members whose community lies entirely inside one faction: {PURE_MEMBERS}")
print(f"exception note at x={sx}, clearance {clr:.1f}px")
