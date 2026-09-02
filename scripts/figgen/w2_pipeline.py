#!/usr/bin/env python
"""Generate assets/figures/fig-w2-pipeline.html — the classical graph-ML pipeline of
Lecture 2 §6 as one slide-ready diagram: "assemble features by a fixed recipe, then
fit a classical model". Two lanes share the cast graph as input:
  node lane  — a real feature table for the cast graph (degree, closeness, betweenness,
               clustering, community), every value computed below, then any classifier
  graph lane — one round of WL color refinement, its color histogram, the histogram
               inner-product kernel, an SVM
Protocol F1: every displayed number derives from the edge list in this file; the
lecture's stated values are asserted, so a drift shows up as a failed run.
Run: python scripts/figgen/w2_pipeline.py   (writes the figure, prints the numbers)
"""
from collections import Counter, deque
from fractions import Fraction
from itertools import combinations
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "assets" / "figures" / "fig-w2-pipeline.html"

SANS = "'Source Sans 3', sans-serif"
MONO = "'JetBrains Mono', monospace"
TEXT = "var(--dkr-text, #1c1c21)"
MUTED = "var(--dkr-muted, #6e6e7a)"
EDGE = "var(--dkr-muted, #8e8e9a)"
ACCENT = "var(--dkr-accent, #d9603b)"
GREEN = "var(--dkr-green, #199473)"
PAPER = "var(--dkr-paper, #ffffff)"
BORDER = "var(--dkr-border, #e8e7e3)"
BG = "var(--dkr-bg, #fff)"

# ── the cast graph (docs/figure-style.md) ─────────────────────────────────────
NAMES = "ABCDEF"
EDGES = [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("C", "E"), ("C", "F"), ("E", "F")]
IDENT = {"A": "#d9a62e", "B": "#cf4a30", "C": "#199473", "D": "#7c5cd6", "E": "#d1567e", "F": "#0f8377"}
# the same drawing as fig-w2-wl (CPOS in w2_figs.py) so the reader recognizes the graph
CPOS = {"A": (120, 105), "B": (215, 48), "C": (280, 118), "D": (68, 172), "E": (224, 185), "F": (368, 172)}
CPOS_CENTER = (218, 116.5)   # bbox centre of CPOS
# KEEP IN SYNC with WLPAL in scripts/figgen/w2_figs.py and assets/d3/w2-wl.js:
# WL color id -> hex, so a round-1 class has the same color here as in the WL figure
WLPAL = ["#8e8e9a", "#d9603b", "#0f8377", "#7c5cd6", "#d9a62e",
         "#d1567e", "#2e7dd1", "#8a5a33", "#cf4a30", "#199473"]

adj = {u: set() for u in NAMES}
for a, b in EDGES:
    adj[a].add(b)
    adj[b].add(a)
n, m = len(NAMES), len(EDGES)


# ═══════════════════════ node lane: the feature table ═══════════════════════
def bfs(s):
    dist = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        for v in sorted(adj[u]):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


degree = {u: len(adj[u]) for u in NAMES}
# closeness (eq-w02-closeness): (n-1) / sum of distances to everyone else
closeness = {u: Fraction(n - 1, sum(bfs(u).values())) for u in NAMES}


# betweenness (eq-w02-betweenness) by explicit enumeration of every shortest path,
# raw pair counts — the lecture's "a betweenness of 6"
def shortest_paths(s, t):
    d = bfs(s)
    out = []

    def back(u, tail):
        if u == s:
            out.append([u] + tail)
            return
        for v in sorted(adj[u]):
            if d[v] == d[u] - 1:
                back(v, [u] + tail)

    back(t, [])
    return out


betweenness = {u: Fraction(0) for u in NAMES}
for s, t in combinations(NAMES, 2):
    paths = shortest_paths(s, t)
    for v in NAMES:
        if v not in (s, t):
            betweenness[v] += Fraction(sum(v in p for p in paths), len(paths))


# local clustering coefficient (def-clustering): edges among neighbors / C(d_u, 2)
def clustering(u):
    k = degree[u]
    if k < 2:
        return Fraction(0)
    nb = sorted(adj[u])
    tri = sum(1 for a, b in combinations(nb, 2) if b in adj[a])
    return Fraction(tri, k * (k - 1) // 2)


clust = {u: clustering(u) for u in NAMES}


# community id = the modularity-maximizing partition (eq-w02-modularity, community
# form), found by brute force over all 203 set partitions of six nodes
def modularity(blocks):
    q = Fraction(0)
    for blk in blocks:
        e_c = sum(1 for a, b in EDGES if a in blk and b in blk)
        d_c = sum(degree[u] for u in blk)
        q += Fraction(e_c, m) - Fraction(d_c * d_c, 4 * m * m)
    return q


def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in set_partitions(rest):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


scored = sorted(((modularity([set(b) for b in p]), sorted(sorted(b) for b in p))
                 for p in set_partitions(list(NAMES))), reverse=True)
bestQ, best = scored[0]
assert bestQ > scored[1][0], "the community id needs a unique modularity maximum"
assert best == [["A", "B", "D"], ["C", "E", "F"]], best   # the lecture's better split, §4.2
community = {u: i + 1 for i, blk in enumerate(best) for u in blk}

# the lecture's stated numbers (§1, §3, §4.2, Q2) must be reproduced exactly
assert [degree[u] for u in NAMES] == [3, 2, 4, 1, 2, 2]
assert (closeness["A"], closeness["B"], closeness["C"], closeness["D"], closeness["E"], closeness["F"]) == \
    (Fraction(5, 7), Fraction(5, 8), Fraction(5, 6), Fraction(5, 11), Fraction(5, 9), Fraction(5, 9))
assert betweenness["C"] == 6 and betweenness["A"] == 4 and all(betweenness[u] == 0 for u in "BDEF")
assert [clust[u] for u in NAMES] == [Fraction(1, 3), 1, Fraction(1, 3), 0, 1, 1]
assert bestQ == Fraction(10, 49)   # 0.204, the lecture's "better modularity"


def fmt2(v):
    """two decimals, rounding half up (5/8 -> 0.63 as in the lecture, not banker's 0.62)"""
    q = int((v * 100 + Fraction(1, 2)).__floor__())
    return f"{q // 100}.{q % 100:02d}"


def frac(v):
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


COLUMNS = [  # header, value formatter
    ("degree", lambda u: str(degree[u])),
    ("closeness", lambda u: fmt2(closeness[u])),
    ("betweenness", lambda u: str(int(betweenness[u]))),
    ("clustering", lambda u: frac(clust[u])),
    ("community", lambda u: str(community[u])),
]
ROWS = {u: [f(u) for _, f in COLUMNS] for u in NAMES}


# ═══════════════════════ graph lane: one WL round ════════════════════════════
# round 0: everyone color 0; round 1: hash(own color, sorted multiset of neighbor
# colors), ids handed out in order of first appearance — exactly as in w2_figs.py
color0 = {u: 0 for u in NAMES}
table = {}
color1 = {}
for u in NAMES:
    sig = (color0[u], tuple(sorted(color0[v] for v in adj[u])))
    if sig not in table:
        table[sig] = len(table) + 1
    color1[u] = table[sig]
# after round 1 a color is exactly a degree class; present the histogram by degree, 1 → 4
deg_of_id = {color1[u]: degree[u] for u in NAMES}
classes = sorted(deg_of_id.items(), key=lambda kv: kv[1])        # [(color id, degree)]
counts = Counter(color1.values())
hist = [counts[cid] for cid, _ in classes]
assert [d for _, d in classes] == [1, 2, 3, 4] and hist == [1, 3, 1, 1], (classes, hist)
hist_str = "(" + ", ".join(str(h) for h in hist) + ")"


# ═══════════════════════ drawing ═════════════════════════════════════════════
W, H = 760, 498


def text(x, y, s, size=12.5, fill=TEXT, weight=None, anchor="middle", font=SANS):
    w = f' font-weight="{weight}"' if weight else ""
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{font}" font-size="{size}"{w} fill="{fill}">{s}</text>'


def box(x, y, w, h):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{PAPER}" stroke="{BORDER}" stroke-width="1.5"/>'


def arrow(d):
    return f'<path d="{d}" fill="none" stroke="{EDGE}" stroke-width="1.6" stroke-dasharray="4,4" marker-end="url(#w2pArrow)"/>'


def chip(x, y, w, label):
    # text in the page ground color, as the house claim bands do: white on terracotta in
    # light, dark on the lighter dark-theme accent — AA in both (white fails there)
    return (f'<rect x="{x}" y="{y}" width="{w}" height="28" rx="14" fill="{ACCENT}"/>\n  '
            + text(x + w / 2, y + 18.5, label, 13, BG, 700))


def cast_graph(cx, cy, scale, fills, r=11):
    """the cast graph centred at (cx, cy), CPOS scaled; letters stay 12.5 px (never scaled)"""
    pos = {u: (round(cx + (x - CPOS_CENTER[0]) * scale, 1), round(cy + (y - CPOS_CENTER[1]) * scale, 1))
           for u, (x, y) in CPOS.items()}
    out = [f'<g stroke="{EDGE}" stroke-width="1.8" opacity="0.65">']
    for a, b in EDGES:
        out.append(f'  <line x1="{pos[a][0]}" y1="{pos[a][1]}" x2="{pos[b][0]}" y2="{pos[b][1]}"/>')
    out.append("</g>")
    for u in NAMES:
        x, y = pos[u]
        out.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fills[u]}"/>')
        out.append(text(x, y + 4.5, u, 12.5, "#fff", 700))
    return out


S = []  # svg body lines
S.append(f'<defs><marker id="w2pArrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">'
         f'<path d="M0,0 L7,3 L0,6 Z" fill="{EDGE}"/></marker></defs>')
S.append(text(380, 20, "the classical graph-ML pipeline: assemble features with a fixed recipe, then fit a classical model",
              12.5, MUTED))

# ── shared input: the cast graph, left, between the lanes ──
GX, GY, GS = 76, 250, 0.38
S.append(text(GX, 200, "input graph", 13, TEXT, 700))
S += cast_graph(GX, GY, GS, IDENT)
S.append(text(GX, 306, f"the cast, m = {m} edges", 12.5, MUTED))
# one dashed bus splits into the two lanes
S.append(arrow("M142,250 L156,250 L156,136 L167,136"))
S.append(arrow("M142,250 L156,250 L156,358 L167,358"))

# ── top lane: NODE level ──
S.append(f'<text x="168" y="44" text-anchor="start" font-family="{SANS}" font-size="13" fill="{TEXT}">'
         f'<tspan font-weight="700">NODE level</tspan><tspan fill="{MUTED}"> · one row of features per node</tspan></text>')
S.append(box(168, 52, 380, 188))
S.append(text(178, 71, "structural features, one row per node", 13, TEXT, 700, "start"))
COLX = [191, 231, 289, 364, 440, 507]          # node disc, then the five feature columns
S.append(text(COLX[0], 92, "node", 12.5, MUTED))
for cx, (hdr, _) in zip(COLX[1:], COLUMNS):
    S.append(text(cx, 92, hdr, 12.5, MUTED))
S.append(f'<line x1="176" y1="98" x2="540" y2="98" stroke="{BORDER}" stroke-width="1"/>')
for i, u in enumerate(NAMES):
    y = 112 + 19 * i
    S.append(f'<circle cx="{COLX[0]}" cy="{y - 4.5}" r="9" fill="{IDENT[u]}"/>')
    S.append(text(COLX[0], y, u, 12.5, "#fff", 700))
    for cx, val in zip(COLX[1:], ROWS[u]):
        S.append(text(cx, y, val, 12.5, TEXT, None, "middle", MONO))
S.append(text(358, 228, "columns chosen by hand, fixed before any label is seen", 12.5, ACCENT, 600))
S.append(arrow("M548,136 L571,136"))
S.append(box(572, 104, 178, 64))
S.append(text(661, 126, "any classifier", 13, TEXT, 700))
S.append(text(661, 144, "logistic regression,", 12.5, MUTED))
S.append(text(661, 160, "gradient boosting", 12.5, MUTED))
S.append(arrow("M661,168 L661,187"))
S.append(chip(576, 188, 170, "one label per node"))

# ── bottom lane: GRAPH level ──
S.append(f'<text x="168" y="266" text-anchor="start" font-family="{SANS}" font-size="13" fill="{TEXT}">'
         f'<tspan font-weight="700">GRAPH level</tspan><tspan fill="{MUTED}"> · one vector of counts per graph</tspan></text>')
S.append(box(168, 276, 168, 164))
S.append(text(176, 295, "WL colors, h rounds", 13, TEXT, 700, "start"))
S.append(text(176, 312, "shown after round 1", 12.5, MUTED, None, "start"))
S += cast_graph(252, 380, GS, {u: WLPAL[color1[u]] for u in NAMES})
S.append(text(252, 432, "one color per degree", 12.5, MUTED))
S.append(arrow("M336,358 L355,358"))

S.append(box(356, 276, 130, 164))
S.append(text(364, 295, "color histogram", 13, TEXT, 700, "start"))
S.append(text(421, 314, hist_str, 13, TEXT, 700, "middle", MONO))
BASE, UNIT, BW = 392, 17, 20
for k, ((cid, deg), cnt) in enumerate(zip(classes, hist)):
    bx = 379 + 28 * k
    h = UNIT * cnt
    S.append(f'<rect x="{bx - BW / 2}" y="{BASE - h}" width="{BW}" height="{h}" fill="{WLPAL[cid]}"/>')
    S.append(text(bx, BASE - h - 5, str(cnt), 12.5, TEXT, None, "middle", MONO))
    S.append(text(bx, 408, str(deg), 12.5, MUTED))
S.append(f'<line x1="366" y1="{BASE}" x2="476" y2="{BASE}" stroke="{EDGE}" stroke-width="1"/>')
S.append(text(421, 426, "degree class 1 → 4", 12.5, MUTED))
S.append(arrow("M486,358 L503,358"))

S.append(box(504, 276, 246, 68))
S.append(text(627, 296, "kernel = inner product", 13, TEXT, 700))
S.append(text(627, 313, "of the two graphs' color histograms", 12.5, MUTED))
S.append(text(627, 333, "K(G₁, G₂) = ⟨h(G₁), h(G₂)⟩", 12.5, TEXT, None, "middle", MONO))
S.append(arrow("M627,344 L627,361"))
S.append(box(504, 362, 246, 36))
S.append(f'<text x="627" y="385" text-anchor="middle" font-family="{SANS}" font-size="13" fill="{TEXT}">'
         f'<tspan font-weight="700">SVM</tspan><tspan fill="{MUTED}"> · a support-vector machine on K</tspan></text>')
S.append(arrow("M627,398 L627,411"))
S.append(chip(542, 412, 170, "one label per graph"))

# ── claim band ──
S.append(f'<rect x="110" y="456" width="540" height="30" rx="15" fill="{GREEN}"/>')
S.append(text(380, 476, "computed, not learned — nothing here improves with more data", 13.5, BG, 700))

ARIA = ("The classical graph-ML pipeline as two lanes sharing the cast graph as input. Node lane: a feature "
        "table with one row per node — degree " + ",".join(ROWS[u][0] for u in NAMES) +
        "; closeness " + ",".join(ROWS[u][1] for u in NAMES) +
        "; betweenness " + ",".join(ROWS[u][2] for u in NAMES) +
        "; clustering " + ",".join(ROWS[u][3] for u in NAMES) +
        "; community " + ",".join(ROWS[u][4] for u in NAMES) +
        " — feeds any classifier and yields one label per node. Graph lane: one round of Weisfeiler-Leman "
        f"coloring splits the graph into its four degree classes, the histogram {hist_str} by degree 1 to 4 "
        "feeds an inner-product kernel and an SVM, yielding one label per graph. Bottom band: computed, not "
        "learned — nothing here improves with more data.")

CAPTION = ("Both lanes have the same shape: a fixed recipe turns the graph into a table of numbers, and a "
           "classical model — logistic regression, boosting, an SVM — is fit on that table. The node lane's "
           "rows are this lecture's centralities, clustering coefficients and community ids, computed here for "
           "the cast graph; the graph lane's row is the Weisfeiler–Leman color histogram, "
           f"{hist_str} after one round. Every number is fixed before the model sees a single label, so the "
           "person who chose the columns is the bottleneck — whatever the task needed that the recipe "
           "discarded is gone before any fitting starts. Next week the feature step itself becomes what "
           "training improves.")

body = "\n  ".join(S)
OUT.write_text(
    "```{=html}\n<figure class=\"dkr-fig\">\n"
    f"<svg viewBox=\"0 0 {W} {H}\" role=\"img\" aria-label=\"{ARIA}\">\n"
    "  <!-- generated by scripts/figgen/w2_pipeline.py — every number is computed there -->\n"
    f"  {body}\n</svg>\n<figcaption class=\"fig-caption\">{CAPTION}</figcaption>\n</figure>\n```\n",
    encoding="utf-8")

# ── report the numbers the figure shows ──
print("wrote", OUT.relative_to(OUT.parents[2]))
print("node  " + "  ".join(f"{h:>11}" for h, _ in COLUMNS))
for u in NAMES:
    print(f"{u:<4}  " + "  ".join(f"{v:>11}" for v in ROWS[u]))
print(f"modularity of {best}: {bestQ} = {float(bestQ):.3f}")
print("WL round 1 colors:", {u: color1[u] for u in NAMES})
print("histogram by degree class 1..4:", hist_str)
