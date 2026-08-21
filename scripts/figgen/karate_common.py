"""Shared Zachary-karate-club data + deterministic layout + label placement,
used by karate.py (Week 1 hero) and karate_fiedler.py (Week 2 spectral split)."""
import math
import random

EDGES = [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),(0,12),(0,13),
         (0,17),(0,19),(0,21),(0,31),(1,2),(1,3),(1,7),(1,13),(1,17),(1,19),(1,21),(1,30),
         (2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),(2,32),(3,7),(3,12),(3,13),(4,6),
         (4,10),(5,6),(5,10),(5,16),(6,16),(8,30),(8,32),(8,33),(9,33),(13,33),(14,32),
         (14,33),(15,32),(15,33),(18,32),(18,33),(19,33),(20,32),(20,33),(22,32),(22,33),
         (23,25),(23,27),(23,29),(23,32),(23,33),(24,25),(24,27),(24,31),(25,31),(26,29),
         (26,33),(27,33),(28,31),(28,33),(29,32),(29,33),(30,32),(30,33),(31,32),(31,33),
         (32,33)]
N = 34
MR_HI = {0,1,2,3,4,5,6,7,8,10,11,12,13,16,17,19,21}   # actual post-split membership
C_HI, C_OFF = "#d9a62e", "#0f8377"


def layout(x_span=(60, 620), y_span=(58, 318), seed=7, iters=600):
    """Deterministic spring embedding; returns list of (x, y) rounded positions."""
    rng = random.Random(seed)
    pos = [[rng.uniform(-1, 1), rng.uniform(-1, 1)] for _ in range(N)]
    for it in range(iters):
        t = 0.08 * (1 - it / iters)
        disp = [[0.0, 0.0] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                dx, dy = pos[i][0] - pos[j][0], pos[i][1] - pos[j][1]
                d2 = dx * dx + dy * dy + 1e-6
                f = 0.02 / d2
                disp[i][0] += f * dx; disp[i][1] += f * dy
        for a, b in EDGES:
            dx, dy = pos[a][0] - pos[b][0], pos[a][1] - pos[b][1]
            d = math.sqrt(dx * dx + dy * dy) + 1e-6
            disp[a][0] -= dx * 0.045; disp[a][1] -= dy * 0.045
            disp[b][0] += dx * 0.045; disp[b][1] += dy * 0.045
        for i in range(N):
            m = math.sqrt(disp[i][0] ** 2 + disp[i][1] ** 2) + 1e-9
            pos[i][0] += disp[i][0] / m * min(m, t)
            pos[i][1] += disp[i][1] / m * min(m, t)
    xs, ys = [p[0] for p in pos], [p[1] for p in pos]
    (x0, x1), (y0, y1) = x_span, y_span
    sx = lambda x: x0 + (x - min(xs)) / (max(xs) - min(xs)) * (x1 - x0)
    sy = lambda y: y0 + (y - min(ys)) / (max(ys) - min(ys)) * (y1 - y0)
    return [(round(sx(x), 1), round(sy(y), 1)) for x, y in pos]


def make_label_placer(P, x_max=740, y_max=336):
    """Returns place(node, text, size) choosing offsets that avoid nodes and prior labels."""
    placed = []

    def place(anchor_node, text, size=12.5):
        ax, ay = P[anchor_node]
        w = len(text) * size * 0.52
        best, best_score = None, -1.0
        for k in range(16):
            ang = 2 * math.pi * k / 16
            for rad in (36, 50, 66, 84):
                cx, cy = ax + rad * math.cos(ang), ay + rad * math.sin(ang)
                if not (20 + w / 2 <= cx <= x_max - w / 2 and 24 <= cy <= y_max):
                    continue
                score = min(math.hypot(cx + dx - px, cy - py)
                            for (px, py) in P for dx in (-w / 2, 0, w / 2))
                for (ox, oy, ow) in placed:
                    gap_x = max(0.0, abs(cx - ox) - (w + ow) / 2)
                    score = min(score, math.hypot(gap_x, abs(cy - oy)))
                if score > best_score:
                    best_score, best = score, (cx, cy)
        placed.append((best[0], best[1], w))
        return best

    return place


def fiedler_vector(edges=EDGES, n=N, iters=6000):
    """Second-smallest Laplacian eigenvector via power iteration on cI − L with
    deflation of the constant vector. Pure python, deterministic."""
    deg = [0] * n
    for a, b in edges:
        deg[a] += 1; deg[b] += 1
    c = 2 * max(deg)
    rng = random.Random(3)
    v = [rng.uniform(-1, 1) for _ in range(n)]
    for _ in range(iters):
        mean = sum(v) / n
        v = [x - mean for x in v]                      # deflate the constant eigenvector
        w = [(c - deg[i]) * v[i] for i in range(n)]    # (cI − L)v = (c−d_i)v_i + Σ_nbrs v_j
        for a, b in edges:
            w[a] += v[b]; w[b] += v[a]
        norm = math.sqrt(sum(x * x for x in w)) + 1e-12
        v = [x / norm for x in w]
    mean = sum(v) / n
    return [x - mean for x in v]
