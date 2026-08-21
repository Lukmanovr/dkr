/* Widget 3.4 — Label propagation: the no-learning baseline.
 * The karate club with only the two leaders labeled. Each sweep, every member
 * takes the majority label among labeled neighbors (synchronous update, ties
 * keep the current state). Deterministic — no randomness at all. The contrast
 * with Widget 3.3: edges-only, converges in a few sweeps, produces no vectors
 * and no scores for any downstream task.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const N = 34;
  const EDGES = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],[0,12],[0,13],
    [0,17],[0,19],[0,21],[0,31],[1,2],[1,3],[1,7],[1,13],[1,17],[1,19],[1,21],[1,30],
    [2,3],[2,7],[2,8],[2,9],[2,13],[2,27],[2,28],[2,32],[3,7],[3,12],[3,13],[4,6],
    [4,10],[5,6],[5,10],[5,16],[6,16],[8,30],[8,32],[8,33],[9,33],[13,33],[14,32],
    [14,33],[15,32],[15,33],[18,32],[18,33],[19,33],[20,32],[20,33],[22,32],[22,33],
    [23,25],[23,27],[23,29],[23,32],[23,33],[24,25],[24,27],[24,31],[25,31],[26,29],
    [26,33],[27,33],[28,31],[28,33],[29,32],[29,33],[30,32],[30,33],[31,32],[31,33],
    [32,33]];
  const MR_HI = new Set([0,1,2,3,4,5,6,7,8,10,11,12,13,16,17,19,21]);
  const ADJ = Array.from({ length: N }, () => []);
  EDGES.forEach(([a, b]) => { ADJ[a].push(b); ADJ[b].push(a); });
  const C_HI = "#d9a62e", C_OFF = "#0f8377";

  // deterministic layout: reuse Week 1's published node positions would need the
  // generator; a fixed spring result is inlined instead (computed once, seed 7 —
  // same layout family as the lecture figures)
  let POS = null;
  function layout() {
    // tiny deterministic spring embedding (same recipe as scripts/figgen)
    const rng = (function (seed) { let a = seed >>> 0; return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })(7);
    const pos = Array.from({ length: N }, () => [rng() * 2 - 1, rng() * 2 - 1]);
    for (let it = 0; it < 350; it++) {
      const t = 0.08 * (1 - it / 350);
      const disp = Array.from({ length: N }, () => [0, 0]);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          if (i === j) continue;
          const dx = pos[i][0] - pos[j][0], dy = pos[i][1] - pos[j][1];
          const d2 = dx * dx + dy * dy + 1e-6;
          disp[i][0] += 0.02 * dx / d2; disp[i][1] += 0.02 * dy / d2;
        }
      }
      EDGES.forEach(([a, b]) => {
        const dx = pos[a][0] - pos[b][0], dy = pos[a][1] - pos[b][1];
        disp[a][0] -= dx * 0.045; disp[a][1] -= dy * 0.045;
        disp[b][0] += dx * 0.045; disp[b][1] += dy * 0.045;
      });
      for (let i = 0; i < N; i++) {
        const m = Math.hypot(disp[i][0], disp[i][1]) + 1e-9;
        pos[i][0] += disp[i][0] / m * Math.min(m, t);
        pos[i][1] += disp[i][1] / m * Math.min(m, t);
      }
    }
    const xs = pos.map((p) => p[0]), ys = pos.map((p) => p[1]);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    return pos.map(([x, y]) => [
      50 + (x - xmin) / (xmax - xmin) * 620,
      30 + (y - ymin) / (ymax - ymin) * 220,
    ]);
  }

  let labels, sweeps, stable;
  function reset() {
    labels = Array(N).fill(-1);
    labels[0] = 0; labels[33] = 1;             // the two leaders — all we know
    sweeps = 0; stable = false;
  }
  reset();

  function sweep() {
    if (stable) return;
    const next = labels.slice();
    for (let i = 0; i < N; i++) {
      if (i === 0 || i === 33) continue;       // seeds stay pinned
      let c0 = 0, c1 = 0;
      ADJ[i].forEach((j) => { if (labels[j] === 0) c0 += 1; else if (labels[j] === 1) c1 += 1; });
      if (c0 > c1) next[i] = 0;
      else if (c1 > c0) next[i] = 1;           // tie or no labeled neighbor: keep
    }
    stable = next.every((v, i) => v === labels[i]);
    if (!stable) { labels = next; sweeps += 1; }
  }

  function render() {
    const P = U.pal();
    if (!POS) POS = layout();
    const svg = U.svgIn("w3-lp-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    g.selectAll("line").data(EDGES).join("line")
      .attr("x1", (e) => POS[e[0]][0]).attr("y1", (e) => POS[e[0]][1])
      .attr("x2", (e) => POS[e[1]][0]).attr("y2", (e) => POS[e[1]][1])
      .attr("stroke", P.muted).attr("stroke-width", 1).attr("opacity", 0.35);

    for (let i = 0; i < N; i++) {
      const [x, y] = POS[i];
      const l = labels[i];
      const seed = i === 0 || i === 33;
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", seed ? 13 : 8)
        .attr("fill", l === 0 ? C_HI : l === 1 ? C_OFF : P.paper)
        .attr("stroke", l === -1 ? P.muted : seed ? P.accentDark : "none")
        .attr("stroke-width", seed ? 3 : 1.4)
        .attr("opacity", l === -1 ? 0.9 : 0.92);
    }

    const labeled = labels.filter((l) => l !== -1).length;
    let correct = 0;
    for (let i = 0; i < N; i++) {
      if (labels[i] === (MR_HI.has(i) ? 0 : 1)) correct += 1;
    }
    const msg = stable
      ? `stable after ${sweeps} sweeps · ${labeled}/34 labeled · ${correct}/34 match the real split`
      : sweeps === 0
        ? "two labels known (the ringed leaders) — press “sweep” to let them spread"
        : `sweep ${sweeps} · ${labeled}/34 labeled · ${correct}/34 currently correct`;
    g.append("text").attr("x", 380).attr("y", 288).attr("text-anchor", "middle")
      .attr("font-size", 13.5).attr("font-weight", 700)
      .attr("fill", stable ? P.green : P.text).text(msg);
    g.append("text").attr("x", 380).attr("y", 308).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", P.muted)
      .text("majority vote of labeled neighbors, synchronous; ties keep their state. No vectors, no features, no learning.");
  }

  document.getElementById("w3-lp-step").addEventListener("click", () => { sweep(); render(); });
  document.getElementById("w3-lp-run").addEventListener("click", () => {
    for (let k = 0; k < 50 && !stable; k++) sweep();
    render();
  });
  document.getElementById("w3-lp-reset").addEventListener("click", () => { reset(); render(); });

  U.onThemeChange(render);
  U.lazyBoot("w3-lp-svg", render);
})();
