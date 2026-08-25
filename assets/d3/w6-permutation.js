/* Widget 6.4 — Permutation equivariance, or: why not just an MLP?
 * One 5-node graph, two models:
 *   GNN layer  : y_u = x_u + sum_{v in N(u)} x_v   (sum aggregation, weights = 1)
 *   MLP        : flatten A row-major, score = w · vec(A) + v · x   (fixed weights)
 * "Shuffle ordering" renames the nodes with a random permutation. The graph is
 * unchanged — only the storage order moved — so a correct model must not care.
 * The GNN's outputs travel with the node identities (equivariance); the MLP's
 * score changes, because vec(A) is not a permutation-invariant representation.
 */
(function () {
  "use strict";
  const U = window.DKR;

  const EDGES = [[0, 1], [0, 2], [1, 2], [2, 3], [3, 4]];
  const NAMES = ["A", "B", "C", "D", "E"];
  const n = 5;
  const x = [1, 2, 3, 4, 5]; // feature of each *identity* (A carries 1, B carries 2, ...)

  // fixed pseudo-random MLP weights (seeded, deterministic)
  let seed = 7;
  function rnd() { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; }
  const wA = d3.range(n * n).map(rnd).map((v) => Math.round(v * 20) / 10);
  const wX = d3.range(n).map(rnd).map((v) => Math.round(v * 20) / 10);

  let perm = d3.range(n); // perm[identity] = current index/slot of that identity

  function adjUnder(p) {
    const M = Array.from({ length: n }, () => Array(n).fill(0));
    EDGES.forEach(([a, b]) => { M[p[a]][p[b]] = 1; M[p[b]][p[a]] = 1; });
    return M;
  }
  function featUnder(p) {
    const f = Array(n).fill(0);
    for (let id = 0; id < n; id++) f[p[id]] = x[id];
    return f;
  }
  function gnnOut(M, f) { return M.map((row, i) => f[i] + row.reduce((a, v, j) => a + v * f[j], 0)); }
  function mlpOut(M, f) {
    let s = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) s += wA[i * n + j] * M[i][j];
    for (let i = 0; i < n; i++) s += wX[i] * f[i];
    return s;
  }

  const identityA = adjUnder(d3.range(n));
  const identityF = featUnder(d3.range(n));
  const gnnRef = gnnOut(identityA, identityF); // per-identity reference outputs
  const mlpRef = mlpOut(identityA, identityF);

  const W = 760, Hgt = 335;
  const gg = U.graph(n, EDGES);
  U.circleLayout(gg, 105, 140, 82);

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w6-pm-svg", W, Hgt);
    const M = adjUnder(perm);
    const f = featUnder(perm);
    const gnnNow = gnnOut(M, f);       // indexed by slot
    const mlpNow = mlpOut(M, f);
    const shuffled = perm.some((v, i) => v !== i);

    // --- the graph (identities never move; slot labels shown next to nodes) ---
    const gp = svg.append("g").attr("transform", "translate(8,10)");
    gp.selectAll("line").data(EDGES).join("line")
      .attr("x1", (e) => gg.nodes[e[0]].x).attr("y1", (e) => gg.nodes[e[0]].y)
      .attr("x2", (e) => gg.nodes[e[1]].x).attr("y2", (e) => gg.nodes[e[1]].y)
      .attr("stroke", P.border).attr("stroke-width", 1.6);
    const nd = gp.selectAll("g").data(gg.nodes).join("g")
      .attr("transform", (d) => `translate(${d.x},${d.y})`);
    nd.append("circle").attr("r", 16).attr("fill", P.paper)
      .attr("stroke", P.blue).attr("stroke-width", 2);
    nd.append("text").attr("text-anchor", "middle").attr("dy", 4.5)
      .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.text)
      .text((d) => NAMES[d.id]);
    nd.append("text").attr("x", 0).attr("y", 32).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.accent)
      .text((d) => `row ${perm[d.id]}`);
    gp.append("text").attr("x", 105).attr("y", 262).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("same graph — only the");
    gp.append("text").attr("x", 105).attr("y", 279).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("storage order changes");

    // --- adjacency matrix under current ordering ---
    const mp = svg.append("g").attr("transform", "translate(235,52)");
    mp.append("text").attr("x", 65).attr("y", -30).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text)
      .text(shuffled ? "PAPᵀ (shuffled rows)" : "A (original order)");
    U.drawMatrix(mp, M, {
      cell: 26,
      color: (v) => (v ? P.blue : "transparent"),
      fmt: (v) => (v ? "1" : "·"),
      textColor: P.text, borderColor: P.border, fontSize: 12.5,
    });
    // row + column labels = which identity sits in each slot
    const slotName = Array(n).fill("");
    for (let id = 0; id < n; id++) slotName[perm[id]] = NAMES[id];
    mp.selectAll("text.rl").data(slotName).join("text")
      .attr("class", "rl").attr("x", -10).attr("y", (d, i) => i * 26 + 18)
      .attr("text-anchor", "end").attr("font-size", 12.5).attr("fill", P.muted)
      .text((d) => d);
    mp.selectAll("text.cl").data(slotName).join("text")
      .attr("class", "cl").attr("x", (d, i) => i * 26 + 13).attr("y", -8)
      .attr("text-anchor", "middle").attr("font-size", 12.5).attr("fill", P.muted)
      .text((d) => d);

    // --- outputs table ---
    const tp = svg.append("g").attr("transform", "translate(465,40)");
    tp.append("text").attr("x", 130).attr("y", -12).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text)
      .text("outputs, tracked by node identity");
    const header = ["node", "GNN before", "GNN after", "ok?"];
    header.forEach((h, j) =>
      tp.append("text").attr("x", [15, 95, 180, 250][j]).attr("y", 12)
        .attr("text-anchor", "middle").attr("font-size", 12.5).attr("fill", P.muted).text(h));
    for (let id = 0; id < n; id++) {
      const y = 36 + id * 23;
      const after = gnnNow[perm[id]];
      const same = Math.abs(after - gnnRef[id]) < 1e-9;
      tp.append("text").attr("x", 15).attr("y", y).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.text).text(NAMES[id]);
      tp.append("text").attr("x", 95).attr("y", y).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("fill", P.text).text(gnnRef[id]);
      tp.append("text").attr("x", 180).attr("y", y).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("fill", P.text).text(after);
      tp.append("text").attr("x", 250).attr("y", y).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("fill", same ? P.green : P.red).text(same ? "✓" : "✗");
    }
    const mlpSame = Math.abs(mlpNow - mlpRef) < 1e-9;
    tp.append("text").attr("x", 15).attr("y", 170).attr("font-size", 12.5)
      .attr("fill", P.text)
      .text(`MLP on vec(A): ${mlpRef.toFixed(2)} → ${mlpNow.toFixed(2)}`);
    tp.append("text").attr("x", 15).attr("y", 191).attr("font-size", 12.5)
      .attr("font-weight", 700)
      .attr("fill", mlpSame ? P.green : P.red)
      .text(mlpSame ? "unchanged (by luck — shuffle again!)" : "changed — the MLP is fooled by renaming");
  }

  document.getElementById("w6-pm-shuffle").addEventListener("click", () => {
    do { perm = d3.shuffle(d3.range(n)); } while (perm.every((v, i) => v === i));
    render();
  });
  document.getElementById("w6-pm-reset").addEventListener("click", () => {
    perm = d3.range(n); render();
  });

  U.onThemeChange(render);
  U.lazyBoot("w6-pm-svg", render);
})();
