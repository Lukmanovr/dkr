/* Widget 9.1 — WL color refinement, stepped by hand.
 * Three graph pairs; Step runs one refinement round on both graphs with a
 * SHARED color table (same signature → same color across graphs), so equal
 * histograms are visible as equal palettes. Deterministic.
 */
(function () {
  "use strict";
  const U = window.DKR;

  function ring(cx, cy, R, n, ids, rot) {
    const pos = {};
    ids.forEach((v, i) => {
      const a = (rot || -Math.PI / 2) + (2 * Math.PI * i) / n;
      pos[v] = [cx + R * Math.cos(a), cy + R * Math.sin(a)];
    });
    return pos;
  }

  const TRI2 = { edges: [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]], n: 6,
    pos: Object.assign(ring(95, 120, 48, 3, [0, 1, 2]), ring(215, 120, 48, 3, [3, 4, 5])) };
  const HEX = { edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]], n: 6,
    pos: ring(155, 120, 66, 6, [0, 1, 2, 3, 4, 5]) };
  const DEC = { edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [6, 7], [7, 8], [8, 9], [9, 1]], n: 10,
    pos: { 0: [155, 78], 1: [155, 150], 2: [98, 184], 3: [40, 150], 4: [40, 78], 5: [98, 44],
           9: [212, 184], 8: [270, 150], 7: [270, 78], 6: [212, 44] } };
  const BIC = { edges: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [5, 6], [6, 7], [7, 8], [8, 9], [9, 5], [0, 5]], n: 10,
    pos: { 0: [120, 114], 1: [85, 66], 2: [31, 86], 3: [31, 142], 4: [85, 162],
           5: [200, 114], 6: [235, 66], 7: [289, 86], 8: [289, 142], 9: [235, 162] } };
  const P43 = { edges: [[0, 1], [1, 2], [2, 3], [4, 5], [5, 6]], n: 7,
    pos: { 0: [40, 90], 1: [110, 90], 2: [180, 90], 3: [250, 90], 4: [75, 165], 5: [145, 165], 6: [215, 165] } };
  const P52 = { edges: [[0, 1], [1, 2], [2, 3], [3, 4], [5, 6]], n: 7,
    pos: { 0: [25, 90], 1: [90, 90], 2: [155, 90], 3: [220, 90], 4: [285, 90], 5: [110, 165], 6: [180, 165] } };

  const PAIRS = [
    { a: TRI2, b: HEX, la: "two triangles", lb: "one hexagon" },
    { a: DEC, b: BIC, la: "decalin", lb: "bicyclopentyl" },
    { a: P43, b: P52, la: "P₄ ∪ P₃", lb: "P₅ ∪ P₂" },
  ];

  let pi = 0, round = 0, colA = [], colB = [], history = [];

  function adj(g) {
    const A = Array.from({ length: g.n }, () => []);
    g.edges.forEach(([a, b]) => { A[a].push(b); A[b].push(a); });
    return A;
  }

  function refineOnce() {
    const P = PAIRS[pi];
    const aA = adj(P.a), aB = adj(P.b);
    const sig = (cols, av, v) => JSON.stringify([cols[v], av[v].map((u) => cols[u]).sort()]);
    const sigsA = P.a.pos && aA.map((_, v) => sig(colA, aA, v));
    const sigsB = aB.map((_, v) => sig(colB, aB, v));
    const table = {};
    let next = 0;
    [...sigsA, ...sigsB].sort().forEach((s) => { if (!(s in table)) table[s] = next++; });
    colA = sigsA.map((s) => table[s]);
    colB = sigsB.map((s) => table[s]);
    round += 1;
  }

  function hist(cols) {
    const h = {};
    cols.forEach((c) => { h[c] = (h[c] || 0) + 1; });
    return h;
  }
  const histsEqual = () => JSON.stringify(hist(colA)) === JSON.stringify(hist(colB));

  function reset() {
    const P = PAIRS[pi];
    colA = Array(P.a.n).fill(0);
    colB = Array(P.b.n).fill(0);
    round = 0;
  }
  reset();

  function drawGraph(g0, gr, cols, P, PAL, label, x0) {
    const g = gr.append("g").attr("transform", `translate(${x0},34)`);
    g0.edges.forEach(([a, b]) => {
      const [x1, y1] = g0.pos[a], [x2, y2] = g0.pos[b];
      g.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
        .attr("stroke", P.muted).attr("stroke-width", 1.8).attr("opacity", 0.55);
    });
    Object.keys(g0.pos).forEach((v) => {
      const [x, y] = g0.pos[v];
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", 11)
        .attr("fill", PAL[cols[v] % PAL.length]).attr("stroke", P.bg).attr("stroke-width", 1.5);
    });
    g.append("text").attr("x", 155).attr("y", 218).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text).text(label);
  }

  function histText(cols) {
    const h = hist(cols);
    return Object.keys(h).sort((a, b) => a - b).map((c) => `${h[c]}×c${c}`).join("  ");
  }

  function render() {
    const P = U.pal();
    const PAL = [P.blue, P.yellow, P.green, "#7c5cd6", P.accent, "#d1567e", "#cf4a30", P.muted];
    const svg = U.svgIn("w9-wl-svg", 760, 356);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const Pr = PAIRS[pi];

    g.append("text").attr("x", 380).attr("y", 22).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`round ${round} · same signature = same color in BOTH graphs · step to refine`);

    drawGraph(Pr.a, g, colA, P, PAL, Pr.la, 35);
    drawGraph(Pr.b, g, colB, P, PAL, Pr.lb, 415);

    g.append("text").attr("x", 190).attr("y", 296).attr("text-anchor", "middle")
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
      .attr("fill", P.text).text(histText(colA));
    g.append("text").attr("x", 570).attr("y", 296).attr("text-anchor", "middle")
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
      .attr("fill", P.text).text(histText(colB));

    const eq = histsEqual();
    let verdict, color;
    if (round === 0) { verdict = "all nodes start the same color — press step"; color = P.muted; }
    else if (eq) {
      verdict = `histograms still match after round ${round} — WL cannot distinguish these graphs`;
      color = P.accentDark;
    } else {
      verdict = `histograms differ at round ${round} — non-isomorphism PROVED`;
      color = P.green;
    }
    g.append("text").attr("x", 380).attr("y", 336).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", color).text(verdict);
  }

  document.getElementById("w9-wl-step").addEventListener("click", () => { refineOnce(); render(); });
  document.getElementById("w9-wl-reset").addEventListener("click", () => { reset(); render(); });
  document.querySelectorAll("#w9-wl-widget [data-pair]").forEach((b) =>
    b.addEventListener("click", () => {
      pi = +b.getAttribute("data-pair");
      document.querySelectorAll("#w9-wl-widget [data-pair]").forEach((x) => x.classList.toggle("active", x === b));
      reset(); render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w9-wl-svg", render);
})();
