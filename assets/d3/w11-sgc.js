/* Widget 11.3 — SGC: propagation as preprocessing.
 * The karate club with a two-community signal (+1 left faction, -1 right).
 * Slide K: node colors show Â^K x — the feature table an SGC classifier
 * would train on. K=0 is raw features; K=2 is the sweet spot; K=8 is Week
 * 9's oversmoothing arriving on schedule. Deterministic linear algebra.
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
    [26,33],[27,33],[28,31],[28,33],[29,32],[29,33],[30,32],[30,33],[31,32],[31,33],[32,33]];
  const MRHI = [0,1,2,3,4,5,6,7,10,11,12,13,16,17,19,21];   // instructor faction
  // noisy starting signal: faction sign, but three nodes mislabeled
  const X0 = Array.from({ length: N }, (_, v) => (MRHI.includes(v) ? 1 : -1));
  [8, 19, 30].forEach((v) => { X0[v] = -X0[v]; });

  // Â = D^-1/2 (A+I) D^-1/2, dense
  const A = Array.from({ length: N }, () => Array(N).fill(0));
  EDGES.forEach(([a, b]) => { A[a][b] = 1; A[b][a] = 1; });
  for (let i = 0; i < N; i++) A[i][i] = 1;
  const d = A.map((r) => r.reduce((s, x) => s + x, 0));
  const Ah = A.map((r, i) => r.map((x, j) => x / Math.sqrt(d[i] * d[j])));
  const propagate = (x) => Ah.map((row) => row.reduce((s, w, j) => s + w * x[j], 0));

  const POS = [];
  for (let v = 0; v < N; v++) {
    const ang = (2 * Math.PI * v) / N - Math.PI / 2;
    POS.push([260 + 175 * Math.cos(ang), 160 + 115 * Math.sin(ang)]);
  }

  let K = 2;

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w11-sg2-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    let x = X0.slice();
    for (let k = 0; k < K; k++) x = propagate(x);
    const vmax = Math.max(...x.map(Math.abs), 1e-9);

    EDGES.forEach(([a, b]) => {
      g.append("line").attr("x1", POS[a][0]).attr("y1", POS[a][1])
        .attr("x2", POS[b][0]).attr("y2", POS[b][1])
        .attr("stroke", P.muted).attr("stroke-width", 0.8).attr("opacity", 0.3);
    });
    let correct = 0;
    for (let v = 0; v < N; v++) {
      const val = x[v] / vmax;
      const col = val >= 0 ? P.blue : P.accent;
      const truth = MRHI.includes(v) ? 1 : -1;
      if (Math.sign(x[v] || 1) === truth) correct += 1;
      g.append("circle").attr("cx", POS[v][0]).attr("cy", POS[v][1])
        .attr("r", 6 + 6 * Math.abs(val)).attr("fill", col)
        .attr("opacity", 0.35 + 0.6 * Math.abs(val));
    }
    g.append("text").attr("x", 380).attr("y", 24).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`Â^${K} x — the table SGC's classifier trains on · color = sign, size = magnitude`);
    const rows = [
      `sign matches true faction: ${correct}/34`,
      `spread of |values|: ${(Math.max(...x.map(Math.abs)) - Math.min(...x.map(Math.abs))).toFixed(3)}`,
    ];
    rows.forEach((t, i) => {
      g.append("text").attr("x", 520).attr("y", 110 + i * 26)
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
        .attr("fill", P.text).text(t);
    });
    const verdict = K === 0 ? "raw signal: three planted errors intact — no propagation, no repair"
      : K <= 3 ? "a few hops of averaging fix the planted errors — this is ALL a shallow GCN was doing"
      : "Week 9 called it: everything collapses toward the dominant direction — K is a dial, not a virtue";
    g.append("text").attr("x", 380).attr("y", 316).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600)
      .attr("fill", K <= 3 && K > 0 ? P.green : P.accentDark).text(verdict);
  }

  document.querySelectorAll("#w11-sg2-widget [data-k]").forEach((btn) =>
    btn.addEventListener("click", () => {
      K = +btn.getAttribute("data-k");
      document.querySelectorAll("#w11-sg2-widget [data-k]").forEach((x) => x.classList.toggle("active", x === btn));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w11-sg2-svg", render);
})();
