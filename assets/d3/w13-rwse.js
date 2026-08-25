/* Widget 13.2 — RWSE fingerprints on the karate club.
 * Pick two nodes; their return-probability vectors (k = 1..8) are computed
 * live by dense matrix powers and drawn as paired bars. Structural twins
 * share fingerprints; hubs and leaves do not. Deterministic.
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
  const A = Array.from({ length: N }, () => Array(N).fill(0));
  EDGES.forEach(([a, b]) => { A[a][b] = 1; A[b][a] = 1; });
  const deg = A.map((r) => r.reduce((s, x) => s + x, 0));
  const M = A.map((r, i) => r.map((x) => x / deg[i]));
  // precompute diagonals of M^k for k=1..8
  const mul = (X, Y) => X.map((row) => row.map((_, j) => row.reduce((s, v, l) => s + v * Y[l][j], 0)));
  const RW = [];
  let Pm = M.map((r) => r.slice());
  RW.push(Pm.map((r, i) => r[i]));
  for (let k = 1; k < 8; k++) {
    Pm = mul(Pm, M);
    RW.push(Pm.map((r, i) => r[i]));
  }
  const PAIRS = [[33, 11], [14, 15], [0, 33]];
  let pi = 0;

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w13-rw-svg", 760, 300);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const [a, b] = PAIRS[pi];

    g.append("text").attr("x", 380).attr("y", 22).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`return probabilities k = 1..8 — node ${a} (deg ${deg[a]}) vs node ${b} (deg ${deg[b]})`);

    const x0 = 90, bw = 34, gap = 78;
    // visible zero baseline: zeros are data here (k=1 is PROVABLY zero — no
    // self-loops), so they must read as measured zeros, not missing bars
    g.append("line").attr("x1", x0 - 10).attr("y1", 210)
      .attr("x2", x0 + 7 * gap + bw + 10).attr("y2", 210)
      .attr("stroke", P.muted).attr("stroke-width", 1.5).attr("opacity", 0.7);
    for (let k = 0; k < 8; k++) {
      const va = RW[k][a], vb = RW[k][b];
      const xx = x0 + k * gap;
      g.append("rect").attr("x", xx).attr("y", 210 - 420 * va)
        .attr("width", bw / 2 - 2).attr("height", 420 * va).attr("fill", P.accent).attr("opacity", 0.85);
      g.append("rect").attr("x", xx + bw / 2).attr("y", 210 - 420 * vb)
        .attr("width", bw / 2 - 2).attr("height", 420 * vb).attr("fill", P.blue).attr("opacity", 0.85);
      if (va < 1e-9 && vb < 1e-9) {
        g.append("text").attr("x", xx + bw / 2).attr("y", 202).attr("text-anchor", "middle")
          .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
          .attr("fill", P.muted).text("0");
      }
      g.append("text").attr("x", xx + bw / 2).attr("y", 230).attr("text-anchor", "middle")
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
        .attr("fill", P.muted).text("k" + (k + 1));
    }
    g.append("text").attr("x", 120).attr("y", 52).attr("font-size", 12.5)
      .attr("font-weight", 700).attr("fill", P.accent).text(`node ${a}`);
    g.append("text").attr("x", 200).attr("y", 52).attr("font-size", 12.5)
      .attr("font-weight", 700).attr("fill", P.blue).text(`node ${b}`);
    const dist = RW.reduce((s, row) => s + Math.abs(row[a] - row[b]), 0);
    g.append("text").attr("x", 380).attr("y", 262).attr("text-anchor", "middle")
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
      .attr("font-weight", 700).attr("fill", P.text)
      .text(`fingerprint L1 distance: ${dist.toFixed(3)}`);
    const verdict = dist < 0.01
      ? "structural twins: identical fingerprints — and no sign to flip, unlike Laplacian PEs"
      : "different roles, different fingerprints — this vector is what the model reads as position";
    g.append("text").attr("x", 380).attr("y", 284).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.accentDark).text(verdict);
  }

  document.querySelectorAll("#w13-rw-widget [data-pair]").forEach((btn) =>
    btn.addEventListener("click", () => {
      pi = +btn.getAttribute("data-pair");
      document.querySelectorAll("#w13-rw-widget [data-pair]").forEach((x) => x.classList.toggle("active", x === btn));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w13-rw-svg", render);
})();
