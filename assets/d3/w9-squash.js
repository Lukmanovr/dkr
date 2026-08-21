/* Widget 9.4 — Oversquashing on the barbell, measured live.
 * Two 5-cliques joined by a two-node path. The display shows exactly how much
 * the starred source can influence every node after K layers — the entry
 * (Â^K)[src][v], computed in the browser. One rescue edge, one order of
 * magnitude. Deterministic.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const N = 12;
  const CLIQUE_L = [0, 1, 2, 3, 4], CLIQUE_R = [7, 8, 9, 10, 11];
  const EDGES = [];
  for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) EDGES.push([CLIQUE_L[i], CLIQUE_L[j]]);
  for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) EDGES.push([CLIQUE_R[i], CLIQUE_R[j]]);
  EDGES.push([4, 5], [5, 6], [6, 7]);
  const RESCUE = [1, 10];
  const POS = { 0: [98, 96], 1: [60, 150], 2: [98, 204], 3: [140, 96], 4: [172, 150],
                5: [330, 150], 6: [430, 150],
                7: [588, 150], 8: [662, 96], 9: [700, 150], 10: [662, 204], 11: [620, 96] };
  const SRC = 0, FAR = 11;

  let K = 5, rescue = false;

  function ahatPow(edges, k) {
    const A = Array.from({ length: N }, () => Array(N).fill(0));
    edges.forEach(([a, b]) => { A[a][b] = 1; A[b][a] = 1; });
    for (let i = 0; i < N; i++) A[i][i] = 1;
    const d = A.map((row) => row.reduce((s, x) => s + x, 0));
    const Ah = A.map((row, i) => row.map((x, j) => x / Math.sqrt(d[i] * d[j])));
    let P = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => (i === j ? 1 : 0)));
    const mul = (X, Y) => X.map((row) => row.map((_, j) => row.reduce((s, v, l) => s + v * Y[l][j], 0)));
    for (let s = 0; s < k; s++) P = mul(P, Ah);
    return P;
  }

  const fmt = (v) => (v === 0 ? "0" : v >= 5e-3 ? v.toFixed(3) : v.toExponential(0).replace("e-", "e-"));

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w9-sq-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const edges = rescue ? [...EDGES, RESCUE] : EDGES;
    const S = ahatPow(edges, K)[SRC];
    const vmax = Math.max(...S);

    g.append("text").attr("x", 380).attr("y", 22).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`influence of the starred node after K = ${K} layers — the exact entry of Â^${K}, per node`);

    edges.forEach(([a, b]) => {
      const [x1, y1] = POS[a], [x2, y2] = POS[b];
      const isRescue = rescue && a === RESCUE[0] && b === RESCUE[1];
      g.append("line").attr("x1", x1).attr("y1", y1 + 40).attr("x2", x2).attr("y2", y2 + 40)
        .attr("stroke", isRescue ? P.green : P.muted)
        .attr("stroke-width", isRescue ? 3 : 1.5).attr("opacity", isRescue ? 0.9 : 0.5);
    });
    Object.keys(POS).forEach((v) => {
      const [x, y] = POS[v];
      const frac = S[v] / vmax;
      g.append("circle").attr("cx", x).attr("cy", y + 40).attr("r", 9 + 10 * Math.sqrt(frac))
        .attr("fill", P.accent).attr("opacity", 0.22 + 0.78 * frac);
      const yoff = y < 150 ? -20 : 32;
      g.append("text").attr("x", x).attr("y", y + 40 + yoff).attr("text-anchor", "middle")
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12)
        .attr("fill", P.text).text(fmt(S[v]));
    });
    g.append("text").attr("x", POS[SRC][0]).attr("y", POS[SRC][1] + 45)
      .attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 700)
      .attr("fill", P.bg).text("★");

    const near = S[4], far = S[FAR];
    const line1 = far === 0
      ? `far clique: EXACTLY 0 — distance 5 > K = ${K}: not squashed, simply unreachable`
      : `same clique ${near.toFixed(3)} vs far clique ${fmt(far)} — ratio ${Math.round(near / far)}×`;
    g.append("text").attr("x", 380).attr("y", 296).attr("text-anchor", "middle")
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
      .attr("fill", P.text).text(line1);
    const verdict = rescue
      ? "one added edge re-routes the flow — the bottleneck, not the distance, was the problem"
      : "everything bound for the far clique squeezes through two path nodes — that is oversquashing";
    g.append("text").attr("x", 380).attr("y", 320).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600)
      .attr("fill", rescue ? P.green : P.accentDark).text(verdict);
  }

  document.querySelectorAll("#w9-sq-widget [data-k]").forEach((b) =>
    b.addEventListener("click", () => {
      K = +b.getAttribute("data-k");
      document.querySelectorAll("#w9-sq-widget [data-k]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.getElementById("w9-sq-rescue").addEventListener("click", (ev) => {
    rescue = !rescue;
    ev.target.classList.toggle("active", rescue);
    ev.target.textContent = rescue ? "remove the rescue edge" : "add one edge across";
    render();
  });

  U.onThemeChange(render);
  U.lazyBoot("w9-sq-svg", render);
})();
