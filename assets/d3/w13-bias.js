/* Widget 13.3 — Graphormer's spatial bias, driven by hand.
 * A 10-node branched path. Attention weights are computed from a distance
 * bias alone: softmax over b[dist(u,v)] with three selectable bias
 * profiles. Decaying bias rediscovers locality; flat dissolves the graph;
 * inverted specializes in long range. Deterministic BFS distances.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const N = 10;
  const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [2, 7], [7, 8], [8, 9]];
  const ADJ = Array.from({ length: N }, () => []);
  EDGES.forEach(([a, b]) => { ADJ[a].push(b); ADJ[b].push(a); });
  const POS = [[70, 90], [150, 90], [230, 90], [310, 90], [390, 90], [470, 90], [550, 90],
               [230, 190], [310, 190], [390, 190]];
  // all-pairs BFS distance
  const DIST = [];
  for (let s = 0; s < N; s++) {
    const d = Array(N).fill(Infinity);
    d[s] = 0;
    let fr = [s];
    while (fr.length) {
      const nxt = [];
      fr.forEach((v) => ADJ[v].forEach((u) => {
        if (d[u] === Infinity) { d[u] = d[v] + 1; nxt.push(u); }
      }));
      fr = nxt;
    }
    DIST.push(d);
  }
  const PROFILES = {
    decay: { b: [4, 2.5, 1, 0, -1, -2, -3], name: "decaying with distance" },
    flat: { b: [0, 0, 0, 0, 0, 0, 0], name: "flat (no structure)" },
    invert: { b: [-3, -2, -1, 0, 1, 2.5, 4], name: "growing with distance" },
  };
  let prof = "decay";
  const SRC = 2;

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w13-bi-svg", 760, 320);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const bias = PROFILES[prof].b;
    const logits = DIST[SRC].map((d, v) => (v === SRC ? -1e9 : bias[Math.min(d, 6)]));
    const mx = Math.max(...logits);
    const exps = logits.map((l) => Math.exp(l - mx));
    const Zs = exps.reduce((a, b2) => a + b2, 0);
    const att = exps.map((e) => e / Zs);

    g.append("text").attr("x", 380).attr("y", 22).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`attention from the starred node, bias ${PROFILES[prof].name} — content terms zeroed to isolate the prior`);

    EDGES.forEach(([a, b]) => {
      g.append("line").attr("x1", POS[a][0]).attr("y1", POS[a][1] + 20)
        .attr("x2", POS[b][0]).attr("y2", POS[b][1] + 20)
        .attr("stroke", P.muted).attr("stroke-width", 1.5).attr("opacity", 0.4);
    });
    const amax = Math.max(...att);
    for (let v = 0; v < N; v++) {
      const w = att[v] / amax;
      g.append("circle").attr("cx", POS[v][0]).attr("cy", POS[v][1] + 20)
        .attr("r", v === SRC ? 13 : 8 + 8 * w)
        .attr("fill", v === SRC ? P.text : P.accent)
        .attr("opacity", v === SRC ? 0.9 : 0.25 + 0.7 * w);
      if (v !== SRC) {
        g.append("text").attr("x", POS[v][0]).attr("y", POS[v][1] - 6)
          .attr("text-anchor", "middle").attr("font-family", "'JetBrains Mono', monospace")
          .attr("font-size", 12.5).attr("fill", P.text).text(att[v].toFixed(2));
      }
    }
    g.append("text").attr("x", POS[SRC][0]).attr("y", POS[SRC][1] + 25)
      .attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 700)
      .attr("fill", P.bg).text("★");
    // bias profile mini-chart (label sits ABOVE the inset so no bar can reach it)
    g.append("text").attr("x", 626).attr("y", 220).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text("b[dist] profile");
    bias.slice(0, 6).forEach((bv, d) => {
      g.append("rect").attr("x", 560 + d * 22).attr("y", 290 - 8 * (bv + 3.5))
        .attr("width", 18).attr("height", 8 * (bv + 3.5)).attr("fill", "#7c5cd6").attr("opacity", 0.7);
    });
    const verdict = prof === "decay"
      ? "a decaying prior rediscovers message passing — locality as a LEARNED preference, not a law"
      : prof === "flat"
      ? "flat prior: attention sees a bag of atoms — the graph has left the building"
      : "an inverted prior attends far first — the anti-oversquashing specialist no MPNN can be";
    g.append("text").attr("x", 380).attr("y", 314).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark).text(verdict);
  }

  document.querySelectorAll("#w13-bi-widget [data-p]").forEach((btn) =>
    btn.addEventListener("click", () => {
      prof = btn.getAttribute("data-p");
      document.querySelectorAll("#w13-bi-widget [data-p]").forEach((x) => x.classList.toggle("active", x === btn));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w13-bi-svg", render);
})();
