/* Widget 11.2 — What one gradient step actually loads.
 * The karate club; pick a strategy and see exactly which nodes and edges a
 * single 4-node mini-batch touches: full-batch (everything), neighbor
 * sampling with fanout 3 (bounded), or one random cluster (partition).
 * Seeded PRNG; reseed to feel the variance.
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
  const ADJ = Array.from({ length: N }, () => []);
  EDGES.forEach(([a, b]) => { ADJ[a].push(b); ADJ[b].push(a); });

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // fixed circular-ish layout (reuse polar by community for stability)
  const POS = [];
  for (let v = 0; v < N; v++) {
    const ang = (2 * Math.PI * v) / N - Math.PI / 2;
    POS.push([380 + 195 * Math.cos(ang), 158 + 118 * Math.sin(ang)]);
  }

  let mode = "full", seed = 3;

  function computeBatch() {
    const rng = mulberry32(seed);
    const targets = [];
    while (targets.length < 4) {
      const v = Math.floor(rng() * N);
      if (!targets.includes(v)) targets.push(v);
    }
    if (mode === "full") {
      return { targets, nodes: new Set(Array.from({ length: N }, (_, i) => i)),
               edges: new Set(EDGES.map((_, i) => i)) };
    }
    if (mode === "sample") {
      const nodes = new Set(targets);
      const eset = new Set();
      let frontier = targets;
      for (let hop = 0; hop < 2; hop++) {
        const nxt = [];
        frontier.forEach((v) => {
          const nbrs = ADJ[v].slice();
          const chosen = [];
          for (let k = 0; k < 3 && nbrs.length; k++) {
            chosen.push(nbrs.splice(Math.floor(rng() * nbrs.length), 1)[0]);
          }
          chosen.forEach((u) => {
            nodes.add(u); nxt.push(u);
            const ei = EDGES.findIndex(([a, b]) => (a === v && b === u) || (a === u && b === v));
            if (ei >= 0) eset.add(ei);
          });
        });
        frontier = nxt;
      }
      return { targets, nodes, edges: eset };
    }
    // cluster: random 4-way partition, use the cluster of the first target
    const part = Array.from({ length: N }, () => Math.floor(rng() * 4));
    const c = part[targets[0]];
    const nodes = new Set();
    part.forEach((p, v) => { if (p === c) nodes.add(v); });
    const eset = new Set();
    EDGES.forEach(([a, b], i) => { if (nodes.has(a) && nodes.has(b)) eset.add(i); });
    return { targets: targets.filter((t) => nodes.has(t)), nodes, edges: eset, cluster: c };
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w11-sm-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const { targets, nodes, edges } = computeBatch();

    EDGES.forEach(([a, b], i) => {
      const hot = edges.has(i);
      g.append("line").attr("x1", POS[a][0]).attr("y1", POS[a][1])
        .attr("x2", POS[b][0]).attr("y2", POS[b][1])
        .attr("stroke", hot ? P.accent : P.muted).attr("stroke-width", hot ? 1.8 : 0.7)
        .attr("opacity", hot ? 0.8 : 0.2);
    });
    for (let v = 0; v < N; v++) {
      const loaded = nodes.has(v), isT = targets.includes(v);
      g.append("circle").attr("cx", POS[v][0]).attr("cy", POS[v][1])
        .attr("r", isT ? 11 : 7)
        .attr("fill", isT ? P.yellow : loaded ? P.blue : P.muted)
        .attr("opacity", loaded || isT ? 0.95 : 0.25)
        .attr("stroke", isT ? P.text : "none").attr("stroke-width", 1.5);
    }
    const labels = { full: "full-batch: everything in memory, every step",
                     sample: "neighbor sampling (fanout 3, 2 hops)",
                     cluster: "cluster training: one random part only" };
    g.append("text").attr("x", 380).attr("y", 24).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`${labels[mode]} — batch of ${targets.length} target nodes (gold)`);
    g.append("text").attr("x", 380).attr("y", 306).attr("text-anchor", "middle")
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 13)
      .attr("font-weight", 700).attr("fill", P.accentDark)
      .text(`loaded: ${nodes.size}/34 nodes · ${edges.size}/78 edges`);
    g.append("text").attr("x", 380).attr("y", 326).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", P.muted)
      .text(mode === "cluster" ? "note the lost between-cluster edges — Cluster-GCN's bias, visible"
            : mode === "sample" ? "reseed: different neighbors each time — sampling's variance, visible"
            : "the memory bill of exactness");
  }

  document.querySelectorAll("#w11-sm-widget [data-mode]").forEach((btn) =>
    btn.addEventListener("click", () => {
      mode = btn.getAttribute("data-mode");
      document.querySelectorAll("#w11-sm-widget [data-mode]").forEach((x) => x.classList.toggle("active", x === btn));
      render();
    }));
  document.getElementById("w11-sm-reseed").addEventListener("click", () => { seed += 1; render(); });

  U.onThemeChange(render);
  U.lazyBoot("w11-sm-svg", render);
})();
