/* Widget 12.2 — Heuristics vs ten hidden edges.
 * Ten karate edges are held out (seeded); the message graph is the rest.
 * Pick a heuristic; it ranks ALL message-graph non-edges and the widget
 * highlights its top ten guesses. Hits = guesses that are truly hidden
 * edges. Deterministic.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const N = 34;
  const ALLE = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],[0,12],[0,13],
    [0,17],[0,19],[0,21],[0,31],[1,2],[1,3],[1,7],[1,13],[1,17],[1,19],[1,21],[1,30],
    [2,3],[2,7],[2,8],[2,9],[2,13],[2,27],[2,28],[2,32],[3,7],[3,12],[3,13],[4,6],
    [4,10],[5,6],[5,10],[5,16],[6,16],[8,30],[8,32],[8,33],[9,33],[13,33],[14,32],
    [14,33],[15,32],[15,33],[18,32],[18,33],[19,33],[20,32],[20,33],[22,32],[22,33],
    [23,25],[23,27],[23,29],[23,32],[23,33],[24,25],[24,27],[24,31],[25,31],[26,29],
    [26,33],[27,33],[28,31],[28,33],[29,32],[29,33],[30,32],[30,33],[31,32],[31,33],[32,33]];
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const rng = mulberry32(12);
  const heldIdx = new Set();
  while (heldIdx.size < 10) heldIdx.add(Math.floor(rng() * ALLE.length));
  const HELD = [...heldIdx].map((i) => ALLE[i]);
  const MSG = ALLE.filter((_, i) => !heldIdx.has(i));
  const ADJ = Array.from({ length: N }, () => new Set());
  MSG.forEach(([a, b]) => { ADJ[a].add(b); ADJ[b].add(a); });

  const inter = (A, B) => [...A].filter((x) => B.has(x));
  const SCORERS = {
    cn: (a, b) => inter(ADJ[a], ADJ[b]).length,
    aa: (a, b) => inter(ADJ[a], ADJ[b]).reduce((s, c) => s + (ADJ[c].size > 1 ? 1 / Math.log(ADJ[c].size) : 0), 0),
    jac: (a, b) => {
      const i = inter(ADJ[a], ADJ[b]).length;
      const u = new Set([...ADJ[a], ...ADJ[b]]).size;
      return u ? i / u : 0;
    },
    pa: (a, b) => ADJ[a].size * ADJ[b].size,
  };
  const NAMES = { cn: "common neighbors", aa: "Adamic–Adar", jac: "Jaccard", pa: "pref. attachment" };

  let heur = "cn";
  const POS = [];
  for (let v = 0; v < N; v++) {
    const ang = (2 * Math.PI * v) / N - Math.PI / 2;
    POS.push([300 + 185 * Math.cos(ang), 165 + 115 * Math.sin(ang)]);
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w12-he-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    // rank all non-edges of the message graph
    const isEdge = new Set(MSG.map(([a, b]) => a * 100 + b));
    const cands = [];
    for (let a = 0; a < N; a++)
      for (let b = a + 1; b < N; b++)
        if (!isEdge.has(a * 100 + b)) cands.push([a, b, SCORERS[heur](a, b)]);
    cands.sort((x, y) => y[2] - x[2]);
    const top = cands.slice(0, 10);
    const heldSet = new Set(HELD.map(([a, b]) => a * 100 + b));
    const hits = top.filter(([a, b]) => heldSet.has(a * 100 + b)).length;

    g.append("text").attr("x", 380).attr("y", 22).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`${NAMES[heur]} ranks every non-edge of the message graph — its top 10 guesses drawn bold`);

    MSG.forEach(([a, b]) => {
      g.append("line").attr("x1", POS[a][0]).attr("y1", POS[a][1] + 25)
        .attr("x2", POS[b][0]).attr("y2", POS[b][1] + 25)
        .attr("stroke", P.muted).attr("stroke-width", 0.7).attr("opacity", 0.25);
    });
    HELD.forEach(([a, b]) => {
      g.append("line").attr("x1", POS[a][0]).attr("y1", POS[a][1] + 25)
        .attr("x2", POS[b][0]).attr("y2", POS[b][1] + 25)
        .attr("stroke", P.yellow).attr("stroke-width", 2.4)
        .attr("stroke-dasharray", "5 4").attr("opacity", 0.9);
    });
    top.forEach(([a, b]) => {
      const hit = heldSet.has(a * 100 + b);
      g.append("line").attr("x1", POS[a][0]).attr("y1", POS[a][1] + 25)
        .attr("x2", POS[b][0]).attr("y2", POS[b][1] + 25)
        .attr("stroke", hit ? P.green : P.accent).attr("stroke-width", 3).attr("opacity", 0.9);
    });
    for (let v = 0; v < N; v++) {
      g.append("circle").attr("cx", POS[v][0]).attr("cy", POS[v][1] + 25).attr("r", 6)
        .attr("fill", P.blue).attr("opacity", 0.8);
    }
    g.append("text").attr("x", 380).attr("y", 306).attr("text-anchor", "middle")
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 13)
      .attr("font-weight", 700).attr("fill", hits >= 4 ? P.green : P.accentDark)
      .text(`hits: ${hits}/10 hidden edges found (gold dashed = hidden truth)`);
    g.append("text").attr("x", 380).attr("y", 326).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", P.muted)
      .text(heur === "pa" ? "degree × degree finds hubs, not relationships — the weakest lens"
            : "closed triangles are most of what these heuristics know — Week 2, monetized");
  }

  document.querySelectorAll("#w12-he-widget [data-h]").forEach((b) =>
    b.addEventListener("click", () => {
      heur = b.getAttribute("data-h");
      document.querySelectorAll("#w12-he-widget [data-h]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w12-he-svg", render);
})();
