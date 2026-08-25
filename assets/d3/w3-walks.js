/* Widget 3.1 — Walks become co-occurrence counts.
 * The cast graph: grow a random walk step by step, and watch each finished
 * walk's skip-gram windows (±2) pour pairs into a live co-occurrence matrix —
 * the similarity S(u,v) that this week's loss will chase. Seeded PRNG, so a
 * reset replays the identical sequence.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const NAMES = ["A", "B", "C", "D", "E", "F"];
  const COLORS = ["#d9a62e", "#cf4a30", "#199473", "#7c5cd6", "#d1567e", "#0f8377"];
  const EDGES = [[0, 1], [0, 2], [0, 3], [1, 2], [2, 4], [2, 5], [4, 5]];
  const POS = [[150, 100], [245, 43], [310, 113], [98, 167], [254, 180], [398, 167]];
  const ADJ = Array.from({ length: 6 }, () => []);
  EDGES.forEach(([a, b]) => { ADJ[a].push(b); ADJ[b].push(a); });
  const LEN = 8, WIN = 2;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  let rng, C, walk, walksDone;
  function reset() {
    rng = mulberry32(42);
    C = Array.from({ length: 6 }, () => Array(6).fill(0));
    walk = [];
    walksDone = 0;
  }
  reset();

  function pick(arr) { return arr[Math.floor(rng() * arr.length)]; }

  function absorb(w) {
    for (let i = 0; i < w.length; i++) {
      for (let j = Math.max(0, i - WIN); j <= Math.min(w.length - 1, i + WIN); j++) {
        if (j !== i) C[w[i]][w[j]] += 1;
      }
    }
    walksDone += 1;
  }

  function step() {
    if (walk.length >= LEN) walk = [];
    if (walk.length === 0) walk.push(Math.floor(rng() * 6));
    else walk.push(pick(ADJ[walk[walk.length - 1]]));
    if (walk.length === LEN) absorb(walk);
  }

  function runWalks(k) {
    walk = [];
    for (let n = 0; n < k; n++) {
      const w = [Math.floor(rng() * 6)];
      for (let i = 1; i < LEN; i++) w.push(pick(ADJ[w[w.length - 1]]));
      absorb(w);
    }
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w3-wk-svg", 760, 320);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g").attr("transform", "translate(10,26)");

    g.selectAll("line").data(EDGES).join("line")
      .attr("x1", (e) => POS[e[0]][0]).attr("y1", (e) => POS[e[0]][1])
      .attr("x2", (e) => POS[e[1]][0]).attr("y2", (e) => POS[e[1]][1])
      .attr("stroke", P.muted).attr("stroke-width", 2).attr("opacity", 0.5);
    // trail of the walk in progress
    for (let i = 1; i < walk.length; i++) {
      g.append("line")
        .attr("x1", POS[walk[i - 1]][0]).attr("y1", POS[walk[i - 1]][1])
        .attr("x2", POS[walk[i]][0]).attr("y2", POS[walk[i]][1])
        .attr("stroke", P.accent).attr("stroke-width", 3.5).attr("opacity", 0.28 + 0.7 * (i / LEN));
    }
    const cur = walk.length ? walk[walk.length - 1] : -1;
    const nd = g.selectAll("g.n").data(d3.range(6)).join("g")
      .attr("transform", (i) => `translate(${POS[i][0]},${POS[i][1]})`);
    nd.append("circle").attr("r", (i) => (i === cur ? 19 : 15))
      .attr("fill", (i) => COLORS[i])
      .attr("stroke", (i) => (i === cur ? P.accentDark : "none")).attr("stroke-width", 3.5);
    nd.append("text").attr("text-anchor", "middle").attr("dy", 4.5)
      .attr("font-size", 13).attr("font-weight", 700).attr("fill", "#fff")
      .text((i) => NAMES[i]);

    const tokens = walk.map((i) => NAMES[i]).join(" → ") || "press “step” to start a walk";
    g.append("text").attr("x", 245).attr("y", 232).attr("text-anchor", "middle")
      .attr("font-size", 13.5).attr("font-weight", 600)
      .attr("fill", walk.length === LEN ? P.green : P.text)
      .text(walk.length === LEN ? `${tokens}   ✓ pairs added` : tokens);
    g.append("text").attr("x", 245).attr("y", 256).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`walks absorbed so far: ${walksDone}`);
    g.append("text").attr("x", 245).attr("y", 274).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`each adds its (center, context) pairs, window ±${WIN}`);

    // co-occurrence matrix
    const cmax = Math.max(1, ...C.flat());
    const mg = g.append("g").attr("transform", "translate(510,48)");
    mg.append("text").attr("x", 91).attr("y", -2).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.text)
      .text("co-occurrence counts C");
    U.drawMatrix(mg, C, {
      x: 26, y: 10, cell: 26, fontSize: 12.5,
      color: (v) => d3.interpolateRgb(P.paper, P.blue)(Math.pow(v / cmax, 0.6) * 0.85),
      textColor: P.text, borderColor: P.border,
      fmt: (v) => (v === 0 ? "·" : String(v)),
    });
    for (let i = 0; i < 6; i++) {
      mg.append("text").attr("x", 16).attr("y", 10 + i * 26 + 17).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", COLORS[i]).text(NAMES[i]);
      mg.append("text").attr("x", 26 + i * 26 + 13).attr("y", 182).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", COLORS[i]).text(NAMES[i]);
    }
    mg.append("text").attr("x", 91).attr("y", 206).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text("this matrix IS the similarity S");
  }

  document.getElementById("w3-wk-step").addEventListener("click", () => { step(); render(); });
  document.getElementById("w3-wk-run").addEventListener("click", () => { runWalks(100); render(); });
  document.getElementById("w3-wk-reset").addEventListener("click", () => { reset(); render(); });

  U.onThemeChange(render);
  U.lazyBoot("w3-wk-svg", render);
})();
