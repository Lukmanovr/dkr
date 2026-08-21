/* Widget 3.2 — p and q steer the walker.
 * A lollipop graph (a tight clique with a long tail) makes the BFS↔DFS dial
 * visible: pick p and q, launch 200 biased walks from the junction, and read
 * the empirical visit distribution off the node sizes. Seeded PRNG per run —
 * identical settings give identical counts.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const N = 8;
  // clique 0-3, tail 3-4-5-6-7 — node 3 is the junction and every walk's start
  const EDGES = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3],
                 [3, 4], [4, 5], [5, 6], [6, 7]];
  const POS = [[95, 60], [55, 165], [175, 190], [185, 95],
               [300, 120], [415, 145], [530, 120], [645, 145]];
  const ADJ = Array.from({ length: N }, () => []);
  EDGES.forEach(([a, b]) => { ADJ[a].push(b); ADJ[b].push(a); });
  const NBRSET = ADJ.map((ns) => new Set(ns));
  const STEPS = [0.25, 0.5, 1, 2, 4];
  const START = 3, WALKS = 200, LEN = 10;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  let visits = null;

  function pv() { return STEPS[+document.getElementById("w3-pq-p").value]; }
  function qv() { return STEPS[+document.getElementById("w3-pq-q").value]; }

  function runWalks() {
    const p = pv(), q = qv();
    const rng = mulberry32(7);
    visits = Array(N).fill(0);
    for (let w = 0; w < WALKS; w++) {
      let prev = -1, cur = START;
      for (let s = 0; s < LEN; s++) {
        const nbrs = ADJ[cur];
        let next;
        if (prev < 0) {
          next = nbrs[Math.floor(rng() * nbrs.length)];
        } else {
          const wts = nbrs.map((x) => (x === prev ? 1 / p : NBRSET[prev].has(x) ? 1 : 1 / q));
          const z = wts.reduce((a, b) => a + b, 0);
          let r = rng() * z;
          next = nbrs[nbrs.length - 1];
          for (let i = 0; i < nbrs.length; i++) { r -= wts[i]; if (r <= 0) { next = nbrs[i]; break; } }
        }
        visits[next] += 1;
        prev = cur; cur = next;
      }
    }
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w3-pq-svg", 760, 300);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g").attr("transform", "translate(30,40)");

    g.selectAll("line").data(EDGES).join("line")
      .attr("x1", (e) => POS[e[0]][0]).attr("y1", (e) => POS[e[0]][1])
      .attr("x2", (e) => POS[e[1]][0]).attr("y2", (e) => POS[e[1]][1])
      .attr("stroke", P.muted).attr("stroke-width", 2).attr("opacity", 0.55);

    const total = visits ? visits.reduce((a, b) => a + b, 0) : 0;
    const vmax = visits ? Math.max(...visits) : 1;
    const nd = g.selectAll("g.n").data(d3.range(N)).join("g")
      .attr("transform", (i) => `translate(${POS[i][0]},${POS[i][1]})`);
    nd.append("circle")
      .attr("r", (i) => 11 + (visits ? 17 * (visits[i] / vmax) : 0))
      .attr("fill", (i) => (i === START ? P.accent : i <= 3 ? P.blue : P.purple))
      .attr("opacity", (i) => (visits ? 0.35 + 0.65 * (visits[i] / vmax) : 0.75));
    nd.append("text").attr("text-anchor", "middle").attr("dy", 4.5)
      .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", "#fff").text((i) => i);
    if (visits) {
      nd.append("text").attr("text-anchor", "middle")
        .attr("y", (i) => -(13 + 17 * (visits[i] / vmax)) - 5)
        .attr("font-size", 12).attr("fill", P.muted)
        .text((i) => `${Math.round(100 * visits[i] / total)}%`);
    }

    g.append("text").attr("x", 110).attr("y", 218).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.blue).text("the clique (teal)");
    g.append("text").attr("x", 480).attr("y", 218).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.purple).text("the tail (purple)");

    let verdict = "choose p and q, then launch the walkers from node 3";
    if (visits) {
      const clique = visits.slice(0, 4).reduce((a, b) => a + b, 0) / total;
      const deep = (visits[6] + visits[7]) / total;
      verdict = `${WALKS} walks × ${LEN} steps · clique ${Math.round(clique * 100)}% · deep tail (6,7) ${Math.round(deep * 100)}% — ` +
        (deep < 0.02 ? "orbiting the start: BFS-flavored, sees local structure" :
         deep > 0.12 ? "escaping down the tail: DFS-flavored, sees far-away structure" :
         "balanced exploration");
    }
    g.append("text").attr("x", 350).attr("y", 248).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", visits ? P.text : P.muted)
      .text(verdict);

    document.getElementById("w3-pq-pval").textContent = pv();
    document.getElementById("w3-pq-qval").textContent = qv();
  }

  document.getElementById("w3-pq-go").addEventListener("click", () => { runWalks(); render(); });
  document.getElementById("w3-pq-p").addEventListener("input", () => { visits = null; render(); });
  document.getElementById("w3-pq-q").addEventListener("input", () => { visits = null; render(); });

  U.onThemeChange(render);
  U.lazyBoot("w3-pq-svg", render);
})();
