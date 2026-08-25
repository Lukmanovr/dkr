/* Widget 7.2 — Neighbor sampling: pay less, accept noise.
 * GraphSAGE's core bargain on the karate club: aggregate a SAMPLE of the
 * neighborhood instead of all of it. Pick a hub, choose the sample size K,
 * resample — the aggregated estimate jitters around the exact mean, and the
 * running spread quantifies exactly what the speedup costs. Seeded PRNG.
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
    [26,33],[27,33],[28,31],[28,33],[29,32],[29,33],[30,32],[30,33],[31,32],[31,33],
    [32,33]];
  const ADJ = Array.from({ length: N }, () => []);
  EDGES.forEach(([a, b]) => { ADJ[a].push(b); ADJ[b].push(a); });
  const DEG = ADJ.map((ns) => ns.length);       // the scalar feature we aggregate
  const CENTERS = [33, 0, 2, 8];

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  let rng = mulberry32(11);
  let ci = 0, K = 3, sample = [], estimates = [];

  function resample() {
    const center = CENTERS[ci];
    const nbrs = ADJ[center];
    if (K >= nbrs.length) {
      sample = nbrs.slice();
    } else {
      const pool = nbrs.slice();
      sample = [];
      for (let i = 0; i < K; i++) {
        sample.push(pool.splice(Math.floor(rng() * pool.length), 1)[0]);
      }
    }
    const est = sample.reduce((a, v) => a + DEG[v], 0) / sample.length;
    estimates.push(est);
    if (estimates.length > 200) estimates.shift();
  }

  function reset() {
    rng = mulberry32(11);
    estimates = [];
    resample();
  }
  reset();

  function render() {
    const P = U.pal();
    const center = CENTERS[ci];
    const nbrs = ADJ[center];
    const exact = nbrs.reduce((a, v) => a + DEG[v], 0) / nbrs.length;
    const svg = U.svgIn("w7-sg-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    // the star: center + its neighborhood, sampled ones highlighted
    const cx0 = 210, cy0 = 150, R = 110;
    nbrs.forEach((v, i) => {
      const ang = (2 * Math.PI * i) / nbrs.length - Math.PI / 2;
      const x = cx0 + R * Math.cos(ang), y = cy0 + R * Math.sin(ang);
      const inS = sample.includes(v);
      g.append("line").attr("x1", cx0).attr("y1", cy0).attr("x2", x).attr("y2", y)
        .attr("stroke", inS ? P.accent : P.muted).attr("stroke-width", inS ? 2.6 : 1.2)
        .attr("opacity", inS ? 0.9 : 0.35);
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", inS ? 11 : 7)
        .attr("fill", inS ? P.accent : P.muted).attr("opacity", inS ? 0.95 : 0.45);
      g.append("text").attr("x", x).attr("y", y + 4).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", "#fff")
        .text(inS ? DEG[v] : "");
    });
    g.append("circle").attr("cx", cx0).attr("cy", cy0).attr("r", 16).attr("fill", P.yellow);
    g.append("text").attr("x", cx0).attr("y", cy0 + 5).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 700).attr("fill", "#5c4508").text(center);

    // full-width bottom caption line — fills the band under both columns
    g.append("text").attr("x", 380).attr("y", 306).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`node ${center}: ${nbrs.length} neighbors — sampling ${Math.min(K, nbrs.length)} (numbers = their degrees)`);

    // estimate panel
    const est = estimates[estimates.length - 1];
    const mean = estimates.reduce((a, b) => a + b, 0) / estimates.length;
    const sd = estimates.length > 1
      ? Math.sqrt(estimates.reduce((a, b) => a + (b - mean) ** 2, 0) / (estimates.length - 1))
      : 0;
    const rows = [
      [`this sample's mean-degree estimate: ${est.toFixed(2)}`, P.accentDark, 700],
      [`exact neighborhood mean: ${exact.toFixed(2)}`, P.text, 700],
      [`${estimates.length} resamples · spread (std): ${sd.toFixed(2)}`, P.muted, 400],
      [`cost this hop: ${Math.min(K, nbrs.length)} msgs, not ${nbrs.length}`, P.muted, 400],
    ];
    rows.forEach(([txt, color, w], i) => {
      g.append("text").attr("x", 430).attr("y", 104 + i * 34)
        .attr("font-size", 13).attr("font-weight", w).attr("fill", color).text(txt);
    });
    const verdict = K >= nbrs.length
      ? "K ≥ degree: no noise — and no savings"
      : sd < 0.65 ? "cheap AND steady — the bargain holds"
      : "noisy — average over epochs, or raise K";
    g.append("text").attr("x", 430).attr("y", 104 + 4 * 34 + 6)
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.green).text(verdict);
  }

  document.getElementById("w7-sg-resample").addEventListener("click", () => { resample(); render(); });
  document.getElementById("w7-sg-center").addEventListener("click", () => {
    ci = (ci + 1) % CENTERS.length; reset(); render();
  });
  document.querySelectorAll("#w7-sg-widget [data-k]").forEach((b) =>
    b.addEventListener("click", () => {
      K = +b.getAttribute("data-k");
      document.querySelectorAll("#w7-sg-widget [data-k]").forEach((x) => x.classList.toggle("active", x === b));
      reset(); render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w7-sg-svg", render);
})();
