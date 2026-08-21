/* Widget 3.3 — Watch embeddings learn.
 * Real skip-gram with negative sampling, trained live in your browser on the
 * karate club, in d = 2 — so the embedding space IS the picture, no projection.
 * Walks and initialization use a seeded PRNG: reset replays the same run.
 * The separation readout is a nearest-centroid probe against the true 1977
 * factions (the labels are never shown to the training).
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
  const MR_HI = new Set([0,1,2,3,4,5,6,7,8,10,11,12,13,16,17,19,21]);
  const ADJ = Array.from({ length: N }, () => []);
  EDGES.forEach(([a, b]) => { ADJ[a].push(b); ADJ[b].push(a); });
  const C_HI = "#d9a62e", C_OFF = "#0f8377";
  const WIN = 3, NEG = 2, LR = 0.08;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const sigma = (x) => 1 / (1 + Math.exp(-x));

  let rng, Z, pairs, cursor, trained, auto = false;

  function reset() {
    auto = false;
    rng = mulberry32(9);
    Z = Array.from({ length: N }, () => [rng() * 1.6 - 0.8, rng() * 1.6 - 0.8]);
    pairs = [];
    for (let r = 0; r < 10; r++) {
      for (let s = 0; s < N; s++) {
        const walk = [s];
        for (let i = 1; i < 12; i++) {
          const nb = ADJ[walk[i - 1]];
          walk.push(nb[Math.floor(rng() * nb.length)]);
        }
        for (let i = 0; i < walk.length; i++) {
          for (let j = Math.max(0, i - WIN); j <= Math.min(walk.length - 1, i + WIN); j++) {
            if (j !== i) pairs.push([walk[i], walk[j]]);
          }
        }
      }
    }
    for (let i = pairs.length - 1; i > 0; i--) {           // seeded shuffle
      const j = Math.floor(rng() * (i + 1));
      [pairs[i], pairs[j]] = [pairs[j], pairs[i]];
    }
    cursor = 0; trained = 0;
  }
  reset();

  function sgnsStep([u, v]) {
    const zu = Z[u], zv = Z[v];
    const s = sigma(zu[0] * zv[0] + zu[1] * zv[1]);
    const g = 1 - s;
    const gu = [g * zv[0], g * zv[1]];
    zv[0] += LR * g * zu[0]; zv[1] += LR * g * zu[1];
    for (let k = 0; k < NEG; k++) {
      const n = Math.floor(rng() * N);
      if (n === u || n === v) continue;
      const zn = Z[n];
      const sn = sigma(zu[0] * zn[0] + zu[1] * zn[1]);
      gu[0] -= sn * zn[0]; gu[1] -= sn * zn[1];
      zn[0] -= LR * sn * zu[0]; zn[1] -= LR * sn * zu[1];
    }
    zu[0] += LR * gu[0]; zu[1] += LR * gu[1];
  }

  function train(k) {
    for (let i = 0; i < k; i++) {
      sgnsStep(pairs[cursor]);
      cursor = (cursor + 1) % pairs.length;
    }
    trained += k;
  }

  function separation() {
    const c = [[0, 0, 0], [0, 0, 0]];                     // [sumx, sumy, count]
    for (let i = 0; i < N; i++) {
      const cl = MR_HI.has(i) ? 0 : 1;
      c[cl][0] += Z[i][0]; c[cl][1] += Z[i][1]; c[cl][2] += 1;
    }
    const cen = c.map(([x, y, n]) => [x / n, y / n]);
    let ok = 0;
    for (let i = 0; i < N; i++) {
      const d0 = (Z[i][0] - cen[0][0]) ** 2 + (Z[i][1] - cen[0][1]) ** 2;
      const d1 = (Z[i][0] - cen[1][0]) ** 2 + (Z[i][1] - cen[1][1]) ** 2;
      if ((d0 < d1) === MR_HI.has(i)) ok += 1;
    }
    return ok;
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w3-em-svg", 760, 350);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g").attr("transform", "translate(20,30)");

    const xs = Z.map((z) => z[0]), ys = Z.map((z) => z[1]);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    const sx = (x) => 40 + ((x - xmin) / (xmax - xmin || 1)) * 640;
    const sy = (y) => 14 + ((y - ymin) / (ymax - ymin || 1)) * 230;

    for (let i = 0; i < N; i++) {
      const hot = i === 0 || i === 33;
      const ring = i === 2 || i === 8;
      g.append("circle")
        .attr("cx", sx(Z[i][0])).attr("cy", sy(Z[i][1]))
        .attr("r", hot ? 11 : 7)
        .attr("fill", MR_HI.has(i) ? C_HI : C_OFF).attr("opacity", 0.9)
        .attr("stroke", ring ? "#cf4a30" : "none").attr("stroke-width", 2.4);
    }

    const sep = separation();
    g.append("text").attr("x", 350).attr("y", 278).attr("text-anchor", "middle")
      .attr("font-size", 13.5).attr("font-weight", 700)
      .attr("fill", sep >= 30 ? P.green : P.text)
      .text(trained === 0
        ? "random initialization — the factions are shuffled together"
        : `${trained.toLocaleString()} pairs trained · nearest-centroid separation: ${sep}/34`);
    g.append("text").attr("x", 350).attr("y", 296).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", P.muted)
      .text("d = 2: this plane IS the embedding space");
    g.append("text").attr("x", 350).attr("y", 312).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", P.muted)
      .text("gold Mr. Hi's · teal the Officer's · rings = boundary members 2, 8");
  }

  function autoLoop() {
    if (!auto) return;
    train(200);
    render();
    requestAnimationFrame(autoLoop);
  }

  document.getElementById("w3-em-train").addEventListener("click", () => { train(500); render(); });
  document.getElementById("w3-em-auto").addEventListener("click", (ev) => {
    if (!U.motionOK()) { train(2000); render(); return; }   // reduced motion: one big jump
    auto = !auto;
    ev.currentTarget.classList.toggle("active", auto);
    if (auto) requestAnimationFrame(autoLoop);
  });
  document.getElementById("w3-em-reset").addEventListener("click", () => {
    const btn = document.getElementById("w3-em-auto");
    btn.classList.remove("active");
    reset(); render();
  });

  U.onThemeChange(render);
  U.lazyBoot("w3-em-svg", render);
})();
