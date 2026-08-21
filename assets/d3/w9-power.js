/* Widget 9.3 — Drilling through the ceiling (and the honesty test).
 * Choose an initial feature; WL runs to stability on TWO test pairs at once:
 *   pair A — two triangles vs one hexagon (genuinely different graphs)
 *   pair B — two relabeled copies of the hexagon (the SAME graph)
 * A good augmentation separates A and leaves B alone. Random IDs separate
 * both — a false alarm the honesty panel catches. Seeded PRNG for reseeds.
 */
(function () {
  "use strict";
  const U = window.DKR;

  const TRI2 = [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3]];
  const HEX = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]];
  // the same hexagon, nodes relabeled by the bijection v -> (5v + 2) mod 6
  const HEX2 = HEX.map(([a, b]) => [(5 * a + 2) % 6, (5 * b + 2) % 6]);

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function adj(edges, n) {
    const A = Array.from({ length: n }, () => []);
    edges.forEach(([a, b]) => { A[a].push(b); A[b].push(a); });
    return A;
  }

  function triangles(A) {          // per-node triangle count
    return A.map((nbrs, v) => {
      let t = 0;
      for (let i = 0; i < nbrs.length; i++)
        for (let j = i + 1; j < nbrs.length; j++)
          if (A[nbrs[i]].includes(nbrs[j])) t++;
      return t;
    });
  }

  function rwReturn(A, k) {        // return probability after k steps, per node
    const n = A.length;
    let P = A.map((_, i) => A.map((_, j) => (i === j ? 1 : 0)));
    const M = A.map((nbrs) => {
      const row = Array(n).fill(0);
      nbrs.forEach((u) => { row[u] = 1 / nbrs.length; });
      return row;
    });
    for (let s = 0; s < k; s++) {
      P = P.map((row) => row.map((_, j) => row.reduce((acc, val, l) => acc + val * M[l][j], 0)));
    }
    return P.map((row, i) => Math.round(row[i] * 1e6));
  }

  let feature = "uniform", seed = 5;

  function initColors(edges, n, rng) {
    const A = adj(edges, n);
    if (feature === "uniform") return Array(n).fill(0);
    if (feature === "degree") return A.map((x) => x.length);
    if (feature === "triangles") return triangles(A);
    if (feature === "rw") return rwReturn(A, 3);
    return A.map(() => Math.floor(rng() * 1e9));   // random IDs
  }

  function wlSeparates(eA, eB, n, rng) {
    let cA = initColors(eA, n, rng), cB = initColors(eB, n, rng);
    const aA = adj(eA, n), aB = adj(eB, n);
    for (let r = 0; r < n; r++) {
      const sig = (c, Ad, v) => JSON.stringify([c[v], Ad[v].map((u) => c[u]).sort()]);
      const sA = cA.map((_, v) => sig(cA, aA, v)), sB = cB.map((_, v) => sig(cB, aB, v));
      const tab = {};
      let next = 0;
      [...sA, ...sB].sort().forEach((s) => { if (!(s in tab)) tab[s] = next++; });
      cA = sA.map((s) => tab[s]); cB = sB.map((s) => tab[s]);
    }
    const h = (c) => JSON.stringify(c.slice().sort((x, y) => x - y));
    return h(cA) !== h(cB);
  }

  const FEATURES = [
    ["uniform", "no features"], ["degree", "+ degree"], ["triangles", "+ triangle count"],
    ["rw", "+ RW return prob"], ["random", "+ random IDs"],
  ];

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w9-pw-svg", 760, 300);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    const rng1 = mulberry32(seed), rng2 = mulberry32(seed + 1);
    const sepA = wlSeparates(TRI2, HEX, 6, rng1);
    const sepB = wlSeparates(HEX, HEX2, 6, rng2);

    g.append("text").attr("x", 380).attr("y", 24).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("an augmentation is only good if it separates DIFFERENT graphs and spares the SAME graph");

    const rows = [
      ["pair A · two triangles vs one hexagon", "different graphs", sepA,
       sepA ? "separated — the blindness is cured" : "NOT separated — still blind",
       sepA ? P.green : P.accentDark, sepA],
      ["pair B · hexagon vs relabeled hexagon", "the same graph", sepB,
       sepB ? "“separated” — a FALSE ALARM on the very same graph" : "not separated — correctly so",
       sepB ? "#cf4a30" : P.green, !sepB],
    ];
    rows.forEach(([title, truth, , verdict, color, ok], i) => {
      const y = 66 + i * 92;
      g.append("rect").attr("x", 40).attr("y", y - 30).attr("width", 680).attr("height", 74)
        .attr("rx", 10).attr("fill", color).attr("opacity", 0.09);
      g.append("text").attr("x", 60).attr("y", y - 6).attr("font-size", 13.5)
        .attr("font-weight", 700).attr("fill", P.text).text(title);
      g.append("text").attr("x", 60).attr("y", y + 16).attr("font-size", 12.5)
        .attr("fill", P.muted).text(`ground truth: ${truth}`);
      g.append("text").attr("x", 60).attr("y", y + 36).attr("font-size", 13)
        .attr("font-weight", 600).attr("fill", color).text(`WL verdict with this feature: ${verdict}`);
      g.append("text").attr("x", 690).attr("y", y + 8).attr("text-anchor", "end")
        .attr("font-size", 20).text(ok ? "✓" : "✗").attr("fill", color).attr("font-weight", 700);
    });

    let moral;
    if (feature === "triangles" || feature === "rw") {
      moral = "structural features: computed FROM the graph, so isomorphic copies agree — power without lies";
    } else if (feature === "random") {
      moral = "random IDs break every symmetry — including the ones that were telling the truth";
    } else {
      moral = "uniform and degree features leave regular graphs uniform — the ceiling in its pure form";
    }
    g.append("text").attr("x", 380).attr("y", 280).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.text).text(moral);
  }

  document.querySelectorAll("#w9-pw-widget [data-feature]").forEach((b) =>
    b.addEventListener("click", () => {
      feature = b.getAttribute("data-feature");
      document.querySelectorAll("#w9-pw-widget [data-feature]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.getElementById("w9-pw-reseed").addEventListener("click", () => { seed += 7; render(); });

  U.onThemeChange(render);
  U.lazyBoot("w9-pw-svg", render);
})();
