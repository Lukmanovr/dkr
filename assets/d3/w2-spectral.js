/* Widget 2.x — Two facts about the Laplacian, computed live (lecture §4.1).
 *
 * Left: a graph whose nodes are coloured by the SIGN of their Fiedler-vector
 * entry and labelled with the entry itself; the edges that cross the sign
 * boundary — the cut — are dashed red and counted. Right: every eigenvalue of
 * L = D − A on a vertical strip, the zero eigenvalues counted (their number is
 * the number of connected components, @prp-components) and λ₂ singled out.
 *
 * Three graphs: the cast graph (connected), the cast graph with C–E and C–F
 * deleted (two pieces, so eigenvalue 0 appears twice), and a barbell — two
 * 4-cliques joined by one bridge — where λ₂ is tiny and the sign change falls
 * exactly on the bridge.
 *
 * Everything shown is computed here from the edge lists with U.symEig
 * (Jacobi), which was checked against numpy on these graphs before shipping.
 * Sign convention: eigenvectors are defined up to sign, so each vector is
 * flipped to make its first node's entry negative. On the disconnected graph
 * λ₂ = 0 has a two-dimensional eigenspace and "the" Fiedler vector is not
 * unique; the widget shows the zero-eigenvector orthogonal to the constant
 * vector 𝟏 (unique up to sign, exactly the lecture's variational definition),
 * which is constant on each piece, and colours the nodes grey.
 * The "cheapest k | n−k split" claims are checked by brute force over every
 * balanced split, not asserted.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 440;

  // ── the three graphs ────────────────────────────────────────────────────
  const CAST = ["A", "B", "C", "D", "E", "F"];
  const CAST_EDGES = [["A", "B"], ["A", "C"], ["A", "D"], ["B", "C"], ["C", "E"], ["C", "F"], ["E", "F"]];
  const CAST_POS = { A: [150, 128], B: [290, 92], C: [282, 212], D: [88, 232], E: [150, 300], F: [292, 315] };

  const BAR = ["1", "2", "3", "4", "5", "6", "7", "8"];
  const clique = (ids) => {
    const out = [];
    for (let i = 0; i < ids.length; i++) for (let j = i + 1; j < ids.length; j++) out.push([ids[i], ids[j]]);
    return out;
  };
  const BAR_EDGES = [...clique(["1", "2", "3", "4"]), ...clique(["5", "6", "7", "8"]), ["4", "5"]];
  // two diamonds; node 4 and node 5 are the bridge ends, facing each other
  const BAR_POS = {
    "1": [92, 125], "2": [47, 195], "3": [92, 265], "4": [137, 195],
    "5": [263, 195], "6": [308, 125], "7": [353, 195], "8": [308, 265],
  };

  const same = (e, f) => (e[0] === f[0] && e[1] === f[1]) || (e[0] === f[1] && e[1] === f[0]);
  const DELETED = [["C", "E"], ["C", "F"]];

  const GRAPHS = {
    cast: {
      names: CAST, edges: CAST_EDGES, pos: CAST_POS, ghost: [],
      head: (n, m) => `the cast graph · ${n} people, ${m} edges`,
    },
    pieces: {
      names: CAST, edges: CAST_EDGES.filter((e) => !DELETED.some((d) => same(e, d))), pos: CAST_POS, ghost: DELETED,
      head: (n, m) => `cast graph minus C–E, C–F · ${n} people, ${m} edges`,
    },
    barbell: {
      names: BAR, edges: BAR_EDGES, pos: BAR_POS, ghost: [], bridge: ["4", "5"],
      head: (n, m) => `two 4-cliques + one bridge · ${n} nodes, ${m} edges`,
    },
  };

  // ── the arithmetic ──────────────────────────────────────────────────────
  const ZERO = 1e-7;

  function analyse(G) {
    const n = G.names.length;
    const E = G.edges.map(([a, b]) => [G.names.indexOf(a), G.names.indexOf(b)]);
    const L = Array.from({ length: n }, () => Array(n).fill(0));
    E.forEach(([a, b]) => { L[a][b] -= 1; L[b][a] -= 1; L[a][a] += 1; L[b][b] += 1; });
    const { values, vectors } = U.symEig(L);
    const zeros = values.filter((v) => Math.abs(v) < ZERO).length;

    let f;
    if (zeros === 1) {
      f = vectors[1].slice();
    } else {
      // λ₂ = 0 too: pick, inside the null space, the vector orthogonal to 𝟏.
      for (let k = 0; k < zeros && !f; k++) {
        const v = vectors[k];
        const mean = v.reduce((s, x) => s + x, 0) / n;
        const w = v.map((x) => x - mean);
        const norm = Math.sqrt(w.reduce((s, x) => s + x * x, 0));
        if (norm > 1e-9) f = w.map((x) => x / norm);
      }
    }
    // sign convention: the first node's entry is negative
    const first = f.findIndex((x) => Math.abs(x) > 1e-9);
    if (first >= 0 && f[first] > 0) f = f.map((x) => -x);

    const neg = [], pos = [];
    f.forEach((x, i) => (x < 0 ? neg : pos).push(i));
    const cut = E.filter(([a, b]) => (f[a] < 0) !== (f[b] < 0));

    // cheapest balanced split, by brute force over every subset of size ⌊n/2⌋
    const k = Math.floor(n / 2);
    let best = Infinity, bestCount = 0;
    for (let mask = 1; mask < (1 << n); mask++) {
      let size = 0;
      for (let i = 0; i < n; i++) if (mask & (1 << i)) size++;
      if (size !== k) continue;
      let c = 0;
      E.forEach(([a, b]) => { if (((mask >> a) & 1) !== ((mask >> b) & 1)) c++; });
      if (c < best) { best = c; bestCount = 1; } else if (c === best) bestCount++;
    }
    const balanced = neg.length === k || pos.length === k;
    return { n, m: E.length, E, values, vectors, zeros, f, ambiguous: zeros > 1, neg, pos, cut, k, best, balanced };
  }

  const RES = {};
  for (const key of Object.keys(GRAPHS)) RES[key] = analyse(GRAPHS[key]);

  let current = "cast";

  // ── formatting ──────────────────────────────────────────────────────────
  const SUB = ["₀", "₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"];
  const sub = (i) => String(i).split("").map((d) => SUB[+d]).join("");
  const fmt = (v) => (Math.abs(v) < 0.005 ? "0.00" : v.toFixed(2).replace("-", "−"));
  const fmtShort = (v) => (Math.abs(v) < 0.005 ? "0" : v.toFixed(2));
  const setOf = (G, ids) => "{" + ids.map((i) => G.names[i]).join(", ") + "}";
  const times = (c) => (c === 1 ? "once" : c === 2 ? "twice" : `${c} times`);
  const MONO = "'JetBrains Mono', monospace";

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w2-sp-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const G = GRAPHS[current], R = RES[current];
    const lam2 = R.values[1];
    const NEG = P.blue, POS = P.accent, GREY = P.muted;

    // ── left: the graph, coloured by the sign of the Fiedler entry ──
    g.append("text").attr("x", 20).attr("y", 24).attr("font-size", 12.5).attr("fill", P.muted)
      .text(G.head(R.n, R.m));

    // legend
    const legend = R.ambiguous
      ? [["disc", GREY, "no unique Fiedler vector"], ["dots", P.muted, "deleted edge"]]
      : [["disc", NEG, "entry < 0"], ["disc", POS, "entry > 0"], ["dash", P.red, "cut edge"]];
    let lx = 20;
    legend.forEach(([kind, col, label]) => {
      if (kind === "disc") {
        g.append("circle").attr("cx", lx + 6).attr("cy", 44).attr("r", 6).attr("fill", col);
        lx += 18;
      } else {
        g.append("line").attr("x1", lx).attr("y1", 44).attr("x2", lx + 22).attr("y2", 44)
          .attr("stroke", col).attr("stroke-width", kind === "dash" ? 2.6 : 1.4)
          .attr("stroke-dasharray", kind === "dash" ? "6 4" : "2 4").attr("opacity", kind === "dash" ? 0.9 : 0.6);
        lx += 28;
      }
      g.append("text").attr("x", lx).attr("y", 48.5).attr("font-size", 12.5).attr("fill", P.text).text(label);
      lx += label.length * 7.6 + 18;
    });

    const XY = (name) => G.pos[name];
    // deleted edges as ghosts, so the reader sees what changed
    G.ghost.forEach(([a, b]) => {
      const [x1, y1] = XY(a), [x2, y2] = XY(b);
      g.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
        .attr("stroke", P.muted).attr("stroke-width", 1.4).attr("stroke-dasharray", "2 4").attr("opacity", 0.6);
    });
    const isCut = (a, b) => R.cut.some(([p, q]) => (p === a && q === b) || (p === b && q === a));
    G.edges.forEach(([a, b]) => {
      const ia = G.names.indexOf(a), ib = G.names.indexOf(b);
      const hot = isCut(ia, ib);
      const [x1, y1] = XY(a), [x2, y2] = XY(b);
      g.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
        .attr("stroke", hot ? P.red : P.muted)
        .attr("stroke-width", hot ? 2.8 : 1.7)
        .attr("stroke-dasharray", hot ? "6 4" : null)
        .attr("opacity", hot ? 0.9 : 0.5);
    });
    if (G.bridge) {
      const [a, b] = G.bridge;
      const ia = G.names.indexOf(a), ib = G.names.indexOf(b);
      const [x1, y1] = XY(a), [x2, y2] = XY(b);
      g.append("text").attr("x", (x1 + x2) / 2).attr("y", (y1 + y2) / 2 - 13)
        .attr("text-anchor", "middle").attr("font-size", 12.5).attr("font-weight", 600)
        .attr("fill", isCut(ia, ib) ? P.red : P.muted).text("the bridge");
    }
    G.names.forEach((nm, i) => {
      const [cx, cy] = XY(nm);
      const v = R.f[i];
      const fill = R.ambiguous ? GREY : v < 0 ? NEG : POS;
      g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 25).attr("fill", fill)
        .attr("stroke", P.paper).attr("stroke-width", 1.5);
      g.append("text").attr("x", cx).attr("y", cy - 3.5).attr("text-anchor", "middle")
        .attr("font-size", 13).attr("font-weight", 700).attr("fill", "#fff").text(nm);
      g.append("text").attr("x", cx).attr("y", cy + 12.5).attr("text-anchor", "middle")
        .attr("font-family", MONO).attr("font-size", 12.5).attr("fill", "#fff").attr("opacity", 0.94)
        .text(fmt(v));
    });

    const cutWord = R.cut.length === 1 ? "edge" : "edges";
    g.append("text").attr("x", 20).attr("y", 366).attr("font-family", MONO)
      .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", R.ambiguous ? P.text : P.accentDark)
      .text(`cut = ${R.cut.length} ${cutWord}: ${setOf(G, R.neg)} | ${setOf(G, R.pos)}`);
    const note = R.ambiguous
      ? "the sign split just names the two pieces"
      : R.balanced && R.cut.length === R.best
        ? `no other ${R.k} | ${R.n - R.k} split cuts fewer edges`
        : `some ${R.k} | ${R.n - R.k} split cuts only ${R.best}`;
    g.append("text").attr("x", 20).attr("y", 386).attr("font-size", 12.5).attr("fill", P.muted).text(note);

    // ── divider ──
    g.append("line").attr("x1", 395).attr("y1", 12).attr("x2", 395).attr("y2", 398)
      .attr("stroke", P.border).attr("stroke-width", 1);

    // ── right: the spectrum ──
    const RX = 410;
    g.append("text").attr("x", RX).attr("y", 24).attr("font-size", 13.5).attr("font-weight", 700)
      .attr("fill", P.text).text("eigenvalues of L = D − A, ascending");

    const AX = 480, Y0 = 300, Y1 = 80, ROW = 26, LX = 520;
    const lmax = Math.max(1, R.values[R.values.length - 1]);
    const yOf = (v) => Y0 - (Math.max(0, v) / lmax) * (Y0 - Y1);
    g.append("line").attr("x1", AX).attr("y1", Y1 - 8).attr("x2", AX).attr("y2", Y0 + 6)
      .attr("stroke", P.muted).attr("stroke-width", 1.2).attr("opacity", 0.7);
    g.append("line").attr("x1", AX - 30).attr("y1", Y0).attr("x2", AX + 30).attr("y2", Y0)
      .attr("stroke", P.green).attr("stroke-width", 1).attr("stroke-dasharray", "3 3").attr("opacity", 0.8);
    g.append("text").attr("x", AX - 36).attr("y", Y0 + 4.5).attr("text-anchor", "end")
      .attr("font-family", MONO).attr("font-size", 12.5).attr("fill", P.green).text("0");

    // repeated eigenvalues sit side by side on the strip, one dot each
    const groups = [];
    R.values.forEach((v, i) => {
      const gp = groups.find((q) => Math.abs(q.v - v) < 1e-6);
      if (gp) gp.idx.push(i); else groups.push({ v, idx: [i] });
    });
    const dotX = {};
    groups.forEach((gp) => gp.idx.forEach((i, j) => { dotX[i] = AX + (j - (gp.idx.length - 1) / 2) * 10; }));

    R.values.forEach((v, i) => {
      const isZero = Math.abs(v) < ZERO;
      const isL2 = i === 1;
      const col = isZero ? P.green : isL2 ? P.accent : P.muted;
      const yt = yOf(v), yr = Y0 - i * ROW;
      g.append("line").attr("x1", dotX[i] + 5).attr("y1", yt).attr("x2", LX - 8).attr("y2", yr)
        .attr("stroke", col).attr("stroke-width", isZero || isL2 ? 1.2 : 0.9).attr("opacity", 0.55);
      g.append("circle").attr("cx", dotX[i]).attr("cy", yt).attr("r", isZero || isL2 ? 5 : 3.6)
        .attr("fill", col).attr("stroke", P.paper).attr("stroke-width", 1);
      g.append("text").attr("x", LX).attr("y", yr + 4.5).attr("font-family", MONO)
        .attr("font-size", 12.5).attr("font-weight", isZero || isL2 ? 700 : 400)
        .attr("fill", isZero ? P.green : isL2 ? P.accentDark : P.text)
        .text(`λ${sub(i + 1)} = ${fmt(v)}`);
      if (isL2) {
        g.append("text").attr("x", LX + 78).attr("y", yr + 4.5).attr("font-size", 12.5)
          .attr("font-weight", 600).attr("fill", P.accentDark)
          .text(R.ambiguous ? "← also 0" : "← Fiedler");
      }
    });

    g.append("text").attr("x", RX).attr("y", 334).attr("font-size", 12.5).attr("font-weight", 700)
      .attr("fill", P.green)
      .text(`eigenvalue 0 appears ${times(R.zeros)} → ${R.zeros} component${R.zeros === 1 ? "" : "s"}`);
    const reading = current === "barbell"
      ? `λ₂ = ${fmt(lam2)} — tiny: a bottleneck, the bridge`
      : current === "pieces"
        ? "λ₂ = 0 too — nothing to cut, already apart"
        : `λ₂ = ${fmt(lam2)} — small λ₂ means a cheap split`;
    g.append("text").attr("x", RX).attr("y", 356).attr("font-size", 12.5).attr("font-weight", 700)
      .attr("fill", P.accentDark).text(reading);
    const others = ["cast", "barbell", "pieces"].filter((k) => k !== current)
      .map((k) => `${k === "pieces" ? "two pieces" : k} ${fmtShort(RES[k].values[1])}`).join(" · ");
    g.append("text").attr("x", RX).attr("y", 376).attr("font-size", 12.5).attr("fill", P.muted)
      .text(`compare λ₂: ${others}`);

    // ── verdict ──
    const verdict = current === "barbell"
      ? `λ₂ is tiny and the sign change falls exactly on the bridge: ${R.neg.length} | ${R.pos.length} for the price of ${R.cut.length} edge`
      : current === "pieces"
        ? `eigenvalue 0 ${times(R.zeros)}, so ${R.zeros} pieces — the second zero-eigenvector is constant on each piece`
        : `thresholding the Fiedler vector at 0 cuts ${setOf(G, R.neg)} from ${setOf(G, R.pos)} — the cheapest ${R.k} | ${R.n - R.k} split`;
    g.append("text").attr("x", W / 2).attr("y", 426).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark).text(verdict);
  }

  for (const b of document.querySelectorAll("#w2-sp-widget [data-graph]")) {
    b.addEventListener("click", () => {
      current = b.dataset.graph;
      for (const o of document.querySelectorAll("#w2-sp-widget [data-graph]")) {
        o.classList.toggle("active", o === b);
      }
      render();
    });
  }
  U.onThemeChange(render);
  U.lazyBoot("w2-sp-svg", render);
})();
