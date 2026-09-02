/* Widget 2.x — The resolution limit, with the arithmetic done live.
 *
 * A ring of c cliques of k nodes each, neighbouring cliques joined by a single
 * edge: the least ambiguous community structure imaginable. Two partitions are
 * scored — the ground truth (every clique its own community) and "pairs"
 * (adjacent cliques fused two by two) — and both scores are computed from the
 * real edge list with the definition verbatim, Q = Σ_c [ e_c/m − (D_c/2m)² ].
 * No closed form is used anywhere in this file; the closed forms in the lecture
 * (Q_truth = 3/4 − 1/c and Q_pairs = 7/8 − 2/c for triangles) were checked
 * against this computation separately, and they agree to 1e-12.
 *
 * The teaching point: once the ring is long enough, merging perfect communities
 * RAISES modularity. On 24 triangles (m = 96) the ground truth scores 0.708 and
 * the merged pairs 0.792, so the maximum-modularity partition is not the truth.
 * The crossover c* is found from the two curves, never assumed.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 440;
  const CS = d3.range(4, 31, 2);                 // ring sizes the chart scans: 4, 6, …, 30
  const TOL = 1e-9;                              // ties at the crossover are exact up to rounding

  let c = 24, k = 3, part = "truth";

  // ── geometry first, then the graph is read off the drawing ───────────────
  // Clique i sits at angle θ_i on a ring of radius R; its k nodes sit on a small
  // circle around that centre with node 0 pointing radially outward. The single
  // edge to the next clique leaves from whichever node lies nearest to it, so
  // the edge list and the picture cannot disagree.
  const RING = { cx: 185, cy: 205, R: 120 };

  function layout(c, k) {
    const { cx, cy, R } = RING;
    const spacing = (2 * Math.PI * R) / c;      // arc length between clique centres
    const detailed = c <= 12;                    // draw every node, or one disc per clique
    const rc = detailed ? Math.min(24, 0.3 * spacing) : 0;                 // clique radius
    const rn = detailed ? Math.max(3.5, Math.min(6, 0.2 * rc)) : 0;        // node radius
    const rd = detailed ? 0 : Math.min(14, 0.36 * spacing);                // disc radius
    const centres = [], nodes = [];
    for (let i = 0; i < c; i++) {
      const th = (2 * Math.PI * i) / c - Math.PI / 2;
      const C = [cx + R * Math.cos(th), cy + R * Math.sin(th)];
      centres.push(C);
      for (let j = 0; j < k; j++) {
        const ph = th + (2 * Math.PI * j) / k;
        nodes.push([C[0] + rc * Math.cos(ph), C[1] + rc * Math.sin(ph)]);
      }
    }
    return { centres, nodes, spacing, detailed, rc, rn, rd, outer: detailed ? rc + rn : rd };
  }

  function buildGraph(c, k, L) {
    const edges = [];
    const nearest = (i, target) => {           // node of clique i closest to a point
      let best = 0, bd = Infinity;
      for (let j = 0; j < k; j++) {
        const p = L.nodes[i * k + j];
        const d = Math.hypot(p[0] - target[0], p[1] - target[1]);
        if (d < bd) { bd = d; best = j; }
      }
      return i * k + best;
    };
    for (let i = 0; i < c; i++) {
      for (let a = 0; a < k; a++) for (let b = a + 1; b < k; b++) edges.push([i * k + a, i * k + b]);
      const nx = (i + 1) % c;
      edges.push([nearest(i, L.centres[nx]), nearest(nx, L.centres[i])]);
    }
    return { n: c * k, edges };
  }

  // Q = Σ_c [ e_c/m − (D_c/2m)² ] over the communities of a partition — the
  // definition, computed from degrees and edge endpoints, nothing else.
  function modularity(g, comm) {
    const m = g.edges.length;
    const deg = Array(g.n).fill(0);
    g.edges.forEach(([a, b]) => { deg[a] += 1; deg[b] += 1; });
    const inside = new Map(), degSum = new Map();
    g.edges.forEach(([a, b]) => {
      if (comm[a] === comm[b]) inside.set(comm[a], (inside.get(comm[a]) || 0) + 1);
    });
    deg.forEach((d, u) => degSum.set(comm[u], (degSum.get(comm[u]) || 0) + d));
    let q = 0;
    degSum.forEach((D, id) => { q += (inside.get(id) || 0) / m - (D / (2 * m)) ** 2; });
    return q;
  }

  const truthOf = (n, k) => d3.range(n).map((u) => Math.floor(u / k));
  const pairsOf = (n, k) => d3.range(n).map((u) => Math.floor(Math.floor(u / k) / 2));

  // One row per ring size: both scores, from a freshly built graph each time.
  function scan(k) {
    return CS.map((cc) => {
      const g = buildGraph(cc, k, layout(cc, k));
      return { c: cc, m: g.edges.length,
               qt: modularity(g, truthOf(g.n, k)), qp: modularity(g, pairsOf(g.n, k)) };
    });
  }

  // Where the pairs curve overtakes the truth curve. A tie on the grid is
  // reported as that c; otherwise the crossing is interpolated between grid points.
  function crossover(rows) {
    if (rows[0].qp - rows[0].qt > TOL) return { c: rows[0].c, q: rows[0].qp, mode: "always" };
    for (let i = 1; i < rows.length; i++) {
      const a = rows[i - 1], b = rows[i];
      const da = a.qp - a.qt, db = b.qp - b.qt;
      if (da <= TOL && db > TOL) {
        if (Math.abs(da) <= TOL) return { c: a.c, q: a.qt, mode: "exact" };
        const t = -da / (db - da);
        return { c: a.c + t * (b.c - a.c), q: a.qt + t * (b.qt - a.qt), mode: "between" };
      }
    }
    return null;
  }

  // Colour count communities around a cycle so neighbours never share a colour:
  // prefer plain cycling, and only the closing community may deviate.
  function cycleColours(count, palette) {
    const out = [];
    for (let i = 0; i < count; i++) {
      let col = palette[i % palette.length];
      const clash = (x) => (i > 0 && x === out[i - 1]) || (i === count - 1 && x === out[0]);
      if (clash(col)) col = palette.find((p) => !clash(p)) || col;
      out.push(col);
    }
    return out;
  }

  const cliqueName = (k) => (k === 3 ? "triangles" : `cliques of ${k}`);

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w2-rl-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const MONO = "'JetBrains Mono', monospace";
    // Community identity on the ring, ordered so neighbouring hues never resemble
    // each other; the chart's two partitions use colours the ring does not:
    // neutral text for the ground truth, the accent for the merge.
    const IDENT = [P.purple, P.green, P.yellow, P.blue];
    const C_TRUTH = P.text, C_PAIRS = P.accent;

    // ── the numbers, all from the graph as drawn ──
    const L = layout(c, k);
    const G = buildGraph(c, k, L);
    const m = G.edges.length;
    const truth = truthOf(G.n, k), pairs = pairsOf(G.n, k);
    const qt = modularity(G, truth), qp = modularity(G, pairs);
    const comm = part === "truth" ? truth : pairs;
    const nComm = new Set(comm).size;
    const qShown = part === "truth" ? qt : qp;
    const eInt = G.edges.filter(([a, b]) => truth[a] === 0 && truth[b] === 0).length; // inside clique 0
    const scale = Math.sqrt(2 * m);
    const gap = qp - qt;
    const verdict = gap > TOL ? "merge" : gap < -TOL ? "separate" : "tie";
    const rows = scan(k);
    const cs = crossover(rows);

    // ── top line ──
    g.append("text").attr("x", W / 2).attr("y", 22).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(L.detailed
        ? `${c} ${cliqueName(k)} joined in a ring by single edges — every clique drawn in full`
        : `${c} ${cliqueName(k)} joined in a ring by single edges — each disc hides ${eInt} internal edges`);

    // ── the ring ──
    const colours = cycleColours(nComm, IDENT);
    const colourOf = (u) => colours[comm[u]];
    const ring = g.append("g");
    const { cx, cy, R } = RING;

    if (part === "pairs") {                     // a soft band along the ring hugs each merged pair
      const pad = Math.max(1, Math.min(3, (L.spacing - 2 * L.outer) / 2 - 2));
      for (let i = 0; i + 1 < c; i += 2) {
        const p = L.centres[i], q = L.centres[i + 1];
        ring.append("path")
          .attr("d", `M ${p[0]} ${p[1]} A ${R} ${R} 0 0 1 ${q[0]} ${q[1]}`)
          .attr("fill", "none").attr("stroke", colours[Math.floor(i / 2)])
          .attr("stroke-width", 2 * (L.outer + pad)).attr("stroke-linecap", "round")
          .attr("opacity", 0.22);
      }
    }

    const endpoint = (u) => (L.detailed ? L.nodes[u] : L.centres[truth[u]]);
    G.edges.forEach(([a, b]) => {
      const internal = truth[a] === truth[b];
      if (internal && !L.detailed) return;      // the disc stands in for these
      const same = comm[a] === comm[b];
      const [x1, y1] = endpoint(a), [x2, y2] = endpoint(b);
      const line = ring.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2);
      if (internal) {
        line.attr("stroke", colourOf(a)).attr("stroke-width", 1.3).attr("opacity", 0.75);
      } else if (same) {                        // a ring edge that now lies inside a community
        line.attr("stroke", colourOf(a)).attr("stroke-width", 2.2).attr("opacity", 0.9);
      } else {
        line.attr("stroke", P.muted).attr("stroke-width", 1.4).attr("opacity", 0.7)
          .attr("stroke-dasharray", "3,3");
      }
    });
    if (L.detailed) {
      L.nodes.forEach((p, u) => {
        ring.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", L.rn).attr("fill", colourOf(u));
      });
    } else {
      L.centres.forEach((p, i) => {
        ring.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", L.rd).attr("fill", colourOf(i * k));
      });
    }

    // the ring's own readout, in its empty middle
    ring.append("text").attr("x", cx).attr("y", cy - 30).attr("text-anchor", "middle")
      .attr("font-size", 14).attr("font-weight", 700).attr("fill", P.text)
      .text(`${c} ${cliqueName(k)}`);
    ring.append("text").attr("x", cx).attr("y", cy - 10).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text(`m = ${m} edges`);
    ring.append("text").attr("x", cx).attr("y", cy + 10).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(part === "truth" ? `${nComm} communities` : `${nComm} merged pairs`);
    ring.append("text").attr("x", cx).attr("y", cy + 32).attr("text-anchor", "middle")
      .attr("font-family", MONO).attr("font-size", 13.5).attr("font-weight", 700)
      .attr("fill", part === "truth" ? C_TRUTH : C_PAIRS).text(`Q = ${qShown.toFixed(3)}`);

    // ── the chart: both scores against c ──
    const x0 = 380, px0 = 414, px1 = 734, py0 = 82, py1 = 236;
    const all = [3, 4, 5].flatMap((kk) => scan(kk)).flatMap((r) => [r.qt, r.qp]);
    const lo = Math.floor(d3.min(all) * 20) / 20, hi = Math.ceil(d3.max(all) * 20) / 20;
    const x = d3.scaleLinear().domain([CS[0], CS[CS.length - 1]]).range([px0, px1]);
    const y = d3.scaleLinear().domain([lo, hi]).range([py1, py0]);
    const ch = g.append("g");

    [[C_TRUTH, "Q(truth): each clique its own community", 46],
     [C_PAIRS, "Q(pairs): adjacent cliques merged in twos", 64]].forEach(([col, label, yy]) => {
      ch.append("line").attr("x1", x0).attr("x2", x0 + 18).attr("y1", yy - 4).attr("y2", yy - 4)
        .attr("stroke", col).attr("stroke-width", 2.6);
      ch.append("text").attr("x", x0 + 24).attr("y", yy).attr("font-size", 12.5)
        .attr("font-weight", 600).attr("fill", col).text(label);
    });

    for (let v = Math.ceil(lo * 10) / 10; v <= hi + 1e-9; v = Math.round((v + 0.1) * 10) / 10) {
      ch.append("line").attr("x1", px0).attr("x2", px1).attr("y1", y(v)).attr("y2", y(v))
        .attr("stroke", P.border).attr("stroke-width", 1);
      ch.append("text").attr("x", px0 - 6).attr("y", y(v) + 4).attr("text-anchor", "end")
        .attr("font-size", 12.5).attr("fill", P.muted).text(v.toFixed(1));
    }
    ch.append("line").attr("x1", px0).attr("x2", px1).attr("y1", py1).attr("y2", py1)
      .attr("stroke", P.muted).attr("stroke-width", 1);
    const ticks = d3.range(4, 31, 4);
    if (!ticks.includes(c)) ticks.push(c);
    ticks.forEach((t) => {
      const now = t === c;
      ch.append("text").attr("x", x(t)).attr("y", py1 + 16).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", now ? 700 : 400)
        .attr("fill", now ? P.accentDark : P.muted).text(t);
    });
    ch.append("text").attr("x", (px0 + px1) / 2).attr("y", py1 + 34).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text("number of cliques c");

    const lineT = d3.line().x((d) => x(d.c)).y((d) => y(d.qt)).curve(d3.curveMonotoneX);
    const lineP = d3.line().x((d) => x(d.c)).y((d) => y(d.qp)).curve(d3.curveMonotoneX);
    ch.append("path").attr("d", lineT(rows)).attr("fill", "none").attr("stroke", C_TRUTH)
      .attr("stroke-width", part === "truth" ? 3 : 1.8).attr("opacity", part === "truth" ? 1 : 0.8);
    ch.append("path").attr("d", lineP(rows)).attr("fill", "none").attr("stroke", C_PAIRS)
      .attr("stroke-width", part === "pairs" ? 3 : 1.8).attr("opacity", part === "pairs" ? 1 : 0.8);

    // the crossover, read off the curves
    if (cs) {
      const label = cs.mode === "always" ? "pairs beat truth at every c shown"
        : `pairs beat truth for c > ${cs.mode === "exact" ? cs.c : cs.c.toFixed(1)}`;
      const est = label.length * 7.5;           // JetBrains Mono at 12.5 px
      // to the right of the marker when that fits the canvas; otherwise to the
      // left, but never so far left that it runs into the y-axis labels
      const fits = x(cs.c) + 5 + est <= W - 5;
      const lx = fits ? x(cs.c) + 5 : Math.max(x(cs.c) - 5, px0 + est);
      ch.append("line").attr("x1", x(cs.c)).attr("x2", x(cs.c)).attr("y1", y(cs.q)).attr("y2", py1)
        .attr("stroke", P.text).attr("stroke-width", 1).attr("stroke-dasharray", "1.5,3").attr("opacity", 0.6);
      ch.append("circle").attr("cx", x(cs.c)).attr("cy", y(cs.q)).attr("r", 4.2)
        .attr("fill", P.paper).attr("stroke", P.text).attr("stroke-width", 1.6);
      ch.append("text").attr("x", lx).attr("y", py1 - 6)
        .attr("text-anchor", fits ? "start" : "end").attr("font-size", 12.5)
        .attr("font-weight", 600).attr("fill", P.text).text(label);
    } else {
      ch.append("text").attr("x", px1).attr("y", py1 - 6).attr("text-anchor", "end")
        .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.text)
        .text("truth wins at every c shown");
    }

    // the current ring on the chart
    ch.append("line").attr("x1", x(c)).attr("x2", x(c)).attr("y1", py0).attr("y2", py1)
      .attr("stroke", P.muted).attr("stroke-width", 1).attr("stroke-dasharray", "4,3");
    [[qt, C_TRUTH], [qp, C_PAIRS]].forEach(([v, col]) => {
      ch.append("circle").attr("cx", x(c)).attr("cy", y(v)).attr("r", 4.6)
        .attr("fill", col).attr("stroke", P.paper).attr("stroke-width", 1.6);
    });

    // ── the readout ──
    const gapCol = verdict === "merge" ? P.accentDark : verdict === "separate" ? P.blueDark : P.muted;
    ch.append("text").attr("x", x0).attr("y", 300).attr("font-family", MONO)
      .attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.text)
      .text(`c = ${c}, k = ${k}, m = ${m}`);
    ch.append("text").attr("x", x0).attr("y", 320).attr("font-family", MONO)
      .attr("font-size", 13.5).attr("fill", P.text)
      .text(`Q(truth) = ${qt.toFixed(3)} · Q(pairs) = ${qp.toFixed(3)}`);
    ch.append("text").attr("x", x0).attr("y", 341).attr("font-family", MONO)
      .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", gapCol)
      .text(`Q(pairs) − Q(truth) = ${gap > TOL ? "+" : gap < -TOL ? "−" : ""}${Math.abs(gap).toFixed(3)}`);

    // ── the lecture's scale rule, evaluated on this ring ──
    g.append("text").attr("x", W / 2).attr("y", 372).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`cliques with fewer than about √(2m) = ${scale.toFixed(1)} internal edges are at risk` +
            (eInt < scale ? ` — here each has only ${eInt}` : ` — ${eInt} each is enough`));

    // ── the verdict ──
    const bandCol = verdict === "merge" ? P.accent : verdict === "separate" ? P.green : P.muted;
    const bandText = verdict === "merge"
      ? "modularity prefers to MERGE perfect communities — the resolution limit"
      : verdict === "separate"
        ? "modularity keeps every clique separate — below the resolution limit"
        : "a dead heat: merging neighbors leaves Q unchanged — exactly at the limit";
    g.append("rect").attr("x", 20).attr("y", 394).attr("width", W - 40).attr("height", 32)
      .attr("rx", 10).attr("fill", bandCol);
    g.append("text").attr("x", W / 2).attr("y", 415).attr("text-anchor", "middle")
      .attr("font-size", 13.5).attr("font-weight", 700).attr("fill", "#fff").text(bandText);
  }

  // ── controls ──
  const slider = document.getElementById("w2-rl-c");
  const cval = document.getElementById("w2-rl-cval");
  function readSlider() {
    let v = Math.round(Number(slider.value));
    if (v % 2) v += 1;                          // the ring only pairs up evenly
    c = Math.max(4, Math.min(30, v));
    if (cval) cval.textContent = c;
  }
  if (slider) {
    readSlider();
    slider.addEventListener("input", () => { readSlider(); render(); });
  }
  for (const b of document.querySelectorAll("#w2-rl-widget [data-k]")) {
    b.addEventListener("click", () => {
      k = Number(b.dataset.k);
      for (const o of document.querySelectorAll("#w2-rl-widget [data-k]")) o.classList.toggle("active", o === b);
      render();
    });
  }
  for (const b of document.querySelectorAll("#w2-rl-widget [data-part]")) {
    b.addEventListener("click", () => {
      part = b.dataset.part;
      for (const o of document.querySelectorAll("#w2-rl-widget [data-part]")) o.classList.toggle("active", o === b);
      render();
    });
  }
  U.onThemeChange(render);
  U.lazyBoot("w2-rl-svg", render);
})();
