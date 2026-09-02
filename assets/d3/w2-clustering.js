/* Widget 2.x — Clustering: one node's coefficient, and two ways to average it.
 *
 * View "node": the cast graph with one person in focus. Their friends are
 * ringed, every friend-pair that is itself an edge is drawn as a shaded
 * triangle, every pair that is not is a dashed arc, and the readout does the
 * arithmetic of the lecture's Definition (local clustering) and Proposition
 * (the diagonal of A³) on the real adjacency matrix:
 *     C_u = closed pairs / C(d_u, 2)     and     (A³)[u,u] = 2 × triangles at u.
 * The cube is computed by matrix multiplication and the triangles by search,
 * independently; the widget reports a disagreement rather than hide one.
 *
 * View "global": two purpose-built graphs on which the two "global clustering"
 * numbers — the average clustering C̄ (mean of the local values, one vote per
 * node) and the transitivity T (3 × triangles / wedges, one vote per wedge) —
 * disagree in OPPOSITE directions. A windmill (one hub, six triangles) has
 * C̄ ≈ 0.93 but T ≈ 0.23; a 5-clique with ten leaves has C̄ ≈ 0.13 but T = 0.40.
 * Every number on screen is computed from the edge lists in this file.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 460;
  const MONO = "'JetBrains Mono', monospace";

  // ── the cast graph ───────────────────────────────────────────────────────
  const NAMES = ["A", "B", "C", "D", "E", "F"];
  const COLOR = { A: "#d9a62e", B: "#cf4a30", C: "#199473", D: "#7c5cd6", E: "#d1567e", F: "#0f8377" };
  const CAST_EDGES = [["A","B"],["A","C"],["A","D"],["B","C"],["C","E"],["C","F"],["E","F"]]
    .map(([a, b]) => [NAMES.indexOf(a), NAMES.indexOf(b)]);
  const POS = {                              // hand-placed: C central, D off to the side
    A: [95, 130], B: [225, 65], C: [210, 205], D: [40, 235], E: [120, 310], F: [290, 320],
  };

  // ── the two purpose-built graphs of the second view ─────────────────────
  // Windmill: hub 0 joined to `blades` pendant pairs; each pair is an edge, so
  // every blade is a triangle through the hub.
  function windmill(blades) {
    const E = [];
    for (let k = 0; k < blades; k++) {
      const a = 1 + 2 * k, b = 2 + 2 * k;
      E.push([0, a], [0, b], [a, b]);
    }
    return { n: 1 + 2 * blades, edges: E };
  }
  // A k-clique on nodes 0..k-1, with `per` leaves hanging off every clique node.
  function cliqueWithLeaves(k, per) {
    const E = [];
    for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) E.push([i, j]);
    for (let i = 0; i < k; i++) for (let l = 0; l < per; l++) E.push([i, k + per * i + l]);
    return { n: k + k * per, edges: E };
  }

  // ── clustering arithmetic, shared by every view ─────────────────────────
  function stats(n, edges) {
    const A = Array.from({ length: n }, () => Array(n).fill(0));
    edges.forEach(([a, b]) => { A[a][b] = 1; A[b][a] = 1; });
    const nb = A.map((row) => row.flatMap((v, j) => (v ? [j] : [])));
    const deg = nb.map((x) => x.length);
    const pairs = deg.map((d) => (d * (d - 1)) / 2);      // C(d_u, 2) — the wedges at u
    const closed = nb.map((x) => {                         // edges among the neighbours
      let c = 0;
      for (let i = 0; i < x.length; i++) for (let j = i + 1; j < x.length; j++) if (A[x[i]][x[j]]) c++;
      return c;                                            // = triangles at u
    });
    const C = pairs.map((p, u) => (p ? closed[u] / p : 0)); // convention: 0 when d_u < 2
    const triangles = closed.reduce((s, c) => s + c, 0) / 3;
    const wedges = pairs.reduce((s, p) => s + p, 0);
    return {
      n, m: edges.length, A, nb, deg, pairs, closed, C, triangles, wedges,
      Cbar: C.reduce((s, c) => s + c, 0) / n,
      T: wedges ? (3 * triangles) / wedges : 0,
    };
  }
  // Every triangle as a triple u<v<w (for shading).
  function triangleList(S) {
    const out = [];
    for (let u = 0; u < S.n; u++)
      for (const v of S.nb[u]) {
        if (v <= u) continue;
        for (const w of S.nb[v]) if (w > v && S.A[u][w]) out.push([u, v, w]);
      }
    return out;
  }

  const CAST = stats(NAMES.length, CAST_EDGES);
  const A3 = U.matmul(U.matmul(CAST.A, CAST.A), CAST.A);   // the Proposition, by multiplication
  const WM = windmill(6), CL = cliqueWithLeaves(5, 2);
  const WIND = stats(WM.n, WM.edges), CLIQ = stats(CL.n, CL.edges);

  // Deterministic positions for the two built graphs (polar, hand-tuned).
  const polar = (cx, cy, r, deg) => [cx + r * Math.cos((deg * Math.PI) / 180), cy + r * Math.sin((deg * Math.PI) / 180)];
  function windmillPos(cx, cy) {
    const pos = [[cx, cy]];
    for (let k = 0; k < 6; k++) {
      const a = -90 + 60 * k;
      pos.push(polar(cx, cy, 90, a - 12), polar(cx, cy, 90, a + 12));
    }
    return pos;
  }
  function cliquePos(cx, cy) {
    const pos = [];
    for (let i = 0; i < 5; i++) pos.push(polar(cx, cy, 44, -90 + 72 * i));
    for (let i = 0; i < 5; i++) {
      const a = -90 + 72 * i;
      pos.push(polar(cx, cy, 92, a - 16), polar(cx, cy, 92, a + 16));
    }
    return pos;
  }

  let view = "node";
  let focus = 2;                              // C — the lecture's worked example

  // Control point for the arc p–q, pushed away from the ego at (ex, ey) so the
  // dashed "open pair" never runs through the person whose pairs we count.
  function bow(p, q, ex, ey) {
    const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
    let vx = mx - ex, vy = my - ey, len = Math.hypot(vx, vy);
    if (len < 1) { vx = -(q[1] - p[1]); vy = q[0] - p[0]; len = Math.hypot(vx, vy) || 1; }
    const push = Math.max(34, 2 * (52 - len));  // apex stays ≥ 52 px from the ego
    return [mx + (vx / len) * push, my + (vy / len) * push];
  }

  // Two bars, C̄ in the accent and T in teal — the same encoding in both views.
  function drawSummary(g, P, x0, y0, S, formulaC, formulaT) {
    const rows = [["C̄", S.Cbar, P.accent, P.accentDark, formulaC],
                  ["T", S.T, P.blue, P.blueDark, formulaT]];
    rows.forEach(([lab, val, col, dark, formula], i) => {
      const y = y0 + i * 26;
      g.append("text").attr("x", x0).attr("y", y + 11).attr("font-family", MONO)
        .attr("font-size", 13).attr("font-weight", 700).attr("fill", dark).text(lab);
      g.append("rect").attr("x", x0 + 30).attr("y", y).attr("width", Math.max(2, val * 120))
        .attr("height", 14).attr("rx", 3).attr("fill", col).attr("opacity", 0.8);
      g.append("text").attr("x", x0 + 158).attr("y", y + 11).attr("font-family", MONO)
        .attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text).text(val.toFixed(2));
      g.append("text").attr("x", x0 + 200).attr("y", y + 11).attr("font-size", 12.5)
        .attr("fill", P.muted).text(formula);
    });
  }

  // ── view 1: one node's coefficient on the cast graph ────────────────────
  function drawNodeView(g, P) {
    const S = CAST, u = focus, nm = NAMES[u], nbs = S.nb[u];
    const [ex, ey] = POS[nm];

    g.append("text").attr("x", 20).attr("y", 24).attr("font-size", 12.5).attr("fill", P.muted)
      .text("click any person to focus on them — their friends are ringed, closed friend-pairs shaded");

    // structure underneath
    CAST_EDGES.forEach(([a, b]) => {
      const touches = a === u || b === u;
      g.append("line").attr("x1", POS[NAMES[a]][0]).attr("y1", POS[NAMES[a]][1])
        .attr("x2", POS[NAMES[b]][0]).attr("y2", POS[NAMES[b]][1])
        .attr("stroke", P.muted).attr("stroke-width", touches ? 2.4 : 1.6)
        .attr("opacity", touches ? 0.75 : 0.4);
    });
    // the friend-pairs: closed ones as shaded triangles, open ones as dashed arcs
    const pairs = [];
    for (let i = 0; i < nbs.length; i++)
      for (let j = i + 1; j < nbs.length; j++)
        pairs.push({ v: nbs[i], w: nbs[j], closed: S.A[nbs[i]][nbs[j]] === 1 });
    pairs.forEach((pr) => {
      const p = POS[NAMES[pr.v]], q = POS[NAMES[pr.w]];
      if (pr.closed) {
        g.append("path").attr("d", `M ${ex} ${ey} L ${p[0]} ${p[1]} L ${q[0]} ${q[1]} Z`)
          .attr("fill", P.accent).attr("opacity", 0.12).attr("stroke", "none");
        g.append("line").attr("x1", p[0]).attr("y1", p[1]).attr("x2", q[0]).attr("y2", q[1])
          .attr("stroke", P.accent).attr("stroke-width", 3.4).attr("opacity", 0.95);
      } else {
        const c = bow(p, q, ex, ey);
        g.append("path").attr("d", `M ${p[0]} ${p[1]} Q ${c[0]} ${c[1]} ${q[0]} ${q[1]}`)
          .attr("fill", "none").attr("stroke", P.muted).attr("stroke-width", 1.4)
          .attr("stroke-dasharray", "3 4").attr("opacity", 0.7);
      }
    });
    // people
    NAMES.forEach((name, i) => {
      const [x, y] = POS[name];
      const isEgo = i === u, isNb = S.A[u][i] === 1;
      if (isNb) g.append("circle").attr("cx", x).attr("cy", y).attr("r", 19.5)
        .attr("fill", "none").attr("stroke", P.accent).attr("stroke-width", 3);
      if (isEgo) g.append("circle").attr("cx", x).attr("cy", y).attr("r", 24)
        .attr("fill", "none").attr("stroke", P.text).attr("stroke-width", 1.6).attr("opacity", 0.7);
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", isEgo ? 19 : 15)
        .attr("fill", COLOR[name]).attr("data-node", name)
        .style("cursor", "pointer").on("click", () => { focus = i; render(); });
      g.append("text").attr("x", x).attr("y", y + (isEgo ? 5.5 : 4.5)).attr("text-anchor", "middle")
        .attr("font-size", isEgo ? 15 : 13).attr("font-weight", 700).attr("fill", "#fff")
        .style("pointer-events", "none").text(name);
    });

    // ── readout for the focused person ──
    const x0 = 395, d = S.deg[u], np = S.pairs[u], nc = S.closed[u];
    g.append("text").attr("x", x0).attr("y", 58).attr("font-size", 13).attr("font-weight", 700)
      .attr("fill", P.text)
      .text(`focus on ${nm} · friend${d === 1 ? "" : "s"} ${nbs.map((v) => NAMES[v]).join(", ")}`);
    g.append("text").attr("x", x0).attr("y", 84).attr("font-family", MONO).attr("font-size", 13)
      .attr("fill", P.text).text(`d_${nm} = ${d} → C(${d},2) = ${np} pair${np === 1 ? "" : "s"}`);
    if (np) {
      pairs.forEach((pr, i) => {
        const cx = x0 + i * 50;
        g.append("rect").attr("x", cx).attr("y", 100).attr("width", 44).attr("height", 22).attr("rx", 5)
          .attr("fill", pr.closed ? P.accent : "none")
          .attr("stroke", pr.closed ? "none" : P.muted).attr("stroke-width", 1.2)
          .attr("stroke-dasharray", pr.closed ? null : "3 3");
        g.append("text").attr("x", cx + 22).attr("y", 115.5).attr("text-anchor", "middle")
          .attr("font-family", MONO).attr("font-size", 12.5).attr("font-weight", 700)
          .attr("fill", pr.closed ? "#fff" : P.muted).text(NAMES[pr.v] + NAMES[pr.w]);
      });
    } else {
      g.append("text").attr("x", x0).attr("y", 115.5).attr("font-size", 12.5).attr("fill", P.muted)
        .text("no pair of friends to check");
    }
    if (d >= 2) {
      g.append("text").attr("x", x0).attr("y", 146).attr("font-family", MONO).attr("font-size", 13)
        .attr("fill", P.text)
        .text(`${nc} of ${np} pair${np === 1 ? "" : "s"} ${nc === 1 ? "is an edge" : "are edges"}`);
      g.append("text").attr("x", x0).attr("y", 174).attr("font-family", MONO).attr("font-size", 15)
        .attr("font-weight", 700).attr("fill", P.accentDark)
        .text(`C_${nm} = ${nc}/${np} = ${(nc / np).toFixed(2)}`);
    } else {
      g.append("text").attr("x", x0).attr("y", 146).attr("font-family", MONO).attr("font-size", 13)
        .attr("fill", P.text).text(`C_${nm} undefined — fewer than two friends,`);
      g.append("text").attr("x", x0).attr("y", 174).attr("font-family", MONO).attr("font-size", 15)
        .attr("font-weight", 700).attr("fill", P.accentDark).text("reported as 0 by convention");
    }
    const agree = A3[u][u] === 2 * nc;       // the Proposition, checked live
    if (!agree) console.error(`w2-clustering: (A³)[${nm},${nm}] = ${A3[u][u]} but ${nc} triangles found`);
    g.append("text").attr("x", x0).attr("y", 204).attr("font-family", MONO).attr("font-size", 13)
      .attr("fill", agree ? P.text : P.red)
      .text(`(A³)[${nm},${nm}] = ${A3[u][u]} ${agree ? "=" : "≠"} 2 × ${nc} triangle${nc === 1 ? "" : "s"}`);
    g.append("text").attr("x", x0).attr("y", 222).attr("font-size", 12.5).attr("fill", P.muted)
      .text(nc ? `closed 3-walks at ${nm} — every triangle twice` : `no closed 3-walk starts at ${nm}`);

    // ── the whole graph, both summaries ──
    g.append("line").attr("x1", x0).attr("y1", 250).attr("x2", 740).attr("y2", 250)
      .attr("stroke", P.border).attr("stroke-width", 1);
    g.append("text").attr("x", x0).attr("y", 280).attr("font-size", 12.5).attr("fill", P.muted)
      .text("the whole graph, summarized both ways:");
    drawSummary(g, P, x0, 292, S, "mean of six, D as 0", `3·${S.triangles} / ${S.wedges} wedges`);
    g.append("text").attr("x", x0).attr("y", 356).attr("font-size", 12.5).attr("fill", P.muted)
      .text("close here — but they need not be: next view");

    // ── all six local values, same recipe ──
    g.append("text").attr("x", 20).attr("y", 372).attr("font-size", 12.5).attr("fill", P.muted)
      .text("C_u for all six — click a disc to focus:");
    NAMES.forEach((name, i) => {
      const cx = 42 + i * 54, cy = 397;
      if (i === u) g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 16)
        .attr("fill", "none").attr("stroke", P.text).attr("stroke-width", 1.6).attr("opacity", 0.7);
      g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 12).attr("fill", COLOR[name])
        .attr("data-node", name).style("cursor", "pointer").on("click", () => { focus = i; render(); });
      g.append("text").attr("x", cx).attr("y", cy + 4.5).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", "#fff")
        .style("pointer-events", "none").text(name);
      const defined = S.deg[i] >= 2;
      g.append("text").attr("x", cx).attr("y", 424).attr("text-anchor", "middle")
        .attr("font-family", MONO).attr("font-size", 12.5).attr("font-weight", i === u ? 700 : 400)
        .attr("fill", defined ? P.text : P.muted).text(defined ? S.C[i].toFixed(2) : "0");
    });

    g.append("text").attr("x", W / 2).attr("y", 450).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
      .text("C_u = closed pairs / all pairs, and the diagonal of A³ counts the closed ones for you.");
  }

  // ── view 2: the two global measures, disagreeing both ways ───────────────
  function drawGlobalPanel(g, P, S, pos, cx, title, hubIndex, classLines, formulaC, formulaT) {
    const x0 = cx - 168;
    g.append("text").attr("x", cx).attr("y", 52).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text).text(title);
    g.append("text").attr("x", cx).attr("y", 70).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`n = ${S.n}, m = ${S.m}, ${S.triangles} triangles, ${S.wedges} wedges`);

    triangleList(S).forEach(([a, b, c]) => {
      g.append("path")
        .attr("d", `M ${pos[a][0]} ${pos[a][1]} L ${pos[b][0]} ${pos[b][1]} L ${pos[c][0]} ${pos[c][1]} Z`)
        .attr("fill", P.accent).attr("opacity", 0.1).attr("stroke", "none");
    });
    for (let a = 0; a < S.n; a++) for (const b of S.nb[a]) {
      if (b <= a) continue;
      g.append("line").attr("x1", pos[a][0]).attr("y1", pos[a][1]).attr("x2", pos[b][0]).attr("y2", pos[b][1])
        .attr("stroke", P.muted).attr("stroke-width", 1.5).attr("opacity", 0.6);
    }
    const fill = d3.interpolateRgb(P.paper, P.accent);
    for (let a = 0; a < S.n; a++) {
      const hub = a === hubIndex;
      g.append("circle").attr("cx", pos[a][0]).attr("cy", pos[a][1]).attr("r", hub ? 15 : S.deg[a] >= 2 ? 10.5 : 9)
        .attr("fill", fill(S.C[a])).attr("stroke", P.muted).attr("stroke-width", 1.2);
      if (hub) g.append("text").attr("x", pos[a][0]).attr("y", pos[a][1] + 4.5).attr("text-anchor", "middle")
        .attr("font-size", 13).attr("font-weight", 700).attr("fill", S.C[a] < 0.5 ? P.text : "#fff").text("H");
    }
    classLines.forEach((line, i) => {
      g.append("text").attr("x", x0).attr("y", 306 + i * 18).attr("font-family", MONO)
        .attr("font-size", 12.5).attr("fill", P.text).text(line);
    });
    drawSummary(g, P, x0, 340, S, formulaC, formulaT);
    const cbarWins = S.Cbar > S.T;
    const ratio = (Math.max(S.Cbar, S.T) / Math.min(S.Cbar, S.T)).toFixed(1);
    g.append("text").attr("x", cx).attr("y", 404).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
      .text(`C̄ says ${cbarWins ? "tight-knit" : "sparse"}, T says ${cbarWins ? "sparse" : "tight-knit"} (${ratio}×)`);
  }

  function drawGlobalView(g, P) {
    g.append("text").attr("x", W / 2).attr("y", 24).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("node fill shows that node's C_u — hollow for 0, full for 1 — and every triangle is shaded");

    const S1 = WIND, blade = S1.nb[0][0];                 // any blade node
    drawGlobalPanel(g, P, S1, windmillPos(190, 182), 190, "windmill: hub H plus six triangles", 0,
      [`${S1.n - 1} blade nodes: C = ${S1.closed[blade]}/${S1.pairs[blade]} = ${S1.C[blade].toFixed(2)} each`,
       `hub H: C = ${S1.closed[0]}/${S1.pairs[0]} = ${S1.C[0].toFixed(2)}`],
      `mean of ${S1.n} C_u`, `3·${S1.triangles} / ${S1.wedges} wedges`);

    const S2 = CLIQ, leaves = S2.deg.filter((d) => d < 2).length, k = S2.n - leaves;
    drawGlobalPanel(g, P, S2, cliquePos(570, 182), 570, "clique with leaves: K₅ plus ten leaves", -1,
      [`${leaves} leaves: C = 0 by convention (d = 1)`,
       `${k} clique nodes: C = ${S2.closed[0]}/${S2.pairs[0]} = ${S2.C[0].toFixed(2)} each`],
      `mean of ${S2.n} C_u`, `3·${S2.triangles} / ${S2.wedges} wedges`);

    g.append("line").attr("x1", 380).attr("y1", 44).attr("x2", 380).attr("y2", 416)
      .attr("stroke", P.border).attr("stroke-width", 1);
    g.append("text").attr("x", W / 2).attr("y", 450).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
      .text("Same word, opposite verdicts — a clustering number means nothing until you say which one.");
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w2-cc-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    if (view === "node") drawNodeView(g, P); else drawGlobalView(g, P);
  }

  for (const b of document.querySelectorAll("#w2-cc-widget [data-view]")) {
    b.addEventListener("click", () => {
      view = b.dataset.view;
      for (const o of document.querySelectorAll("#w2-cc-widget [data-view]")) {
        o.classList.toggle("active", o === b);
      }
      render();
    });
  }
  U.onThemeChange(render);
  U.lazyBoot("w2-cc-svg", render);
})();
