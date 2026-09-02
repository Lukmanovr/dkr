/* Widget 2.x — Graphlets: the census that degree and clustering cannot see.
 *
 * An 11-node graph built so that two nodes, P and Q, have the SAME degree (3)
 * and the SAME local clustering coefficient (1/3) yet sit in different local
 * architectures: at P the third friend circles back through a square, at Q
 * the third friend leads away down a path. The right-hand panel is the
 * graphlet degree vector restricted to seven orbits on shapes of 3 and 4
 * nodes (wedge end/center, triangle, 4-path end/middle, 3-star center,
 * square), computed here by exhaustive enumeration of every 3- and 4-node
 * subset and classification of the INDUCED subgraph — the standard graphlet
 * convention (Przulj, 2007): a shape counts only when no extra edge sits among
 * its nodes.
 *
 * Nothing is asserted: degree, clustering, every orbit count and the verdict
 * line are recomputed from the edge list on every render, and clicking any
 * node re-runs the whole census with that node as P.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 460;

  // Hand-placed so the square at P is a literal square and Q's path is a staircase.
  const POS = [
    [236, 62],    // 1  the far end of Q's path
    [72, 168],    // 2  square corner above P
    [154, 168],   // 3  square corner, also the bridge to the right half
    [236, 128],   // 4  bridge node on Q's path
    [318, 168],   // 5  Q's third friend
    [72, 250],    // 6  P (default)
    [154, 250],   // 7  square corner, also in P's triangle
    [318, 250],   // 8  Q
    [113, 330],   // 9  apex of P's triangle
    [277, 330],   // 10 Q's triangle
    [359, 330],   // 11 Q's triangle
  ];
  const N = POS.length;
  const EDGES = [
    [6, 7], [6, 9], [6, 2], [7, 9], [2, 3], [3, 7],      // P's side: triangle 6-7-9, square 6-7-3-2
    [8, 10], [8, 11], [8, 5], [10, 11], [5, 4], [4, 1],  // Q's side: triangle 8-10-11, path 8-5-4-1
    [3, 4],                                              // the bridge
  ].map(([a, b]) => [a - 1, b - 1]);
  const DEFAULT_P = 5, Q = 7;                            // 0-based: node 6 and node 8
  let pIdx = DEFAULT_P;

  const A = Array.from({ length: N }, () => Array(N).fill(0));
  EDGES.forEach(([a, b]) => { A[a][b] = 1; A[b][a] = 1; });
  const DEG = A.map((r) => r.reduce((s, x) => s + x, 0));

  // Local clustering exactly as the lecture defines it (undefined below degree 2).
  function clustering(u) {
    if (DEG[u] < 2) return null;
    const nb = d3.range(N).filter((v) => A[u][v]);
    let t = 0;
    for (let i = 0; i < nb.length; i++) for (let j = i + 1; j < nb.length; j++) t += A[nb[i]][nb[j]];
    return t / ((DEG[u] * (DEG[u] - 1)) / 2);
  }

  // The seven orbits on display, each with a tiny glyph (● = the orbit position).
  const ORBITS = [
    { key: "wedgeEnd", label: "wedge, end", size: 3,
      pts: [[-12, 8], [0, -8], [12, 8]], edges: [[0, 1], [1, 2]], mark: 0 },
    { key: "wedgeMid", label: "wedge, center", size: 3,
      pts: [[-12, 8], [0, -8], [12, 8]], edges: [[0, 1], [1, 2]], mark: 1 },
    { key: "triangle", label: "triangle", size: 3,
      pts: [[-11, 8], [11, 8], [0, -9]], edges: [[0, 1], [1, 2], [0, 2]], mark: 2 },
    { key: "p4End", label: "4-path, end", size: 4,
      pts: [[-15, 7], [-5, -7], [5, 7], [15, -7]], edges: [[0, 1], [1, 2], [2, 3]], mark: 0 },
    { key: "p4Mid", label: "4-path, middle", size: 4,
      pts: [[-15, 7], [-5, -7], [5, 7], [15, -7]], edges: [[0, 1], [1, 2], [2, 3]], mark: 1 },
    { key: "starMid", label: "3-star, center", size: 4,
      pts: [[0, 0], [-13, 8], [13, 8], [0, -12]], edges: [[0, 1], [0, 2], [0, 3]], mark: 0 },
    { key: "square", label: "square", size: 4,
      pts: [[-9, -9], [9, -9], [9, 9], [-9, 9]], edges: [[0, 1], [1, 2], [2, 3], [3, 0]], mark: 0 },
  ];
  const KEYS = ORBITS.map((o) => o.key);

  // ── the census: every 3- and 4-node subset, classified as an induced subgraph ──
  function classify(S) {
    const d = S.map((u) => S.reduce((s, v) => s + A[u][v], 0));    // degrees inside S
    const m = d.reduce((s, x) => s + x, 0) / 2;
    const seen = new Set([S[0]]), q = [S[0]];                       // connected inside S?
    while (q.length) { const u = q.pop(); for (const v of S) if (A[u][v] && !seen.has(v)) { seen.add(v); q.push(v); } }
    if (seen.size !== S.length) return null;
    if (S.length === 3) {
      if (m === 2) return d.map((x) => (x === 2 ? "wedgeMid" : "wedgeEnd"));
      return d.map(() => "triangle");
    }
    const sig = d.slice().sort().join("");
    if (m === 3 && sig === "1122") return d.map((x) => (x === 1 ? "p4End" : "p4Mid"));
    if (m === 3 && sig === "1113") return d.map((x) => (x === 3 ? "starMid" : null));
    if (m === 4 && sig === "2222") return d.map(() => "square");
    return null;                                    // paw, diamond, K4: not among the seven
  }
  function subsets(k) {
    const out = [];
    (function rec(start, acc) {
      if (acc.length === k) { out.push(acc.slice()); return; }
      for (let i = start; i < N; i++) { acc.push(i); rec(i + 1, acc); acc.pop(); }
    })(0, []);
    return out;
  }
  // SUBS[u][key] = list of node sets in which u occupies that orbit; counts follow.
  const SUBS = Array.from({ length: N }, () => Object.fromEntries(KEYS.map((k) => [k, []])));
  for (const k of [3, 4]) for (const S of subsets(k)) {
    const roles = classify(S);
    if (!roles) continue;
    S.forEach((u, i) => { if (roles[i]) SUBS[u][roles[i]].push(S); });
  }
  const count = (u, key) => SUBS[u][key].length;

  // Walk a cycle (triangle or square) in drawing order for a filled polygon.
  function cycleOrder(S) {
    const ord = [S[0]];
    while (ord.length < S.length) {
      const last = ord[ord.length - 1];
      ord.push(S.find((v) => A[last][v] && !ord.includes(v)));
    }
    return ord;
  }
  const inducedEdges = (S) => EDGES.filter(([a, b]) => S.includes(a) && S.includes(b));

  let selected = "square";
  let hover = null;
  const live = {};                                  // handles for in-place hover updates

  function fmtC(c) { return c === null ? "—" : c.toFixed(2); }

  function paintShade(P) {
    const key = hover || selected;
    const g = live.shade;
    if (!g) return;
    g.selectAll("*").remove();
    [[pIdx, P.accent], [Q, P.blue]].forEach(([u, color]) => {
      for (const S of SUBS[u][key]) {
        if (key === "triangle" || key === "square") {
          g.append("polygon")
            .attr("points", cycleOrder(S).map((i) => POS[i].join(",")).join(" "))
            .attr("fill", color).attr("opacity", 0.14);
        }
        inducedEdges(S).forEach(([a, b]) => {
          g.append("line").attr("x1", POS[a][0]).attr("y1", POS[a][1])
            .attr("x2", POS[b][0]).attr("y2", POS[b][1])
            .attr("stroke", color).attr("stroke-width", 12).attr("stroke-linecap", "round")
            .attr("opacity", 0.2);
        });
      }
    });
    const orb = ORBITS.find((o) => o.key === key);
    live.note.text(`shaded — ${orb.label}: ${count(pIdx, key)} at P, ${count(Q, key)} at Q`);
    live.bands.forEach((b, i) => b.attr("opacity", ORBITS[i].key === key ? 0.11 : 0));
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w2-gl-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const MONO = "'JetBrains Mono', monospace";

    const dP = DEG[pIdx], dQ = DEG[Q];
    const cP = clustering(pIdx), cQ = clustering(Q);
    const sameDeg = dP === dQ;
    const sameC = fmtC(cP) === fmtC(cQ);
    const gdvP = KEYS.map((k) => count(pIdx, k)), gdvQ = KEYS.map((k) => count(Q, k));
    const nDiff = KEYS.filter((k, i) => gdvP[i] !== gdvQ[i]).length;

    // ── header: the two numbers that fail to separate P from Q ──
    const header = sameDeg && sameC
      ? `degree: P = Q = ${dP} · clustering: P = Q = ${fmtC(cP)}`
      : `degree: P = ${dP}, Q = ${dQ} · clustering: P = ${fmtC(cP)}, Q = ${fmtC(cQ)}`;
    g.append("text").attr("x", W / 2).attr("y", 26).attr("text-anchor", "middle")
      .attr("font-family", MONO).attr("font-size", 14).attr("font-weight", 700)
      .attr("fill", P.text).text(header);

    g.append("line").attr("x1", 392).attr("y1", 52).attr("x2", 392).attr("y2", 416)
      .attr("stroke", P.border).attr("stroke-width", 1);

    // ── the graph (left) ──
    live.shade = g.append("g");                      // shading sits under the edges
    EDGES.forEach(([a, b]) => {
      g.append("line").attr("x1", POS[a][0]).attr("y1", POS[a][1])
        .attr("x2", POS[b][0]).attr("y2", POS[b][1])
        .attr("stroke", P.muted).attr("stroke-width", 2).attr("opacity", 0.6);
    });
    POS.forEach(([x, y], i) => {
      const isP = i === pIdx, isQ = i === Q, hot = isP || isQ;
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", hot ? 17 : 13)
        .attr("fill", isP ? P.accent : isQ ? P.blue : P.muted)
        .attr("opacity", hot ? 1 : 0.8)
        .attr("stroke", P.paper).attr("stroke-width", hot ? 2.5 : 0)
        .style("cursor", isQ ? "default" : "pointer")
        .on("click", () => { if (i !== Q) { pIdx = i; render(); } });
      g.append("text").attr("x", x).attr("y", y + (hot ? 5.5 : 4.5)).attr("text-anchor", "middle")
        .attr("font-size", hot ? 15 : 12.5).attr("font-weight", 700).attr("fill", "#fff")
        .style("pointer-events", "none").text(isP ? "P" : isQ ? "Q" : String(i + 1));
    });
    live.note = g.append("text").attr("x", 206).attr("y", 374).attr("text-anchor", "middle")
      .attr("font-family", MONO).attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text);
    g.append("text").attr("x", 206).attr("y", 396).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text("click any node to make it P");
    g.append("text").attr("x", 206).attr("y", 414).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text("hover a row to shade its shapes");

    // ── the census (right): two bar groups over the orbit list ──
    const x0 = 404, x1 = 752, rowY = 84, rowH = 42;
    const legend = g.append("g");
    legend.append("circle").attr("cx", x0 + 8).attr("cy", 62).attr("r", 6.5).attr("fill", P.accent);
    legend.append("text").attr("x", x0 + 19).attr("y", 66.5).attr("font-size", 12.5)
      .attr("font-weight", 700).attr("fill", P.text).text("P");
    legend.append("circle").attr("cx", x0 + 44).attr("cy", 62).attr("r", 6.5).attr("fill", P.blue);
    legend.append("text").attr("x", x0 + 55).attr("y", 66.5).attr("font-size", 12.5)
      .attr("font-weight", 700).attr("fill", P.text).text("Q");
    legend.append("text").attr("x", x0 + 78).attr("y", 66.5).attr("font-size", 12.5)
      .attr("fill", P.muted).text("induced shapes with the node at ●");

    const maxCount = Math.max(1, ...gdvP, ...gdvQ);
    const unit = Math.min(28, 130 / maxCount);
    const barX = 570;
    live.bands = [];
    ORBITS.forEach((o, i) => {
      const y = rowY + i * rowH;
      const row = g.append("g").style("cursor", "pointer")
        .on("mouseenter", () => { hover = o.key; paintShade(P); })
        .on("mouseleave", () => { hover = null; paintShade(P); })
        .on("click", () => { selectOrbit(o.key); });
      const band = row.append("rect").attr("x", x0).attr("y", y).attr("width", x1 - x0).attr("height", rowH - 2)
        .attr("rx", 6).attr("fill", P.accent).attr("opacity", 0);
      live.bands.push(band);
      // an invisible hit area so hovering anywhere on the row counts
      row.append("rect").attr("x", x0).attr("y", y).attr("width", x1 - x0).attr("height", rowH - 2)
        .attr("fill", "transparent");
      // glyph
      const gx = x0 + 24, gy = y + 20;
      o.edges.forEach(([a, b]) => {
        row.append("line").attr("x1", gx + o.pts[a][0]).attr("y1", gy + o.pts[a][1])
          .attr("x2", gx + o.pts[b][0]).attr("y2", gy + o.pts[b][1])
          .attr("stroke", P.muted).attr("stroke-width", 1.5);
      });
      o.pts.forEach((p, k) => {
        const on = k === o.mark;
        row.append("circle").attr("cx", gx + p[0]).attr("cy", gy + p[1]).attr("r", on ? 4.2 : 3.2)
          .attr("fill", on ? P.text : P.paper).attr("stroke", on ? P.text : P.muted).attr("stroke-width", 1.4);
      });
      row.append("text").attr("x", x0 + 48).attr("y", y + 24.5).attr("font-size", 12.5)
        .attr("fill", P.text).text(o.label);
      // the two bars
      [[gdvP[i], P.accent, y + 7], [gdvQ[i], P.blue, y + 22]].forEach(([c, color, by]) => {
        if (c > 0) {
          row.append("rect").attr("x", barX).attr("y", by).attr("width", c * unit).attr("height", 12)
            .attr("rx", 2).attr("fill", color).attr("opacity", 0.85);
        }
        row.append("text").attr("x", barX + c * unit + 6).attr("y", by + 10.5)
          .attr("font-family", MONO).attr("font-size", 12.5).attr("font-weight", 700)
          .attr("fill", c > 0 ? P.text : P.muted).text(String(c));
      });
    });
    g.append("text").attr("x", x0).attr("y", 398).attr("font-size", 12.5).attr("fill", P.muted)
      .text("orbit = the ● position inside the shape");
    g.append("text").attr("x", x0).attr("y", 416).attr("font-size", 12.5).attr("fill", P.muted)
      .text("induced = no extra edge inside the shape");

    // ── the verdict band ──
    let verdict, tone;
    if (!sameDeg) { verdict = `different degree (P = ${dP}, Q = ${dQ}) — no graphlets needed`; tone = P.muted; }
    else if (!sameC) { verdict = "same degree, different clustering — already told apart"; tone = P.muted; }
    else if (nDiff > 0) { verdict = "identical by degree and clustering — told apart by graphlets"; tone = P.green; }
    else { verdict = "identical on all seven orbits — this census cannot separate them"; tone = P.accentDark; }
    const bw = Math.min(724, verdict.length * 8.2 + 44);
    g.append("rect").attr("x", W / 2 - bw / 2).attr("y", 428).attr("width", bw).attr("height", 30)
      .attr("rx", 15).attr("fill", tone);
    g.append("text").attr("x", W / 2).attr("y", 448).attr("text-anchor", "middle")
      .attr("font-size", 13.5).attr("font-weight", 700).attr("fill", "#fff").text(verdict);

    paintShade(P);
  }

  function selectOrbit(key) {
    selected = key; hover = null;
    for (const o of document.querySelectorAll("#w2-gl-widget [data-orbit]")) {
      o.classList.toggle("active", o.dataset.orbit === key);
    }
    render();
  }
  for (const b of document.querySelectorAll("#w2-gl-widget [data-orbit]")) {
    b.addEventListener("click", () => selectOrbit(b.dataset.orbit));
  }
  const reset = document.getElementById("w2-gl-reset");
  if (reset) reset.addEventListener("click", () => { pIdx = DEFAULT_P; render(); });

  U.onThemeChange(render);
  U.lazyBoot("w2-gl-svg", render);
})();
