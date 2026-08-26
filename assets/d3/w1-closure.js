/* Widget 1.9 — Triadic closure: friends of friends are friends.
 *
 * View "neighbourhood": two karate-club members with the SAME degree (6) whose
 * friend-pairs close at wildly different rates — member 3, buried inside Mr Hi's
 * knot, versus member 31, a broker whose friends include both faction leaders.
 * Every closed pair is a triangle; the ratio closed/total is exactly the local
 * clustering coefficient Week 2 formalizes.
 *
 * View "chance": the whole club against a random graph of the SAME size and
 * density — 34 nodes, 78 edges, uniform wiring, seeded so each reshuffle is
 * reproducible. (A degree-preserving null was tried and rejected: with a
 * degree-17 hub among 34 nodes the configuration model is degenerate and
 * produces MORE triangles than the real club, which would teach the opposite
 * of the truth. Separating hub effects from closure needs Week 2's tools.)
 *
 * All counts are computed here from the real edge list; nothing is asserted.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 440;

  const EDGES = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],
    [0,12],[0,13],[0,17],[0,19],[0,21],[0,31],[1,2],[1,3],[1,7],[1,13],[1,17],
    [1,19],[1,21],[1,30],[2,3],[2,7],[2,8],[2,9],[2,13],[2,27],[2,28],[2,32],
    [3,7],[3,12],[3,13],[4,6],[4,10],[5,6],[5,10],[5,16],[6,16],[8,30],[8,32],
    [8,33],[9,33],[13,33],[14,32],[14,33],[15,32],[15,33],[18,32],[18,33],
    [19,33],[20,32],[20,33],[22,32],[22,33],[23,25],[23,27],[23,29],[23,32],
    [23,33],[24,25],[24,27],[24,31],[25,31],[26,29],[26,33],[27,33],[28,31],
    [28,33],[29,32],[29,33],[30,32],[30,33],[31,32],[31,33],[32,33]];
  const N = 34;

  const EGOS = [
    { id: 3,  tag: "inside Mr Hi's knot" },
    { id: 31, tag: "a broker between the factions" },
  ];

  let view = "hood";
  let seed = 0;                       // which reshuffle of the random graph

  // ── graph helpers ────────────────────────────────────────────────────────
  const key = (a, b) => (a < b ? a + "," + b : b + "," + a);

  function adjOf(edges) {
    const S = new Set(edges.map(([a, b]) => key(a, b)));
    const nb = Array.from({ length: N }, () => []);
    edges.forEach(([a, b]) => { nb[a].push(b); nb[b].push(a); });
    return { has: (a, b) => S.has(key(a, b)), nb };
  }

  function triangles(edges) {
    const A = adjOf(edges);
    let t = 0;
    for (let u = 0; u < N; u++)
      for (const v of A.nb[u]) {
        if (v <= u) continue;
        for (const w of A.nb[v]) { if (w > v && A.has(u, w)) t++; }
      }
    return t;
  }

  // Global clustering (transitivity): 3 x triangles / paths of length two.
  function transitivity(edges) {
    const A = adjOf(edges);
    let wedges = 0;
    for (let u = 0; u < N; u++) { const d = A.nb[u].length; wedges += (d * (d - 1)) / 2; }
    return wedges ? (3 * triangles(edges)) / wedges : 0;
  }

  function prng(s) {                          // mulberry32, deterministic
    let x = 0x9e3779b9 ^ Math.imul(s + 1, 0x85ebca6b);
    return () => {
      x |= 0; x = (x + 0x6D2B79F5) | 0;
      let t = Math.imul(x ^ (x >>> 15), 1 | x);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Erdos-Renyi: the same n and the same m, wired uniformly at random.
  function erdosRenyi(m, s) {
    const rnd = prng(s * 7 + 3);
    const S = new Set(), E = [];
    while (E.length < m) {
      const a = Math.floor(rnd() * N), b = Math.floor(rnd() * N);
      if (a === b || S.has(key(a, b))) continue;
      S.add(key(a, b)); E.push([a, b]);
    }
    return E;
  }

  // Arc between two ring nodes, bowed AWAY from the panel centre so it never
  // collides with the ego node sitting there.
  function bow(p, q, cx, cy) {
    let mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
    let vx = mx - cx, vy = my - cy;
    const len = Math.hypot(vx, vy);
    if (len < 12) {                      // diametrically opposite: bow sideways
      vx = -(q[1] - p[1]); vy = q[0] - p[0];
    }
    const L = Math.hypot(vx, vy) || 1;
    const push = 34;
    return [mx + (vx / L) * push, my + (vy / L) * push];
  }

  // ── the two views ────────────────────────────────────────────────────────
  function drawHood(g, P, ego, cx, cy) {
    const A = adjOf(EDGES);
    const friends = A.nb[ego.id].slice().sort((a, b) => a - b);
    const R = 104;
    const pos = friends.map((f, i) => {
      const a = (2 * Math.PI * i) / friends.length - Math.PI / 2;
      return [cx + R * Math.cos(a), cy + R * Math.sin(a)];
    });

    let closed = 0;
    const total = (friends.length * (friends.length - 1)) / 2;
    for (let i = 0; i < friends.length; i++) {
      for (let j = i + 1; j < friends.length; j++) {
        const isClosed = A.has(friends[i], friends[j]);
        const c = bow(pos[i], pos[j], cx, cy);
        const arc = `M ${pos[i][0]} ${pos[i][1]} Q ${c[0]} ${c[1]} ${pos[j][0]} ${pos[j][1]}`;
        if (isClosed) {
          closed++;
          g.append("path")           // the triangle this closure creates
            .attr("d", `M ${cx} ${cy} L ${pos[i][0]} ${pos[i][1]} Q ${c[0]} ${c[1]} ${pos[j][0]} ${pos[j][1]} Z`)
            .attr("fill", P.accent).attr("opacity", 0.1).attr("stroke", "none");
          g.append("path").attr("d", arc).attr("fill", "none")
            .attr("stroke", P.accent).attr("stroke-width", 2).attr("opacity", 0.85);
        } else {
          g.append("path").attr("d", arc).attr("fill", "none")
            .attr("stroke", P.muted).attr("stroke-width", 1.2)
            .attr("stroke-dasharray", "3 4").attr("opacity", 0.5);
        }
      }
    }
    friends.forEach((f, i) => {
      g.append("line").attr("x1", cx).attr("y1", cy).attr("x2", pos[i][0]).attr("y2", pos[i][1])
        .attr("stroke", P.muted).attr("stroke-width", 1.6).attr("opacity", 0.65);
    });
    friends.forEach((f, i) => {
      g.append("circle").attr("cx", pos[i][0]).attr("cy", pos[i][1]).attr("r", 13)
        .attr("fill", P.blue);
      g.append("text").attr("x", pos[i][0]).attr("y", pos[i][1] + 4.5)
        .attr("text-anchor", "middle").attr("font-size", 12.5).attr("font-weight", 700)
        .attr("fill", "#fff").text(f);
    });
    g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 18).attr("fill", P.accent);
    g.append("text").attr("x", cx).attr("y", cy + 5.5).attr("text-anchor", "middle")
      .attr("font-size", 14).attr("font-weight", 700).attr("fill", "#fff").text(ego.id);

    const C = closed / total;
    g.append("text").attr("x", cx).attr("y", cy - 148).attr("text-anchor", "middle")
      .attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.text)
      .text(`member ${ego.id} · 6 friends`);
    g.append("text").attr("x", cx).attr("y", cy - 130).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text(ego.tag);
    g.append("text").attr("x", cx).attr("y", cy + 148).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.text)
      .text(`${closed} of ${total} friend-pairs know each other`);
    g.append("text").attr("x", cx).attr("y", cy + 170).attr("text-anchor", "middle")
      .attr("font-family", "'JetBrains Mono', monospace")
      .attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.accentDark)
      .text(`clustering C = ${C.toFixed(2)}`);
  }

  function drawWhole(g, P, edges, cx, cy, title, sub) {
    const R = 112;
    const pos = Array.from({ length: N }, (_, i) => {
      const a = (2 * Math.PI * i) / N - Math.PI / 2;
      return [cx + R * Math.cos(a), cy + R * Math.sin(a)];
    });
    const A = adjOf(edges);
    // Mark the edges that belong to at least one triangle. We highlight EDGES,
    // not filled triangle areas: a random triangle spans the whole circle while
    // a real one is local, so shaded areas would show the sparser graph as the
    // inkier one — the exact opposite of the counts underneath.
    const inTri = new Set();
    for (let u = 0; u < N; u++)
      for (const v of A.nb[u]) {
        if (v <= u) continue;
        for (const w of A.nb[v]) {
          if (w > v && A.has(u, w)) { inTri.add(key(u, v)); inTri.add(key(v, w)); inTri.add(key(u, w)); }
        }
      }
    edges.forEach(([a, b]) => {                       // plain edges underneath
      if (inTri.has(key(a, b))) return;
      g.append("line").attr("x1", pos[a][0]).attr("y1", pos[a][1])
        .attr("x2", pos[b][0]).attr("y2", pos[b][1])
        .attr("stroke", P.muted).attr("stroke-width", 0.8).attr("opacity", 0.4);
    });
    edges.forEach(([a, b]) => {                       // closure on top
      if (!inTri.has(key(a, b))) return;
      g.append("line").attr("x1", pos[a][0]).attr("y1", pos[a][1])
        .attr("x2", pos[b][0]).attr("y2", pos[b][1])
        .attr("stroke", P.accent).attr("stroke-width", 1.7).attr("opacity", 0.8);
    });
    pos.forEach((p) => {
      g.append("circle").attr("cx", p[0]).attr("cy", p[1]).attr("r", 4.6)
        .attr("fill", P.blue).attr("opacity", 0.9);
    });

    const t = triangles(edges), C = transitivity(edges);
    g.append("text").attr("x", cx).attr("y", cy - 150).attr("text-anchor", "middle")
      .attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.text).text(title);
    g.append("text").attr("x", cx).attr("y", cy - 132).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text(sub);
    g.append("text").attr("x", cx).attr("y", cy + 148).attr("text-anchor", "middle")
      .attr("font-family", "'JetBrains Mono', monospace")
      .attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.accentDark)
      .text(`${t} triangles`);
    g.append("text").attr("x", cx).attr("y", cy + 168).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.text).text(`clustering C = ${C.toFixed(3)}`);
    return t;
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w1-cl-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    if (view === "hood") {
      g.append("text").attr("x", W / 2).attr("y", 24).attr("text-anchor", "middle")
        .attr("font-size", 13).attr("fill", P.muted)
        .text("Same degree, opposite closure — solid arcs join friends who know each other, dashed join strangers");
      drawHood(g, P, EGOS[0], 192, 212);
      drawHood(g, P, EGOS[1], 568, 212);
      g.append("line").attr("x1", 380).attr("y1", 70).attr("x2", 380).attr("y2", 400)
        .attr("stroke", P.border).attr("stroke-width", 1);
      g.append("text").attr("x", W / 2).attr("y", 428).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
        .text("Degree counts your friends; clustering asks whether they are friends with each other.");
    } else {
      g.append("text").attr("x", W / 2).attr("y", 24).attr("text-anchor", "middle")
        .attr("font-size", 13).attr("fill", P.muted)
        .text("Same 34 members and same 78 friendships — highlighted edges belong to a triangle");
      const tReal = drawWhole(g, P, EDGES, 192, 206, "the real karate club", "friendships as recorded, 1977");
      const rnd = erdosRenyi(EDGES.length, seed);
      const tRnd = drawWhole(g, P, rnd, 568, 206, "random graph, same density",
        "uniform wiring #" + (seed + 1));
      g.append("line").attr("x1", 380).attr("y1", 60).attr("x2", 380).attr("y2", 396)
        .attr("stroke", P.border).attr("stroke-width", 1);
      const ratio = tRnd ? (tReal / tRnd).toFixed(1) : "∞";
      g.append("text").attr("x", W / 2).attr("y", 426).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
        .text(`Same size, same density, ${ratio}× fewer triangles — closure is not chance.`);
    }
  }

  // Controls that only mean anything in the comparison view are dimmed elsewhere.
  function syncControls() {
    const on = view === "chance";
    for (const el of document.querySelectorAll("#w1-cl-widget .cl-chance-only")) {
      el.style.opacity = on ? "1" : "0.35";
    }
  }
  for (const b of document.querySelectorAll("#w1-cl-widget [data-view]")) {
    b.addEventListener("click", () => {
      view = b.dataset.view;
      for (const o of document.querySelectorAll("#w1-cl-widget [data-view]")) {
        o.classList.toggle("active", o === b);
      }
      syncControls();
      render();
    });
  }
  const rs = document.getElementById("w1-cl-reshuffle");
  if (rs) {
    rs.addEventListener("click", () => {
      if (view !== "chance") return;
      seed = (seed + 1) % 6;
      render();
    });
  }
  U.onThemeChange(render);
  U.lazyBoot("w1-cl-svg", render);
})();
