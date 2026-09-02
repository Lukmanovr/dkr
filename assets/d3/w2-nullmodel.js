/* Widget 2.x — Modularity's null model, made literal (lecture §4.2).
 *
 * The cast graph (A..F; edges AB, AC, AD, BC, CE, CF, EF; degrees 3,2,4,1,2,2;
 * m = 7) is drawn with every edge cut in half: each node keeps d_u short
 * stubs, 14 loose ends in all. "Rewire once" draws ONE uniformly random
 * perfect matching of those 14 stubs and redraws the resulting multigraph
 * honestly — self-loops as loops around the node, parallel edges as bowed
 * duplicates — because that is exactly the configuration model modularity's
 * d_u d_v / 2m term approximates. "Rewire 100×" runs a hundred matchings and
 * only updates the tallies. The right panel tracks one pair (u, v): the
 * formula d_u d_v / 2m with the numbers substituted, the running empirical
 * mean number of u–v edges across every rewiring so far, the exact
 * stub-matching value d_u d_v / (2m − 1), and observed − expected, which is
 * the modularity-matrix entry B_Q[u, v] of @eq-w02-modmatrix.
 *
 * Randomness is a seeded mulberry32 stream, so every session sees the same
 * sequence of rewirings; every number on screen is computed here from the
 * edge list and the tallies, never typed in.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 484;
  const MONO = "'JetBrains Mono', monospace";

  const NAMES = ["A", "B", "C", "D", "E", "F"];
  const COL = { A: "#d9a62e", B: "#cf4a30", C: "#199473", D: "#7c5cd6", E: "#d1567e", F: "#0f8377" };
  const EDGES = [["A","B"],["A","C"],["A","D"],["B","C"],["C","E"],["C","F"],["E","F"]]
    .map(([a, b]) => [NAMES.indexOf(a), NAMES.indexOf(b)]);
  const n = 6, m = EDGES.length, twoM = 2 * m;
  const DEG = Array(n).fill(0);
  EDGES.forEach(([a, b]) => { DEG[a]++; DEG[b]++; });
  const POS = {                             // hand-placed; every stub direction distinct
    A: [150, 140], B: [292, 86], C: [284, 224], D: [86, 250], E: [150, 328], F: [294, 350],
  };
  const R = 17, STUB = 12;                  // node radius, stub length beyond the rim

  // ── the 14 stubs: one per edge end, pointing at the neighbour it came from ──
  const STUBS = [];                         // {node, dir:[dx,dy], tip:[x,y]}
  const NODE_STUBS = Array.from({ length: n }, () => []);
  const REAL = [];                          // the real graph as a stub matching
  EDGES.forEach(([a, b]) => {
    const ids = [[a, b], [b, a]].map(([u, v]) => {
      const [ux, uy] = POS[NAMES[u]], [vx, vy] = POS[NAMES[v]];
      const L = Math.hypot(vx - ux, vy - uy);
      const dir = [(vx - ux) / L, (vy - uy) / L];
      const s = { node: u, dir, tip: [ux + dir[0] * (R + STUB), uy + dir[1] * (R + STUB)] };
      STUBS.push(s); NODE_STUBS[u].push(STUBS.length - 1);
      return STUBS.length - 1;
    });
    REAL.push(ids);
  });

  function prng(seed) {                     // mulberry32, deterministic
    let x = 0x9e3779b9 ^ Math.imul(seed + 1, 0x85ebca6b);
    return () => {
      x |= 0; x = (x + 0x6D2B79F5) | 0;
      let t = Math.imul(x ^ (x >>> 15), 1 | x);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  // Seed chosen so the deterministic session is a representative sample, not
  // a fluke: the first draw shows both a self-loop and a repeat, and the A–C
  // mean after 100 draws (0.900) sits within one standard error of the exact
  // 12/13 = 0.923 (20,000 draws with this code give 0.922).
  const SEED = 42;

  // ── state ──
  let rnd, nRewire, tally, current, pair, pick;
  function reset() {
    rnd = prng(SEED);
    nRewire = 0;
    tally = Array.from({ length: n }, () => Array(n).fill(0));   // edges per pair, summed
    current = null;                          // null = the real graph
    pick = null;
  }
  pair = [0, 2];                             // A–C, the worked example
  reset();

  // One uniformly random perfect matching of the 14 stubs (Fisher–Yates, pair off).
  function randomMatching() {
    const s = STUBS.map((_, i) => i);
    for (let i = s.length - 1; i > 0; i--) {
      const j = Math.floor(rnd() * (i + 1));
      [s[i], s[j]] = [s[j], s[i]];
    }
    const M = [];
    for (let i = 0; i < s.length; i += 2) M.push([s[i], s[i + 1]]);
    return M;
  }

  // Count matrix of a matching: cnt[u][v] = cnt[v][u] = edges between u and v; cnt[u][u] = self-loops.
  function counts(M) {
    const c = Array.from({ length: n }, () => Array(n).fill(0));
    M.forEach(([s1, s2]) => {
      const u = STUBS[s1].node, v = STUBS[s2].node;
      if (u === v) c[u][u]++; else { c[u][v]++; c[v][u]++; }
    });
    return c;
  }

  function rewire(times) {
    for (let t = 0; t < times; t++) {
      current = randomMatching();
      const c = counts(current);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) tally[i][j] += c[i][j];
      nRewire++;
    }
  }

  const f3 = (v) => v.toFixed(3);
  const sgn = (v) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(3);
  const plural = (k, w) => `${k} ${w}${k === 1 ? "" : "s"}`;

  // ── drawing helpers ──
  // Spliced edge between two stubs of different nodes: a cubic whose handles
  // continue each stub's own direction, so the real graph's edges come out
  // straight and a rewired pair reads as two loose ends bent together.
  function splicePath(s1, s2, dup) {
    const a = STUBS[s1], b = STUBS[s2];
    const [px, py] = a.tip, [qx, qy] = b.tip;
    const d = Math.hypot(qx - px, qy - py);
    const k = Math.min(60, 0.3 * d + 10);
    let c1 = [px + a.dir[0] * k, py + a.dir[1] * k];
    let c2 = [qx + b.dir[0] * k, qy + b.dir[1] * k];
    if (dup > 0) {                           // bow duplicates sideways so repeats stay visible
      const nx = -(qy - py) / d, ny = (qx - px) / d;
      const off = 18 * Math.ceil(dup / 2) * (dup % 2 ? 1 : -1);
      c1 = [c1[0] + nx * off, c1[1] + ny * off];
      c2 = [c2[0] + nx * off, c2[1] + ny * off];
    }
    return `M ${px} ${py} C ${c1[0]} ${c1[1]} ${c2[0]} ${c2[1]} ${qx} ${qy}`;
  }

  // Self-loop: leave stub 1, round the node the short way at radius R+24(+10 per extra loop), return at stub 2.
  function loopPath(s1, s2, dup) {
    const a = STUBS[s1], b = STUBS[s2];
    const [ox, oy] = POS[NAMES[a.node]];
    const Ra = R + 24 + 10 * dup;
    const a1 = Math.atan2(a.dir[1], a.dir[0]), a2 = Math.atan2(b.dir[1], b.dir[0]);
    let delta = a2 - a1;
    while (delta > Math.PI) delta -= 2 * Math.PI;
    while (delta <= -Math.PI) delta += 2 * Math.PI;
    const sweep = delta > 0 ? 1 : 0;
    const q1 = [ox + a.dir[0] * Ra, oy + a.dir[1] * Ra];
    const q2 = [ox + b.dir[0] * Ra, oy + b.dir[1] * Ra];
    return `M ${a.tip[0]} ${a.tip[1]} L ${q1[0]} ${q1[1]} A ${Ra} ${Ra} 0 0 ${sweep} ${q2[0]} ${q2[1]} L ${b.tip[0]} ${b.tip[1]}`;
  }

  function drawGraph(g, P) {
    const M = current || REAL;
    const [tu, tv] = pair;
    const seen = {};                         // multiplicity so far per unordered pair
    M.forEach(([s1, s2]) => {
      const u = STUBS[s1].node, v = STUBS[s2].node;
      const key = Math.min(u, v) + "-" + Math.max(u, v);
      const dup = seen[key] || 0; seen[key] = dup + 1;
      const tracked = (u === tu && v === tv) || (u === tv && v === tu);
      g.append("path")
        .attr("d", u === v ? loopPath(s1, s2, dup) : splicePath(s1, s2, dup))
        .attr("fill", "none")
        .attr("stroke", tracked ? P.accent : P.muted)
        .attr("stroke-width", tracked ? 3.2 : 1.9)
        .attr("opacity", tracked ? 0.95 : 0.7)
        .attr("stroke-linecap", "round");
    });
    // stubs on top of the edges, in the owner's identity colour
    STUBS.forEach((s) => {
      const [ox, oy] = POS[NAMES[s.node]];
      g.append("line")
        .attr("x1", ox + s.dir[0] * (R + 1)).attr("y1", oy + s.dir[1] * (R + 1))
        .attr("x2", s.tip[0]).attr("y2", s.tip[1])
        .attr("stroke", COL[NAMES[s.node]]).attr("stroke-width", 3.4)
        .attr("stroke-linecap", "round");
    });
    NAMES.forEach((nm, i) => {
      const [x, y] = POS[nm];
      const isTracked = i === tu || i === tv;
      if (isTracked || i === pick) {
        g.append("circle").attr("cx", x).attr("cy", y).attr("r", R + 5)
          .attr("fill", "none").attr("stroke", P.accent).attr("stroke-width", 2.5)
          .attr("stroke-dasharray", i === pick && !isTracked ? "4 3" : null);
      }
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", R)
        .attr("fill", COL[nm]).style("cursor", "pointer")
        .on("click", () => {
          if (pick === null) { pick = i; }
          else if (pick === i) { pick = null; }
          else { pair = [pick, i]; pick = null; syncPills(); }
          render();
        });
      g.append("text").attr("x", x).attr("y", y + 5).attr("text-anchor", "middle")
        .attr("font-size", 14).attr("font-weight", 700).attr("fill", "#fff")
        .style("pointer-events", "none").text(nm);
      // degree tag beside each node: the number of stubs it owns
      const tagDir = nm === "B" || nm === "C" || nm === "F" ? [1, 0] : [-1, 0];
      g.append("text").attr("x", x + tagDir[0] * (R + 20)).attr("y", y + 4.5)
        .attr("text-anchor", tagDir[0] > 0 ? "start" : "end")
        .attr("font-family", MONO).attr("font-size", 12.5).attr("fill", P.muted)
        .text(`d=${DEG[i]}`);
    });
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w2-nm-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const [u, v] = pair;
    const Un = NAMES[u], Vn = NAMES[v];
    const formula = DEG[u] * DEG[v] / twoM;
    const exact = DEG[u] * DEG[v] / (twoM - 1);
    const observed = EDGES.some(([a, b]) => (a === u && b === v) || (a === v && b === u)) ? 1 : 0;
    const emp = nRewire ? tally[u][v] / nRewire : null;

    // ── left: the stub picture ──
    g.append("text").attr("x", 14).attr("y", 22).attr("font-size", 12.5).attr("fill", P.muted)
      .text(current ? `${twoM} stubs re-spliced at random: rewiring #${nRewire}`
                    : `cut all ${m} edges in half: ${twoM} loose stubs`);
    drawGraph(g, P);

    let status, loops = 0, repeats = 0, here = observed;
    if (current) {
      const c = counts(current);
      for (let i = 0; i < n; i++) {
        loops += c[i][i];
        for (let j = i + 1; j < n; j++) if (c[i][j] > 1) repeats += c[i][j] - 1;
      }
      here = c[u][v];
      status = `${plural(here, `${Un}–${Vn} edge`)} · ${plural(loops, "self-loop")} · ${plural(repeats, "repeat")}`;
    } else {
      status = `real graph: ${plural(observed, `${Un}–${Vn} edge`)} — press “rewire once”`;
    }
    g.append("text").attr("x", 14).attr("y", 396).attr("font-size", 13).attr("font-weight", 700)
      .attr("fill", current ? P.accentDark : P.text).text(status);
    g.append("text").attr("x", 14).attr("y", 418).attr("font-size", 12.5).attr("fill", P.muted)
      .text("Self-loops and repeats are allowed in the null —");
    g.append("text").attr("x", 14).attr("y", 436).attr("font-size", 12.5).attr("fill", P.muted)
      .text("that is why the formula is only ≈ dᵤdᵥ/2m.");

    g.append("line").attr("x1", 392).attr("y1", 10).attr("x2", 392).attr("y2", 446)
      .attr("stroke", P.border).attr("stroke-width", 1);

    // ── right: the tracked pair ──
    const x0 = 404, BW = 236;
    g.append("text").attr("x", x0).attr("y", 24).attr("font-size", 14).attr("font-weight", 700)
      .attr("fill", P.text).text(`tracked pair: ${Un}–${Vn}`);
    g.append("text").attr("x", x0).attr("y", 44).attr("font-family", MONO).attr("font-size", 12.5)
      .attr("fill", P.muted).text(`d(${Un}) = ${DEG[u]}, d(${Vn}) = ${DEG[v]}, 2m = ${twoM}`);

    const hi = Math.max(1.5, formula, exact, emp || 0);
    const sx = (val) => x0 + (val / hi) * BW;
    // formula bar
    // the exact stub-matching value as a dashed reference line, drawn first so
    // the bar labels (which carry a paper halo) sit on top of it
    g.append("line").attr("x1", sx(exact)).attr("y1", 78).attr("x2", sx(exact)).attr("y2", 156)
      .attr("stroke", P.text).attr("stroke-width", 1.3).attr("stroke-dasharray", "4 3").attr("opacity", 0.7);
    const halo = (t) => t.attr("stroke", P.paper).attr("stroke-width", 4).attr("paint-order", "stroke");
    halo(g.append("text").attr("x", x0).attr("y", 72).attr("font-family", MONO).attr("font-size", 12.5)
      .attr("fill", P.text)
      .text(`formula d(${Un})·d(${Vn})/2m = ${DEG[u]}·${DEG[v]}/${twoM} = ${f3(formula)}`));
    g.append("rect").attr("x", x0).attr("y", 80).attr("width", sx(formula) - x0).attr("height", 18)
      .attr("rx", 3).attr("fill", P.accent).attr("opacity", 0.8);
    // empirical bar
    const empLabel = emp === null
      ? "mean of 0 rewirings so far = —"
      : `mean of ${plural(nRewire, "rewiring")} so far = ${f3(emp)}`;
    halo(g.append("text").attr("x", x0).attr("y", 124).attr("font-family", MONO).attr("font-size", 12.5)
      .attr("fill", P.text).text(empLabel));
    g.append("rect").attr("x", x0).attr("y", 132).attr("width", emp === null ? 0 : Math.max(0, sx(emp) - x0))
      .attr("height", 18).attr("rx", 3).attr("fill", P.blue).attr("opacity", 0.8);
    // axis
    g.append("line").attr("x1", x0).attr("y1", 156).attr("x2", x0 + BW).attr("y2", 156)
      .attr("stroke", P.border).attr("stroke-width", 1);
    for (let t = 0; t <= hi + 1e-9; t += 0.5) {
      g.append("line").attr("x1", sx(t)).attr("y1", 156).attr("x2", sx(t)).attr("y2", 160)
        .attr("stroke", P.muted).attr("stroke-width", 1);
      g.append("text").attr("x", sx(t)).attr("y", 173).attr("text-anchor", "middle")
        .attr("font-family", MONO).attr("font-size", 12.5).attr("fill", P.muted)
        .text(t % 1 ? t.toFixed(1) : String(t));
    }
    g.append("text").attr("x", x0).attr("y", 196).attr("font-size", 12.5).attr("fill", P.text)
      .text(`dashed: exact value d(${Un})·d(${Vn})/(2m−1) = ${f3(exact)}`);
    g.append("text").attr("x", x0).attr("y", 214).attr("font-size", 12.5).attr("fill", P.muted)
      .text("the 2m version is the standard approximation");

    // observed vs expected: the modularity-matrix entry
    g.append("text").attr("x", x0).attr("y", 250).attr("font-size", 12.5).attr("fill", P.muted)
      .text("in the real graph");
    g.append("text").attr("x", x0).attr("y", 270).attr("font-family", MONO).attr("font-size", 12.5)
      .attr("fill", P.text).text(`observed  A[${Un},${Vn}] = ${observed}`);
    g.append("text").attr("x", x0).attr("y", 292).attr("font-family", MONO).attr("font-size", 13.5)
      .attr("font-weight", 700).attr("fill", P.accentDark)
      .text(`observed − expected = ${observed} − ${f3(formula)} = ${sgn(observed - formula)}`);
    g.append("text").attr("x", x0).attr("y", 312).attr("font-size", 12.5).attr("fill", P.text)
      .text(`= B_Q[${Un},${Vn}], the modularity-matrix entry`);
    g.append("text").attr("x", x0).attr("y", 330).attr("font-size", 12.5).attr("fill", P.muted)
      .text("B_Q = A − ddᵀ/2m: the null model as a matrix");

    g.append("text").attr("x", x0).attr("y", 372).attr("font-size", 12.5)
      .attr("fill", pick === null ? P.muted : P.accentDark)
      .text(pick === null ? "click two people to track another pair"
                          : `${NAMES[pick]} picked — now click its partner`);

    g.append("text").attr("x", W / 2).attr("y", 470).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
      .text("Hubs meet everyone by chance, so an edge between hubs earns a community almost no credit.");
  }

  // ── controls ──
  const PAIRS = { AC: [0, 2], DE: [3, 4] };
  function syncPills() {
    for (const b of document.querySelectorAll("#w2-nm-widget [data-pair]")) {
      const p = PAIRS[b.dataset.pair];
      b.classList.toggle("active", p && p[0] === pair[0] && p[1] === pair[1]);
    }
  }
  for (const b of document.querySelectorAll("#w2-nm-widget [data-pair]")) {
    b.addEventListener("click", () => { pair = PAIRS[b.dataset.pair].slice(); pick = null; syncPills(); render(); });
  }
  const once = document.getElementById("w2-nm-once");
  if (once) once.addEventListener("click", () => { rewire(1); render(); });
  const many = document.getElementById("w2-nm-many");
  if (many) many.addEventListener("click", () => { rewire(100); render(); });
  const rs = document.getElementById("w2-nm-reset");
  if (rs) rs.addEventListener("click", () => { reset(); render(); });

  U.onThemeChange(render);
  U.lazyBoot("w2-nm-svg", render);
})();
