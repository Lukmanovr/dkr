/* Widget 2.x — Katz centrality: one dial from degree to the leading eigenvector.
 *
 * x_Katz = sum_{k>=1} beta^k A^k 1. Small beta counts one-step walks only, which
 * is degree; as beta approaches 1/lambda_max the long walks dominate and the
 * leading eigenvector takes over. The graph here is built so the two extremes
 * DISAGREE: a hub S with five pendant leaves (degree 6, the most edges) joined
 * by one bridge S–K1 to a 4-clique (degree 4 at K1, but the clique's spectral
 * radius is higher). By degree S wins; by eigenvector K1 wins; Katz crosses
 * over at a tipping point beta* that the widget locates by bisection.
 *
 * Everything shown is computed here from the 12-edge list: lambda_max and the
 * eigenvector by the shared Jacobi routine, Katz scores by solving
 * (I − beta A) x = beta A 1 exactly (the truncated series drifts near
 * 1/lambda_max), and beta* by bisection on x_K1 − x_S.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 450;

  const NAMES = ["S", "L1", "L2", "L3", "L4", "L5", "K1", "K2", "K3", "K4"];
  const n = NAMES.length;
  const EDGES = [["S", "L1"], ["S", "L2"], ["S", "L3"], ["S", "L4"], ["S", "L5"], ["S", "K1"],
                 ["K1", "K2"], ["K1", "K3"], ["K1", "K4"], ["K2", "K3"], ["K2", "K4"], ["K3", "K4"]]
    .map(([a, b]) => [NAMES.indexOf(a), NAMES.indexOf(b)]);
  const S = 0, K1 = 6;
  const hubSide = (i) => i <= 5;                 // S and its leaves vs the clique

  // Hand-placed, deterministic layout: hub and leaves left, clique right.
  const CY = 166;
  const POS = { S: [140, CY], K1: [240, CY], K2: [305, CY - 65], K3: [305, CY + 65], K4: [370, CY] };
  [244, 212, 180, 148, 116].forEach((deg, k) => {   // leaves on an arc around S
    const a = (deg * Math.PI) / 180;
    POS["L" + (k + 1)] = [POS.S[0] + 100 * Math.cos(a), POS.S[1] + 100 * Math.sin(a)];
  });

  const A = Array.from({ length: n }, () => Array(n).fill(0));
  EDGES.forEach(([a, b]) => { A[a][b] = 1; A[b][a] = 1; });
  const deg = U.degrees(A);

  // Leading eigenpair (Jacobi; sign fixed so the Perron vector is positive).
  const eig = U.symEig(A);
  const lam = eig.values[n - 1];
  let ev = eig.vectors[n - 1];
  if (ev[S] < 0) ev = ev.map((v) => -v);
  const betaMax = 1 / lam;

  // Gaussian elimination with partial pivoting (n = 10, exact enough).
  function solve(M0, b0) {
    const M = M0.map((r, i) => r.concat([b0[i]]));
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = 0; r < n; r++) {
        if (r === c) continue;
        const f = M[r][c] / M[c][c];
        if (f === 0) continue;
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    return M.map((r, i) => r[n] / r[i]);
  }
  // Katz: (I − βA) x = βA1, i.e. x = ((I − βA)^{-1} − I) 1.
  function katz(beta) {
    const M = A.map((row, i) => row.map((a, j) => (i === j ? 1 : 0) - beta * a));
    return solve(M, deg.map((d) => beta * d));
  }

  // The tipping point: the β at which K1's score passes S's.
  const gap = (beta) => { const x = katz(beta); return x[K1] - x[S]; };
  let bstar = NaN;
  {
    let lo = 0.001 * betaMax, hi = 0.999 * betaMax;
    if (gap(lo) < 0 && gap(hi) > 0) {
      for (let it = 0; it < 60; it++) { const mid = (lo + hi) / 2; if (gap(mid) < 0) lo = mid; else hi = mid; }
      bstar = (lo + hi) / 2;
    }
  }

  const PRESETS = { low: 5, mid: 50, high: 95 };  // percent of 1/λmax
  let pct = PRESETS.mid;
  const beta = () => (pct / 100) * betaMax;
  const f3 = (v) => v.toFixed(3);

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w2-kz-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const MONO = "'JetBrains Mono', monospace";

    const b = beta();
    const x = katz(b);
    const xmax = Math.max(...x);
    const rel = x.map((v) => v / xmax);
    const degMax = Math.max(...deg), evMax = Math.max(...ev);
    const top = x.indexOf(xmax);
    const past = Number.isFinite(bstar) && b > bstar;
    const side = (i) => (hubSide(i) ? P.accent : P.blue);
    const radius = (i) => 9 + 13 * rel[i];

    g.append("text").attr("x", W / 2).attr("y", 20).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("one dial: β near 0 ranks like degree, β near 1/λmax ranks like the leading eigenvector");

    // ── the graph ──
    g.append("text").attr("x", 110).attr("y", 46).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
      .text(`popular hub S (degree ${deg[S]})`);
    g.append("text").attr("x", 305).attr("y", 46).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.blueDark)
      .text("well-connected 4-clique");
    EDGES.forEach(([a, c]) => {
      const [x1, y1] = POS[NAMES[a]], [x2, y2] = POS[NAMES[c]];
      g.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
        .attr("stroke", P.muted).attr("stroke-width", 1.8).attr("opacity", 0.6);
    });
    NAMES.forEach((nm, i) => {
      const [cx, cy] = POS[nm];
      g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", radius(i))
        .attr("fill", side(i))
        .attr("stroke", i === top ? P.text : "none").attr("stroke-width", 2.5);
      g.append("text").attr("x", cx).attr("y", cy + 4.5).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", "#fff")
        .style("pointer-events", "none").text(nm);
    });

    // ── ranked bars: Katz now, with ghost ticks for the two extremes ──
    const bx = 462, bw = 190, y0 = 58, row = 20, bh = 10;
    g.append("text").attr("x", 430).attr("y", 46).attr("font-size", 12.5)
      .attr("font-weight", 600).attr("fill", P.text).text("Katz score, ranked (top node = 1)");
    const order = d3.range(n).slice().sort((i, j) => (Math.abs(x[j] - x[i]) > 1e-9 ? x[j] - x[i] : i - j));
    const tri = (cx, yTip, up, col) => {
      const base = up ? yTip + 4 : yTip - 4;
      g.append("path")
        .attr("d", `M ${cx - 3.5} ${base} L ${cx + 3.5} ${base} L ${cx} ${yTip} Z`)
        .attr("fill", col).attr("opacity", 0.9);
    };
    order.forEach((i, k) => {
      const y = y0 + k * row;
      const isTop = i === top;
      g.append("text").attr("x", 452).attr("y", y + 10).attr("text-anchor", "end")
        .attr("font-size", 12.5).attr("font-weight", isTop ? 700 : 400)
        .attr("fill", isTop ? side(i) : P.text).text(NAMES[i]);
      g.append("rect").attr("x", bx).attr("y", y).attr("width", Math.max(2, bw * rel[i]))
        .attr("height", bh).attr("rx", 2).attr("fill", side(i)).attr("opacity", isTop ? 1 : 0.5);
      tri(bx + bw * (deg[i] / degMax), y - 1, false, P.muted);        // pure degree, above
      tri(bx + bw * (ev[i] / evMax), y + bh + 1, true, P.purple);       // pure eigenvector, below
      g.append("text").attr("x", bx + bw + 8).attr("y", y + 10)
        .attr("font-family", MONO).attr("font-size", 12.5).attr("font-weight", isTop ? 700 : 400)
        .attr("fill", isTop ? P.text : P.muted).text(f3(rel[i]));
    });
    const ly = y0 + n * row + 18;
    tri(451, ly - 3, false, P.muted);
    g.append("text").attr("x", 462).attr("y", ly).attr("font-size", 12.5).attr("fill", P.muted)
      .text("tick above a bar: pure degree");
    tri(451, ly + 12, true, P.purple);
    g.append("text").attr("x", 462).attr("y", ly + 18).attr("font-size", 12.5).attr("fill", P.muted)
      .text("tick below a bar: pure eigenvector");

    // ── the dial itself: 0 … β* … 1/λmax, colored by who wins ──
    const ax0 = 40, ax1 = 720, ay = 328;
    const ax = (v) => ax0 + (ax1 - ax0) * (v / betaMax);
    const xs = Number.isFinite(bstar) ? ax(bstar) : ax1;
    g.append("line").attr("x1", ax0).attr("y1", ay).attr("x2", xs - 2).attr("y2", ay)
      .attr("stroke", P.accent).attr("stroke-width", 4).attr("opacity", 0.55);
    g.append("line").attr("x1", xs + 2).attr("y1", ay).attr("x2", ax1).attr("y2", ay)
      .attr("stroke", P.blue).attr("stroke-width", 4).attr("opacity", 0.55);
    if (Number.isFinite(bstar)) {
      g.append("line").attr("x1", xs).attr("y1", ay - 7).attr("x2", xs).attr("y2", ay + 7)
        .attr("stroke", P.text).attr("stroke-width", 1.5);
      g.append("text").attr("x", xs).attr("y", ay - 12).attr("text-anchor", "middle")
        .attr("font-family", MONO).attr("font-size", 12.5).attr("font-weight", 700)
        .attr("fill", P.text).text(`β* = ${f3(bstar)}`);
    }
    g.append("circle").attr("cx", ax(b)).attr("cy", ay).attr("r", 6)
      .attr("fill", P.accentDark).attr("stroke", P.paper).attr("stroke-width", 2);
    g.append("text").attr("x", ax0).attr("y", ay + 20).attr("font-family", MONO)
      .attr("font-size", 12.5).attr("fill", P.muted).text("0");
    g.append("text").attr("x", ax1).attr("y", ay + 20).attr("text-anchor", "end")
      .attr("font-family", MONO).attr("font-size", 12.5).attr("fill", P.muted)
      .text(`1/λmax = ${f3(betaMax)}`);

    // ── readouts ──
    g.append("text").attr("x", 20).attr("y", 374).attr("font-family", MONO)
      .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.text)
      .text(`β = ${f3(b)}   (λmax = ${lam.toFixed(2)}, so β must stay below 1/λmax = ${f3(betaMax)})`);
    const v1 = past
      ? `top node: ${NAMES[top]} — past the tipping point β* = ${f3(bstar)}:`
      : `top node: ${NAMES[top]}, the popular hub — the tipping point lies ahead at β* = ${f3(bstar)}`;
    const v2 = past
      ? "the well-connected clique now outranks the popular hub"
      : "below it, short walks dominate and Katz still counts edges much as degree does";
    g.append("text").attr("x", 20).attr("y", 396).attr("font-size", 12.5)
      .attr("font-weight", 600).attr("fill", past ? P.blueDark : P.accentDark).text(v1);
    g.append("text").attr("x", 20).attr("y", 418).attr("font-size", 12.5)
      .attr("font-weight", 600).attr("fill", past ? P.blueDark : P.accentDark).text(v2);
    g.append("text").attr("x", 20).attr("y", 440).attr("font-size", 12.5).attr("fill", P.muted)
      .text("a leaf's endorsement is worth little because leaves themselves are endorsed by nobody");

    const val = document.getElementById("w2-kz-betaval");
    if (val) val.textContent = `β = ${f3(b)} (${Math.round(pct)} % of 1/λmax)`;
  }

  const slider = document.getElementById("w2-kz-beta");
  const pills = document.querySelectorAll("#w2-kz-widget [data-beta]");
  function syncPills() {
    for (const o of pills) o.classList.toggle("active", PRESETS[o.dataset.beta] === pct);
  }
  if (slider) {
    slider.value = String(pct);
    slider.addEventListener("input", () => { pct = parseFloat(slider.value); syncPills(); render(); });
  }
  for (const bt of pills) {
    bt.addEventListener("click", () => {
      pct = PRESETS[bt.dataset.beta];
      if (slider) slider.value = String(pct);
      syncPills();
      render();
    });
  }
  U.onThemeChange(render);
  U.lazyBoot("w2-kz-svg", render);
})();
