/* Week 2 widget — Two power iterations by hand, animated.
 *
 * The lecture's three-page web (X → Y, X → Z, Y → X, Z → Y) with β = 1, so the
 * update is the bare r ← M·r and the first steps are clean fractions. Each press
 * of "step" splits the mass at every page down its outlinks exactly as the
 * column of M says — half of X to Y and half to Z, all of Y to X, all of Z to Y —
 * while the matrix–vector product is written out entry by entry in the middle
 * and the distance to the fixed point is plotted on the right.
 *
 * Nothing shown is typed in: M is built from the link list, the iteration runs
 * in exact rational arithmetic (BigInt) so the printed fractions are the
 * lecture's verbatim, the inlink counts in the verdict are counted from the same
 * list, and r* is found by iterating the floating-point update 200 times.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 460;
  const MAXK = 40;                 // "run" stops here; the chart's x-axis ends here too
  const TOL = 0.0005;              // converged once the L1 distance to r* is below this

  const PAGES = ["X", "Y", "Z"];
  const LINKS = [["X", "Y"], ["X", "Z"], ["Y", "X"], ["Z", "Y"]];   // u → v
  const n = PAGES.length;
  const ix = (p) => PAGES.indexOf(p);
  const dout = PAGES.map((u) => LINKS.filter(([a]) => a === u).length);
  const din = PAGES.map((v) => LINKS.filter(([, b]) => b === v).length);

  // ── exact rationals (BigInt), so steps 0–2 print the lecture's fractions verbatim ──
  const gcd = (a, b) => (b === 0n ? a : gcd(b, a % b));
  const mag = (x) => (x < 0n ? -x : x);
  const Q = (nu, de) => { const g = gcd(mag(nu), mag(de)) || 1n; return { n: nu / g, d: de / g }; };
  const ZERO = Q(0n, 1n);
  const qadd = (a, b) => Q(a.n * b.d + b.n * a.d, a.d * b.d);
  const qmul = (a, b) => Q(a.n * b.n, a.d * b.d);
  const qnum = (a) => Number(a.n) / Number(a.d);
  const qstr = (a) => (a.d === 1n ? String(a.n) : `${a.n}/${a.d}`);
  const isZero = (a) => a.n === 0n;
  const isOne = (a) => a.n === a.d;

  // column-stochastic M: M[v][u] = 1/dout(u) when u → v — column u says where u's mass goes
  const M = PAGES.map(() => PAGES.map(() => ZERO));
  LINKS.forEach(([u, v]) => { M[ix(v)][ix(u)] = Q(1n, BigInt(dout[ix(u)])); });
  const Mf = M.map((row) => row.map(qnum));
  const colSum = (j) => M.reduce((s, row) => qadd(s, row[j]), ZERO);
  const applyM = (r) => M.map((row) => row.reduce((s, m, j) => qadd(s, qmul(m, r[j])), ZERO));
  const applyMf = (r) => Mf.map((row) => row.reduce((s, m, j) => s + m * r[j], 0));

  const STARTS = {
    uniform: PAGES.map(() => Q(1n, BigInt(n))),
    allx: PAGES.map((p) => (p === "X" ? Q(1n, 1n) : ZERO)),
  };

  // r*: the fixed point, found by iterating far past convergence — never typed in.
  const RSTAR = (() => {
    let r = PAGES.map(() => 1 / n);
    for (let k = 0; k < 200; k++) r = applyMf(r);
    return r;
  })();
  const f3 = (v) => v.toFixed(3);
  const l1 = (a, b) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0);
  if (RSTAR.map(f3).join(",") !== "0.400,0.400,0.200") {
    console.warn("w2-poweriter: r* did not come out as (0.400, 0.400, 0.200):", RSTAR);
  }
  const TIE = Math.abs(RSTAR[ix("X")] - RSTAR[ix("Y")]) < 1e-9;

  const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";
  const sup = (k) => String(k).split("").map((c) => SUP[+c]).join("");
  const rk = (k) => `r⁽${sup(k)}⁾`;
  const MONO = "'JetBrains Mono', monospace";

  // ── state ──
  let start = "uniform";
  let hist = [STARTS[start]];                  // hist[k] = r⁽ᵏ⁾, exact
  const K = () => hist.length - 1;
  // fractions for the hand-worked steps 0–2, three decimals afterwards
  const fmt = (q, k) => (k <= 2 ? qstr(q) : f3(qnum(q)));
  const dists = () => hist.map((r) => l1(r.map(qnum), RSTAR));

  // ── geometry of the little web (left panel) ──
  const R = 22;
  const POS = { X: [128, 122], Y: [206, 250], Z: [50, 250] };
  const CEN = [0, 1].map((a) => PAGES.reduce((s, p) => s + POS[p][a], 0) / n);
  const unit = (p, q) => { const ex = q[0] - p[0], ey = q[1] - p[1], l = Math.hypot(ex, ey) || 1; return [ex / l, ey / l]; };

  // Quadratic arc from the rim of u to just short of the rim of v. A two-way pair
  // bows to the right of travel so X→Y and Y→X never overlap.
  function curve(u, v) {
    const a = POS[u], b = POS[v];
    const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const [dx, dy] = unit(a, b);
    const twoWay = LINKS.some(([p, q]) => p === v && q === u);
    const bow = twoWay ? 24 : 0;
    const c = [mid[0] - bow * dy, mid[1] + bow * dx];
    const ua = unit(a, c), ub = unit(b, c);
    const p0 = [a[0] + R * ua[0], a[1] + R * ua[1]];
    const p1 = [b[0] + (R + 3) * ub[0], b[1] + (R + 3) * ub[1]];
    // label offset: along the bow for arcs, away from the triangle's center for straight edges
    const off = twoWay ? [-dy, dx] : unit(CEN, mid);
    return { p0, c, p1, off };
  }
  const bez = (q, t) => [
    (1 - t) * (1 - t) * q.p0[0] + 2 * (1 - t) * t * q.c[0] + t * t * q.p1[0],
    (1 - t) * (1 - t) * q.p0[1] + 2 * (1 - t) * t * q.c[1] + t * t * q.p1[1],
  ];

  const FLY = 650;                              // packet flight time, ms

  function render(opts) {
    const { from = null, reveal = false } = opts || {};
    const P = U.pal();
    const svg = U.svgIn("w2-pi-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const anim = U.motionOK() && (from !== null || reveal);
    const k = K();
    const cur = hist[k];
    const prev = k >= 1 ? hist[k - 1] : hist[0];   // what the last multiplication consumed
    const kk = Math.max(k, 1);                     // the step whose product is written out
    const D = dists();
    const dist = D[k];
    const converged = dist < TOL;

    const defs = svg.append("defs");
    defs.append("marker").attr("id", "w2pi-arrow").attr("markerUnits", "userSpaceOnUse")
      .attr("markerWidth", 10).attr("markerHeight", 10).attr("refX", 9).attr("refY", 5).attr("orient", "auto")
      .append("path").attr("d", "M0,0 L10,5 L0,10 Z").attr("fill", P.muted);
    PAGES.forEach((p) => {
      defs.append("clipPath").attr("id", `w2pi-clip-${p}`)
        .append("circle").attr("cx", POS[p][0]).attr("cy", POS[p][1]).attr("r", R - 1);
    });
    const g = svg.append("g");

    // ════════ LEFT — the three-page web, mass as a fill level ════════
    g.append("text").attr("x", 16).attr("y", 22).attr("font-size", 12.5).attr("fill", P.muted)
      .text("the three-page web");

    const curves = {};
    LINKS.forEach(([u, v]) => {
      const q = curve(u, v);
      curves[u + v] = q;
      g.append("path")
        .attr("d", `M${q.p0[0]},${q.p0[1]} Q${q.c[0]},${q.c[1]} ${q.p1[0]},${q.p1[1]}`)
        .attr("fill", "none").attr("stroke", P.muted).attr("stroke-width", 1.8).attr("opacity", 0.7)
        .attr("marker-end", "url(#w2pi-arrow)");
      const m = bez(q, 0.5);
      g.append("text").attr("x", m[0] + 11 * q.off[0]).attr("y", m[1] + 11 * q.off[1] + 4.5)
        .attr("text-anchor", "middle").attr("font-family", MONO).attr("font-size", 12.5)
        .attr("fill", P.muted).attr("stroke", P.paper).attr("stroke-width", 3.5).attr("paint-order", "stroke")
        .text(qstr(M[ix(v)][ix(u)]));
    });

    PAGES.forEach((p, i) => {
      const [cx, cy] = POS[p];
      const mass = qnum(cur[i]);
      const massPrev = qnum(prev[i]);
      g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", R).attr("fill", P.paper);
      const lvl = g.append("rect").attr("clip-path", `url(#w2pi-clip-${p})`)
        .attr("x", cx - R).attr("width", 2 * R)
        .attr("fill", P.blue).attr("fill-opacity", 0.85);
      const yOf = (ms) => cy + R - 2 * R * ms, hOf = (ms) => 2 * R * ms;
      if (anim && from !== null) {
        lvl.attr("y", yOf(massPrev)).attr("height", hOf(massPrev))
          .transition().delay(FLY).duration(320).ease(d3.easeCubicOut)
          .attr("y", yOf(mass)).attr("height", hOf(mass));
      } else {
        lvl.attr("y", yOf(mass)).attr("height", hOf(mass));
      }
      g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", R)
        .attr("fill", "none").attr("stroke", P.blue).attr("stroke-width", 2);
      g.append("text").attr("x", cx).attr("y", cy + 5.5).attr("text-anchor", "middle")
        .attr("font-size", 15).attr("font-weight", 700).attr("fill", P.text)
        .attr("stroke", P.paper).attr("stroke-width", 3.5).attr("paint-order", "stroke")
        .text(p);
      // the number: above X, below Y and Z — always outside the triangle
      const ly = p === "X" ? cy - R - 12 : cy + R + 18;
      g.append("text").attr("x", cx).attr("y", ly).attr("text-anchor", "middle")
        .attr("font-family", MONO).attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text)
        .text(fmt(cur[i], k));
    });

    g.append("text").attr("x", 16).attr("y", 334)
      .attr("font-family", MONO).attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text)
      .text(`${rk(k)} = (${cur.map((q) => fmt(q, k)).join(", ")})`);
    g.append("text").attr("x", 16).attr("y", 354).attr("font-size", 12.5).attr("fill", P.muted)
      .text(`sum = ${qstr(cur.reduce(qadd, ZERO))} — mass is conserved`);

    // ════════ MIDDLE — M, its column sums, and the product entry by entry ════════
    const mx = 304, my = 68, cell = 52;
    g.append("text").attr("x", 268).attr("y", 22).attr("font-size", 12.5).attr("fill", P.muted)
      .text("M: where each page sends its mass");
    PAGES.forEach((p, j) => {
      g.append("text").attr("x", mx + j * cell + cell / 2).attr("y", my - 10)
        .attr("text-anchor", "middle").attr("font-size", 12.5).attr("font-weight", 600)
        .attr("fill", P.muted).text(`from ${p}`);
      g.append("text").attr("x", mx - 6).attr("y", my + j * cell + cell / 2 + 4.5)
        .attr("text-anchor", "end").attr("font-size", 12.5).attr("font-weight", 600)
        .attr("fill", P.muted).text(`to ${p}`);
    });
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const z = isZero(M[i][j]);
        const cx = mx + j * cell, cy = my + i * cell;
        g.append("rect").attr("x", cx).attr("y", cy).attr("width", cell).attr("height", cell)
          .attr("fill", z ? "none" : P.blue).attr("fill-opacity", z ? 1 : 0.16)
          .attr("stroke", P.border).attr("stroke-width", 1);
        g.append("text").attr("x", cx + cell / 2).attr("y", cy + cell / 2 + 5)
          .attr("text-anchor", "middle").attr("font-family", MONO).attr("font-size", 13.5)
          .attr("font-weight", z ? 400 : 700).attr("fill", z ? P.muted : P.text)
          .text(qstr(M[i][j]));
      }
    }
    PAGES.forEach((p, j) => {
      g.append("text").attr("x", mx + j * cell + cell / 2).attr("y", my + n * cell + 18)
        .attr("text-anchor", "middle").attr("font-family", MONO).attr("font-size", 12.5)
        .attr("fill", P.blueDark).attr("font-weight", 600)
        .text(`Σ = ${qstr(colSum(j))}`);
    });
    // the vector being multiplied, as a column beside M
    const vx = mx + n * cell + 18;
    g.append("text").attr("x", vx - 9).attr("y", my + (n * cell) / 2 + 5).attr("text-anchor", "middle")
      .attr("font-size", 15).attr("fill", P.muted).text("·");
    g.append("text").attr("x", vx + cell / 2).attr("y", my - 10).attr("text-anchor", "middle")
      .attr("font-family", MONO).attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.muted)
      .text(rk(Math.max(k - 1, 0)));
    PAGES.forEach((p, i) => {
      const cy = my + i * cell;
      g.append("rect").attr("x", vx).attr("y", cy).attr("width", cell).attr("height", cell)
        .attr("fill", P.paper).attr("stroke", P.border).attr("stroke-width", 1);
      g.append("text").attr("x", vx + cell / 2).attr("y", cy + cell / 2 + 5).attr("text-anchor", "middle")
        .attr("font-family", MONO).attr("font-size", 13).attr("fill", P.text)
        .text(fmt(prev[i], kk));
    });

    const sy = 270;
    g.append("text").attr("x", 268).attr("y", sy)
      .attr("font-family", MONO).attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.text)
      .text(k >= 1 ? `${rk(k)} = M·${rk(k - 1)}, entry by entry:` : `${rk(1)} = M·${rk(0)} will be:`);
    PAGES.forEach((p, i) => {
      const terms = PAGES.map((u, j) => (isZero(M[i][j]) ? null : `${qstr(M[i][j])}·${fmt(prev[j], kk)}`))
        .filter(Boolean).join(" + ");
      const y = sy + 24 + i * 22;
      g.append("text").attr("x", 268).attr("y", y).attr("font-size", 13).attr("font-weight", 700)
        .attr("fill", P.accentDark).text(p);
      g.append("text").attr("x", 284).attr("y", y).attr("font-family", MONO).attr("font-size", 12.5)
        .attr("fill", P.text).text(`${terms} = ${k >= 1 ? fmt(cur[i], k) : "?"}`);
    });
    if (k === 0) {
      g.append("text").attr("x", 268).attr("y", sy + 24 + n * 22 + 2).attr("font-size", 12.5)
        .attr("fill", P.muted).text("predict each sum, then press step");
    }

    // ════════ RIGHT — L1 distance to r* per step ════════
    const px0 = 578, px1 = 744, py0 = 52, py1 = 290;
    g.append("text").attr("x", 540).attr("y", 22).attr("font-size", 12.5).attr("fill", P.muted)
      .text("L1 distance to r*, per step");
    const xs = d3.scaleLinear().domain([0, MAXK]).range([px0, px1]);
    const ys = d3.scaleLog().domain([1e-7, 2]).range([py1, py0]);
    const yOfD = (d) => ys(Math.max(d, 1e-7));
    [[1, "1"], [1e-2, "10⁻²"], [1e-4, "10⁻⁴"], [1e-6, "10⁻⁶"]].forEach(([v, lab]) => {
      g.append("line").attr("x1", px0).attr("x2", px1).attr("y1", ys(v)).attr("y2", ys(v))
        .attr("stroke", P.border).attr("stroke-width", 1);
      g.append("text").attr("x", px0 - 8).attr("y", ys(v) + 4.5).attr("text-anchor", "end")
        .attr("font-family", MONO).attr("font-size", 12.5).attr("fill", P.muted).text(lab);
    });
    g.append("line").attr("x1", px0).attr("x2", px1).attr("y1", py1).attr("y2", py1)
      .attr("stroke", P.muted).attr("stroke-width", 1);
    [0, 10, 20, 30, 40].forEach((v) => {
      g.append("text").attr("x", xs(v)).attr("y", py1 + 16).attr("text-anchor", "middle")
        .attr("font-family", MONO).attr("font-size", 12.5).attr("fill", P.muted).text(v);
    });
    // the convergence threshold
    g.append("line").attr("x1", px0).attr("x2", px1).attr("y1", ys(TOL)).attr("y2", ys(TOL))
      .attr("stroke", P.green).attr("stroke-width", 1.4).attr("stroke-dasharray", "4,4");
    g.append("text").attr("x", px1).attr("y", ys(TOL) - 20).attr("text-anchor", "end")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.green).text("converged");
    g.append("text").attr("x", px1).attr("y", ys(TOL) - 6).attr("text-anchor", "end")
      .attr("font-family", MONO).attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.green)
      .text(`< ${TOL}`);

    const pts = D.map((d, i) => [xs(i), yOfD(d)]);
    if (pts.length > 1) {
      g.append("path").attr("d", d3.line()(pts)).attr("fill", "none")
        .attr("stroke", P.blue).attr("stroke-width", 1.2).attr("opacity", 0.45);
    }
    pts.forEach(([x, y], i) => {
      const last = i === k;
      const dot = g.append("circle").attr("cx", x).attr("cy", y)
        .attr("fill", last ? P.accent : D[i] < TOL ? P.green : P.blue)
        .attr("stroke", P.paper).attr("stroke-width", 1);
      if (anim && reveal) {
        dot.attr("r", last ? 5 : 3.5).attr("opacity", 0)
          .transition().delay(i * 22).duration(120).attr("opacity", 1);
      } else if (anim && last) {
        dot.attr("r", 0).transition().delay(FLY).duration(220).attr("r", 5);
      } else {
        dot.attr("r", last ? 5 : 3.5);
      }
    });

    g.append("text").attr("x", 540).attr("y", 330).attr("font-family", MONO)
      .attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text)
      .text(`step ${k} of ${MAXK}`);
    g.append("text").attr("x", 540).attr("y", 352).attr("font-family", MONO)
      .attr("font-size", 12.5).attr("fill", P.text)
      .text(`r* = (${RSTAR.map(f3).join(", ")})`);
    g.append("text").attr("x", 540).attr("y", 370).attr("font-size", 12.5).attr("fill", P.muted)
      .text("(from 200 iterations)");
    if (k >= 10 && D[k] > 0 && D[2] > 0) {
      // average per-step shrink factor, measured on the trajectory itself
      const factor = Math.pow(D[k] / D[2], 1 / (k - 2));
      g.append("text").attr("x", 540).attr("y", 390).attr("font-family", MONO)
        .attr("font-size", 12.5).attr("fill", P.text)
        .text(`shrinks ×${factor.toFixed(2)} per step`);
    }

    // ════════ bottom — the verdict ════════
    if (converged) {
      const l1txt = `converged: r = (${cur.map((q) => f3(qnum(q))).join(", ")})${TIE ? " — X and Y tie" : ""}`;
      const carries = isOne(M[ix("X")][ix("Y")]) ? "all" : qstr(M[ix("X")][ix("Y")]);
      const l2txt = `X has ${din[ix("X")]} inlink, Y has ${din[ix("Y")]} — but X's single inlink carries ${carries} of Y's mass`;
      const bw = Math.min(740, Math.max(l1txt.length * 0.6 * 13, l2txt.length * 0.6 * 12.5) + 28);
      g.append("rect").attr("x", 10).attr("y", 404).attr("width", bw).attr("height", 48).attr("rx", 12)
        .attr("fill", P.green);
      g.append("text").attr("x", 24).attr("y", 423).attr("font-family", MONO)
        .attr("font-size", 13).attr("font-weight", 700).attr("fill", "#fff").text(l1txt);
      g.append("text").attr("x", 24).attr("y", 442).attr("font-size", 12.5).attr("fill", "#fff").text(l2txt);
    } else if (k === 0) {
      g.append("text").attr("x", 16).attr("y", 423).attr("font-size", 13).attr("font-weight", 600)
        .attr("fill", P.accentDark).text("step 0 — press step to apply r ← M·r once");
      g.append("text").attr("x", 16).attr("y", 442).attr("font-size", 12.5).attr("fill", P.muted)
        .text("the fixed point will be the same from either start");
    } else {
      g.append("text").attr("x", 16).attr("y", 423).attr("font-size", 13).attr("font-weight", 600)
        .attr("fill", P.accentDark)
        .text(`step ${k} — still moving: distance to r* = ${f3(dist)}`);
      g.append("text").attr("x", 16).attr("y", 442).attr("font-size", 12.5).attr("fill", P.muted)
        .text(k >= MAXK ? "at the step limit — press reset" : "keep stepping, or run to step 40 — then try the other start");
    }

    // ════════ packets of mass flying down the outlinks (decorative overlay) ════════
    if (anim && from !== null) {
      const layer = svg.append("g").style("pointer-events", "none");
      LINKS.forEach(([u, v]) => {
        const share = qnum(from[ix(u)]) * Mf[ix(v)][ix(u)];
        if (share <= 0) return;
        const q = curves[u + v];
        const dot = layer.append("circle").attr("r", 2 + 12 * Math.sqrt(share))
          .attr("fill", P.accent).attr("opacity", 0.9).attr("cx", q.p0[0]).attr("cy", q.p0[1]);
        dot.transition().duration(FLY).ease(d3.easeCubicInOut)
          .tween("fly", () => (t) => { const p = bez(q, t); dot.attr("cx", p[0]).attr("cy", p[1]); })
          .transition().duration(200).attr("opacity", 0).remove();
      });
    }
  }

  // ── controls ──
  function step() {
    if (K() >= MAXK) { render(); return; }
    const from = hist[K()];
    hist.push(applyM(from));
    render({ from });
  }
  function run() {
    while (K() < MAXK) hist.push(applyM(hist[K()]));
    render({ reveal: true });
  }
  function reset() {
    hist = [STARTS[start]];
    render();
  }
  const on = (id, fn) => { const el = document.getElementById(id); if (el) el.addEventListener("click", fn); };
  on("w2-pi-step", step);
  on("w2-pi-run", run);
  on("w2-pi-reset", reset);
  for (const b of document.querySelectorAll("#w2-pi-widget [data-start]")) {
    b.addEventListener("click", () => {
      start = b.dataset.start;
      for (const o of document.querySelectorAll("#w2-pi-widget [data-start]")) {
        o.classList.toggle("active", o === b);
      }
      reset();
    });
  }
  U.onThemeChange(() => render());
  U.lazyBoot("w2-pi-svg", () => render());
})();
