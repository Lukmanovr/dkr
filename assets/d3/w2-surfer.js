/* Widget 2.x — Dead ends, spider traps, and the teleport that fixes both.
 *
 * A six-page toy web drawn as a wheel around the hub page P4. Each press of
 * "step" applies one power-iteration update
 *     r ← β M r + (1 − β)·1/n
 * to the surfer's mass vector, starting uniform. Two pathologies can be
 * switched on: a DEAD END (P6 loses both outlinks) and a SPIDER TRAP (P5 and
 * P6 link only to each other). With β = 1 the dead end's mass simply vanishes
 * (that is the leak: a zero column makes M sub-stochastic) and the trap
 * absorbs everything; with β = 0.85 the dangling mass is spread uniformly,
 * the total stays exactly 1, and the trap leaks 15 % of its mass per step.
 *
 * Every number on screen is computed here from the edge list; the verdict
 * line is chosen from the MEASURED state (Σ r, the trap share), never from
 * which button happens to be pressed.
 *
 * The healthy web is strongly connected and aperiodic (cycles of length 3, 4
 * and 6 meet at P4), so the plain walk converges there. The trap is a
 * two-cycle, so inside it the two pages pass the mass back and forth — only
 * the trap's TOTAL settles, which is the quantity we read out.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 428;

  const n = 6;
  const NAMES = ["P1", "P2", "P3", "P4", "P5", "P6"];
  // healthy web: out-neighbor lists (0-indexed). P4 is the hub with three
  // outlinks, one of which is the trap's only entrance.
  const HEALTHY = [[1], [2], [3], [0, 1, 4], [5], [0, 3]];

  function outlinks(scen) {
    const g = HEALTHY.map((o) => o.slice());
    if (scen === "dead") g[5] = [];               // P6: no outlinks at all
    if (scen === "trap") { g[4] = [5]; g[5] = [4]; } // P5 ⇄ P6 and nothing else
    return g;
  }

  // wheel layout: P4 in the middle, the other five on a regular pentagon
  const CX = 225, CY = 218, R = 130;
  const ANG = { P2: -90, P3: -18, P5: 54, P6: 126, P1: 198 };
  const POS = { P4: [CX, CY] };
  for (const nm of Object.keys(ANG)) {
    const a = (ANG[nm] * Math.PI) / 180;
    POS[nm] = [CX + R * Math.cos(a), CY + R * Math.sin(a)];
  }
  // outward direction of each page's mass label (away from the hub; the hub's
  // own label sits in the free wedge between the P3 and P5 spokes)
  const DIR = {};
  for (const nm of Object.keys(ANG)) DIR[nm] = [Math.cos((ANG[nm] * Math.PI) / 180), Math.sin((ANG[nm] * Math.PI) / 180)];
  DIR.P4 = [Math.cos((18 * Math.PI) / 180), Math.sin((18 * Math.PI) / 180)];

  // ── state ──
  let scen = "healthy";
  let beta = 1;
  let r = Array(n).fill(1 / n);
  let k = 0;

  function reset() { r = Array(n).fill(1 / n); k = 0; }

  // one power-iteration step. With β = 1 a dangling page's mass is NOT
  // redistributed — that is the leak. With teleport on, the dangling column
  // is treated as uniform (the standard choice), so Σ r is preserved.
  function step() {
    const g = outlinks(scen);
    const nx = Array(n).fill((1 - beta) / n);
    let dangling = 0;
    for (let u = 0; u < n; u++) {
      if (g[u].length === 0) { dangling += r[u]; continue; }
      const s = (beta * r[u]) / g[u].length;
      for (const v of g[u]) nx[v] += s;
    }
    if (beta < 1) for (let v = 0; v < n; v++) nx[v] += (beta * dangling) / n;
    r = nx;
    k += 1;
  }

  const f3 = (x) => x.toFixed(3);
  const radius = (mass) => Math.max(6, 56 * Math.sqrt(Math.max(0, mass)));

  // quadratic curve from u to v, bent to the right of travel so reciprocal
  // links form a lens, trimmed to the two disc boundaries so the arrowhead
  // lands on the target's rim whatever its current radius.
  function edgePath(u, v, ru, rv) {
    const [x1, y1] = POS[NAMES[u]], [x2, y2] = POS[NAMES[v]];
    const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy);
    const bend = 0.13 * L;
    const cx = (x1 + x2) / 2 - (dy / L) * bend, cy = (y1 + y2) / 2 + (dx / L) * bend;
    const B = (t) => [(1 - t) * (1 - t) * x1 + 2 * (1 - t) * t * cx + t * t * x2,
                      (1 - t) * (1 - t) * y1 + 2 * (1 - t) * t * cy + t * t * y2];
    let t0 = 0, t1 = 1;
    for (let i = 0; i <= 200; i++) { const t = i / 200; const [x, y] = B(t); if (Math.hypot(x - x1, y - y1) <= ru + 1.5) t0 = t; }
    for (let i = 200; i >= 0; i--) { const t = i / 200; const [x, y] = B(t); if (Math.hypot(x - x2, y - y2) <= rv + 2.5) t1 = t; }
    const [ax, ay] = B(t0), [bx, by] = B(t1);
    const qx = (1 - t0) * (1 - t1) * x1 + (t0 * (1 - t1) + t1 * (1 - t0)) * cx + t0 * t1 * x2;
    const qy = (1 - t0) * (1 - t1) * y1 + (t0 * (1 - t1) + t1 * (1 - t0)) * cy + t0 * t1 * y2;
    return `M${ax.toFixed(1)},${ay.toFixed(1)} Q${qx.toFixed(1)},${qy.toFixed(1)} ${bx.toFixed(1)},${by.toFixed(1)}`;
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w2-sf-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const MONO = "'JetBrains Mono', monospace";

    const total = r.reduce((a, b) => a + b, 0);
    const trapMass = r[4] + r[5];
    const gNow = outlinks(scen);
    const bad = (u) => (scen === "dead" && u === 5) || (scen === "trap" && u >= 4);
    const rad = r.map(radius);

    const defs = svg.append("defs");
    const marker = (id, fill, op) => defs.append("marker").attr("id", id)
      .attr("viewBox", "0 0 10 10").attr("refX", 10).attr("refY", 5)
      .attr("markerUnits", "userSpaceOnUse").attr("markerWidth", 9).attr("markerHeight", 9)
      .attr("orient", "auto")
      .append("path").attr("d", "M0,0 L10,5 L0,10 Z").attr("fill", fill).attr("opacity", op);
    marker("w2sfArrow", P.muted, 0.9);
    marker("w2sfArrowHot", P.red, 0.95);
    marker("w2sfArrowGhost", P.red, 0.3);

    // ── hint line ──
    const hint = scen === "healthy"
      ? "a healthy web: every page can reach every other page"
      : scen === "dead"
        ? "dead end: P6 lost its outlinks (dashed)"
        : "spider trap: P5 and P6 now link only to each other";
    g.append("text").attr("x", 20).attr("y", 24).attr("font-size", 12.5).attr("fill", P.muted).text(hint);

    // ── links: live ones with arrowheads, removed ones as dashed ghosts ──
    for (let u = 0; u < n; u++) {
      for (const v of HEALTHY[u]) {
        if (gNow[u].includes(v)) continue;
        g.append("path").attr("d", edgePath(u, v, rad[u], rad[v]))
          .attr("fill", "none").attr("stroke", P.red).attr("stroke-width", 1.6)
          .attr("stroke-dasharray", "3,4").attr("opacity", 0.3)
          .attr("marker-end", "url(#w2sfArrowGhost)");
      }
    }
    for (let u = 0; u < n; u++) {
      for (const v of gNow[u]) {
        const hot = scen === "trap" && u >= 4 && v >= 4;
        g.append("path").attr("d", edgePath(u, v, rad[u], rad[v]))
          .attr("fill", "none").attr("stroke", hot ? P.red : P.muted)
          .attr("stroke-width", hot ? 2.4 : 1.8).attr("opacity", hot ? 0.9 : 0.7)
          .attr("marker-end", hot ? "url(#w2sfArrowHot)" : "url(#w2sfArrow)");
      }
    }

    // ── pages: area ∝ mass, name inside when there is room, mass beside ──
    NAMES.forEach((nm, u) => {
      const [x, y] = POS[nm], rr = rad[u];
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", rr)
        .attr("fill", bad(u) ? P.red : P.blue).attr("opacity", 0.92);
      const nameInside = rr >= 12;
      if (nameInside) {
        g.append("text").attr("x", x).attr("y", y + 4.5).attr("text-anchor", "middle")
          .attr("font-size", 13).attr("font-weight", 700).attr("fill", "#fff").text(nm);
      }
      const [dx, dy] = DIR[nm];
      const lx = x + dx * (rr + 7), ly = y + dy * (rr + 7);
      const anchor = dx > 0.35 ? "start" : dx < -0.35 ? "end" : "middle";
      const base = dy > 0.35 ? 12 : dy < -0.35 ? -4 : 4.5;
      g.append("text").attr("x", lx).attr("y", ly + base).attr("text-anchor", anchor)
        .attr("font-family", MONO).attr("font-size", 12.5).attr("font-weight", 600)
        .attr("fill", bad(u) ? P.red : P.text)
        .text(nameInside ? f3(r[u]) : `${nm} ${f3(r[u])}`);
    });

    // ── right panel: the six masses as bars, then the total ──
    const bx = 445, bw = 260, barX = 478;
    g.append("text").attr("x", bx).attr("y", 44).attr("font-size", 12.5).attr("fill", P.muted)
      .text("mass at each page");
    NAMES.forEach((nm, u) => {
      const y = 58 + u * 22;
      const w = Math.min(bw, bw * Math.max(0, r[u]));
      g.append("text").attr("x", bx).attr("y", y + 11.5).attr("font-size", 12.5).attr("font-weight", 700)
        .attr("fill", bad(u) ? P.red : P.text).text(nm);
      g.append("rect").attr("x", barX).attr("y", y).attr("width", w).attr("height", 14)
        .attr("fill", bad(u) ? P.red : P.blue).attr("opacity", 0.85).attr("rx", 2);
      g.append("text").attr("x", barX + w + 6).attr("y", y + 11.5)
        .attr("font-family", MONO).attr("font-size", 12.5).attr("fill", P.text).text(f3(r[u]));
    });
    const ty = 206;
    g.append("text").attr("x", bx).attr("y", ty)
      .attr("font-family", MONO).attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.text)
      .text(`Σ r = ${f3(total)}`);
    g.append("rect").attr("x", barX).attr("y", ty + 8).attr("width", bw).attr("height", 12)
      .attr("fill", "none").attr("stroke", P.border).attr("stroke-width", 1.2).attr("rx", 2);
    g.append("rect").attr("x", barX).attr("y", ty + 8).attr("width", Math.min(bw, bw * Math.max(0, total)))
      .attr("height", 12).attr("fill", P.accent).attr("opacity", 0.85).attr("rx", 2);
    g.append("text").attr("x", barX).attr("y", ty + 34).attr("text-anchor", "middle")
      .attr("font-family", MONO).attr("font-size", 12.5).attr("fill", P.muted).text("0");
    g.append("text").attr("x", barX + bw).attr("y", ty + 34).attr("text-anchor", "middle")
      .attr("font-family", MONO).attr("font-size", 12.5).attr("fill", P.muted).text("1");

    // ── readouts ──
    g.append("text").attr("x", bx).attr("y", 270)
      .attr("font-family", MONO).attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.text)
      .text(`step k = ${k}`);
    if (scen === "trap") {
      g.append("text").attr("x", bx).attr("y", 292)
        .attr("font-family", MONO).attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.red)
        .text(`mass inside the trap = ${f3(trapMass)}`);
    }
    g.append("text").attr("x", bx).attr("y", 322).attr("font-size", 12.5).attr("fill", P.muted)
      .text(beta === 1 ? "each step: r ← M r   (β = 1)" : "each step: r ← 0.85·M r + 0.15/6");
    if (scen === "dead") {
      g.append("text").attr("x", bx).attr("y", 340).attr("font-size", 12.5).attr("fill", P.muted)
        .text(beta === 1 ? "P6's mass has nowhere to go: lost" : "P6's mass is spread evenly, 1/6 each");
    } else {
      g.append("text").attr("x", bx).attr("y", 340).attr("font-size", 12.5).attr("fill", P.muted)
        .text(scen === "trap"
          ? (beta === 1 ? "the trap keeps 100% of its mass per step" : "the trap keeps 85% of its mass per step")
          : "no dangling pages, nothing to lose");
    }

    // ── verdict, decided from the measured state ──
    let verdict, color;
    if (beta < 1) {
      if (Math.abs(total - 1) < 5e-4) {
        verdict = scen === "trap"
          ? "teleport keeps Σ r = 1 and the trap leaks 15% per step"
          : "teleport keeps Σ r = 1 — dangling mass is spread, not lost";
      } else {
        verdict = "teleport is pulling Σ r back to 1: each step Σ r ← 0.85·Σ r + 0.15";
      }
      color = P.green;
    } else if (scen === "dead") {
      verdict = "mass is leaking out of the web — the fixed point is r = 0";
      color = P.red;
    } else if (scen === "trap") {
      verdict = "the trap is swallowing everything";
      color = P.red;
    } else {
      verdict = "a healthy web: Σ r stays 1 and the walk settles without teleport";
      color = P.muted;
    }
    g.append("text").attr("x", 20).attr("y", H - 16).attr("font-size", 13.5).attr("font-weight", 700)
      .attr("fill", color).text(verdict);
  }

  // ── controls ──
  const root = document.getElementById("w2-sf-widget");
  const setActive = (attr, val) => {
    for (const o of root.querySelectorAll(`[${attr}]`)) o.classList.toggle("active", o.getAttribute(attr) === val);
  };
  for (const b of root.querySelectorAll("[data-scen]")) {
    b.addEventListener("click", () => {
      scen = b.dataset.scen;
      setActive("data-scen", scen);
      reset();                                   // a new web starts from uniform
      render();
    });
  }
  for (const b of root.querySelectorAll("[data-beta]")) {
    b.addEventListener("click", () => {
      beta = Number(b.dataset.beta);
      setActive("data-beta", b.dataset.beta);
      render();                                  // keep the state: watch teleport act on it
    });
  }
  document.getElementById("w2-sf-step").addEventListener("click", () => { step(); render(); });
  document.getElementById("w2-sf-run").addEventListener("click", () => { for (let i = 0; i < 30; i++) step(); render(); });
  document.getElementById("w2-sf-reset").addEventListener("click", () => { reset(); render(); });

  U.onThemeChange(render);
  U.lazyBoot("w2-sf-svg", render);
})();
