/* Widget 2.2 — The random surfer, one step at a time.
 * The ten-page toy web of the opening figure. Each press of "step" applies one
 * power-iteration update r ← β·M r + (1−β)·t with the current damping β and
 * teleport vector t (uniform, or all-on-Q for the personalized variant).
 *
 * What the reader sees, and why: the update itself is written on the panel with
 * the current β and t substituted; every link is drawn with a width proportional
 * to the mass it carries THIS step, so the flow is visible, not implied; disc size
 * and a number give each page's current estimate; a ranked bar list reorders live;
 * and a chart of the per-step change |Δr|₁ shows convergence happening, with the
 * contraction bound of the proposition printed beside it. Everything is computed
 * from the edge list at every render.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 480;
  const WEB = {
    Hub: ["Q", "N1", "N2"],
    N1: ["Hub"], N2: ["Hub"], N3: ["Hub"], N4: ["Hub", "V"],
    L1: ["V"], L2: ["V"], L3: ["V"],
    Q: ["Hub"],
    V: ["N3"],
  };
  const NODES = Object.keys(WEB);
  const n = NODES.length;
  // the opening figure's arrangement, compressed into the left half
  const POS = { Hub: [118, 196], N1: [50, 100], N2: [50, 292], N3: [205, 82], N4: [205, 306],
                Q: [300, 158], V: [368, 262], L1: [292, 344], L2: [368, 356], L3: [438, 336] };
  const LINKS = [];
  NODES.forEach((u) => WEB[u].forEach((v) => LINKS.push([u, v])));

  let pr = {}, iter = 0, hist = [];
  function reset() {
    NODES.forEach((u) => { pr[u] = 1 / n; });
    iter = 0; hist = [];
  }
  reset();

  function beta() { return parseFloat(document.getElementById("w2-pr-beta").value); }
  function personalized() { return document.getElementById("w2-pr-tele").checked; }

  function step() {
    const b = beta(), pers = personalized();
    const nxt = {};
    NODES.forEach((u) => { nxt[u] = pers ? (u === "Q" ? 1 - b : 0) : (1 - b) / n; });
    NODES.forEach((u) => {
      const share = (b * pr[u]) / WEB[u].length;
      WEB[u].forEach((v) => { nxt[v] += share; });
    });
    hist.push(NODES.reduce((a, u) => a + Math.abs(nxt[u] - pr[u]), 0));
    pr = nxt;
    iter += 1;
  }

  const radius = (u) => 7 + 34 * Math.sqrt(pr[u]);
  const flow = (u) => beta() * pr[u] / WEB[u].length;      // mass per outlink this step

  // quadratic arc from u's rim to just short of v's rim, bowed left of travel
  function arc(u, v) {
    const [x1, y1] = POS[u], [x2, y2] = POS[v];
    const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy);
    const px = -dy / L, py = dx / L, bow = 0.16 * L;
    const cx = (x1 + x2) / 2 + px * bow, cy = (y1 + y2) / 2 + py * bow;
    const r1 = radius(u) + 2, r2 = radius(v) + 10;
    let d1x = cx - x1, d1y = cy - y1, l1 = Math.hypot(d1x, d1y); d1x /= l1; d1y /= l1;
    let d2x = x2 - cx, d2y = y2 - cy, l2 = Math.hypot(d2x, d2y); d2x /= l2; d2y /= l2;
    return `M ${(x1 + d1x * r1).toFixed(1)} ${(y1 + d1y * r1).toFixed(1)} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${(x2 - d2x * r2).toFixed(1)} ${(y2 - d2y * r2).toFixed(1)}`;
  }

  function render() {
    const P = U.pal();
    const b = beta(), pers = personalized();
    const svg = U.svgIn("w2-pr-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const defs = svg.append("defs");
    for (const [id, col] of [["w2prA", P.muted], ["w2prQ", P.accent], ["w2prV", P.purple]]) {
      defs.append("marker").attr("id", id).attr("markerWidth", 9).attr("markerHeight", 9)
        .attr("refX", 7).attr("refY", 4.5).attr("orient", "auto").attr("markerUnits", "userSpaceOnUse")
        .append("path").attr("d", "M0,0.5 L8,4.5 L0,8.5 Z").attr("fill", col);
    }
    g.append("text").attr("x", 20).attr("y", 22).attr("font-size", 12.5).attr("fill", P.muted)
      .text("the opening figure's web · disc = current estimate · arrow width = mass flowing this step");

    // ── links, width ∝ mass carried this step ──
    const lg = g.append("g").attr("fill", "none").attr("stroke-linecap", "round");
    const colourOf = (v) => (v === "Q" ? P.accent : v === "V" ? P.purple : P.muted);
    LINKS.forEach(([u, v]) => {
      const w = 1 + 30 * flow(u);
      lg.append("path").attr("d", arc(u, v)).attr("stroke", colourOf(v)).attr("stroke-width", w.toFixed(1))
        .attr("opacity", v === "Q" || v === "V" ? 0.9 : 0.5)
        .attr("marker-end", `url(#${v === "Q" ? "w2prQ" : v === "V" ? "w2prV" : "w2prA"})`);
    });

    // ── teleport target ──
    if (pers) {
      const [qx, qy] = POS.Q;
      g.append("circle").attr("cx", qx).attr("cy", qy).attr("r", radius("Q") + 9).attr("fill", "none")
        .attr("stroke", P.accent).attr("stroke-width", 2).attr("stroke-dasharray", "5,4");
      g.append("text").attr("x", qx).attr("y", qy - radius("Q") - 16).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.accent).text("every teleport lands here");
    }

    // ── pages ──
    const nd = g.selectAll("g.n").data(NODES).join("g").attr("class", "n")
      .attr("transform", (u) => `translate(${POS[u][0]},${POS[u][1]})`);
    nd.append("circle").attr("r", (u) => radius(u))
      .attr("fill", (u) => (u === "Q" ? P.accent : u === "V" ? P.purple : u.startsWith("L") ? P.muted : P.blue))
      .attr("opacity", (u) => (u === "Q" || u === "V" ? 1 : u === "Hub" ? 0.92 : u.startsWith("L") ? 0.5 : 0.55));
    nd.append("text").attr("text-anchor", "middle").attr("dy", 4.5)
      .attr("font-size", (u) => (u === "Hub" ? 13.5 : 12.5)).attr("font-weight", 700).attr("fill", "#fff")
      .text((u) => (u === "Q" || u === "V" ? u : u === "Hub" ? "hub" : ""));
    const SIDE = { L1: -1, N3: 1, Q: 1, V: 1, L2: 1, L3: 2 };
    nd.append("text")
      .attr("text-anchor", (u) => (SIDE[u] === 1 ? "start" : SIDE[u] === -1 ? "end" : "middle"))
      .attr("x", (u) => (SIDE[u] === 1 ? radius(u) + 6 : SIDE[u] === -1 ? -(radius(u) + 6) : 0))
      .attr("y", (u) => (SIDE[u] === 2 ? -(radius(u) + 6) : SIDE[u] ? 4 : radius(u) + 15))
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5).attr("fill", P.text)
      .attr("stroke", P.paper).attr("stroke-width", 3.5).attr("paint-order", "stroke")
      .text((u) => pr[u].toFixed(3));

    // ── right panel ──
    const px = 470;
    const panel = g.append("g").attr("transform", `translate(${px},0)`);
    panel.append("text").attr("y", 52).attr("font-family", "'JetBrains Mono', monospace")
      .attr("font-size", 14).attr("font-weight", 700).attr("fill", P.text)
      .text("r ← β·M r + (1−β)·t");
    panel.append("text").attr("y", 72).attr("font-size", 12.5).attr("fill", P.text)
      .text(`β = ${b.toFixed(2)}: links ${Math.round(b * 100)}%, teleport ${Math.round((1 - b) * 100)}%`);
    panel.append("text").attr("y", 90).attr("font-size", 12.5).attr("fill", pers ? P.accent : P.muted)
      .text(pers ? "t = all on Q (personalized)" : "t = 1/10 everywhere (uniform)");

    // ranked bars
    const ranked = NODES.slice().sort((a, c) => pr[c] - pr[a]);
    const bmax = pr[ranked[0]] || 1;
    panel.append("text").attr("y", 118).attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.text)
      .text(`after ${iter} step${iter === 1 ? "" : "s"}, ranked`);
    ranked.forEach((u, i) => {
      const y = 126 + i * 21;
      const w = 4 + 150 * (pr[u] / bmax);
      panel.append("rect").attr("x", 40).attr("y", y).attr("width", w).attr("height", 14).attr("rx", 3)
        .attr("fill", u === "Q" ? P.accent : u === "V" ? P.purple : P.blue)
        .attr("opacity", u === "Q" || u === "V" ? 0.95 : u === "Hub" ? 0.8 : 0.4);
      panel.append("text").attr("x", 34).attr("y", y + 11.5).attr("text-anchor", "end")
        .attr("font-size", 12.5).attr("font-weight", u === "Q" || u === "V" ? 700 : 400)
        .attr("fill", u === "Q" ? P.accent : u === "V" ? P.purple : P.text).text(u === "Hub" ? "hub" : u);
      panel.append("text").attr("x", 46 + w).attr("y", y + 11.5)
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5).attr("fill", P.muted)
        .text(pr[u].toFixed(3));
    });

    // convergence chart: |Δr|₁ per step on a log scale
    const cx0 = 40, cy0 = 352, cw = 200, ch = 74;
    const chart = panel.append("g").attr("transform", `translate(${cx0},${cy0})`);
    const LO = -6, HI = 0;
    const ys = (d) => ch - (Math.max(LO, Math.min(HI, Math.log10(Math.max(d, 1e-9)))) - LO) / (HI - LO) * ch;
    const nx = Math.max(20, hist.length);
    const xs = (k) => (k / (nx - 1)) * cw;
    chart.append("line").attr("x1", 0).attr("x2", cw).attr("y1", ch).attr("y2", ch).attr("stroke", P.border);
    chart.append("line").attr("x1", 0).attr("x2", cw).attr("y1", ys(1e-3)).attr("y2", ys(1e-3))
      .attr("stroke", P.green).attr("stroke-width", 1.3).attr("stroke-dasharray", "4,3");
    chart.append("text").attr("x", cw + 4).attr("y", ys(1e-3) + 4).attr("font-size", 12.5).attr("fill", P.green).text("10⁻³");
    chart.append("text").attr("x", -6).attr("y", ys(1) + 4).attr("text-anchor", "end").attr("font-size", 12.5).attr("fill", P.muted).text("1");
    chart.append("text").attr("x", -6).attr("y", ys(1e-6) + 4).attr("text-anchor", "end").attr("font-size", 12.5).attr("fill", P.muted).text("10⁻⁶");
    if (hist.length) {
      const line = d3.line().x((d, k) => xs(k + 1)).y((d) => ys(d));
      chart.append("path").attr("d", line(hist)).attr("fill", "none").attr("stroke", P.accent).attr("stroke-width", 2);
      chart.selectAll("circle.h").data(hist).join("circle").attr("cx", (d, k) => xs(k + 1)).attr("cy", (d) => ys(d))
        .attr("r", 2.6).attr("fill", P.accent);
    }
    chart.append("text").attr("x", cw / 2).attr("y", -8).attr("text-anchor", "middle").attr("font-size", 12.5)
      .attr("fill", P.muted).text("change per step, |Δr|₁ (log scale)");

    // ── verdicts ──
    const last = hist.length ? hist[hist.length - 1] : null;
    const conv = last !== null && last < 1e-3;
    const firstConv = hist.findIndex((d) => d < 1e-3) + 1;
    const bound = b > 0 && b < 1 ? Math.ceil(Math.log(1e-3) / Math.log(b)) : null;
    let status, note;
    if (iter === 0) {
      status = "everyone starts equal at 1/10 = 0.100 — press step";
      note = b === 0 ? "β = 0: no link is ever followed, so one step gives r = t exactly"
           : `contraction: the error shrinks by ×${b.toFixed(2)} per step, so expect ≈ ${bound} steps to 10⁻³`;
    } else {
      status = `iteration ${iter} · total change |Δr|₁ = ${last.toFixed(4)}${conv ? " — converged (for the eye)" : ""}`;
      if (b === 0) note = "β = 0: no link is ever followed — every page holds exactly its teleport share, r = t";
      else if (conv) note = `first below 10⁻³ at step ${firstConv} · the ×β contraction predicted ≈ ${bound}`;
      else note = `contraction: expect ≈ ${bound} steps to 10⁻³ at β = ${b.toFixed(2)}${b >= 0.95 ? " — a weak contraction, so it crawls" : ""}`;
    }
    g.append("text").attr("x", W / 2).attr("y", 452).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", conv ? P.green : P.text).text(status);
    g.append("text").attr("x", W / 2).attr("y", 471).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text(note);

    document.getElementById("w2-pr-betaval").textContent = b.toFixed(2);
  }

  document.getElementById("w2-pr-step").addEventListener("click", () => { step(); render(); });
  document.getElementById("w2-pr-step10").addEventListener("click", () => {
    for (let k = 0; k < 10; k++) step();
    render();
  });
  document.getElementById("w2-pr-reset").addEventListener("click", () => { reset(); render(); });
  document.getElementById("w2-pr-beta").addEventListener("input", render);
  document.getElementById("w2-pr-tele").addEventListener("change", () => { reset(); render(); });

  U.onThemeChange(render);
  U.lazyBoot("w2-pr-svg", render);
})();
