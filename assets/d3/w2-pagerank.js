/* Widget 2.2 — The random surfer, one step at a time.
 * The same ten-page toy web as the hero figure. Each press of "step" applies one
 * power-iteration update r ← (1−β)·t + β·Mr with the current damping β and
 * teleport vector t (uniform, or all-on-Q for the personalized variant).
 * Node areas track the current rank estimate; the change readout shows
 * convergence happening.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const WEB = {
    Hub: ["Q", "N1", "N2"],
    N1: ["Hub"], N2: ["Hub"], N3: ["Hub"], N4: ["Hub", "V"],
    L1: ["V"], L2: ["V"], L3: ["V"],
    Q: ["Hub"],
    V: ["N3"],
  };
  const NODES = Object.keys(WEB);
  const POS = { Hub: [170, 165], N1: [75, 75], N2: [75, 250], N3: [280, 75], N4: [280, 250],
                Q: [400, 155], V: [455, 268], L1: [370, 315], L2: [432, 328], L3: [540, 315] };
  const n = NODES.length;

  let pr = {}, iter = 0, delta = null;
  function reset() {
    NODES.forEach((u) => { pr[u] = 1 / n; });
    iter = 0; delta = null;
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
    delta = NODES.reduce((a, u) => a + Math.abs(nxt[u] - pr[u]), 0);
    pr = nxt;
    iter += 1;
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w2-pr-svg", 760, 360);
    const g = svg.append("g").attr("transform", "translate(10,4)");

    const defs = svg.append("defs");
    defs.append("marker").attr("id", "w2prArrow").attr("markerWidth", 8).attr("markerHeight", 8)
      .attr("refX", 24).attr("refY", 3).attr("orient", "auto")
      .append("path").attr("d", "M0,0 L7,3 L0,6 Z").attr("fill", P.muted);

    const links = [];
    NODES.forEach((u) => WEB[u].forEach((v) => links.push([u, v])));
    g.selectAll("line").data(links).join("line")
      .attr("x1", (e) => POS[e[0]][0]).attr("y1", (e) => POS[e[0]][1])
      .attr("x2", (e) => POS[e[1]][0]).attr("y2", (e) => POS[e[1]][1])
      .attr("stroke", P.muted).attr("stroke-width", 1.3).attr("opacity", 0.6)
      .attr("marker-end", "url(#w2prArrow)");

    const nd = g.selectAll("g.n").data(NODES).join("g")
      .attr("transform", (u) => `translate(${POS[u][0]},${POS[u][1]})`);
    nd.append("circle")
      .attr("r", (u) => 9 + 85 * pr[u])
      .attr("fill", (u) => (u === "Q" ? P.accent : u === "V" ? P.purple : P.blue))
      .attr("opacity", (u) => (u === "Q" || u === "V" ? 0.95 : u === "Hub" ? 0.8 : 0.55));
    nd.append("text").attr("text-anchor", "middle").attr("dy", 4)
      .attr("font-size", 12).attr("font-weight", 700).attr("fill", "#fff")
      .text((u) => (u === "Q" || u === "V" ? u : u === "Hub" ? "hub" : ""));
    nd.append("text").attr("text-anchor", "middle")
      .attr("y", (u) => -(9 + 85 * pr[u]) - 5)
      .attr("font-size", 12).attr("fill", P.muted)
      .text((u) => pr[u].toFixed(3));

    // right panel: ranked bars of the current estimate
    const panel = g.append("g").attr("transform", "translate(600,26)")
      .attr("font-family", "'Source Sans 3', sans-serif");
    panel.append("text").attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text)
      .text(`after ${iter} step${iter === 1 ? "" : "s"}`);
    const ranked = NODES.slice().sort((a, b) => pr[b] - pr[a]);
    const bmax = pr[ranked[0]];
    ranked.forEach((u, i) => {
      const y = 16 + i * 27;
      panel.append("rect").attr("x", 0).attr("y", y).attr("height", 16).attr("rx", 4)
        .attr("width", 4 + 96 * (pr[u] / bmax))
        .attr("fill", u === "Q" ? P.accent : u === "V" ? P.purple : P.blue)
        .attr("opacity", u === "Q" || u === "V" ? 0.95 : 0.45);
      panel.append("text").attr("x", 8 + 96 * (pr[u] / bmax)).attr("y", y + 12.5)
        .attr("font-size", 12).attr("fill", P.muted).text(u);
    });

    const status = iter === 0
      ? "everyone starts equal at 1/10 = 0.100 — press step"
      : `iteration ${iter} · total change |Δr|₁ = ${delta.toFixed(4)}${delta < 1e-3 ? " — converged (for the eye)" : ""}`;
    g.append("text").attr("x", 290).attr("y", 352).attr("text-anchor", "middle")
      .attr("font-family", "'Source Sans 3', sans-serif").attr("font-size", 13)
      .attr("fill", iter && delta < 1e-3 ? P.green : P.text).attr("font-weight", 600)
      .text(status);

    document.getElementById("w2-pr-betaval").textContent = beta().toFixed(2);
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
