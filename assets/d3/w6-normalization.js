/* Widget 6.3 — Why the D^{-1/2} A D^{-1/2} normalization?
 * The same hub-and-spokes graph is propagated under three operators:
 *   sum   : h <- (A + I) h                      (no normalization)
 *   mean  : h <- D̃^{-1} Ã h                     (random-walk normalization)
 *   GCN   : h <- D̃^{-1/2} Ã D̃^{-1/2} h          (symmetric normalization)
 * with Ã = A + I. Node color/label shows the scalar feature; a log-scale chart
 * tracks ||h||₂ per operator across steps — sum explodes, the others stay bounded.
 */
(function () {
  "use strict";
  const U = window.DKR;

  const g = U.graph(6, [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [3, 4]]);
  const A = U.adjacency(g);
  const n = g.n;
  const NAMES = ["hub", "b", "c", "d", "e", "f"];
  const x0 = [1.0, 0.8, 0.3, 0.6, 0.2, 0.9];

  // Ã = A + I, D̃ its degree matrix
  const At = A.map((row, i) => row.map((v, j) => (i === j ? 1 : v)));
  const dt = U.degrees(At);
  const OPS = {
    sum: { M: At, label: "sum · (A+I)h", color: "red" },
    mean: { M: At.map((row, i) => row.map((v) => v / dt[i])), label: "mean · D̃⁻¹Ãh", color: "yellow" },
    gcn: { M: At.map((row, i) => row.map((v, j) => v / Math.sqrt(dt[i] * dt[j]))), label: "GCN · D̃⁻½ÃD̃⁻½h", color: "green" },
  };

  let state = { sum: x0.slice(), mean: x0.slice(), gcn: x0.slice() };
  let norms = { sum: [norm(x0)], mean: [norm(x0)], gcn: [norm(x0)] };
  let step = 0;

  function norm(v) { return Math.sqrt(v.reduce((a, b) => a + b * b, 0)); }
  function apply(M, v) { return M.map((row) => row.reduce((a, m, j) => a + m * v[j], 0)); }

  const W = 760, Hgt = 290;
  // three copies of the same layout
  const lay = g.nodes.map((d, i) => {
    if (i === 0) return { x: 0, y: 0 };
    const a = (2 * Math.PI * (i - 1)) / 5 - Math.PI / 2;
    return { x: 62 * Math.cos(a), y: 62 * Math.sin(a) };
  });

  function drawPanel(svg, ox, key, P) {
    const p = svg.append("g").attr("transform", `translate(${ox},95)`);
    const vals = state[key];
    const vmax = Math.max(1e-9, d3.max(vals, (v) => Math.abs(v)));
    p.selectAll("line").data(g.edges).join("line")
      .attr("x1", (e) => lay[e[0]].x).attr("y1", (e) => lay[e[0]].y)
      .attr("x2", (e) => lay[e[1]].x).attr("y2", (e) => lay[e[1]].y)
      .attr("stroke", P.border).attr("stroke-width", 1.3);
    const nd = p.selectAll("g").data(g.nodes).join("g")
      .attr("transform", (d) => `translate(${lay[d.id].x},${lay[d.id].y})`);
    nd.append("circle").attr("r", 16)
      .attr("fill", (d) => U.signedColor(vals[d.id], vmax, P))
      .attr("stroke", P.muted).attr("stroke-width", 1);
    nd.append("text").attr("text-anchor", "middle").attr("dy", 4.5)
      .attr("font-size", 12.5).attr("fill", P.text)
      .text((d) => {
        const v = vals[d.id];
        return Math.abs(v) >= 100 ? v.toExponential(0) : v.toFixed(Math.abs(v) >= 10 ? 0 : 1);
      });
    p.append("text").attr("x", 0).attr("y", -82).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 700).attr("fill", P[OPS[key].color])
      .text(OPS[key].label);
    p.append("text").attr("x", 0).attr("y", 94).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`‖h‖ = ${norms[key][step] >= 1000 ? norms[key][step].toExponential(1) : norms[key][step].toFixed(2)}`);
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w6-nm-svg", W, Hgt);
    drawPanel(svg, 105, "sum", P);
    drawPanel(svg, 315, "mean", P);
    drawPanel(svg, 525, "gcn", P);

    // log-norm chart
    const ch = svg.append("g").attr("transform", "translate(660,60)");
    const maxStep = Math.max(6, step);
    const xs = d3.scaleLinear().domain([0, maxStep]).range([0, 85]);
    const ymaxLog = Math.max(1, Math.log10(d3.max(norms.sum) || 1));
    const ys = d3.scaleLinear().domain([-1, ymaxLog]).range([170, 0]);
    ch.append("line").attr("x1", 0).attr("x2", 85).attr("y1", ys(0)).attr("y2", ys(0))
      .attr("stroke", P.border).attr("stroke-dasharray", "3,3");
    Object.keys(OPS).forEach((key) => {
      const line = d3.line().x((v, i) => xs(i)).y((v) => ys(Math.log10(Math.max(1e-1, v))));
      ch.append("path").attr("d", line(norms[key]))
        .attr("fill", "none").attr("stroke", P[OPS[key].color]).attr("stroke-width", 2);
    });
    ch.append("text").attr("x", 40).attr("y", 195).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text("log₁₀‖h‖ vs step");

    document.getElementById("w6-nm-step-label").textContent = String(step);
  }

  document.getElementById("w6-nm-step").addEventListener("click", () => {
    Object.keys(OPS).forEach((key) => {
      state[key] = apply(OPS[key].M, state[key]);
      norms[key].push(norm(state[key]));
    });
    step += 1; render();
  });
  document.getElementById("w6-nm-reset").addEventListener("click", () => {
    state = { sum: x0.slice(), mean: x0.slice(), gcn: x0.slice() };
    norms = { sum: [norm(x0)], mean: [norm(x0)], gcn: [norm(x0)] };
    step = 0; render();
  });

  U.onThemeChange(render);
  U.lazyBoot("w6-nm-svg", render);
})();
