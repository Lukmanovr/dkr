/* Widget 6.2 — The spectral playground.
 * Three graphs (ring, grid, two communities). Two modes:
 *   eigenvectors — color nodes by the k-th Laplacian eigenvector u_k (slider on k);
 *   filter       — set w0,w1,w2 of p(L) = w0 I + w1 L + w2 L^2, see the frequency
 *                  response p(lambda) over the spectrum and the node-domain result
 *                  p(L)x for a delta signal x placed by clicking a node.
 * All eigendecompositions are computed in-browser (Jacobi, graphs are small).
 */
(function () {
  "use strict";
  const U = window.DKR;

  function ring(n) { return U.graph(n, d3.range(n).map((i) => [i, (i + 1) % n])); }
  function grid(r, c) {
    const e = [];
    for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) {
      const u = i * c + j;
      if (j + 1 < c) e.push([u, u + 1]);
      if (i + 1 < r) e.push([u, u + c]);
    }
    return U.graph(r * c, e);
  }
  function communities() {
    const e = [];
    for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) { e.push([i, j]); e.push([i + 5, j + 5]); }
    e.push([4, 5]);
    return U.graph(10, e);
  }

  const GRAPHS = {
    ring: { g: ring(12), label: "ring (12)" },
    grid: { g: grid(4, 4), label: "grid (4×4)" },
    comm: { g: communities(), label: "two communities" },
  };

  // layouts
  U.circleLayout(GRAPHS.ring.g, 150, 150, 105);
  GRAPHS.grid.g.nodes.forEach((d, i) => { d.x = 60 + (i % 4) * 60; d.y = 60 + Math.floor(i / 4) * 60; });
  GRAPHS.comm.g.nodes.forEach((d, i) => {
    const c = i < 5 ? 0 : 1;
    const a = (2 * Math.PI * (i % 5)) / 5 - Math.PI / 2;
    d.x = (c === 0 ? 85 : 215) + 42 * Math.cos(a);
    d.y = 150 + 42 * Math.sin(a);
  });

  let cur = "comm";
  let mode = "eig";
  let k = 1;
  let delta = 0;
  let w = [1.0, -0.5, 0.0];

  const eigCache = {};
  function eig(key) {
    if (!eigCache[key]) {
      const g = GRAPHS[key].g, A = U.adjacency(g), d = U.degrees(A);
      const L = A.map((row, i) => row.map((v, j) => (i === j ? d[i] : -v)));
      eigCache[key] = { L, ...U.symEig(L) };
    }
    return eigCache[key];
  }

  function applyFilter(key, x) {
    const { L } = eig(key);
    const n = L.length;
    const Lx = U.matmul(L, x.map((v) => [v])).map((r) => r[0]);
    const LLx = U.matmul(L, Lx.map((v) => [v])).map((r) => r[0]);
    return d3.range(n).map((i) => w[0] * x[i] + w[1] * Lx[i] + w[2] * LLx[i]);
  }

  const W = 760, Hgt = 330;

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w6-sp-svg", W, Hgt);
    const g = GRAPHS[cur].g;
    const E = eig(cur);
    const n = g.n;

    // node values for current mode
    let vals, title;
    if (mode === "eig") {
      vals = E.vectors[Math.min(k, n - 1)];
      title = `u_${Math.min(k, n - 1)} · λ = ${E.values[Math.min(k, n - 1)].toFixed(2)}`;
    } else {
      const x = Array(n).fill(0); x[Math.min(delta, n - 1)] = 1;
      vals = applyFilter(cur, x);
      title = `y = p(L)·δ_${Math.min(delta, n - 1)}`;
    }
    const vmax = Math.max(1e-9, d3.max(vals, (v) => Math.abs(v)));

    // --- graph panel ---
    const gp = svg.append("g").attr("transform", "translate(10,5)");
    gp.selectAll("line").data(g.edges).join("line")
      .attr("x1", (e) => g.nodes[e[0]].x).attr("y1", (e) => g.nodes[e[0]].y)
      .attr("x2", (e) => g.nodes[e[1]].x).attr("y2", (e) => g.nodes[e[1]].y)
      .attr("stroke", P.border).attr("stroke-width", 1.5);
    const node = gp.selectAll("g.n").data(g.nodes).join("g")
      .attr("transform", (d) => `translate(${d.x},${d.y})`)
      .attr("tabindex", mode === "filter" ? 0 : null)
      .attr("role", mode === "filter" ? "button" : null)
      .style("cursor", mode === "filter" ? "pointer" : "default")
      .on("click", (ev, d) => { if (mode === "filter") { delta = d.id; render(); } })
      .on("keydown", (ev, d) => {
        if (mode === "filter" && (ev.key === "Enter" || ev.key === " ")) { delta = d.id; render(); ev.preventDefault(); }
      });
    node.append("circle").attr("r", 12)
      .attr("fill", (d) => U.signedColor(vals[d.id], vmax, P))
      .attr("stroke", (d) => (mode === "filter" && d.id === delta) ? P.accentDark : P.muted)
      .attr("stroke-width", (d) => (mode === "filter" && d.id === delta) ? 2.5 : 1);
    node.append("text").attr("text-anchor", "middle").attr("dy", 3.5)
      .attr("font-size", 8.5).attr("fill", P.text)
      .text((d) => vals[d.id].toFixed(1).replace("-0.0", "0.0"));
    gp.append("text").attr("x", 150).attr("y", 315).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", P.muted).text(title);

    // --- spectrum / response panel ---
    const sp = svg.append("g").attr("transform", "translate(360,20)");
    const lmax = Math.max(1e-9, E.values[n - 1]);
    const xs = d3.scaleLinear().domain([0, lmax * 1.05]).range([0, 360]);
    const allResp = E.values.map((l) => w[0] + w[1] * l + w[2] * l * l);
    const respOn = mode === "filter";
    const ymaxCand = respOn ? d3.max(allResp.map(Math.abs)) : 1;
    const ymax = Math.max(1, ymaxCand);
    const ys = d3.scaleLinear().domain([-ymax, ymax]).range([250, 0]);

    sp.append("line").attr("x1", 0).attr("x2", 360).attr("y1", ys(0)).attr("y2", ys(0))
      .attr("stroke", P.border);
    sp.append("text").attr("x", 180).attr("y", 285).attr("text-anchor", "middle")
      .attr("font-size", 11).attr("fill", P.muted)
      .text(respOn ? "frequency response p(λ) over the spectrum" : "the spectrum of L (each dot one eigenvalue)");

    if (respOn) {
      const line = d3.line().x((l) => xs(l)).y((l) => ys(w[0] + w[1] * l + w[2] * l * l));
      const dense = d3.range(0, lmax * 1.05, lmax / 120);
      sp.append("path").attr("d", line(dense))
        .attr("fill", "none").attr("stroke", P.accent).attr("stroke-width", 2);
    }
    sp.selectAll("circle.ev").data(E.values).join("circle")
      .attr("cx", (l) => xs(l))
      .attr("cy", (l) => respOn ? ys(w[0] + w[1] * l + w[2] * l * l) : ys(0))
      .attr("r", 4)
      .attr("fill", (l, i) => (mode === "eig" && i === Math.min(k, n - 1)) ? P.accent : P.blue)
      .attr("opacity", 0.85);
    sp.append("text").attr("x", 0).attr("y", -6).attr("font-size", 11).attr("fill", P.muted)
      .text(respOn ? `p(λ) = ${w[0].toFixed(1)} + ${w[1].toFixed(1)}·λ + ${w[2].toFixed(1)}·λ²` : "λ = 0 (constant) → large λ (oscillatory)");

    // slider visibility
    document.getElementById("w6-sp-eigrow").style.display = mode === "eig" ? "flex" : "none";
    document.getElementById("w6-sp-wrow").style.display = mode === "filter" ? "flex" : "none";
    document.getElementById("w6-sp-k").max = String(n - 1);
    document.getElementById("w6-sp-klabel").textContent = String(Math.min(k, n - 1));
    ["0", "1", "2"].forEach((i) => {
      document.getElementById("w6-sp-w" + i + "label").textContent = w[+i].toFixed(1);
    });
  }

  // controls
  document.querySelectorAll("#w6-sp-widget [data-graph]").forEach((b) =>
    b.addEventListener("click", () => {
      cur = b.getAttribute("data-graph");
      document.querySelectorAll("#w6-sp-widget [data-graph]").forEach((x) => x.classList.toggle("active", x === b));
      delta = 0; render();
    }));
  document.querySelectorAll("#w6-sp-widget [data-mode]").forEach((b) =>
    b.addEventListener("click", () => {
      mode = b.getAttribute("data-mode");
      document.querySelectorAll("#w6-sp-widget [data-mode]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.getElementById("w6-sp-k").addEventListener("input", (e) => { k = +e.target.value; render(); });
  ["0", "1", "2"].forEach((i) =>
    document.getElementById("w6-sp-w" + i).addEventListener("input", (e) => { w[+i] = +e.target.value; render(); }));

  U.onThemeChange(render);
  render();
})();
