/* Widget 1.1 — Build the graph, watch its representations follow.
 * Click (or keyboard-select) two nodes to toggle the edge between them.
 * Tabs show the same graph as adjacency matrix A, degree matrix D,
 * Laplacian L = D − A, and as adjacency/edge lists — all updating live.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const NAMES = ["A", "B", "C", "D", "E", "F"];
  const COLORS = ["#d9a62e", "#cf4a30", "#199473", "#7c5cd6", "#d1567e", "#0f8377"];
  const CAST = [[0, 1], [0, 2], [0, 3], [1, 2], [2, 4], [2, 5], [4, 5]];
  const POS = [[150, 118], [255, 55], [327, 131], [92, 192], [265, 205], [388, 190]];
  const n = 6;

  let edges = new Set(CAST.map(([a, b]) => `${a}-${b}`));
  let pending = null;
  let tab = "A";

  const key = (a, b) => (a < b ? `${a}-${b}` : `${b}-${a}`);
  function matA() {
    const M = Array.from({ length: n }, () => Array(n).fill(0));
    edges.forEach((k) => { const [a, b] = k.split("-").map(Number); M[a][b] = 1; M[b][a] = 1; });
    return M;
  }

  const W = 760, Hgt = 300;

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w1-bd-svg", W, Hgt);
    const A = matA();
    const deg = A.map((r) => r.reduce((x, y) => x + y, 0));

    // ── graph panel ──
    const gp = svg.append("g").attr("transform", "translate(4,18)");
    const E = [...edges].map((k) => k.split("-").map(Number));
    gp.selectAll("line").data(E).join("line")
      .attr("x1", (e) => POS[e[0]][0]).attr("y1", (e) => POS[e[0]][1])
      .attr("x2", (e) => POS[e[1]][0]).attr("y2", (e) => POS[e[1]][1])
      .attr("stroke", P.muted).attr("stroke-width", 2).attr("opacity", 0.7);
    const nd = gp.selectAll("g.n").data(d3.range(n)).join("g")
      .attr("transform", (i) => `translate(${POS[i][0]},${POS[i][1]})`)
      .attr("tabindex", 0).attr("role", "button")
      .attr("aria-label", (i) => `node ${NAMES[i]}, degree ${deg[i]}`)
      .style("cursor", "pointer")
      .on("click", (ev, i) => pick(i))
      .on("keydown", (ev, i) => {
        if (ev.key === "Enter" || ev.key === " ") { pick(i); ev.preventDefault(); }
      });
    nd.append("circle").attr("r", 24).attr("fill", "transparent");
    nd.append("circle").attr("r", 17)
      .attr("fill", (i) => COLORS[i])
      .attr("stroke", (i) => (i === pending ? P.accentDark : "none"))
      .attr("stroke-width", 3.5)
      .attr("stroke-dasharray", (i) => (i === pending ? "4,3" : null));
    nd.append("text").attr("text-anchor", "middle").attr("dy", 5)
      .attr("font-size", 14).attr("font-weight", 700).attr("fill", "#fff")
      .text((i) => NAMES[i]);
    nd.append("text").attr("x", 0).attr("y", -24).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text((i) => `d=${deg[i]}`);
    gp.append("text").attr("x", 235).attr("y", 262).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", pending === null ? P.muted : P.accentDark)
      .text(pending === null
        ? "click two nodes to toggle the edge between them"
        : `now click a second node to connect/disconnect ${NAMES[pending]} …`);

    // ── representation panel ──
    const rp = svg.append("g").attr("transform", "translate(480,30)");
    const m = [...edges].length;
    if (tab === "A" || tab === "D" || tab === "L") {
      let M, fmt, color, title;
      if (tab === "A") {
        M = A; title = "adjacency matrix A";
        color = (v) => (v ? P.blue : "transparent");
        fmt = (v) => (v ? "1" : "·");
      } else if (tab === "D") {
        M = A.map((r, i) => r.map((_, j) => (i === j ? deg[i] : 0)));
        title = "degree matrix D";
        color = (v, i, j) => (i === j && v > 0 ? P.yellow : "transparent");
        fmt = (v, i, j) => (i === j ? String(v) : "·");
      } else {
        M = A.map((r, i) => r.map((v, j) => (i === j ? deg[i] : -v)));
        title = "Laplacian L = D − A";
        color = (v) => (v > 0 ? P.yellow : v < 0 ? P.blue : "transparent");
        fmt = (v) => (v === 0 ? "·" : String(v));
      }
      U.drawMatrix(rp, M, { cell: 30, color, fmt, textColor: P.text, borderColor: P.border, fontSize: 12.5 });
      d3.range(n).forEach((i) => {
        rp.append("text").attr("x", -9).attr("y", i * 30 + 20).attr("text-anchor", "end")
          .attr("font-size", 12.5).attr("fill", COLORS[i]).attr("font-weight", 700).text(NAMES[i]);
        rp.append("text").attr("x", i * 30 + 15).attr("y", -8).attr("text-anchor", "middle")
          .attr("font-size", 12.5).attr("fill", COLORS[i]).attr("font-weight", 700).text(NAMES[i]);
      });
      rp.append("text").attr("x", 90).attr("y", 212).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.text).text(title);
      if (tab === "L") {
        rp.append("text").attr("x", 90).attr("y", 232).attr("text-anchor", "middle")
          .attr("font-size", 12.5).attr("fill", P.muted).text("row sums = 0 — check it!");
      }
    } else {
      // lists
      const adj = d3.range(n).map((i) =>
        d3.range(n).filter((j) => A[i][j]).map((j) => NAMES[j]).join(", ") || "—");
      d3.range(n).forEach((i) => {
        rp.append("text").attr("x", 0).attr("y", i * 22)
          .attr("font-size", 13.5)
          .call((t) => {
            t.append("tspan").attr("fill", COLORS[i]).attr("font-weight", 700).text(NAMES[i]);
            t.append("tspan").attr("fill", P.text).text(` → ${adj[i]}`);
          });
      });
      const E2 = [...edges].map((k) => k.split("-").map(Number)).sort((x, y) => x[0] - y[0] || x[1] - y[1]);
      rp.append("text").attr("x", 0).attr("y", 158).attr("font-size", 12.5).attr("fill", P.muted).text("edge_index (COO):");
      rp.append("text").attr("x", 0).attr("y", 178).attr("font-size", 12.5)
        .attr("font-family", "'JetBrains Mono', monospace").attr("fill", P.text)
        .text("src " + E2.map((e) => e[0]).join(" "));
      rp.append("text").attr("x", 0).attr("y", 196).attr("font-size", 12.5)
        .attr("font-family", "'JetBrains Mono', monospace").attr("fill", P.text)
        .text("dst " + E2.map((e) => e[1]).join(" "));
    }
    rp.append("text").attr("x", 90).attr("y", 258).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`n = ${n} · m = ${m} edges`);
  }

  function pick(i) {
    if (pending === null) pending = i;
    else if (pending === i) pending = null;
    else {
      const k = key(pending, i);
      edges.has(k) ? edges.delete(k) : edges.add(k);
      pending = null;
    }
    render();
  }

  document.querySelectorAll("#w1-bd-widget [data-tab]").forEach((b) =>
    b.addEventListener("click", () => {
      tab = b.getAttribute("data-tab");
      document.querySelectorAll("#w1-bd-widget [data-tab]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.getElementById("w1-bd-reset").addEventListener("click", () => {
    edges = new Set(CAST.map(([a, b]) => `${a}-${b}`)); pending = null; render();
  });

  U.onThemeChange(render);
  U.lazyBoot("w1-bd-svg", render);
})();
