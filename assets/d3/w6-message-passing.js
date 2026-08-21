/* Widget 6.1 — Message passing, step by step.
 * A fixed 6-node graph carries a 2-d feature vector per node. Each press of
 * "Step" applies one round of self-loop mean aggregation
 *     h_u <- ( h_u + sum_{v in N(u)} h_v ) / (1 + d_u),
 * i.e. the GCN-style propagation with all weights fixed to identity.
 * Clicking a node (or pressing arrow keys) unrolls its 2-layer computation tree.
 * Repeated stepping also previews oversmoothing: watch the vectors converge.
 */
(function () {
  "use strict";
  const U = window.DKR;

  const g = U.graph(6, [[0, 1], [0, 2], [1, 2], [1, 3], [3, 4], [3, 5], [4, 5]]);
  const A = U.adjacency(g);
  const deg = U.degrees(A);
  const H0 = [[1.0, 0.1], [0.1, 1.0], [0.9, 0.9], [0.1, 0.1], [1.0, 0.5], [0.2, 0.8]];
  const NAMES = ["A", "B", "C", "D", "E", "F"];

  let H = H0.map((r) => r.slice());
  let layer = 0;
  let selected = 1;

  function stepOnce(Hin) {
    return Hin.map((h, u) => {
      const acc = h.slice();
      U.neighbors(A, u).forEach((v) => { acc[0] += Hin[v][0]; acc[1] += Hin[v][1]; });
      return [acc[0] / (1 + deg[u]), acc[1] / (1 + deg[u])];
    });
  }

  const W = 760, Hgt = 380;
  U.circleLayout(g, 190, 195, 130);

  function fmt(v) { return v.toFixed(2); }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w6-mp-svg", W, Hgt);

    // --- graph panel ---
    const gp = svg.append("g");
    gp.selectAll("line").data(g.edges).join("line")
      .attr("x1", (e) => g.nodes[e[0]].x).attr("y1", (e) => g.nodes[e[0]].y)
      .attr("x2", (e) => g.nodes[e[1]].x).attr("y2", (e) => g.nodes[e[1]].y)
      .attr("stroke", (e) => (e[0] === selected || e[1] === selected) ? P.accent : P.border)
      .attr("stroke-width", (e) => (e[0] === selected || e[1] === selected) ? 2.5 : 1.5);

    const node = gp.selectAll("g.node").data(g.nodes).join("g")
      .attr("class", "node")
      .attr("transform", (d) => `translate(${d.x},${d.y})`)
      .attr("tabindex", 0)
      .attr("role", "button")
      .attr("aria-label", (d) => `select node ${NAMES[d.id]}`)
      .style("cursor", "pointer")
      .on("click", (ev, d) => { selected = d.id; render(); })
      .on("keydown", (ev, d) => {
        if (ev.key === "Enter" || ev.key === " ") { selected = d.id; render(); ev.preventDefault(); }
      });

    node.append("circle")
      .attr("r", 17)
      .attr("fill", (d) => d.id === selected ? P.accent : P.paper)
      .attr("stroke", (d) => d.id === selected ? P.accentDark : P.blue)
      .attr("stroke-width", 2);
    node.append("circle").attr("r", 24).attr("fill", "transparent");
    node.append("text")
      .attr("text-anchor", "middle").attr("dy", 5)
      .attr("font-size", 13).attr("font-weight", 700)
      .attr("fill", (d) => d.id === selected ? P.paper : P.text)
      .text((d) => NAMES[d.id]);

    // feature vector readout beside each node (two bars + numbers)
    const fx = node.append("g").attr("transform", "translate(21,-12)");
    [0, 1].forEach((k) => {
      fx.append("rect")
        .attr("x", 0).attr("y", k * 12)
        .attr("width", (d) => 3 + 30 * Math.min(1.2, H[d.id][k]))
        .attr("height", 8)
        .attr("rx", 2)
        .attr("fill", k === 0 ? P.blue : P.green)
        .attr("opacity", 0.85);
      fx.append("text")
        .attr("x", (d) => 7 + 30 * Math.min(1.2, H[d.id][k]))
        .attr("y", k * 12 + 7.5)
        .attr("font-size", 9.5).attr("fill", P.muted)
        .text((d) => fmt(H[d.id][k]));
    });

    gp.append("text").attr("x", 190).attr("y", 372).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", P.muted)
      .text(`layer k = ${layer} · h = [dim 1, dim 2]`);

    // --- computation tree panel for the selected node ---
    const tp = svg.append("g").attr("transform", "translate(420,10)");
    tp.append("text").attr("x", 160).attr("y", 12).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("font-weight", 700).attr("fill", P.text)
      .text(`computation tree of ${NAMES[selected]} (2 layers)`);

    const lvl1 = U.neighbors(A, selected);
    const treeNodes = [{ id: selected, x: 160, y: 55, depth: 0 }];
    const span1 = 320 / (lvl1.length + 1);
    lvl1.forEach((v, i) => {
      const x1 = span1 * (i + 1);
      treeNodes.push({ id: v, x: x1, y: 160, depth: 1, parentX: 160, parentY: 55 });
      const lvl2 = U.neighbors(A, v);
      const span2 = span1 / (lvl2.length + 1);
      lvl2.forEach((w2, j) => {
        treeNodes.push({ id: w2, x: x1 - span1 / 2 + span2 * (j + 1), y: 265, depth: 2, parentX: x1, parentY: 160 });
      });
    });

    tp.selectAll("line").data(treeNodes.filter((d) => d.depth > 0)).join("line")
      .attr("x1", (d) => d.parentX).attr("y1", (d) => d.parentY + 12)
      .attr("x2", (d) => d.x).attr("y2", (d) => d.y - 12)
      .attr("stroke", P.border).attr("stroke-width", 1.2);

    const tn = tp.selectAll("g.tn").data(treeNodes).join("g")
      .attr("class", "tn").attr("transform", (d) => `translate(${d.x},${d.y})`);
    tn.append("circle").attr("r", 12)
      .attr("fill", (d) => d.depth === 0 ? P.accent : (d.depth === 1 ? P.blue : P.paper))
      .attr("stroke", P.blueDark).attr("stroke-width", 1.2);
    tn.append("text").attr("text-anchor", "middle").attr("dy", 4)
      .attr("font-size", 10.5).attr("font-weight", 600)
      .attr("fill", (d) => d.depth === 2 ? P.text : P.paper)
      .text((d) => NAMES[d.id]);

    tp.append("text").attr("x", 160).attr("y", 310).attr("text-anchor", "middle")
      .attr("font-size", 10.5).attr("fill", P.muted)
      .text("leaves supply h(0); each level is one AGGREGATE + UPDATE");

    document.getElementById("w6-mp-layer").textContent = String(layer);
  }

  document.getElementById("w6-mp-step").addEventListener("click", () => {
    H = stepOnce(H); layer += 1; render();
  });
  document.getElementById("w6-mp-reset").addEventListener("click", () => {
    H = H0.map((r) => r.slice()); layer = 0; render();
  });
  document.addEventListener("keydown", (ev) => {
    if (!document.getElementById("w6-mp-svg")) return;
    const within = document.activeElement && document.getElementById("w6-mp-widget").contains(document.activeElement);
    if (!within) return;
    if (ev.key === "ArrowRight") { selected = (selected + 1) % g.n; render(); ev.preventDefault(); }
    if (ev.key === "ArrowLeft") { selected = (selected + g.n - 1) % g.n; render(); ev.preventDefault(); }
  });

  U.onThemeChange(render);
  U.lazyBoot("w6-mp-svg", render);
})();
