/* Widget 1.4 — Three levels of asking.
 * The same cast graph, three prediction levels: node-level (classify D),
 * edge-level (score candidate links), graph-level (one verdict per graph).
 */
(function () {
  "use strict";
  const U = window.DKR;
  const NAMES = ["A", "B", "C", "D", "E", "F"];
  const COLORS = ["#d9a62e", "#cf4a30", "#199473", "#7c5cd6", "#d1567e", "#0f8377"];
  const EDGES = [[0, 1], [0, 2], [0, 3], [1, 2], [2, 4], [2, 5], [4, 5]];
  const POS = [[210, 120], [330, 55], [420, 132], [140, 205], [345, 212], [520, 200]];

  const INFO = {
    node: {
      title: "node-level — attach a label to every node",
      body: "two labels are known (green = genuine, red = bot); predict the rest. Industrial form: fraud and integrity scoring, topic tagging.",
    },
    edge: {
      title: "edge-level — score pairs that are NOT yet connected",
      body: "should B–F or A–E exist? Industrial form: friend/product recommendation, knowledge-graph completion (Weeks 4–5, 12).",
    },
    graph: {
      title: "graph-level — one prediction for a whole graph",
      body: "each graph in a collection gets a verdict. Industrial form: molecule property prediction, malware call-graph screening (Week 13's lab).",
    },
  };

  let mode = "node";
  const W = 760, Hgt = 290;

  function baseGraph(g, P, opts = {}) {
    g.selectAll("line.e").data(EDGES).join("line")
      .attr("x1", (e) => POS[e[0]][0]).attr("y1", (e) => POS[e[0]][1])
      .attr("x2", (e) => POS[e[1]][0]).attr("y2", (e) => POS[e[1]][1])
      .attr("stroke", P.muted).attr("stroke-width", 2).attr("opacity", 0.7);
    const nd = g.selectAll("g.n").data(d3.range(6)).join("g")
      .attr("transform", (i) => `translate(${POS[i][0]},${POS[i][1]})`);
    nd.append("circle").attr("r", 16)
      .attr("fill", (i) => (opts.fill ? opts.fill(i) : COLORS[i]))
      .attr("stroke", (i) => (opts.stroke ? opts.stroke(i) : "none"))
      .attr("stroke-width", 2.5);
    nd.append("text").attr("text-anchor", "middle").attr("dy", 4.5)
      .attr("font-size", 13).attr("font-weight", 700)
      .attr("fill", (i) => (opts.textFill ? opts.textFill(i) : "#fff"))
      .text((i) => (opts.text ? opts.text(i) : NAMES[i]));
    return nd;
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w1-tk-svg", W, Hgt);
    const g = svg.append("g").attr("transform", "translate(30,16)");

    if (mode === "node") {
      const known = { 1: P.green, 4: P.red };
      baseGraph(g, P, {
        fill: (i) => known[i] || P.paper,
        stroke: (i) => (known[i] ? "none" : i === 3 ? P.accentDark : P.muted),
        text: (i) => (known[i] ? NAMES[i] : "?"),
        textFill: (i) => (known[i] ? "#fff" : P.text),
      });
      g.append("text").attr("x", 140).attr("y", 168).attr("text-anchor", "middle")
        .attr("font-size", 12).attr("font-weight", 700).attr("fill", P.accentDark)
        .text("classify D");
    } else if (mode === "edge") {
      baseGraph(g, P, {});
      const cand = [[1, 5], [0, 4]];
      g.selectAll("line.c").data(cand).join("line")
        .attr("x1", (e) => POS[e[0]][0]).attr("y1", (e) => POS[e[0]][1])
        .attr("x2", (e) => POS[e[1]][0]).attr("y2", (e) => POS[e[1]][1])
        .attr("stroke", P.accent).attr("stroke-width", 2.4).attr("stroke-dasharray", "6,5");
      g.selectAll("g.q").data(cand).join("g")
        .attr("transform", (e) => `translate(${(POS[e[0]][0] + POS[e[1]][0]) / 2},${(POS[e[0]][1] + POS[e[1]][1]) / 2})`)
        .call((q) => {
          q.append("circle").attr("r", 11).attr("fill", P.paper).attr("stroke", P.accent).attr("stroke-width", 2);
          q.append("text").attr("text-anchor", "middle").attr("dy", 4)
            .attr("font-size", 12).attr("font-weight", 700).attr("fill", P.accentDark).text("?");
        });
    } else {
      // graph-level: two boxed mini-graphs, one verdict each
      const mk = (ox, edges, verdict, good) => {
        const bx = g.append("g").attr("transform", `translate(${ox},18)`);
        bx.append("rect").attr("x", -20).attr("y", -14).attr("width", 260).attr("height", 190).attr("rx", 12)
          .attr("fill", "none").attr("stroke", P.border).attr("stroke-width", 1.4);
        const pos = [[50, 40], [160, 26], [200, 96], [40, 120], [120, 140]];
        bx.selectAll("line").data(edges).join("line")
          .attr("x1", (e) => pos[e[0]][0]).attr("y1", (e) => pos[e[0]][1])
          .attr("x2", (e) => pos[e[1]][0]).attr("y2", (e) => pos[e[1]][1])
          .attr("stroke", P.muted).attr("stroke-width", 1.8).attr("opacity", 0.7);
        bx.selectAll("circle").data(pos).join("circle")
          .attr("cx", (p) => p[0]).attr("cy", (p) => p[1]).attr("r", 10)
          .attr("fill", P.blue);
        bx.append("rect").attr("x", 44).attr("y", 148).attr("width", 132).attr("height", 24).attr("rx", 12)
          .attr("fill", good ? P.green : P.red);
        bx.append("text").attr("x", 110).attr("y", 164).attr("text-anchor", "middle")
          .attr("font-size", 12).attr("font-weight", 700).attr("fill", "#fff").text(verdict);
      };
      mk(30, [[0, 1], [1, 2], [2, 4], [4, 3], [3, 0], [0, 4]], "verdict: safe ✓", true);
      mk(390, [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2]], "verdict: toxic ✗", false);
    }

    document.getElementById("w1-tk-title").textContent = INFO[mode].title;
    document.getElementById("w1-tk-body").textContent = INFO[mode].body;
  }

  document.querySelectorAll("#w1-tk-widget [data-mode]").forEach((b) =>
    b.addEventListener("click", () => {
      mode = b.getAttribute("data-mode");
      document.querySelectorAll("#w1-tk-widget [data-mode]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w1-tk-svg", render);
})();
