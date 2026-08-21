/* Widget 1.3 — The same little world, five graph types.
 * Four people and two things they like; switching the type re-draws the SAME
 * relationships under a different modeling choice: undirected friendships,
 * directed follows, weighted hours, bipartite likes, full heterogeneous graph.
 */
(function () {
  "use strict";
  const U = window.DKR;

  const PEOPLE = ["Ann", "Bo", "Cy", "Di"];
  const ITEMS = ["Solaris (film)", "Kino (band)"];
  const PCOLOR = ["#d9a62e", "#cf4a30", "#199473", "#7c5cd6"];
  const ICOLOR = ["#d1567e", "#0f8377"];

  const MODES = {
    undirected: {
      label: "friendships are symmetric: one undirected edge per pair — the default simple graph.",
      pos: { p: [[190, 70], [420, 60], [160, 210], [430, 215]] },
      edges: [[0, 1], [0, 2], [1, 3], [2, 3], [0, 3]],
    },
    directed: {
      label: "“follows” is not symmetric: Ann follows Bo, Bo does not follow back — edges get direction.",
      pos: { p: [[190, 70], [420, 60], [160, 210], [430, 215]] },
      arcs: [[0, 1], [2, 0], [3, 1], [3, 2], [1, 3]],
    },
    weighted: {
      label: "hours talked per week become edge weights — same structure, quantified.",
      pos: { p: [[190, 70], [420, 60], [160, 210], [430, 215]] },
      wedges: [[0, 1, 6], [0, 2, 1], [1, 3, 3], [2, 3, 8], [0, 3, 0.5]],
    },
    bipartite: {
      label: "people and things they like: edges only ever cross between the two sets — bipartite, the recommender-system shape.",
      pos: { p: [[150, 55], [150, 120], [150, 185], [150, 250]], i: [[470, 95], [470, 210]] },
      likes: [[0, 0], [1, 0], [1, 1], [2, 1], [3, 0], [3, 1]],
    },
    hetero: {
      label: "everything at once: two node types, two edge types — a heterogeneous graph, one step from a knowledge graph.",
      pos: { p: [[165, 65], [390, 55], [150, 225], [400, 235]], i: [[560, 140], [280, 145]] },
      edges: [[0, 1], [2, 3], [0, 2]],
      likes: [[1, 0], [3, 0], [2, 1], [0, 1]],
    },
  };

  let mode = "undirected";
  const W = 760, Hgt = 300;

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w1-ty-svg", W, Hgt);
    svg.append("defs").append("marker").attr("id", "tyArrow")
      .attr("markerWidth", 9).attr("markerHeight", 9).attr("refX", 24).attr("refY", 3.5).attr("orient", "auto")
      .append("path").attr("d", "M0,0 L8,3.5 L0,7 Z").attr("fill", P.muted);
    const M = MODES[mode];
    const g = svg.append("g").attr("transform", "translate(30,4)");

    const pxy = (i) => M.pos.p[i];
    const ixy = (i) => M.pos.i[i];

    if (M.edges) g.selectAll("line.f").data(M.edges).join("line")
      .attr("x1", (e) => pxy(e[0])[0]).attr("y1", (e) => pxy(e[0])[1])
      .attr("x2", (e) => pxy(e[1])[0]).attr("y2", (e) => pxy(e[1])[1])
      .attr("stroke", P.muted).attr("stroke-width", 2.2).attr("opacity", 0.75);
    if (M.arcs) g.selectAll("line.a").data(M.arcs).join("line")
      .attr("x1", (e) => pxy(e[0])[0]).attr("y1", (e) => pxy(e[0])[1])
      .attr("x2", (e) => pxy(e[1])[0]).attr("y2", (e) => pxy(e[1])[1])
      .attr("stroke", P.muted).attr("stroke-width", 2.2).attr("opacity", 0.85)
      .attr("marker-end", "url(#tyArrow)");
    if (M.wedges) {
      g.selectAll("line.w").data(M.wedges).join("line")
        .attr("x1", (e) => pxy(e[0])[0]).attr("y1", (e) => pxy(e[0])[1])
        .attr("x2", (e) => pxy(e[1])[0]).attr("y2", (e) => pxy(e[1])[1])
        .attr("stroke", P.blue).attr("stroke-width", (e) => 1 + e[2]).attr("opacity", 0.7);
      g.selectAll("text.wl").data(M.wedges).join("text")
        .attr("x", (e) => (pxy(e[0])[0] + pxy(e[1])[0]) / 2 + 8)
        .attr("y", (e) => (pxy(e[0])[1] + pxy(e[1])[1]) / 2 - 6)
        .attr("font-size", 12).attr("fill", P.blueDark).attr("font-weight", 600)
        .text((e) => e[2] + "h");
    }
    if (M.likes) g.selectAll("line.l").data(M.likes).join("line")
      .attr("x1", (e) => pxy(e[0])[0]).attr("y1", (e) => pxy(e[0])[1])
      .attr("x2", (e) => ixy(e[1])[0]).attr("y2", (e) => ixy(e[1])[1])
      .attr("stroke", P.accent).attr("stroke-width", 2).attr("opacity", 0.65)
      .attr("stroke-dasharray", mode === "hetero" ? "5,4" : null);

    // people
    const pp = g.selectAll("g.p").data(d3.range(4)).join("g")
      .attr("transform", (i) => `translate(${pxy(i)[0]},${pxy(i)[1]})`);
    pp.append("circle").attr("r", 17).attr("fill", (i) => PCOLOR[i]);
    pp.append("text").attr("text-anchor", "middle").attr("dy", 4.5)
      .attr("font-size", 12).attr("font-weight", 700).attr("fill", "#fff").text((i) => PEOPLE[i]);
    // items
    if (M.pos.i) {
      const ii = g.selectAll("g.i").data(d3.range(2)).join("g")
        .attr("transform", (i) => `translate(${ixy(i)[0]},${ixy(i)[1]})`);
      ii.append("rect").attr("x", -16).attr("y", -16).attr("width", 32).attr("height", 32).attr("rx", 7)
        .attr("fill", (i) => ICOLOR[i]);
      ii.append("text").attr("text-anchor", "middle").attr("y", 34)
        .attr("font-size", 11.5).attr("fill", P.muted).text((i) => ITEMS[i]);
    }
    if (mode === "hetero") {
      const lg = g.append("g").attr("transform", "translate(520,238)")
        .attr("font-size", 11.5);
      lg.append("line").attr("x1", 0).attr("x2", 26).attr("y1", 0).attr("y2", 0)
        .attr("stroke", P.muted).attr("stroke-width", 2.2);
      lg.append("text").attr("x", 32).attr("y", 4).attr("fill", P.muted).text("friend_of");
      lg.append("line").attr("x1", 0).attr("x2", 26).attr("y1", 20).attr("y2", 20)
        .attr("stroke", P.accent).attr("stroke-width", 2).attr("stroke-dasharray", "5,4");
      lg.append("text").attr("x", 32).attr("y", 24).attr("fill", P.muted).text("likes");
    }
    document.getElementById("w1-ty-caption").textContent = M.label;
  }

  document.querySelectorAll("#w1-ty-widget [data-mode]").forEach((b) =>
    b.addEventListener("click", () => {
      mode = b.getAttribute("data-mode");
      document.querySelectorAll("#w1-ty-widget [data-mode]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w1-ty-svg", render);
})();
