/* Widget 2.3 — Weisfeiler–Lehman color refinement, one round at a time.
 * Two scenes: the cast graph (refines to a stable 5-color partition in two
 * rounds) and the two-triangles-vs-hexagon pair (WL is stuck at one color
 * forever — its famous blind spot). Refinement ids are global and match the
 * palette used in the lecture's WL figure.
 */
(function () {
  "use strict";
  const U = window.DKR;
  // KEEP IN SYNC with WLPAL in scripts/figgen/w2_figs.py (the lecture's WL figure
  // uses the same refinement ids; ids 5-9 are hue-separated for the stable state)
  const WLPAL = ["#8e8e9a", "#d9603b", "#0f8377", "#7c5cd6", "#d9a62e",
                 "#d1567e", "#2e7dd1", "#8a5a33", "#cf4a30", "#199473"];

  const SCENES = {
    cast: {
      n: 6,
      edges: [[0, 1], [0, 2], [0, 3], [1, 2], [2, 4], [2, 5], [4, 5]],
      pos: [[120, 105], [215, 48], [280, 118], [68, 172], [224, 185], [368, 172]],
      names: ["A", "B", "C", "D", "E", "F"],
    },
    rings: {
      n: 12,
      edges: [[0, 1], [1, 2], [2, 0], [3, 4], [4, 5], [5, 3],
              [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 6]],
      pos: [[70, 30], [30, 110], [110, 110], [200, 30], [160, 110], [240, 110],
            [390, 20], [442, 50], [442, 110], [390, 140], [338, 110], [338, 50]],
      names: null,
    },
  };

  let scene = "cast";
  let colors, round, stable, nextId;

  function reset() {
    colors = Array(SCENES[scene].n).fill(0);
    round = 0; stable = false; nextId = 1;
  }
  reset();

  function neighborsOf() {
    const S = SCENES[scene];
    const nb = Array.from({ length: S.n }, () => []);
    S.edges.forEach(([a, b]) => { nb[a].push(b); nb[b].push(a); });
    return nb;
  }

  function refine() {
    if (stable) return;
    const S = SCENES[scene];
    const nb = neighborsOf();
    const sigs = d3.range(S.n).map((i) =>
      colors[i] + "|" + nb[i].map((j) => colors[j]).sort((a, b) => a - b).join(","));
    const seen = new Map();
    const nc = sigs.map((s) => {
      if (!seen.has(s)) seen.set(s, nextId++);
      return seen.get(s);
    });
    const same = d3.range(S.n).every((i) => d3.range(S.n).every((j) =>
      (colors[i] === colors[j]) === (nc[i] === nc[j])));
    if (same) { stable = true; nextId -= seen.size; return; }
    colors = nc;
    round += 1;
  }

  function histo() {
    const h = new Map();
    colors.forEach((c) => h.set(c, (h.get(c) || 0) + 1));
    return [...h.entries()].sort((a, b) => a[0] - b[0]);
  }

  function drawGraph(g, S, ids, P) {
    g.selectAll("line").data(S.edges.filter(([a]) => ids.includes(a))).join("line")
      .attr("x1", (e) => S.pos[e[0]][0]).attr("y1", (e) => S.pos[e[0]][1])
      .attr("x2", (e) => S.pos[e[1]][0]).attr("y2", (e) => S.pos[e[1]][1])
      .attr("stroke", P.muted).attr("stroke-width", 2).attr("opacity", 0.6);
    const nd = g.selectAll("g.n").data(ids).join("g")
      .attr("transform", (i) => `translate(${S.pos[i][0]},${S.pos[i][1]})`);
    nd.append("circle").attr("r", 15).attr("fill", (i) => WLPAL[colors[i] % 10]);
    if (S.names) {
      // dark letters on the shared gray (round-0) color — white fails contrast there
      nd.append("text").attr("text-anchor", "middle").attr("dy", 4.5)
        .attr("font-size", 13).attr("font-weight", 700)
        .attr("fill", (i) => (colors[i] === 0 ? "#1c1c21" : "#fff"))
        .text((i) => S.names[i]);
    }
  }

  function drawHist(g, entries, P, title, axisNote) {
    g.append("text").attr("y", -10).attr("font-size", 12.5).attr("font-weight", 700)
      .attr("fill", P.text).text(title);
    entries.forEach(([c, count], k) => {
      const h = count * 14;
      g.append("rect").attr("x", k * 32).attr("y", 100 - h).attr("width", 22).attr("height", h)
        .attr("rx", 4).attr("fill", WLPAL[c % 10]);
      g.append("text").attr("x", k * 32 + 11).attr("y", 94 - h).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("fill", P.muted).text(count);
    });
    const w = Math.max(entries.length * 32 - 10, 22);
    g.append("line").attr("x1", -4).attr("x2", w + 4).attr("y1", 101).attr("y2", 101)
      .attr("stroke", P.muted).attr("stroke-width", 1);
    if (axisNote) {
      g.append("text").attr("x", -4).attr("y", 116).attr("font-size", 12.5)
        .attr("fill", P.muted).text(axisNote);
    }
  }

  function render() {
    const P = U.pal();
    const S = SCENES[scene];
    const svg = U.svgIn("w2-wl-svg", 760, 300);
    svg.attr("font-family", "'Source Sans 3', sans-serif");

    if (scene === "cast") {
      drawGraph(svg.append("g").attr("transform", "translate(110,30)"), S, d3.range(6), P);
      drawHist(svg.append("g").attr("transform", "translate(580,80)"), histo(), P, "color histogram", "nodes per color");
    } else {
      const g = svg.append("g").attr("transform", "translate(20,40)");
      drawGraph(g, S, d3.range(12), P);
      g.append("text").attr("x", 135).attr("y", 180).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("fill", P.muted).text("two 3-rings");
      g.append("text").attr("x", 390).attr("y", 180).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("fill", P.muted).text("one 6-ring");
      const hl = histo().map(([c]) => [c, colors.slice(0, 6).filter((x) => x === c).length]).filter((e) => e[1]);
      const hr = histo().map(([c]) => [c, colors.slice(6).filter((x) => x === c).length]).filter((e) => e[1]);
      drawHist(svg.append("g").attr("transform", "translate(560,90)"), hl, P, "3-rings");
      drawHist(svg.append("g").attr("transform", "translate(670,90)"), hr, P, "6-ring");
      svg.append("text").attr("x", 640).attr("y", 150).attr("text-anchor", "middle")
        .attr("font-size", 22).attr("font-weight", 800).attr("fill", P.accent).text("=");
    }

    let status;
    if (scene === "cast") {
      status = stable
        ? `stable after round ${round} — ${histo().length} colors: this histogram is the graph's WL fingerprint`
        : `round ${round} · ${histo().length} color${histo().length > 1 ? "s" : ""} — press “refine once”`;
    } else {
      status = stable
        ? "stable immediately: every node is “a degree-2 node whose neighbors are degree-2 nodes”. WL says SAME — you can see it's wrong. Week 9: GNNs inherit exactly this blind spot."
        : "every node starts identical — press “refine once” and watch nothing change";
    }
    svg.append("text").attr("x", 380).attr("y", 288).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600)
      .attr("fill", stable ? (scene === "cast" ? P.green : P.accentDark) : P.text)
      .call((t) => {
        // wrap the long rings verdict onto two lines
        if (status.length <= 90) { t.text(status); return; }
        const cut = status.lastIndexOf(" ", 90);
        t.append("tspan").attr("x", 380).text(status.slice(0, cut));
        t.append("tspan").attr("x", 380).attr("dy", 16).text(status.slice(cut + 1));
        t.attr("y", 272);
      });
  }

  document.querySelectorAll("#w2-wl-widget [data-scene]").forEach((b) =>
    b.addEventListener("click", () => {
      scene = b.getAttribute("data-scene");
      document.querySelectorAll("#w2-wl-widget [data-scene]").forEach((x) => x.classList.toggle("active", x === b));
      reset();
      render();
    }));
  document.getElementById("w2-wl-step").addEventListener("click", () => { refine(); render(); });
  document.getElementById("w2-wl-reset").addEventListener("click", () => { reset(); render(); });

  U.onThemeChange(render);
  U.lazyBoot("w2-wl-svg", render);
})();
