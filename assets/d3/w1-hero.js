/* Widget 1.0 — Zachary's karate club, live force layout.
 * The real 1977 dataset: 34 members, 78 friendship ties, colored by the
 * faction each member actually joined after the split. Node 8 wears the
 * red ring: the single member Zachary's min-cut model misplaced (33/34).
 * The simulation is run to convergence synchronously before first paint
 * (deterministic — initial positions are a fixed circle, so shots and lint
 * are stable); dragging a node reheats it live.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const EDGES = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,10],[0,11],
    [0,12],[0,13],[0,17],[0,19],[0,21],[0,31],[1,2],[1,3],[1,7],[1,13],[1,17],
    [1,19],[1,21],[1,30],[2,3],[2,7],[2,8],[2,9],[2,13],[2,27],[2,28],[2,32],
    [3,7],[3,12],[3,13],[4,6],[4,10],[5,6],[5,10],[5,16],[6,16],[8,30],[8,32],
    [8,33],[9,33],[13,33],[14,32],[14,33],[15,32],[15,33],[18,32],[18,33],
    [19,33],[20,32],[20,33],[22,32],[22,33],[23,25],[23,27],[23,29],[23,32],
    [23,33],[24,25],[24,27],[24,31],[25,31],[26,29],[26,33],[27,33],[28,31],
    [28,33],[29,32],[29,33],[30,32],[30,33],[31,32],[31,33],[32,33]];
  const CLUB = [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,1,0,1,1,1,1,1,1,1,1,
    1,1,1,1];
  const MISS = 8;               // sided with Mr. Hi; the model said Officer
  const W = 760, H = 480;

  const nodes = CLUB.map((c, i) => ({
    id: i, club: c,
    // deterministic start: a circle, factions on opposite arcs
    x: W / 2 + 300 * Math.cos((i / 34) * 2 * Math.PI),
    y: H / 2 + 130 * Math.sin((i / 34) * 2 * Math.PI),
  }));
  const links = EDGES.map(([s, t]) => ({ source: s, target: t }));

  const sim = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).distance(62).strength(0.4))
    .force("charge", d3.forceManyBody().strength(-210))
    .force("center", d3.forceCenter(W / 2, H / 2 + 6))
    .force("collide", d3.forceCollide(17))
    .force("x", d3.forceX(W / 2).strength(0.018))
    .force("y", d3.forceY(H / 2).strength(0.10))
    .stop();
  for (let i = 0; i < 300; i++) sim.tick();   // converge before first paint

  let svg, linkSel, nodeSel, labelSel, missLabel;

  function colors() {
    const P = U.pal();
    return { hi: P.yellow, of: P.blue, miss: P.red, muted: P.muted, text: P.text };
  }

  function build() {
    const C = colors();
    svg = U.svgIn("w1-he-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    linkSel = g.append("g").selectAll("line").data(links).join("line")
      .attr("stroke", C.muted).attr("stroke-width", 1.3).attr("opacity", 0.45);

    nodeSel = g.append("g").selectAll("circle").data(nodes).join("circle")
      .attr("r", (d) => (d.id === 0 || d.id === 33 ? 15 : 10))
      .attr("fill", (d) => (d.club === 0 ? C.hi : C.of))
      .attr("stroke", (d) => (d.id === MISS ? C.miss : "none"))
      .attr("stroke-width", 3.5)
      .attr("stroke-dasharray", (d) => (d.id === MISS ? "5,3" : null))
      .style("cursor", "grab")
      .call(d3.drag()
        .on("start", (ev, d) => { sim.alphaTarget(0.25).restart(); d.fx = d.x; d.fy = d.y; })
        .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; })
        .on("end", (ev, d) => { sim.alphaTarget(0); d.fx = null; d.fy = null; }));
    nodeSel.append("title")
      .text((d) => `member ${d.id} — joined ${d.club === 0 ? "Mr. Hi" : "the Officer"}` +
        (d.id === MISS ? " (the one the model missed)" : ""));

    labelSel = g.append("g").selectAll("text")
      .data(nodes.filter((d) => d.id === 0 || d.id === 33)).join("text")
      .attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 700)
      .attr("fill", C.text)
      .attr("stroke", U.pal().bg).attr("stroke-width", 4).attr("paint-order", "stroke")
      .text((d) => (d.id === 0 ? "Mr. Hi (instructor)" : "the Officer (president)"));

    missLabel = g.append("text")
      .attr("text-anchor", "middle").attr("font-size", 12.5)
      .attr("font-weight", 600).attr("fill", C.miss)
      .attr("stroke", U.pal().bg).attr("stroke-width", 4).attr("paint-order", "stroke")
      .text("the one miss");

    g.append("text").attr("x", W / 2).attr("y", 20).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", C.muted)
      .text("friendship = edge · 34 members · 78 ties · drag any node — the layout is a live physical simulation");

    sim.on("tick", position);
    position();
  }

  function position() {
    for (const d of nodes) {
      d.x = Math.max(24, Math.min(W - 24, d.x));
      d.y = Math.max(44, Math.min(H - 20, d.y));
    }
    linkSel.attr("x1", (d) => d.source.x).attr("y1", (d) => d.source.y)
      .attr("x2", (d) => d.target.x).attr("y2", (d) => d.target.y);
    nodeSel.attr("cx", (d) => d.x).attr("cy", (d) => d.y);
    labelSel.attr("x", (d) => d.x)
      .attr("y", (d) => d.y + (d.y > H / 2 ? 36 : -26));
    const m = nodes[MISS];
    missLabel.attr("x", m.x).attr("y", m.y - 18);
  }

  function render() {
    const el = document.getElementById("w1-he-svg");
    if (el) el.innerHTML = "";
    build();
  }

  document.getElementById("w1-he-shuffle").addEventListener("click", () => {
    if (!U.motionOK()) { render(); return; }
    for (const d of nodes) { d.x += (d.id % 7 - 3) * 60; d.y += (d.id % 5 - 2) * 60; }
    sim.alpha(0.9).restart();
  });

  U.onThemeChange(render);
  U.lazyBoot("w1-he-svg", render);
})();
