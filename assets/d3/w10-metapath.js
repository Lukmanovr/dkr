/* Widget 10.3 — Metapaths: different lenses, different neighbors.
 * A ten-node academic instance. Pick an author and a metapath; the widget
 * composes the typed hops and highlights exactly which authors become your
 * "neighbors" under that lens. Deterministic BFS over typed edges.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const WRITES = [["a1", "p1"], ["a2", "p1"], ["a2", "p2"], ["a3", "p2"], ["a4", "p3"], ["a5", "p4"]];
  const PUB = [["p1", "KDD"], ["p2", "KDD"], ["p3", "ICML"], ["p4", "ICML"]];
  const CITES = [["p3", "p1"], ["p4", "p2"]];
  const POS = {
    a1: [90, 70], a2: [230, 70], a3: [370, 70], a4: [510, 70], a5: [650, 70],
    p1: [160, 170], p2: [300, 170], p3: [510, 170], p4: [650, 170],
    KDD: [230, 262], ICML: [580, 262],
  };

  const nbrs = (pairs, fwd) => {
    const m = {};
    pairs.forEach(([a, b]) => {
      const [k, v] = fwd ? [a, b] : [b, a];
      (m[k] = m[k] || new Set()).add(v);
    });
    return m;
  };
  const AP = nbrs(WRITES, true), PA = nbrs(WRITES, false);
  const PV = nbrs(PUB, true), VP = nbrs(PUB, false);
  const CITED_BY = nbrs(CITES, false);   // paper -> papers that cite it

  function compose(start, hops) {
    let cur = new Set([start]);
    hops.forEach((m) => {
      const nxt = new Set();
      cur.forEach((v) => (m[v] || new Set()).forEach((u) => nxt.add(u)));
      cur = nxt;
    });
    cur.delete(start);
    return cur;
  }

  const METAPATHS = {
    apa: { hops: [AP, PA], label: "A–P–A · co-authors" },
    apvpa: { hops: [AP, PV, VP, PA], label: "A–P–V–P–A · same-venue authors" },
    cite: { hops: [AP, CITED_BY, PA], label: "A–P–cited-by–P–A · authors citing my work" },
  };

  let author = "a1", mp = "apa";

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w10-mp-svg", 760, 342);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const found = compose(author, METAPATHS[mp].hops);
    const OFF = 14;   // vertical offset applied to every node position

    g.append("text").attr("x", 380).attr("y", 22).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`${METAPATHS[mp].label} — who counts as ${author}'s neighborhood under this lens?`);

    const eStyle = (rel) => rel === "w" ? P.green : rel === "p" ? P.accent : "#7c5cd6";
    [[WRITES, "w"], [PUB, "p"], [CITES, "c"]].forEach(([pairs, rel]) =>
      pairs.forEach(([a, b]) => {
        const [x1, y1] = POS[a], [x2, y2] = POS[b];
        g.append("line").attr("x1", x1).attr("y1", y1 + OFF).attr("x2", x2).attr("y2", y2 + OFF)
          .attr("stroke", eStyle(rel)).attr("stroke-width", 1.8)
          .attr("stroke-dasharray", rel === "c" ? "5 3" : null).attr("opacity", 0.45);
      }));
    Object.keys(POS).forEach((v) => {
      const [x, y] = POS[v];
      const isA = v[0] === "a", isV = v === "KDD" || v === "ICML";
      const base = isA ? P.yellow : isV ? P.accent : P.blue;
      const hot = v === author || found.has(v);
      const r = isV ? 22 : 14;
      g.append("circle").attr("cx", x).attr("cy", y + OFF).attr("r", v === author ? 18 : r)
        .attr("fill", base).attr("opacity", hot ? 1 : 0.35)
        .attr("stroke", v === author ? P.text : found.has(v) ? P.accentDark : "none")
        .attr("stroke-width", found.has(v) ? 3 : 2);
      // dimmed circles get a pale fill — keep their labels dark (theme text),
      // full-strength fills keep the background-colored label
      g.append("text").attr("x", x).attr("y", y + OFF + 4).attr("text-anchor", "middle")
        .attr("font-size", 13).attr("font-weight", 700)
        .attr("fill", hot ? P.bg : P.text).text(v);
    });

    // result line lives BELOW the drawing's bounding box — never through it
    const list = [...found].sort().join(", ") || "nobody";
    g.append("text").attr("x", 380).attr("y", 330).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.accentDark)
      .text(`${author} + ${METAPATHS[mp].label.split("·")[0].trim()} → { ${list} } — a different graph from the same data`);
  }

  document.querySelectorAll("#w10-mp-widget [data-author]").forEach((b) =>
    b.addEventListener("click", () => {
      author = b.getAttribute("data-author");
      document.querySelectorAll("#w10-mp-widget [data-author]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.querySelectorAll("#w10-mp-widget [data-mp]").forEach((b) =>
    b.addEventListener("click", () => {
      mp = b.getAttribute("data-mp");
      document.querySelectorAll("#w10-mp-widget [data-mp]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w10-mp-svg", render);
})();
