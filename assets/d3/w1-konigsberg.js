/* Widget 1.6 — Königsberg, 1736: the founding modeling act.
 * City view: the real Merian-Erben engraving of 1652 (public domain, via
 * Wikimedia Commons). Graph view: Euler's abstraction drawn OVER a ghost of
 * the same engraving — four nodes pinned to the actual landmasses, seven
 * multigraph edges, degrees 3, 3, 5, 3. Deterministic; city view at boot.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 442;
  const IMG = { x: 115, y: 14, w: 530, h: 369 };
  // node anchors over the engraving's landmasses (canvas coords)
  const PT = { A: [368, 140], B: [330, 352], C: [478, 278], D: [592, 300] };
  const DEG = { A: 3, B: 3, C: 5, D: 3 };
  const NAME = { A: "north bank", B: "south bank", C: "the island", D: "east land" };
  const BR = [["C", "A", -24], ["C", "A", 26], ["C", "B", -26], ["C", "B", 24],
              ["C", "D", 0], ["A", "D", 30], ["B", "D", -30]];
  let view = "city";

  const path = (a, b, bend) => {
    const [x1, y1] = PT[a], [x2, y2] = PT[b];
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy);
    return `M ${x1} ${y1} Q ${mx - (dy / L) * bend} ${my + (dx / L) * bend} ${x2} ${y2}`;
  };

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w1-kb-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const isCity = view === "city";

    g.append("image")
      .attr("href", "../assets/images/koenigsberg-1652.jpg")
      .attr("x", IMG.x).attr("y", IMG.y).attr("width", IMG.w).attr("height", IMG.h)
      .attr("opacity", isCity ? 1 : 0.18)
      .attr("preserveAspectRatio", "xMidYMid meet");

    if (!isCity) {
      for (const [a, b, bend] of BR) {
        g.append("path").attr("d", path(a, b, bend)).attr("fill", "none")
          .attr("stroke", P.blueDark).attr("stroke-width", 2.4).attr("opacity", 0.85);
      }
      for (const k of Object.keys(PT)) {
        const [x, y] = PT[k];
        g.append("circle").attr("cx", x).attr("cy", y).attr("r", 17)
          .attr("fill", P.yellow).attr("stroke", P.accentDark).attr("stroke-width", 2);
        g.append("text").attr("x", x).attr("y", y + 5).attr("text-anchor", "middle")
          .attr("font-size", 14).attr("font-weight", 700).attr("fill", "#fff").text(k);
        g.append("text").attr("x", x).attr("y", y - 25).attr("text-anchor", "middle")
          .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
          .attr("font-weight", 700).attr("fill", P.accentDark)
          .attr("stroke", P.bg).attr("stroke-width", 4).attr("paint-order", "stroke")
          .text("deg " + DEG[k]);
      }
      g.append("text").attr("x", W / 2).attr("y", H - 32).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
        .text("degrees 3, 3, 5, 3 — FOUR odd landmasses; Euler: the walk is impossible");
    } else {
      g.append("text").attr("x", W / 2).attr("y", H - 32).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
        .text("seven bridges, one question: can a walk cross each exactly once?");
    }

    g.append("text").attr("x", W / 2).attr("y", 10).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(isCity
        ? "Königsberg — the city itself, as the map records it"
        : "the same city with the noise deleted: meeting points and connections only");
    g.append("text").attr("x", W / 2).attr("y", H - 8).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("engraving: Merian-Erben, 1652 — public domain, via Wikimedia Commons");
  }

  for (const b of document.querySelectorAll("#w1-kb-widget [data-kb]")) {
    b.addEventListener("click", () => {
      view = b.dataset.kb;
      for (const o of document.querySelectorAll("#w1-kb-widget [data-kb]")) {
        o.classList.toggle("active", o === b);
      }
      render();
    });
  }
  U.onThemeChange(render);
  U.lazyBoot("w1-kb-svg", render);
})();
