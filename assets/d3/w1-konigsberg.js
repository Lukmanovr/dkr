/* Widget 1.6 — Königsberg, 1736: the founding modeling act, in three stages.
 * (1) The real Merian-Erben engraving of 1652 (public domain, Wikimedia
 * Commons). (2) A clean AI-generated illustration of the same city (Gemini
 * "Nano Banana", 2026 — labeled as such). (3) Euler's abstraction drawn over
 * a ghost of the illustration: four nodes pinned to the landmasses, seven
 * multigraph edges, degrees 3, 3, 5, 3. Deterministic; illustration at boot.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 470;
  const ILLO = { href: "../assets/images/koenigsberg-illustration.jpg",
                 x: 20, y: 16, w: 720, h: 402 };
  const ENGR = { href: "../assets/images/koenigsberg-1652.jpg",
                 x: 100, y: 16, w: 560, h: 390 };
  // node anchors over the ILLUSTRATION's landmasses (canvas coords)
  const PT = { A: [360, 62], B: [344, 358], C: [334, 214], D: [674, 200] };
  const DEG = { A: 3, B: 3, C: 5, D: 3 };
  const BR = [["C", "A", -26], ["C", "A", 28], ["C", "B", -28], ["C", "B", 26],
              ["C", "D", 0], ["A", "D", 34], ["B", "D", -34]];
  let view = "illo";

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
    const img = view === "engraving" ? ENGR : ILLO;

    g.append("image").attr("href", img.href)
      .attr("x", img.x).attr("y", img.y).attr("width", img.w).attr("height", img.h)
      .attr("opacity", view === "graph" ? 0.16 : 1)
      .attr("preserveAspectRatio", "xMidYMid meet");

    if (view === "graph") {
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
    }

    const line = view === "graph"
      ? "degrees 3, 3, 5, 3 — FOUR odd landmasses; Euler: the walk is impossible"
      : "seven bridges, one question: can a walk cross each exactly once?";
    g.append("text").attr("x", W / 2).attr("y", H - 34).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
      .text(line);

    const attr = view === "engraving"
      ? "engraving: Merian-Erben, 1652 — public domain, via Wikimedia Commons"
      : "illustration — the seven bridges in their historical arrangement";
    g.append("text").attr("x", W / 2).attr("y", H - 12).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text(attr);
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
