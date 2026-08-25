/* Widget 5.2 — Queries as boxes: projection, intersection, and why not points.
 * The 2D geometry from the boxes figure, made steppable. Three scenes:
 * the 2-hop path query, an intersection query (two branches meet), and the
 * points-fail demonstration. All containment computed live from coordinates.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const ENT = {
    "Aspirin": [5.2, 5.6], "Ibuprofen": [6.1, 4.9], "Naproxen": [5.7, 6.4],
    "Metformin": [1.5, 1.6], "COX-1": [6.9, 2.3], "COX-2": [4.1, 3.1],
    "AMPK": [1.2, 3.0], "Inflammation": [2.6, 1.0],
  };
  const B1 = { c: [3.9, 2.9], o: [0.9, 0.9], color: "#7c5cd6", label: "assoc⁻¹(Inflammation)" };
  const B2 = { c: [5.7, 5.6], o: [1.3, 1.4], color: "#199473", label: "targets⁻¹(·)" };
  // second branch for the intersection scene: drugs that target COX-1
  const B3 = { c: [5.0, 6.0], o: [0.8, 1.0], color: "#d9a62e", label: "targets⁻¹(COX-1)" };
  const DRUGS = ["Aspirin", "Ibuprofen", "Naproxen", "Metformin"];

  const inside = (p, b) => Math.abs(p[0] - b.c[0]) <= b.o[0] && Math.abs(p[1] - b.c[1]) <= b.o[1];
  function intersect(a, b) {
    const lo = [Math.max(a.c[0] - a.o[0], b.c[0] - b.o[0]), Math.max(a.c[1] - a.o[1], b.c[1] - b.o[1])];
    const hi = [Math.min(a.c[0] + a.o[0], b.c[0] + b.o[0]), Math.min(a.c[1] + a.o[1], b.c[1] + b.o[1])];
    if (lo[0] >= hi[0] || lo[1] >= hi[1]) return null;
    return { c: [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2], o: [(hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2],
             color: "#d9603b", label: "∩" };
  }

  const S = 40;
  const bx = (v) => 70 + v * S;
  const by = (v) => 300 - v * S;

  let scene = "path", step = 0;
  const MAXSTEP = { path: 2, intersect: 3, points: 2 };

  function drawBox(g, b, dash) {
    const x0 = bx(b.c[0] - b.o[0]), y0 = by(b.c[1] + b.o[1]);
    const w = 2 * b.o[0] * S, h = 2 * b.o[1] * S;
    g.append("rect").attr("x", x0).attr("y", y0).attr("width", w).attr("height", h)
      .attr("rx", 8).attr("fill", b.color).attr("opacity", 0.12);
    g.append("rect").attr("x", x0).attr("y", y0).attr("width", w).attr("height", h)
      .attr("rx", 8).attr("fill", "none").attr("stroke", b.color).attr("stroke-width", 2.2)
      .attr("stroke-dasharray", dash ? "6,5" : null);
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w5-bx-svg", 760, 340);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    let boxes = [], statusLines = [];
    if (scene === "path") {
      if (step >= 1) boxes.push(B1);
      if (step >= 2) boxes.push(B2);
      statusLines = [
        "the anchor is a point — press “step” to apply the first projection",
        "① one box: everything assoc-related to Inflammation should fall inside",
        "② project the whole box: the answers are simply “every drug inside” — all three, Naproxen included",
      ];
    } else if (scene === "intersect") {
      if (step >= 1) boxes.push(B2);
      if (step >= 2) boxes.push(B3);
      if (step >= 3) {
        const meet = intersect(B2, B3);
        if (meet) boxes.push(meet);
      }
      statusLines = [
        "intersection query: drugs for Inflammation AND targeting COX-1 — press “step”",
        "branch 1: the Inflammation box (green)",
        "branch 2: the COX-1 box (gold)",
        "∩: shrink to the overlap — boxes are CLOSED under intersection; the meet is again a box",
      ];
    } else {
      statusLines = [
        "same query, but embed it as a POINT — press “step”",
        "the point lands between the three answers — nearest to none of them",
        "a point must choose ONE location; a three-answer query needs a REGION. That is the whole argument.",
      ];
    }

    boxes.forEach((b, i) => drawBox(g, b, false));
    if (scene === "points" && step >= 1) {
      const q = [5.65, 5.65];
      g.append("circle").attr("cx", bx(q[0])).attr("cy", by(q[1])).attr("r", 7)
        .attr("fill", "#d9603b");
      g.append("text").attr("x", bx(q[0])).attr("y", by(q[1]) - 12).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.accentDark).text("q (a point)");
      if (step >= 2) {
        ["Aspirin", "Ibuprofen", "Naproxen"].forEach((d) => {
          g.append("line").attr("x1", bx(q[0])).attr("y1", by(q[1]))
            .attr("x2", bx(ENT[d][0])).attr("y2", by(ENT[d][1]))
            .attr("stroke", P.muted).attr("stroke-width", 1.4).attr("stroke-dasharray", "3,3");
        });
      }
    }

    const active = boxes.length ? boxes[boxes.length - 1] : null;
    Object.entries(ENT).forEach(([name, p]) => {
      const x = bx(p[0]), y = by(p[1]);
      const isDrug = DRUGS.includes(name);
      const inAns = active && isDrug && inside(p, active);
      const anchor = name === "Inflammation";
      if (anchor) {   // the query starts here: keep the anchor visibly special
        g.append("circle").attr("cx", x).attr("cy", y).attr("r", 13)
          .attr("fill", "none").attr("stroke", P.yellow).attr("stroke-width", 1.6)
          .attr("stroke-dasharray", "3,3").attr("opacity", 0.9);
      }
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", inAns || anchor ? 9 : 7)
        .attr("fill", inAns ? P.green : anchor ? P.yellow : P.muted)
        .attr("opacity", inAns || anchor ? 0.95 : 0.6);
      const dy = anchor ? 30 : ["COX-2", "COX-1"].includes(name) ? 24 : -13;
      g.append("text").attr("x", x).attr("y", y + dy).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.text).text(name);
      if (anchor && step === 0) {
        g.append("text").attr("x", x).attr("y", y + 48).attr("text-anchor", "middle")
          .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.yellow)
          .text("anchor — the query starts here");
      }
    });

    if (active && scene !== "points") {
      const contained = DRUGS.filter((d) => inside(ENT[d], active));
      g.append("text").attr("x", 620).attr("y", 40).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.text)
        .text("drugs inside the box:");
      contained.forEach((d, i) => {
        g.append("text").attr("x", 620).attr("y", 62 + i * 20).attr("text-anchor", "middle")
          .attr("font-size", 12.5).attr("fill", P.green).text(d);
      });
      if (!contained.length) {
        g.append("text").attr("x", 620).attr("y", 62).attr("text-anchor", "middle")
          .attr("font-size", 12.5).attr("fill", P.muted).text("—");
      }
    } else if (scene !== "points") {
      g.append("text").attr("x", 620).attr("y", 40).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.muted)
        .text("drugs inside the box:");
      g.append("text").attr("x", 620).attr("y", 62).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("fill", P.muted).text("— none yet: no box —");
    }

    g.append("text").attr("x", 380).attr("y", 328).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.text)
      .text(statusLines[Math.min(step, statusLines.length - 1)]);
  }

  document.querySelectorAll("#w5-bx-widget [data-scene]").forEach((b) =>
    b.addEventListener("click", () => {
      scene = b.getAttribute("data-scene");
      step = 0;
      document.querySelectorAll("#w5-bx-widget [data-scene]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.getElementById("w5-bx-step").addEventListener("click", () => {
    step = Math.min(MAXSTEP[scene], step + 1);
    render();
  });
  document.getElementById("w5-bx-reset").addEventListener("click", () => { step = 0; render(); });

  U.onThemeChange(render);
  U.lazyBoot("w5-bx-svg", render);
})();
