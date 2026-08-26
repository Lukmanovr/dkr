/* Widget 1.5 — The multigraph collapse, and what it throws away.
 * Two transfer histories between accounts A and B, both 24 transfers in one
 * year: "steady" (rent-like, evenly spaced) and "burst" (mule-like, packed
 * into three days). Collapsed to a weighted edge, both become the identical
 * w = 24 — the timeline underneath keeps the story the collapse destroys.
 * Deterministic timestamps; identity view at boot is the multigraph.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 320;
  const N = 24;
  // day-of-year for each transfer (deterministic)
  const STEADY = Array.from({ length: N }, (_, i) => 8 + i * 15 + (i % 3) * 2);
  const BURST = [22, 61, 118, 170].concat(
    Array.from({ length: 20 }, (_, i) => 283 + (i % 5) + Math.floor(i / 5)));
  let scen = "steady";
  let view = "multi";

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w1-mg-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const days = scen === "steady" ? STEADY : BURST;

    // ── the pair of accounts ──
    const ax = 230, bx = 530, ny = 92;
    if (view === "multi") {
      days.forEach((d, i) => {
        const bend = (i - (N - 1) / 2) * 5.2;
        g.append("path")
          .attr("d", `M ${ax + 20} ${ny} Q ${(ax + bx) / 2} ${ny + bend} ${bx - 20} ${ny}`)
          .attr("fill", "none").attr("stroke", P.blue)
          .attr("stroke-width", 1.2).attr("opacity", 0.55);
      });
      g.append("text").attr("x", (ax + bx) / 2).attr("y", ny - 72)
        .attr("text-anchor", "middle").attr("font-size", 12.5).attr("fill", P.text)
        .text("24 separate edges — one per transfer, each with its own moment");
    } else {
      g.append("line").attr("x1", ax + 20).attr("y1", ny).attr("x2", bx - 20)
        .attr("y2", ny).attr("stroke", P.blue).attr("stroke-width", 7)
        .attr("opacity", 0.85);
      g.append("text").attr("x", (ax + bx) / 2).attr("y", ny - 14)
        .attr("text-anchor", "middle")
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 14)
        .attr("font-weight", 700).attr("fill", P.text).text("w = 24");
      g.append("text").attr("x", (ax + bx) / 2).attr("y", ny - 72)
        .attr("text-anchor", "middle").attr("font-size", 12.5).attr("fill", P.text)
        .text("one weighted edge — identical for BOTH histories");
    }
    for (const [x, nm] of [[ax, "A"], [bx, "B"]]) {
      g.append("circle").attr("cx", x).attr("cy", ny).attr("r", 19)
        .attr("fill", nm === "A" ? P.yellow : P.blue);
      g.append("text").attr("x", x).attr("y", ny + 5).attr("text-anchor", "middle")
        .attr("font-size", 14).attr("font-weight", 700).attr("fill", "#fff").text(nm);
    }

    // ── the timeline the collapse forgets ──
    const ty = 218, tx0 = 80, tx1 = 700;
    g.append("line").attr("x1", tx0).attr("y1", ty).attr("x2", tx1).attr("y2", ty)
      .attr("stroke", P.muted).attr("stroke-width", 1.5);
    ["Jan", "Jul", "Dec"].forEach((mn, i) => {
      const x = tx0 + (i / 2) * (tx1 - tx0);
      g.append("text").attr("x", x).attr("y", ty + 22).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("fill", P.muted).text(mn);
    });
    days.forEach((d) => {
      const x = tx0 + (d / 365) * (tx1 - tx0);
      g.append("circle").attr("cx", x).attr("cy", ty).attr("r", 4.5)
        .attr("fill", P.accent).attr("opacity", 0.8);
    });
    g.append("text").attr("x", tx0).attr("y", ty - 32).attr("font-size", 12.5)
      .attr("font-weight", 700).attr("fill", P.accentDark)
      .text("the timeline the weight forgets:");
    // Week 14's wall
    const wx = tx0 + 0.8 * (tx1 - tx0);
    g.append("line").attr("x1", wx).attr("y1", ty - 26).attr("x2", wx)
      .attr("y2", ty + 10).attr("stroke", P.accent).attr("stroke-width", 2)
      .attr("stroke-dasharray", "5,4");
    g.append("text").attr("x", wx).attr("y", ty - 32).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.accent).text("Week 14 splits HERE");

    const verdict = view === "multi"
      ? (scen === "steady"
        ? "evenly spaced all year — this pattern says rent, salary, subscription"
        : "20 of 24 packed into three October days — this pattern says investigate")
      : "steady or burst, the collapsed graph is the same picture: the pattern is gone";
    g.append("text").attr("x", W / 2).attr("y", H - 12).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
      .text(verdict);
  }

  for (const b of document.querySelectorAll("#w1-mg-widget [data-scen]")) {
    b.addEventListener("click", () => {
      scen = b.dataset.scen;
      for (const o of document.querySelectorAll("#w1-mg-widget [data-scen]")) {
        o.classList.toggle("active", o === b);
      }
      render();
    });
  }
  for (const b of document.querySelectorAll("#w1-mg-widget [data-view]")) {
    b.addEventListener("click", () => {
      view = b.dataset.view;
      for (const o of document.querySelectorAll("#w1-mg-widget [data-view]")) {
        o.classList.toggle("active", o === b);
      }
      render();
    });
  }
  U.onThemeChange(render);
  U.lazyBoot("w1-mg-svg", render);
})();
