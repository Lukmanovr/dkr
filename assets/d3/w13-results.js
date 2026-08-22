/* Widget 13.4 — The ZINC ladder, measured, with the budget lens.
 * Numbers from scratchpad/w13_experiment.py (ZINC-subset, 100 epochs,
 * cosine LR, seed 0, course GPU, 2026-08-22). Toggle the reference to see
 * what ~10x tuned schedules report — budget moves everything.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const ROWS = [
    ["MPNN-only (GINE ×4)", 0.349, 188, "teal"],
    ["MPNN + RWSE", 0.296, 194, "green"],
    ["GPS-lite (∥ attention + RWSE)", 0.303, 529, "purple"],
  ];
  let ref = false;

  function render() {
    const P = U.pal();
    const colOf = { teal: P.blue, green: P.green, purple: "#7c5cd6" };
    const svg = U.svgIn("w13-re-svg", 760, 300);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    g.append("text").attr("x", 380).attr("y", 22).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("ZINC-subset test MAE (lower is better) · 100 epochs each · wall-clock shown");

    const lo = ref ? 0.05 : 0.28;
    const scale = (m) => 480 * (0.42 - m) / (0.42 - lo);
    ROWS.forEach(([name, mae, secs, ck], i) => {
      const yy = 62 + i * 52;
      const w = Math.max(6, scale(mae));
      g.append("text").attr("x", 240).attr("y", yy + 4).attr("text-anchor", "end")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", colOf[ck]).text(name);
      g.append("text").attr("x", 240).attr("y", yy + 21).attr("text-anchor", "end")
        .attr("font-size", 12).attr("fill", P.muted).text(secs + " s");
      g.append("rect").attr("x", 252).attr("y", yy - 8).attr("width", w).attr("height", 22)
        .attr("rx", 6).attr("fill", colOf[ck]).attr("opacity", 0.85);
      g.append("text").attr("x", 252 + w + 8).attr("y", yy + 8)
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
        .attr("font-weight", 700).attr("fill", P.text).text(mae.toFixed(3));
    });
    if (ref) {
      const xr = 252 + scale(0.07);
      g.append("line").attr("x1", xr).attr("y1", 44).attr("x2", xr).attr("y2", 212)
        .attr("stroke", P.accent).attr("stroke-width", 2).attr("stroke-dasharray", "6 4");
      g.append("text").attr("x", xr).attr("y", 232).attr("text-anchor", "middle")
        .attr("font-size", 12).attr("font-weight", 700).attr("fill", P.accent)
        .text("tuned GPS, ~10× schedule: ≈0.070");
    }
    const msg = ref
      ? "same architecture family, 10× the budget, 4× better MAE — never compare across budgets"
      : "the encoding paid (−0.053, free); the attention didn't (+0.007 at 2.7× time) — at THIS scale and budget";
    g.append("text").attr("x", 380).attr("y", 266).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.accentDark).text(msg);
    g.append("text").attr("x", 380).attr("y", 288).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", P.muted)
      .text("molecules average 23 atoms: four local layers already reach everything — range wasn't the bottleneck");
  }

  document.getElementById("w13-re-ref").addEventListener("click", (ev) => {
    ref = !ref;
    ev.target.classList.toggle("active", ref);
    render();
  });

  U.onThemeChange(render);
  U.lazyBoot("w13-re-svg", render);
})();
