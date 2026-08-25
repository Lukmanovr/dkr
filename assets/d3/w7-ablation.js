/* Widget 7.4 — A real ablation, explorable.
 * The numbers are measured, not invented: Cora public split, hidden 64,
 * dropout 0.5, Adam(0.01), early stopping on validation, THREE SEEDS per
 * configuration — the reference run of Lab 7's ablation grid (reproduce it
 * there). Toggle one factor at a time and read the delta like a scientist.
 */
(function () {
  "use strict";
  const U = window.DKR;
  // measured 2026-08-21; mean ± std over seeds {0,1,2}; test accuracy %
  const R = {
    "GCN|2|plain": [81.0, 0.6], "GCN|2|res": [80.3, 0.9],
    "GCN|4|plain": [76.9, 2.4], "GCN|4|res": [80.0, 1.1],
    "SAGE|2|plain": [79.8, 1.1], "SAGE|2|res": [79.9, 0.1],
    "SAGE|4|plain": [77.8, 1.1], "SAGE|4|res": [79.5, 0.9],
    "GAT|2|plain": [75.8, 1.2], "GAT|2|res": [77.1, 2.2],
    "GAT|4|plain": [26.2, 9.8], "GAT|4|res": [80.5, 1.0],
    "GIN|2|plain": [77.3, 1.2], "GIN|2|res": [75.9, 0.8],
    "GIN|4|plain": [44.2, 19.5], "GIN|4|res": [75.2, 1.3],
  };
  const ARCHS = ["GCN", "SAGE", "GAT", "GIN"];

  let arch = "GCN", depth = 2, res = false;
  const key = () => `${arch}|${depth}|${res ? "res" : "plain"}`;

  function render() {
    const P = U.pal();
    // four clearly separated architecture hues, theme-aware (gold/teal/salmon/purple)
    const ACOL = { GCN: P.yellow, SAGE: P.blue, GAT: P.accent, GIN: P.purple };
    const svg = U.svgIn("w7-ab-svg", 760, 360);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    const [m, s] = R[key()];
    const [bm] = R["GCN|2|plain"];
    const delta = m - bm;

    const headTxt = `${arch} · ${depth} layers · ${res ? "residual" : "no skips"}`;
    g.append("rect").attr("x", 200 - headTxt.length * 3.8 - 28).attr("y", 48)
      .attr("width", 13).attr("height", 13).attr("rx", 3).attr("fill", ACOL[arch]);
    g.append("text").attr("x", 200).attr("y", 60).attr("text-anchor", "middle")
      .attr("font-size", 15).attr("font-weight", 700).attr("fill", P.text)
      .text(headTxt);
    g.append("text").attr("x", 200).attr("y", 108).attr("text-anchor", "middle")
      .attr("font-size", 40).attr("font-weight", 800)
      .attr("fill", m > 70 ? P.text : "#cf4a30")
      .text(`${m.toFixed(1)}%`);
    g.append("text").attr("x", 200).attr("y", 134).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("fill", P.muted).text(`± ${s.toFixed(1)} over 3 seeds`);
    g.append("text").attr("x", 200).attr("y", 162).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600)
      .attr("fill", Math.abs(delta) <= 1.0 ? P.muted : delta > 0 ? P.green : "#cf4a30")
      .text(`${delta >= 0 ? "+" : ""}${delta.toFixed(1)} vs the 2-layer GCN baseline` +
            (Math.abs(delta) <= 2 * Math.max(s, 0.6) ? "  (within noise!)" : ""));

    // ranked strip of all 16 configs
    const entries = Object.entries(R).sort((a, b) => b[1][0] - a[1][0]);
    const x0 = 420, y0 = 42, rowH = 18;
    g.append("text").attr("x", x0 + 140).attr("y", y0 - 14).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.text)
      .text("all 16 measured configurations");
    entries.forEach(([k, [mm, ss]], i) => {
      const y = y0 + i * rowH;
      const [a] = k.split("|");
      const w = Math.max(2, (mm / 85) * 190);
      const hot = k === key();
      g.append("rect").attr("x", x0 + 78).attr("y", y).attr("width", w).attr("height", 13)
        .attr("rx", 4).attr("fill", ACOL[a]).attr("opacity", hot ? 1 : 0.35);
      g.append("text").attr("x", x0 + 72).attr("y", y + 11).attr("text-anchor", "end")
        .attr("font-size", 12.5).attr("font-weight", hot ? 700 : 400)
        .attr("fill", hot ? P.text : P.muted)
        .text(k.replace("|res", "·R").replace("|plain", "").replace("|", "·"));
      if (hot || i === 0 || i === entries.length - 1) {
        g.append("text").attr("x", x0 + 82 + w).attr("y", y + 11)
          .attr("font-size", 12.5).attr("font-weight", hot ? 700 : 400)
          .attr("fill", hot ? P.text : P.muted)
          .text(hot ? `${mm.toFixed(1)} ± ${ss.toFixed(1)}` : mm.toFixed(1));
      }
    });

    const notes = {
      "GAT|4|plain": "the crash: deep attention with no highway for gradients — and s.d. = 9.8 means seeds disagree wildly",
      "GIN|4|plain": "sum aggregation explodes activations with depth; s.d. = 19.5 is a model gambling, not learning",
      "GAT|4|res": "same GAT, one residual connection: +54 points. The safest upgrade in the design space",
      "GCN|2|plain": "the boring baseline is the best single cell — on THIS dataset. That sentence is the whole method",
    };
    const note = notes[key()] ||
      (depth === 2 ? "shallow models barely care about skips — there is little depth to rescue"
                   : "at depth 4, residuals decide whether training works at all");
    g.append("text").attr("x", 200).attr("y", 210).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.text)
      .call((t) => {
        const words = note.split(" ");
        let line = "", lines = [];
        words.forEach((w) => {
          if ((line + " " + w).length > 42) { lines.push(line); line = w; }
          else line = (line + " " + w).trim();
        });
        lines.push(line);
        lines.forEach((ln, i) => t.append("tspan").attr("x", 200).attr("dy", i ? 17 : 0).text(ln));
      });

    g.append("text").attr("x", 380).attr("y", 350).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("measured on Cora (public split, 3 seeds, early stopping) — Lab 7 reproduces this grid");
  }

  document.querySelectorAll("#w7-ab-widget [data-arch]").forEach((b) =>
    b.addEventListener("click", () => {
      arch = b.getAttribute("data-arch");
      document.querySelectorAll("#w7-ab-widget [data-arch]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.querySelectorAll("#w7-ab-widget [data-depth]").forEach((b) =>
    b.addEventListener("click", () => {
      depth = +b.getAttribute("data-depth");
      document.querySelectorAll("#w7-ab-widget [data-depth]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.getElementById("w7-ab-res").addEventListener("change", (ev) => {
    res = ev.currentTarget.checked;
    render();
  });

  U.onThemeChange(render);
  U.lazyBoot("w7-ab-svg", render);
})();
