/* Widget 10.4 — The encoder showdown, measured.
 * Numbers are real: FB15k-237, Lab-4 protocol (margin loss, 2,000 test facts,
 * both directions, filtered), the consistent ±0.06-init series, run
 * 2026-08-22 on this course's hardware (scripts/experiments/w10_experiment3.py).
 * Two lenses: the budget table, and TransE's per-category MRR.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const ROWS = [
    ["TransE (lookup)", 5, "7 s", 0.178, 0.301, "gold"],
    ["DistMult (lookup)", 5, "7 s", 0.179, 0.311, "gold"],
    ["DistMult (lookup)", 15, "22 s", 0.145, 0.281, "muted"],
    ["R-GCN → DistMult", 5, "156 s", 0.126, 0.257, "teal"],
    ["R-GCN → DistMult", 15, "1143 s", 0.122, 0.257, "muted"],
  ];
  const CATS = [["1-1", 0.196, 22], ["1-N", 0.152, 264], ["N-1", 0.313, 864], ["N-N", 0.139, 2850]];

  let view = "budget";

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w10-sd-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const colOf = { gold: P.yellow, teal: P.blue, muted: P.muted };

    if (view === "budget") {
      g.append("text").attr("x", 380).attr("y", 24).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("fill", P.muted)
        .text("same loss, same eval, our hardware — filtered MRR (bar) and Hits@10, with the wall-clock bill");
      ROWS.forEach(([name, ep, secs, mrr, h10, ckey], i) => {
        const y = 60 + i * 46;
        const w = 340 * mrr / 0.2;
        g.append("text").attr("x", 175).attr("y", y + 8).attr("text-anchor", "end")
          .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.text)
          .text(name);
        g.append("text").attr("x", 175).attr("y", y + 24).attr("text-anchor", "end")
          .attr("font-size", 12).attr("fill", P.muted).text(ep + " ep · " + secs);
        g.append("rect").attr("x", 188).attr("y", y - 4).attr("width", w).attr("height", 20)
          .attr("rx", 5).attr("fill", colOf[ckey]).attr("opacity", 0.85);
        g.append("text").attr("x", 188 + w + 8).attr("y", y + 11)
          .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
          .attr("fill", P.text).text(`MRR ${mrr.toFixed(3)} · H@10 ${h10.toFixed(3)}`);
      });
      g.append("text").attr("x", 380).attr("y", 300).attr("text-anchor", "middle")
        .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.accentDark)
        .text("the lookup wins at every budget, at 1/22nd the compute — and more epochs helped NEITHER pipeline");
      g.append("text").attr("x", 380).attr("y", 320).attr("text-anchor", "middle")
        .attr("font-size", 12).attr("fill", P.muted)
        .text("well-tuned literature numbers (100+ epochs, tuned losses) reach MRR ≈ 0.30–0.34 — also led by lookups");
    } else {
      g.append("text").attr("x", 380).attr("y", 24).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("fill", P.muted)
        .text("TransE's MRR by relation category (test sample, both directions) — Week 4's crush, quantified");
      CATS.forEach(([cat, mrr, n], i) => {
        const y = 70 + i * 52;
        const w = 480 * mrr / 0.35;
        g.append("text").attr("x", 96).attr("y", y + 10).attr("text-anchor", "end")
          .attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text).text(cat);
        g.append("rect").attr("x", 110).attr("y", y - 6).attr("width", w).attr("height", 24)
          .attr("rx", 6).attr("fill", cat === "N-1" ? P.green : cat === "N-N" ? P.accent : P.blue)
          .attr("opacity", 0.85);
        g.append("text").attr("x", 110 + w + 8).attr("y", y + 11)
          .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
          .attr("fill", P.text).text(`MRR ${mrr.toFixed(3)}  (n=${n})`);
      });
      g.append("text").attr("x", 380).attr("y", 296).attr("text-anchor", "middle")
        .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.accentDark)
        .text("N-1 queries (many heads, one true tail) are easy; N-N — 71% of the test set — is where MRR goes to die");
      g.append("text").attr("x", 380).attr("y", 316).attr("text-anchor", "middle")
        .attr("font-size", 12).attr("fill", P.muted)
        .text("categories from train-set head/tail fan-out, threshold 1.5 — the TransE paper's own recipe");
    }
  }

  document.querySelectorAll("#w10-sd-widget [data-view]").forEach((b) =>
    b.addEventListener("click", () => {
      view = b.getAttribute("data-view");
      document.querySelectorAll("#w10-sd-widget [data-view]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w10-sd-svg", render);
})();
