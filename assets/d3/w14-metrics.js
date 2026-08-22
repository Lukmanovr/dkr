/* Widget 14.2 — Recall@K and NDCG@K, anatomized.
 * A fixed ranked list of 10 items; the relevant ones sit at ranks 1, 4, 9
 * (3 relevant total). Slide K and watch both metrics recompute from their
 * definitions — every number checkable by hand. Recall is capped by
 * min(|P|, K); NDCG divides by the ideal ordering's DCG at the same K.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const POS = [1, 4, 9];         // 1-based ranks of the relevant items
  const NP = POS.length;
  let K = 5;

  const lg2 = (x) => Math.log(x) / Math.log(2);

  function calc(k) {
    let hits = 0, dcg = 0, idcg = 0;
    for (const r of POS) if (r <= k) { hits++; dcg += 1 / lg2(r + 1); }
    for (let r = 1; r <= Math.min(NP, k); r++) idcg += 1 / lg2(r + 1);
    return { hits, rec: hits / Math.min(NP, k), ndcg: dcg / idcg, dcg, idcg };
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w14-me-svg", 760, 300);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const c = calc(K);

    g.append("text").attr("x", 380).attr("y", 20).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("the model ranked 10 items · the user actually liked 3 of them (ranks 1, 4, 9)");

    for (let r = 1; r <= 10; r++) {
      const x = 60 + (r - 1) * 65;
      const isPos = POS.includes(r);
      const inK = r <= K;
      g.append("rect").attr("x", x).attr("y", 52).attr("width", 52).attr("height", 52)
        .attr("rx", 8).attr("fill", isPos ? P.green : P.muted)
        .attr("opacity", inK ? (isPos ? 0.9 : 0.35) : 0.12);
      g.append("text").attr("x", x + 26).attr("y", 84).attr("text-anchor", "middle")
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 13)
        .attr("font-weight", 700).attr("fill", inK ? P.text : P.muted).text(r);
      if (isPos) {
        g.append("text").attr("x", x + 26).attr("y", 124).attr("text-anchor", "middle")
          .attr("font-size", 12).attr("font-weight", 700)
          .attr("fill", inK ? P.green : P.muted).text("liked");
      }
    }
    const bx = 60 + K * 65 - 13 + 6;
    g.append("line").attr("x1", bx).attr("y1", 44).attr("x2", bx).attr("y2", 134)
      .attr("stroke", P.accent).attr("stroke-width", 2.5).attr("stroke-dasharray", "6 4");
    g.append("text").attr("x", bx).attr("y", 150).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.accent)
      .text("K = " + K);

    const mono = "'JetBrains Mono', monospace";
    g.append("text").attr("x", 80).attr("y", 192).attr("font-family", mono)
      .attr("font-size", 12.5).attr("fill", P.text)
      .text("Recall@" + K + " = hits / min(|P|, K) = " + c.hits + " / " +
        Math.min(NP, K) + " = " + c.rec.toFixed(3));
    g.append("text").attr("x", 80).attr("y", 220).attr("font-family", mono)
      .attr("font-size", 12.5).attr("fill", P.text)
      .text("DCG@" + K + " = " + c.dcg.toFixed(4) + "   IDCG@" + K + " = " +
        c.idcg.toFixed(4) + "   NDCG@" + K + " = " + c.ndcg.toFixed(3));
    const msg = K === 10
      ? "at K=10 everything is found (recall 1.000) — but NDCG stays 0.813: " +
        "it remembers rank 9 arrived late"
      : K >= 4
        ? "recall counts WHAT you found; NDCG also charges for WHERE it sat"
        : "the rank-4 hit is invisible below K=4 — a metric only sees inside its window";
    g.append("text").attr("x", 380).attr("y", 254).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
      .text(msg);
    g.append("text").attr("x", 380).attr("y", 278).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", P.muted)
      .text("1/log₂(r+1) discounting: rank 1 pays 1.0, rank 4 pays 0.43, rank 9 pays 0.30");
  }

  for (const b of document.querySelectorAll("#w14-me-widget [data-k]")) {
    b.addEventListener("click", () => {
      K = +b.dataset.k;
      for (const o of document.querySelectorAll("#w14-me-widget [data-k]")) {
        o.classList.toggle("active", o === b);
      }
      render();
    });
  }
  U.onThemeChange(render);
  U.lazyBoot("w14-me-svg", render);
})();
