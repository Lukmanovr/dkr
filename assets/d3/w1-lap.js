/* Widget 1.11 — The Laplacian as a disagreement meter.
 *
 * The cast graph carries a signal x with one number per node. Every edge is
 * labelled with its squared difference, and the total is x^T L x — so the
 * quadratic form stops being an identity to memorise and becomes a running
 * tally you can watch move as you click nodes.
 *
 * The honest comparison: with the SAME six values reshuffled among the six
 * nodes, what would the total be on average? That expectation is computed
 * exactly here, by averaging over all 720 permutations, and it is the line
 * that separates homophily (below chance) from heterophily (above it).
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 400;

  const NAMES = ["A", "B", "C", "D", "E", "F"];
  const n = 6;
  const EDGES = [["A","B"],["A","C"],["A","D"],["B","C"],["C","E"],["C","F"],["E","F"]]
    .map(([a, b]) => [NAMES.indexOf(a), NAMES.indexOf(b)]);
  const POS = {
    A: [150, 108], B: [286, 52], C: [280, 190], D: [92, 214], E: [152, 292], F: [292, 312],
  };

  // Presets over the same multiset {+1,+1,+1,-1,-1,-1}, so the chance
  // baseline is identical for all three and only the ARRANGEMENT differs.
  // Verified by exhaustive search over all balanced splits: 8 is the minimum
  // this graph admits, 20 the maximum, and 16 sits nearest the 16.8 average.
  const PRESETS = {
    agree:    [1, 1, -1, 1, -1, -1],     // A B D on one side, C E F on the other
    mixed:    [1, -1, 1, 1, -1, -1],
    disagree: [1, -1, -1, -1, 1, 1],
  };
  let x = PRESETS.agree.slice();

  const quad = (v) => EDGES.reduce((s, [a, b]) => s + (v[a] - v[b]) ** 2, 0);

  // Exact mean of x^T L x over every rearrangement of the same values.
  function chanceMean(v) {
    const vals = v.slice();
    let total = 0, count = 0;
    (function perm(arr, k) {
      if (k === arr.length) { total += quad(arr); count++; return; }
      const seen = new Set();
      for (let i = k; i < arr.length; i++) {
        if (seen.has(arr[i])) continue;      // skip duplicate values
        seen.add(arr[i]);
        [arr[k], arr[i]] = [arr[i], arr[k]];
        perm(arr, k + 1);
        [arr[k], arr[i]] = [arr[i], arr[k]];
      }
    })(vals, 0);
    return total / count;
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w1-lp-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const total = quad(x);
    const mean = chanceMean(x);

    g.append("text").attr("x", 20).attr("y", 24).attr("font-size", 12.5).attr("fill", P.muted)
      .text("click any person to flip their side — the tally on the right follows");

    EDGES.forEach(([a, b]) => {
      const d2 = (x[a] - x[b]) ** 2;
      const [x1, y1] = POS[NAMES[a]], [x2, y2] = POS[NAMES[b]];
      g.append("line").attr("x1", x1).attr("y1", y1).attr("x2", x2).attr("y2", y2)
        .attr("stroke", d2 ? P.red : P.muted)
        .attr("stroke-width", d2 ? 3 : 1.6)
        .attr("opacity", d2 ? 0.85 : 0.4);
      g.append("text").attr("x", (x1 + x2) / 2).attr("y", (y1 + y2) / 2 - 5)
        .attr("text-anchor", "middle")
        .attr("font-family", "'JetBrains Mono', monospace")
        .attr("font-size", 12.5).attr("font-weight", d2 ? 700 : 400)
        .attr("fill", d2 ? P.red : P.muted)
        .attr("stroke", P.bg).attr("stroke-width", 3.5).attr("paint-order", "stroke")
        .text(d2);
    });
    NAMES.forEach((nm, i) => {
      const [cx, cy] = POS[nm];
      g.append("circle").attr("cx", cx).attr("cy", cy).attr("r", 23)
        .attr("fill", x[i] > 0 ? P.accent : P.blue)
        .style("cursor", "pointer")
        .on("click", () => { x[i] = -x[i]; render(); });
      g.append("text").attr("x", cx).attr("y", cy - 3).attr("text-anchor", "middle")
        .attr("font-size", 13).attr("font-weight", 700).attr("fill", "#fff")
        .style("pointer-events", "none").text(nm);
      g.append("text").attr("x", cx).attr("y", cy + 13).attr("text-anchor", "middle")
        .attr("font-family", "'JetBrains Mono', monospace")
        .attr("font-size", 12.5).attr("fill", "#fff").attr("opacity", 0.92)
        .style("pointer-events", "none").text(x[i] > 0 ? "+1" : "−1");
    });

    // ── the tally ──
    const px = 400;
    g.append("text").attr("x", px).attr("y", 60)
      .attr("font-size", 13).attr("fill", P.text)
      .text("each edge adds the square of its gap:");
    g.append("text").attr("x", px).attr("y", 92)
      .attr("font-family", "'JetBrains Mono', monospace")
      .attr("font-size", 15).attr("font-weight", 700).attr("fill", P.text)
      .text(`xᵀLx  =  ${total}`);
    const disagreeing = EDGES.filter(([a, b]) => x[a] !== x[b]).length;
    g.append("text").attr("x", px).attr("y", 116).attr("font-size", 12.5).attr("fill", P.muted)
      .text(`${disagreeing} of 7 friendships cross the divide, 4 each`);

    // the chance line, drawn as a small scale
    const bx = px, by = 186, bw = 264, hi = Math.max(total, mean, 20);
    g.append("text").attr("x", bx).attr("y", by - 30).attr("font-size", 12.5).attr("fill", P.text)
      .text("against the same values, reshuffled:");
    [["this arrangement", total, P.accent, 0],
     ["chance average", mean, P.muted, 34]].forEach(([label, val, col, dy]) => {
      g.append("rect").attr("x", bx).attr("y", by + dy).attr("width", (val / hi) * bw)
        .attr("height", 20).attr("fill", col).attr("opacity", 0.75).attr("rx", 3);
      g.append("text").attr("x", bx + (val / hi) * bw + 8).attr("y", by + dy + 15)
        .attr("font-family", "'JetBrains Mono', monospace")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.text)
        .text(typeof val === "number" && val % 1 ? val.toFixed(1) : String(val));
      g.append("text").attr("x", bx).attr("y", by + dy - 3).attr("font-size", 12.5)
        .attr("fill", P.muted).text(label);
    });

    const verdict = total < mean - 1.5
      ? "Below chance: friends agree — homophily."
      : total > mean + 1.5
        ? "Above chance: friends disagree — heterophily."
        : "About chance: as good as scattered.";
    g.append("text").attr("x", px).attr("y", 268).attr("font-size", 12.5)
      .attr("font-weight", 600).attr("fill", P.accentDark).text(verdict);
    g.append("text").attr("x", px).attr("y", 300).attr("font-size", 12.5).attr("fill", P.text)
      .text("Zachary's cut minimised this exact quantity,");
    g.append("text").attr("x", px).attr("y", 318).attr("font-size", 12.5).attr("fill", P.text)
      .text("with +1 and −1 as the two clubs.");
    g.append("text").attr("x", px).attr("y", 350).attr("font-size", 12.5).attr("fill", P.muted)
      .text("Put everyone on one side and it hits 0 —");
    g.append("text").attr("x", px).attr("y", 368).attr("font-size", 12.5).attr("fill", P.muted)
      .text("constants are what the Laplacian cannot see.");
  }

  for (const b of document.querySelectorAll("#w1-lp-widget [data-preset]")) {
    b.addEventListener("click", () => {
      x = PRESETS[b.dataset.preset].slice();
      for (const o of document.querySelectorAll("#w1-lp-widget [data-preset]")) {
        o.classList.toggle("active", o === b);
      }
      render();
    });
  }
  U.onThemeChange(render);
  U.lazyBoot("w1-lp-svg", render);
})();
