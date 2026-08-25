/* Widget 7.1 — The multiset discriminator.
 * Build two neighborhoods out of teal/gold one-hot features; mean, max, and sum
 * are computed live, and the verdict says which aggregators can tell your two
 * multisets apart. Presets give the three classic confusion cases.
 */
(function () {
  "use strict";
  const U = window.DKR;
  let A = [0], B = [0, 0];                      // 0 = teal (1,0), 1 = gold (0,1)
  const PRESETS = [
    [[0], [0, 0]],
    [[0, 1], [0, 0, 1, 1]],
    [[0, 0, 1], [0, 1, 1]],
  ];

  function vecs(ms) {
    const arr = ms.map((c) => (c === 0 ? [1, 0] : [0, 1]));
    const mean = [0, 1].map((i) => arr.reduce((a, v) => a + v[i], 0) / arr.length);
    const mx = [0, 1].map((i) => Math.max(...arr.map((v) => v[i])));
    const sum = [0, 1].map((i) => arr.reduce((a, v) => a + v[i], 0));
    return { mean, mx, sum };
  }
  const eq = (a, b) => Math.abs(a[0] - b[0]) < 1e-9 && Math.abs(a[1] - b[1]) < 1e-9;
  const fmt = (v) => `(${+v[0].toFixed(2)}, ${+v[1].toFixed(2)})`;

  function drawSet(g, ms, x0, y, label, P, which) {
    g.append("text").attr("x", x0 - 14).attr("y", y + 5).attr("text-anchor", "end")
      .attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text).text(label);
    ms.forEach((c, i) => {
      const cx = x0 + 16 + i * 34;
      g.append("circle").attr("cx", cx).attr("cy", y).attr("r", 12)
        .attr("fill", c === 0 ? P.blue : P.yellow).style("cursor", "pointer")
        .on("click", () => {                     // click toggles color
          ms[i] = 1 - ms[i];
          render();
        });
    });
    // add / remove controls
    const bx = x0 + 16 + ms.length * 34 + 6;
    [["+", () => { if (ms.length < 6) ms.push(0); }],
     ["−", () => { if (ms.length > 1) ms.pop(); }]].forEach(([sym, fn], k) => {
      g.append("circle").attr("cx", bx + k * 28).attr("cy", y).attr("r", 11)
        .attr("fill", "none").attr("stroke", P.muted).attr("stroke-width", 1.6)
        .style("cursor", "pointer").on("click", () => { fn(); render(); });
      g.append("text").attr("x", bx + k * 28).attr("y", y + 5).attr("text-anchor", "middle")
        .attr("font-size", 14).attr("fill", P.muted).style("pointer-events", "none").text(sym);
    });
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w7-ag-svg", 760, 300);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    g.append("text").attr("x", 380).attr("y", 26).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("click a node to flip its color · +/− to resize · can you make two different multisets that fool ALL three?");

    // builder centered: label + up to 6 nodes + controls sit in the middle band
    drawSet(g, A, 258, 62, "N(a):", P);
    drawSet(g, B, 258, 102, "N(b):", P);

    const a = vecs(A), b = vecs(B);
    const rows = [["mean", a.mean, b.mean], ["max", a.mx, b.mx], ["sum", a.sum, b.sum]];
    const identical = A.length === B.length &&
      [...A].sort().join() === [...B].sort().join();
    rows.forEach(([name, va, vb], i) => {
      const y = 168 + i * 34;
      const same = eq(va, vb);
      g.append("text").attr("x", 120).attr("y", y).attr("text-anchor", "end")
        .attr("font-size", 13).attr("font-weight", 700)
        .attr("fill", same ? "#cf4a30" : P.green)
        .text(`${same ? "✗" : "✓"} ${name}`);
      g.append("text").attr("x", 150).attr("y", y).attr("font-size", 13)
        .attr("font-family", "'JetBrains Mono', monospace").attr("fill", P.text)
        .text(`${fmt(va)}   vs   ${fmt(vb)}`);
      g.append("text").attr("x", 462).attr("y", y).attr("font-size", 12.5)
        .attr("fill", P.muted)
        .text(same ? "confused — a and b look identical"
                   : "separated — the difference survives");
    });

    const anyDistinguish = rows.some(([, va, vb]) => !eq(va, vb));
    let verdict;
    if (identical) verdict = "the multisets are identical — no aggregator can (or should) separate them";
    else if (!anyDistinguish) verdict = "all three fooled by genuinely different multisets — you found a blind spot!";
    else verdict = "different multisets, and at least one aggregator sees it";
    g.append("text").attr("x", 380).attr("y", 286).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600)
      .attr("fill", identical ? P.muted : anyDistinguish ? P.text : P.accentDark)
      .text(verdict);
  }

  document.querySelectorAll("#w7-ag-widget [data-preset]").forEach((btn) =>
    btn.addEventListener("click", () => {
      const k = +btn.getAttribute("data-preset");
      A = PRESETS[k][0].slice();
      B = PRESETS[k][1].slice();
      document.querySelectorAll("#w7-ag-widget [data-preset]").forEach((x) => x.classList.toggle("active", x === btn));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w7-ag-svg", render);
})();
