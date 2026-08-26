/* Widget 1.10 — Matrix powers count walks, made clickable.
 *
 * The cast graph (A..F; edges AB, AC, AD, BC, CE, CF, EF) beside its k-th
 * power. Pick k, click any cell of A^k, and the walks that cell counts are
 * enumerated and drawn on the graph. The point students usually miss —
 * that row-times-column is a sum over MIDDLE nodes — becomes visible: the
 * cell's number is exactly the length of the list underneath it.
 *
 * Every entry is computed here by repeated multiplication of the real
 * adjacency matrix, and every listed walk is found by search, so the number
 * in the cell and the walks on the graph cannot drift apart.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 430;

  const NAMES = ["A", "B", "C", "D", "E", "F"];
  const n = 6;
  const EDGES = [["A","B"],["A","C"],["A","D"],["B","C"],["C","E"],["C","F"],["E","F"]]
    .map(([a, b]) => [NAMES.indexOf(a), NAMES.indexOf(b)]);
  const POS = {                              // hand-placed, readable at 360 px
    A: [92, 96], B: [206, 40], C: [200, 168], D: [40, 196], E: [96, 268], F: [214, 288],
  };

  const A = Array.from({ length: n }, () => Array(n).fill(0));
  EDGES.forEach(([a, b]) => { A[a][b] = 1; A[b][a] = 1; });
  const NB = Array.from({ length: n }, (_, u) => A[u].flatMap((v, j) => (v ? [j] : [])));

  function power(k) {                        // A^k by repeated multiplication
    let M = A.map((r) => r.slice());
    for (let p = 1; p < k; p++) M = U.matmul(M, A);
    return M;
  }

  // Every walk of length k from u to v, as arrays of node indices.
  function walksBetween(u, v, k) {
    const out = [];
    (function step(at, path) {
      if (path.length - 1 === k) { if (at === v) out.push(path.slice()); return; }
      for (const w of NB[at]) { path.push(w); step(w, path); path.pop(); }
    })(u, [u]);
    return out;
  }

  let k = 2;
  let sel = [0, 2];                          // default: A -> C, the worked example

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w1-wk-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const M = power(k);
    const [su, sv] = sel;
    const found = walksBetween(su, sv, k);
    const used = new Set();                  // edges any listed walk travels
    found.forEach((w) => {
      for (let i = 1; i < w.length; i++) used.add(Math.min(w[i-1], w[i]) + "-" + Math.max(w[i-1], w[i]));
    });

    // ── the graph ──
    EDGES.forEach(([a, b]) => {
      const hot = used.has(Math.min(a, b) + "-" + Math.max(a, b));
      g.append("line")
        .attr("x1", POS[NAMES[a]][0]).attr("y1", POS[NAMES[a]][1])
        .attr("x2", POS[NAMES[b]][0]).attr("y2", POS[NAMES[b]][1])
        .attr("stroke", hot ? P.accent : P.muted)
        .attr("stroke-width", hot ? 3.4 : 1.6)
        .attr("opacity", hot ? 0.95 : 0.45);
    });
    NAMES.forEach((nm, i) => {
      const [x, y] = POS[nm];
      const isEnd = i === su || i === sv;
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", isEnd ? 19 : 15)
        .attr("fill", isEnd ? P.accent : P.blue)
        .attr("stroke", P.paper).attr("stroke-width", isEnd ? 2.5 : 0);
      g.append("text").attr("x", x).attr("y", y + 5)
        .attr("text-anchor", "middle").attr("font-size", isEnd ? 15 : 13)
        .attr("font-weight", 700).attr("fill", "#fff").text(nm);
    });
    g.append("text").attr("x", 20).attr("y", 24).attr("font-size", 12.5).attr("fill", P.muted)
      .text("the cast graph — 6 people, 7 edges");
    g.append("text").attr("x", 20).attr("y", 340).attr("font-size", 13).attr("fill", P.text)
      .attr("font-weight", 700)
      .text(`walks of length ${k} from ${NAMES[su]} to ${NAMES[sv]}`);
    if (!found.length) {
      g.append("text").attr("x", 20).attr("y", 362).attr("font-size", 12.5).attr("fill", P.muted)
        .text(`none — you cannot get from ${NAMES[su]} to ${NAMES[sv]} in exactly ${k} step${k > 1 ? "s" : ""}`);
    } else {
      found.slice(0, 5).forEach((w, i) => {
        g.append("text").attr("x", 20).attr("y", 362 + i * 17)
          .attr("font-family", "'JetBrains Mono', monospace")
          .attr("font-size", 12.5).attr("fill", P.accentDark)
          .text(w.map((z) => NAMES[z]).join(" → "));
      });
      if (found.length > 5) {
        g.append("text").attr("x", 20).attr("y", 362 + 5 * 17).attr("font-size", 12.5)
          .attr("fill", P.muted).text(`… and ${found.length - 5} more`);
      }
    }

    // ── the matrix ──
    const cell = 40, mx = 400, my = 82;
    g.append("text").attr("x", mx).attr("y", 24)
      .attr("font-family", "'JetBrains Mono', monospace")
      .attr("font-size", 14).attr("font-weight", 700).attr("fill", P.text)
      .text(k === 1 ? "A  (one step)" : `A^${k}  (${k} steps)`);
    g.append("text").attr("x", mx).attr("y", 44).attr("font-size", 12.5).attr("fill", P.muted)
      .text("click any cell — its number is the walk count");
    NAMES.forEach((nm, j) => {
      g.append("text").attr("x", mx + 28 + j * cell + cell / 2).attr("y", my - 8)
        .attr("text-anchor", "middle").attr("font-size", 12.5).attr("font-weight", 700)
        .attr("fill", j === sv ? P.accentDark : P.muted).text(nm);
      g.append("text").attr("x", mx + 18).attr("y", my + j * cell + cell / 2 + 4.5)
        .attr("text-anchor", "middle").attr("font-size", 12.5).attr("font-weight", 700)
        .attr("fill", j === su ? P.accentDark : P.muted).text(nm);
    });
    const maxV = Math.max(1, ...M.flat());
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const on = i === su && j === sv;
        const t = M[i][j] / maxV;
        const cx = mx + 28 + j * cell, cy = my + i * cell;
        g.append("rect").attr("x", cx).attr("y", cy).attr("width", cell).attr("height", cell)
          .attr("fill", M[i][j] ? P.accent : P.paper)
          .attr("fill-opacity", M[i][j] ? 0.12 + 0.5 * t : 1)
          .attr("stroke", on ? P.accentDark : P.border)
          .attr("stroke-width", on ? 3 : 0.8)
          .style("cursor", "pointer")
          .on("click", () => { sel = [i, j]; render(); });
        g.append("text").attr("x", cx + cell / 2).attr("y", cy + cell / 2 + 5)
          .attr("text-anchor", "middle")
          .attr("font-family", "'JetBrains Mono', monospace")
          .attr("font-size", 13.5).attr("font-weight", on ? 700 : 500)
          .attr("fill", M[i][j] ? P.text : P.muted)
          .style("pointer-events", "none")
          .text(M[i][j]);
      }
    }
    g.append("text").attr("x", mx).attr("y", my + 6 * cell + 30)
      .attr("font-family", "'JetBrains Mono', monospace")
      .attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.accentDark)
      .text(`(A^${k})[${NAMES[su]},${NAMES[sv]}] = ${M[su][sv]}`);
    g.append("text").attr("x", mx).attr("y", my + 6 * cell + 50)
      .attr("font-size", 12.5).attr("fill", P.text)
      .text(`and exactly ${found.length} walk${found.length === 1 ? "" : "s"} ${found.length === 1 ? "is" : "are"} listed on the left`);
  }

  for (const b of document.querySelectorAll("#w1-wk-widget [data-k]")) {
    b.addEventListener("click", () => {
      k = Number(b.dataset.k);
      for (const o of document.querySelectorAll("#w1-wk-widget [data-k]")) {
        o.classList.toggle("active", o === b);
      }
      render();
    });
  }
  U.onThemeChange(render);
  U.lazyBoot("w1-wk-svg", render);
})();
