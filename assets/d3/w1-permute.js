/* Widget 1.6 — Permutation invariance vs equivariance, made visible.
 * The cast graph never moves; only its STORAGE ORDER shuffles. One readout
 * per node (the degree vector) travels with its node — equivariant,
 * f(PAP^T) = P f(A). One readout per graph (m = 7 edges; the sorted degree
 * sequence) cannot feel the shuffle — invariant, f(PAP^T) = f(A).
 * Deterministic: mulberry32-seeded permutation sequence, identity at boot.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const NAMES = ["A", "B", "C", "D", "E", "F"];
  const COLORS = ["#d9a62e", "#cf4a30", "#199473", "#7c5cd6", "#d1567e", "#0f8377"];
  const EDGES = [[0, 1], [0, 2], [0, 3], [1, 2], [2, 4], [2, 5], [4, 5]];
  const DEG = [3, 2, 4, 1, 2, 2];
  const POS = [[130, 96], [222, 52], [286, 120], [78, 168], [232, 196], [330, 182]];
  const W = 760, H = 330;

  const rng = U.mulberry32 ? U.mulberry32(7) : (() => { let s = 7; return () => {
    s = (s + 0x6D2B79F5) | 0; let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; })();

  let perm = [0, 1, 2, 3, 4, 5];       // perm[slot] = node stored in that slot
  let shuffled = false;

  function shuffle() {
    const p = perm.slice();
    for (let i = p.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [p[i], p[j]] = [p[j], p[i]];
    }
    perm = p;
    shuffled = true;
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w1-pe-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    g.append("text").attr("x", W / 2).attr("y", 20).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("the graph never moves — only the storage order (the gray #slots) shuffles");

    // ── left: the fixed drawing, with storage-slot badges ──
    const slotOf = [];
    perm.forEach((node, slot) => { slotOf[node] = slot; });
    for (const [a, b] of EDGES) {
      g.append("line").attr("x1", POS[a][0]).attr("y1", POS[a][1] + 30)
        .attr("x2", POS[b][0]).attr("y2", POS[b][1] + 30)
        .attr("stroke", P.muted).attr("stroke-width", 1.7).attr("opacity", 0.5);
    }
    NAMES.forEach((nm, i) => {
      const [x, y0] = POS[i]; const y = y0 + 30;
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", 16).attr("fill", COLORS[i]);
      g.append("text").attr("x", x).attr("y", y + 4.5).attr("text-anchor", "middle")
        .attr("font-size", 13).attr("font-weight", 700).attr("fill", "#fff").text(nm);
      g.append("rect").attr("x", x + 8).attr("y", y - 31).attr("width", 32)
        .attr("height", 18).attr("rx", 9).attr("fill", P.muted).attr("opacity", 0.85);
      g.append("text").attr("x", x + 24).attr("y", y - 18).attr("text-anchor", "middle")
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
        .attr("font-weight", 700).attr("fill", P.bg).text("#" + (slotOf[i] + 1));
    });

    // ── right: the two readouts, laid out by STORAGE slot ──
    const rx = 415, cw = 47;
    const row = (y, label, hot) => {
      g.append("text").attr("x", rx).attr("y", y).attr("font-size", 12.5)
        .attr("font-weight", 700).attr("fill", hot ? P.accentDark : P.blueDark)
        .text(label);
    };
    row(58, "storage slots  #1 … #6", false);
    perm.forEach((node, slot) => {
      const x = rx + slot * cw;
      g.append("circle").attr("cx", x + 16).attr("cy", 84).attr("r", 13)
        .attr("fill", COLORS[node]);
      g.append("text").attr("x", x + 16).attr("y", 88.5).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", "#fff")
        .text(NAMES[node]);
    });

    row(128, "equivariant — degrees TRAVEL with their nodes", true);
    perm.forEach((node, slot) => {
      const x = rx + slot * cw;
      g.append("rect").attr("x", x).attr("y", 140).attr("width", 32)
        .attr("height", DEG[node] * 11).attr("rx", 4)
        .attr("fill", COLORS[node]).attr("opacity", 0.85);
      g.append("text").attr("x", x + 16).attr("y", 140 + DEG[node] * 11 + 15)
        .attr("text-anchor", "middle").attr("font-family", "'JetBrains Mono', monospace")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", P.text)
        .text(DEG[node]);
    });
    g.append("text").attr("x", rx).attr("y", 222).attr("font-family",
      "'JetBrains Mono', monospace").attr("font-size", 12.5).attr("fill", P.muted)
      .text("f(PAPᵀ) = P f(A)");

    row(252, "invariant — sorted degrees & m CANNOT move", false);
    g.append("text").attr("x", rx).attr("y", 274).attr("font-family",
      "'JetBrains Mono', monospace").attr("font-size", 13).attr("fill", P.text)
      .text("sorted: [" + DEG.slice().sort().join(", ") + "]   ·   m = " +
        EDGES.length + " edges");
    g.append("text").attr("x", rx).attr("y", 295).attr("font-family",
      "'JetBrains Mono', monospace").attr("font-size", 12.5).attr("fill", P.muted)
      .text("f(PAPᵀ) = f(A)" + (shuffled ? "   — unchanged ✓" : ""));

    g.append("text").attr("x", W / 2).attr("y", H - 8).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
      .text(shuffled
        ? "the per-node values changed seats but kept their owners; the per-graph values never moved"
        : "press shuffle: the drawing will not move, because nothing about the graph changes");
  }

  document.getElementById("w1-pe-shuffle").addEventListener("click", () => {
    shuffle(); render();
  });
  document.getElementById("w1-pe-reset").addEventListener("click", () => {
    perm = [0, 1, 2, 3, 4, 5]; shuffled = false; render();
  });

  U.onThemeChange(render);
  U.lazyBoot("w1-pe-svg", render);
})();
