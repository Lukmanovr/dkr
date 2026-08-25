/* Widget 13.1 — Reach vs cost: message passing against attention.
 * A 12-node "molecule": two rings joined by a bridge. Choose MPNN depth or
 * full attention; the widget highlights what the starred node can hear and
 * counts the cost. Deterministic.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const N = 12;
  const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0],       // ring A
                 [7, 8], [8, 9], [9, 10], [10, 11], [11, 7],   // ring B
                 [4, 5], [5, 6], [6, 7]];                      // bridge
  const ADJ = Array.from({ length: N }, () => []);
  EDGES.forEach(([a, b]) => { ADJ[a].push(b); ADJ[b].push(a); });
  const POS = [];
  for (let k = 0; k < 5; k++) {
    POS.push([150 + 62 * Math.cos(-Math.PI / 2 + k * 2 * Math.PI / 5),
              150 + 62 * Math.sin(-Math.PI / 2 + k * 2 * Math.PI / 5)]);
  }
  POS.push([300, 190], [380, 150]);
  for (let k = 0; k < 5; k++) {
    POS.push([540 + 62 * Math.cos(Math.PI / 2 + k * 2 * Math.PI / 5),
              150 + 62 * Math.sin(Math.PI / 2 + k * 2 * Math.PI / 5)]);
  }
  const SRC = 0;
  let mode = "1";   // "1","2","3","attn"

  function khop(L) {
    let frontier = [SRC];
    const dist = Array(N).fill(Infinity);
    dist[SRC] = 0;
    for (let d = 1; d <= L; d++) {
      const nxt = [];
      frontier.forEach((v) => ADJ[v].forEach((u) => {
        if (dist[u] === Infinity) { dist[u] = d; nxt.push(u); }
      }));
      frontier = nxt;
    }
    return dist;
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w13-at-svg", 760, 300);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const attn = mode === "attn";
    const L = attn ? 0 : +mode;
    const dist = khop(attn ? 0 : L);
    const reached = attn ? N : dist.filter((d) => d < Infinity).length;

    if (attn) {
      for (let v = 1; v < N; v++) {
        g.append("line").attr("x1", POS[SRC][0]).attr("y1", POS[SRC][1] + 8)
          .attr("x2", POS[v][0]).attr("y2", POS[v][1] + 8)
          .attr("stroke", "#7c5cd6").attr("stroke-width", 1.4).attr("opacity", 0.5);
      }
    }
    EDGES.forEach(([a, b]) => {
      g.append("line").attr("x1", POS[a][0]).attr("y1", POS[a][1] + 8)
        .attr("x2", POS[b][0]).attr("y2", POS[b][1] + 8)
        .attr("stroke", P.muted).attr("stroke-width", 1.5)
        .attr("opacity", attn ? 0.2 : 0.5);
    });
    for (let v = 0; v < N; v++) {
      const hot = attn ? true : dist[v] < Infinity;
      g.append("circle").attr("cx", POS[v][0]).attr("cy", POS[v][1] + 8)
        .attr("r", v === SRC ? 12 : 8)
        .attr("fill", v === SRC ? P.accent : attn ? "#7c5cd6" : P.blue)
        .attr("opacity", hot ? 0.95 : 0.25);
    }
    g.append("text").attr("x", POS[SRC][0]).attr("y", POS[SRC][1] + 13)
      .attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 700)
      .attr("fill", P.bg).text("★");
    const label = attn ? "one attention layer: everyone, immediately"
      : `${L} message-passing layer${L > 1 ? "s" : ""}: ${reached}/12 nodes reachable`;
    g.append("text").attr("x", 380).attr("y", 24).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text(label);
    const cost = attn ? `${N * N} pair scores (n²)` : `${2 * EDGES.length * L} messages (L·2E)`;
    g.append("text").attr("x", 380).attr("y", 264).attr("text-anchor", "middle")
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
      .attr("font-weight", 700).attr("fill", P.text).text(`cost this configuration: ${cost}`);
    const verdict = attn
      ? "range for free, structure for nothing — the graph must come back as features"
      : reached < N ? "the far ring is silent: bridges cost hops (Week 9's two walls, in one picture)"
      : "deep enough to hear everyone — and paying oversmoothing risk for it";
    g.append("text").attr("x", 380).attr("y", 286).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.accentDark).text(verdict);
  }

  document.querySelectorAll("#w13-at-widget [data-m]").forEach((b) =>
    b.addEventListener("click", () => {
      mode = b.getAttribute("data-m");
      document.querySelectorAll("#w13-at-widget [data-m]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w13-at-svg", render);
})();
