/* Widget 14.1 — LightGCN propagation, driven by hand.
 * u1's unit of signal spreads over the normalized bipartite graph
 * (weights 1/sqrt(du*di), LightGCN's propagation exactly). Hand-checkable:
 * after 2 steps u2 holds 0.250 (via shared item i2); u3 first hears at
 * K=4 with mass 0.063 (i3 carries both u2's step-2 mass and its echo).
 */
(function () {
  "use strict";
  const U = window.DKR;
  const EDGES = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [2, 3]];
  const DU = [2, 2, 2], DI = [1, 2, 2, 1];
  const UP = [[150, 70], [150, 150], [150, 230]];
  const IP = [[420, 50], [420, 116], [420, 182], [420, 248]];
  let K = 0;

  function masses(k) {
    let m = [1, 0, 0, 0, 0, 0, 0];
    for (let s = 0; s < k; s++) {
      const n = [0, 0, 0, 0, 0, 0, 0];
      for (const [u, i] of EDGES) {
        const w = 1 / Math.sqrt(DU[u] * DI[i]);
        n[3 + i] += w * m[u];
        n[u] += w * m[3 + i];
      }
      m = n;
    }
    return m;
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w14-pr-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const m = masses(K);

    g.append("text").attr("x", 380).attr("y", 20).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("one unit of signal starts on u1 · each step multiplies by 1/√(dᵤdᵢ)");

    for (const [u, i] of EDGES) {
      // trim endpoints to the node borders (circle r=16, chip 28x28) so no
      // edge ever strikes through a label (figure grammar: no collisions)
      const [ux, uy] = UP[u], [ix, iy] = IP[i];
      const dx = ix - ux, dy = iy - uy, L = Math.hypot(dx, dy);
      const sx = ux + (dx / L) * 19, sy = uy + (dy / L) * 19;
      const trim = Math.min(17 / Math.abs(dx), dy === 0 ? Infinity : 17 / Math.abs(dy));
      g.append("line").attr("x1", sx).attr("y1", sy)
        .attr("x2", ix - dx * trim).attr("y2", iy - dy * trim)
        .attr("stroke", P.muted).attr("stroke-width", 1.6).attr("opacity", 0.4);
    }
    const fmt = (v) => v.toFixed(3);
    UP.forEach(([x, y], k) => {
      const v = m[k];
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", 16)
        .attr("fill", P.blue).attr("opacity", v > 0 ? 0.35 + 0.65 * Math.min(1, v) : 0.15);
      g.append("text").attr("x", x).attr("y", y + 4).attr("text-anchor", "middle")
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
        .attr("font-weight", 700).attr("fill", P.text).text("u" + (k + 1));
      g.append("text").attr("x", x - 26).attr("y", y + 4).attr("text-anchor", "end")
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
        .attr("fill", v > 0 ? P.text : P.muted).text(fmt(v));
    });
    IP.forEach(([x, y], k) => {
      const v = m[3 + k];
      g.append("rect").attr("x", x - 14).attr("y", y - 14).attr("width", 28)
        .attr("height", 28).attr("rx", 5).attr("fill", P.yellow)
        .attr("opacity", v > 0 ? 0.35 + 0.65 * Math.min(1, v) : 0.15);
      g.append("text").attr("x", x).attr("y", y + 4).attr("text-anchor", "middle")
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
        .attr("font-weight", 700).attr("fill", P.text).text("i" + (k + 1));
      g.append("text").attr("x", x + 24).attr("y", y + 4)
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
        .attr("fill", v > 0 ? P.text : P.muted).text(fmt(v));
    });

    g.append("text").attr("x", 380).attr("y", 292).attr("text-anchor", "middle")
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
      .attr("fill", P.text)
      .text("after " + K + " step" + (K === 1 ? "" : "s") + ":  mass(u2) = " +
        fmt(m[1]) + "  ·  mass(u3) = " + fmt(m[2]));
    const verdict = K < 2
      ? "no other USER has heard from u1 yet — items are not recommendations"
      : K < 4
        ? "K=2 is the co-consumption hop: u2 hears u1 through shared item i2 — " +
          "this is the signal K=0 (plain MF) never receives"
        : "by K=4 even u3 hears (0.063, via u2's items) — deep layers whisper";
    g.append("text").attr("x", 380).attr("y", 314).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
      .text(verdict);
  }

  for (const b of document.querySelectorAll("#w14-pr-widget [data-k]")) {
    b.addEventListener("click", () => {
      K = +b.dataset.k;
      for (const o of document.querySelectorAll("#w14-pr-widget [data-k]")) {
        o.classList.toggle("active", o === b);
      }
      render();
    });
  }
  U.onThemeChange(render);
  U.lazyBoot("w14-pr-svg", render);
})();
