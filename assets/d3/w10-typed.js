/* Widget 10.1 — One typed update, dials exposed.
 * The lecture's five-node academic toy. The update at paper P1 is recomputed
 * live: choose the cites-channel weight, or collapse all types into one
 * shared weight, and watch which voices survive. Deterministic arithmetic.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const POS = { A1: [180, 60], A2: [330, 60], P1: [205, 165], P2: [365, 165], V: [285, 255] };
  const FEAT = { A1: 1, A2: 2, P1: 1, P2: 3, V: 2 };
  const EDGES = [["A1", "P1", "writes"], ["A2", "P1", "writes"], ["A2", "P2", "writes"],
    ["P2", "P1", "cites"], ["P1", "V", "pub"], ["P2", "V", "pub"]];

  let wCites = 1, typed = true;

  function render() {
    const P = U.pal();
    const RELCOL = { writes: P.green, cites: "#7c5cd6", pub: P.accent };
    const svg = U.svgIn("w10-ty-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    g.append("text").attr("x", 380).attr("y", 24).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(typed ? "each relation has its own weight — the cites dial is yours"
                  : "types collapsed: ONE shared weight ½, one plain mean over all neighbors");

    svg.append("defs").html('<marker id="w10ty-arr" viewBox="0 0 10 10" refX="8" refY="5" ' +
      'markerWidth="6" markerHeight="6" orient="auto">' +
      '<path d="M 0 0 L 10 5 L 0 10 z" fill="' + P.muted + '"/></marker>');
    EDGES.forEach(([a, b, rel]) => {
      const [x1, y1] = POS[a], [x2, y2] = POS[b];
      const dx = x2 - x1, dy = y2 - y1, ln = Math.hypot(dx, dy);
      g.append("line")
        .attr("x1", x1 + (dx / ln) * 18).attr("y1", y1 + (dy / ln) * 18)
        .attr("x2", x2 - (dx / ln) * 20).attr("y2", y2 - (dy / ln) * 20)
        .attr("stroke", typed ? RELCOL[rel] : P.muted).attr("stroke-width", 2)
        .attr("stroke-dasharray", rel === "cites" ? "5 3" : null)
        .attr("marker-end", "url(#w10ty-arr)");
    });
    Object.keys(POS).forEach((v) => {
      const [x, y] = POS[v];
      const col = v[0] === "A" ? P.yellow : v[0] === "P" ? P.blue : P.accent;
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", v === "P1" ? 20 : 15)
        .attr("fill", col).attr("stroke", v === "P1" ? P.text : "none").attr("stroke-width", 2);
      g.append("text").attr("x", x).attr("y", y + 4).attr("text-anchor", "middle")
        .attr("font-size", 13).attr("font-weight", 700).attr("fill", P.bg).text(v);
      // P1's upper-right is a three-arrowhead junction: its h-label moves to
      // the free lower-left corner with a short leader instead
      if (v === "P1") {
        g.append("line").attr("x1", x - 15).attr("y1", y + 15).attr("x2", x - 24).attr("y2", y + 24)
          .attr("stroke", P.muted).attr("stroke-width", 1);
        g.append("text").attr("x", x - 27).attr("y", y + 32).attr("text-anchor", "end")
          .attr("font-size", 12.5)
          .attr("font-family", "'JetBrains Mono', monospace").attr("fill", P.muted).text("h=" + FEAT[v]);
      } else {
        g.append("text").attr("x", x + 26).attr("y", y - 10).attr("font-size", 12.5)
          .attr("font-family", "'JetBrains Mono', monospace").attr("fill", P.muted).text("h=" + FEAT[v]);
      }
    });

    // the ledger
    const L = 470;
    let lines, total;
    if (typed) {
      const rows = [
        ["self      1 × 1", 1.0, P.text],
        ["writes    ½ × mean{1,2}", 0.75, P.green],
        [`cites     ${wCites} × 3`, wCites * 3, "#7c5cd6"],
        ["pub⁻¹     ¼ × 2", 0.5, P.accent],
      ];
      total = rows.reduce((s, r) => s + r[1], 0);
      lines = rows.map(([t, v, c]) => [t + " = " + v.toFixed(2), c]);
    } else {
      total = 1 + 0.5 * (1 + 2 + 3 + 2) / 4;
      lines = [["self     1 × 1 = 1.00", P.text],
               ["all      ½ × mean{1,2,3,2} = 1.00", P.muted]];
    }
    g.append("text").attr("x", L).attr("y", 78).attr("font-size", 13)
      .attr("font-weight", 700).attr("fill", P.text).text("update at P1:");
    lines.forEach(([t, c], i) => {
      g.append("text").attr("x", L).attr("y", 106 + i * 26)
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
        .attr("fill", c).text(t);
    });
    g.append("text").attr("x", L).attr("y", 116 + lines.length * 26)
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 14)
      .attr("font-weight", 700).attr("fill", P.text).text("h′(P1) = " + total.toFixed(2));

    let verdict;
    if (!typed) verdict = "one volume for every voice — the citation is just another neighbor";
    else if (wCites === 0) verdict = "citations muted: P2's signal is gone from P1 entirely";
    else if (wCites >= 2) verdict = "citations amplified 2×: one relation now dominates the update";
    else verdict = "typed weights let the network learn which relations matter — per relation";
    g.append("text").attr("x", 380).attr("y", 316).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.accentDark).text(verdict);
  }

  document.querySelectorAll("#w10-ty-widget [data-wc]").forEach((b) =>
    b.addEventListener("click", () => {
      wCites = +b.getAttribute("data-wc");
      typed = true;
      document.querySelectorAll("#w10-ty-widget [data-wc]").forEach((x) => x.classList.toggle("active", x === b));
      document.getElementById("w10-ty-untyped").classList.remove("active");
      render();
    }));
  document.getElementById("w10-ty-untyped").addEventListener("click", (ev) => {
    typed = !typed;
    ev.target.classList.toggle("active", !typed);
    render();
  });

  U.onThemeChange(render);
  U.lazyBoot("w10-ty-svg", render);
})();
