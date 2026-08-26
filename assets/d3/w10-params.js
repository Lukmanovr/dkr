/* Widget 10.2 — The parameter bill, itemized.
 * Exact per-layer parameter counts for full per-relation weights, basis
 * decomposition, and block-diagonal weights, at your choice of R, d, B.
 * Pure arithmetic, log-scale bars. Deterministic.
 */
(function () {
  "use strict";
  const U = window.DKR;
  let R = 237, d = 100, B = 30;

  const fmt = (n) => n.toLocaleString("en-US");

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w10-pm-svg", 760, 300);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    const full = (2 * R + 1) * d * d;
    const basis = B * d * d + 2 * R * B + d * d;
    const block = 2 * R * Math.floor((d * d) / 10) + d * d;

    g.append("text").attr("x", 380).attr("y", 24).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`one R-GCN layer at R = ${R} relations (both directions), d = ${d}, B = ${B} bases`);

    const rows = [
      [`full  (2R+1)·d²`, full, P.accent],
      [`block-diagonal (10 blocks)  2R·d²/10 + d²`, block, "#7c5cd6"],
      [`basis  B·d² + 2R·B + d²`, basis, P.green],
    ];
    const lmax = Math.log10(Math.max(full, block, basis, 1e6));
    const lmin = 4;   // 10k floor
    rows.forEach(([lbl, n, col], i) => {
      const y = 72 + i * 62;
      const w = Math.max(8, 560 * (Math.log10(n) - lmin) / (lmax - lmin));
      g.append("text").attr("x", 60).attr("y", y - 8).attr("font-size", 12.5)
        .attr("fill", P.text).text(lbl);
      g.append("rect").attr("x", 60).attr("y", y).attr("width", w).attr("height", 22)
        .attr("rx", 6).attr("fill", col).attr("opacity", 0.85);
      g.append("text").attr("x", 60 + w + 10).attr("y", y + 16)
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
        .attr("fill", P.text).text(fmt(n));
    });

    const ratio = full / basis;
    g.append("text").attr("x", 380).attr("y", 262).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.green)
      .text(`basis decomposition: ${ratio.toFixed(1)}× fewer parameters — `
        + `${B} shared matrices + ${fmt(2 * R * B)} coefficients`);
    const warn = R >= 200 && B <= 40
      ? "at KG scale the full version is untrainable luggage; the basis version fits in a coat pocket"
      : R <= 30
      ? "at few relations the decomposition buys little — full weights are affordable and stronger"
      : "the trade: fewer parameters, but relations must SHARE structure through the bases";
    g.append("text").attr("x", 380).attr("y", 286).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text(warn);
  }

  const wire = (attr, setter) => document.querySelectorAll(`#w10-pm-widget [${attr}]`).forEach((b) =>
    b.addEventListener("click", () => {
      setter(+b.getAttribute(attr));
      document.querySelectorAll(`#w10-pm-widget [${attr}]`).forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  wire("data-r", (v) => { R = v; });
  wire("data-d", (v) => { d = v; });
  wire("data-b", (v) => { B = v; });

  U.onThemeChange(render);
  U.lazyBoot("w10-pm-svg", render);
})();
