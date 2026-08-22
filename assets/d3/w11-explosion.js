/* Widget 11.1 — The receptive-field bill, computed live.
 * Choose depth L, branching b, and a fanout cap f: the widget computes the
 * worst-case number of nodes one update touches, sum over hops of min(b,f)^k,
 * and overlays the MEASURED ogbn-arxiv numbers (which are smaller than the
 * bound because real neighborhoods overlap). Deterministic arithmetic.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const MEASURED = { 1: 18, 2: 4577, 3: 22663 };   // arxiv, 200-seed average
  let L = 3, b = 14, f = Infinity;

  const fmt = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n.toLocaleString("en-US"));

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w11-ex-svg", 760, 300);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const eff = Math.min(b, f);
    let terms = [], total = 1;
    for (let k = 1; k <= L; k++) {
      const t = Math.pow(eff, k);
      terms.push(t);
      total += t;
    }
    g.append("text").attr("x", 380).attr("y", 24).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`worst case: 1 + ${terms.map((t, i) => `${eff}^${i + 1}`).join(" + ")} nodes loaded to update ONE node`);

    const y0 = 70;
    terms.forEach((t, i) => {
      const w = Math.max(6, 480 * Math.log10(t + 1) / Math.log10(3e6));
      g.append("text").attr("x", 120).attr("y", y0 + i * 44 + 14).attr("text-anchor", "end")
        .attr("font-size", 12.5).attr("fill", P.text).text(`hop ${i + 1}`);
      g.append("rect").attr("x", 132).attr("y", y0 + i * 44).attr("width", w).attr("height", 22)
        .attr("rx", 6).attr("fill", P.accent).attr("opacity", 0.55 + 0.15 * i);
      g.append("text").attr("x", 138 + w).attr("y", y0 + i * 44 + 15)
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
        .attr("fill", P.text).text(fmt(t));
    });
    const yT = y0 + L * 44 + 10;
    g.append("text").attr("x", 132).attr("y", yT + 8).attr("font-size", 13.5)
      .attr("font-weight", 700).attr("fill", P.text)
      .text(`total ≤ ${fmt(total)} nodes per updated node`);

    let note;
    if (f === Infinity && b === 14 && MEASURED[L]) {
      note = `measured on arxiv (overlaps shrink the bound): ${fmt(MEASURED[L])} — still ${Math.round(100 * MEASURED[L] / 169343)}% of the graph at L=3`;
    } else if (f !== Infinity) {
      note = `the cap does the work: cost is ${fmt(total)} REGARDLESS of how hubby the graph is`;
    } else {
      note = "unbounded: one hub in the neighborhood and the batch loads its whole crowd";
    }
    g.append("text").attr("x", 380).attr("y", 282).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", f === Infinity ? P.accentDark : P.green)
      .text(note);
  }

  const wire = (attr, setter) => document.querySelectorAll(`#w11-ex-widget [${attr}]`).forEach((btn) =>
    btn.addEventListener("click", () => {
      setter(btn.getAttribute(attr));
      document.querySelectorAll(`#w11-ex-widget [${attr}]`).forEach((x) => x.classList.toggle("active", x === btn));
      render();
    }));
  wire("data-l", (v) => { L = +v; });
  wire("data-b", (v) => { b = +v; });
  wire("data-f", (v) => { f = v === "inf" ? Infinity : +v; });

  U.onThemeChange(render);
  U.lazyBoot("w11-ex-svg", render);
})();
