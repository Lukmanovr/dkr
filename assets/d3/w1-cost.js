/* Widget 1.2 — What does storing a graph cost?
 * Sliders set n (nodes) and mean degree d̄; log-scale bars compare the entry
 * counts of the dense adjacency matrix (n²), the adjacency list (n + 2m), and
 * the edge list (2m), with real-dataset presets and a verdict line.
 */
(function () {
  "use strict";
  const U = window.DKR;

  const PRESETS = {
    cast: { label: "cast graph", n: 6, deg: 2.33 },
    cora: { label: "Cora", n: 2708, deg: 3.9 },
    arxiv: { label: "ogbn-arxiv", n: 169343, deg: 13.7 },
    products: { label: "ogbn-products", n: 2449029, deg: 50.5 },
    social: { label: "a social network", n: 1e9, deg: 200 },
  };

  let logN = Math.log10(PRESETS.cora.n);
  let logD = Math.log10(PRESETS.cora.deg);

  const fmt = (v) => {
    if (v >= 1e18) return (v / 1e18).toFixed(1) + " E";
    if (v >= 1e15) return (v / 1e15).toFixed(1) + " P";
    if (v >= 1e12) return (v / 1e12).toFixed(1) + " T";
    if (v >= 1e9) return (v / 1e9).toFixed(1) + " B";
    if (v >= 1e6) return (v / 1e6).toFixed(1) + " M";
    if (v >= 1e3) return (v / 1e3).toFixed(1) + " k";
    return String(Math.round(v));
  };
  const bytes = (entries) => {
    const b = entries * 4;
    if (b >= 1e18) return (b / 1e18).toFixed(1) + " EB";
    if (b >= 1e15) return (b / 1e15).toFixed(1) + " PB";
    if (b >= 1e12) return (b / 1e12).toFixed(1) + " TB";
    if (b >= 1e9) return (b / 1e9).toFixed(1) + " GB";
    if (b >= 1e6) return (b / 1e6).toFixed(1) + " MB";
    if (b >= 1e3) return (b / 1e3).toFixed(1) + " kB";
    return b.toFixed(0) + " B";
  };

  const W = 760, Hgt = 230;

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w1-ct-svg", W, Hgt);
    const n = Math.round(10 ** logN);
    const deg = 10 ** logD;
    const m = (n * deg) / 2;
    const rows = [
      { name: "dense matrix A", val: n * n, color: P.red, note: "n²" },
      { name: "adjacency list", val: n + 2 * m, color: P.yellow, note: "n + 2m" },
      { name: "edge list (COO)", val: 2 * m, color: P.green, note: "2m" },
    ];
    const xmax = Math.max(...rows.map((r) => r.val));
    const xs = d3.scaleLog().domain([1, Math.max(10, xmax)]).range([0, 420]);

    const g = svg.append("g").attr("transform", "translate(190,36)");
    rows.forEach((r, i) => {
      const y = i * 62;
      g.append("text").attr("x", -12).attr("y", y + 20).attr("text-anchor", "end")
        .attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.text).text(r.name);
      g.append("text").attr("x", -12).attr("y", y + 37).attr("text-anchor", "end")
        .attr("font-size", 12.5).attr("fill", P.muted).text(r.note);
      const bw = Math.max(3, xs(Math.max(1, r.val)));
      g.append("rect").attr("x", 0).attr("y", y).attr("height", 30).attr("rx", 6)
        .attr("width", bw)
        .attr("fill", r.color).attr("opacity", 0.85);
      const lbl = `${fmt(r.val)} entries ≈ ${bytes(r.val)}`;
      const fits = bw + 10 + lbl.length * 7 <= 560;   // keep labels inside the 760 canvas
      g.append("text")
        .attr("x", fits ? bw + 10 : bw)
        .attr("y", fits ? y + 20 : y + 45)
        .attr("text-anchor", fits ? "start" : "end")
        .attr("font-size", 12.5).attr("fill", P.text)
        .text(lbl);
    });
    g.append("text").attr("x", 210).attr("y", -14).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("fill", P.muted)
      .text(`n = ${fmt(n)} nodes · d̄ ≈ ${deg.toFixed(1)} · m ≈ ${fmt(m)} edges  (log-scale bars)`);

    const dense = n * n;
    const verdictEl = document.getElementById("w1-ct-verdict");
    if (dense * 4 > 64e9) {
      verdictEl.textContent = `verdict: the dense matrix needs ${bytes(dense)} — it will never exist. Sparse or nothing.`;
      verdictEl.style.color = P.red;
    } else if (dense * 4 > 16e9) {
      verdictEl.textContent = `verdict: ${bytes(dense)} dense — bigger than your GPU. Sparse wins.`;
      verdictEl.style.color = P.red;
    } else {
      verdictEl.textContent = `verdict: ${bytes(dense)} dense is storable — but ${(dense / (2 * m)).toFixed(0)}× the edge list, for the same information.`;
      verdictEl.style.color = "";
    }
    document.getElementById("w1-ct-nlabel").textContent = fmt(n);
    document.getElementById("w1-ct-dlabel").textContent = deg.toFixed(1);
  }

  document.getElementById("w1-ct-n").addEventListener("input", (e) => { logN = +e.target.value; render(); });
  document.getElementById("w1-ct-d").addEventListener("input", (e) => { logD = +e.target.value; render(); });
  document.querySelectorAll("#w1-ct-widget [data-preset]").forEach((b) =>
    b.addEventListener("click", () => {
      const p = PRESETS[b.getAttribute("data-preset")];
      logN = Math.log10(p.n); logD = Math.log10(p.deg);
      document.getElementById("w1-ct-n").value = logN;
      document.getElementById("w1-ct-d").value = logD;
      document.querySelectorAll("#w1-ct-widget [data-preset]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w1-ct-svg", render);
})();
