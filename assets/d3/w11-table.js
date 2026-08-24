/* Widget 11.4 — The scaling table, sortable, with the products stress test.
 * Measured on ogbn-arxiv (scripts/experiments/w11_experiment.py, 2026-08-22, course
 * GPU): accuracy / s-per-epoch / TRAINING-peak MB. The "products" toggle
 * scales memory by the node ratio (2.4M / 169k ≈ 14.5×) — SGC's feature
 * table scales too, but its GPU-resident training slice does not.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const ROWS = [
    ["full-batch GCN", 0.665, "0.10", 2821, "everything resident", true],
    ["sampled SAGE (f=10)", 0.605, "12.2", 64, "batch-bounded", false],
    ["Cluster-GCN (random parts)", 0.628, "0.51", 31, "one part resident", false],
    ["SGC (K=2 precompute)", 0.638, "0.0015", 171, "train split resident", false],
  ];
  const SCALE = 2449029 / 169343;
  let sortBy = "acc", products = false;

  function render() {
    const P = U.pal();
    const COLS = { "full-batch GCN": P.accent, "sampled SAGE (f=10)": P.blue,
                   "Cluster-GCN (random parts)": "#7c5cd6", "SGC (K=2 precompute)": P.green };
    const svg = U.svgIn("w11-tb-svg", 760, 310);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const rows = ROWS.slice().sort((a, b) =>
      sortBy === "acc" ? b[1] - a[1] : sortBy === "time" ? parseFloat(a[2]) - parseFloat(b[2]) : a[3] - b[3]);

    g.append("text").attr("x", 380).attr("y", 22).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(products
        ? "the products stress test: memory × 14.5 (node ratio) — who survives an 8 GB card?"
        : "measured on ogbn-arxiv — sorted by " + (sortBy === "acc" ? "accuracy" : sortBy === "time" ? "epoch time" : "memory"));

    rows.forEach(([name, acc, spe, mem, note, resident], i) => {
      const y = 62 + i * 52;
      const m = products && resident ? mem * SCALE : products ? Math.round(mem * (name.startsWith("SGC") ? 1 : SCALE)) : mem;
      const fits = m < 8192;
      g.append("text").attr("x", 60).attr("y", y).attr("font-size", 13)
        .attr("font-weight", 700).attr("fill", COLS[name]).text(name);
      g.append("text").attr("x", 60).attr("y", y + 18).attr("font-size", 11.5 < 12 ? 12 : 12)
        .attr("fill", P.muted).text(note);
      g.append("text").attr("x", 330).attr("y", y + 8)
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
        .attr("fill", P.text).text(`acc ${acc.toFixed(3)}`);
      g.append("text").attr("x", 440).attr("y", y + 8)
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
        .attr("fill", P.text).text(`${spe} s/ep`);
      const memtxt = m >= 1024 ? (m / 1024).toFixed(1) + " GB" : Math.round(m) + " MB";
      g.append("text").attr("x", 560).attr("y", y + 8)
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
        .attr("font-weight", 700).attr("fill", products ? (fits ? P.green : "#cf4a30") : P.text)
        .text(memtxt + (products ? (fits ? "  ✓ fits" : "  ✗ OOM") : ""));
    });
    g.append("text").attr("x", 380).attr("y", 296).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.accentDark)
      .text(products
        ? "full-batch dies first — exactly the wall the samplers were invented for"
        : "no row wins every column: pick the constraint that binds YOUR deployment, then read its row");
  }

  document.querySelectorAll("#w11-tb-widget [data-sort]").forEach((btn) =>
    btn.addEventListener("click", () => {
      sortBy = btn.getAttribute("data-sort");
      products = false;
      document.querySelectorAll("#w11-tb-widget [data-sort]").forEach((x) => x.classList.toggle("active", x === btn));
      document.getElementById("w11-tb-products").classList.remove("active");
      render();
    }));
  document.getElementById("w11-tb-products").addEventListener("click", (ev) => {
    products = !products;
    ev.target.classList.toggle("active", products);
    render();
  });

  U.onThemeChange(render);
  U.lazyBoot("w11-tb-svg", render);
})();
