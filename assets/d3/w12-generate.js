/* Widget 12.4 — Judge the generator by its statistics.
 * Real sample graphs baked from our experiment: training community graphs,
 * GraphRNN-S samples, and ER baseline samples. The widget draws four of the
 * chosen set and computes degree/clustering histograms plus total-variation
 * distance to the TRAINING distribution, live. Deterministic.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const DATA = window.DKR_W12_GEN;

  function stats(edgeLists) {
    const degs = [], clus = [];
    edgeLists.forEach((edges) => {
      const adj = {};
      edges.forEach(([a, b]) => {
        (adj[a] = adj[a] || new Set()).add(b);
        (adj[b] = adj[b] || new Set()).add(a);
      });
      Object.keys(adj).forEach((v) => {
        const nb = [...adj[v]];
        degs.push(nb.length);
        let t = 0;
        for (let i = 0; i < nb.length; i++)
          for (let j = i + 1; j < nb.length; j++)
            if (adj[nb[i]] && adj[nb[i]].has(nb[j])) t++;
        clus.push(nb.length > 1 ? 2 * t / (nb.length * (nb.length - 1)) : 0);
      });
    });
    return { degs, clus };
  }

  function hist(vals, bins, lo, hi) {
    const h = new Array(bins).fill(0);
    vals.forEach((v) => {
      const k = Math.min(bins - 1, Math.max(0, Math.floor((v - lo) / (hi - lo) * bins)));
      h[k] += 1;
    });
    const s = h.reduce((a, b) => a + b, 0) || 1;
    return h.map((x) => x / s);
  }
  const tv = (h1, h2) => h1.reduce((s, v, i) => s + Math.abs(v - h2[i]), 0) / 2;

  const TRAIN_STATS = stats(DATA.train);
  const TH_DEG = hist(TRAIN_STATS.degs, 14, 0, 14);
  const TH_CLU = hist(TRAIN_STATS.clus, 10, 0, 1);

  let src = "model";

  function drawGraph(g, edges, x0, y0, w, P, col) {
    const nodes = new Set();
    edges.forEach(([a, b]) => { nodes.add(a); nodes.add(b); });
    const ns = [...nodes];
    const pos = {};
    ns.forEach((v, i) => {
      const ang = (2 * Math.PI * i) / ns.length;
      pos[v] = [x0 + w / 2 + (w / 2 - 8) * Math.cos(ang), y0 + w / 2 + (w / 2 - 8) * Math.sin(ang)];
    });
    edges.forEach(([a, b]) => {
      g.append("line").attr("x1", pos[a][0]).attr("y1", pos[a][1])
        .attr("x2", pos[b][0]).attr("y2", pos[b][1])
        .attr("stroke", col).attr("stroke-width", 1).attr("opacity", 0.5);
    });
    ns.forEach((v) => {
      g.append("circle").attr("cx", pos[v][0]).attr("cy", pos[v][1]).attr("r", 2.6)
        .attr("fill", col);
    });
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w12-ge-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const COLS = { train: P.yellow, model: P.green, er: P.accent };
    const NAMES = { train: "training graphs", model: "GraphRNN-S samples", er: "ER baseline samples" };

    g.append("text").attr("x", 380).attr("y", 22).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`${NAMES[src]} — four real samples, and their statistics against the training distribution`);

    DATA[src].slice(0, 4).forEach((edges, i) => {
      drawGraph(g, edges, 40 + i * 120, 40, 100, P, COLS[src]);
    });

    const S = stats(DATA[src]);
    const hd = hist(S.degs, 14, 0, 14);
    const hc = hist(S.clus, 10, 0, 1);
    const tvd = tv(hd, TH_DEG), tvc = tv(hc, TH_CLU);

    // degree histogram bars vs train
    const bx = 70;
    g.append("text").attr("x", bx).attr("y", 188).attr("font-size", 12.5)
      .attr("font-weight", 700).attr("fill", P.text).text("degree distribution (outline = training)");
    hd.forEach((v, i) => {
      g.append("rect").attr("x", bx + i * 22).attr("y", 260 - 55 * v / 0.4)
        .attr("width", 18).attr("height", 55 * v / 0.4).attr("fill", COLS[src]).attr("opacity", 0.75);
      g.append("rect").attr("x", bx + i * 22).attr("y", 260 - 55 * TH_DEG[i] / 0.4)
        .attr("width", 18).attr("height", 55 * TH_DEG[i] / 0.4)
        .attr("fill", "none").attr("stroke", P.text).attr("stroke-width", 1);
    });
    const cx = 470;
    g.append("text").attr("x", cx).attr("y", 188).attr("font-size", 12.5)
      .attr("font-weight", 700).attr("fill", P.text).text("clustering (outline = training)");
    hc.forEach((v, i) => {
      g.append("rect").attr("x", cx + i * 24).attr("y", 260 - 55 * v / 0.6)
        .attr("width", 20).attr("height", 55 * v / 0.6).attr("fill", COLS[src]).attr("opacity", 0.75);
      g.append("rect").attr("x", cx + i * 24).attr("y", 260 - 55 * TH_CLU[i] / 0.6)
        .attr("width", 20).attr("height", 55 * TH_CLU[i] / 0.6)
        .attr("fill", "none").attr("stroke", P.text).attr("stroke-width", 1);
    });
    g.append("text").attr("x", 380).attr("y", 296).attr("text-anchor", "middle")
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
      .attr("font-weight", 700).attr("fill", P.text)
      .text(`TV to training: degree ${tvd.toFixed(3)} · clustering ${tvc.toFixed(3)}`);
    const verdict = src === "train" ? "the reference against itself — the floor any generator chases"
      : src === "model" ? "the RNN learned the community signature: triangles inside, sparsity between"
      : "right density, wrong texture: ER has no communities and the clustering column says so";
    g.append("text").attr("x", 380).attr("y", 318).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.accentDark).text(verdict);
  }

  document.querySelectorAll("#w12-ge-widget [data-src]").forEach((b) =>
    b.addEventListener("click", () => {
      src = b.getAttribute("data-src");
      document.querySelectorAll("#w12-ge-widget [data-src]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w12-ge-svg", render);
})();
