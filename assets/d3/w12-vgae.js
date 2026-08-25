/* Widget 12.3 — A real VGAE latent space, decoded before your eyes.
 * The 2-D latent means of a VGAE trained on the karate club (baked by
 * scripts/figgen/w12_figs.py; reconstruction AUC 0.929). Slide the decoder
 * threshold: an edge is predicted wherever sigma(z_u . z_v) > t. Counts of
 * true/false/missed edges update live. Deterministic.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const D = window.DKR_W12_VGAE;
  const N = D.z.length;
  const edgeSet = new Set(D.edges.map(([a, b]) => a * 100 + b));
  let thr = 0.7;

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w12-vg-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    const xs = D.z.map((p) => p[0]), ys = D.z.map((p) => p[1]);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    const px = (x) => 90 + 580 * (x - xmin) / (xmax - xmin);
    const py = (y) => 60 + 210 * (y - ymin) / (ymax - ymin);

    let tp = 0, fp = 0;
    const logit = (a, b) => D.z[a][0] * D.z[b][0] + D.z[a][1] * D.z[b][1];
    const sig = (v) => 1 / (1 + Math.exp(-v));
    for (let a = 0; a < N; a++) {
      for (let b = a + 1; b < N; b++) {
        if (sig(logit(a, b)) > thr) {
          const real = edgeSet.has(a * 100 + b);
          if (real) tp++; else fp++;
          g.append("line").attr("x1", px(D.z[a][0])).attr("y1", py(D.z[a][1]))
            .attr("x2", px(D.z[b][0])).attr("y2", py(D.z[b][1]))
            .attr("stroke", real ? P.green : P.accent)
            .attr("stroke-width", real ? 1.6 : 1.7).attr("opacity", real ? 0.55 : 0.65);
        }
      }
    }
    for (let v = 0; v < N; v++) {
      g.append("circle").attr("cx", px(D.z[v][0])).attr("cy", py(D.z[v][1])).attr("r", 6)
        .attr("fill", P.blue).attr("opacity", 0.9);
    }
    const missed = D.edges.length - tp;
    g.append("text").attr("x", 380).attr("y", 24).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`the karate club, as its VGAE sees it — decode: edge wherever sigmoid(z·z) > ${thr.toFixed(2)}`);
    g.append("text").attr("x", 380).attr("y", 300).attr("text-anchor", "middle")
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
      .attr("fill", P.text)
      .text(`true edges recovered ${tp}/${D.edges.length} · false edges invented ${fp} · missed ${missed}`);
    const verdict = thr <= 0.55 ? "low bar: nearly everything close gets an edge — recall bought with fiction"
      : thr >= 0.85 ? "high bar: only the surest pairs — precision bought with amnesia"
      : "the trade every decoder makes: geometry proposes, the threshold disposes";
    g.append("text").attr("x", 380).attr("y", 322).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.accentDark)
      .text(verdict + `  (baked model: recon AUC ${D.auc})`);
  }

  document.querySelectorAll("#w12-vg-widget [data-t]").forEach((b) =>
    b.addEventListener("click", () => {
      thr = +b.getAttribute("data-t");
      document.querySelectorAll("#w12-vg-widget [data-t]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w12-vg-svg", render);
})();
