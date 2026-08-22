/* Widget 14.3 — Three protocols, one truth.
 * One user's 12 interactions on a timeline; each protocol paints them
 * train/val/test. The measured LightGCN K=3 Recall@20 for each protocol
 * (scratchpad/w14_experiment.py + _2.py, ML-100k, seed 0, 2026-08-22):
 * random 0.3462 · leave-last-out 0.2096 · global temporal 0.1953.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const N = 12;
  const PROTO = {
    random: {
      te: [2, 6, 10], va: [5], score: "0.346", users: "943 test users",
      note: "interactions 11 and 12 are TRAINED ON, then the model is asked " +
        "to 'predict' number 6 — the future informs the past",
      verdict: "flattered: ×1.77 vs the honest number",
    },
    llo: {
      te: [12], va: [11], score: "0.210", users: "940 test users",
      note: "each user's last item held out — per-user time is respected, " +
        "but user A's 'past' may postdate user B's 'future'",
      verdict: "the literature's usual compromise",
    },
    temporal: {
      te: [11, 12], va: [10], score: "0.195", users: "66 test users survive the wall",
      note: "a hard wall at the 80% timestamp: nothing after it is ever " +
        "seen in training — deployment's actual contract",
      verdict: "honest — and the number everyone else inflates",
    },
  };
  let mode = "temporal";

  function render() {
    const P = U.pal();
    const cfg = PROTO[mode];
    const svg = U.svgIn("w14-sp-svg", 760, 250);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    g.append("text").attr("x", 380).attr("y", 20).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("one user's 12 interactions, in time order →");

    g.append("line").attr("x1", 70).attr("y1", 92).attr("x2", 700).attr("y2", 92)
      .attr("stroke", P.muted).attr("stroke-width", 1.5);
    for (let t = 1; t <= N; t++) {
      const x = 85 + (t - 1) * 54;
      const role = cfg.te.includes(t) ? "test" : cfg.va.includes(t) ? "val" : "train";
      const col = role === "test" ? P.accent : role === "val" ? P.yellow : P.blue;
      g.append("circle").attr("cx", x).attr("cy", 92).attr("r", 13)
        .attr("fill", col).attr("opacity", role === "train" ? 0.55 : 0.95);
      g.append("text").attr("x", x).attr("y", 96).attr("text-anchor", "middle")
        .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12)
        .attr("font-weight", 700).attr("fill", "#fff").text(t);
      g.append("text").attr("x", x).attr("y", 124).attr("text-anchor", "middle")
        .attr("font-size", 12).attr("fill", col)
        .text(role === "train" ? "" : role);
    }
    if (mode === "temporal") {
      const wx = 85 + 8.5 * 54;
      g.append("line").attr("x1", wx).attr("y1", 56).attr("x2", wx).attr("y2", 128)
        .attr("stroke", P.accent).attr("stroke-width", 2.5)
        .attr("stroke-dasharray", "6 4");
      g.append("text").attr("x", wx).attr("y", 48).attr("text-anchor", "middle")
        .attr("font-size", 12).attr("font-weight", 700).attr("fill", P.accent)
        .text("the time wall");
    }

    g.append("text").attr("x", 380).attr("y", 158).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.text).text(cfg.note);
    g.append("text").attr("x", 380).attr("y", 192).attr("text-anchor", "middle")
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 15)
      .attr("font-weight", 700).attr("fill", P.text)
      .text("measured LightGCN K=3 test Recall@20: " + cfg.score);
    g.append("text").attr("x", 380).attr("y", 214).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", P.muted).text(cfg.users);
    g.append("text").attr("x", 380).attr("y", 236).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.accentDark)
      .text(cfg.verdict);
  }

  for (const b of document.querySelectorAll("#w14-sp-widget [data-p]")) {
    b.addEventListener("click", () => {
      mode = b.dataset.p;
      for (const o of document.querySelectorAll("#w14-sp-widget [data-p]")) {
        o.classList.toggle("active", o === b);
      }
      render();
    });
  }
  U.onThemeChange(render);
  U.lazyBoot("w14-sp-svg", render);
})();
