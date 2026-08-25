/* Widget 7.3 — How attentive is attention? GAT vs GATv2, trained live.
 * The GATv2 paper's dictionary-lookup toy: five query nodes must each attend to
 * their own key. Both scoring functions train here, in your browser, by real
 * gradient descent from the same seed. GAT's scores are provably "static" —
 * one neighbor ranking shared by every query — so it cannot learn the diagonal;
 * GATv2's one-line reordering can. Watch the theorem happen.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const NQ = 5, D = 10, H = 8, LR = 0.15, STEPS = 800;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const lrelu = (x) => (x > 0 ? x : 0.2 * x);
  const dlrelu = (x) => (x > 0 ? 1 : 0.2);

  // features: query q -> one-hot dim q; key k -> one-hot dim 5+k
  function softmaxRow(s) {
    const m = Math.max(...s);
    const e = s.map((x) => Math.exp(x - m));
    const z = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / z);
  }

  function trainGAT(seed) {
    // s(q,k) = lrelu(a1·W[:,q] + a2·W[:,5+k]) ; W: H x D, a1,a2: H
    const rng = mulberry32(seed);
    const W = Array.from({ length: H }, () => Array.from({ length: D }, () => rng() * 0.4 - 0.2));
    const a1 = Array.from({ length: H }, () => rng() * 0.4 - 0.2);
    const a2 = Array.from({ length: H }, () => rng() * 0.4 - 0.2);
    for (let step = 0; step < STEPS; step++) {
      for (let q = 0; q < NQ; q++) {
        const zq = W.map((row) => row[q]);
        const pre = [], sc = [];
        for (let k = 0; k < NQ; k++) {
          const zk = W.map((row) => row[5 + k]);
          let u = 0;
          for (let i = 0; i < H; i++) u += a1[i] * zq[i] + a2[i] * zk[i];
          pre.push(u); sc.push(lrelu(u));
        }
        const p = softmaxRow(sc);
        for (let k = 0; k < NQ; k++) {
          const gs = (p[k] - (k === q ? 1 : 0)) * dlrelu(pre[k]);
          const zk = W.map((row) => row[5 + k]);
          for (let i = 0; i < H; i++) {
            a1[i] -= LR * gs * zq[i];
            a2[i] -= LR * gs * zk[i];
            W[i][q] -= LR * gs * a1[i];
            W[i][5 + k] -= LR * gs * a2[i];
          }
        }
      }
    }
    return attnMatrix((q, k) => {
      let u = 0;
      for (let i = 0; i < H; i++) u += a1[i] * W[i][q] + a2[i] * W[i][5 + k];
      return lrelu(u);
    });
  }

  function trainGATv2(seed) {
    // s(q,k) = a · lrelu(W [hq ; hk]) ; W: H x 2D, a: H
    const rng = mulberry32(seed);
    const W = Array.from({ length: H }, () => Array.from({ length: 2 * D }, () => rng() * 0.4 - 0.2));
    const a = Array.from({ length: H }, () => rng() * 0.4 - 0.2);
    for (let step = 0; step < STEPS; step++) {
      for (let q = 0; q < NQ; q++) {
        const pres = [], sc = [];
        for (let k = 0; k < NQ; k++) {
          const u = W.map((row) => row[q] + row[D + 5 + k]);   // one-hots pick columns
          pres.push(u);
          sc.push(u.reduce((acc, ui, i) => acc + a[i] * lrelu(ui), 0));
        }
        const p = softmaxRow(sc);
        for (let k = 0; k < NQ; k++) {
          const gs = p[k] - (k === q ? 1 : 0);
          const u = pres[k];
          for (let i = 0; i < H; i++) {
            const gu = gs * a[i] * dlrelu(u[i]);
            a[i] -= LR * gs * lrelu(u[i]);
            W[i][q] -= LR * gu;
            W[i][D + 5 + k] -= LR * gu;
          }
        }
      }
    }
    return attnMatrix((q, k) => {
      const u = W.map((row) => row[q] + row[D + 5 + k]);
      return u.reduce((acc, ui, i) => acc + a[i] * lrelu(ui), 0);
    });
  }

  function attnMatrix(score) {
    const M = [];
    let correct = 0;
    for (let q = 0; q < NQ; q++) {
      const row = softmaxRow(Array.from({ length: NQ }, (_, k) => score(q, k)));
      M.push(row);
      if (row.indexOf(Math.max(...row)) === q) correct += 1;
    }
    return { M, correct };
  }

  let seed = 7, gat = null, gv2 = null;
  function run() {
    gat = trainGAT(seed);
    gv2 = trainGATv2(seed);
  }
  run();

  function heat(g, M, x0, y0, title, correct, P, color) {
    const C = 34;
    g.append("text").attr("x", x0 + 2.5 * C + 17).attr("y", y0 - 30).attr("text-anchor", "middle")
      .attr("font-size", 14).attr("font-weight", 700).attr("fill", color).text(title);
    g.append("text").attr("x", x0 + 2.5 * C + 17).attr("y", y0 - 12).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", correct >= 4 ? P.green : "#cf4a30")
      .text(`attends to own key: ${correct}/5`);
    for (let q = 0; q < NQ; q++) {
      g.append("text").attr("x", x0 - 8).attr("y", y0 + q * C + C / 2 + 4).attr("text-anchor", "end")
        .attr("font-size", 12.5).attr("fill", P.muted).text("q" + q);
      const am = M[q].indexOf(Math.max(...M[q]));
      for (let k = 0; k < NQ; k++) {
        const v = M[q][k];
        g.append("rect").attr("x", x0 + k * C + 17).attr("y", y0 + q * C)
          .attr("width", C - 3).attr("height", C - 3).attr("rx", 5)
          .attr("fill", color).attr("opacity", 0.08 + 0.85 * v)
          .attr("stroke", k === am ? P.text : "none").attr("stroke-width", 2);
        g.append("text").attr("x", x0 + k * C + C / 2 + 15).attr("y", y0 + q * C + C / 2 + 4)
          .attr("text-anchor", "middle").attr("font-size", 12.5)
          .attr("fill", v > 0.45 ? "#fff" : P.muted).text(v > 0.045 ? String(Math.min(99, Math.round(v * 100))) : "·");
      }
    }
    for (let k = 0; k < NQ; k++) {
      g.append("text").attr("x", x0 + k * C + C / 2 + 15).attr("y", y0 + NQ * C + 16)
        .attr("text-anchor", "middle").attr("font-size", 12.5).attr("fill", P.muted).text("k" + k);
    }
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w7-gt-svg", 760, 320);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    heat(g, gat.M, 90, 78, "GAT (2018)", gat.correct, P, "#cf4a30");
    heat(g, gv2.M, 450, 78, "GATv2 (2022)", gv2.correct, P, "#199473");
    g.append("text").attr("x", 380).attr("y", 30).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`task: query qᵢ must attend to key kᵢ · both trained ${STEPS} steps from seed ${seed} · outline = each row's argmax`);
    g.append("text").attr("x", 380).attr("y", 296).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.text)
      .text("GAT's argmax column is (nearly) the same in every row — static attention, the theorem in pixels");
    g.append("text").attr("x", 380).attr("y", 314).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("the one-line fix: nonlinearity inside — score = aᵀLeakyReLU(W[h‖h′]), not LeakyReLU(aᵀ[Wh‖Wh′])");
  }

  document.getElementById("w7-gt-retrain").addEventListener("click", () => {
    seed += 1; run(); render();
  });
  document.getElementById("w7-gt-reset").addEventListener("click", () => {
    seed = 7; run(); render();
  });

  U.onThemeChange(render);
  U.lazyBoot("w7-gt-svg", render);
})();
