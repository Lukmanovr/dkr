/* Widget 4.4 — Filtered ranking, on facts the model never saw.
 * Two facts are held out; a tiny TransE trains on the other ten at boot
 * (seeded, instant). Each query ranks every entity as the missing head; the
 * filter toggle removes known-true training competitors before ranking —
 * the protocol every KG paper reports. MRR and Hits@1 update across queries.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const ENT = ["IU", "KFU", "Innopolis", "Kazan", "Tatarstan", "Russia", "Volga", "university", "city", "region"];
  const RNAME = ["located_in", "instance_of", "capital_of", "part_of", "flows_through"];
  const ALL = [
    [0, 1, 7], [0, 0, 2], [2, 0, 4], [3, 2, 4], [3, 0, 4], [4, 3, 5],
    [1, 1, 7], [1, 0, 3], [2, 1, 8], [3, 1, 8], [4, 1, 9], [6, 4, 4],
  ];
  // held-out test facts (head prediction): (Kazan, located_in, Tatarstan), (KFU, instance_of, university)
  const TEST = [[3, 0, 4], [1, 1, 7]];
  const TRAIN = ALL.filter(([h, r, t]) => !TEST.some(([a, b, c]) => a === h && b === r && c === t));
  const NE = ENT.length, NR = 5;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const rng = mulberry32(29);
  const E = Array.from({ length: NE }, () => [rng() * 4 - 2, rng() * 4 - 2]);
  const R = Array.from({ length: NR }, () => [rng() * 1 - 0.5, rng() * 1 - 0.5]);
  const resid = (h, r, t) => Math.hypot(E[h][0] + R[r][0] - E[t][0], E[h][1] + R[r][1] - E[t][1]);
  for (let s = 0; s < 6000; s++) {
    const [h, r, t] = TRAIN[Math.floor(rng() * TRAIN.length)];
    let ch = h, ct = t;
    if (rng() < 0.5) { do { ch = Math.floor(rng() * NE); } while (ch === h); }
    else { do { ct = Math.floor(rng() * NE); } while (ct === t); }
    const dp = resid(h, r, t), dn = resid(ch, r, ct);
    if (1.0 + dp - dn <= 0) continue;
    const up = [(E[h][0] + R[r][0] - E[t][0]) / (dp + 1e-9), (E[h][1] + R[r][1] - E[t][1]) / (dp + 1e-9)];
    const un = [(E[ch][0] + R[r][0] - E[ct][0]) / (dn + 1e-9), (E[ch][1] + R[r][1] - E[ct][1]) / (dn + 1e-9)];
    for (const k of [0, 1]) {
      E[h][k] -= 0.04 * up[k]; E[t][k] += 0.04 * up[k];
      E[ch][k] += 0.04 * un[k]; E[ct][k] -= 0.04 * un[k];
      R[r][k] -= 0.04 * (up[k] - un[k]);
    }
  }

  const TRAIN_SET = new Set(TRAIN.map(([h, r, t]) => `${h},${r},${t}`));
  let qi = 0;

  function rankingFor(test, filtered) {
    const [th, tr, tt] = test;
    let cands = [];
    for (let e = 0; e < NE; e++) {
      const known = TRAIN_SET.has(`${e},${tr},${tt}`);
      if (filtered && known) continue;
      cands.push({ e, score: -resid(e, tr, tt), known });
    }
    cands.sort((a, b) => b.score - a.score);
    const rank = cands.findIndex((c) => c.e === th) + 1;
    return { cands, rank };
  }

  function filteredOn() { return document.getElementById("w4-rk-filter").checked; }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w4-rk-svg", 760, 350);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    const filt = filteredOn();
    const [th, tr, tt] = TEST[qi];
    const { cands, rank } = rankingFor(TEST[qi], filt);
    const { rank: rawRank } = rankingFor(TEST[qi], false);
    const { rank: filtRank } = rankingFor(TEST[qi], true);

    g.append("text").attr("x", 380).attr("y", 24).attr("text-anchor", "middle")
      .attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.text)
      .text(`held-out fact: (?, ${RNAME[tr]}, ${ENT[tt]}) — truth: ${ENT[th]} (the model never saw it)`);

    const smax = cands[0].score, smin = cands[cands.length - 1].score;
    cands.slice(0, 8).forEach((c, i) => {
      const y = 44 + i * 29;
      const isTruth = c.e === th;
      const bw = 20 + 300 * ((c.score - smin) / (smax - smin || 1));
      g.append("rect").attr("x", 200).attr("y", y).attr("width", bw).attr("height", 21).attr("rx", 7)
        .attr("fill", isTruth ? P.green : c.known ? "#7c5cd6" : P.muted)
        .attr("opacity", isTruth ? 0.95 : c.known ? 0.85 : 0.4);
      g.append("text").attr("x", 192).attr("y", y + 15.5).attr("text-anchor", "end")
        .attr("font-size", 12.5).attr("font-weight", isTruth || c.known ? 700 : 400)
        .attr("fill", isTruth ? P.green : c.known ? "#7c5cd6" : P.text)
        .text(`${i + 1} · ${ENT[c.e]}`);
      g.append("text").attr("x", 208 + bw).attr("y", y + 15.5)
        .attr("font-size", 12).attr("fill", P.muted).text(c.score.toFixed(2));
      if (c.known) {
        g.append("text").attr("x", 208 + bw + 46).attr("y", y + 15.5)
          .attr("font-size", 12).attr("fill", "#7c5cd6").text("known true (train)");
      }
    });

    // metrics over both test queries
    const ranksRaw = TEST.map((q) => rankingFor(q, false).rank);
    const ranksFilt = TEST.map((q) => rankingFor(q, true).rank);
    const mrr = (rs) => rs.reduce((a, r) => a + 1 / r, 0) / rs.length;
    const hits1 = (rs) => rs.filter((r) => r === 1).length / rs.length;
    g.append("text").attr("x", 380).attr("y", 300).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", P.text)
      .text(`this query: raw rank ${rawRank} → filtered rank ${filtRank}`);
    g.append("text").attr("x", 380).attr("y", 322).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 700)
      .attr("fill", filt ? P.green : P.muted)
      .text(`both test facts — raw: MRR ${mrr(ranksRaw).toFixed(2)}, Hits@1 ${hits1(ranksRaw).toFixed(2)} · ` +
            `filtered: MRR ${mrr(ranksFilt).toFixed(2)}, Hits@1 ${hits1(ranksFilt).toFixed(2)}`);
    g.append("text").attr("x", 380).attr("y", 342).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", P.muted)
      .text(filt ? "purple competitors removed before ranking — the number papers report"
                 : "tick “filtered” — the purple bars are training facts the raw metric wrongly punishes");
  }

  document.getElementById("w4-rk-next").addEventListener("click", () => {
    qi = (qi + 1) % TEST.length;
    render();
  });
  document.getElementById("w4-rk-filter").addEventListener("change", render);

  U.onThemeChange(render);
  U.lazyBoot("w4-rk-svg", render);
})();
