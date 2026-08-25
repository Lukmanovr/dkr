/* Widget 5.1 — Two hops over an incomplete KG: traversal vs soft completion.
 * The biomedical toy graph from the hero figure. Traversal mode follows stated
 * edges only; soft mode scores candidates with a tiny DistMult trained at boot
 * (seeded, instant) and admits the strongest unstated candidate per hop.
 * Naproxen's recovery is computed, not scripted.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const ENT = ["Aspirin", "Ibuprofen", "Naproxen", "Metformin", "COX-1", "COX-2", "AMPK", "Inflammation", "Diabetes"];
  const TARGETS = 0, ASSOC = 1;
  const T = [[0, TARGETS, 4], [0, TARGETS, 5], [1, TARGETS, 5], [2, TARGETS, 4], [3, TARGETS, 6],
             [5, ASSOC, 7], [6, ASSOC, 8]];
  const POS = [[120, 62], [120, 132], [120, 202], [120, 272], [350, 82], [350, 172], [350, 262],
               [585, 127], [585, 262]];
  const NE = ENT.length;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const sigma = (x) => 1 / (1 + Math.exp(-x));

  // boot-train DistMult (similarity scorer — kind to 1-to-N, per Week 4)
  const rng = mulberry32(23);
  const D = 6;
  const E = Array.from({ length: NE }, () => Array.from({ length: D }, () => rng() * 0.6 - 0.3));
  const R = Array.from({ length: 2 }, () => Array.from({ length: D }, () => rng() * 0.6 - 0.3));
  const score = (h, r, t) => {
    let s = 0;
    for (let i = 0; i < D; i++) s += E[h][i] * R[r][i] * E[t][i];
    return s;
  };
  for (let step = 0; step < 9000; step++) {
    const [h, r, t] = T[Math.floor(rng() * T.length)];
    let ch = h, ct = t;
    if (rng() < 0.5) { do { ch = Math.floor(rng() * NE); } while (ch === h); }
    else { do { ct = Math.floor(rng() * NE); } while (ct === t); }
    const gp = 1 - sigma(score(h, r, t));
    const gn = sigma(score(ch, r, ct));
    for (let i = 0; i < D; i++) {
      const lr = 0.05;
      const dh = gp * R[r][i] * E[t][i], dt = gp * R[r][i] * E[h][i];
      const dr = gp * E[h][i] * E[t][i];
      const nh = gn * R[r][i] * E[ct][i], nt = gn * R[r][i] * E[ch][i];
      const nr = gn * E[ch][i] * E[ct][i];
      E[h][i] += lr * dh; E[t][i] += lr * dt; R[r][i] += lr * (dr - nr);
      E[ch][i] -= lr * nh; E[ct][i] -= lr * nt;
    }
  }

  const statedHeads = (r, t) => T.filter(([, rr, tt]) => rr === r && tt === t).map(([h]) => h);

  function softHeads(r, t, pool) {
    // stated heads + the single strongest unstated candidate from the pool
    const stated = statedHeads(r, t).filter((h) => pool.includes(h));
    const cands = pool.filter((h) => !stated.includes(h))
      .map((h) => ({ h, s: score(h, r, t) }))
      .sort((a, b) => b.s - a.s);
    return { stated, extra: cands.length ? cands[0].h : null };
  }

  let mode = "traversal", hop = 0;
  const DRUG_POOL = [0, 1, 2, 3], PROT_POOL = [4, 5, 6];

  function state() {
    // hop 0: anchor. hop 1: proteins (always exact — keeps the story crisp).
    // hop 2: drugs; soft mode admits the strongest unstated candidate here.
    const s = { prots: [], drugs: [], extraProt: null, extraDrug: null };
    if (hop >= 1) {
      s.prots = softHeads(ASSOC, 7, PROT_POOL).stated;
    }
    if (hop >= 2) {
      const drugSet = new Set();
      let best = null, bestScore = -Infinity;
      s.prots.forEach((p) => {
        const r2 = softHeads(TARGETS, p, DRUG_POOL);
        r2.stated.forEach((d) => drugSet.add(d));
        if (mode === "soft" && r2.extra !== null) {
          const sc = score(r2.extra, TARGETS, p);
          if (sc > bestScore) { bestScore = sc; best = r2.extra; }
        }
      });
      s.drugs = [...drugSet];
      if (mode === "soft" && best !== null && !drugSet.has(best)) s.extraDrug = best;
    }
    return s;
  }

  function render() {
    const P = U.pal();
    const st = state();
    const svg = U.svgIn("w5-qy-svg", 760, 350);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    T.forEach(([h, , t]) => {
      g.append("line").attr("x1", POS[h][0] + 44).attr("y1", POS[h][1])
        .attr("x2", POS[t][0] - 48).attr("y2", POS[t][1])
        .attr("stroke", P.muted).attr("stroke-width", 1.6).attr("opacity", 0.5);
    });
    if (mode === "soft" && st.extraDrug === 2) {
      g.append("line").attr("x1", POS[2][0] + 44).attr("y1", POS[2][1])
        .attr("x2", POS[5][0] - 48).attr("y2", POS[5][1])
        .attr("stroke", P.accent).attr("stroke-width", 2.2).attr("stroke-dasharray", "6,5");
    }

    ENT.forEach((name, i) => {
      const [x, y] = POS[i];
      const isAnchor = i === 7;
      const inProts = st.prots.includes(i) || st.extraProt === i;
      const inDrugs = st.drugs.includes(i) || st.extraDrug === i;
      const isExtra = st.extraProt === i || st.extraDrug === i;
      const w = 8.2 * name.length + 22;
      let fill = P.paper, stroke = P.muted, tfill = P.muted;
      if (isAnchor) { fill = P.yellow; stroke = "none"; tfill = "#5c4508"; }
      else if (inDrugs) { fill = isExtra ? P.paper : P.green; stroke = isExtra ? P.accent : "none"; tfill = isExtra ? P.accentDark : "#fff"; }
      else if (inProts) { fill = isExtra ? P.paper : P.purple; stroke = isExtra ? P.accent : "none"; tfill = isExtra ? P.accentDark : "#fff"; }
      g.append("rect").attr("x", x - w / 2).attr("y", y - 14).attr("width", w).attr("height", 28)
        .attr("rx", 14).attr("fill", fill).attr("stroke", stroke).attr("stroke-width", 1.6)
        .attr("stroke-dasharray", isExtra ? "5,4" : null);
      g.append("text").attr("x", x).attr("y", y + 4.5).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", tfill).text(name);
    });

    const names = (ids) => ids.map((i) => ENT[i]).join(", ") || "—";
    const answers = st.drugs.concat(st.extraDrug !== null ? [st.extraDrug] : []);
    let status;
    if (hop === 0) status = "anchor: Inflammation — press “hop 1” to walk associated_with backwards";
    else if (hop === 1) status = `hop 1 → proteins: ${names(st.prots.concat(st.extraProt !== null ? [st.extraProt] : []))}`;
    else status = `hop 2 → answer set: {${names(answers)}}` +
      (mode === "soft" ? (st.extraDrug === 2 ? " — Naproxen recovered by the model, not by any path" : "")
                       : (answers.length === 2 ? " — two of the three true drugs. The third has no path." : ""));
    g.append("text").attr("x", 380).attr("y", 320).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600)
      .attr("fill", hop === 2 && mode === "soft" && st.extraDrug === 2 ? P.green : P.text)
      .text(status);
    g.append("text").attr("x", 380).attr("y", 340).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(mode === "soft"
        ? "soft mode: stated edges + the strongest unstated candidate per hop, scored by a DistMult trained on this graph"
        : "traversal mode: stated edges only — the mode every database query runs in");
  }

  document.querySelectorAll("#w5-qy-widget [data-mode]").forEach((b) =>
    b.addEventListener("click", () => {
      mode = b.getAttribute("data-mode");
      document.querySelectorAll("#w5-qy-widget [data-mode]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.getElementById("w5-qy-hop").addEventListener("click", () => {
    hop = Math.min(2, hop + 1);
    render();
  });
  document.getElementById("w5-qy-reset").addEventListener("click", () => { hop = 0; render(); });

  U.onThemeChange(render);
  U.lazyBoot("w5-qy-svg", render);
})();
