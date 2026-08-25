/* Widget 4.3 — Who gets the gradient? Corruption and self-adversarial weights.
 * A tiny TransE trains itself on the fragment at boot (seeded, instant). Pick a
 * true triple: every tail-corruption is scored, and the temperature slider α
 * turns uniform negative sampling into RotatE's self-adversarial weighting —
 * watch the gradient budget concentrate on the plausible lies.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const ENT = ["IU", "KFU", "Innopolis", "Kazan", "Tatarstan", "Russia", "Volga", "university", "city", "region"];
  const T = [
    [0, 1, 7], [0, 0, 2], [2, 0, 4], [3, 2, 4], [3, 0, 4], [4, 3, 5],
    [1, 1, 7], [1, 0, 3], [2, 1, 8], [3, 1, 8], [4, 1, 9], [6, 4, 4],
  ];
  const RNAME = ["located_in", "instance_of", "capital_of", "part_of", "flows_through"];
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

  // boot-train the tiny TransE (same recipe as Widget 1, its own seeded copy)
  const rng = mulberry32(13);
  const E = Array.from({ length: NE }, () => [rng() * 4 - 2, rng() * 4 - 2]);
  const R = Array.from({ length: NR }, () => [rng() * 1 - 0.5, rng() * 1 - 0.5]);
  const resid = (h, r, t) => Math.hypot(E[h][0] + R[r][0] - E[t][0], E[h][1] + R[r][1] - E[t][1]);
  for (let s = 0; s < 6000; s++) {
    const [h, r, t] = T[Math.floor(rng() * T.length)];
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

  const TRUE_SET = new Set(T.map(([h, r, t]) => `${h},${r},${t}`));
  // pills: three true facts; the third corrupts the HEAD, where an accidentally
  // true corruption exists (Innopolis is also an instance of city)
  const QUERIES = [
    { triple: [0, 0, 2], side: "tail" },
    { triple: [6, 4, 4], side: "tail" },
    { triple: [3, 1, 8], side: "head" },
  ];
  let qi = 0;

  function alpha() { return parseFloat(document.getElementById("w4-ng-alpha").value); }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w4-ng-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    const { triple: [h, r, t], side } = QUERIES[qi];
    const a = alpha();
    const cands = [];
    for (let e = 0; e < NE; e++) {
      if (e === (side === "tail" ? t : h)) continue;
      const score = side === "tail" ? -resid(h, r, e) : -resid(e, r, t);
      const key = side === "tail" ? `${h},${r},${e}` : `${e},${r},${t}`;
      cands.push({ e, score, isTrue: TRUE_SET.has(key) });
    }
    cands.sort((x, y) => y.score - x.score);
    const mx = Math.max(...cands.map((c) => c.score));
    const expw = cands.map((c) => Math.exp(a * (c.score - mx)));
    const Zs = expw.reduce((x, y) => x + y, 0);
    cands.forEach((c, i) => { c.w = expw[i] / Zs; });

    g.append("text").attr("x", 380).attr("y", 24).attr("text-anchor", "middle")
      .attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.text)
      .text(`true fact: (${ENT[h]}, ${RNAME[r]}, ${ENT[t]}) — nine candidate corruptions of the ${side}`);

    const smax = Math.max(...cands.map((c) => c.score));
    const smin = Math.min(...cands.map((c) => c.score));
    cands.forEach((c, i) => {
      const y = 46 + i * 27;
      const bw = 8 + 210 * ((c.score - smin) / (smax - smin || 1));
      g.append("rect").attr("x", 180).attr("y", y).attr("width", bw).attr("height", 11)
        .attr("rx", 4).attr("fill", P.blue).attr("opacity", 0.5);
      const ww = 300 * c.w;
      g.append("rect").attr("x", 180).attr("y", y + 12).attr("width", Math.max(1.5, ww))
        .attr("height", 7).attr("rx", 3).attr("fill", P.accent).attr("opacity", 0.9);
      g.append("text").attr("x", 172).attr("y", y + 14).attr("text-anchor", "end")
        .attr("font-size", 12.5).attr("font-weight", c.w > 0.2 ? 700 : 400)
        .attr("fill", c.isTrue ? "#cf4a30" : P.text).text(ENT[c.e] + (c.isTrue ? " ⚠" : ""));
      g.append("text").attr("x", 190 + Math.max(bw, ww)).attr("y", y + 16)
        .attr("font-size", 12.5).attr("fill", P.muted)
        .text(`${(100 * c.w).toFixed(0)}%`);
    });

    g.append("text").attr("x", 560).attr("y", 52).attr("font-size", 12.5).attr("fill", P.blue).text("teal: model score");
    g.append("text").attr("x", 560).attr("y", 70).attr("font-size", 12.5).attr("fill", P.accentDark).text("terracotta: sampling weight");
    const top = cands[0];
    const verdict = a === 0
      ? "α = 0: uniform — most of the budget lands on easy absurdities"
      : `α = ${a.toFixed(1)}: the hardest negative (${ENT[top.e]}) now takes ${(100 * top.w).toFixed(0)}% of the gradient`;
    g.append("text").attr("x", 380).attr("y", 316).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", a > 0 ? P.accentDark : P.muted)
      .text(verdict);
    document.getElementById("w4-ng-aval").textContent = a.toFixed(1);
  }

  document.querySelectorAll("#w4-ng-widget [data-q]").forEach((b) =>
    b.addEventListener("click", () => {
      qi = +b.getAttribute("data-q");
      document.querySelectorAll("#w4-ng-widget [data-q]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.getElementById("w4-ng-alpha").addEventListener("input", render);

  U.onThemeChange(render);
  U.lazyBoot("w4-ng-svg", render);
})();
