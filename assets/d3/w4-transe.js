/* Widget 4.1 — TransE learns our twelve facts, live.
 * Real margin-loss SGD in 2D on the Lab-1 Wikidata fragment. Watch the
 * geometry succeed (chains line up) and fail honestly: same-tail heads
 * collapse (the 1-to-N crush), located_in and capital_of merge, and the
 * transitive fact the hero figure asks about stays mediocre — as the pattern
 * cheat sheet predicts. Seeded: reset replays the identical run.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const ENT = ["IU", "KFU", "Innopolis", "Kazan", "Tatarstan", "Russia", "Volga", "university", "city", "region"];
  const REL = ["located_in", "instance_of", "capital_of", "part_of", "flows_through"];
  const RCOL = ["#0f8377", "#8e8e9a", "#7c5cd6", "#199473", "#d1567e"];
  const T = [ // [head, rel, tail] as indices
    [0, 1, 7], [0, 0, 2], [2, 0, 4], [3, 2, 4], [3, 0, 4], [4, 3, 5],
    [1, 1, 7], [1, 0, 3], [2, 1, 8], [3, 1, 8], [4, 1, 9], [6, 4, 4],
  ];
  const NE = ENT.length, NR = REL.length, GAMMA = 1.0, LR = 0.04;

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  let rng, E, R, steps, showRel = 0, auto = false;
  function reset() {
    auto = false;
    rng = mulberry32(18);
    E = Array.from({ length: NE }, () => [rng() * 4 - 2, rng() * 4 - 2]);
    R = Array.from({ length: NR }, () => [rng() * 1 - 0.5, rng() * 1 - 0.5]);
    steps = 0;
  }
  reset();

  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  const resid = (h, r, t) => Math.hypot(E[h][0] + R[r][0] - E[t][0], E[h][1] + R[r][1] - E[t][1]);

  function sgdStep() {
    const [h, r, t] = T[Math.floor(rng() * T.length)];
    let ch = h, ct = t;                                   // corrupt head or tail
    if (rng() < 0.5) { do { ch = Math.floor(rng() * NE); } while (ch === h); }
    else { do { ct = Math.floor(rng() * NE); } while (ct === t); }
    const dpos = resid(h, r, t), dneg = resid(ch, r, ct);
    if (GAMMA + dpos - dneg <= 0) return;
    // gradient of ||h+r-t||: unit vector u; positive pulls together, negative pushes
    const up = [(E[h][0] + R[r][0] - E[t][0]) / (dpos + 1e-9), (E[h][1] + R[r][1] - E[t][1]) / (dpos + 1e-9)];
    const un = [(E[ch][0] + R[r][0] - E[ct][0]) / (dneg + 1e-9), (E[ch][1] + R[r][1] - E[ct][1]) / (dneg + 1e-9)];
    for (const k of [0, 1]) {
      E[h][k] -= LR * up[k]; E[t][k] += LR * up[k];
      E[ch][k] += LR * un[k]; E[ct][k] -= LR * un[k];
      R[r][k] -= LR * (up[k] - un[k]);
    }
  }

  function train(k) { for (let i = 0; i < k; i++) sgdStep(); steps += k; }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w4-te-svg", 760, 380);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    const xs = E.map((e) => e[0]), ys = E.map((e) => e[1]);
    const xmin = Math.min(...xs), xmax = Math.max(...xs);
    const ymin = Math.min(...ys), ymax = Math.max(...ys);
    const sx = (x) => 55 + ((x - xmin) / (xmax - xmin || 1)) * 560;
    const sy = (y) => 30 + ((y - ymin) / (ymax - ymin || 1)) * 225;

    // arrows of the selected relation: h → h+r (solid), h+r ⇢ t (dashed gap)
    T.filter(([, r]) => r === showRel).forEach(([h, r, t]) => {
      const hx = sx(E[h][0]), hy = sy(E[h][1]);
      const gx = sx(E[h][0] + R[r][0]), gy = sy(E[h][1] + R[r][1]);
      const tx = sx(E[t][0]), ty = sy(E[t][1]);
      g.append("line").attr("x1", hx).attr("y1", hy).attr("x2", gx).attr("y2", gy)
        .attr("stroke", RCOL[showRel]).attr("stroke-width", 2.2).attr("opacity", 0.85);
      g.append("line").attr("x1", gx).attr("y1", gy).attr("x2", tx).attr("y2", ty)
        .attr("stroke", "#cf4a30").attr("stroke-width", 1.6)
        .attr("stroke-dasharray", "3,3").attr("opacity", 0.8);
      g.append("circle").attr("cx", gx).attr("cy", gy).attr("r", 3).attr("fill", RCOL[showRel]);
    });

    ENT.forEach((name, i) => {
      const x = sx(E[i][0]), y = sy(E[i][1]);
      const isType = i >= 7;
      g.append("circle").attr("cx", x).attr("cy", y).attr("r", isType ? 6 : 8)
        .attr("fill", isType ? P.muted : P.yellow).attr("opacity", 0.9);
      g.append("text").attr("x", x).attr("y", y - 11).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("font-weight", 600)
        .attr("paint-order", "stroke").attr("stroke", P.paper).attr("stroke-width", 3)
        .attr("stroke-linejoin", "round")
        .attr("fill", isType ? P.muted : P.text).text(name);
    });

    const avg = T.reduce((a, [h, r, t]) => a + resid(h, r, t), 0) / T.length;
    const crushCities = dist(E[2], E[3]);                 // Innopolis vs Kazan
    const crushRel = dist(R[0], R[2]);                    // located_in vs capital_of
    const inferred = resid(0, 0, 4);                      // (IU, located_in, Tatarstan)
    const nonsense = resid(6, 0, 7);                      // (Volga, located_in, university)

    const rows = [
      [`${steps.toLocaleString()} SGD steps · mean residual ‖h+r−t‖ = ${avg.toFixed(2)}`, avg < 0.45 ? P.green : P.text],
      [`the crush: |Innopolis − Kazan| = ${crushCities.toFixed(2)} · |located_in − capital_of| = ${crushRel.toFixed(2)}`, P.muted],
      [`hero fact (IU, located_in, Tatarstan): ${inferred.toFixed(2)} · nonsense (Volga, located_in, university): ${nonsense.toFixed(2)}`, P.muted],
    ];
    rows.forEach(([txt, color], i) => {
      g.append("text").attr("x", 380).attr("y", 300 + i * 19).attr("text-anchor", "middle")
        .attr("font-size", i === 0 ? 13.5 : 12.5).attr("font-weight", i === 0 ? 700 : 400)
        .attr("fill", color).text(txt);
    });
    g.append("text").attr("x", 380).attr("y", 300 + 3 * 19).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("solid = h + r · dashed red = the miss · smaller numbers = truer, per TransE");
  }

  function autoLoop() {
    if (!auto) return;
    train(60); render();
    requestAnimationFrame(autoLoop);
  }

  document.querySelectorAll("#w4-te-widget [data-rel]").forEach((b) =>
    b.addEventListener("click", () => {
      showRel = +b.getAttribute("data-rel");
      document.querySelectorAll("#w4-te-widget [data-rel]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.getElementById("w4-te-train").addEventListener("click", () => { train(300); render(); });
  document.getElementById("w4-te-auto").addEventListener("click", (ev) => {
    if (!U.motionOK()) { train(1500); render(); return; }
    auto = !auto;
    ev.currentTarget.classList.toggle("active", auto);
    if (auto) requestAnimationFrame(autoLoop);
  });
  document.getElementById("w4-te-reset").addEventListener("click", () => {
    document.getElementById("w4-te-auto").classList.remove("active");
    reset(); render();
  });

  U.onThemeChange(render);
  U.lazyBoot("w4-te-svg", render);
})();
