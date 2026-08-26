/* Widget 2.4 — Greedy modularity, move by move (Louvain, phase 1).
 * Ten nodes, two planted communities, two bridge edges. Every node starts
 * alone. Each press of "next move" considers one node in sweep order: it
 * computes ΔQ for joining each neighboring community and takes the best
 * positive move, exactly as Louvain's local-move phase does. Q is recomputed
 * from the definition at every state — no caching tricks, so what you see is
 * what the formula says.
 */
(function () {
  "use strict";
  const U = window.DKR;
  // last entry is blue, NOT gray — node 9's singleton community must not share
  // the neutral gray used for inter-community edges
  const PAL = ["#d9603b", "#0f8377", "#7c5cd6", "#d9a62e", "#d1567e",
               "#199473", "#8a5a33", "#cf4a30", "#5f7a1e", "#3f7fc4"];
  const N = 10;
  const EDGES = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 4], [2, 4], [0, 3],
                 [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [8, 9], [7, 9], [5, 8],
                 [4, 5], [4, 6]];
  const POS = [[80, 80], [190, 55], [140, 170], [250, 140], [330, 110],
               [440, 85], [450, 195], [555, 65], [580, 175], [675, 115]];
  const M = EDGES.length;
  const DEG = Array(N).fill(0);
  EDGES.forEach(([a, b]) => { DEG[a] += 1; DEG[b] += 1; });
  const NBR = Array.from({ length: N }, () => []);
  EDGES.forEach(([a, b]) => { NBR[a].push(b); NBR[b].push(a); });

  let comm, ptr, noGain, converged, msg;
  function reset() {
    comm = d3.range(N);
    ptr = 0; noGain = 0; converged = false;
    msg = "every node is its own community — press “next move”";
  }
  reset();

  function modularity(c) {
    let q = 0;
    EDGES.forEach(([a, b]) => { if (c[a] === c[b]) q += 1 / M; });   // 2·(A_ij/2m) over pairs
    const degSum = new Map();
    c.forEach((ci, i) => degSum.set(ci, (degSum.get(ci) || 0) + DEG[i]));
    degSum.forEach((d) => { q -= (d / (2 * M)) * (d / (2 * M)); });
    return q;
  }

  function nextMove() {
    if (converged) return;
    const i = ptr % N;
    ptr += 1;
    const qBase = modularity(comm);
    const own = comm[i];
    let best = own, bestQ = qBase;
    [...new Set(NBR[i].map((j) => comm[j]))].forEach((c) => {
      if (c === own) return;
      const trial = comm.slice();
      trial[i] = c;
      const q = modularity(trial);
      if (q > bestQ + 1e-12) { bestQ = q; best = c; }
    });
    if (best !== own) {
      comm[i] = best;
      noGain = 0;
      msg = `node ${i} joins node ${best}'s community · ΔQ = +${(bestQ - qBase).toFixed(3)}`;
    } else {
      noGain += 1;
      msg = `node ${i}: no move improves Q — stays put`;
      if (noGain >= N) {
        converged = true;
        msg = `phase-1 optimum: no single node wants to move · Q = ${qBase.toFixed(3)} — try phase 2`;
      }
    }
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w2-lv-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g").attr("transform", "translate(15,10)");

    g.selectAll("line").data(EDGES).join("line")
      .attr("x1", (e) => POS[e[0]][0]).attr("y1", (e) => POS[e[0]][1])
      .attr("x2", (e) => POS[e[1]][0]).attr("y2", (e) => POS[e[1]][1])
      .attr("stroke", (e) => (comm[e[0]] === comm[e[1]] ? PAL[comm[e[0]] % 10] : P.muted))
      .attr("stroke-width", (e) => (comm[e[0]] === comm[e[1]] ? 3 : 1.8))
      .attr("opacity", (e) => (comm[e[0]] === comm[e[1]] ? 0.8 : 0.65))
      .attr("stroke-dasharray", (e) => (comm[e[0]] === comm[e[1]] ? null : "4,4"));

    const upNext = converged ? -1 : ptr % N;
    const nd = g.selectAll("g.n").data(d3.range(N)).join("g")
      .attr("transform", (i) => `translate(${POS[i][0]},${POS[i][1]})`);
    nd.append("circle").attr("r", 16)
      .attr("fill", (i) => PAL[comm[i] % 10])
      .attr("stroke", (i) => (i === upNext ? P.text : "none"))
      .attr("stroke-width", 2.5)
      .attr("stroke-dasharray", (i) => (i === upNext ? "3,3" : null));
    nd.append("text").attr("text-anchor", "middle").attr("dy", 4.5)
      .attr("font-size", 13).attr("font-weight", 700).attr("fill", "#fff")
      .text((i) => i);

    // Q meter — scaled: zero tick, end labels, and the planted-split target, so
    // the track reads as an axis rather than a broken slider
    const q = modularity(comm);
    const meter = g.append("g").attr("transform", "translate(90,258)")
      .attr("font-family", "'Source Sans 3', sans-serif");
    const QLO = -0.12, QHI = 0.5;                                // display range
    const qx = (v) => 540 * (v - QLO) / (QHI - QLO);
    meter.append("rect").attr("x", 0).attr("y", 0).attr("width", 540).attr("height", 12)
      .attr("rx", 6).attr("fill", P.border);
    const frac = Math.max(0, Math.min(1, (q - QLO) / (QHI - QLO)));
    meter.append("rect").attr("x", 0).attr("y", 0).attr("width", 540 * frac).attr("height", 12)
      .attr("rx", 6).attr("fill", q > 0.3 ? P.green : P.accent);
    meter.append("line").attr("x1", qx(0)).attr("x2", qx(0)).attr("y1", -4).attr("y2", 16)
      .attr("stroke", P.text).attr("stroke-width", 1.2);
    meter.append("text").attr("x", qx(0)).attr("y", -8).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text("Q = 0");
    const qStar = 0.389;                                         // the planted 2-community split
    meter.append("line").attr("x1", qx(qStar)).attr("x2", qx(qStar)).attr("y1", -4).attr("y2", 16)
      .attr("stroke", P.green).attr("stroke-width", 1.4).attr("stroke-dasharray", "3,2");
    meter.append("text").attr("x", qx(qStar)).attr("y", -8).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.green).text("planted split");
    meter.append("text").attr("x", 556).attr("y", 11).attr("font-size", 13.5)
      .attr("font-weight", 700).attr("fill", P.text).text(`Q = ${q.toFixed(3)}`);
    meter.append("text").attr("x", 280).attr("y", 30).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`${new Set(comm).size} communities · dashed ring = considered next · solid colored edges are inside a community`);

    g.append("text").attr("x", 365).attr("y", 315).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600)
      .attr("fill", converged ? P.green : P.text).text(msg);
  }

  document.getElementById("w2-lv-step").addEventListener("click", () => { nextMove(); render(); });
  document.getElementById("w2-lv-run").addEventListener("click", () => {
    for (let k = 0; k < 500 && !converged; k++) nextMove();
    render();
  });
  document.getElementById("w2-lv-phase2").addEventListener("click", () => {
    // Louvain's aggregation phase, in effect: greedily merge whole communities
    // while Q improves, then run the local-move sweep again.
    let improved = true;
    while (improved) {
      improved = false;
      const ids = [...new Set(comm)];
      let bQ = modularity(comm), bPair = null;
      for (let a = 0; a < ids.length; a++) {
        for (let b = a + 1; b < ids.length; b++) {
          const trial = comm.map((c) => (c === ids[b] ? ids[a] : c));
          const q = modularity(trial);
          if (q > bQ + 1e-12) { bQ = q; bPair = [ids[a], ids[b]]; }
        }
      }
      if (bPair) { comm = comm.map((c) => (c === bPair[1] ? bPair[0] : c)); improved = true; }
    }
    converged = false; noGain = 0;
    for (let k = 0; k < 500 && !converged; k++) nextMove();
    msg = `after phase 2 (merge whole communities, then resweep): ${new Set(comm).size} communities · Q = ${modularity(comm).toFixed(3)}`;
    render();
  });
  document.getElementById("w2-lv-reset").addEventListener("click", () => { reset(); render(); });

  U.onThemeChange(render);
  U.lazyBoot("w2-lv-svg", render);
})();
