/* Widget 2.4 — Greedy modularity, move by move (Louvain, phase 1).
 * Ten nodes, two planted communities, two bridge edges. Every node starts
 * alone. Each press of "next move" considers one node in sweep order: it
 * computes ΔQ for joining each neighboring community and takes the best
 * positive move, exactly as Louvain's local-move phase does.
 *
 * What the reader sees, and why: the two planted groups are laid out as two
 * clusters with node 4 as the hinge; communities of two or more members wear
 * a soft hull in their colour while nodes still alone stay hollow; a decision
 * panel shows, BEFORE each move, every option the next node weighs and its
 * ΔQ, so the greedy step is watched rather than trusted; and a chart keeps
 * the history of Q against the planted split. Q is recomputed from the
 * definition at every state — no caching tricks.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 432;
  const N = 10;
  const EDGES = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [3, 4], [2, 4], [0, 3],
                 [5, 6], [5, 7], [6, 7], [6, 8], [7, 8], [8, 9], [7, 9], [5, 8],
                 [4, 5], [4, 6]];
  const POS = [[92, 128], [172, 62], [150, 196], [248, 132], [368, 126],
               [486, 66], [492, 190], [590, 62], [604, 194], [676, 128]];
  const PLANTED = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
  const M = EDGES.length;
  const DEG = Array(N).fill(0);
  EDGES.forEach(([a, b]) => { DEG[a] += 1; DEG[b] += 1; });
  const NBR = Array.from({ length: N }, () => []);
  EDGES.forEach(([a, b]) => { NBR[a].push(b); NBR[b].push(a); });

  function modularity(c) {
    let q = 0;
    EDGES.forEach(([a, b]) => { if (c[a] === c[b]) q += 1 / M; });   // 2·(A_ij/2m) over pairs
    const degSum = new Map();
    c.forEach((ci, i) => degSum.set(ci, (degSum.get(ci) || 0) + DEG[i]));
    degSum.forEach((d) => { q -= (d / (2 * M)) * (d / (2 * M)); });
    return q;
  }
  const Q_PLANTED = modularity(PLANTED);

  let comm, ptr, noGain, converged, msg, hist, colorOf, nextColor, lastMoved;
  function reset() {
    comm = d3.range(N);
    ptr = 0; noGain = 0; converged = false; lastMoved = -1;
    msg = "every node is its own community — press “next move”";
    hist = [modularity(comm)];
    colorOf = new Map(); nextColor = 0;
  }
  reset();

  // a community earns a colour the moment it has two members, and keeps it
  function paletteFor(P) { return [P.accent, P.blue, P.purple, P.yellow, P.red, P.green]; }
  function assignColors() {
    const size = new Map();
    comm.forEach((c) => size.set(c, (size.get(c) || 0) + 1));
    [...size.keys()].sort((a, b) => a - b).forEach((c) => {
      if (size.get(c) >= 2 && !colorOf.has(c)) { colorOf.set(c, nextColor % 6); nextColor += 1; }
    });
  }

  // the options the node at the sweep pointer would weigh right now
  function options(i) {
    const qBase = modularity(comm);
    const own = comm[i];
    return [...new Set(NBR[i].map((j) => comm[j]))].filter((c) => c !== own).map((c) => {
      const trial = comm.slice(); trial[i] = c;
      return { c, dq: modularity(trial) - qBase, size: comm.filter((x) => x === c).length };
    }).sort((a, b) => b.dq - a.dq);
  }

  function nextMove() {
    if (converged) return;
    const i = ptr % N;
    ptr += 1;
    const opts = options(i);
    const best = opts.length && opts[0].dq > 1e-12 ? opts[0] : null;
    if (best) {
      comm[i] = best.c;
      noGain = 0; lastMoved = i;
      hist.push(modularity(comm));
      msg = `node ${i} joins node ${best.c}'s community · ΔQ = +${best.dq.toFixed(3)}`;
    } else {
      noGain += 1; lastMoved = -1;
      msg = `node ${i}: no move improves Q — stays put`;
      if (noGain >= N) {
        converged = true;
        msg = `phase-1 optimum: no single node wants to move · Q = ${modularity(comm).toFixed(3)} — try phase 2`;
      }
    }
    assignColors();
  }

  function hullPath(pts, pad) {
    if (pts.length === 1) return null;
    const cx = d3.mean(pts, (p) => p[0]), cy = d3.mean(pts, (p) => p[1]);
    const pushed = pts.map(([x, y]) => {
      const dx = x - cx, dy = y - cy, L = Math.hypot(dx, dy) || 1;
      return [x + (dx / L) * pad, y + (dy / L) * pad];
    });
    const hull = pts.length >= 3 ? d3.polygonHull(pushed) : pushed;
    return "M" + (hull || pushed).map((p) => p.join(",")).join("L") + "Z";
  }

  function render() {
    const P = U.pal();
    const PAL = paletteFor(P);
    const colour = (c) => (colorOf.has(c) ? PAL[colorOf.get(c)] : null);
    const svg = U.svgIn("w2-lv-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const q = modularity(comm);
    const upNext = converged ? -1 : ptr % N;

    // ── community hulls ──
    const members = new Map();
    comm.forEach((c, i) => { if (!members.has(c)) members.set(c, []); members.get(c).push(i); });
    members.forEach((ids, c) => {
      if (ids.length < 2) return;
      const d = hullPath(ids.map((i) => POS[i]), 6);
      g.append("path").attr("d", d).attr("fill", colour(c)).attr("fill-opacity", 0.10)
        .attr("stroke", colour(c)).attr("stroke-width", 44).attr("stroke-opacity", 0.14)
        .attr("stroke-linejoin", "round").attr("stroke-linecap", "round");
    });

    // ── edges ──
    g.selectAll("line.e").data(EDGES).join("line").attr("class", "e")
      .attr("x1", (e) => POS[e[0]][0]).attr("y1", (e) => POS[e[0]][1])
      .attr("x2", (e) => POS[e[1]][0]).attr("y2", (e) => POS[e[1]][1])
      .attr("stroke", (e) => (comm[e[0]] === comm[e[1]] && colour(comm[e[0]]) ? colour(comm[e[0]]) : P.muted))
      .attr("stroke-width", (e) => (comm[e[0]] === comm[e[1]] ? 3 : 1.6))
      .attr("opacity", (e) => (comm[e[0]] === comm[e[1]] ? 0.85 : 0.5));

    // ── nodes ──
    const nd = g.selectAll("g.n").data(d3.range(N)).join("g").attr("class", "n")
      .attr("transform", (i) => `translate(${POS[i][0]},${POS[i][1]})`);
    nd.filter((i) => i === upNext).append("circle").attr("r", 23).attr("fill", "none")
      .attr("stroke", P.text).attr("stroke-width", 2).attr("stroke-dasharray", "4,3");
    nd.append("circle").attr("r", 16)
      .attr("fill", (i) => colour(comm[i]) || P.paper)
      .attr("stroke", (i) => (colour(comm[i]) ? "none" : P.muted))
      .attr("stroke-width", 2);
    nd.append("text").attr("text-anchor", "middle").attr("dy", 4.5)
      .attr("font-size", 13).attr("font-weight", 700)
      .attr("fill", (i) => (colour(comm[i]) ? "#fff" : P.text)).text((i) => i);
    g.append("text").attr("x", W / 2).attr("y", 22).attr("text-anchor", "middle").attr("font-size", 12.5)
      .attr("fill", P.muted)
      .text("hollow = still alone · shaded hull = a community · dashed ring = whose turn it is");

    // ── decision panel: the options the next node is weighing ──
    const px = 20, py = 262;
    const panel = g.append("g").attr("transform", `translate(${px},${py})`);
    if (upNext >= 0) {
      const opts = options(upNext);
      panel.append("text").attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.text)
        .text(`node ${upNext} is weighing its options`);
      const scale = 620;                                       // px per unit ΔQ
      const x0 = 248;                                          // bar origin, clear of the labels
      if (!opts.length) {
        panel.append("text").attr("y", 24).attr("font-size", 12.5).attr("fill", P.muted)
          .text("every neighbor is already in its community");
      }
      opts.slice(0, 4).forEach((o, k) => {
        const y = 26 + k * 21;
        const best = k === 0 && o.dq > 1e-12;
        const sw = colour(o.c);
        panel.append("circle").attr("cx", 7).attr("cy", y - 4).attr("r", 6)
          .attr("fill", sw || P.paper).attr("stroke", sw ? "none" : P.muted).attr("stroke-width", 1.5);
        panel.append("text").attr("x", 19).attr("y", y).attr("font-size", 12.5)
          .attr("fill", P.text).attr("font-weight", best ? 700 : 400)
          .text(`join node ${o.c}'s community (${o.size})`);
        const w = Math.abs(o.dq) * scale;
        panel.append("rect").attr("x", o.dq >= 0 ? x0 : x0 - w).attr("y", y - 11)
          .attr("width", Math.max(w, 1)).attr("height", 12).attr("rx", 2)
          .attr("fill", o.dq >= 0 ? (best ? P.green : P.blue) : P.red).attr("opacity", best ? 0.95 : 0.55);
        panel.append("text").attr("x", (o.dq >= 0 ? x0 + w : x0) + 6).attr("y", y)
          .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5)
          .attr("font-weight", best ? 700 : 400).attr("fill", o.dq >= 0 ? P.text : P.red)
          .text(`${o.dq >= 0 ? "+" : "−"}${Math.abs(o.dq).toFixed(3)}${best ? "  ← best" : ""}`);
      });
      panel.append("line").attr("x1", x0).attr("x2", x0).attr("y1", 12).attr("y2", 26 + Math.max(opts.length, 1) * 21 - 6)
        .attr("stroke", P.border).attr("stroke-width", 1);
      panel.append("text").attr("x", x0).attr("y", 26 + Math.max(opts.length, 1) * 21 + 6)
        .attr("text-anchor", "middle").attr("font-size", 12.5).attr("fill", P.muted)
        .text("ΔQ per option · the best is taken if positive");
    } else {
      panel.append("text").attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.green)
        .text("phase 1 has converged");
      panel.append("text").attr("y", 24).attr("font-size", 12.5).attr("fill", P.text)
        .text("no single node can raise Q by moving —");
      panel.append("text").attr("y", 42).attr("font-size", 12.5).attr("fill", P.text)
        .text("only moving whole communities (phase 2) can");
    }

    // ── Q history against the planted split ──
    const cx0 = 430, cy0 = 258, cw = 310, ch = 92;
    const chart = g.append("g").attr("transform", `translate(${cx0},${cy0})`);
    const QLO = -0.12, QHI = 0.46;
    const ys = (v) => ch - (v - QLO) / (QHI - QLO) * ch;
    const nx = Math.max(12, hist.length);
    const xs = (k) => (k / (nx - 1)) * cw;
    chart.append("line").attr("x1", 0).attr("x2", cw).attr("y1", ys(0)).attr("y2", ys(0))
      .attr("stroke", P.muted).attr("stroke-width", 1).attr("opacity", 0.7);
    chart.append("text").attr("x", -6).attr("y", ys(0) + 4).attr("text-anchor", "end")
      .attr("font-size", 12.5).attr("fill", P.muted).text("0");
    chart.append("line").attr("x1", 0).attr("x2", cw).attr("y1", ys(Q_PLANTED)).attr("y2", ys(Q_PLANTED))
      .attr("stroke", P.green).attr("stroke-width", 1.4).attr("stroke-dasharray", "4,3");
    chart.append("text").attr("x", 0).attr("y", ys(Q_PLANTED) - 5).attr("text-anchor", "start")
      .attr("font-size", 12.5).attr("fill", P.green).text(`planted split · Q = ${Q_PLANTED.toFixed(3)}`);
    const path = d3.line().x((v, k) => xs(k)).y((v) => ys(v)).curve(d3.curveStepAfter);
    chart.append("path").attr("d", path(hist)).attr("fill", "none")
      .attr("stroke", P.accent).attr("stroke-width", 2.2);
    chart.selectAll("circle.h").data(hist).join("circle").attr("class", "h")
      .attr("cx", (v, k) => xs(k)).attr("cy", (v) => ys(v)).attr("r", 3).attr("fill", P.accent);
    const last = hist.length - 1;
    chart.append("circle").attr("cx", xs(last)).attr("cy", ys(hist[last])).attr("r", 5.5)
      .attr("fill", P.accent).attr("stroke", P.paper).attr("stroke-width", 2);
    // current value sits just right of its point (or left, near the chart's edge),
    // haloed in paper so it stays legible over the reference lines
    const labelLeft = xs(last) > cw - 90;
    const onPlanted = Math.abs(ys(hist[last]) - ys(Q_PLANTED)) < 12;   // would sit on the dashed line
    chart.append("text").attr("x", xs(last) + (labelLeft ? -11 : 11)).attr("y", ys(hist[last]) + (onPlanted ? 19 : 4.5))
      .attr("text-anchor", labelLeft ? "end" : "start")
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 13.5)
      .attr("font-weight", 700).attr("fill", P.text)
      .attr("stroke", P.paper).attr("stroke-width", 4).attr("paint-order", "stroke")
      .text(`Q = ${q.toFixed(3)}`);
    chart.append("text").attr("x", cw / 2).attr("y", ch + 16).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`Q after each accepted move (${last} so far)`);
    chart.append("text").attr("x", cw).attr("y", -6).attr("text-anchor", "end")
      .attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.text)
      .text(`${new Set(comm).size} communities now`);

    // ── message line ──
    g.append("text").attr("x", W / 2).attr("y", H - 12).attr("text-anchor", "middle")
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
          const qq = modularity(trial);
          if (qq > bQ + 1e-12) { bQ = qq; bPair = [ids[a], ids[b]]; }
        }
      }
      if (bPair) {
        comm = comm.map((c) => (c === bPair[1] ? bPair[0] : c));
        hist.push(modularity(comm));
        improved = true;
      }
    }
    converged = false; noGain = 0;
    for (let k = 0; k < 500 && !converged; k++) nextMove();
    assignColors();
    msg = `after phase 2 (merge whole communities, then resweep): ${new Set(comm).size} communities · Q = ${modularity(comm).toFixed(3)}`;
    render();
  });
  document.getElementById("w2-lv-reset").addEventListener("click", () => { reset(); render(); });

  U.onThemeChange(render);
  U.lazyBoot("w2-lv-svg", render);
})();
