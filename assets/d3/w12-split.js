/* Widget 12.1 — Assign every edge a job, then face the auditor.
 * A 10-node graph; click an edge to cycle its role: message → supervision →
 * test. The auditor checks the three anti-leakage rules live and names the
 * exact violation. Deterministic.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const POS = [[90, 70], [200, 50], [310, 80], [420, 55], [530, 90],
               [120, 190], [230, 210], [340, 180], [450, 210], [560, 170]];
  const EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [1, 6], [2, 7], [3, 8], [4, 9],
                 [5, 6], [6, 7], [7, 8], [8, 9], [1, 7], [2, 6]];
  const ROLES = ["message", "supervision", "test"];
  const COLKEY = { message: "blue", supervision: "green", test: "accent" };
  // start: a deliberately broken assignment (everything message = classic leak)
  let role = EDGES.map(() => 0);

  function audit(P) {
    const counts = [0, 0, 0];
    role.forEach((r) => counts[r]++);
    if (counts[2] === 0) return ["no test edges: nothing to evaluate on — assign some", P.accentDark, false];
    if (counts[1] === 0) return ["no supervision edges: nothing to train the scorer on", P.accentDark, false];
    if (counts[0] < 5) return ["message graph too thin: the GNN has almost nothing to propagate over", P.accentDark, false];
    return ["split legal: test unseen, supervision scored-not-propagated, messages flow ✓", P.green, true];
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w12-sp-svg", 760, 300);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const colOf = { message: P.blue, supervision: P.green, test: P.accent };

    g.append("text").attr("x", 380).attr("y", 22).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("click an edge to cycle: message → supervision → test — the auditor below never sleeps");

    EDGES.forEach(([a, b], i) => {
      const r = ROLES[role[i]];
      g.append("line")
        .attr("x1", POS[a][0]).attr("y1", POS[a][1] + 30)
        .attr("x2", POS[b][0]).attr("y2", POS[b][1] + 30)
        .attr("stroke", colOf[r]).attr("stroke-width", r === "message" ? 2.5 : 3.5)
        .attr("stroke-dasharray", r === "test" ? "6 4" : r === "supervision" ? "2 3" : null)
        .attr("opacity", 0.85).style("cursor", "pointer")
        .on("click", () => { role[i] = (role[i] + 1) % 3; render(); });
    });
    POS.forEach(([x, y], v) => {
      g.append("circle").attr("cx", x).attr("cy", y + 30).attr("r", 11).attr("fill", P.muted);
      g.append("text").attr("x", x).attr("y", y + 34).attr("text-anchor", "middle")
        .attr("font-size", 12).attr("font-weight", 700).attr("fill", P.bg).text(v);
    });
    const counts = [0, 0, 0];
    role.forEach((r) => counts[r]++);
    g.append("text").attr("x", 380).attr("y", 262).attr("text-anchor", "middle")
      .attr("font-family", "'JetBrains Mono', monospace").attr("font-size", 12.5).attr("fill", P.text)
      .text(`message ${counts[0]} · supervision ${counts[1]} · test ${counts[2]}`);
    const [verdict, col] = audit(P);
    g.append("text").attr("x", 380).attr("y", 288).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600).attr("fill", col).text(verdict);
  }

  document.getElementById("w12-sp-reset").addEventListener("click", () => {
    role = EDGES.map(() => 0);
    render();
  });
  document.getElementById("w12-sp-auto").addEventListener("click", () => {
    // a legal 70/15/15-ish assignment, deterministic
    role = EDGES.map((_, i) => (i % 7 === 3 ? 1 : i % 7 === 6 ? 2 : 0));
    render();
  });

  U.onThemeChange(render);
  U.lazyBoot("w12-sp-svg", render);
})();
