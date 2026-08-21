/* Widget 2.1 — Centrality playground.
 * Krackhardt's kite: pick a measure, watch who wins; click two nodes to add or
 * remove an edge and watch every score recompute. All measures computed here,
 * live, from the current edge set (BFS closeness, Brandes betweenness, power
 * iteration for eigenvector and PageRank).
 */
(function () {
  "use strict";
  const U = window.DKR;
  const N = 10;
  const NAMES = ["Andre", "Beverly", "Carol", "Diane", "Ed", "Fernando", "Garth", "Heather", "Ike", "Jane"];
  const POS = [[89, 39], [295, 25], [60, 122], [189, 82], [320, 122],
               [112, 183], [272, 190], [192, 240], [192, 273], [192, 305]];
  const KITE = [[0, 1], [0, 2], [0, 3], [0, 5], [1, 3], [1, 4], [1, 6], [2, 3], [2, 5],
                [3, 4], [3, 5], [3, 6], [4, 6], [5, 6], [5, 7], [6, 7], [7, 8], [8, 9]];

  const MEASURES = {
    degree: { color: "accent", desc: "count my edges", fmt: (v) => String(v) },
    closeness: { color: "green", desc: "1 / my average distance to everyone", fmt: (v) => v.toFixed(2) },
    betweenness: { color: "purple", desc: "how many shortest paths pass through me", fmt: (v) => v.toFixed(1) },
    eigenvector: { color: "yellow", desc: "my neighbors' importance, recursively", fmt: (v) => v.toFixed(2) },
    pagerank: { color: "blue", desc: "the random surfer's visit rate (β = 0.85)", fmt: (v) => v.toFixed(3) },
  };
  const PALKEY = { accent: "#d9603b", green: "#199473", purple: "#7c5cd6", yellow: "#d9a62e", blue: "#0f8377" };

  let edges = KITE.map((e) => e.slice());
  let measure = "degree";
  let selected = null;

  function adj() {
    const A = Array.from({ length: N }, () => []);
    edges.forEach(([a, b]) => { A[a].push(b); A[b].push(a); });
    return A;
  }

  function bfs(A, s) {
    const dist = Array(N).fill(-1);
    dist[s] = 0;
    const q = [s];
    for (let h = 0; h < q.length; h++) {
      const u = q[h];
      A[u].forEach((v) => { if (dist[v] < 0) { dist[v] = dist[u] + 1; q.push(v); } });
    }
    return dist;
  }

  function compute(kind) {
    const A = adj();
    if (kind === "degree") return A.map((nb) => nb.length);
    if (kind === "closeness") {
      // networkx formula (Wasserman–Faust), correct on disconnected graphs
      return d3.range(N).map((s) => {
        const dist = bfs(A, s);
        const reach = dist.filter((d) => d > 0);
        if (!reach.length) return 0;
        const total = reach.reduce((a, b) => a + b, 0);
        return (reach.length / total) * (reach.length / (N - 1));
      });
    }
    if (kind === "betweenness") {
      // Brandes (2001)
      const bc = Array(N).fill(0);
      for (let s = 0; s < N; s++) {
        const dist = Array(N).fill(-1), sigma = Array(N).fill(0), delta = Array(N).fill(0);
        dist[s] = 0; sigma[s] = 1;
        const order = [s];
        for (let h = 0; h < order.length; h++) {
          const u = order[h];
          A[u].forEach((v) => {
            if (dist[v] < 0) { dist[v] = dist[u] + 1; order.push(v); }
            if (dist[v] === dist[u] + 1) sigma[v] += sigma[u];
          });
        }
        for (let h = order.length - 1; h >= 0; h--) {
          const w = order[h];
          A[w].forEach((v) => {
            if (dist[v] === dist[w] + 1) delta[w] += (sigma[w] / sigma[v]) * (1 + delta[v]);
          });
          if (w !== s) bc[w] += delta[w];
        }
      }
      return bc.map((b) => b / 2);
    }
    if (kind === "eigenvector") {
      let v = Array(N).fill(1);
      for (let it = 0; it < 300; it++) {
        const w = d3.range(N).map((i) => A[i].reduce((a, j) => a + v[j], 0));
        const nrm = Math.sqrt(w.reduce((a, x) => a + x * x, 0)) || 1;
        v = w.map((x) => x / nrm);
      }
      return v;
    }
    // pagerank on the undirected graph, β = 0.85
    const beta = 0.85;
    let pr = Array(N).fill(1 / N);
    for (let it = 0; it < 200; it++) {
      const nxt = Array(N).fill((1 - beta) / N);
      for (let u = 0; u < N; u++) {
        if (!A[u].length) continue;
        const share = (beta * pr[u]) / A[u].length;
        A[u].forEach((v) => { nxt[v] += share; });
      }
      const total = nxt.reduce((a, b) => a + b, 0);
      pr = nxt.map((x) => x / total);
    }
    return pr;
  }

  function render() {
    const P = U.pal();
    const vals = compute(measure);
    const vmax = Math.max(...vals), vmin = Math.min(...vals);
    const champs = d3.range(N).filter((i) => vals[i] >= vmax - 1e-9);
    const color = P[MEASURES[measure].color] || PALKEY[MEASURES[measure].color];

    const svg = U.svgIn("w2-ce-svg", 760, 368);
    const g = svg.append("g").attr("transform", "translate(30,30)");

    g.selectAll("line").data(edges).join("line")
      .attr("x1", (e) => POS[e[0]][0]).attr("y1", (e) => POS[e[0]][1])
      .attr("x2", (e) => POS[e[1]][0]).attr("y2", (e) => POS[e[1]][1])
      .attr("stroke", P.muted).attr("stroke-width", 2).attr("opacity", 0.55);

    const nd = g.selectAll("g.n").data(d3.range(N)).join("g")
      .attr("transform", (i) => `translate(${POS[i][0]},${POS[i][1]})`)
      .style("cursor", "pointer");
    nd.append("circle")
      .attr("r", (i) => 9 + 13 * ((vals[i] - vmin) / ((vmax - vmin) || 1)))
      .attr("fill", (i) => (champs.includes(i) ? color : P.muted))
      .attr("opacity", (i) => (champs.includes(i) ? 1 : 0.5))
      .attr("stroke", (i) => (i === selected ? P.accentDark : "none"))
      .attr("stroke-width", 3);
    nd.append("text").attr("text-anchor", "middle").attr("dy", 4.5)
      .attr("font-size", 12).attr("font-weight", 700).attr("fill", "#fff")
      .text((i) => NAMES[i][0]);
    nd.append("text")
      .attr("x", (i) => 9 + 13 * ((vals[i] - vmin) / ((vmax - vmin) || 1)) + 5)
      .attr("dy", 4.5).attr("font-size", 12).attr("fill", P.muted)
      .text((i) => MEASURES[measure].fmt(vals[i]));
    nd.on("click", (ev, i) => {
      if (selected === null) { selected = i; }
      else if (selected === i) { selected = null; }
      else {
        const k = edges.findIndex(([a, b]) => (a === selected && b === i) || (a === i && b === selected));
        if (k >= 0) edges.splice(k, 1); else edges.push([Math.min(selected, i), Math.max(selected, i)]);
        selected = null;
      }
      render();
    });

    const panel = g.append("g").attr("transform", "translate(420,30)")
      .attr("font-family", "'Source Sans 3', sans-serif");
    panel.append("text").attr("font-size", 17).attr("font-weight", 700).attr("fill", color)
      .text(measure);
    panel.append("text").attr("y", 24).attr("font-size", 13).attr("fill", P.text)
      .text(MEASURES[measure].desc);
    panel.append("text").attr("y", 52).attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.text)
      .text((champs.length > 1 ? "champions (tied): " : "champion: ") + champs.map((i) => NAMES[i]).join(" & "));
    const note = selected === null
      ? "click two nodes to add or remove an edge"
      : `selected ${NAMES[selected]} — click another node to toggle their edge`;
    panel.append("text").attr("y", 84).attr("font-size", 12).attr("fill", selected === null ? P.muted : P.accentDark)
      .text(note);
    if (edges.length !== KITE.length ||
        !edges.every(([a, b]) => KITE.some(([x, y]) => x === a && y === b))) {
      panel.append("text").attr("y", 108).attr("font-size", 12).attr("fill", P.muted)
        .text("(edited graph — press reset to restore the kite)");
    }
  }

  document.querySelectorAll("#w2-ce-widget [data-measure]").forEach((b) =>
    b.addEventListener("click", () => {
      measure = b.getAttribute("data-measure");
      document.querySelectorAll("#w2-ce-widget [data-measure]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.getElementById("w2-ce-reset").addEventListener("click", () => {
    edges = KITE.map((e) => e.slice());
    selected = null;
    render();
  });

  U.onThemeChange(render);
  U.lazyBoot("w2-ce-svg", render);
})();
