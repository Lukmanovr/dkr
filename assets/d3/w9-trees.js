/* Widget 9.2 — Computation trees, unrolled and compared.
 * Pick a pair of nodes (possibly from different graphs); their neighborhoods
 * unroll into computation trees to the chosen depth. A canonical form decides
 * "identical" — the exact condition under which message passing must give
 * both nodes the same embedding. Deterministic.
 */
(function () {
  "use strict";
  const U = window.DKR;

  const GRAPHS = {
    tri: [[0, 1], [1, 2], [2, 0]],
    hex: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]],
    p4: [[0, 1], [1, 2], [2, 3]],
    p3: [[0, 1], [1, 2]],
    dec: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0], [0, 6], [6, 7], [7, 8], [8, 9], [9, 1]],
    bic: [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [5, 6], [6, 7], [7, 8], [8, 9], [9, 5], [0, 5]],
  };
  const PRESETS = [
    { a: ["tri", 0, "a triangle node"], b: ["hex", 0, "a hexagon node"] },
    { a: ["p3", 1, "P₃'s middle node"], b: ["p4", 1, "P₄'s 2nd node"] },
    { a: ["dec", 0, "decalin, a fused carbon"], b: ["bic", 0, "bicyclopentyl, a bridge carbon"] },
  ];

  function adj(edges) {
    const A = {};
    edges.forEach(([a, b]) => {
      (A[a] = A[a] || []).push(b);
      (A[b] = A[b] || []).push(a);
    });
    return A;
  }

  // unrolled tree: children of (v) are ALL neighbors of v (backtracking allowed)
  function unroll(A, v, depth) {
    const node = { v, kids: [] };
    if (depth > 0) A[v].forEach((u) => node.kids.push(unroll(A, u, depth - 1)));
    return node;
  }
  function canon(t) {
    return "(" + t.kids.map(canon).sort().join("") + ")";
  }
  function leafCount(t) {
    return t.kids.length ? t.kids.reduce((s, k) => s + leafCount(k), 0) : 1;
  }

  let pi = 0, depth = 2;

  function layout(t, x0, width, y, dy, out) {
    const n = leafCount(t);
    let x = x0;
    const centers = t.kids.map((k) => {
      const w = (width * leafCount(k)) / n;
      const c = layout(k, x, w, y + dy, dy, out);
      x += w;
      return c;
    });
    const cx = t.kids.length ? (centers[0] + centers[centers.length - 1]) / 2 : x0 + width / 2;
    out.push({ x: cx, y, kids: centers.map((c, i) => ({ x: c, y: y + dy })) });
    return cx;
  }

  function drawTree(g, A, v, x0, width, P, root) {
    const t = unroll(A, v, depth);
    const nodes = [];
    layout(t, x0, width, 56, 62, nodes);
    nodes.forEach((n) => n.kids.forEach((k) =>
      g.append("line").attr("x1", n.x).attr("y1", n.y).attr("x2", k.x).attr("y2", k.y)
        .attr("stroke", P.muted).attr("stroke-width", 1.3)));
    nodes.forEach((n, i) => {
      const isRoot = i === nodes.length - 1;
      g.append("circle").attr("cx", n.x).attr("cy", n.y).attr("r", isRoot ? 9 : 6)
        .attr("fill", isRoot ? P.accent : P.blue);
    });
    return canon(t);
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w9-tr-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const Pr = PRESETS[pi];

    g.append("text").attr("x", 380).attr("y", 24).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text(`unrolled to depth ${depth} — children of a node are ALL its neighbors (walks may backtrack)`);

    const cA = drawTree(g, adj(GRAPHS[Pr.a[0]]), Pr.a[1], 30, 330, P);
    const cB = drawTree(g, adj(GRAPHS[Pr.b[0]]), Pr.b[1], 400, 330, P);

    g.append("text").attr("x", 195).attr("y", 276).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text).text(Pr.a[2]);
    g.append("text").attr("x", 565).attr("y", 276).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text).text(Pr.b[2]);

    const same = cA === cB;
    const verdict = same
      ? `identical trees at depth ${depth} — every message-passing GNN gives these nodes the SAME embedding`
      : `trees differ at depth ${depth} — a GNN with enough layers CAN separate these nodes`;
    g.append("text").attr("x", 380).attr("y", 312).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 600)
      .attr("fill", same ? P.accentDark : P.green).text(verdict);
  }

  document.querySelectorAll("#w9-tr-widget [data-preset]").forEach((b) =>
    b.addEventListener("click", () => {
      pi = +b.getAttribute("data-preset");
      document.querySelectorAll("#w9-tr-widget [data-preset]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.querySelectorAll("#w9-tr-widget [data-depth]").forEach((b) =>
    b.addEventListener("click", () => {
      depth = +b.getAttribute("data-depth");
      document.querySelectorAll("#w9-tr-widget [data-depth]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w9-tr-svg", render);
})();
