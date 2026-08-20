/* DKR widget utilities — shared by every interactive figure.
 * Self-contained: depends only on the vendored d3.v7.min.js loaded before it.
 * Widgets register a redraw callback so the light/dark toggle recolors them live.
 */
window.DKR = (function () {
  "use strict";

  // Read the current theme palette from the CSS custom properties the SCSS emits.
  function pal() {
    const s = getComputedStyle(document.documentElement);
    const get = (name, fallback) => (s.getPropertyValue(name) || fallback).trim();
    // NOTE: the "blue"/"blueDark" keys are legacy names for the SECONDARY hue —
    // since the modern palette they carry teal, not blue.
    return {
      accent: get("--dkr-accent", "#d9603b"),
      accentDark: get("--dkr-accent-dark", "#b84a2b"),
      blue: get("--dkr-blue", "#0f8377"),
      blueDark: get("--dkr-blue-dark", "#0a5f56"),
      green: get("--dkr-green", "#199473"),
      yellow: get("--dkr-yellow", "#d9a62e"),
      red: get("--dkr-red", "#cf4a30"),
      purple: get("--dkr-purple", "#7c5cd6"),
      paper: get("--dkr-paper", "#ffffff"),
      muted: get("--dkr-muted", "#8e8e9a"),
      border: get("--dkr-border", "#e8e7e3"),
      text: get("--dkr-text", "#1c1c21"),
      bg: get("--dkr-bg", "#fbfbfa"),
    };
  }

  // Re-render registered widgets when Quarto's theme toggle flips the body class.
  const redrawers = [];
  function onThemeChange(fn) { redrawers.push(fn); }
  const mo = new MutationObserver(() => redrawers.forEach((fn) => { try { fn(); } catch (e) {} }));
  mo.observe(document.body, { attributes: true, attributeFilter: ["class"] });

  // Small-graph helpers -------------------------------------------------------

  // Build {nodes, links} from an edge list over n nodes (0-indexed pairs).
  function graph(n, edges) {
    return {
      nodes: d3.range(n).map((i) => ({ id: i })),
      links: edges.map(([s, t]) => ({ source: s, target: t })),
      edges: edges.map((e) => e.slice()),
      n,
    };
  }

  function adjacency(g) {
    const A = Array.from({ length: g.n }, () => Array(g.n).fill(0));
    g.edges.forEach(([s, t]) => { A[s][t] = 1; A[t][s] = 1; });
    return A;
  }

  function degrees(A) { return A.map((row) => row.reduce((a, b) => a + b, 0)); }

  function neighbors(A, u) {
    const out = [];
    A[u].forEach((v, j) => { if (v) out.push(j); });
    return out;
  }

  // Fixed circular layout (deterministic; widgets should not jiggle).
  function circleLayout(g, cx, cy, r) {
    g.nodes.forEach((d, i) => {
      const a = (2 * Math.PI * i) / g.n - Math.PI / 2;
      d.x = cx + r * Math.cos(a);
      d.y = cy + r * Math.sin(a);
    });
  }

  // Matrix rendering into an SVG group: cells + optional value labels.
  function drawMatrix(gSel, M, opts) {
    const { x = 0, y = 0, cell = 26, fmt = (v) => (v === 0 ? "0" : String(Math.round(v * 100) / 100)),
            color = null, textColor = "#333", borderColor = "#ccc", fontSize = 11 } = opts || {};
    const rows = M.length, cols = M[0].length;
    const grp = gSel.append("g").attr("transform", `translate(${x},${y})`);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        grp.append("rect")
          .attr("x", j * cell).attr("y", i * cell)
          .attr("width", cell).attr("height", cell)
          .attr("fill", color ? color(M[i][j], i, j) : "none")
          .attr("stroke", borderColor).attr("stroke-width", 0.6);
        grp.append("text")
          .attr("x", j * cell + cell / 2).attr("y", i * cell + cell / 2 + fontSize * 0.35)
          .attr("text-anchor", "middle").attr("font-size", fontSize)
          .attr("fill", textColor)
          .text(fmt(M[i][j], i, j));
      }
    }
    return grp;
  }

  // Numeric helpers -----------------------------------------------------------

  function matmul(A, B) {
    const n = A.length, m = B[0].length, k = B.length;
    const C = Array.from({ length: n }, () => Array(m).fill(0));
    for (let i = 0; i < n; i++)
      for (let l = 0; l < k; l++) {
        if (A[i][l] === 0) continue;
        for (let j = 0; j < m; j++) C[i][j] += A[i][l] * B[l][j];
      }
    return C;
  }

  function relu(M) { return M.map((r) => r.map((v) => Math.max(0, v))); }

  // Jacobi eigendecomposition for small symmetric matrices (n <= ~30).
  // Returns { values, vectors } with eigenvalues ascending; vectors[k] is the k-th eigenvector.
  function symEig(Ain) {
    const n = Ain.length;
    const A = Ain.map((r) => r.slice());
    let V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < 100; sweep++) {
      let off = 0;
      for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < 1e-12) break;
      for (let p = 0; p < n - 1; p++) {
        for (let q = p + 1; q < n; q++) {
          if (Math.abs(A[p][q]) < 1e-14) continue;
          const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
          const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < n; k++) {
            const akp = A[k][p], akq = A[k][q];
            A[k][p] = c * akp - s * akq;
            A[k][q] = s * akp + c * akq;
          }
          for (let k = 0; k < n; k++) {
            const apk = A[p][k], aqk = A[q][k];
            A[p][k] = c * apk - s * aqk;
            A[q][k] = s * apk + c * aqk;
          }
          for (let k = 0; k < n; k++) {
            const vkp = V[k][p], vkq = V[k][q];
            V[k][p] = c * vkp - s * vkq;
            V[k][q] = s * vkp + c * vkq;
          }
        }
      }
    }
    const pairs = d3.range(n).map((i) => ({ value: A[i][i], vector: V.map((row) => row[i]) }));
    pairs.sort((a, b) => a.value - b.value);
    return { values: pairs.map((p) => p.value), vectors: pairs.map((p) => p.vector) };
  }

  // Diverging color for signed values in [-max, max]: blue-negative, paper-zero, accent-positive.
  function signedColor(v, max, p) {
    const t = Math.max(-1, Math.min(1, v / (max || 1)));
    const P = p || pal();
    return t >= 0
      ? d3.interpolateRgb(P.paper, P.accent)(t)
      : d3.interpolateRgb(P.paper, P.blue)(-t);
  }

  // Standard widget scaffold: clears container, adds responsive SVG.
  function svgIn(containerId, w, h) {
    const el = d3.select("#" + containerId);
    el.selectAll("*").remove();
    return el.append("svg")
      .attr("viewBox", `0 0 ${w} ${h}`)
      .attr("width", "100%")
      .style("max-width", w + "px")
      .style("display", "block")
      .style("margin", "0 auto");
  }

  return { pal, onThemeChange, graph, adjacency, degrees, neighbors, circleLayout,
           drawMatrix, matmul, relu, symEig, signedColor, svgIn };
})();
