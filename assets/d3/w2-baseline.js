/* Widget 2.x — The ceiling: four pipelines on Cora, and what each one can see.
 *
 * A horizontal bar chart of node-classification accuracy on Cora (7 topics)
 * for the four pipelines the lecture's §6 names: chance, seven structural
 * statistics, the same classifier plus a one-hot of the Louvain community,
 * and — as the Week 6 preview — a GCN computing on structure and the papers'
 * words jointly. Beside the bars, a panel says WHAT the selected pipeline can
 * see, as three chips in a fixed order (statistics · position · words), and
 * one sentence from the lecture's own reasoning explains the jump or the plateau.
 *
 * Data provenance is the whole point of the figure, so it is explicit below:
 * chance is arithmetic on the number of classes; the 19% and 62% are the
 * lecture's Lab 2 measurements; the 81% is the lecture's Week 6 preview.
 * Every other number on the canvas (the 43-point jump, "one time in 7", bar
 * lengths) is derived from that table at render time.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const W = 760, H = 400;

  const CLASSES = 7;                         // Cora has seven topics
  const ROWS = [
    { key: "chance", name: `chance: guess one of ${CLASSES} topics`,
      acc: Math.round(100 / CLASSES), source: "Lab 2 measurement",
      chips: { statistics: "off", position: "off", words: "off" } },
    { key: "stats", name: "seven structural statistics",
      acc: 19, source: "Lab 2 measurement",
      chips: { statistics: "on", position: "off", words: "off" } },
    { key: "community", name: "statistics + community one-hot",
      acc: 62, source: "Lab 2 measurement",
      chips: { statistics: "on", position: "onehot", words: "off" } },
    { key: "gcn", name: "GCN on structure and words",
      acc: 81, source: "Week 6 preview",
      chips: { statistics: "implicit", position: "learned", words: "on" } },
  ];
  const CHIPS = ["statistics", "position", "words"];   // fixed order, never re-sorted

  // How each chip reads in each state: [is it filled?, the short "how"].
  const HOW = {
    statistics: { off: [false, "not used"], on: [true, "how central / clustered"],
                  implicit: [true, "learned implicitly"] },
    position:   { off: [false, "not used"], onehot: [true, "region, as a one-hot"],
                  learned: [true, "region, learned"] },
    words:      { off: [false, "never read"], on: [true, "the paper's text"] },
  };

  const byKey = Object.fromEntries(ROWS.map((r) => [r.key, r]));
  const JUMP = byKey.community.acc - byKey.stats.acc;   // 62 − 19 = 43

  // One plain sentence per row, in the lecture's own words, numbers from the table.
  const WHY = {
    chance: `Nothing is seen, so the best you can do is guess one of the ${CLASSES} topics and be right about one time in ${CLASSES}. Every pipeline has to beat this floor.`,
    stats: `How central or tight-knit a paper is says almost nothing about its topic: statistics are topic-blind, and ${byKey.stats.acc}% is what that blindness costs.`,
    community: `Under homophily, topic lives in where you sit, so a one-hot of the region lifts the same classifier by ${JUMP} points. It then plateaus: a hard partition is a lossy, task-blind compression of position, and the pipeline never reads the words.`,
    gcn: `Computing on structure and the papers' words jointly makes position continuous and learned, and finally reads the text: ${byKey.gcn.acc}% on Cora, a Week 6 preview.`,
  };

  let row = "community";                     // the semester's baseline number

  // ── small helpers ──────────────────────────────────────────────────────────
  // Greedy word wrap by character count: every widget text is monospace on the
  // site (theme SCSS forces JetBrains Mono), so 0.6 em per character is exact.
  function wrap(text, maxChars) {
    const lines = [];
    let cur = "";
    for (const w of text.split(" ")) {
      if (cur && (cur + " " + w).length > maxChars) { lines.push(cur); cur = w; }
      else cur = cur ? cur + " " + w : w;
    }
    if (cur) lines.push(cur);
    return lines;
  }
  function lum(hex) {
    const h = hex.replace("#", "");
    const c = [0, 2, 4].map((i) => parseInt(h.length === 3 ? h[i / 2] + h[i / 2] : h.slice(i, i + 2), 16) / 255)
      .map((v) => (v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4)));
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
  }
  function contrast(a, b) {
    const A = lum(a), B = lum(b);
    return (Math.max(A, B) + 0.05) / (Math.min(A, B) + 0.05);
  }
  // A filled chip needs legible ink in BOTH themes: try each accent shade with
  // white and with the page ground, keep the pairing that contrasts best.
  function bestFill(P) {
    let best = null;
    for (const fill of [P.accentDark, P.accent]) {
      for (const ink of ["#ffffff", P.bg]) {
        const c = contrast(fill, ink);
        if (!best || c > best.c) best = { fill, ink, c };
      }
    }
    return best;
  }

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w2-bl-svg", W, H);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const MONO = "'JetBrains Mono', monospace";
    const sel = byKey[row];

    g.append("text").attr("x", 20).attr("y", 22).attr("font-size", 12.5).attr("fill", P.muted)
      .text(`Cora node classification, ${CLASSES} topics — four pipelines, accuracy`);

    // ── the bars ──
    const x0 = 22, L = 372, y0 = 50, pitch = 50, barH = 18;
    const xOf = (acc) => x0 + (acc / 100) * L;
    const yAxis = y0 + (ROWS.length - 1) * pitch + barH + 30;      // 248

    // chance reference: a faint line from the end of the chance bar to the axis
    const xChance = xOf(byKey.chance.acc);
    g.append("line").attr("x1", xChance).attr("y1", y0 + 18).attr("x2", xChance).attr("y2", yAxis)
      .attr("stroke", P.muted).attr("stroke-width", 1).attr("stroke-dasharray", "3 3").attr("opacity", 0.5);

    ROWS.forEach((r, i) => {
      const yTop = y0 + i * pitch;
      const on = r.key === row;
      const w = xOf(r.acc) - x0;
      const yb = yTop + 18;

      // row label + provenance tag
      g.append("text").attr("x", x0).attr("y", yTop + 11).attr("font-size", 13)
        .attr("font-weight", on ? 700 : 400).attr("fill", P.text).text(r.name);
      g.append("text").attr("x", 420).attr("y", yTop + 11).attr("text-anchor", "end")
        .attr("font-size", 12.5).attr("fill", P.muted).text(r.source);

      // bar: square at the baseline, 4 px rounded data-end
      const rr = Math.min(4, w / 2);
      const bar = g.append("path")
        .attr("d", `M ${x0} ${yb} H ${x0 + w - rr} a ${rr} ${rr} 0 0 1 ${rr} ${rr} V ${yb + barH - rr} a ${rr} ${rr} 0 0 1 ${-rr} ${rr} H ${x0} Z`)
        .attr("fill", on ? P.accent : P.blue)
        .attr("fill-opacity", on ? 1 : 0.5);

      // value label at the bar end
      g.append("text").attr("x", x0 + w + 8).attr("y", yb + barH / 2 + 5)
        .attr("font-family", MONO).attr("font-size", 14).attr("font-weight", 700)
        .attr("fill", P.text).text(`${r.acc}%`);

      // the whole row is the hit target: click selects, hover lifts the bar
      const hit = g.append("rect").attr("x", 12).attr("y", yTop - 4).attr("width", 418).attr("height", 44)
        .attr("fill", P.paper).attr("fill-opacity", 0).style("cursor", "pointer");
      hit.append("title").text(`${r.acc}% — ${r.name} (${r.source})`);
      hit.on("mouseenter", () => { if (!on) bar.attr("fill-opacity", 0.8); })
         .on("mouseleave", () => { if (!on) bar.attr("fill-opacity", 0.5); })
         .on("click", () => select(r.key));
    });

    // baseline, axis and its two ticks
    g.append("line").attr("x1", x0).attr("y1", y0 + 14).attr("x2", x0).attr("y2", yAxis)
      .attr("stroke", P.border).attr("stroke-width", 1);
    g.append("line").attr("x1", x0).attr("y1", yAxis).attr("x2", xOf(100)).attr("y2", yAxis)
      .attr("stroke", P.border).attr("stroke-width", 1);
    [[xOf(50), "50%"], [xOf(100), "100%"]].forEach(([x, t]) => {
      g.append("line").attr("x1", x).attr("y1", yAxis).attr("x2", x).attr("y2", yAxis + 4)
        .attr("stroke", P.border).attr("stroke-width", 1);
      g.append("text").attr("x", x).attr("y", yAxis + 17).attr("text-anchor", "middle")
        .attr("font-family", MONO).attr("font-size", 12.5).attr("fill", P.muted).text(t);
    });
    g.append("text").attr("x", xChance).attr("y", yAxis + 17).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted).text("chance");

    // ── key to the three chips (what a pipeline could see) ──
    g.append("text").attr("x", 20).attr("y", 292).attr("font-size", 12.5).attr("font-weight", 700)
      .attr("fill", P.muted).text("the three things a pipeline could see");
    [["statistics", "how central or clustered a node is"],
     ["position", "which region of the graph it sits in"],
     ["words", "the paper's own text"]].forEach(([k, d], i) => {
      g.append("text").attr("x", 20).attr("y", 310 + i * 17).attr("font-size", 12.5)
        .attr("fill", P.muted).text(`${k} — ${d}`);
    });

    // ── the panel: what the selected pipeline sees ──
    const px = 450, pw = 290;
    g.append("text").attr("x", px).attr("y", 60).attr("font-size", 12.5).attr("fill", P.muted)
      .text("what this pipeline can see");
    g.append("rect").attr("x", px).attr("y", 70).attr("width", 10).attr("height", 10).attr("rx", 2)
      .attr("fill", P.accent);
    g.append("text").attr("x", px + 16).attr("y", 79).attr("font-size", 13).attr("font-weight", 700)
      .attr("fill", P.text).text(sel.name);

    const chipW = 100, chipH = 22, fill = bestFill(P);
    CHIPS.forEach((c, i) => {
      const state = sel.chips[c];
      const [filled, how] = HOW[c][state];
      const y = 96 + i * 28;
      const implicit = state === "implicit";
      const chip = g.append("rect").attr("x", px).attr("y", y).attr("width", chipW).attr("height", chipH)
        .attr("rx", 11);
      if (filled && !implicit) {
        chip.attr("fill", fill.fill);
      } else if (implicit) {
        chip.attr("fill", P.accent).attr("fill-opacity", 0.28)
          .attr("stroke", P.accent).attr("stroke-width", 1.2).attr("stroke-dasharray", "3 3");
      } else {
        chip.attr("fill", "none").attr("stroke", P.muted).attr("stroke-width", 1.2);
      }
      g.append("text").attr("x", px + chipW / 2).attr("y", y + chipH / 2 + 4.5)
        .attr("text-anchor", "middle").attr("font-size", 12.5).attr("font-weight", 700)
        .attr("fill", filled ? (implicit ? P.text : fill.ink) : P.muted).text(c);
      g.append("text").attr("x", px + chipW + 10).attr("y", y + chipH / 2 + 4.5)
        .attr("font-size", 12.5).attr("fill", filled ? P.text : P.muted).text(how);
    });

    const maxChars = Math.floor(pw / (12.5 * 0.6));            // 38 monospace columns
    wrap(WHY[row], maxChars - 1).forEach((line, i) => {
      g.append("text").attr("x", px).attr("y", 196 + i * 16).attr("font-size", 12.5)
        .attr("fill", P.text).text(line);
    });

    // ── verdict ──
    g.append("rect").attr("x", 20).attr("y", 370).attr("width", 4).attr("height", 18).attr("fill", P.green);
    g.append("text").attr("x", 32).attr("y", 384).attr("font-size", 13.5).attr("font-weight", 700)
      .attr("fill", P.text)
      .text(`The ${JUMP}-point jump comes from position, not from a better statistic.`);
  }

  function select(key) {
    row = key;
    for (const o of document.querySelectorAll("#w2-bl-widget [data-row]")) {
      o.classList.toggle("active", o.dataset.row === key);
    }
    render();
  }
  for (const b of document.querySelectorAll("#w2-bl-widget [data-row]")) {
    b.addEventListener("click", () => select(b.dataset.row));
  }
  U.onThemeChange(render);
  U.lazyBoot("w2-bl-svg", render);
})();
