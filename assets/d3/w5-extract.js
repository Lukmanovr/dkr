/* Widget 5.4 — Cleaning up after the extractor.
 * The construction figure's five extracted triples, with the three repair
 * passes as toggles: canonicalization merges aliases, the schema check drops
 * ill-typed facts, and the provenance check drops facts no sentence supports.
 * Readouts (entity count, kept facts) recompute from the actual triple list.
 */
(function () {
  "use strict";
  const U = window.DKR;

  const ALIASES = { "IU": "Innopolis University", "Republic of Tatarstan": "Tatarstan" };
  // [head, rel, tail, supported-by-text, type-ok]
  const RAW = [
    ["Innopolis University", "located_in", "Innopolis", true, true],
    ["IU", "located_in", "Republic of Tatarstan", true, true],
    ["Innopolis University", "founded_in", "2012", true, true],
    ["Volga", "located_in", "Innopolis", false, true],
    ["IU", "founded_in", "Tatarstan", true, false],
  ];

  function pipeline(canon, schema, prov) {
    let triples = RAW.map(([h, r, t, sup, ok]) => ({
      h: canon ? (ALIASES[h] || h) : h,
      r,
      t: canon ? (ALIASES[t] || t) : t,
      sup, ok,
      dropped: null,
    }));
    triples.forEach((tr) => {
      if (schema && !tr.ok) tr.dropped = "schema: founded_in expects a year";
      else if (prov && !tr.sup) tr.dropped = "provenance: no sentence says this";
    });
    // canonicalization can create exact duplicates — merge them
    const seen = new Set();
    triples.forEach((tr) => {
      if (tr.dropped) return;
      const key = `${tr.h}|${tr.r}|${tr.t}`;
      if (seen.has(key)) tr.dropped = "duplicate after canonicalization";
      else seen.add(key);
    });
    const kept = triples.filter((t) => !t.dropped);
    const ents = new Set();
    kept.forEach((t) => { ents.add(t.h); ents.add(t.t); });
    return { triples, kept, ents };
  }

  const box = (id) => document.getElementById(id).checked;

  function render() {
    const P = U.pal();
    const { triples, kept, ents } = pipeline(box("w5-ex-canon"), box("w5-ex-schema"), box("w5-ex-prov"));
    const svg = U.svgIn("w5-ex-svg", 760, 264);
    svg.attr("font-family", "'JetBrains Mono', monospace");
    const g = svg.append("g");

    triples.forEach((tr, i) => {
      const y = 34 + i * 40;
      const dead = !!tr.dropped;
      g.append("text").attr("x", 40).attr("y", y)
        .attr("font-size", 12.5)
        .attr("fill", dead ? P.muted : P.green)
        .attr("opacity", dead ? 0.55 : 1)
        .attr("text-decoration", dead ? "line-through" : null)
        .text(`(${tr.h}, ${tr.r}, ${tr.t})`);
      if (dead) {
        g.append("text").attr("x", 40).attr("y", y + 17)
          .attr("font-family", "'Source Sans 3', sans-serif").attr("font-size", 12.5)
          .attr("fill", "#cf4a30").text("✗ " + tr.dropped);
      }
    });

    const raw_ents = pipeline(false, false, false).ents.size;
    g.append("g").attr("font-family", "'Source Sans 3', sans-serif")
      .call((p) => {
        p.append("text").attr("x", 560).attr("y", 46).attr("text-anchor", "middle")
          .attr("font-size", 13).attr("font-weight", 700).attr("fill", P.text)
          .text("the graph you would ship:");
        p.append("text").attr("x", 560).attr("y", 74).attr("text-anchor", "middle")
          .attr("font-size", 12.5).attr("fill", P.text)
          .text(`entities: ${ents.size}` + (ents.size > 5 ? "  (aliases inflating!)" : ""));
        p.append("text").attr("x", 560).attr("y", 96).attr("text-anchor", "middle")
          .attr("font-size", 12.5).attr("fill", P.text).text(`facts kept: ${kept.length} of ${RAW.length}`);
        const clean = box("w5-ex-canon") && box("w5-ex-schema") && box("w5-ex-prov");
        const falseKept = kept.some((t) => !t.sup);
        p.append("rect").attr("x", 425).attr("y", 116).attr("width", 270).attr("height", 52)
          .attr("rx", 12).attr("fill", clean ? P.green : falseKept ? "#cf4a30" : P.yellow).attr("opacity", 0.9);
        p.append("text").attr("x", 560).attr("y", 138).attr("text-anchor", "middle")
          .attr("font-size", 12.5).attr("font-weight", 700).attr("fill", clean || falseKept ? "#fff" : "#5c4508")
          .text(clean ? "clean: 3 facts, 5 entities" : falseKept ? "a FALSE fact is in your KG" : "cleaner — keep going");
        p.append("text").attr("x", 560).attr("y", 157).attr("text-anchor", "middle")
          .attr("font-size", 12.5).attr("fill", clean || falseKept ? "#fff" : "#5c4508")
          .text(clean ? "ready for GraphRAG duty" : falseKept ? "and GraphRAG will cite it proudly" : "");
      });

    g.append("text").attr("x", 380).attr("y", 250).attr("text-anchor", "middle")
      .attr("font-family", "'Source Sans 3', sans-serif").attr("font-size", 12.5).attr("fill", P.muted)
      .text(`raw extraction: ${RAW.length} triples, ${raw_ents} "entities" — flip the repairs on and both numbers tell the truth`);
  }

  ["w5-ex-canon", "w5-ex-schema", "w5-ex-prov"].forEach((id) =>
    document.getElementById(id).addEventListener("change", render));

  U.onThemeChange(render);
  U.lazyBoot("w5-ex-svg", render);
})();
