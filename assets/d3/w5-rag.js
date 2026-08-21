/* Widget 5.3 — Retrieval bake-off: similarity vs structure.
 * Six tiny documents, three questions, two retrieval strategies. Vector mode
 * scores documents by real cosine similarity over bag-of-words (computed
 * here); graph mode links entities and pulls the k-hop subgraph of a KG
 * extracted from the same documents. The metric: are the gold supporting
 * facts inside the retrieved context?
 */
(function () {
  "use strict";
  const U = window.DKR;

  const DOCS = [
    "Innopolis University is a young IT university located in the city of Innopolis.",
    "Innopolis is a town in the Republic of Tatarstan, founded in 2012.",
    "Kazan Federal University is one of the oldest universities in Russia, located in Kazan.",
    "Kazan is the capital of Tatarstan and a major university city.",
    "The Volga river flows through the Republic of Tatarstan.",
    "Innopolis University hosts an annual robotics olympiad for school students.",
  ];
  // the KG extracted from the docs: [head, rel, tail, source-doc]
  const KG = [
    ["Innopolis University", "located_in", "Innopolis", 0],
    ["Innopolis", "located_in", "Tatarstan", 1],
    ["KFU", "located_in", "Kazan", 2],
    ["Kazan", "capital_of", "Tatarstan", 3],
    ["Volga", "flows_through", "Tatarstan", 4],
    ["Innopolis University", "hosts", "robotics olympiad", 5],
  ];
  const QUESTIONS = [
    { q: "Which region is Innopolis University located in?",
      entities: ["Innopolis University"], hops: 2, gold: [0, 1] },
    { q: "In which city is Kazan Federal University located?",
      entities: ["KFU"], hops: 1, gold: [2] },
    { q: "Do local school students get any outreach events?",
      entities: [], hops: 1, gold: [5] },
  ];
  const STOP = new Set(["is", "a", "the", "in", "of", "for", "which", "do", "any", "get", "and",
                        "one", "an", "on", "to", "it", "its", "was", "are", "through"]);

  function bag(text) {
    const counts = {};
    text.toLowerCase().replace(/[^a-zа-я0-9\s]/g, "").split(/\s+/).forEach((w) => {
      if (w && !STOP.has(w)) counts[w] = (counts[w] || 0) + 1;
    });
    return counts;
  }
  function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (const w in a) { na += a[w] * a[w]; if (b[w]) dot += a[w] * b[w]; }
    for (const w in b) nb += b[w] * b[w];
    return dot / (Math.sqrt(na * nb) || 1);
  }
  const DOC_BAGS = DOCS.map(bag);

  function vectorRetrieve(q, k) {
    const qb = bag(q);
    return DOCS.map((_, i) => ({ i, s: cosine(qb, DOC_BAGS[i]) }))
      .sort((a, b) => b.s - a.s).slice(0, k);
  }

  function graphRetrieve(entities, hops) {
    const frontier = new Set(entities);
    const facts = new Set();
    for (let h = 0; h < hops; h++) {
      const next = new Set(frontier);
      KG.forEach(([hd, , tl], fi) => {
        if (frontier.has(hd) || frontier.has(tl)) { facts.add(fi); next.add(hd); next.add(tl); }
      });
      next.forEach((e) => frontier.add(e));
    }
    return [...facts];
  }

  let qi = 0, mode = "vector";
  const K = 2;

  function render() {
    const P = U.pal();
    const Q = QUESTIONS[qi];
    const svg = U.svgIn("w5-rg-svg", 760, 330);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    g.append("text").attr("x", 380).attr("y", 26).attr("text-anchor", "middle")
      .attr("font-size", 13.5).attr("font-weight", 700).attr("fill", P.text)
      .text(`“${Q.q}”`);

    let retrievedFacts, contextLines, title;
    if (mode === "vector") {
      const top = vectorRetrieve(Q.q, K);
      title = `top-${K} documents by cosine similarity (scores computed live)`;
      contextLines = top.map(({ i, s }) => ({ text: `[doc ${i}] (${s.toFixed(2)}) ${DOCS[i]}`, hot: false }));
      const docSet = new Set(top.map((t) => t.i));
      retrievedFacts = KG.map((f, fi) => fi).filter((fi) => docSet.has(KG[fi][3]));
    } else {
      const linked = Q.entities;
      const facts = linked.length ? graphRetrieve(linked, Q.hops) : [];
      title = linked.length
        ? `entity link: ${linked.join(", ")} → ${Q.hops}-hop subgraph`
        : "entity link: NOTHING in the question matches a KG node — retrieval comes back empty";
      contextLines = facts.map((fi) => {
        const [h, r, t, d] = KG[fi];
        return { text: `(${h}, ${r}, ${t})   ← doc ${d}`, hot: true };
      });
      retrievedFacts = facts;
    }

    g.append("text").attr("x", 40).attr("y", 58).attr("font-size", 12.5)
      .attr("font-weight", 600).attr("fill", P.muted).text(title);
    contextLines.slice(0, 4).forEach((ln, i) => {
      g.append("text").attr("x", 40).attr("y", 84 + i * 24).attr("font-size", 12)
        .attr("fill", ln.hot ? P.blue : P.text).text(ln.text.slice(0, 96));
    });
    if (!contextLines.length) {
      g.append("text").attr("x", 40).attr("y", 84).attr("font-size", 12)
        .attr("fill", P.muted).text("(empty context)");
    }

    const covered = Q.gold.filter((fi) => retrievedFacts.includes(fi));
    const full = covered.length === Q.gold.length;
    g.append("text").attr("x", 40).attr("y", 210).attr("font-size", 12.5)
      .attr("font-weight", 600).attr("fill", P.muted)
      .text(`gold supporting facts (${Q.gold.length}):`);
    Q.gold.forEach((fi, i) => {
      const [h, r, t] = KG[fi];
      const got = covered.includes(fi);
      g.append("text").attr("x", 40).attr("y", 232 + i * 22).attr("font-size", 12)
        .attr("fill", got ? P.green : "#cf4a30")
        .text(`${got ? "✓" : "✗"} (${h}, ${r}, ${t})`);
    });

    g.append("rect").attr("x", 470).attr("y", 218).attr("width", 250).attr("height", 56)
      .attr("rx", 12).attr("fill", full ? P.green : "#cf4a30").attr("opacity", 0.9);
    g.append("text").attr("x", 595).attr("y", 242).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("font-weight", 700).attr("fill", "#fff")
      .text(`coverage: ${covered.length}/${Q.gold.length}`);
    g.append("text").attr("x", 595).attr("y", 262).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", "#fff")
      .text(full ? "an LLM could answer from this" : "no reader can answer from this");

    g.append("text").attr("x", 380).attr("y", 316).attr("text-anchor", "middle")
      .attr("font-size", 12).attr("fill", P.muted)
      .text("coverage = are the needed facts inside the retrieved context? (the reader can be any LLM — or you)");
  }

  document.querySelectorAll("#w5-rg-widget [data-q]").forEach((b) =>
    b.addEventListener("click", () => {
      qi = +b.getAttribute("data-q");
      document.querySelectorAll("#w5-rg-widget [data-q]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.querySelectorAll("#w5-rg-widget [data-mode]").forEach((b) =>
    b.addEventListener("click", () => {
      mode = b.getAttribute("data-mode");
      document.querySelectorAll("#w5-rg-widget [data-mode]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w5-rg-svg", render);
})();
