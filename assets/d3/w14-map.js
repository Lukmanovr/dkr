/* Widget 14.4 — Where this course goes to work.
 * Five sectors of the graph-ML job map; each card names the task, the
 * architecture family, and the weeks of THIS course that taught it.
 */
(function () {
  "use strict";
  const U = window.DKR;
  const SECTORS = {
    recsys: {
      title: "Recommendation",
      task: "rank items a user will interact with next",
      arch: "two-tower retrieval · LightGCN · PinSage-style sampling",
      weeks: "Weeks 3, 6, 11, 14",
      where: "marketplaces, feeds, streaming — the largest employer of GNNs",
      col: "blue",
    },
    fraud: {
      title: "Integrity & fraud",
      task: "flag colluding accounts, laundering rings, fake engagement",
      arch: "heterogeneous + temporal GNNs over typed event graphs",
      weeks: "Weeks 2, 10, 14",
      where: "payments, banks, marketplaces — adversarial, time-critical",
      col: "accent",
    },
    molecules: {
      title: "Drug discovery",
      task: "predict properties; generate and screen candidates",
      arch: "MPNNs with edge features · graph transformers (GPS)",
      weeks: "Weeks 7, 9, 12, 13",
      where: "pharma & biotech — where expressiveness genuinely binds",
      col: "green",
    },
    kg: {
      title: "KG platform & GraphRAG",
      task: "curate a knowledge graph; ground LLM answers in it",
      arch: "KG embeddings · query reasoning · retrieval + LLM",
      weeks: "Weeks 4, 5, 14",
      where: "search, enterprise data teams, assistants",
      col: "yellow",
    },
    rdl: {
      title: "ML infra & relational DL",
      task: "learn on the company database itself — tables as temporal graphs",
      arch: "RelBench-style hetero-temporal GNNs · sampling systems",
      weeks: "Weeks 10, 11, 14",
      where: "the frontier: replacing hand-built feature pipelines",
      col: "purple",
    },
  };
  let sel = "recsys";

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w14-mp-svg", 760, 290);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");
    const keys = Object.keys(SECTORS);

    keys.forEach((k, idx) => {
      const s = SECTORS[k];
      const x = 42 + idx * 140;
      const hot = k === sel;
      g.append("rect").attr("x", x).attr("y", 34).attr("width", 128).attr("height", 54)
        .attr("rx", 10).attr("fill", P[s.col]).attr("opacity", hot ? 0.9 : 0.25);
      const words = s.title.split(" & ");
      words.forEach((w, j) => {
        g.append("text").attr("x", x + 64)
          .attr("y", 56 + j * 17 - (words.length - 1) * 7)
          .attr("text-anchor", "middle").attr("font-size", 12.5)
          .attr("font-weight", 700).attr("fill", hot ? "#fff" : P.text)
          .text(words.length > 1 && j === 1 ? "& " + w : w);
      });
    });

    const s = SECTORS[sel];
    g.append("rect").attr("x", 60).attr("y", 112).attr("width", 640).attr("height", 150)
      .attr("rx", 12).attr("fill", P[s.col]).attr("opacity", 0.1);
    const rows = [
      ["the task", s.task],
      ["the tools", s.arch],
      ["taught in", s.weeks],
      ["who hires", s.where],
    ];
    rows.forEach(([label, val], j) => {
      const y = 140 + j * 32;
      g.append("text").attr("x", 92).attr("y", y).attr("font-size", 12.5)
        .attr("font-weight", 700).attr("fill", P[s.col]).text(label);
      g.append("text").attr("x", 185).attr("y", y).attr("font-size", 12.5)
        .attr("fill", P.text).text(val);
    });
    g.append("text").attr("x", 380).attr("y", 282).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("fill", P.muted)
      .text("every sector runs on machinery this course built by hand — the map is the syllabus, employed");
  }

  for (const b of document.querySelectorAll("#w14-mp-widget [data-s]")) {
    b.addEventListener("click", () => {
      sel = b.dataset.s;
      for (const o of document.querySelectorAll("#w14-mp-widget [data-s]")) {
        o.classList.toggle("active", o === b);
      }
      render();
    });
  }
  U.onThemeChange(render);
  U.lazyBoot("w14-mp-svg", render);
})();
