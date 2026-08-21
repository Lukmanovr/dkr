/* Widget 6.5 — linked equation ↔ dataflow diagram (Distill-style multiple
 * representations). Hover/focus an equation term to isolate the matching diagram
 * part; hover a diagram part to highlight the term. No d3 needed. */
(function () {
  "use strict";
  const root = document.getElementById("w6-eq-widget");
  if (!root) return;
  const svg = document.getElementById("w6-eq-svg");
  const terms = Array.from(root.querySelectorAll(".eq-term"));

  function setHot(link) {
    if (link) {
      svg.classList.add("lnk-focus");
      svg.querySelectorAll(".lnk-part").forEach((g) => {
        // the junction "+" stays lit with either aggregation or self, since both flow into it
        const hot = g.id === "lnk-" + link || (g.id === "lnk-plus" && (link === "agg" || link === "self"));
        g.classList.toggle("lnk-hot", hot);
      });
    } else {
      svg.classList.remove("lnk-focus");
      svg.querySelectorAll(".lnk-part").forEach((g) => g.classList.remove("lnk-hot"));
    }
    terms.forEach((t) => t.classList.toggle("hot", t.getAttribute("data-link") === link));
  }

  terms.forEach((t) => {
    const link = t.getAttribute("data-link");
    t.addEventListener("mouseenter", () => setHot(link));
    t.addEventListener("mouseleave", () => setHot(null));
    t.addEventListener("focus", () => setHot(link));
    t.addEventListener("blur", () => setHot(null));
    t.addEventListener("click", () =>
      setHot(t.classList.contains("hot") ? null : link)); // tap-to-toggle on touch
  });

  ["agg", "self", "w", "relu"].forEach((link) => {
    const g = document.getElementById("lnk-" + link);
    if (!g) return;
    g.style.cursor = "pointer";
    g.addEventListener("mouseenter", () => setHot(link));
    g.addEventListener("mouseleave", () => setHot(null));
  });

  document.addEventListener("keydown", (ev) => { if (ev.key === "Escape") setHot(null); });
})();
