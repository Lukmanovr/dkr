/* Widget 4.2 — The cheat sheet, with reasons.
 * Pick a relation pattern and a model: the verdict, the algebraic requirement,
 * and the two-line derivation. Content mirrors the static cheat-sheet figure
 * (verified against RotatE, ICLR 2019, Table 1) — this widget adds the WHY.
 */
(function () {
  "use strict";
  const U = window.DKR;

  const MODELS = ["TransE", "DistMult", "ComplEx", "RotatE"];
  const MCOL = ["#d9603b", "#7c5cd6", "#0f8377", "#199473"];
  const SCORE = ["−‖h + r − t‖", "Σᵢ hᵢ rᵢ tᵢ", "Re Σᵢ hᵢ rᵢ t̄ᵢ", "−‖h ∘ r − t‖, |rᵢ| = 1"];

  const CELLS = {
    // pattern -> per-model [ok, requirement, derivation]
    symmetry: [
      [false, "needs r = 0", "h+r=t and t+r=h ⇒ adding: 2r=0 ⇒ r=0 — but then every entity equals its own tail: the relation collapses."],
      [true, "automatic (for every r)", "Σ hᵢrᵢtᵢ = Σ tᵢrᵢhᵢ by commutativity — the score cannot tell (h,r,t) from (t,r,h). Symmetry is free… and mandatory."],
      [true, "take r real", "with Im(r)=0 the Hermitian product's swap-antisymmetric part vanishes: score(h,t) = score(t,h). Choosing Im(r)≠0 turns it back off."],
      [true, "phases rᵢ ∈ {0, π}", "rᵢ=±1 ⇒ r∘r = 1: rotating twice returns home, so h∘r=t ⇔ t∘r=h. A half-turn is its own inverse."],
    ],
    antisymmetry: [
      [true, "any r ≠ 0", "h+r=t ⇒ t+r = h+2r ≠ h whenever r≠0 — direction is built into translation."],
      [false, "impossible", "the symmetric score above: if (h,r,t) scores high, (t,r,h) scores identically high. No parameter choice escapes commutativity."],
      [true, "take r with Im(r) ≠ 0", "the imaginary part of r feeds the antisymmetric term Im(h t̄): swapping h,t flips its sign — the figure's +2 / −2."],
      [true, "any phases ∉ {0, π}", "a rotation that is not a half-turn distinguishes forward from back: h∘r=t but t∘r≠h."],
    ],
    inversion: [
      [true, "r₂ = −r₁", "h+r₁=t ⇔ t+(−r₁)=h — walk the translation backwards."],
      [false, "impossible", "DistMult would need score(h,r₂,t)=score(t,r₁,h) for all h,t — but both its relations are already symmetric, so r₂ ≡ r₁ and inversion degenerates to symmetry."],
      [true, "r₂ = r̄₁", "conjugating the relation conjugates the product: Re⟨t,r̄₁,h̄⟩ = Re⟨h,r₁,t̄⟩."],
      [true, "r₂ = r₁⁻¹ (rotate back)", "h∘r₁=t ⇔ t∘r₁⁻¹=h — undo the rotation, angle by angle."],
    ],
    composition: [
      [true, "r₃ = r₁ + r₂", "(h+r₁)+r₂ = h+(r₁+r₂): translations add. Chains of facts become sums of vectors."],
      [false, "impossible", "composition needs an operation on relations that mirrors chaining entities; the diagonal product supplies none (and it already failed inversion)."],
      [false, "not in general", "Hermitian products don't chain: score(h,r₁,m)·score(m,r₂,t) has no r₃ reproducing it for all entities."],
      [true, "r₃ = r₁ ∘ r₂ (add angles)", "rotations compose by adding phases — the only model here whose relations form a group."],
    ],
    "1-to-N": [
      [false, "tails collapse", "h+r=t₁ and h+r=t₂ ⇒ t₁=t₂. One head, one translation, ONE landing point: Kazan and Innopolis get crushed together (watch Figure 1)."],
      [true, "similarity, not equality", "the score asks 'is t aligned with h⊙r?' — many tails can align well simultaneously; nothing forces them equal."],
      [true, "same as DistMult", "a similarity scorer: high scores for several tails are not contradictory."],
      [false, "rotation is a bijection", "h∘r is a single point, exactly like TransE: distance-to-one-point scorers cannot love two tails equally without merging them."],
    ],
  };
  const PATTERNS = Object.keys(CELLS);
  const PDESC = {
    symmetry: "r(a,b) ⇒ r(b,a) — married_to, borders",
    antisymmetry: "r(a,b) ⇒ ¬r(b,a) — located_in, parent_of",
    inversion: "r₂(b,a) ⇔ r₁(a,b) — contains / located_in",
    composition: "r₂(b,c) ∧ r₁(a,b) ⇒ r₃(a,c) — grandparent",
    "1-to-N": "one head, many tails — Tatarstan contains …",
  };

  let pat = "symmetry", model = 0;

  function render() {
    const P = U.pal();
    const svg = U.svgIn("w4-pt-svg", 760, 205);
    svg.attr("font-family", "'Source Sans 3', sans-serif");
    const g = svg.append("g");

    const [ok, req, why] = CELLS[pat][model];
    g.append("text").attr("x", 380).attr("y", 30).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("fill", P.muted).text(PDESC[pat]);

    g.append("text").attr("x", 250).attr("y", 78).attr("text-anchor", "middle")
      .attr("font-size", 17).attr("font-weight", 700).attr("fill", MCOL[model])
      .text(MODELS[model]);
    g.append("text").attr("x", 250).attr("y", 100).attr("text-anchor", "middle")
      .attr("font-size", 13).attr("fill", P.muted).text(SCORE[model]);

    g.append("circle").attr("cx", 480).attr("cy", 82).attr("r", 26)
      .attr("fill", ok ? P.green : "#cf4a30").attr("opacity", 0.9);
    g.append("text").attr("x", 480).attr("y", 92).attr("text-anchor", "middle")
      .attr("font-size", 26).attr("font-weight", 800).attr("fill", "#fff")
      .text(ok ? "✓" : "✗");
    g.append("text").attr("x", 480).attr("y", 126).attr("text-anchor", "middle")
      .attr("font-size", 12.5).attr("font-weight", 700)
      .attr("fill", ok ? P.green : "#cf4a30").text(req);

    // the why, wrapped
    const words = why.split(" ");
    const lines = [""];
    words.forEach((w) => {
      if ((lines[lines.length - 1] + " " + w).length > 88) lines.push(w);
      else lines[lines.length - 1] = (lines[lines.length - 1] + " " + w).trim();
    });
    lines.forEach((ln, i) => {
      g.append("text").attr("x", 380).attr("y", 158 + i * 19).attr("text-anchor", "middle")
        .attr("font-size", 12.5).attr("fill", P.text).text(ln);
    });
  }

  document.querySelectorAll("#w4-pt-widget [data-pat]").forEach((b) =>
    b.addEventListener("click", () => {
      pat = b.getAttribute("data-pat");
      document.querySelectorAll("#w4-pt-widget [data-pat]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));
  document.querySelectorAll("#w4-pt-widget [data-model]").forEach((b) =>
    b.addEventListener("click", () => {
      model = +b.getAttribute("data-model");
      document.querySelectorAll("#w4-pt-widget [data-model]").forEach((x) => x.classList.toggle("active", x === b));
      render();
    }));

  U.onThemeChange(render);
  U.lazyBoot("w4-pt-svg", render);
})();
