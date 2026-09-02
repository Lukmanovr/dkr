// Widget smoke test: mount every lecture's widget HTML fragments in jsdom,
// evaluate their scripts in order, poke the controls, and assert per-week
// CORRECTNESS probes (the measured numbers each widget must display).
// Any thrown error or failed probe fails the run.
//
//     npm install            # once, in scripts/
//     node scripts/widget_smoke.mjs     # from the repo root OR anywhere
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { JSDOM } from "jsdom";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const A = join(ROOT, "assets", "d3") + "/";
const strip = (f) =>
  readFileSync(A + f, "utf8").replace(/^```\{=html\}\r?\n/, "").replace(/\r?\n```\r?\n?$/, "");

const body = ["w1-hero.html", "w1-permute.html", "w1-multigraph.html", "w1-konigsberg.html", "w1-closure.html", "w1-walks.html", "w1-lap.html", "w6-message-passing.html", "w6-spectral.html", "w6-normalization.html", "w6-permutation.html", "w6-eq-linked.html",
  "w1-builder.html", "w1-cost.html", "w1-types.html", "w1-tasks.html",
  "w2-centrality.html", "w2-pagerank.html", "w2-wl.html", "w2-louvain.html", "w2-katz.html", "w2-surfer.html", "w2-poweriter.html", "w2-clustering.html", "w2-graphlets.html", "w2-nullmodel.html", "w2-resolution.html", "w2-spectral.html", "w2-baseline.html",
  "w3-walks.html", "w3-pq.html", "w3-embed.html", "w3-labelprop.html",
  "w4-transe.html", "w4-patterns.html", "w4-negatives.html", "w4-rank.html",
  "w5-query.html", "w5-boxes.html", "w5-rag.html", "w5-extract.html",
  "w7-agg.html", "w7-sage.html", "w7-gat.html", "w7-ablation.html",
  "w9-wl.html", "w9-trees.html", "w9-power.html", "w9-squash.html",
  "w10-typed.html", "w10-params.html", "w10-metapath.html", "w10-showdown.html",
  "w11-explosion.html", "w11-sampler.html", "w11-sgc.html", "w11-table.html",
  "w12-split.html", "w12-heuristic.html", "w12-vgae.html", "w12-generate.html",
  "w13-attn.html", "w13-rwse.html", "w13-bias.html", "w13-results.html",
  "w14-propagate.html", "w14-metrics.html", "w14-splits.html", "w14-map.html"]
  .map(strip).join("\n")
  + '\n<span class="term" data-def="test def">term</span>'
  + '\n<div class="predict"><p>Q?</p><div class="predict-answer">A.</div></div>';

const dom = new JSDOM(`<!doctype html><html><body>${body}</body></html>`, {
  runScripts: "outside-only", pretendToBeVisual: true,
});
const { window } = dom;

let failures = 0;
const run = (name, src) => {
  try {
    window.eval(src);
    console.log(`  ${name} loaded ok`);
  } catch (e) {
    failures++;
    console.error(`  FAIL ${name}: ${e.message}\n${(e.stack || "").split("\n").slice(0, 4).join("\n")}`);
  }
};

for (const f of ["d3.v7.min.js", "_dkr.js", "w6-message-passing.js", "w6-spectral.js", "w6-normalization.js", "w6-permutation.js", "w6-eq-linked.js",
  "w1-hero.js", "w1-permute.js", "w1-multigraph.js", "w1-konigsberg.js", "w1-closure.js", "w1-walks.js", "w1-lap.js", "w1-builder.js", "w1-cost.js", "w1-types.js", "w1-tasks.js",
  "w2-centrality.js", "w2-pagerank.js", "w2-wl.js", "w2-louvain.js", "w2-katz.js", "w2-surfer.js", "w2-poweriter.js", "w2-clustering.js", "w2-graphlets.js", "w2-nullmodel.js", "w2-resolution.js", "w2-spectral.js", "w2-baseline.js",
  "w3-walks.js", "w3-pq.js", "w3-embed.js", "w3-labelprop.js",
  "w4-transe.js", "w4-patterns.js", "w4-negatives.js", "w4-rank.js",
  "w5-query.js", "w5-boxes.js", "w5-rag.js", "w5-extract.js",
  "w7-agg.js", "w7-sage.js", "w7-gat.js", "w7-ablation.js",
  "w9-wl.js", "w9-trees.js", "w9-power.js", "w9-squash.js",
  "w10-typed.js", "w10-params.js", "w10-metapath.js", "w10-showdown.js",
  "w11-explosion.js", "w11-sampler.js", "w11-sgc.js", "w11-table.js",
  "w12-vgae-data.js", "w12-gen-data.js",
  "w12-split.js", "w12-heuristic.js", "w12-vgae.js", "w12-generate.js",
  "w13-attn.js", "w13-rwse.js", "w13-bias.js", "w13-results.js",
  "w14-propagate.js", "w14-metrics.js", "w14-splits.js", "w14-map.js"]) {
  run(f, readFileSync(A + f, "utf8"));
}
// page runtime (strip the <script> wrapper)
const pageFoot = readFileSync(join(ROOT, "assets", "page-foot.html"), "utf8")
  .replace(/^\s*<script>/, "").replace(/<\/script>\s*$/, "");
run("page-foot.html", pageFoot);

// exercise the new interactions
const term = window.document.querySelector(".term");
term.dispatchEvent(new window.Event("mouseenter", { bubbles: true }));
if (!window.document.querySelector(".dkr-tooltip.show")) { failures++; console.error("  FAIL tooltip did not show"); }
else console.log("  glossary tooltip shows ✓");
term.dispatchEvent(new window.Event("mouseleave", { bubbles: true }));

const pbtn = window.document.querySelector(".predict .predict-btn");
if (!pbtn) { failures++; console.error("  FAIL predict button not injected"); }
else {
  pbtn.dispatchEvent(new window.Event("click", { bubbles: true }));
  if (!window.document.querySelector(".predict.revealed")) { failures++; console.error("  FAIL predict did not reveal"); }
  else console.log("  predict block reveals ✓");
}

const eqTerm = window.document.querySelector('#w6-eq-widget .eq-term[data-link="agg"]');
eqTerm.dispatchEvent(new window.Event("mouseenter", { bubbles: true }));
const hotAgg = window.document.querySelector("#lnk-agg.lnk-hot");
const hotPlus = window.document.querySelector("#lnk-plus.lnk-hot");
if (!hotAgg || !hotPlus) { failures++; console.error("  FAIL eq-linked highlight (agg/plus)"); }
else console.log("  eq-linked hover isolates aggregation path ✓");
eqTerm.dispatchEvent(new window.Event("mouseleave", { bubbles: true }));
if (window.document.querySelector(".lnk-hot")) { failures++; console.error("  FAIL eq-linked did not clear"); }
else console.log("  eq-linked clears ✓");

// poke the controls
const click = (id) => {
  const el = window.document.getElementById(id);
  if (!el) { failures++; console.error(`  FAIL missing #${id}`); return; }
  try { el.dispatchEvent(new window.Event("click", { bubbles: true })); }
  catch (e) { failures++; console.error(`  FAIL clicking #${id}: ${e.message}`); }
};
const slide = (id, v) => {
  const el = window.document.getElementById(id);
  if (!el) { failures++; console.error(`  FAIL missing #${id}`); return; }
  try { el.value = String(v); el.dispatchEvent(new window.Event("input", { bubbles: true })); }
  catch (e) { failures++; console.error(`  FAIL sliding #${id}: ${e.message}`); }
};

for (let i = 0; i < 3; i++) click("w6-mp-step");
click("w6-mp-reset");
slide("w6-sp-k", 5);
for (const b of window.document.querySelectorAll("#w6-sp-widget [data-mode]")) b.dispatchEvent(new window.Event("click", { bubbles: true }));
for (const b of window.document.querySelectorAll("#w6-sp-widget [data-graph]")) b.dispatchEvent(new window.Event("click", { bubbles: true }));
slide("w6-sp-w1", 1.5);
for (let i = 0; i < 5; i++) click("w6-nm-step");
click("w6-nm-reset");
for (let i = 0; i < 4; i++) click("w6-pm-shuffle");
click("w6-pm-reset");
// week-1 widget interactions
for (const b of window.document.querySelectorAll("#w1-bd-widget [data-tab]")) b.dispatchEvent(new window.Event("click", { bubbles: true }));
click("w1-bd-reset");
slide("w1-ct-n", 6.4); slide("w1-ct-d", 1.7);
for (const b of window.document.querySelectorAll("#w1-ct-widget [data-preset]")) b.dispatchEvent(new window.Event("click", { bubbles: true }));
for (const b of window.document.querySelectorAll("#w1-ty-widget [data-mode]")) b.dispatchEvent(new window.Event("click", { bubbles: true }));
for (const b of window.document.querySelectorAll("#w1-tk-widget [data-mode]")) b.dispatchEvent(new window.Event("click", { bubbles: true }));
// week-2 widget interactions
for (const b of window.document.querySelectorAll("#w2-ce-widget [data-measure]")) b.dispatchEvent(new window.Event("click", { bubbles: true }));
// toggle an edge in the kite: click Heather (idx 7) then Ike (idx 8)
{
  const nodeGroups = window.document.querySelectorAll("#w2-ce-svg svg g g");
  console.log("  (kite has interactive node groups: " + nodeGroups.length + ")");
}
click("w2-ce-reset");
// week-2 PageRank probes: one step from uniform, the beta = 0.85 fixed point, the
// beta = 0 degenerate case, and the personalized fixed point, all measured.
{
  const T = () => [...window.document.querySelectorAll("#w2-pr-svg svg text")].map((t) => t.textContent).join(" | ");
  const want = (label, needles) => {
    const t = T();
    const miss = needles.filter((n) => !t.includes(n));
    if (miss.length) { failures++; console.error(`  FAIL w2-pagerank ${label}: missing ${JSON.stringify(miss)}: ` + t.slice(0, 400)); }
    else console.log(`  w2-pagerank: ${label} ✓`);
  };
  click("w2-pr-reset");
  want("initial state names the update and the bound", ["everyone starts equal at 1/10 = 0.100 — press step", "t = 1/10 everywhere (uniform)", "expect ≈ 43 steps"]);
  click("w2-pr-step");
  want("one step puts 0.398 on the hub", ["after 1 step, ranked | hub | 0.398", "total change |Δr|₁ = 1.0200"]);
  for (let k = 0; k < 5; k++) click("w2-pr-step10");
  want("51 steps reach hub 0.416, Q 0.133, V 0.060 and converge at step 44", ["hub | 0.416", "Q | 0.133", "V | 0.060", "converged (for the eye)", "first below 10⁻³ at step 44"]);
  slide("w2-pr-beta", 0); click("w2-pr-step");
  want("beta = 0 makes every page 0.100", ["β = 0: no link is ever followed", "0.100"]);
  slide("w2-pr-beta", 0.85);
  const tele = window.document.getElementById("w2-pr-tele");
  tele.checked = true; tele.dispatchEvent(new window.Event("change", { bubbles: true }));
  want("personalized teleport is announced and resets", ["t = all on Q (personalized)", "every teleport lands here", "after 0 steps"]);
  for (let k = 0; k < 6; k++) click("w2-pr-step10");
  want("personalized fixed point: Q 0.280, hub 0.459", ["Q | 0.280", "hub | 0.459", "converged (for the eye)"]);
  tele.checked = false; tele.dispatchEvent(new window.Event("change", { bubbles: true }));
  click("w2-pr-reset");
}
for (let i = 0; i < 4; i++) click("w2-wl-step");   // cast: refine to stable + one extra
const wlStatus1 = window.document.querySelector("#w2-wl-svg svg text:last-of-type");
for (const b of window.document.querySelectorAll("#w2-wl-widget [data-scene]")) b.dispatchEvent(new window.Event("click", { bubbles: true }));
click("w2-wl-step"); click("w2-wl-step");           // rings: stable immediately
click("w2-wl-reset");
for (let i = 0; i < 5; i++) click("w2-lv-step");
click("w2-lv-run");
const lvMsg = [...window.document.querySelectorAll("#w2-lv-svg svg text")].map((t) => t.textContent).join(" | ");
if (!/phase-1 optimum: .*Q = 0\.327/.test(lvMsg)) { failures++; console.error("  FAIL louvain phase-1 did not stall at Q=0.327: " + lvMsg.slice(-160)); }
else console.log("  louvain phase 1 stalls at Q = 0.327 as the caption promises ✓");
click("w2-lv-phase2");
const lvMsg2 = [...window.document.querySelectorAll("#w2-lv-svg svg text")].map((t) => t.textContent).join(" | ");
if (!/after phase 2 .*2 communities · Q = 0\.389/.test(lvMsg2)) { failures++; console.error("  FAIL louvain phase 2 did not reach planted split Q=0.389: " + lvMsg2.slice(-160)); }
else console.log("  louvain phase 2 reaches the planted split, Q = 0.389 ✓");
click("w2-lv-reset");
// week-3 widget interactions
for (let i = 0; i < 9; i++) click("w3-wk-step");            // one full walk + one step
click("w3-wk-run");
const wkTexts = [...window.document.querySelectorAll("#w3-wk-svg svg text")].map((t) => t.textContent).join(" | ");
if (!/walks absorbed so far: 101/.test(wkTexts)) { failures++; console.error("  FAIL walks counter wrong: " + wkTexts.slice(0, 160)); }
else console.log("  w3-walks absorbs 101 walks (1 stepped + 100 batch) ✓");
click("w3-wk-reset");
slide("w3-pq-p", 0); slide("w3-pq-q", 4);
click("w3-pq-go");
const pqTexts = [...window.document.querySelectorAll("#w3-pq-svg svg text")].map((t) => t.textContent).join(" | ");
const pqBFS = /orbiting the start/.test(pqTexts);
slide("w3-pq-p", 4); slide("w3-pq-q", 0);
click("w3-pq-go");
const pqTexts2 = [...window.document.querySelectorAll("#w3-pq-svg svg text")].map((t) => t.textContent).join(" | ");
const pqDFS = /escaping down the tail/.test(pqTexts2);
if (!pqBFS || !pqDFS) { failures++; console.error(`  FAIL pq regimes (BFS=${pqBFS}, DFS=${pqDFS}): ` + pqTexts2.slice(-160)); }
else console.log("  w3-pq: low-q escapes, high-q orbits ✓");
for (let i = 0; i < 8; i++) click("w3-em-train");           // 4000 pairs
const emTexts = [...window.document.querySelectorAll("#w3-em-svg svg text")].map((t) => t.textContent).join(" | ");
const emSep = (emTexts.match(/separation: (\d+)\/34/) || [])[1];
if (!emSep || +emSep < 28) { failures++; console.error("  FAIL embed separation after 4000 pairs: " + emTexts.slice(0, 160)); }
else console.log(`  w3-embed separates ${emSep}/34 after 4,000 pairs ✓`);
click("w3-em-auto"); click("w3-em-auto");                   // toggle on/off
click("w3-em-reset");
click("w3-lp-step");
click("w3-lp-run");
const lpTexts = [...window.document.querySelectorAll("#w3-lp-svg svg text")].map((t) => t.textContent).join(" | ");
const lpM = lpTexts.match(/stable after (\d+) sweeps · (\d+)\/34 labeled · (\d+)\/34 match/);
if (!lpM || +lpM[3] < 26) { failures++; console.error("  FAIL labelprop outcome: " + lpTexts.slice(0, 200)); }
else console.log(`  w3-labelprop stable after ${lpM[1]} sweeps, ${lpM[2]}/34 labeled, ${lpM[3]}/34 correct ✓`);
click("w3-lp-reset");
// week-4 widget interactions
for (let i = 0; i < 12; i++) click("w4-te-train");          // 3,600 SGD steps
const teTexts = [...window.document.querySelectorAll("#w4-te-svg svg text")].map((t) => t.textContent).join(" | ");
const teRes = parseFloat((teTexts.match(/mean residual ‖h\+r−t‖ = ([\d.]+)/) || [])[1]);
if (!(teRes < 0.6)) { failures++; console.error("  FAIL transe mean residual after 3600 steps: " + teRes); }
else console.log(`  w4-transe trains: mean residual ${teRes} after 3,600 steps ✓`);
const teCrush = parseFloat((teTexts.match(/\|Innopolis − Kazan\| = ([\d.]+)/) || [])[1]);
const teHero = teTexts.match(/hero fact [^:]+: ([\d.]+) · nonsense [^:]+: ([\d.]+)/);
if (!teHero || !(parseFloat(teHero[1]) < parseFloat(teHero[2]))) {
  failures++; console.error("  FAIL transe hero-vs-nonsense ordering: " + (teHero || teTexts.slice(-160)));
} else console.log(`  w4-transe: crush |Inno−Kazan| = ${teCrush}; hero ${teHero[1]} beats nonsense ${teHero[2]} ✓`);
click("w4-te-auto"); click("w4-te-auto");
click("w4-te-reset");
for (const b of window.document.querySelectorAll("#w4-pt-widget [data-pat]")) b.dispatchEvent(new window.Event("click", { bubbles: true }));
for (const b of window.document.querySelectorAll("#w4-pt-widget [data-model]")) b.dispatchEvent(new window.Event("click", { bubbles: true }));
const ptTexts = [...window.document.querySelectorAll("#w4-pt-svg svg text")].map((t) => t.textContent).join(" ");
if (!/bijection/.test(ptTexts)) { failures++; console.error("  FAIL patterns final cell (RotatE × 1-to-N): " + ptTexts.slice(0, 120)); }
else console.log("  w4-patterns walks all 20 cells, ends on RotatE × 1-to-N ✓");
slide("w4-ng-alpha", 3);
for (const b of window.document.querySelectorAll("#w4-ng-widget [data-q]")) b.dispatchEvent(new window.Event("click", { bubbles: true }));
const ngTexts = [...window.document.querySelectorAll("#w4-ng-svg svg text")].map((t) => t.textContent).join(" | ");
const ngTop = (ngTexts.match(/now takes (\d+)% of the gradient/) || [])[1];
if (!ngTop || +ngTop < 40) { failures++; console.error("  FAIL negatives concentration at α=3: " + ngTexts.slice(-140)); }
else console.log(`  w4-negatives: hardest negative takes ${ngTop}% at α = 3 ✓`);
const warns = [...window.document.querySelectorAll("#w4-ng-svg svg text")].filter((t) => /⚠/.test(t.textContent));
if (!warns.length) { failures++; console.error("  FAIL negatives: accidental-true ⚠ candidate missing on third triple"); }
else console.log("  w4-negatives: accidentally-true corruption flagged ⚠ ✓");
const rkFilter = window.document.getElementById("w4-rk-filter");
const rkRead = () => [...window.document.querySelectorAll("#w4-rk-svg svg text")].map((t) => t.textContent).join(" | ");
click("w4-rk-next"); click("w4-rk-next");
const rkM = rkRead().match(/raw: MRR ([\d.]+), Hits@1 ([\d.]+) · filtered: MRR ([\d.]+), Hits@1 ([\d.]+)/);
if (!rkM || !(parseFloat(rkM[3]) >= parseFloat(rkM[1]))) { failures++; console.error("  FAIL rank metrics: " + rkRead().slice(-200)); }
else console.log(`  w4-rank: MRR raw ${rkM[1]} → filtered ${rkM[3]}, Hits@1 ${rkM[2]} → ${rkM[4]} ✓`);
rkFilter.checked = true; rkFilter.dispatchEvent(new window.Event("change", { bubbles: true }));
// week-5 widget interactions
const qyText = () => [...window.document.querySelectorAll("#w5-qy-svg svg text")].map((t) => t.textContent).join(" | ");
click("w5-qy-hop"); click("w5-qy-hop");
const trav = qyText();
if (!/answer set: \{Aspirin, Ibuprofen\}/.test(trav)) { failures++; console.error("  FAIL w5-query traversal answers: " + trav.slice(-200)); }
else console.log("  w5-query traversal finds exactly {Aspirin, Ibuprofen} ✓");
for (const b of window.document.querySelectorAll("#w5-qy-widget [data-mode]")) {
  if (b.getAttribute("data-mode") === "soft") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
const soft = qyText();
if (!/Naproxen recovered by the model/.test(soft)) { failures++; console.error("  FAIL w5-query soft mode did not recover Naproxen: " + soft.slice(-220)); }
else console.log("  w5-query soft completion recovers Naproxen (DistMult, computed) ✓");
click("w5-qy-reset");
for (const b of window.document.querySelectorAll("#w5-bx-widget [data-scene]")) {
  if (b.getAttribute("data-scene") === "intersect") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
click("w5-bx-step"); click("w5-bx-step"); click("w5-bx-step");
const bxText = [...window.document.querySelectorAll("#w5-bx-svg svg text")].map((t) => t.textContent).join(" | ");
if (!/closed under intersection/i.test(bxText)) { failures++; console.error("  FAIL w5-boxes intersect scene: " + bxText.slice(-160)); }
else console.log("  w5-boxes intersection scene completes ✓");
for (const b of window.document.querySelectorAll("#w5-bx-widget [data-scene]")) b.dispatchEvent(new window.Event("click", { bubbles: true }));
click("w5-bx-step");
const rgText = () => [...window.document.querySelectorAll("#w5-rg-svg svg text")].map((t) => t.textContent).join(" | ");
const rgV = rgText();
if (!/coverage: 1\/2/.test(rgV)) { failures++; console.error("  FAIL w5-rag vector coverage on 2-hop: " + rgV.slice(-160)); }
else console.log("  w5-rag: vector RAG covers 1/2 on the 2-hop question ✓");
for (const b of window.document.querySelectorAll("#w5-rg-widget [data-mode]")) {
  if (b.getAttribute("data-mode") === "graph") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
if (!/coverage: 2\/2/.test(rgText())) { failures++; console.error("  FAIL w5-rag graph coverage on 2-hop: " + rgText().slice(-160)); }
else console.log("  w5-rag: GraphRAG covers 2/2 on the 2-hop question ✓");
for (const b of window.document.querySelectorAll("#w5-rg-widget [data-q]")) {
  if (b.getAttribute("data-q") === "2") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
if (!/coverage: 0\/1/.test(rgText())) { failures++; console.error("  FAIL w5-rag fuzzy question should beat graph mode: " + rgText().slice(-160)); }
else console.log("  w5-rag: fuzzy question defeats entity linking (vector's turn) ✓");
for (const id of ["w5-ex-canon", "w5-ex-schema", "w5-ex-prov"]) {
  const cb = window.document.getElementById(id);
  cb.checked = true; cb.dispatchEvent(new window.Event("change", { bubbles: true }));
}
const exText = [...window.document.querySelectorAll("#w5-ex-svg svg text")].map((t) => t.textContent).join(" | ");
if (!/clean: 3 facts, 5 entities/.test(exText)) { failures++; console.error("  FAIL w5-extract clean state: " + exText.slice(-160)); }
else console.log("  w5-extract: all repairs on → 3 facts, 5 entities ✓");
// week-7 widget interactions
for (const b of window.document.querySelectorAll("#w7-ag-widget [data-preset]")) b.dispatchEvent(new window.Event("click", { bubbles: true }));
const agText = [...window.document.querySelectorAll("#w7-ag-svg svg text")].map((t) => t.textContent).join(" | ");
if (!/✓ mean/.test(agText) || !/✗ max/.test(agText) || !/✓ sum/.test(agText)) { failures++; console.error("  FAIL w7-agg case 3 verdicts: " + agText.slice(0, 200)); }
else console.log("  w7-agg: case 3 shows mean ✓, max ✗, sum ✓ — computed verdicts render ✓");
for (const b of window.document.querySelectorAll("#w7-sg-widget [data-k]")) {
  if (b.getAttribute("data-k") === "2") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
for (let i = 0; i < 20; i++) click("w7-sg-resample");
const sgText = [...window.document.querySelectorAll("#w7-sg-svg svg text")].map((t) => t.textContent).join(" | ");
const sgSpread = parseFloat((sgText.match(/spread \(std\): ([\d.]+)/) || [])[1]);
if (!(sgSpread > 0)) { failures++; console.error("  FAIL w7-sage spread not positive: " + sgText.slice(-160)); }
else console.log(`  w7-sage: 21 resamples at K=2 on node 33, spread ${sgSpread} > 0 ✓`);
click("w7-sg-center");
const gtText = () => [...window.document.querySelectorAll("#w7-gt-svg svg text")].map((t) => t.textContent).join(" | ");
const gtM = gtText().match(/attends to own key: (\d)\/5.*attends to own key: (\d)\/5/);
if (!gtM || +gtM[1] > 2 || +gtM[2] < 4) { failures++; console.error("  FAIL w7-gat static-vs-dynamic: " + (gtM || gtText().slice(0, 160))); }
else console.log(`  w7-gat: GAT ${gtM[1]}/5 (static, capped) vs GATv2 ${gtM[2]}/5 (dynamic) ✓`);
click("w7-gt-retrain");
const gtM2 = gtText().match(/attends to own key: (\d)\/5.*attends to own key: (\d)\/5/);
if (!gtM2 || +gtM2[1] > 2 || +gtM2[2] < 4) { failures++; console.error("  FAIL w7-gat after retrain: " + gtM2); }
else console.log("  w7-gat: verdict survives a reseed ✓");
click("w7-gt-reset");
for (const b of window.document.querySelectorAll("#w7-ab-widget [data-arch]")) {
  if (b.getAttribute("data-arch") === "GAT") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
for (const b of window.document.querySelectorAll("#w7-ab-widget [data-depth]")) {
  if (b.getAttribute("data-depth") === "4") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
const abText = () => [...window.document.querySelectorAll("#w7-ab-svg svg text")].map((t) => t.textContent).join(" | ");
if (!/26\.2%/.test(abText())) { failures++; console.error("  FAIL w7-ablation GAT-4-plain cell: " + abText().slice(0, 160)); }
else console.log("  w7-ablation: GAT·4·plain shows the measured 26.2% crash ✓");
const abRes = window.document.getElementById("w7-ab-res");
abRes.checked = true; abRes.dispatchEvent(new window.Event("change", { bubbles: true }));
if (!/80\.5%/.test(abText())) { failures++; console.error("  FAIL w7-ablation residual rescue: " + abText().slice(0, 160)); }
else console.log("  w7-ablation: residual box rescues to the measured 80.5% ✓");
console.log("  control interactions dispatched");

// week-9 correctness probes
const w9Text = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
for (let i = 0; i < 4; i++) click("w9-wl-step");
if (!/cannot distinguish/.test(w9Text("w9-wl-svg"))) { failures++; console.error("  FAIL w9-wl blind pair: " + w9Text("w9-wl-svg").slice(-160)); }
else console.log("  w9-wl: triangles vs hexagon stays blind ✓");
for (const b of window.document.querySelectorAll("#w9-wl-widget [data-pair]")) {
  if (b.getAttribute("data-pair") === "2") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
click("w9-wl-step"); click("w9-wl-step");
if (!/differ at round 2/.test(w9Text("w9-wl-svg"))) { failures++; console.error("  FAIL w9-wl P43/P52 split: " + w9Text("w9-wl-svg").slice(-160)); }
else console.log("  w9-wl: same-degree-sequence pair splits at round 2 ✓");

if (!/SAME embedding/.test(w9Text("w9-tr-svg"))) { failures++; console.error("  FAIL w9-trees default identical: " + w9Text("w9-tr-svg").slice(-160)); }
else console.log("  w9-trees: triangle vs hexagon trees identical ✓");
for (const b of window.document.querySelectorAll("#w9-tr-widget [data-preset]")) {
  if (b.getAttribute("data-preset") === "1") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
if (!/CAN separate/.test(w9Text("w9-tr-svg"))) { failures++; console.error("  FAIL w9-trees P3/P4 depth-2 differ: " + w9Text("w9-tr-svg").slice(-160)); }
else console.log("  w9-trees: P3-mid vs P4-2nd differ at depth 2 ✓");
for (const b of window.document.querySelectorAll("#w9-tr-widget [data-depth]")) {
  if (b.getAttribute("data-depth") === "1") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
if (!/SAME embedding/.test(w9Text("w9-tr-svg"))) { failures++; console.error("  FAIL w9-trees P3/P4 depth-1 identical: " + w9Text("w9-tr-svg").slice(-160)); }
else console.log("  w9-trees: same pair identical at depth 1 ✓");

for (const b of window.document.querySelectorAll("#w9-pw-widget [data-feature]")) {
  if (b.getAttribute("data-feature") === "triangles") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
let pw = w9Text("w9-pw-svg");
if (!/blindness is cured/.test(pw) || !/correctly so/.test(pw)) { failures++; console.error("  FAIL w9-power triangles: " + pw.slice(0, 200)); }
else console.log("  w9-power: triangle counts separate A, spare B ✓");
for (const b of window.document.querySelectorAll("#w9-pw-widget [data-feature]")) {
  if (b.getAttribute("data-feature") === "random") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
pw = w9Text("w9-pw-svg");
if (!/FALSE ALARM/.test(pw)) { failures++; console.error("  FAIL w9-power random false alarm: " + pw.slice(0, 200)); }
else console.log("  w9-power: random IDs trigger the false alarm ✓");

let sq = w9Text("w9-sq-svg");
if (!/ratio 290/.test(sq)) { failures++; console.error("  FAIL w9-squash K=5 ratio: " + sq.slice(-200)); }
else console.log("  w9-squash: K=5 shows the 290x crush ✓");
click("w9-sq-rescue");
sq = w9Text("w9-sq-svg");
if (!/re-routes the flow/.test(sq) || /ratio 290/.test(sq)) { failures++; console.error("  FAIL w9-squash rescue: " + sq.slice(-200)); }
else console.log("  w9-squash: one rescue edge changes the story ✓");
for (const b of window.document.querySelectorAll("#w9-sq-widget [data-k]")) {
  if (b.getAttribute("data-k") === "2") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
sq = w9Text("w9-sq-svg");
if (!/EXACTLY 0/.test(sq)) { failures++; console.error("  FAIL w9-squash unreachable-vs-squashed: " + sq.slice(-200)); }
else console.log("  w9-squash: K=2 labels the far clique unreachable, not squashed ✓");

// week-10 correctness probes
const w10Text = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
if (!/5\.25/.test(w10Text("w10-ty-svg"))) { failures++; console.error("  FAIL w10-typed default 5.25: " + w10Text("w10-ty-svg").slice(-200)); }
else console.log("  w10-typed: default typed update = 5.25 ✓");
for (const b of window.document.querySelectorAll("#w10-ty-widget [data-wc]")) {
  if (b.getAttribute("data-wc") === "0") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
if (!/2\.25/.test(w10Text("w10-ty-svg"))) { failures++; console.error("  FAIL w10-typed muted cites 2.25: " + w10Text("w10-ty-svg").slice(-200)); }
else console.log("  w10-typed: muting cites drops the update to 2.25 ✓");
click("w10-ty-untyped");
if (!/2\.00/.test(w10Text("w10-ty-svg"))) { failures++; console.error("  FAIL w10-typed untyped 2.00: " + w10Text("w10-ty-svg").slice(-200)); }
else console.log("  w10-typed: collapsed types give 2.00 ✓");

let pm = w10Text("w10-pm-svg");
if (!/4,750,000/.test(pm) || !/324,220/.test(pm) || !/14\.7/.test(pm)) { failures++; console.error("  FAIL w10-params defaults: " + pm.slice(0, 240)); }
else console.log("  w10-params: 4,750,000 vs 324,220 (14.7x) at FB15k-237 defaults ✓");

let mp = w10Text("w10-mp-svg");
if (!/{ a2 }/.test(mp)) { failures++; console.error("  FAIL w10-metapath APA(a1): " + mp.slice(-200)); }
else console.log("  w10-metapath: a1 co-authors = {a2} ✓");
for (const b of window.document.querySelectorAll("#w10-mp-widget [data-mp]")) {
  if (b.getAttribute("data-mp") === "apvpa") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
mp = w10Text("w10-mp-svg");
if (!/{ a2, a3 }/.test(mp)) { failures++; console.error("  FAIL w10-metapath APVPA(a1): " + mp.slice(-200)); }
else console.log("  w10-metapath: a1 same-venue = {a2, a3} ✓");
for (const b of window.document.querySelectorAll("#w10-mp-widget [data-mp]")) {
  if (b.getAttribute("data-mp") === "cite") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
mp = w10Text("w10-mp-svg");
if (!/{ a4 }/.test(mp)) { failures++; console.error("  FAIL w10-metapath cited-by(a1): " + mp.slice(-200)); }
else console.log("  w10-metapath: a1 cited-by authors = {a4} ✓");

let sd = w10Text("w10-sd-svg");
if (!/0\.179/.test(sd) || !/0\.126/.test(sd) || !/0\.145/.test(sd)) { failures++; console.error("  FAIL w10-showdown budget rows: " + sd.slice(0, 240)); }
else console.log("  w10-showdown: measured 0.179 / 0.126 / 0.145 present ✓");
for (const b of window.document.querySelectorAll("#w10-sd-widget [data-view]")) {
  if (b.getAttribute("data-view") === "cats") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
sd = w10Text("w10-sd-svg");
if (!/0\.313/.test(sd) || !/N-N/.test(sd)) { failures++; console.error("  FAIL w10-showdown categories: " + sd.slice(0, 240)); }
else console.log("  w10-showdown: category lens shows N-1 at 0.313 ✓");

// week-11 correctness probes
const w11Text = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
let ex = w11Text("w11-ex-svg");
if (!/22,663/.test(ex) || !/13%/.test(ex)) { failures++; console.error("  FAIL w11-explosion arxiv overlay: " + ex.slice(-200)); }
else console.log("  w11-explosion: measured 22,663 / 13% overlay at defaults \u2713");
for (const b of window.document.querySelectorAll("#w11-ex-widget [data-f]")) {
  if (b.getAttribute("data-f") === "10") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
ex = w11Text("w11-ex-svg");
if (!/1,111/.test(ex)) { failures++; console.error("  FAIL w11-explosion fanout bound: " + ex.slice(0, 200)); }
else console.log("  w11-explosion: fanout-10 bound 1,111 at L=3 \u2713");

let sm = w11Text("w11-sm-svg");
if (!/34\/34 nodes .* 78\/78 edges/.test(sm)) { failures++; console.error("  FAIL w11-sampler full: " + sm.slice(-160)); }
else console.log("  w11-sampler: full-batch loads 34/34 and 78/78 \u2713");
for (const b of window.document.querySelectorAll("#w11-sm-widget [data-mode]")) {
  if (b.getAttribute("data-mode") === "sample") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
sm = w11Text("w11-sm-svg");
const mload = sm.match(/loaded: (\d+)\/34/);
if (!mload || +mload[1] >= 34) { failures++; console.error("  FAIL w11-sampler bounded: " + sm.slice(-160)); }
else console.log(`  w11-sampler: fanout-3 batch loads ${mload[1]}/34 \u2713`);

let sg = w11Text("w11-sg2-svg");
if (!/34\/34|3[0-4]\/34/.test(sg)) { failures++; console.error("  FAIL w11-sgc render: " + sg.slice(0, 200)); }
const kmatch = sg.match(/sign matches true faction: (\d+)\/34/);
for (const b of window.document.querySelectorAll("#w11-sg2-widget [data-k]")) {
  if (b.getAttribute("data-k") === "0") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
const k0 = w11Text("w11-sg2-svg").match(/sign matches true faction: (\d+)\/34/);
if (!kmatch || !k0 || +kmatch[1] <= +k0[1]) { failures++; console.error(`  FAIL w11-sgc repair: K2=${kmatch && kmatch[1]} K0=${k0 && k0[1]}`); }
else console.log(`  w11-sgc: propagation repairs errors (${k0[1]}/34 raw -> ${kmatch[1]}/34 at K=2) \u2713`);

let tb = w11Text("w11-tb-svg");
if (!/0\.665/.test(tb) || !/2,821 MB|2.8 GB/.test(tb)) { failures++; console.error("  FAIL w11-table rows: " + tb.slice(0, 240)); }
else console.log("  w11-table: measured 0.665 / 2,821 MB present \u2713");
click("w11-tb-products");
tb = w11Text("w11-tb-svg");
if (!/OOM/.test(tb) || !/fits/.test(tb)) { failures++; console.error("  FAIL w11-table stress: " + tb.slice(0, 240)); }
else console.log("  w11-table: products stress marks OOM and fits \u2713");

// week-12 correctness probes
const w12Text = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
let sp12 = w12Text("w12-sp-svg");
if (!/no test edges/.test(sp12)) { failures++; console.error("  FAIL w12-split initial audit: " + sp12.slice(-160)); }
else console.log("  w12-split: all-message start flagged (no test edges) \u2713");
click("w12-sp-auto");
sp12 = w12Text("w12-sp-svg");
if (!/split legal/.test(sp12)) { failures++; console.error("  FAIL w12-split legal deal: " + sp12.slice(-160)); }
else console.log("  w12-split: dealt split passes the auditor \u2713");

let he = w12Text("w12-he-svg");
const hitm = he.match(/hits: (\d+)\/10/);
if (!hitm || +hitm[1] < 2) { failures++; console.error("  FAIL w12-heuristic CN hits: " + he.slice(-160)); }
else console.log(`  w12-heuristic: CN finds ${hitm[1]}/10 hidden edges \u2713`);
for (const b of window.document.querySelectorAll("#w12-he-widget [data-h]")) {
  if (b.getAttribute("data-h") === "pa") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
const he2 = w12Text("w12-he-svg").match(/hits: (\d+)\/10/);
if (!he2 || +he2[1] > +hitm[1]) { failures++; console.error("  FAIL w12-heuristic PA should be weaker: " + he2); }
else console.log(`  w12-heuristic: pref. attachment weaker (${he2[1]} vs ${hitm[1]}) \u2713`);

let vg = w12Text("w12-vg-svg");
const vgm = vg.match(/true edges recovered (\d+)\/78/);
if (!vgm) { failures++; console.error("  FAIL w12-vgae counts: " + vg.slice(-200)); }
else {
  for (const b of window.document.querySelectorAll("#w12-vg-widget [data-t]")) {
    if (b.getAttribute("data-t") === "0.9") b.dispatchEvent(new window.Event("click", { bubbles: true }));
  }
  const vg2 = w12Text("w12-vg-svg").match(/true edges recovered (\d+)\/78/);
  if (!vg2 || +vg2[1] >= +vgm[1]) { failures++; console.error("  FAIL w12-vgae threshold monotonicity"); }
  else console.log(`  w12-vgae: raising threshold drops recovered ${vgm[1]} -> ${vg2[1]} \u2713`);
}

let ge = w12Text("w12-ge-svg");
const gem = ge.match(/clustering (\d\.\d+)/);
for (const b of window.document.querySelectorAll("#w12-ge-widget [data-set]")) {
  if (b.getAttribute("data-set") === "er") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
const ge2 = w12Text("w12-ge-svg").match(/clustering (\d\.\d+)/);
if (!gem || !ge2 || +ge2[1] <= +gem[1]) { failures++; console.error(`  FAIL w12-generate: model TV ${gem && gem[1]} vs ER ${ge2 && ge2[1]}`); }
else console.log(`  w12-generate: model clustering TV ${gem[1]} beats ER ${ge2[1]} \u2713`);

// week-13 correctness probes
const w13Text = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
let at13 = w13Text("w13-at-svg");
const rm = at13.match(/(\d+)\/12 nodes reachable/);
if (!rm || +rm[1] >= 12) { failures++; console.error("  FAIL w13-attn L1 partial reach: " + at13.slice(0, 200)); }
else console.log(`  w13-attn: 1 layer reaches ${rm[1]}/12 \u2713`);
for (const b of window.document.querySelectorAll("#w13-at-widget [data-m]")) {
  if (b.getAttribute("data-m") === "attn") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
at13 = w13Text("w13-at-svg");
if (!/144 pair scores/.test(at13)) { failures++; console.error("  FAIL w13-attn n2 cost: " + at13.slice(0, 200)); }
else console.log("  w13-attn: attention costs 144 = n\u00B2 \u2713");

for (const b of window.document.querySelectorAll("#w13-rw-widget [data-pair]")) {
  if (b.getAttribute("data-pair") === "1") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
let rw13 = w13Text("w13-rw-svg");
if (!/L1 distance: 0\.000/.test(rw13)) { failures++; console.error("  FAIL w13-rwse twins: " + rw13.slice(-200)); }
else console.log("  w13-rwse: wallflower twins share fingerprints exactly \u2713");
for (const b of window.document.querySelectorAll("#w13-rw-widget [data-pair]")) {
  if (b.getAttribute("data-pair") === "0") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
rw13 = w13Text("w13-rw-svg");
const dm = rw13.match(/L1 distance: (\d\.\d+)/);
if (!dm || +dm[1] < 0.05) { failures++; console.error("  FAIL w13-rwse hub-vs-leaf: " + rw13.slice(-200)); }
else console.log(`  w13-rwse: hub vs leaf differ (L1 ${dm[1]}) \u2713`);

let bi = w13Text("w13-bi-svg");
if (!/rediscovers message passing/.test(bi)) { failures++; console.error("  FAIL w13-bias decay verdict: " + bi.slice(-200)); }
else console.log("  w13-bias: decay profile = soft MPNN \u2713");
for (const b of window.document.querySelectorAll("#w13-bi-widget [data-p]")) {
  if (b.getAttribute("data-p") === "invert") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
bi = w13Text("w13-bi-svg");
if (!/attends far first/.test(bi)) { failures++; console.error("  FAIL w13-bias invert verdict: " + bi.slice(-200)); }
else console.log("  w13-bias: inverted profile attends far \u2713");

let re13 = w13Text("w13-re-svg");
if (!/0\.349/.test(re13) || !/0\.296/.test(re13) || !/0\.303/.test(re13)) { failures++; console.error("  FAIL w13-results rows: " + re13.slice(0, 240)); }
else console.log("  w13-results: measured 0.349/0.296/0.303 present \u2713");
click("w13-re-ref");
re13 = w13Text("w13-re-svg");
if (!/0\.070/.test(re13)) { failures++; console.error("  FAIL w13-results reference: " + re13.slice(0, 240)); }
else console.log("  w13-results: budget reference overlay shows \u2713");

// week-14 correctness probes
const w14Text = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
for (const b of window.document.querySelectorAll("#w14-pr-widget [data-k]")) {
  if (b.getAttribute("data-k") === "2") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
let pr14 = w14Text("w14-pr-svg");
if (!/mass\(u2\) = 0\.250/.test(pr14) || !/mass\(u3\) = 0\.000/.test(pr14)) { failures++; console.error("  FAIL w14-propagate K=2: " + pr14.slice(-240)); }
else console.log("  w14-propagate: co-consumption hop u2 = 0.250 at K=2, u3 still 0 ✓");
for (const b of window.document.querySelectorAll("#w14-pr-widget [data-k]")) {
  if (b.getAttribute("data-k") === "4") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
pr14 = w14Text("w14-pr-svg");
if (!/mass\(u3\) = 0\.063/.test(pr14)) { failures++; console.error("  FAIL w14-propagate K=4: " + pr14.slice(-240)); }
else console.log("  w14-propagate: u3 first hears at K=4 (0.063) ✓");

let me14 = w14Text("w14-me-svg");
if (!/0\.667/.test(me14) || !/0\.671/.test(me14)) { failures++; console.error("  FAIL w14-metrics K=5: " + me14.slice(-300)); }
else console.log("  w14-metrics: K=5 recall 0.667, NDCG 0.671 ✓");
for (const b of window.document.querySelectorAll("#w14-me-widget [data-k]")) {
  if (b.getAttribute("data-k") === "10") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
me14 = w14Text("w14-me-svg");
if (!/1\.000/.test(me14) || !/0\.813/.test(me14)) { failures++; console.error("  FAIL w14-metrics K=10: " + me14.slice(-300)); }
else console.log("  w14-metrics: K=10 recall 1.000 but NDCG 0.813 ✓");

let sp14 = w14Text("w14-sp-svg");
if (!/0\.195/.test(sp14) || !/time wall/.test(sp14)) { failures++; console.error("  FAIL w14-splits temporal: " + sp14.slice(-300)); }
else console.log("  w14-splits: temporal wall + honest 0.195 ✓");
for (const b of window.document.querySelectorAll("#w14-sp-widget [data-p]")) {
  if (b.getAttribute("data-p") === "random") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
sp14 = w14Text("w14-sp-svg");
if (!/0\.346/.test(sp14) || !/future informs the past/.test(sp14)) { failures++; console.error("  FAIL w14-splits random: " + sp14.slice(-300)); }
else console.log("  w14-splits: random split leak 0.346 exposed ✓");

let mp14 = w14Text("w14-mp-svg");
if (!/Weeks 3, 6, 11, 14/.test(mp14)) { failures++; console.error("  FAIL w14-map recsys: " + mp14.slice(-300)); }
else console.log("  w14-map: recsys sector cites its weeks ✓");
for (const b of window.document.querySelectorAll("#w14-mp-widget [data-s]")) {
  if (b.getAttribute("data-s") === "fraud") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
mp14 = w14Text("w14-mp-svg");
if (!/Weeks 2, 10, 14/.test(mp14)) { failures++; console.error("  FAIL w14-map fraud: " + mp14.slice(-300)); }
else console.log("  w14-map: fraud sector cites Weeks 2, 10, 14 ✓");

// week-1 hero force-layout probes
{
  const heroTexts = [...window.document.querySelectorAll("#w1-he-svg svg text")].map((t) => t.textContent).join(" | ");
  const heroCircles = window.document.querySelectorAll("#w1-he-svg svg circle").length;
  const heroLines = window.document.querySelectorAll("#w1-he-svg svg line").length;
  if (heroCircles !== 34) { failures++; console.error(`  FAIL w1-hero: ${heroCircles} nodes, want 34`); }
  else console.log("  w1-hero: 34 members rendered ✓");
  if (heroLines !== 78) { failures++; console.error(`  FAIL w1-hero: ${heroLines} edges, want 78`); }
  else console.log("  w1-hero: 78 ties rendered ✓");
  if (!/Mr\. Hi \(instructor\)/.test(heroTexts) || !/the Officer \(president\)/.test(heroTexts) || !/the one miss/.test(heroTexts)) {
    failures++; console.error("  FAIL w1-hero labels: " + heroTexts.slice(0, 200));
  } else console.log("  w1-hero: leader labels + the-one-miss annotation ✓");
}

// week-1 multigraph + Konigsberg probes
{
  const T = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
  const dots = window.document.querySelectorAll("#w1-mg-svg svg circle").length;
  if (dots < 26) { failures++; console.error(`  FAIL w1-multigraph: ${dots} circles, want 24 dots + 2 accounts`); }
  else console.log("  w1-multigraph: 24 transfers + 2 accounts rendered ✓");
  for (const b of window.document.querySelectorAll("#w1-mg-widget [data-view]")) {
    if (b.getAttribute("data-view") === "collapsed") b.dispatchEvent(new window.Event("click", { bubbles: true }));
  }
  let mg = T("w1-mg-svg");
  if (!/w = 24/.test(mg) || !/pattern is gone/.test(mg)) { failures++; console.error("  FAIL w1-multigraph collapsed: " + mg.slice(-200)); }
  else console.log("  w1-multigraph: collapse shows identical w = 24 ✓");
  for (const b of window.document.querySelectorAll("#w1-mg-widget [data-scen]")) {
    if (b.getAttribute("data-scen") === "burst") b.dispatchEvent(new window.Event("click", { bubbles: true }));
  }
  mg = T("w1-mg-svg");
  if (!/w = 24/.test(mg)) { failures++; console.error("  FAIL w1-multigraph burst collapsed weight changed"); }
  else console.log("  w1-multigraph: burst scenario collapses to the SAME weight ✓");

  let kb = T("w1-kb-svg");
  if (!/seven bridges/.test(kb)) { failures++; console.error("  FAIL w1-konigsberg city view: " + kb.slice(0, 200)); }
  else console.log("  w1-konigsberg: city view asks the question ✓");
  for (const b of window.document.querySelectorAll("#w1-kb-widget [data-kb]")) {
    if (b.getAttribute("data-kb") === "graph") b.dispatchEvent(new window.Event("click", { bubbles: true }));
  }
  kb = T("w1-kb-svg");
  if (!/deg 5/.test(kb) || !/impossible/.test(kb)) { failures++; console.error("  FAIL w1-konigsberg graph view: " + kb.slice(-220)); }
  else console.log("  w1-konigsberg: degrees 3,3,5,3 -> impossible ✓");
}

// week-1 triadic-closure probes: the measured clustering of two degree-6 members,
// and the triangle surplus of the real club over a degree-preserving rewiring.
{
  const T = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
  let cl = T("w1-cl-svg");
  if (!/10 of 15 friend-pairs/.test(cl) || !/clustering C = 0\.67/.test(cl)) {
    failures++; console.error("  FAIL w1-closure member 3 clustering: " + cl.slice(0, 300));
  } else console.log("  w1-closure: member 3 closes 10/15 pairs, C = 0.67 ✓");
  if (!/3 of 15 friend-pairs/.test(cl) || !/clustering C = 0\.20/.test(cl)) {
    failures++; console.error("  FAIL w1-closure member 31 clustering: " + cl.slice(0, 300));
  } else console.log("  w1-closure: member 31 (same degree) closes 3/15, C = 0.20 ✓");

  for (const b of window.document.querySelectorAll("#w1-cl-widget [data-view]")) {
    if (b.getAttribute("data-view") === "chance") b.dispatchEvent(new window.Event("click", { bubbles: true }));
  }
  cl = T("w1-cl-svg");
  if (!/45 triangles/.test(cl)) {
    failures++; console.error("  FAIL w1-closure real triangle count (want 45): " + cl.slice(0, 300));
  } else console.log("  w1-closure: real karate club has 45 triangles ✓");
  // Every reshuffle must stay well under the real count, not just the first.
  for (let k = 0; k < 6; k++) {
    const c = [...T("w1-cl-svg").matchAll(/(\d+) triangles/g)].map((m) => Number(m[1]));
    if (c.length < 2 || c[0] !== 45 || !(c[0] > 2 * c[1])) {
      failures++; console.error(`  FAIL w1-closure reshuffle ${k}: ` + JSON.stringify(c));
      break;
    }
    if (k === 5) console.log("  w1-closure: all 6 same-density reshuffles stay below half of 45 ✓");
    window.document.getElementById("w1-cl-reshuffle").dispatchEvent(new window.Event("click", { bubbles: true }));
  }

}

// week-1 walk-counting probes: the cell value must equal the number of walks listed,
// and the k=2 diagonal must reproduce the degree sequence.
{
  const T = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
  let wk = T("w1-wk-svg");
  if (!/\(A\^2\)\[A,C\] = 1/.test(wk) || !/A → B → C/.test(wk) || !/exactly 1 walk is listed/.test(wk)) {
    failures++; console.error("  FAIL w1-walks default cell A->C at k=2: " + wk.slice(-320));
  } else console.log("  w1-walks: (A^2)[A,C] = 1 and the single walk A→B→C is listed ✓");

  // click the (C,C) diagonal cell: must read 4, node C's degree
  const cells = window.document.querySelectorAll("#w1-wk-svg svg rect");
  if (cells.length < 36) { failures++; console.error(`  FAIL w1-walks: only ${cells.length} matrix cells`); }
  else {
    cells[2 * 6 + 2].dispatchEvent(new window.Event("click", { bubbles: true }));
    wk = T("w1-wk-svg");
    if (!/\(A\^2\)\[C,C\] = 4/.test(wk)) {
      failures++; console.error("  FAIL w1-walks diagonal must equal degree 4: " + wk.slice(-320));
    } else console.log("  w1-walks: diagonal (A^2)[C,C] = 4 = deg(C) ✓");
  }
}

// week-1 Laplacian probes: the three presets share one value multiset, so they must
// share the 16.8 chance average while their totals separate homophily from heterophily.
{
  const T = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
  let lp = T("w1-lp-svg");
  if (!/xᵀLx  =  8/.test(lp) || !/16\.8/.test(lp) || !/homophily/.test(lp)) {
    failures++; console.error("  FAIL w1-lap agree preset (want 8 vs 16.8, homophily): " + lp.slice(0, 400));
  } else console.log("  w1-lap: friends agree -> 8 against a 16.8 chance average, homophily ✓");

  for (const b of window.document.querySelectorAll("#w1-lp-widget [data-preset]")) {
    if (b.getAttribute("data-preset") === "disagree") b.dispatchEvent(new window.Event("click", { bubbles: true }));
  }
  lp = T("w1-lp-svg");
  if (!/xᵀLx  =  20/.test(lp) || !/16\.8/.test(lp) || !/heterophily/.test(lp)) {
    failures++; console.error("  FAIL w1-lap disagree preset (want 20 vs same 16.8, heterophily): " + lp.slice(0, 400));
  } else console.log("  w1-lap: friends disagree -> 20 against the SAME 16.8 baseline, heterophily ✓");
}



// ===== week-2 additions (2026-09-02): nine new widgets, probes authored per widget =====
// week-2 Katz probes: the same ten-node graph must crown the hub S at a low walk
// discount and the clique member K1 at a high one, with the tipping point β* = 0.200
// and λmax = 3.18 (1/λmax = 0.315) computed live from the 12-edge list.
{
  const T = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
  const kzClick = (preset) => {
    for (const b of window.document.querySelectorAll("#w2-kz-widget [data-beta]")) {
      if (b.getAttribute("data-beta") === preset) b.dispatchEvent(new window.Event("click", { bubbles: true }));
    }
  };
  kzClick("low");
  let kz = T("w2-kz-svg");
  if (!/ranked \(top node = 1\) \| S \| 1\.000 \| K1 \|/.test(kz) || !/top node: S, the popular hub/.test(kz) ||
      !/β = 0\.016   \(λmax = 3\.18, so β must stay below 1\/λmax = 0\.315\)/.test(kz) || !/β\* = 0\.200/.test(kz)) {
    failures++; console.error("  FAIL w2-katz low preset (want S on top, β = 0.016, λmax = 3.18, β* = 0.200): " + kz.slice(0, 600));
  } else console.log("  w2-katz: low β -> S tops the ranking, β* = 0.200 shown ✓");

  kzClick("high");
  kz = T("w2-kz-svg");
  if (!/ranked \(top node = 1\) \| K1 \| 1\.000 \| K2 \|/.test(kz) || !/β = 0\.299/.test(kz) ||
      !/top node: K1 — past the tipping point β\* = 0\.200:/.test(kz) ||
      !/the well-connected clique now outranks the popular hub/.test(kz)) {
    failures++; console.error("  FAIL w2-katz high preset (want K1 on top at β = 0.299, past β* = 0.200): " + kz.slice(0, 600));
  } else console.log("  w2-katz: high β -> K1 overtakes S past the tipping point β* = 0.200 ✓");
  kzClick("mid");
}

// week-2 surfer probes: a dead end leaks mass at β = 1, a spider trap swallows it,
// and teleport (β = 0.85) restores Σ r = 1 while draining the trap to ~0.58.
{
  const T = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
  const sfClick = (sel) => {
    const el = window.document.querySelector(`#w2-sf-widget ${sel}`);
    if (el) el.dispatchEvent(new window.Event("click", { bubbles: true }));
  };

  sfClick('[data-scen="dead"]'); sfClick("#w2-sf-run");           // β = 1 is the default
  let sf = T("w2-sf-svg");
  if (!/Σ r = 0\.027/.test(sf) || !/step k = 30/.test(sf) || !/mass is leaking out of the web/.test(sf)) {
    failures++; console.error("  FAIL w2-surfer dead end at β=1 (want Σ r = 0.027 after 30 steps, leaking verdict): " + sf.slice(0, 500));
  } else console.log("  w2-surfer: dead end, β = 1 -> Σ r = 0.027 after 30 steps, mass is leaking ✓");

  sfClick('[data-scen="trap"]'); sfClick("#w2-sf-run");
  sf = T("w2-sf-svg");
  if (!/mass inside the trap = 0\.978/.test(sf) || !/Σ r = 1\.000/.test(sf) || !/the trap is swallowing everything/.test(sf)) {
    failures++; console.error("  FAIL w2-surfer spider trap at β=1 (want trap = 0.978, Σ r = 1.000, swallowing verdict): " + sf.slice(0, 500));
  } else console.log("  w2-surfer: spider trap, β = 1 -> the trap holds 0.978 after 30 steps ✓");

  sfClick('[data-beta="0.85"]'); sfClick("#w2-sf-run");           // continues from the saturated trap
  sf = T("w2-sf-svg");
  if (!/Σ r = 1\.000/.test(sf) || !/mass inside the trap = 0\.576/.test(sf) || !/step k = 60/.test(sf) ||
      !/teleport keeps Σ r = 1 and the trap leaks 15% per step/.test(sf)) {
    failures++; console.error("  FAIL w2-surfer spider trap at β=0.85 (want Σ r = 1.000, trap = 0.576 at k = 60, teleport verdict): " + sf.slice(0, 500));
  } else console.log("  w2-surfer: β = 0.85 -> Σ r = 1.000 and the trap drains to 0.576 ✓");

  sfClick('[data-beta="1"]'); sfClick('[data-scen="healthy"]');    // leave the widget at its defaults
}

// week-2 power-iteration probes: the lecture's hand-worked steps must print verbatim
// (fractions, X Y Z in order), "run" must land on the fixed point the widget computed
// by iterating 200 times, and the (1, 0, 0) start must take a different route.
{
  const T = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
  const piClick = (id) => window.document.getElementById(id).dispatchEvent(new window.Event("click", { bubbles: true }));
  const piStart = (s) => {
    for (const b of window.document.querySelectorAll("#w2-pi-widget [data-start]")) {
      if (b.getAttribute("data-start") === s) b.dispatchEvent(new window.Event("click", { bubbles: true }));
    }
  };
  piStart("uniform");
  piClick("w2-pi-step");
  let pi = T("w2-pi-svg");
  if (!/X \| 1\/3 \| Y \| 1\/2 \| Z \| 1\/6 \| r⁽¹⁾ = \(1\/3, 1\/2, 1\/6\)/.test(pi) || !/1\/2·1\/3 \+ 1·1\/3 = 1\/2/.test(pi)) {
    failures++; console.error("  FAIL w2-poweriter step 1 (want (1/3, 1/2, 1/6) and the Y sum): " + pi.slice(0, 500));
  } else console.log("  w2-poweriter: uniform start, one step -> (1/3, 1/2, 1/6), r_Y = 1/2·1/3 + 1·1/3 = 1/2 ✓");

  piClick("w2-pi-step");
  pi = T("w2-pi-svg");
  if (!/X \| 1\/2 \| Y \| 1\/3 \| Z \| 1\/6 \| r⁽²⁾ = \(1\/2, 1\/3, 1\/6\)/.test(pi) || !/Σ = 1 \| Σ = 1 \| Σ = 1/.test(pi)) {
    failures++; console.error("  FAIL w2-poweriter step 2 (want (1/2, 1/3, 1/6), columns summing to 1): " + pi.slice(0, 500));
  } else console.log("  w2-poweriter: second step -> (1/2, 1/3, 1/6), every column of M sums to 1 ✓");

  piClick("w2-pi-run");
  pi = T("w2-pi-svg");
  if (!/X \| 0\.400 \| Y \| 0\.400 \| Z \| 0\.200 \| r⁽⁴⁰⁾ = \(0\.400, 0\.400, 0\.200\)/.test(pi) ||
      !/r\* = \(0\.400, 0\.400, 0\.200\)/.test(pi) || !/converged: r = \(0\.400, 0\.400, 0\.200\) — X and Y tie/.test(pi)) {
    failures++; console.error("  FAIL w2-poweriter run (want (0.400, 0.400, 0.200) = r*, converged verdict): " + pi.slice(-600));
  } else console.log("  w2-poweriter: run to 40 -> (0.400, 0.400, 0.200), equal to the 200-iteration r*, X and Y tie ✓");

  piStart("allx");
  piClick("w2-pi-step");
  pi = T("w2-pi-svg");
  if (!/X \| 0 \| Y \| 1\/2 \| Z \| 1\/2 \| r⁽¹⁾ = \(0, 1\/2, 1\/2\)/.test(pi)) {
    failures++; console.error("  FAIL w2-poweriter all-on-X start, one step (want (0, 1/2, 1/2)): " + pi.slice(0, 500));
  } else console.log("  w2-poweriter: (1, 0, 0) start, one step -> (0, 1/2, 1/2) ✓");
  piStart("uniform");
}

// week-2 clustering probes: the node view must do the lecture's worked example on the
// real adjacency matrix (C_C = 2/6, (A³)[C,C] = 4 = 2 × 2 triangles), report D's
// undefined coefficient honestly, and the global view must show the two "global
// clustering" numbers disagreeing in opposite directions on the two built graphs
// (windmill C̄ 0.93 vs T 0.23; clique-with-leaves C̄ 0.13 vs T 0.40).
{
  const T = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
  const clickView = (v) => {
    for (const b of window.document.querySelectorAll("#w2-cc-widget [data-view]")) {
      if (b.getAttribute("data-view") === v) b.dispatchEvent(new window.Event("click", { bubbles: true }));
    }
  };
  let cc = T("w2-cc-svg");
  if (!/d_C = 4 → C\(4,2\) = 6 pairs/.test(cc) || !/C_C = 2\/6 = 0\.33/.test(cc) ||
      !/\(A³\)\[C,C\] = 4 = 2 × 2 triangles/.test(cc) || !/0\.61/.test(cc) || !/0\.50/.test(cc)) {
    failures++; console.error("  FAIL w2-clustering node view at C (want 2/6 = 0.33, (A³)[C,C] = 4, C̄ 0.61, T 0.50): " + cc.slice(0, 500));
  } else console.log("  w2-clustering: focus C -> C_C = 2/6 = 0.33 and (A³)[C,C] = 4 = 2 × 2 triangles; cast C̄ = 0.61, T = 0.50 ✓");

  const dNode = window.document.querySelector('#w2-cc-svg circle[data-node="D"]');
  if (dNode) dNode.dispatchEvent(new window.Event("click", { bubbles: true }));
  cc = T("w2-cc-svg");
  if (!/d_D = 1 → C\(1,2\) = 0 pairs/.test(cc) || !/C_D undefined — fewer than two friends,/.test(cc) ||
      !/reported as 0 by convention/.test(cc) || !/\(A³\)\[D,D\] = 0 = 2 × 0 triangles/.test(cc)) {
    failures++; console.error("  FAIL w2-clustering node view at D (want undefined, 0 by convention, (A³)[D,D] = 0): " + cc.slice(0, 500));
  } else console.log("  w2-clustering: focus D -> undefined, reported as 0 by convention, (A³)[D,D] = 0 ✓");

  const aNode = window.document.querySelector('#w2-cc-svg circle[data-node="A"]');
  if (aNode) aNode.dispatchEvent(new window.Event("click", { bubbles: true }));
  cc = T("w2-cc-svg");
  if (!/1 of 3 pairs is an edge/.test(cc) || !/C_A = 1\/3 = 0\.33/.test(cc) || !/\(A³\)\[A,A\] = 2 = 2 × 1 triangle/.test(cc)) {
    failures++; console.error("  FAIL w2-clustering node view at A (want 1/3 = 0.33, (A³)[A,A] = 2): " + cc.slice(0, 500));
  } else console.log("  w2-clustering: focus A -> C_A = 1/3 = 0.33 and (A³)[A,A] = 2 = 2 × 1 triangle ✓");

  clickView("global");
  cc = T("w2-cc-svg");
  const wind = /n = 13, m = 18, 6 triangles, 78 wedges/.test(cc) && /hub H: C = 6\/66 = 0\.09/.test(cc) &&
               /C̄ \| 0\.93 \| mean of 13 C_u \| T \| 0\.23 \| 3·6 \/ 78 wedges/.test(cc) &&
               /C̄ says tight-knit, T says sparse \(4\.0×\)/.test(cc);
  const cliq = /n = 15, m = 20, 10 triangles, 75 wedges/.test(cc) && /5 clique nodes: C = 6\/15 = 0\.40 each/.test(cc) &&
               /C̄ \| 0\.13 \| mean of 15 C_u \| T \| 0\.40 \| 3·10 \/ 75 wedges/.test(cc) &&
               /C̄ says sparse, T says tight-knit \(3\.0×\)/.test(cc);
  if (!wind || !cliq) {
    failures++; console.error("  FAIL w2-clustering global view (want windmill C̄ 0.93 / T 0.23 and clique C̄ 0.13 / T 0.40, opposite verdicts): " + cc.slice(0, 700));
  } else console.log("  w2-clustering: windmill C̄ 0.93 vs T 0.23, clique-with-leaves C̄ 0.13 vs T 0.40 — opposite verdicts ✓");
  clickView("node");
}

// week-2 graphlet probes: P and Q agree on degree (3) and clustering (0.33) but the
// induced-orbit census separates them — square 1 vs 0 by default, 4-path-middle 1 vs 2 —
// and moving P to a same-degree, different-clustering node (node 3) flips the verdict.
{
  const T = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
  let gl = T("w2-gl-svg");
  if (!/degree: P = Q = 3 · clustering: P = Q = 0\.33/.test(gl) ||
      !/shaded — square: 1 at P, 0 at Q/.test(gl) ||
      !/identical by degree and clustering — told apart by graphlets/.test(gl)) {
    failures++; console.error("  FAIL w2-graphlets default (want deg 3 = 3, C 0.33 = 0.33, square 1 vs 0, graphlet verdict): " + gl.slice(0, 400));
  } else console.log("  w2-graphlets: P = Q on degree 3 and C = 0.33, squares 1 vs 0, told apart by graphlets ✓");

  for (const b of window.document.querySelectorAll("#w2-gl-widget [data-orbit]")) {
    if (b.getAttribute("data-orbit") === "p4Mid") b.dispatchEvent(new window.Event("click", { bubbles: true }));
  }
  gl = T("w2-gl-svg");
  if (!/shaded — 4-path, middle: 1 at P, 2 at Q/.test(gl) || !/wedge, end \| 2 \| 1 \|/.test(gl)) {
    failures++; console.error("  FAIL w2-graphlets 4-path-middle (want 1 at P, 2 at Q; wedge-end 2 vs 1): " + gl.slice(0, 400));
  } else console.log("  w2-graphlets: 4-path middle 1 vs 2 and wedge end 2 vs 1 — three orbits differ ✓");

  // node 3 has degree 3 like Q but clustering 0: the cheaper feature already separates them
  const node3 = window.document.querySelector("#w2-gl-svg svg > g > circle:nth-of-type(3)");
  if (node3) node3.dispatchEvent(new window.Event("click", { bubbles: true }));
  gl = T("w2-gl-svg");
  if (!/degree: P = 3, Q = 3 · clustering: P = 0\.00, Q = 0\.33/.test(gl) || !/same degree, different clustering/.test(gl)) {
    failures++; console.error("  FAIL w2-graphlets node-3-as-P (want deg 3 vs 3, C 0.00 vs 0.33, clustering verdict): " + gl.slice(0, 400));
  } else console.log("  w2-graphlets: node 3 as P -> same degree, clustering 0.00 vs 0.33 already separates ✓");

  const reset = window.document.getElementById("w2-gl-reset");
  if (reset) reset.dispatchEvent(new window.Event("click", { bubbles: true }));
  for (const b of window.document.querySelectorAll("#w2-gl-widget [data-orbit]")) {
    if (b.getAttribute("data-orbit") === "square") b.dispatchEvent(new window.Event("click", { bubbles: true }));
  }
}

// week-2 null-model probes: the formula d_u d_v / 2m with the numbers substituted, the
// modularity-matrix entry observed − expected, and the empirical stub-matching mean after
// 100 seeded rewirings (exact value 12/13 = 0.923; the 100-draw mean must land near it).
{
  const T = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
  const clickSel = (sel) => window.document.querySelector(sel).dispatchEvent(new window.Event("click", { bubbles: true }));
  clickSel("#w2-nm-reset");
  let nm = T("w2-nm-svg");
  if (!/3·4\/14 = 0\.857/.test(nm) || !/1 − 0\.857 = \+0\.143/.test(nm) || !/mean of 0 rewirings/.test(nm)) {
    failures++; console.error("  FAIL w2-nullmodel default (want 3·4/14 = 0.857, +0.143, 0 rewirings): " + nm.slice(0, 500));
  } else console.log("  w2-nullmodel: A–C formula 3·4/14 = 0.857, B_Q[A,C] = +0.143 ✓");

  clickSel("#w2-nm-many");
  nm = T("w2-nm-svg");
  const mm = nm.match(/mean of 100 rewirings so far = (\d\.\d{3})/);
  const emp = mm ? Number(mm[1]) : NaN;
  if (!mm || Math.abs(emp - 0.92) > 0.25 || !/rewiring #100/.test(nm)) {
    failures++; console.error("  FAIL w2-nullmodel after 100 rewirings (want mean within 0.25 of 0.92): " + nm.slice(0, 500));
  } else console.log(`  w2-nullmodel: 100 rewirings -> empirical A–C mean ${emp} against exact 0.923 ✓`);

  for (const b of window.document.querySelectorAll("#w2-nm-widget [data-pair]")) {
    if (b.getAttribute("data-pair") === "DE") b.dispatchEvent(new window.Event("click", { bubbles: true }));
  }
  nm = T("w2-nm-svg");
  if (!/1·2\/14 = 0\.143/.test(nm) || !/0 − 0\.143 = −0\.143/.test(nm) || !/mean of 100 rewirings/.test(nm)) {
    failures++; console.error("  FAIL w2-nullmodel D–E (want 1·2/14 = 0.143, −0.143, tallies kept): " + nm.slice(0, 500));
  } else console.log("  w2-nullmodel: D–E formula 1·2/14 = 0.143, B_Q[D,E] = −0.143, tallies survive the switch ✓");
  clickSel("#w2-nm-reset");
}

// week-2 resolution-limit probes: a ring of c k-cliques scored two ways from the
// definition of Q. At the lecture's 24 triangles (m = 96) merged pairs must beat the
// ground truth (0.792 vs 0.708); at c = 6 the same triangles must be kept apart; and
// with 5-cliques the crossover must move to c = 22.
{
  const T = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
  let rl = T("w2-rl-svg");
  if (!/c = 24, k = 3, m = 96/.test(rl) || !/Q\(truth\) = 0\.708/.test(rl) || !/Q\(pairs\) = 0\.792/.test(rl) ||
      !/pairs beat truth for c > 8/.test(rl) || !/prefers to MERGE perfect communities/.test(rl) ||
      !/√\(2m\) = 13\.9 internal edges are at risk/.test(rl)) {
    failures++; console.error("  FAIL w2-resolution default (want c=24 k=3 m=96, Q(truth)=0.708 < Q(pairs)=0.792, c*=8, merge verdict): " + rl.slice(0, 600));
  } else console.log("  w2-resolution: 24 triangles, m = 96 -> Q(truth) = 0.708 < Q(pairs) = 0.792, crossover c > 8, MERGE verdict ✓");

  const rlSlider = window.document.getElementById("w2-rl-c");
  rlSlider.value = "6";
  rlSlider.dispatchEvent(new window.Event("input", { bubbles: true }));
  rl = T("w2-rl-svg");
  if (!/c = 6, k = 3, m = 24/.test(rl) || !/Q\(truth\) = 0\.583/.test(rl) || !/Q\(pairs\) = 0\.542/.test(rl) ||
      !/keeps every clique separate/.test(rl) || /MERGE/.test(rl)) {
    failures++; console.error("  FAIL w2-resolution c=6 (want m=24, Q(truth)=0.583 > Q(pairs)=0.542, separate verdict): " + rl.slice(0, 600));
  } else console.log("  w2-resolution: 6 triangles, m = 24 -> Q(truth) = 0.583 > Q(pairs) = 0.542, kept separate ✓");

  for (const b of window.document.querySelectorAll("#w2-rl-widget [data-k]")) {
    if (b.getAttribute("data-k") === "5") b.dispatchEvent(new window.Event("click", { bubbles: true }));
  }
  rl = T("w2-rl-svg");
  if (!/c = 6, k = 5, m = 66/.test(rl) || !/pairs beat truth for c > 22/.test(rl)) {
    failures++; console.error("  FAIL w2-resolution k=5 (want m=66 at c=6, crossover c > 22): " + rl.slice(0, 600));
  } else console.log("  w2-resolution: cliques of 5 -> m = 66 at c = 6, crossover moves to c > 22 ✓");

  // restore the defaults for anything probed later
  rlSlider.value = "24";
  rlSlider.dispatchEvent(new window.Event("input", { bubbles: true }));
  for (const b of window.document.querySelectorAll("#w2-rl-widget [data-k]")) {
    if (b.getAttribute("data-k") === "3") b.dispatchEvent(new window.Event("click", { bubbles: true }));
  }
}

// week-2 spectral probes: zero eigenvalues count components (1 for the cast graph,
// 2 once C–E and C–F are deleted) and the Fiedler split lands on the cheapest cut
// (2 edges on the cast graph, exactly the bridge on the barbell). All numbers were
// checked against numpy: cast λ₂ = 0.6314, barbell λ₂ = 0.3542.
{
  const T = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
  const pick = (name) => {
    for (const b of window.document.querySelectorAll("#w2-sp-widget [data-graph]")) {
      if (b.getAttribute("data-graph") === name) b.dispatchEvent(new window.Event("click", { bubbles: true }));
    }
  };
  let sp = T("w2-sp-svg");
  if (!/appears once → 1 component/.test(sp) || !/cut = 2 edges: \{A, B, D\} \| \{C, E, F\}/.test(sp) || !/λ₂ = 0\.63/.test(sp)) {
    failures++; console.error("  FAIL w2-spectral cast (want 1 component, cut = 2 edges {A, B, D} | {C, E, F}, λ₂ = 0.63): " + sp.slice(0, 500));
  } else console.log("  w2-spectral: cast graph -> 1 component, λ₂ = 0.63, Fiedler cut {A, B, D} | {C, E, F} = 2 edges ✓");

  pick("pieces");
  sp = T("w2-sp-svg");
  const zeroLabels = (sp.match(/= 0\.00/g) || []).length;
  if (!/appears twice → 2 components/.test(sp) || zeroLabels !== 2 || !/cut = 0 edges: \{A, B, C, D\} \| \{E, F\}/.test(sp)) {
    failures++; console.error(`  FAIL w2-spectral two pieces (want 2 components, exactly two "= 0.00" labels, got ${zeroLabels}, cut = 0): ` + sp.slice(0, 500));
  } else console.log("  w2-spectral: two pieces -> eigenvalue 0 twice, 2 components, zero-eigenvector constant on each piece ✓");

  pick("barbell");
  sp = T("w2-sp-svg");
  if (!/cut = 1 edge: \{1, 2, 3, 4\} \| \{5, 6, 7, 8\}/.test(sp) || !/λ₂ = 0\.35/.test(sp) || !/appears once → 1 component/.test(sp)) {
    failures++; console.error("  FAIL w2-spectral barbell (want cut = 1 edge {1, 2, 3, 4} | {5, 6, 7, 8}, λ₂ = 0.35): " + sp.slice(0, 500));
  } else console.log("  w2-spectral: barbell -> λ₂ = 0.35, Fiedler split 4 | 4, the bridge is the only cut edge ✓");
  pick("cast");
}

// week-2 baseline probes: the four lecture accuracies are all drawn, the default
// (62%) panel reads position as a hard one-hot with the words never read, the
// verdict computes the 43-point jump from the table, and the GCN panel reads the words.
{
  const T = (id) => [...window.document.querySelectorAll(`#${id} svg text`)].map((t) => t.textContent).join(" | ");
  let bl = T("w2-bl-svg");
  const missing = ["14%", "19%", "62%", "81%"].filter((v) => !bl.split(" | ").includes(v));
  if (missing.length) {
    failures++; console.error("  FAIL w2-baseline value labels missing " + missing.join(", ") + ": " + bl.slice(0, 400));
  } else console.log("  w2-baseline: 14% / 19% / 62% / 81% value labels present ✓");

  if (!/statistics \+ community one-hot \| statistics \| how central \/ clustered \| position \| region, as a one-hot \| words \| never read/.test(bl)
      || !/topic lives in where/.test(bl)) {
    failures++; console.error("  FAIL w2-baseline default panel (want position as a one-hot, words never read): " + bl.slice(-700));
  } else console.log("  w2-baseline: default 62% row -> position chip filled as a one-hot, words never read ✓");

  if (!/The 43-point jump comes from position, not from a better statistic\./.test(bl)) {
    failures++; console.error("  FAIL w2-baseline verdict (want the 43-point jump, 62 − 19): " + bl.slice(-200));
  } else console.log("  w2-baseline: verdict computes the 43-point jump (62 − 19) ✓");

  for (const b of window.document.querySelectorAll("#w2-bl-widget [data-row]")) {
    if (b.getAttribute("data-row") === "gcn") b.dispatchEvent(new window.Event("click", { bubbles: true }));
  }
  bl = T("w2-bl-svg");
  if (!/GCN on structure and words \| statistics \| learned implicitly \| position \| region, learned \| words \| the paper's text/.test(bl)) {
    failures++; console.error("  FAIL w2-baseline GCN panel (want words = the paper's text): " + bl.slice(-700));
  } else console.log("  w2-baseline: GCN row -> words chip filled, the paper's text ✓");
}

// week-1 permutation-invariance probes
{
  const peText = () => [...window.document.querySelectorAll("#w1-pe-svg svg text")].map((t) => t.textContent).join(" | ");
  let t0 = peText();
  if (!/sorted: \[1, 2, 2, 2, 3, 4\]/.test(t0) || !/m = 7 edges/.test(t0)) {
    failures++; console.error("  FAIL w1-permute invariants at identity: " + t0.slice(-200));
  } else console.log("  w1-permute: invariant readouts present ✓");
  const slots0 = t0.match(/#\d/g).join("");
  click("w1-pe-shuffle");
  const t1 = peText();
  if (!/sorted: \[1, 2, 2, 2, 3, 4\]/.test(t1) || !/m = 7 edges/.test(t1) || !/unchanged/.test(t1)) {
    failures++; console.error("  FAIL w1-permute invariants after shuffle: " + t1.slice(-200));
  } else console.log("  w1-permute: invariants survive the shuffle ✓");
  const slots1 = t1.match(/#\d/g).join("");
  if (slots0 === slots1) { failures++; console.error("  FAIL w1-permute: storage badges did not reorder"); }
  else console.log("  w1-permute: storage order actually shuffled ✓");
  click("w1-pe-reset");
}

// week-2 correctness probes: drive WL cast to stability and check the verdict text
for (const b of window.document.querySelectorAll("#w2-wl-widget [data-scene]")) {
  if (b.getAttribute("data-scene") === "cast") b.dispatchEvent(new window.Event("click", { bubbles: true }));
}
for (let i = 0; i < 4; i++) click("w2-wl-step");
const wlTexts = [...window.document.querySelectorAll("#w2-wl-svg svg text")].map((t) => t.textContent).join(" | ");
if (!/stable after round 2/.test(wlTexts)) { failures++; console.error("  FAIL WL cast did not stabilize at round 2: " + wlTexts.slice(-160)); }
else console.log("  WL cast stabilizes after round 2 ✓");

// sanity: every widget rendered SVG content
for (const id of ["w1-he-svg", "w1-pe-svg", "w1-mg-svg", "w1-kb-svg", "w1-cl-svg", "w1-wk-svg", "w1-lp-svg", "w6-mp-svg", "w6-sp-svg", "w6-nm-svg", "w6-pm-svg", "w1-bd-svg", "w1-ct-svg", "w1-ty-svg", "w1-tk-svg",
  "w2-ce-svg", "w2-pr-svg", "w2-wl-svg", "w2-lv-svg",
  "w3-wk-svg", "w3-pq-svg", "w3-em-svg", "w3-lp-svg",
  "w4-te-svg", "w4-pt-svg", "w4-ng-svg", "w4-rk-svg",
  "w5-qy-svg", "w5-bx-svg", "w5-rg-svg", "w5-ex-svg",
  "w7-ag-svg", "w7-sg-svg", "w7-gt-svg", "w7-ab-svg",
  "w9-wl-svg", "w9-tr-svg", "w9-pw-svg", "w9-sq-svg",
  "w10-ty-svg", "w10-pm-svg", "w10-mp-svg", "w10-sd-svg",
  "w11-ex-svg", "w11-sm-svg", "w11-sg2-svg", "w11-tb-svg",
  "w12-sp-svg", "w12-he-svg", "w12-vg-svg", "w12-ge-svg",
  "w13-at-svg", "w13-rw-svg", "w13-bi-svg", "w13-re-svg",
  "w14-pr-svg", "w14-me-svg", "w14-sp-svg", "w14-mp-svg"]) {
  const svg = window.document.querySelector(`#${id} svg`);
  const n = svg ? svg.querySelectorAll("*").length : 0;
  const min = id === "w4-pt-svg" ? 8 : 10;   // the patterns explorer is legitimately sparse
  if (n < min) { failures++; console.error(`  FAIL #${id}: svg has only ${n} elements`); }
  else console.log(`  #${id}: svg with ${n} elements ✓`);
}

console.log(failures === 0 ? "\nALL WIDGET SMOKE TESTS PASSED" : `\n${failures} FAILURE(S)`);
process.exit(failures === 0 ? 0 : 1);
