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

const body = ["w1-hero.html", "w1-permute.html", "w6-message-passing.html", "w6-spectral.html", "w6-normalization.html", "w6-permutation.html", "w6-eq-linked.html",
  "w1-builder.html", "w1-cost.html", "w1-types.html", "w1-tasks.html",
  "w2-centrality.html", "w2-pagerank.html", "w2-wl.html", "w2-louvain.html",
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
  "w1-hero.js", "w1-permute.js", "w1-builder.js", "w1-cost.js", "w1-types.js", "w1-tasks.js",
  "w2-centrality.js", "w2-pagerank.js", "w2-wl.js", "w2-louvain.js",
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
for (let i = 0; i < 3; i++) click("w2-pr-step");
click("w2-pr-step10");
slide("w2-pr-beta", 0.5);
const tele = window.document.getElementById("w2-pr-tele");
tele.checked = true; tele.dispatchEvent(new window.Event("change", { bubbles: true }));
click("w2-pr-step");
tele.checked = false; tele.dispatchEvent(new window.Event("change", { bubbles: true }));
click("w2-pr-reset");
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
for (const id of ["w1-he-svg", "w1-pe-svg", "w6-mp-svg", "w6-sp-svg", "w6-nm-svg", "w6-pm-svg", "w1-bd-svg", "w1-ct-svg", "w1-ty-svg", "w1-tk-svg",
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
