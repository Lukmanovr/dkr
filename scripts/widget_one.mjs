// Mount ONE widget (html fragment + js) in jsdom, exercise its buttons, and
// print every <text> it renders — for authoring smoke probes and catching
// runtime errors before a widget joins widget_smoke.mjs.
//     node scripts/widget_one.mjs w2-katz [click-selector ...]
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { JSDOM } from "jsdom";
const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const A = join(ROOT, "assets", "d3") + "/";
const [, , name, ...clicks] = process.argv;
const strip = (f) => readFileSync(A + f, "utf8").replace(/^```\{=html\}\r?\n/, "").replace(/\r?\n```\r?\n?$/, "");
const dom = new JSDOM(`<!doctype html><html><body>${strip(name + ".html")}</body></html>`, { runScripts: "outside-only", pretendToBeVisual: true });
const { window } = dom;
for (const f of ["d3.v7.min.js", "_dkr.js", name + ".js"]) window.eval(readFileSync(A + f, "utf8"));
const dump = (label) => {
  const t = [...window.document.querySelectorAll("svg text")].map((e) => e.textContent).join(" | ");
  console.log(`--- ${label} (${window.document.querySelectorAll("svg *").length} svg elements)\n${t}\n`);
};
dump("initial");
for (const sel of clicks) {
  const el = window.document.querySelector(sel);
  if (!el) { console.log(`!! no element for ${sel}`); continue; }
  el.dispatchEvent(new window.Event("click", { bubbles: true }));
  dump(`after click ${sel}`);
}
