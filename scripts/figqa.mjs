/* Shared figure-QA infrastructure: theme extraction, harness building, browser boot.
 * Used by figshot.mjs (screenshots) and figlint.mjs (geometry checks). */
import { readFileSync, mkdirSync, writeFileSync, readdirSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import puppeteer from "puppeteer-core";

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
export const FIGDIR = join(ROOT, "assets", "figures");
export const QADIR = join(ROOT, "qa", "figshots");

export function chromePath() {
  if (process.env.CHROME_PATH) return process.env.CHROME_PATH;
  const candidates = [
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium-browser",
  ];
  for (const c of candidates) if (existsSync(c)) return c;
  throw new Error("Chrome not found; set CHROME_PATH");
}

// Pull the --dkr-* token values straight from each theme's SCSS defaults so the
// harness can never drift from the real themes.
export function themeTokens(scssFile) {
  const scss = readFileSync(join(ROOT, "assets", scssFile), "utf8");
  const get = (name) => (scss.match(new RegExp(`\\$${name}:\\s*(#[0-9a-fA-F]{3,8})`)) || [])[1];
  return {
    "--dkr-accent": get("dkr-accent"),
    "--dkr-accent-dark": get("dkr-accent-dark"),
    "--dkr-blue": get("dkr-teal"),
    "--dkr-blue-dark": get("dkr-teal-deep"),
    "--dkr-green": get("dkr-green"),
    "--dkr-yellow": get("dkr-yellow"),
    "--dkr-red": get("dkr-red"),
    "--dkr-purple": get("dkr-purple"),
    "--dkr-paper": get("dkr-paper"),
    "--dkr-muted": get("dkr-muted"),
    "--dkr-border": get("border-color"),
    "--dkr-text": get("body-color"),
    "--dkr-bg": get("body-bg"),
  };
}

export const THEMES = {
  light: themeTokens("theme-light.scss"),
  dark: themeTokens("theme-dark.scss"),
};

export function stripFences(text) {
  return text.replace(/^```\{=html\}\r?\n/, "").replace(/\r?\n```\r?\n?$/, "");
}

export const WIDGETDIR = join(ROOT, "assets", "d3");
// Widget includes with meaningful static (pre-JS) rendering, QA'd like figures.
const STATIC_WIDGETS = ["w6-eq-linked"];
// JS-rendered widgets: the harness loads their real scripts so shot and lint
// exercise the initial rendered state, not an empty container.
const JS_WIDGETS = ["w2-centrality", "w2-pagerank", "w2-wl", "w2-louvain",
                    "w3-walks", "w3-pq", "w3-embed", "w3-labelprop",
                    "w4-transe", "w4-patterns", "w4-negatives", "w4-rank",
                    "w5-query", "w5-boxes", "w5-rag", "w5-extract",
                    "w7-agg", "w7-sage", "w7-gat", "w7-ablation"];

export function listFigures() {
  return readdirSync(FIGDIR).filter((f) => f.endsWith(".html")).map((f) => f.replace(/\.html$/, ""))
    .concat(STATIC_WIDGETS, JS_WIDGETS);
}

export function harnessFor(name, theme) {
  const dir = /^w\d+-/.test(name) ? WIDGETDIR : FIGDIR;
  const frag = stripFences(readFileSync(join(dir, `${name}.html`), "utf8"));
  const vars = Object.entries(THEMES[theme]).map(([k, v]) => `${k}: ${v};`).join(" ");
  const scripts = JS_WIDGETS.includes(name)
    ? ["d3.v7.min.js", "_dkr.js", `${name}.js`]
        .map((f) => `<script src="${pathToFileURL(join(WIDGETDIR, f)).href}"></script>`).join("")
    : "";
  return `<!doctype html><html><head><meta charset="utf-8"><style>
    :root { ${vars} }
    body { margin: 0; padding: 16px; background: var(--dkr-bg); font-family: sans-serif; }
    figure { margin: 0 auto; max-width: 720px; }
    svg { width: 100%; height: auto; display: block; }
    figcaption, .fig-caption { font-size: 13px; color: var(--dkr-muted); text-align: center; }
    /* the site themes force mono text inside widgets — lint must measure that reality */
    .interactive-container svg text { font-family: "JetBrains Mono", Consolas, monospace; }
  </style></head><body>${frag}${scripts}</body></html>`;
}

export async function withBrowser(fn) {
  const browser = await puppeteer.launch({
    executablePath: chromePath(),
    headless: "new",
    args: ["--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=1"],
  });
  try {
    return await fn(browser);
  } finally {
    await browser.close();
  }
}

export function ensureQaDir() {
  mkdirSync(QADIR, { recursive: true });
}

export function writeHarness(name, theme, html) {
  ensureQaDir();
  const p = join(QADIR, `${name}-${theme}.html`);
  writeFileSync(p, html);
  return pathToFileURL(p).href;
}
