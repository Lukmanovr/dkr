// Screenshot a rendered site page in light and dark themes.
// Usage: node shot_page.mjs <path-under-_site> <outprefix>
import puppeteer from "puppeteer-core";
import { existsSync } from "node:fs";

const ROOT = "e:/DKR_course_materials/dkr";
const [, , rel, prefix] = process.argv;

const CHROME = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  process.env.LOCALAPPDATA + "/Google/Chrome/Application/chrome.exe",
].find(existsSync);

const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new" });
const page = await browser.newPage();
await page.setViewport({ width: 900, height: 1400, deviceScaleFactor: 1 });
const url = "file:///" + ROOT + "/_site/" + rel;
await page.goto(url, { waitUntil: "networkidle0", timeout: 60000 });
await new Promise((r) => setTimeout(r, 1200)); // KaTeX
const out = ROOT + "/qa/figshots/" + prefix;
await page.screenshot({ path: out + "-light.png", fullPage: true });

// flip to dark via quarto's toggle
const toggled = await page.evaluate(() => {
  const t = document.querySelector(".quarto-color-scheme-toggle, a.quarto-color-scheme-toggle");
  if (t) { t.click(); return true; }
  return false;
});
await new Promise((r) => setTimeout(r, 800));
await page.screenshot({ path: out + "-dark.png", fullPage: true });
console.log("shot", prefix, "toggled:", toggled);
await browser.close();
