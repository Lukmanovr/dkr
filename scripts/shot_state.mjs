// One-off: open a saved qa harness page, click a selector, screenshot.
// Usage: node shot_state.mjs <harness-html-name> <click-selector> <out-name>
import puppeteer from "puppeteer-core";
import { existsSync } from "node:fs";
const CHROME = ["C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  process.env.LOCALAPPDATA + "/Google/Chrome/Application/chrome.exe"].find(existsSync);
const [, , harness, sel, out] = process.argv;
const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new" });
const page = await browser.newPage();
await page.setViewport({ width: 792, height: 700 });
await page.goto("file:///e:/DKR_course_materials/dkr/qa/figshots/" + harness,
  { waitUntil: "networkidle0", timeout: 60000 });
await page.evaluateHandle("document.fonts.ready");
await page.click(sel);
await new Promise((r) => setTimeout(r, 400));
await page.screenshot({ path: "e:/DKR_course_materials/dkr/qa/figshots/" + out, fullPage: true });
console.log("shot", out);
await browser.close();
