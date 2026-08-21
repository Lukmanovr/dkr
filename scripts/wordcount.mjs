/* Measure each lecture's visible word count the way the site's reading-time
 * plugin does (main.innerText). Usage: node scripts/wordcount.mjs [slug ...] */
import puppeteer from "puppeteer-core";
import { existsSync } from "node:fs";

const chrome = ["C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "/usr/bin/google-chrome"].find(existsSync);
const slugs = process.argv.slice(2).length ? process.argv.slice(2) :
  ["01-why-graphs", "02-classical-graph-ml", "03-embeddings",
   "04-knowledge-graphs", "05-kg-reasoning", "06-gcn"];

const browser = await puppeteer.launch({ executablePath: chrome, headless: "new", args: ["--disable-gpu"] });
const page = await browser.newPage();
for (const slug of slugs) {
  await page.goto(`http://localhost:8765/lectures/${slug}.html`, { waitUntil: "domcontentloaded", timeout: 60000 });
  await new Promise((r) => setTimeout(r, 400));
  const n = await page.evaluate(() =>
    (document.querySelector("main").innerText || "").trim().split(/\s+/).length);
  const ok = n >= 8000 && n <= 9500 ? "✓" : n < 8000 ? `needs +${8000 - n}` : "over";
  console.log(`${slug}: ${n} visible words  ${ok}`);
}
await browser.close();
