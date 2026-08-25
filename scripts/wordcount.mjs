/* Measure each lecture's visible word count the way the site's reading-time
 * plugin does (main.innerText). Usage: node scripts/wordcount.mjs [slug ...] */
import puppeteer from "puppeteer-core";
import { existsSync } from "node:fs";

const chrome = ["C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "/usr/bin/google-chrome"].find(existsSync);
const slugs = process.argv.slice(2).length ? process.argv.slice(2) :
  ["01-why-graphs", "02-classical-graph-ml", "03-embeddings",
   "04-knowledge-graphs", "05-kg-reasoning", "06-gcn", "07-gnn-design",
   "09-expressiveness", "10-hetero-rgcn", "11-scaling", "12-link-generation", "13-transformers",
   "14-production"];

const browser = await puppeteer.launch({ executablePath: chrome, headless: "new", args: ["--disable-gpu"] });
const page = await browser.newPage();
for (const slug of slugs) {
  await page.goto(`http://localhost:8765/lectures/${slug}.html`, { waitUntil: "load", timeout: 60000 });
  await page.waitForSelector(".reading-meta", { timeout: 10000 });
  // read the badge itself — the number the instructor and students see
  const n = await page.evaluate(() =>
    parseInt(document.querySelector(".reading-meta").textContent.match(/([\d,]+) words/)[1].replace(/,/g, ""), 10));
  // Band per instructor 2026-08-25: floor 8,000, ceiling ~10,000 (book pass).
  const ok = n >= 8000 && n <= 10000 ? "✓" : n < 8000 ? `needs +${8000 - n}` : "over";
  console.log(`${slug}: ${n} visible words  ${ok}`);
}
await browser.close();
