/* Figure Protocol step F2: screenshot every figure in both themes at 760 and 360 px.
 * Output: qa/figshots/<name>-<theme>-<width>.png (git-ignored).
 * Usage: node scripts/figshot.mjs [fig-name ...]        (default: all figures) */
import { join } from "node:path";
import { listFigures, harnessFor, withBrowser, writeHarness, QADIR } from "./figqa.mjs";

const targets = process.argv.slice(2).length ? process.argv.slice(2) : listFigures();
const WIDTHS = [760, 360];
const THEMES = ["light", "dark"];

await withBrowser(async (browser) => {
  const page = await browser.newPage();
  for (const name of targets) {
    for (const theme of THEMES) {
      const url = writeHarness(name, theme, harnessFor(name, theme));
      for (const width of WIDTHS) {
        await page.setViewport({ width, height: 200 });
        await page.goto(url, { waitUntil: "networkidle0" });
      // wait for the real webfonts: Linux CI fallback fonts are wider and
      // produced phantom overflow findings on the first CI run (2026-08-26)
      await page.evaluateHandle("document.fonts.ready");
        // let lazyBoot-ed widgets finish their first render
        await page.evaluate(() => new Promise((r) => setTimeout(r, 180)));
        const height = await page.evaluate(() => Math.ceil(document.body.scrollHeight));
        await page.setViewport({ width, height: Math.min(height + 8, 2200) });
        const out = join(QADIR, `${name}-${theme}-${width}.png`);
        await page.screenshot({ path: out, fullPage: false });
        console.log(`shot ${name} ${theme} ${width}px`);
      }
    }
  }
});
console.log("done");
