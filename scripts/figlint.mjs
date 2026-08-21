/* Figure Protocol step F3: automated geometry lint over every figure, both themes.
 * All geometry is measured in SCREEN space (getBoundingClientRect), so ancestor
 * transforms are handled correctly. Checks per figure:
 *   overflow    — no rendered element escapes the SVG's own screen box (3 px tol)
 *   collision   — no two text elements' screen boxes materially intersect
 *   band-clip   — a text whose center sits in a filled rounded band must fit in it
 *   fontsize    — rendered font size ≥ 11 px at 760 px page width
 *   a11y        — svg[aria-label] required (failure); missing caption is a note
 * Usage: node scripts/figlint.mjs [fig-name ...]   → exit 1 on any failure */
import { listFigures, harnessFor, withBrowser, writeHarness } from "./figqa.mjs";

const targets = process.argv.slice(2).length ? process.argv.slice(2) : listFigures();
let failures = 0;

await withBrowser(async (browser) => {
  const page = await browser.newPage();
  await page.setViewport({ width: 760, height: 900 });
  for (const name of targets) {
    for (const theme of ["light", "dark"]) {
      const url = writeHarness(name, theme, harnessFor(name, theme));
      await page.goto(url, { waitUntil: "networkidle0" });
      const { problems, notes } = await page.evaluate(() => {
        const problems = [], notes = [];
        const svg = document.querySelector("svg");
        if (!svg) return { problems: ["no <svg> found"], notes };
        if (!svg.getAttribute("aria-label")) problems.push("a11y: svg lacks aria-label");
        if (!document.querySelector("figcaption, .fig-caption"))
          notes.push("note: no caption element in the include (caption must exist in surrounding prose)");

        const S = svg.getBoundingClientRect();
        const TOL = 3;
        const els = Array.from(svg.querySelectorAll("circle, rect, text, line, path, ellipse"));
        const rect = (el) => el.getBoundingClientRect();
        const label = (el) => (el.textContent || "").trim().slice(0, 30) || el.tagName;

        for (const el of els) {
          const b = rect(el);
          if (b.width === 0 && b.height === 0) continue;
          if (b.left < S.left - TOL || b.top < S.top - TOL ||
              b.right > S.right + TOL || b.bottom > S.bottom + TOL) {
            problems.push(`overflow: "${label(el)}" escapes the canvas by ` +
              `${Math.round(Math.max(S.left - b.left, b.right - S.right, S.top - b.top, b.bottom - S.bottom))}px`);
          }
        }

        const texts = els.filter((e) => e.tagName === "text")
          .map((t) => ({ t, b: rect(t) })).filter(({ b }) => b.width > 0);
        for (let i = 0; i < texts.length; i++) {
          for (let j = i + 1; j < texts.length; j++) {
            const A = texts[i], B = texts[j];
            const ox = Math.min(A.b.right, B.b.right) - Math.max(A.b.left, B.b.left);
            const oy = Math.min(A.b.bottom, B.b.bottom) - Math.max(A.b.top, B.b.top);
            if (ox > 3 && oy > Math.min(A.b.height, B.b.height) * 0.35) {
              problems.push(`collision: "${label(A.t)}" × "${label(B.t)}"`);
            }
          }
        }

        const bands = els.filter((e) => e.tagName === "rect" && +e.getAttribute("rx") > 0 &&
          e.getAttribute("fill") && !["none", "transparent"].includes(e.getAttribute("fill")));
        for (const { t, b } of texts) {
          const cx = (b.left + b.right) / 2, cy = (b.top + b.bottom) / 2;
          for (const band of bands) {
            const r = rect(band);
            if (!(cx > r.left && cx < r.right && cy > r.top && cy < r.bottom)) continue;
            const PAD = 6; // bands must give their text breathing room, not just contain it
            if (b.left < r.left + PAD - 1 || b.right > r.right - PAD + 1) {
              problems.push(`band-clip: "${label(t)}" too wide for its band ` +
                `(text ${Math.round(b.width)}px vs band ${Math.round(r.width)}px, need ${PAD}px padding each side)`);
            }
          }
        }

        // edge-clip: a text whose center is OUTSIDE a stroked shape but whose box
        // cuts into it (the "label clips the box corner" bug class)
        const strokedRects = els.filter((e) => e.tagName === "rect" &&
          e.getAttribute("stroke") && e.getAttribute("stroke") !== "none");
        for (const { t, b } of texts) {
          const cx = (b.left + b.right) / 2, cy = (b.top + b.bottom) / 2;
          for (const shp of strokedRects) {
            const r = rect(shp);
            const centerIn = cx > r.left && cx < r.right && cy > r.top && cy < r.bottom;
            if (centerIn) continue; // that's band territory, handled above
            const ox = Math.min(b.right, r.right) - Math.max(b.left, r.left);
            const oy = Math.min(b.bottom, r.bottom) - Math.max(b.top, r.top);
            if (ox > 3 && oy > 3) {
              problems.push(`edge-clip: "${label(t)}" cuts into a box edge by ${Math.round(Math.min(ox, oy))}px`);
            }
          }
        }

        for (const { t } of texts) {
          const fs = parseFloat(getComputedStyle(t).fontSize) *
            (S.width / svg.viewBox.baseVal.width);
          if (fs < 11) problems.push(`fontsize: "${label(t)}" renders at ${fs.toFixed(1)}px`);
        }
        return { problems: [...new Set(problems)], notes };
      });
      if (problems.length) {
        failures += problems.length;
        console.log(`✗ ${name} [${theme}]`);
        problems.forEach((p) => console.log(`    ${p}`));
      } else {
        console.log(`✓ ${name} [${theme}]${notes.length ? "  (" + notes.length + " note)" : ""}`);
      }
    }
  }
});
console.log(failures ? `\n${failures} problem(s)` : "\nall figures pass lint");
process.exit(failures ? 1 : 0);
