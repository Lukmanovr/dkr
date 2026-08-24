# DKR — Data and Knowledge Representation

Course materials for **Data and Knowledge Representation** (Innopolis University, Fall 2026):
a 15-week course on representing, learning from, and reasoning over graphs and knowledge graphs —
from graph statistics and shallow embeddings through GNNs, KG embeddings and reasoning, graph
transformers, and production systems (recommenders, relational deep learning, GraphRAG).

**Course site:** https://lukmanovr.github.io/dkr/ (built with [Quarto](https://quarto.org), deployed by GitHub Actions)

## Repository layout

```
_quarto.yml            Quarto site configuration (theme, navigation, formats)
lectures/              13 lecture pages (.qmd); weeks 8 and 15 are exam slots, not lectures
labs/                  14 Colab lab notebooks (student versions) + two session pages (8, 15)
homeworks/             Homework notebooks (student versions) + rubrics
project/               Course project: requirements, milestones, rubric, topic list
exams/                 Exam information + public sample exams with solutions
assets/d3/             Interactive d3 widgets (one self-contained JS file each) + vendored d3
assets/figures/        Static figures (SVG/PNG)
docs/                  Course design document, notation reference, contributor notes
scripts/               Build/QA tooling — see "Verification" below
scripts/labgen/        Single-source lab/HW generators (student + private solution notebooks)
scripts/figgen/        Figure generators (every data-derived number asserted in the generator)
scripts/experiments/   The measurement scripts behind every baked number, with results + README
.github/workflows/     CI: site build + deploy, notebook execution smoke tests
```

Instructor-only material (exam problem banks, lab/homework solution notebooks, answer keys)
lives in a separate **private** repository and is never part of this repo or the site.

## Working on this repo from a fresh clone

Requirements: [Quarto CLI](https://quarto.org/docs/get-started/) ≥ 1.5, Python 3.11+,
Node 22+ (for the figure/widget QA tooling), and a local Chrome (figure screenshots).

```bash
git clone https://github.com/lukmanovr/dkr.git
cd dkr
pip install -r labs/requirements.txt      # torch, torch_geometric, nbclient, ...
(cd scripts && npm install)               # puppeteer-core (figure shots) + jsdom (widget smoke)
```

Datasets are **not** committed — every notebook and experiment downloads what it needs
into `data/` on first run (gitignored). Budget for that: Cora/MovieLens seconds, ZINC a
minute, ogbn-arxiv ~90 MB and slow on some networks, so keep the cache once you have it.

On Windows, export `KMP_DUPLICATE_LIB_OK=TRUE` and `PYTHONIOENCODING=utf-8` before
running anything that imports torch.

```bash
make site        # quarto render  -> _site/
make preview     # quarto preview (live-reloading dev server)
make test        # execute all lab/homework notebooks headlessly (SMOKE=1: reduced epochs)
make linkcheck   # verify all external links resolve
```

## Verification

Nothing ships without these; CI runs the same set.

```bash
python scripts/check_no_solutions.py      # no solution content leaked into the public tree
python scripts/check_links.py _site       # every internal + external link resolves
node scripts/figlint.mjs                  # figure geometry: overflow, collisions, font sizes
node scripts/figshot.mjs                  # screenshots both themes at 760/360 -> qa/figshots
node scripts/widget_smoke.mjs             # mounts all widgets, asserts per-week probes
node scripts/wordcount.mjs                # every lecture's on-page badge (standard: >= 8,000)
python scripts/run_notebooks.py labs homeworks   # notebook CI contract
```

`wordcount.mjs` reads the page badge, so serve the rendered site first:
`python -m http.server 8765 --directory _site`.

The figure protocol (F1 author -> F2 shoot -> F3 lint -> F4 read the screenshots ->
F5 sign off in the commit) and the lecture/lab quality checklists are in
[`docs/design.md`](docs/design.md) §14 — read that before adding a week.

Instructor-only material (exam banks, solution notebooks) lives in the separate private
repo; clone it as a **sibling directory** — the lab generators write solutions to
`../dkr-private/solutions/labs/`.

## Course facts (Fall 2026)

- 15 weeks, Thursdays, 2026-08-27 → 2026-12-03; one 90-minute lecture + one lab session per week
- Midterm: week 8 (Oct 15, written, in the lecture slot). Final: week 15 (Dec 3, written)
- Grading: Labs 15% · Homeworks 30% (3×10%) · Midterm 15% · Project 20% · Final 20%
- Labs run top-to-bottom on free-tier Google Colab with PyTorch Geometric

## License and attribution

License to be finalized before publication (proposed: CC BY-NC-SA 4.0 for prose/figures, MIT for code —
pending instructor sign-off at the design-review gate). Several lab exercises are inspired by
Stanford CS224W and the PyTorch Geometric tutorials; such derivations are attributed in place.
