# DKR — Data and Knowledge Representation

Course materials for **Data and Knowledge Representation** (Innopolis University, Fall 2026):
a 15-week course on representing, learning from, and reasoning over graphs and knowledge graphs —
from graph statistics and shallow embeddings through GNNs, KG embeddings and reasoning, graph
transformers, and production systems (recommenders, relational deep learning, GraphRAG).

**Course site:** https://lukmanovr.github.io/dkr/ (built with [Quarto](https://quarto.org), deployed by GitHub Actions)

## Repository layout

```
_quarto.yml            Quarto site configuration (theme, navigation, formats)
lectures/              15 lecture pages (.qmd) — each also renders a revealjs slide deck
labs/                  15 Colab lab notebooks (student versions; "Open in Colab" badges)
homeworks/             Homework notebooks (student versions) + rubrics
project/               Course project: requirements, milestones, rubric, topic list
exams/                 Exam information + public sample exams with solutions
assets/d3/             Interactive d3 widgets (one self-contained JS file each) + vendored d3
assets/figures/        Static figures (SVG/PNG)
docs/                  Course design document, notation reference, contributor notes
scripts/               Build/QA tooling (notebook smoke tests, link checks, solution stripping)
.github/workflows/     CI: site build + deploy, notebook execution smoke tests
```

Instructor-only material (exam problem banks, lab/homework solution notebooks, answer keys)
lives in a separate **private** repository and is never part of this repo or the site.

## Building the site locally

Requirements: [Quarto CLI](https://quarto.org/docs/get-started/) ≥ 1.5, Python 3.11+.

```bash
make site        # quarto render  -> _site/
make preview     # quarto preview (live-reloading dev server)
make test        # execute all lab/homework notebooks headlessly (SMOKE=1: reduced epochs)
make linkcheck   # verify all external links resolve
```

## Course facts (Fall 2026)

- 15 weeks, Thursdays, 2026-08-27 → 2026-12-03; one 90-minute lecture + one lab session per week
- Midterm: week 8 (Oct 15, written, in the lecture slot). Final: week 15 (Dec 3, written)
- Grading: Labs 15% · Homeworks 30% (3×10%) · Midterm 15% · Project 20% · Final 20%
- Labs run top-to-bottom on free-tier Google Colab with PyTorch Geometric

## License and attribution

License to be finalized before publication (proposed: CC BY-NC-SA 4.0 for prose/figures, MIT for code —
pending instructor sign-off at the design-review gate). Several lab exercises are inspired by
Stanford CS224W and the PyTorch Geometric tutorials; such derivations are attributed in place.
