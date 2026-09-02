# DKR figure grammar

Every static figure (`assets/figures/*.html`) and widget (`assets/d3/*`) follows this
grammar so a reader can identify a DKR figure at a glance across all fifteen weeks.
Deviations are design bugs. See also Checklist C in `docs/design.md`.

## The cast

Six identity-colored nodes, reused in every conceptual diagram of every week. Identity
colors are data colors — never used for UI chrome, never re-assigned:

| node | color | | node | color |
|---|---|---|---|---|
| A | `#d9a62e` gold | | D | `#7c5cd6` purple |
| B | `#cf4a30` red | | E | `#d1567e` pink |
| C | `#199473` green | | F | `#0f8377` teal |

Cast graph (weeks that need "a" graph use this one): edges
A–B, A–C, A–D, B–C, C–E, C–F, E–F. A is the default target node.

## Geometry

- **Canvas**: `viewBox="0 0 760 H"`; content column max-width 700 px on the page.
- **Nodes**: circles; primary r 15–19, secondary/leaf r 8–11; white bold Source Sans
  labels (font 13–17 primary, 10–11 leaf).
- **Edges (structure)**: solid, `var(--dkr-muted)`, width 1.5–2.6, opacity 0.55–0.75.
- **Messages/dataflow**: dashed `4,4`, width 1.4–1.8, triangular marker arrowheads.
- **Function boxes**: rounded rects rx 8–10, fill `var(--dkr-paper)`,
  stroke `var(--dkr-border)` 1.5–2.2 (or a role color when the box *is* the concept).
- **Matrices**: cell grids stroke `var(--dkr-border)` 1.0; filled cells in a single role
  color at 0.85 opacity.

## Color roles inside figures

Structural colors always via CSS custom properties with light-theme fallbacks —
`var(--dkr-accent, #d9603b)` etc. — so every figure recolors with the theme toggle:

- `--dkr-accent` terracotta: annotations, the current focus, callout bands and chips.
- `--dkr-blue` (carries teal): secondary structure, aggregation, matrices.
- `--dkr-green`: verdict/claim bands (the "green overlay" statement).
- `--dkr-muted`: edges, de-emphasized labels, axis text.
- Identity colors (the cast) are literal hexes — they must not flip with the theme.

## Text in figures

- All figure text is Source Sans 3; JetBrains Mono only for numeric readouts in widgets.
- Annotation labels 13–15 px, weight 600, in the color of the thing they annotate.
- One bottom takeaway line per figure, 13–14 px, `var(--dkr-muted)` — or a filled
  green/accent claim band (rx 10–15, white bold text) when the figure earns a verdict.
- Math inside SVG uses Unicode super/subscripts (h⁽ᵏ⁾ᵤ), never images of LaTeX.

## Captions

`figcaption`/`.fig-caption`: numbered bold prefix (**Figure N.**), 2–4 sentences that
*teach* (what to see, why it matters), and — for widgets — one **Try this:** experiment.

## Files & wrappers

- Static figure: `assets/figures/fig-<name>.html`, a ```{=html} block containing
  `<figure class="dkr-fig"><svg …>…</svg></figure>`, aria-label on the svg.
- Widget: `assets/d3/w6-<name>.js` + matching `.html` include with
  `.interactive-container`, fig-label, controls, `<noscript>` static description,
  caption. First render deferred via `DKR.lazyBoot`; theme reactivity via
  `DKR.onThemeChange`; reduced motion respected via `DKR.motionOK`.

## The Figure Protocol (F1–F5, mandatory for every new or edited figure)

**F1 · Author** to the grammar above. Any content that derives from data — adjacency
patterns, computed values, eigen-colorings — is emitted by a generator in
`scripts/figgen/` (committed beside the figure), never hand-placed.
**F2 · Render**: `node scripts/figshot.mjs [name…]` — screenshots every figure in both
themes (tokens extracted live from the theme SCSS) at 760 px and 360 px into
`qa/figshots/` (git-ignored).
**F3 · Lint**: `node scripts/figlint.mjs [name…]` — screen-space geometry checks:
canvas overflow, text–text collisions, band padding (≥6 px per side), minimum rendered
font size (≥11 px), svg aria-label. Wired into CI; exit 1 on any failure.
**F4 · Look**: the author reads every screenshot (light + dark) and walks Checklist F
(design.md §14) explicitly. The lint catches geometry; only the eye catches meaning.
Any edit re-runs F2–F4.
**F5 · Sign off** (describing the change, never the request that prompted it): the commit lists each touched figure with its lint result and a
one-line visual verdict. No recorded F4 pass → the figure does not ship.
