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
