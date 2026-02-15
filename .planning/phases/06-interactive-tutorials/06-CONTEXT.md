# Phase 6: Image Input Support, Interactive Tutorials & README Overhaul - Context

**Gathered:** 2026-02-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver three capabilities: (1) abstract frame loading to support image directories alongside video files via a FrameSet interface, (2) Jupyter notebook tutorials demonstrating end-to-end calibration workflows integrated into Sphinx docs, and (3) a concise README with visual assets linking to docs for details. The pipeline, documentation site, and example data infrastructure already exist from prior phases.

</domain>

<decisions>
## Implementation Decisions

### Tutorial content & depth
- Audience: researchers who know Python and OpenCV basics but are new to AquaCal
- Each notebook is self-contained — no required reading order, some setup repetition is acceptable
- Failure modes covered via inline warning callouts where relevant, not separate sections
- Data source: runtime-toggleable — cell near top lets user switch between synthetic, in-package preset, or Zenodo download
- Rich visualization emphasis: 3D camera rig plots, ray traces, error heatmaps, before/after comparisons
- Diagnostics notebook covers reprojection error analysis, parameter convergence, AND 3D rig visualization
- Synthetic validation notebook focuses on refractive vs non-refractive comparison (set n_air=n_water=1.0) — interactive extension of tests/synthetic/experiments.py, showing users first-hand why refractive calibration matters

### Image input behavior
- Auto-detect input type: directory = images, file = video — no config flag needed
- Supported image formats: JPEG and PNG only
- Image ordering: alphabetical/natural sort by filename
- No subsampling for image directories — assumed pre-curated; subsampling only applies to video

### README structure & visuals
- Hero visual: ray trace visualization (Snell's law refraction diagram — immediately communicates "refractive")
- Quick start: 3-step minimal (pip install, generate config, run calibrate)
- Short feature list: 4-5 bullets covering key capabilities
- Citation: brief one-liner in README with link to docs for full BibTeX and details
- Bulk content (CLI reference, config reference, methodology, output details) moves to docs with links

### Notebook integration
- Pre-rendered outputs committed with notebooks — fast doc builds, curated outputs, no build-time deps
- Notebooks live in docs/tutorials/ — picked up naturally by nbsphinx
- "Tutorials" as top-level section in docs sidebar alongside Theory, API Reference
- Google Colab badge on each notebook for interactive cloud use

### Claude's Discretion
- Narrative verbosity per notebook — Claude picks appropriate balance of explanation vs code
- Exact matplotlib styling and plot layouts
- Cell structure and markdown formatting within notebooks
- How the data source toggle is implemented (config dict, enum, etc.)

</decisions>

<specifics>
## Specific Ideas

- Synthetic validation notebook should mirror/extend tests/synthetic/experiments.py — the refractive vs non-refractive comparison shows users first-hand when the refractive model matters vs when the approximation is "good enough"
- Runtime data source toggle: a cell near the top of each notebook where user sets `DATA_SOURCE = "synthetic"` (or `"preset"` or `"zenodo"`) and the rest adapts
- User has experiment result plots at C:\Users\tucke\Desktop\results that could inform the refractive comparison visualizations

</specifics>

<deferred>
## Deferred Ideas

- Dedicated docs page proving necessity of refractive model (when it matters vs when approximation is good enough) using experiment output plots — new documentation capability beyond Phase 6 scope
- TIFF and other scientific image format support — keep it simple with JPEG+PNG for now

</deferred>

---

*Phase: 06-interactive-tutorials*
*Context gathered: 2026-02-15*
