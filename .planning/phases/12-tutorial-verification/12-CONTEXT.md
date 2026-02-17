# Phase 12: Tutorial Verification - Context

**Gathered:** 2026-02-17
**Status:** Ready for planning

<domain>
## Phase Boundary

Run all Jupyter tutorials end-to-end, verify they execute without errors, restructure from 3 tutorials to 2, rewrite the synthetic validation tutorial to absorb three experiments from `test_refractive_comparison.py`, and ensure a clear learning path. Delete the old test file and update the index.

</domain>

<decisions>
## Implementation Decisions

### Tutorial restructuring
- Merge tutorial 02 (Diagnostics) content into tutorial 01 (Full Pipeline)
- Delete tutorial 02 as a standalone notebook
- Renumber: current 03 becomes 02 (clean sequential numbering)
- Final structure: **01** (pipeline + diagnostics) and **02** (synthetic validation)
- Curated subset of diagnostics content — Claude picks what's most valuable for a "run and diagnose" pipeline tutorial flow

### Tutorial 02 rewrite (formerly 03)
- Replace existing content entirely — the 3 experiments from the brief become the full notebook
- Progressive narrative: each experiment builds on the previous ("refraction matters" → "it matters more at depth" → "here's how accuracy scales")
- Follow the brief exactly for which plots to keep/drop; Claude may add a summary visualization if it strengthens the narrative
- RIG_SIZE toggle with "small" and "large" presets, consistent with existing pattern
- Pre-execute with **large** preset for production (compelling results); use small preset during development/testing
- Source experiments from `experiments.py` and `experiment_helpers.py` (keep those files)
- Delete `tests/synthetic/test_refractive_comparison.py` after conversion

### Verification approach
- "Passing" = all cells execute without errors (no numeric spot-checks)
- Clear all existing outputs and re-run from scratch for both tutorials
- Development: quick test with small preset; production: full run with large preset, wait for completion

### Learning path
- Practical → Scientific arc: Tutorial 01 is "here's how to calibrate your rig" (practical workflow + diagnostics), Tutorial 02 is "here's why it works and how accurate it is" (validation/science)
- Tutorial 01 ends with a forward link to Tutorial 02 as "next step"
- Index page structure and tutorial titles at Claude's discretion
- Update `docs/tutorials/index.md` to reflect new 2-tutorial structure

### Claude's Discretion
- Which diagnostics from tutorial 02 to keep in the merged tutorial 01
- Tutorial titles (can be more inviting than current names)
- Index page format (list vs brief guide)
- Whether to add a summary comparison visualization across all 3 experiments
- Markdown explanation depth between notebook cells

</decisions>

<specifics>
## Specific Ideas

- The experiment-to-tutorial conversion brief (`.planning/experiment-to-tutorial-brief.md`) has exact specifications for which experiments, plots, and files are involved
- Progressive story across experiments: "refraction matters" → "depth generalization" → "depth scaling"
- Large preset = 13 cameras, 30 frames, full depth sweep (for pre-executed outputs)
- Small preset = 4 cameras, 10 frames, 3 depths (for development testing)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 12-tutorial-verification*
*Context gathered: 2026-02-17*
