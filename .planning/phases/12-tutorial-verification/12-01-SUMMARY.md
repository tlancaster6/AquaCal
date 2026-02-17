---
phase: 12-tutorial-verification
plan: 01
subsystem: docs
tags: [jupyter, tutorials, notebooks, diagnostics, restructure]

requires:
  - phase: 11-documentation-visuals
    provides: "documentation site with visual aids and style guide"

provides:
  - "2-tutorial structure replacing old 3-tutorial layout"
  - "Tutorial 01 with merged diagnostics section (reprojection error, interface distance recovery, 3D error, checklist)"
  - "Tutorial 02 (renamed from 03) ready for rewrite in Plan 02"
  - "Updated tutorial index with accurate descriptions and toctree"

affects:
  - 12-02 (tutorial execution and rewrite)
  - documentation-site

tech-stack:
  added: []
  patterns:
    - "Diagnostics integrated into pipeline tutorial: calibrate then immediately diagnose in same notebook"
    - "Notebook outputs cleared before committing: re-executed cleanly in Plan 02"

key-files:
  created: []
  modified:
    - docs/tutorials/01_full_pipeline.ipynb
    - docs/tutorials/02_synthetic_validation.ipynb
    - docs/tutorials/index.md

key-decisions:
  - "Diagnostics merged into tutorial 01 (not standalone): practical flow of calibrate-then-diagnose"
  - "Tutorial 02 renamed from 03 unchanged: content rewrite deferred to Plan 02"
  - "Old 02_diagnostics.ipynb deleted: content subsumed by merged sections in tutorial 01"

patterns-established:
  - "Tutorial arc: practical workflow (01) then scientific validation (02)"

duration: 12min
completed: 2026-02-17
---

# Phase 12 Plan 01: Tutorial Restructure Summary

**3-to-2 tutorial restructure: diagnostics merged into pipeline tutorial, synthetic validation renamed, index updated with practical-to-scientific arc descriptions**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-02-17
- **Completed:** 2026-02-17
- **Tasks:** 2 completed
- **Files modified:** 3 (plus 1 deleted, 1 renamed)

## Accomplishments

- Merged 5 diagnostic sections into tutorial 01: per-camera RMS bar chart, reprojection error histogram, interface distance recovery comparison, 3D reconstruction error histogram, and common issues checklist table
- Added a calibration run (Stages 2-3 via `generate_synthetic_detections`) in the diagnostics section so plots show real optimization results rather than placeholder text
- Deleted `docs/tutorials/02_diagnostics.ipynb` and renamed `03_synthetic_validation.ipynb` to `02_synthetic_validation.ipynb`
- Updated tutorial 01 summary cell: forward link to `02_synthetic_validation.ipynb`, removed stale links to old tutorials
- Cleared all 33 cell outputs from tutorial 01 (ready for re-execution in Plan 02)
- Rewrote tutorial index for 2-tutorial structure with practical descriptions and corrected toctree

## Task Commits

1. **Task 1: Merge diagnostics + restructure files** - `695cf8a` (feat)
2. **Task 2: Update tutorial index** - `0d89566` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `docs/tutorials/01_full_pipeline.ipynb` - Extended from 22 to 33 cells; added Diagnostics section after Stage 3; outputs cleared; summary cell updated
- `docs/tutorials/02_synthetic_validation.ipynb` - Renamed from `03_synthetic_validation.ipynb`; content unchanged
- `docs/tutorials/index.md` - Rewritten for 2-tutorial structure; stale Phase 6 note removed; toctree corrected

**Deleted:**
- `docs/tutorials/02_diagnostics.ipynb` - Content subsumed into tutorial 01

## Decisions Made

- Diagnostics merged into tutorial 01 rather than kept standalone: creates a natural calibrate-then-diagnose workflow in a single notebook
- Tutorial 02 content kept unchanged for now: full rewrite deferred to Plan 02 where it will be restructured around three experiments (parameter fidelity, depth generalization, accuracy scaling)
- Selected 5 of 8 diagnostic cells: excluded spatial heatmap (cell 8), camera Z position comparison (cell 11), multi-panel 3D visualization (cell 13) per plan guidance; these are more useful for deep debugging than a pipeline walkthrough

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- Pre-commit `end-of-file-fixer` hook modified the notebook on first commit attempt (missing trailing newline); resolved by re-staging the hook-fixed file and committing again. Standard workflow behavior.

## Next Phase Readiness

- Ready for Plan 02: tutorial 01 re-execution and tutorial 02 full rewrite
- Tutorial 01 has 33 cells with cleared outputs; will be executed end-to-end in Plan 02
- Tutorial 02 (`02_synthetic_validation.ipynb`) contains the old synthetic validation content; Plan 02 will rewrite it around the three-experiment structure

---
*Phase: 12-tutorial-verification*
*Completed: 2026-02-17*

## Self-Check: PASSED

- FOUND: `docs/tutorials/01_full_pipeline.ipynb` (33 cells, all outputs cleared)
- FOUND: `docs/tutorials/02_synthetic_validation.ipynb` (renamed from 03)
- FOUND: `docs/tutorials/index.md` (2-tutorial toctree)
- FOUND: `.planning/phases/12-tutorial-verification/12-01-SUMMARY.md`
- CONFIRMED: `695cf8a` — task 1 commit (merge diagnostics + restructure)
- CONFIRMED: `0d89566` — task 2 commit (update index)
- CONFIRMED: `9e7c230` — metadata commit
