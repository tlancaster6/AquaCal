---
phase: 05-documentation-site
plan: 03
subsystem: documentation
tags: [api-reference, docstrings, sphinx, autodoc]
dependency_graph:
  requires: [05-01]
  provides: [complete-api-reference, docstring-examples, theory-crosslinks]
  affects: [docs]
tech_stack:
  added: []
  patterns: [sphinx-autodoc, rst-api-docs, google-docstrings]
key_files:
  created:
    - docs/api/index.rst
    - docs/api/core.rst
    - docs/api/calibration.rst
    - docs/api/config.rst
    - docs/api/io.rst
    - docs/api/validation.rst
    - docs/api/triangulation.rst
    - docs/api/datasets.rst
  modified:
    - src/aquacal/core/refractive_geometry.py
    - src/aquacal/core/camera.py
    - src/aquacal/calibration/pipeline.py
    - src/aquacal/calibration/interface_estimation.py
    - src/aquacal/validation/reprojection.py
decisions:
  - decision: "Use RST for API pages (not Markdown) since autodoc directives work best in RST"
    rationale: "Research recommendation from Plan 01, confirmed by heavy autodoc usage"
  - decision: "Group API functions by category (e.g., 'Refractive Projection', 'Snell's Law') not flat module dumps"
    rationale: "Improves discoverability and matches user mental models"
  - decision: "Add 3-5 line usage examples to key functions only (not all functions)"
    rationale: "Balances usefulness with maintenance burden; focus on entry points"
  - decision: "Cross-link API docs to theory pages via Sphinx :doc: directive"
    rationale: "Connects practical API reference to conceptual understanding"
metrics:
  duration: 970s
  tasks_completed: 3
  files_modified: 13
  completed_date: 2026-02-15
---

# Phase 05 Plan 03: API Reference and Docstrings Summary

**One-liner:** Complete API reference pages with autodoc, usage examples for 5 key functions, and cross-links to theory guides.

## What Was Built

Created comprehensive API reference documentation using Sphinx autodoc:

**API Reference Structure:**
- Landing page (docs/api/index.rst) with 7 subpackage sections
- Core module grouped by functionality (projection, Snell's law, camera models, board, interface)
- Calibration module organized by pipeline stages (intrinsics, extrinsics, interface, refinement)
- Config, io, validation, triangulation, datasets modules with full autodoc coverage

**Docstring Improvements:**
- Added 3-5 line usage examples to 5 priority functions:
  - `refractive_project()` / `refractive_project_batch()` / `snells_law_3d()`
  - `Camera` class
  - `run_calibration()`
- Added cross-reference notes linking API to theory pages:
  - Refractive functions → /guide/refractive_geometry
  - Camera class → /guide/coordinates
  - Pipeline functions → /guide/optimizer
  - Validation functions → /guide/refractive_geometry

**Integration:**
- Removed old placeholder index.md
- API pages integrate with main docs toctree
- Build succeeds with no errors (warnings for non-existent guide pages expected until Plan 02 completes)
- All 572 tests pass

## Tasks Completed

| Task | Commit | Files | Description |
|------|--------|-------|-------------|
| 1 | 09c557a | 9 | Created API reference RST pages with autodoc directives |
| 2a | 7dc5671 | 4 | Improved docstrings for core and calibration modules |
| 2b | 29ffca1 | 1 | Improved docstrings for io, validation, and other modules |

## Deviations from Plan

None - plan executed exactly as written.

**Note on Plan 05-02 overlap:**
During execution, Plan 05-02 (theory pages) was completed by another agent, which included adding a usage example to `CalibrationResult`. This was already part of our Plan 03 Task 2b priority list, so we confirmed it was present rather than duplicating work.

## Verification Completed

- [x] `python -m sphinx docs docs/_build -b html` builds successfully
- [x] API landing page lists all 7 subpackages with descriptions
- [x] Core API page shows `refractive_project()` with usage example and theory link
- [x] Calibration API page shows pipeline stages in order
- [x] Config API page shows all dataclasses with field documentation
- [x] `python -m pytest tests/ -m "not slow" -x` passes (572 tests)
- [x] 5+ key functions have inline usage examples visible in rendered docs
- [x] Cross-reference links to theory pages work (warnings expected for non-existent pages)

## Success Criteria Met

- [x] API reference auto-generated from docstrings via autodoc
- [x] All public API has complete docstrings
- [x] Key functions (refractive_project, snells_law_3d, Camera, run_calibration) have 3-5 line usage examples
- [x] Cross-links to theory pages work (Sphinx :doc: directives in place)
- [x] Tests pass (no docstring-related breakage)

## Self-Check: PASSED

**Created files verified:**
- [x] docs/api/index.rst exists
- [x] docs/api/core.rst exists
- [x] docs/api/calibration.rst exists
- [x] docs/api/config.rst exists
- [x] docs/api/io.rst exists
- [x] docs/api/validation.rst exists
- [x] docs/api/triangulation.rst exists
- [x] docs/api/datasets.rst exists

**Commits verified:**
- [x] 09c557a exists (Task 1)
- [x] 7dc5671 exists (Task 2a)
- [x] 29ffca1 exists (Task 2b)

**Modified files verified:**
- [x] src/aquacal/core/refractive_geometry.py has usage examples
- [x] src/aquacal/core/camera.py has usage example
- [x] src/aquacal/calibration/pipeline.py has usage example
- [x] src/aquacal/calibration/interface_estimation.py has cross-link
- [x] src/aquacal/validation/reprojection.py has cross-link

## Impact

**Documentation completeness:** API reference now provides auto-generated, comprehensive documentation for all 7 subpackages. Users can discover functions, understand parameters, and see concrete usage examples.

**Theory integration:** Cross-links bridge the gap between "how to call this function" (API reference) and "why does this work" (theory guides). Users can jump from API docs to detailed explanations.

**Discoverability:** Grouping functions by category (not just module listing) helps users find what they need faster. Example: "I need to project a 3D point" → find it in "Refractive Projection" section.

**Maintenance:** Google-style docstrings with type hints provide single source of truth. Sphinx autodoc ensures API docs stay synchronized with code.

---

*Plan: 05-03*
*Completed: 2026-02-15*
*Duration: 970s (16m 10s)*
