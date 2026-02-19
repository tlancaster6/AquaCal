---
phase: 14-geometry-rewiring
plan: 03
subsystem: core, triangulation
tags: [aquakit, bridge, refraction, rewiring, public-api]

# Dependency graph
requires:
  - phase: 14-02
    provides: All refractive_project call sites rewired; fast shims deleted; originals marked DEPRECATED

provides:
  - refractive_back_project call site in triangulate.py routes through _bridge_refractive_back_project
  - core/__init__.py exports bridge-backed versions of all 5 geometry functions under original public names
  - AquaKit InterfaceParams NOT leaked into public API (schema InterfaceParams unaffected)
  - refractive_project_batch removed from public API
affects: [15-equivalence, 16-cleanup, 17-release]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Public API aliasing: bridge functions exported under original names (e.g., _bridge_refractive_project as refractive_project) for zero-breaking-change API surface"
    - "Per-camera _make_interface_params() factory inside triangulation loop (each camera has its own water_z)"

key-files:
  created: []
  modified:
    - src/aquacal/triangulation/triangulate.py
    - src/aquacal/core/__init__.py

key-decisions:
  - "core/__init__.py uses aliased bridge imports (_bridge_X as X) so external callers are unaffected"
  - "AquaKit InterfaceParams deliberately excluded from core/__init__.py to prevent silent type collision with aquacal.config.schema.InterfaceParams"
  - "refractive_project_batch removed from public API (fast shims were deleted in 14-02; no bridge batch equivalent yet)"
  - "triangulate.py moves _make_interface_params() call inside the per-camera loop (was: one shared Interface for all cameras)"

# Metrics
duration: 2min
completed: 2026-02-19
---

# Phase 14 Plan 03: Back-projection Rewiring and Public API Bridge Summary

**All 5 geometry functions route through AquaKit bridge at every internal call site and at the public aquacal.core API boundary, with zero breaking changes to external callers**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-02-19T17:59:55Z
- **Completed:** 2026-02-19T18:02:00Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Rewired `triangulate_point()` to use `_bridge_refractive_back_project` + `_make_interface_params()` per camera — no more `Interface` object construction, no more direct `refractive_geometry` import
- Updated `core/__init__.py` to export all 5 geometry functions via bridge aliases under their original public names — external code using `from aquacal.core import refractive_project` continues to work unchanged
- AquaKit's `InterfaceParams` type deliberately excluded from the public API (schema `InterfaceParams` remains the only one visible in `aquacal.core`)
- Removed `refractive_project_batch` from the public API (fast shims deleted in Plan 02)

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewire refractive_back_project in triangulate.py** - `8b85d87` (feat)
2. **Task 2: Update core/__init__.py to export bridge functions** - `97cee73` (feat)

**Plan metadata:** see final docs commit below

## Files Created/Modified

- `src/aquacal/triangulation/triangulate.py` - Uses `_bridge_refractive_back_project` + `_make_interface_params()` per-camera; `Interface` import and `camera_distances` dict removed
- `src/aquacal/core/__init__.py` - All 5 geometry functions re-exported via bridge aliases; `refractive_project_batch` removed; no AquaKit types leaked

## Decisions Made

- `core/__init__.py` uses the aliased import pattern (`_bridge_refractive_project as refractive_project`) so the public API surface is unchanged for external callers
- AquaKit `InterfaceParams` is NOT exported from `aquacal.core` — the schema type of the same name would silently conflict for any caller also importing from `aquacal.config.schema`
- `refractive_project_batch` removed from `__all__` — the fast shim was deleted in Plan 02, and no bridge batch function has been designed yet (deferred to Phase 16/17)
- `triangulate.py` moves the `_make_interface_params()` call inside the per-camera loop because each camera has its own `water_z` value (matches the pattern established in `pipeline.py` during Plan 02)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Ruff I001 import ordering in triangulate.py**
- **Found during:** Task 1 (initial ruff check after edit)
- **Issue:** Ruff flagged I001 (import block un-sorted) because the new bridge import was placed in the middle of the local import block
- **Fix:** Applied `python -m ruff check --fix` to auto-sort; ruff reformatted the multi-name bridge import into a single parenthesized block
- **Files modified:** `src/aquacal/triangulation/triangulate.py`
- **Verification:** `python -m ruff check src/aquacal/triangulation/triangulate.py` passes
- **Committed in:** `8b85d87`

**2. [Rule 3 - Blocking] Ruff I001 import ordering in core/__init__.py**
- **Found during:** Task 2 (initial ruff check after edit)
- **Issue:** Ruff flagged I001 for the bridge import block; ruff auto-fix additionally split the single multi-alias import into five separate per-alias imports (alphabetical sort requirement)
- **Fix:** Applied `python -m ruff check --fix`; result is five single-alias imports — functionally identical
- **Files modified:** `src/aquacal/core/__init__.py`
- **Verification:** `python -m ruff check src/aquacal/core/__init__.py` passes
- **Committed in:** `97cee73`

---

**Total deviations:** 2 auto-fixed (2 Rule 3 - blocking lint/format)
**Impact on plan:** Both fixes required for code style compliance. No scope creep.

## Issues Encountered

- AquaKit/torch not installed locally — import-level verification done via AST parse and pattern grep rather than live Python imports. Expected; CI has torch/aquakit installed.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Phase 14 geometry rewiring is **complete**: all 5 geometry functions route through AquaKit bridge at every internal call site and at the public API boundary
- Original implementations remain in `refractive_geometry.py` and `interface_model.py` (marked `# DEPRECATED`) for Phase 15 equivalence testing
- Phase 15 (equivalence testing) can proceed to verify AquaKit bridge produces numerically equivalent results to the original implementations before Phase 16 deletes the originals

## Self-Check: PASSED

- `src/aquacal/triangulation/triangulate.py` — FOUND
- `src/aquacal/core/__init__.py` — FOUND
- `.planning/phases/14-geometry-rewiring/14-03-SUMMARY.md` — FOUND
- Commit `8b85d87` (Task 1) — FOUND
- Commit `97cee73` (Task 2) — FOUND
- `python -m ruff check src/aquacal/core/ src/aquacal/triangulation/` — PASSED
- No `from aquacal.core.refractive_geometry import refractive_back_project` remaining — PASSED
- `refractive_project_batch` removed from `core/__init__.py` — PASSED

---
*Phase: 14-geometry-rewiring*
*Completed: 2026-02-19*
