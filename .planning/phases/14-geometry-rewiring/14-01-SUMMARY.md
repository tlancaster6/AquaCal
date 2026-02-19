---
phase: 14-geometry-rewiring
plan: 01
subsystem: core
tags: [aquakit, torch, pytorch, numpy, bridge, refraction, geometry]

# Dependency graph
requires:
  - phase: 13-setup
    provides: aquakit declared as hard dep, torch guard in __init__.py
provides:
  - AquaKit bridge module centralizing all numpy<->torch conversion
  - All 5 geometry bridge wrappers with numpy-in/numpy-out signatures
  - _make_interface_params() factory hiding AquaKit type collision from call sites
  - Private _to_torch() / _to_numpy() helpers for future use by all rewiring plans
affects: [14-02, 14-03, 15-equivalence, 16-cleanup, 17-release]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Bridge pattern: all AquaKit calls go through _aquakit_bridge.py — torch never leaks to call sites"
    - "Factory pattern: _make_interface_params() constructs AquaKit types without exposing type names"
    - "Local import pattern: AquaKit InterfaceParams imported only inside _make_interface_params() to avoid name collision"

key-files:
  created:
    - src/aquacal/core/_aquakit_bridge.py
  modified: []

key-decisions:
  - "_make_interface_params() factory prevents aquakit.types.InterfaceParams name collision with aquacal.config.schema.InterfaceParams — call sites use factory, not direct import"
  - "InterfaceParams excluded from __all__ and only imported locally inside factory function"
  - "_bridge_refractive_project uses two-step approach: AquaKit finds interface point, then camera.project() maps it to pixel"
  - "_bridge_trace_ray_air_to_water and _bridge_refractive_back_project both call camera.pixel_to_ray_world() to get ray before delegating to AquaKit"

patterns-established:
  - "Bridge module pattern: single file owns all torch imports; callers import bridge functions, never torch directly"
  - "Unsqueeze(0)/squeeze(0) pattern: AquaKit operates on batched tensors (N,3); bridge wraps single numpy Vec3 in batch of 1"

# Metrics
duration: 10min
completed: 2026-02-19
---

# Phase 14 Plan 01: AquaKit Bridge Module Summary

**numpy<->torch bridge module with all 5 geometry wrappers and _make_interface_params() factory, isolating torch from the rest of AquaCal**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-02-19T17:45:19Z
- **Completed:** 2026-02-19T17:55:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created `src/aquacal/core/_aquakit_bridge.py` with all 5 bridge wrappers matching AquaCal's current numpy-in/numpy-out API shapes
- Implemented `_make_interface_params(water_z, n_air, n_water)` factory so call sites construct AquaKit InterfaceParams without ever naming or importing the AquaKit type directly
- Private `_to_torch()` and `_to_numpy()` helpers centralize all numpy<->torch conversion in one place
- `InterfaceParams` is NOT in `__all__` and not importable from the bridge — call sites must use the factory
- Ruff lint and format pass with all pre-commit hooks

## Task Commits

Each task was committed atomically:

1. **Task 1: Create AquaKit bridge module** - `290066d` (feat)

**Plan metadata:** see final docs commit below

## Files Created/Modified
- `src/aquacal/core/_aquakit_bridge.py` - Bridge module with 5 wrappers, factory, and private conversion helpers

## Decisions Made
- Used `_make_interface_params()` factory with local import (`from aquakit.types import InterfaceParams as _AquaKitInterfaceParams`) to avoid module-level name collision
- `_bridge_refractive_project` implements the two-step pattern: AquaKit returns interface point, `camera.project()` converts to pixel with distortion applied
- For `_bridge_trace_ray_air_to_water` and `_bridge_refractive_back_project`: Camera back-projection happens in the bridge (calling `camera.pixel_to_ray_world()`), then rays are passed to AquaKit — this keeps Camera as a numpy-only object

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed import ordering to satisfy ruff I001**
- **Found during:** Task 1 (initial ruff check)
- **Issue:** Ruff flagged I001 (import block un-sorted) because `numpy` and `torch` were in a separate block from `aquakit`
- **Fix:** Moved `aquakit` import to the same block as `numpy` and `torch` (all third-party); sorted alphabetically
- **Files modified:** `src/aquacal/core/_aquakit_bridge.py`
- **Verification:** `python -m ruff check src/aquacal/core/_aquakit_bridge.py` passes with no errors
- **Committed in:** `290066d` (Task 1 commit)

**2. [Rule 3 - Blocking] ruff format auto-reformatted file on first commit attempt**
- **Found during:** Task 1 commit (pre-commit hook)
- **Issue:** ruff-format modified the file (whitespace/line-length adjustments)
- **Fix:** Re-staged reformatted file, committed again
- **Files modified:** `src/aquacal/core/_aquakit_bridge.py`
- **Verification:** All pre-commit hooks pass on second commit attempt
- **Committed in:** `290066d` (final Task 1 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule 3 - blocking lint/format issues)
**Impact on plan:** Both fixes required for code style compliance. No scope creep.

## Issues Encountered
- AquaKit is not installed in the local conda environment (`AquaCal` env). Import verification fails with the torch guard from Phase 13. This is expected behavior — CI has torch and aquakit installed via the updated workflows. The bridge module is syntactically valid and ruff-clean.

## Self-Check

**1. Check created files exist:**
- `src/aquacal/core/_aquakit_bridge.py` — FOUND (committed as `290066d`)

**2. Check commits exist:**
- `290066d` — FOUND (feat(14-01): create AquaKit bridge module)

## Self-Check: PASSED

## User Setup Required
None - no external service configuration required. AquaKit import will succeed once torch is installed in the environment.

## Next Phase Readiness
- Bridge module is the foundation for all subsequent Phase 14 rewiring plans
- Plans 14-02 onwards can import bridge functions and begin rewiring call sites
- All 5 GEOM functions are wrapped: snells_law_3d, trace_ray_air_to_water, refractive_project, refractive_back_project, ray_plane_intersection

---
*Phase: 14-geometry-rewiring*
*Completed: 2026-02-19*
