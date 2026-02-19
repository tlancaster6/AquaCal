---
phase: 14-geometry-rewiring
plan: 02
subsystem: calibration, datasets, validation, core
tags: [aquakit, bridge, refraction, rewiring, deprecated]

# Dependency graph
requires:
  - phase: 14-01
    provides: _aquakit_bridge.py with all 5 bridge wrappers and _make_interface_params()
provides:
  - All 7 refractive_project call sites routed through _bridge_refractive_project
  - refractive_project_fast and refractive_project_fast_batch deleted
  - Original geometry implementations marked # DEPRECATED
affects: [14-03, 15-equivalence, 16-cleanup, 17-release]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Factory pattern at call sites: _make_interface_params(water_z, n_air, n_water) constructs AquaKit InterfaceParams without naming the type"
    - "Bridge-only projection: all call sites use _bridge_refractive_project instead of refractive_project"

key-files:
  created: []
  modified:
    - src/aquacal/calibration/_optim_common.py
    - src/aquacal/calibration/extrinsics.py
    - src/aquacal/calibration/interface_estimation.py
    - src/aquacal/calibration/pipeline.py
    - src/aquacal/datasets/rendering.py
    - src/aquacal/datasets/synthetic.py
    - src/aquacal/validation/reprojection.py
    - src/aquacal/core/refractive_geometry.py
    - src/aquacal/core/interface_model.py

key-decisions:
  - "rendering.py and reprojection.py keep Interface import for function signatures (type hints); only refractive_project call is replaced with bridge"
  - "compute_reprojection_error_single builds _make_interface_params from Interface object fields at the call site, bridging old API to new"
  - "pipeline.py frame_residuals closure builds interface_aq per-camera inside the loop (each camera has its own water_z)"
  - "synthetic.py hardcodes n_air=1.0, n_water=1.333 as defaults since Interface was constructed without explicit n values"

# Metrics
duration: 7min
completed: 2026-02-19
---

# Phase 14 Plan 02: Call Site Rewiring Summary

**7 call-site files rewired to _bridge_refractive_project; fast shims deleted; original implementations marked DEPRECATED**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-02-19T17:49:59Z
- **Completed:** 2026-02-19T17:57:07Z
- **Tasks:** 2
- **Files modified:** 9

## Accomplishments

- Rewired all 7 call-site files to use `_bridge_refractive_project` and `_make_interface_params` from `_aquakit_bridge.py`
- Zero `from aquacal.core.refractive_geometry import refractive_project` imports remain in the 7 target files
- Zero `import torch` added to any call site
- Deleted `refractive_project_fast()` and `refractive_project_fast_batch()` from `refractive_geometry.py`
- Added `# DEPRECATED` comments to all 5 original geometry functions in `refractive_geometry.py`
- Added `# DEPRECATED` comment to `ray_plane_intersection` in `interface_model.py`
- Ruff lint and format pass with all pre-commit hooks

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewire refractive_project call sites to bridge** - `a923f0f` (feat)
2. **Task 2: Remove fast shims and mark originals deprecated** - `a20d6e0` (feat)

**Plan metadata:** see final docs commit below

## Files Modified

### Task 1 (call site rewiring)
- `src/aquacal/calibration/_optim_common.py` - `compute_residuals()` uses bridge
- `src/aquacal/calibration/extrinsics.py` - `refractive_solve_pnp()` uses bridge
- `src/aquacal/calibration/interface_estimation.py` - `register_auxiliary_camera()` lazy imports bridge
- `src/aquacal/calibration/pipeline.py` - `_estimate_validation_poses()` lazy imports bridge
- `src/aquacal/datasets/rendering.py` - `render_synthetic_frame()` uses bridge; `Interface` kept for type hint
- `src/aquacal/datasets/synthetic.py` - `generate_synthetic_detections()` uses bridge; unused `interface_normal` removed
- `src/aquacal/validation/reprojection.py` - both projection functions use bridge; `Interface` kept for `compute_reprojection_error_single` signature

### Task 2 (cleanup)
- `src/aquacal/core/refractive_geometry.py` - fast shims deleted; 5 functions marked `# DEPRECATED`
- `src/aquacal/core/interface_model.py` - `ray_plane_intersection` marked `# DEPRECATED`

## Decisions Made

- `rendering.py` and `reprojection.py` retain `Interface` import because public functions accept `Interface` as a parameter (type hints). The bridge wrapping happens internally by extracting `water_z`, `n_air`, `n_water` from the object.
- `pipeline.py` frame_residuals closure constructs `interface_aq` per-camera inside the loop — each camera has a different `water_z` from `water_z_values[cam_name]`, so one shared `interface_aq` cannot be reused.
- `synthetic.py` used `Interface` with default `n_air=1.0, n_water=1.333`. These defaults are now hardcoded at the `_make_interface_params` call site. This matches the `Interface.__init__` defaults.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Ruff I001 import ordering in all 7 modified files**
- **Found during:** Task 1 (initial ruff check)
- **Issue:** Ruff flagged I001 (import block unsorted) because bridge imports were placed in the middle of local import blocks
- **Fix:** Applied `python -m ruff check --fix` to auto-sort imports; ruff reformatted multi-line import parens
- **Files modified:** All 7 call-site files
- **Verification:** `python -m ruff check src/aquacal/calibration/ src/aquacal/datasets/ src/aquacal/validation/` passes
- **Committed in:** `a923f0f`

**2. [Rule 1 - Bug] Unused variable `interface_normal` in synthetic.py**
- **Found during:** Task 1 (ruff F841 error)
- **Issue:** `generate_synthetic_detections()` created `interface_normal = np.array([0.0, 0.0, -1.0])` that was only needed to construct the old `Interface` object. After rewiring, `_make_interface_params` doesn't need the normal (it's hardcoded to [0,0,-1] in the factory).
- **Fix:** Removed the unused assignment
- **Files modified:** `src/aquacal/datasets/synthetic.py`
- **Verification:** `python -m ruff check src/aquacal/datasets/synthetic.py` passes
- **Committed in:** `a923f0f`

**3. [Rule 3 - Blocking] Ruff format reformatted refractive_geometry.py on Task 2 commit**
- **Found during:** Task 2 pre-commit hook
- **Issue:** Deleting the fast shim functions left trailing whitespace that ruff-format fixed
- **Fix:** Re-staged and recommitted
- **Files modified:** `src/aquacal/core/refractive_geometry.py`
- **Verification:** All pre-commit hooks pass on second attempt
- **Committed in:** `a20d6e0`

---

**Total deviations:** 3 auto-fixed (2 Rule 3 - blocking lint/format; 1 Rule 1 - unused variable cleanup)
**Impact on plan:** All fixes required for code style compliance. No scope creep.

## Issues Encountered
- AquaKit/torch not installed locally — import-level tests fail with the torch guard from Phase 13. This is expected; CI has torch/aquakit installed. Syntax was verified via `ast.parse()` on all modified files.

## Self-Check

**1. Check commits exist:**
- `a923f0f` — Task 1: Rewire call sites
- `a20d6e0` — Task 2: Remove fast shims and mark deprecated

**2. Check key verification criteria:**
- `grep -n "refractive_project_fast" src/aquacal/core/refractive_geometry.py` → NOT FOUND (good)
- `grep -rn "from aquacal.core.refractive_geometry import refractive_project" ...` → NOT FOUND (good)
- `grep -rn "_bridge_refractive_project" src/ | grep -v _aquakit_bridge.py` → 17 matches in 7 files (good)
- `grep -c "DEPRECATED" src/aquacal/core/refractive_geometry.py` → 5 (good)
- `grep -c "DEPRECATED" src/aquacal/core/interface_model.py` → 1 (good)
- `python -m ruff check src/` → All checks passed

## Self-Check: PASSED

## Next Phase Readiness
- Forward projection path (the most-used geometry operation) now routes through AquaKit
- Plan 14-03 can proceed to rewire back-projection and ray-tracing call sites
- Original implementations remain in `refractive_geometry.py` and `interface_model.py` for Phase 16 equivalence testing

---
*Phase: 14-geometry-rewiring*
*Completed: 2026-02-19*
