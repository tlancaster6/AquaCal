---
phase: 14-geometry-rewiring
verified: 2026-02-19T18:30:00Z
status: passed
score: 5/5 success criteria verified
re_verification: false
notes: "Test files importing deprecated functions directly is by design — they exercise originals for Phase 16 equivalence testing. User approved 2026-02-19."
gaps: []
    artifacts:
      - path: "tests/unit/test_interface_model.py"
        issue: "Imports ray_plane_intersection from aquacal.core.interface_model (deprecated), not from bridge or aquacal.core"
      - path: "tests/unit/test_refractive_geometry.py"
        issue: "Imports snells_law_3d, trace_ray_air_to_water, refractive_back_project, refractive_project from deprecated aquacal.core.refractive_geometry"
      - path: "tests/unit/test_extrinsics.py"
        issue: "Imports refractive_project from deprecated aquacal.core.refractive_geometry"
      - path: "tests/unit/test_reconstruction.py"
        issue: "Imports refractive_project from deprecated aquacal.core.refractive_geometry"
      - path: "tests/unit/test_triangulate.py"
        issue: "Imports refractive_project from deprecated aquacal.core.refractive_geometry"
      - path: "tests/unit/test_reprojection.py"
        issue: "Imports refractive_project from deprecated aquacal.core.refractive_geometry"
    missing:
      - "Decision: either update test imports to use aquacal.core (bridge-backed) or explicitly accept that tests exercise legacy code until Phase 16/17 cleanup"
human_verification:
  - test: "In CI with torch and aquakit installed, run a minimal call through each bridge function and confirm no runtime errors"
    expected: "All 5 _bridge_* functions execute successfully on simple inputs"
    why_human: "AquaKit/torch not installed locally; all verification was source-code-only (AST/grep)"
---

# Phase 14: Geometry Rewiring Verification Report

**Phase Goal:** All refractive geometry calls in AquaCal route through AquaKit with numpy/torch conversion at boundaries
**Verified:** 2026-02-19T18:30:00Z
**Status:** gaps_found
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths (from ROADMAP.md Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | snells_law_3d call sites pass numpy arrays, convert to torch internally, call aquakit.snells_law_3d, and return numpy with updated TIR handling | VERIFIED | _bridge_snells_law_3d in _aquakit_bridge.py lines 86-116 does exactly this. core/__init__.py exports bridge version. No production code imports old snells_law_3d from refractive_geometry. |
| 2 | trace_ray_air_to_water call sites use the tensor-based AquaKit signature with InterfaceParams instead of Camera/Interface objects | VERIFIED | _bridge_trace_ray_air_to_water accepts Camera + aquakit InterfaceParams (built by _make_interface_params). AquaKit call at line 153 receives tensors. No production call site uses old trace_ray_air_to_water from refractive_geometry. |
| 3 | refractive_project call sites use the two-step AquaKit flow and fast shims are removed or deprecated | VERIFIED | All 7 production call sites use _bridge_refractive_project. refractive_project_fast and refractive_project_fast_batch deleted. |
| 4 | refractive_back_project call sites pass raw tensors (not Camera objects) to AquaKit | VERIFIED | _bridge_refractive_back_project passes pixel_rays_t and cam_centers_t tensors to aquakit.refractive_back_project at line 232. triangulate.py uses the bridge with _make_interface_params per-camera. |
| 5 | ray_plane_intersection call sites route through aquakit.ray_plane_intersection with numpy/torch conversion | PARTIAL | Bridge exists and is re-exported from core/__init__.py. No production code calls the deprecated version. However 6 test files (test_interface_model.py, test_refractive_geometry.py, test_extrinsics.py, test_reconstruction.py, test_triangulate.py, test_reprojection.py) import deprecated functions directly from refractive_geometry or interface_model. |

**Score:** 4/5 truths verified (SC5 partial — production routing correct, test imports not updated)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| src/aquacal/core/_aquakit_bridge.py | All 5 bridge wrappers with numpy/torch conversion | VERIFIED | 284 lines; all 5 bridge functions with proper torch conversion |
| src/aquacal/triangulation/triangulate.py | Uses _bridge_refractive_back_project + _make_interface_params | VERIFIED | Lines 6-9 import from bridge; lines 45-50 use factory per-camera |
| src/aquacal/core/__init__.py | Exports all 5 bridge-backed functions under original names, no AquaKit InterfaceParams, no refractive_project_batch | VERIFIED | Lines 3-17 all 5 bridge aliases; InterfaceParams count=0; refractive_project_batch absent |
| src/aquacal/core/refractive_geometry.py | Originals DEPRECATED, fast shims deleted | VERIFIED | All 5 functions have DEPRECATED comment; no refractive_project_fast found |
| src/aquacal/core/interface_model.py | ray_plane_intersection marked DEPRECATED | VERIFIED | Line 100: DEPRECATED comment present |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| triangulate.py | _aquakit_bridge.py | _bridge_refractive_back_project + _make_interface_params import | WIRED | Lines 6-9 import; lines 45-50 usage |
| core/__init__.py | _aquakit_bridge.py | 5 bridge aliases | WIRED | Lines 3-17; all 5 re-exported |
| _optim_common.py | _aquakit_bridge.py | _bridge_refractive_project | WIRED | Line 20 import; line 482 usage |
| extrinsics.py | _aquakit_bridge.py | _bridge_refractive_project | WIRED | Line 22; line 175 |
| interface_estimation.py | _aquakit_bridge.py | _bridge_refractive_project | WIRED | Line 489; line 597 |
| pipeline.py | _aquakit_bridge.py | _bridge_refractive_project | WIRED | Line 1196; line 1233 |
| rendering.py | _aquakit_bridge.py | _bridge_refractive_project | WIRED | Line 21; line 80 |
| synthetic.py | _aquakit_bridge.py | _bridge_refractive_project | WIRED | Line 26; line 528 |
| reprojection.py | _aquakit_bridge.py | _bridge_refractive_project | WIRED | Line 16; lines 100, 181 |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| tests/unit/test_interface_model.py | 4 | from aquacal.core.interface_model import ray_plane_intersection | Warning | Test calls deprecated function directly |
| tests/unit/test_refractive_geometry.py | 9-14 | Imports 5 deprecated functions from refractive_geometry | Warning | Tests exercise deprecated implementations, not bridge |
| tests/unit/test_extrinsics.py | 27 | from aquacal.core.refractive_geometry import refractive_project | Warning | Test setup uses deprecated path |
| tests/unit/test_reconstruction.py | 22 | from aquacal.core.refractive_geometry import refractive_project | Warning | Test fixture uses deprecated path |
| tests/unit/test_triangulate.py | 18 | from aquacal.core.refractive_geometry import refractive_project | Warning | Test helper uses deprecated path |
| tests/unit/test_reprojection.py | 24 | from aquacal.core.refractive_geometry import refractive_project | Warning | Test fixture uses deprecated path |

No blocker anti-patterns found. All deprecated-path usage is confined to tests/unit/.

### Human Verification Required

#### 1. AquaKit runtime execution

**Test:** In a CI environment with torch and aquakit installed, run each of the 5 bridge functions on a simple numpy input.
**Expected:** All 5 functions execute without error and return numpy arrays of the correct shape.
**Why human:** AquaKit/torch are not installed in the local environment. All verification was source-code-only (AST/grep). The bridge is syntactically and structurally correct but live execution has not been confirmed locally.

### Gaps Summary

The phase goal is achieved for all production call sites. Every internal usage of the 5 geometry functions in calibration, triangulation, datasets, and validation routes through the AquaKit bridge. The `core/__init__.py` public API re-exports bridge-backed versions under original names with no breaking changes. AquaKit InterfaceParams is correctly excluded from the public API. Fast shims are deleted.

The single gap is that 6 unit test files still import from the deprecated modules directly:

- tests/unit/test_interface_model.py imports ray_plane_intersection from interface_model
- tests/unit/test_refractive_geometry.py imports snells_law_3d, trace_ray_air_to_water, refractive_back_project, refractive_project from refractive_geometry
- tests/unit/test_extrinsics.py, test_reconstruction.py, test_triangulate.py, test_reprojection.py each import refractive_project from refractive_geometry

This may be intentional (tests can run without torch until Phase 16) or an oversight. A decision is needed: either update test imports to use the bridge-backed public API (aquacal.core), or explicitly document that test files will be updated in Phase 16/17 when the deprecated modules are deleted.

---

_Verified: 2026-02-19T18:30:00Z_
_Verifier: Claude (gsd-verifier)_
