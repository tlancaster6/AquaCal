# Phase 14: Geometry Rewiring - Context

**Gathered:** 2026-02-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Route all refractive geometry calls in AquaCal through AquaKit, with numpy/torch conversion at call boundaries. Covers 5 functions: `snells_law_3d`, `trace_ray_air_to_water`, `refractive_project`, `refractive_back_project`, `ray_plane_intersection`. Original implementations are kept for Phase 16 equivalence testing and deleted in Phase 17.

</domain>

<decisions>
## Implementation Decisions

### Conversion boundary design
- Create a bridge module (e.g. `core/_aquakit_bridge.py`) that centralizes all numpy-to-torch-to-numpy conversion
- All AquaCal code calls bridge functions, never AquaKit directly
- Bridge adopts AquaKit's function signatures (InterfaceParams, raw tensors, etc.) — call sites update to match
- Bridge also re-exports AquaKit types (InterfaceParams, CameraIntrinsics, etc.) so AquaCal imports them from one place
- Conversion helpers (`_to_torch()`, `_to_numpy()`) are private to the bridge — no other module needs to know about torch
- Rationale: when AquaCal eventually moves to full PyTorch, the bridge conversion layer is removed rather than hunting through call sites

### Fast shim fate
- Delete `refractive_project_fast` and `refractive_project_fast_batch` entirely — AquaKit's projection replaces them
- Also remove the `use_fast` parameter from any public API that exposes it (clean break — TypeError signals callers to update)
- Keep original `refractive_project` (bisection-based) until Phase 16 equivalence tests pass, then delete in Phase 17
- The sparse Jacobian callable (which currently uses `refractive_project_fast` internally) goes through the bridge like everything else — equivalence first, optimization later during the full PyTorch port
- Mark original geometry functions with `# DEPRECATED` comment so it's clear they're superseded, but no runtime warnings

### TIR / edge-case handling
- Match AquaCal's current TIR behavior exactly through the bridge (Claude checks what AquaCal currently does and replicates it)
- Trust AquaKit's TIR detection as source of truth — no compatibility shim for minor numerical threshold differences
- Back-projection unpacking (Camera objects vs raw arrays): Claude's discretion based on call-site analysis

### Claude's Discretion
- Whether bridge exposes refractive_project as one combined function or two separate steps (based on whether any call site needs intermediate interface points)
- How bridge unpacks Camera objects for back-projection (bridge does it vs call sites pass raw arrays)
- Exact grouping of functions within rewiring plans

</decisions>

<specifics>
## Specific Ideas

- "Pick whichever option will make our lives easier when we eventually update to full PyTorch support in AquaCal" — future torch migration is a key design driver
- "At this stage we are focused on equivalence. We'll tackle optimization when we do the full pytorch port" — correctness over performance

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

### Migration ordering
- Bridge module built first as its own plan (Plan 1: bridge + conversion helpers + type re-exports)
- All 5 geometry functions rewired together in a single sweep (not one-by-one)
- Periodic test checkpoints during rewiring (e.g. after bridge built, after projection, after back-projection)

---

*Phase: 14-geometry-rewiring*
*Context gathered: 2026-02-19*
