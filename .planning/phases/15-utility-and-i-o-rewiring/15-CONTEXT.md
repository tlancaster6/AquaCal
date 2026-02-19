# Phase 15: Utility and I/O Rewiring - Context

**Gathered:** 2026-02-19
**Status:** Ready for planning

<domain>
## Phase Boundary

Route pose transforms, schema types, and I/O utilities in AquaCal through AquaKit, replacing redundant implementations with AquaKit equivalents. The Camera class and calibration pipeline internals are NOT in scope — Camera replacement is deferred to Phase 17.

Reference: `.planning/rewiring/REWIRING.md` contains the complete import replacement table and signature change documentation.

</domain>

<decisions>
## Implementation Decisions

### Type replacement strategy
- **Vec2, Vec3, Mat3**: Import from AquaKit (now `torch.Tensor` type aliases instead of numpy)
- **CameraIntrinsics, CameraExtrinsics**: Replace with AquaKit types (torch tensor fields). If missing critical fields/methods with no AquaKit equivalent, flag for user to implement upstream in AquaKit first
- **InterfaceParams**: Replace AquaCal's InterfaceParams with AquaKit's everywhere. Phase 14's collision avoidance was temporary — resolve the dual-type situation now
- **INTERFACE_NORMAL**: Check if AquaKit exports it during research; if yes, import from AquaKit; if no, keep AquaCal's

### Missing AquaKit equivalents
- **All 5 pose transform functions** (rvec_to_matrix, matrix_to_rvec, compose_poses, invert_pose, camera_center) exist in AquaKit — wire them all through
- **All I/O types** (VideoSet, ImageSet, FrameSet, create_frameset) exist in AquaKit — wire them all through
- **create_frameset**: AquaKit implements it at `src/aquakit/io/images.py` — add to AquaCal's API
- **Camera class**: Keep in AquaCal for Phase 15. It's heavily used in calibration (not ported to AquaKit). Defer replacement to Phase 17 cleanup
- **Functions not in AquaKit**: Decide case-by-case whether to implement upstream (based on anticipated reuse by other libraries) or keep in AquaCal

### Bridge organization
- **Separate bridge modules per domain**: `core/_aquakit_bridge.py` (geometry, existing from Phase 14), `utils/_aquakit_bridge.py` (transforms), `io/_aquakit_bridge.py` (I/O)
- **Conversion pattern**: Claude decides per function whether callers pass numpy (bridge converts) or torch directly, based on how deep in the numpy-based pipeline each caller is
- **Conversion helpers**: Provide helpers to create AquaKit types from numpy values (e.g., `make_intrinsics(K_np, dist_np, ...)`) since the calibration pipeline works with numpy arrays

### Public API surface
- **Import paths updated**: Old import paths removed (clean break). Since v1.5 is pre-release, no deprecation warnings needed
- **Naming**: Adopt AquaKit names where they differ — `load_calibration_data` replaces `load_calibration`
- **Type exposure**: Claude decides whether to re-export AquaKit types directly or provide numpy-friendly wrappers, based on downstream user expectations

</decisions>

<specifics>
## Specific Ideas

- REWIRING.md at `.planning/rewiring/REWIRING.md` is the authoritative reference for all import mappings and signature changes
- Phase 14's alias pattern in `core/__init__.py` is superseded — Phase 15 updates import paths rather than aliasing
- AquaKit source is at `/c/Users/tucke/PycharmProjects/AquaKit` for cross-reference during research

</specifics>

<deferred>
## Deferred Ideas

- Camera class replacement with `create_camera()` + protocol pattern — Phase 17 cleanup
- `refractive_project_batch` bridge (no AquaKit batch equivalent yet) — Phase 16/17

</deferred>

---

*Phase: 15-utility-and-i-o-rewiring*
*Context gathered: 2026-02-19*
