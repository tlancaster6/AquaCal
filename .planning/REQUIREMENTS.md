# Requirements: AquaCal v1.5 AquaKit Integration

**Defined:** 2026-02-19
**Core Value:** Accurate refractive camera calibration from standard ChArUco board observations — researchers can pip install aquacal, point it at their videos, and get a calibration result they trust.

**Note:** Bug fixes and small revisions to AquaKit will be performed as necessary during this milestone as issues are discovered during rewiring.

## v1 Requirements

Requirements for this milestone. Each maps to roadmap phases.

### Setup

- [ ] **SETUP-01**: AquaCal declares `aquakit` as a pip dependency in pyproject.toml
- [ ] **SETUP-02**: PyTorch compatibility is documented (prerequisite install) and handled gracefully at import time

### Geometry Rewiring

- [ ] **GEOM-01**: `snells_law_3d` calls route through AquaKit with numpy↔torch conversion at boundaries
- [ ] **GEOM-02**: `trace_ray_air_to_water` calls route through AquaKit with numpy↔torch conversion
- [ ] **GEOM-03**: `refractive_project` calls route through AquaKit (replaces `refractive_project_fast` and `refractive_project_fast_batch`)
- [ ] **GEOM-04**: `refractive_back_project` calls route through AquaKit with numpy↔torch conversion
- [ ] **GEOM-05**: `ray_plane_intersection` calls route through AquaKit with numpy↔torch conversion

### Utility Rewiring

- [ ] **UTIL-01**: Pose transforms (`rvec_to_matrix`, `matrix_to_rvec`, `compose_poses`, `invert_pose`, `camera_center`) route through AquaKit
- [ ] **UTIL-02**: Schema types (`CameraIntrinsics`, `CameraExtrinsics`, `InterfaceParams`, `Vec2/Vec3/Mat3`, `INTERFACE_NORMAL`) are imported from AquaKit or wrapped with conversion

### I/O Rewiring

- [ ] **IO-01**: `load_calibration_data` routes through AquaKit
- [ ] **IO-02**: `VideoSet`, `ImageSet`, `FrameSet`, `create_frameset` route through AquaKit
- [ ] **IO-03**: `triangulate_rays` routes through AquaKit

### Testing

- [ ] **TEST-01**: Numerical equivalence tests verify rewired geometry functions match originals within 1e-4 relative tolerance
- [ ] **TEST-02**: Existing test suite (584 tests) passes after rewiring
- [ ] **TEST-03**: End-to-end calibration pipeline produces equivalent results before and after rewiring

### Cleanup

- [ ] **CLEAN-01**: Redundant AquaCal implementations deleted after equivalence is verified
- [ ] **CLEAN-02**: Unused imports and dead internal helpers removed

### Documentation

- [ ] **DOCS-01**: Sphinx docs updated with new import paths where user-facing
- [ ] **DOCS-02**: Jupyter notebooks updated to reflect AquaKit dependency
- [ ] **DOCS-03**: README updated to mention AquaKit dependency

### Migration Report

- [ ] **MIGR-01**: Written migration report identifies additional AquaCal modules with cross-library potential for future AquaKit migration (conservative — only modules with strong cross-library utility, e.g. synthetic data generation)

## Future Requirements

Deferred to next milestone (full PyTorch migration):

- **TORCH-01**: AquaCal internals converted from NumPy to PyTorch throughout
- **TORCH-02**: NumPy↔torch conversion boundary eliminated
- **CI-01**: GitHub Actions CI matrix updated with torch installation

## Out of Scope

| Feature | Reason |
|---------|--------|
| AquaMVS rewiring | Separate library, handled independently |
| Full PyTorch migration of AquaCal internals | Next milestone — this round keeps NumPy internals |
| Hero image redesign | Deferred creative task, unrelated to rewiring |
| Memory/CPU optimization | Separate concern, deferred |
| GPU acceleration | AquaKit supports CUDA but AquaCal stays CPU for now |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| *(populated by roadmapper)* | | |

**Coverage:**
- v1 requirements: 19 total
- Mapped to phases: 0
- Unmapped: 19 ⚠️

---
*Requirements defined: 2026-02-19*
*Last updated: 2026-02-19 after initial definition*
