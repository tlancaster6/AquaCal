# Roadmap: AquaCal

## Milestones

- ✅ **v1.2 MVP** — Phases 1-6 (shipped 2026-02-15)
- ✅ **v1.4 QA & Polish** — Phases 7-12 (shipped 2026-02-19)
- 🚧 **v1.5 AquaKit Integration** — Phases 13-17 (in progress)

## Phases

<details>
<summary>✅ v1.2 MVP (Phases 1-6) — SHIPPED 2026-02-15</summary>

- [x] Phase 1: Foundation and Cleanup (3/3 plans) — completed 2026-02-14
- [x] Phase 2: CI/CD Automation (3/3 plans) — completed 2026-02-14
- [x] Phase 3: Public Release (3/3 plans) — completed 2026-02-14
- [x] Phase 4: Example Data (3/3 plans) — completed 2026-02-14
- [x] Phase 5: Documentation Site (4/4 plans) — completed 2026-02-14
- [x] Phase 6: Interactive Tutorials (4/4 plans) — completed 2026-02-15

See `.planning/milestones/v1.2-ROADMAP.md` for full details.

</details>

<details>
<summary>✅ v1.4 QA & Polish (Phases 7-12) — SHIPPED 2026-02-19</summary>

- [x] Phase 7: Infrastructure Check (1/1 plans) — completed 2026-02-15
- [x] Phase 8: CLI QA Execution (1/1 plans) — completed 2026-02-15
- [x] Phase 9: Bug Triage (0/0 plans — no bugs found) — completed 2026-02-17
- [x] Phase 10: Documentation Audit (3/3 plans) — completed 2026-02-16
- [x] Phase 11: Documentation Visuals (2/2 plans) — completed 2026-02-17
- [x] Phase 12: Tutorial Verification (3/3 plans) — completed 2026-02-19

See `.planning/milestones/v1.4-ROADMAP.md` for full details.

</details>

### 🚧 v1.5 AquaKit Integration (In Progress)

**Milestone Goal:** Wire AquaKit shared library into AquaCal, replace redundant code, verify numerical equivalence, and produce a migration report for future candidates.

- [ ] **Phase 13: Setup** — Add AquaKit dependency and PyTorch compatibility handling
- [ ] **Phase 14: Geometry Rewiring** — Route refractive geometry functions through AquaKit
- [ ] **Phase 15: Utility and I/O Rewiring** — Route transforms, schema types, and I/O through AquaKit
- [ ] **Phase 16: Testing** — Verify numerical equivalence and confirm full suite health
- [ ] **Phase 17: Cleanup, Docs, and Migration Report** — Delete dead code, update docs, document future candidates

## Phase Details

### Phase 13: Setup
**Goal**: AquaCal can declare and load AquaKit as a dependency with graceful PyTorch handling
**Depends on**: Nothing (first phase of this milestone)
**Requirements**: SETUP-01, SETUP-02
**Success Criteria** (what must be TRUE):
  1. `pip install aquacal` also installs `aquakit` as a transitive dependency
  2. Importing `aquacal` when PyTorch is not installed raises a clear, actionable error message (not a cryptic `ModuleNotFoundError` deep in the stack)
  3. Sphinx docs and README document the PyTorch prerequisite install step
**Plans**: TBD

### Phase 14: Geometry Rewiring
**Goal**: All refractive geometry calls in AquaCal route through AquaKit with numpy/torch conversion at boundaries
**Depends on**: Phase 13
**Requirements**: GEOM-01, GEOM-02, GEOM-03, GEOM-04, GEOM-05
**Success Criteria** (what must be TRUE):
  1. `snells_law_3d` call sites pass numpy arrays, convert to torch internally, call `aquakit.snells_law_3d`, and return numpy — TIR handling updated for the `(directions, valid)` tuple return
  2. `trace_ray_air_to_water` call sites use the tensor-based AquaKit signature with `InterfaceParams` instead of Camera/Interface objects
  3. `refractive_project` call sites use the two-step AquaKit flow (find interface point, then project through camera) and the `refractive_project_fast` / `refractive_project_fast_batch` shims are removed or deprecated
  4. `refractive_back_project` call sites pass raw tensors (not Camera objects) to AquaKit
  5. `ray_plane_intersection` call sites route through `aquakit.ray_plane_intersection` with numpy/torch conversion
**Plans**: TBD

### Phase 15: Utility and I/O Rewiring
**Goal**: Pose transforms, schema types, and I/O utilities in AquaCal route through AquaKit
**Depends on**: Phase 13
**Requirements**: UTIL-01, UTIL-02, IO-01, IO-02, IO-03
**Success Criteria** (what must be TRUE):
  1. `rvec_to_matrix`, `matrix_to_rvec`, `compose_poses`, `invert_pose`, and `camera_center` call sites route through AquaKit with numpy/torch conversion wrappers
  2. `CameraIntrinsics`, `CameraExtrinsics`, `InterfaceParams`, `Vec2`, `Vec3`, `Mat3`, and `INTERFACE_NORMAL` are imported from AquaKit (or wrapped) throughout AquaCal; tensor-bearing fields are handled correctly at numpy boundaries
  3. `load_calibration_data` routes through AquaKit and the `CalibrationData` / `CameraData` return types are handled correctly
  4. `VideoSet`, `ImageSet`, `FrameSet`, and `create_frameset` import from AquaKit; existing call sites behave identically
  5. `triangulate_rays` call sites use the AquaKit `(origin, direction)` tuple-list signature
**Plans**: TBD

### Phase 16: Testing
**Goal**: Rewired AquaCal is numerically equivalent to the original and the full test suite is green
**Depends on**: Phase 14, Phase 15
**Requirements**: TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. Equivalence tests exist for every rewired geometry function and all pass with relative tolerance 1e-4 against the original AquaCal implementations (or saved reference outputs)
  2. All 584 existing tests pass after rewiring (no regressions)
  3. An end-to-end calibration run on the synthetic test dataset produces reprojection errors within 1e-4 of pre-rewiring baseline
**Plans**: TBD

### Phase 17: Cleanup, Docs, and Migration Report
**Goal**: Dead code is deleted, user-facing docs reflect the new dependency, and future AquaKit migration candidates are documented
**Depends on**: Phase 16
**Requirements**: CLEAN-01, CLEAN-02, DOCS-01, DOCS-02, DOCS-03, MIGR-01
**Success Criteria** (what must be TRUE):
  1. Redundant AquaCal geometry and I/O implementations that have been superseded by AquaKit are deleted and no longer importable from `aquacal.*`
  2. All unused imports and dead internal helper functions are removed; linting passes cleanly
  3. Sphinx API docs reflect updated import paths for any user-facing symbols that moved; notebook cells that import from old paths are corrected
  4. README mentions AquaKit as a dependency with a PyTorch prerequisite note
  5. A written migration report (`MIGRATION_REPORT.md`) identifies additional AquaCal modules with strong cross-library potential (conservative list) and explains why each is or is not a candidate
**Plans**: TBD

## Progress

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Foundation and Cleanup | v1.2 | 3/3 | Complete | 2026-02-14 |
| 2. CI/CD Automation | v1.2 | 3/3 | Complete | 2026-02-14 |
| 3. Public Release | v1.2 | 3/3 | Complete | 2026-02-14 |
| 4. Example Data | v1.2 | 3/3 | Complete | 2026-02-14 |
| 5. Documentation Site | v1.2 | 4/4 | Complete | 2026-02-14 |
| 6. Interactive Tutorials | v1.2 | 4/4 | Complete | 2026-02-15 |
| 7. Infrastructure Check | v1.4 | 1/1 | Complete | 2026-02-15 |
| 8. CLI QA Execution | v1.4 | 1/1 | Complete | 2026-02-15 |
| 9. Bug Triage | v1.4 | 0/0 | Complete | 2026-02-17 |
| 10. Documentation Audit | v1.4 | 3/3 | Complete | 2026-02-16 |
| 11. Documentation Visuals | v1.4 | 2/2 | Complete | 2026-02-17 |
| 12. Tutorial Verification | v1.4 | 3/3 | Complete | 2026-02-19 |
| 13. Setup | v1.5 | 0/TBD | Not started | - |
| 14. Geometry Rewiring | v1.5 | 0/TBD | Not started | - |
| 15. Utility and I/O Rewiring | v1.5 | 0/TBD | Not started | - |
| 16. Testing | v1.5 | 0/TBD | Not started | - |
| 17. Cleanup, Docs, and Migration Report | v1.5 | 0/TBD | Not started | - |
