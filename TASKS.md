# Task List

**This file is the authoritative source for task IDs and phase numbering.**
Other documents (`development_plan.md`, `agent_implementation_spec.md`) may organize content by conceptual phases, but task assignments should reference the IDs below.

Status key: `[ ]` not started | `[~]` in progress | `[x]` complete

---

## Phase 0: Project Setup

- [x] **0.1** Create `pyproject.toml` with dependencies and package config
- [x] **0.2** Add `__init__.py` files to all packages with explicit exports
- [x] **0.3** Add `requirements.txt` for pip-only users

## Phase 1: Foundation

- [x] **1.1** Define configuration and output schemas (`config/schema.py`)
- [x] **1.2** Implement rotation/transform utilities (`utils/transforms.py`)
- [x] **1.3** Implement board geometry (`core/board.py`)

## Phase 2: Core Geometry

- [x] **2.1** Implement camera model (`core/camera.py`)
- [x] **2.2** Implement interface model (`core/interface_model.py`)
- [x] **2.3** Implement refractive geometry (`core/refractive_geometry.py`)

## Phase 3: Data Pipeline

- [x] **3.1** Implement video loading (`io/video.py`)
- [x] **3.2** Implement ChArUco detection (`io/detection.py`)
- [x] **3.3** Implement serialization (`io/serialization.py`)

## Phase 4: Calibration Stages

- [x] **4.1** Implement intrinsic calibration (`calibration/intrinsics.py`)
- [x] **4.2** Implement extrinsic initialization (`calibration/extrinsics.py`)
- [x] **4.3** Implement interface/pose optimization (`calibration/interface_estimation.py`)
- [x] **4.4** Implement joint refinement (`calibration/refinement.py`)

## Phase 5: Validation

- [x] **5.1** Implement reprojection error computation (`validation/reprojection.py`)
- [x] **5.2** Implement 3D reconstruction metrics (`validation/reconstruction.py`)
- [x] **5.3** Implement diagnostics (`validation/diagnostics.py`)

## Phase 6: Integration

- [x] **6.1** Implement triangulation module (`triangulation/triangulate.py`)
- [x] **6.2** Implement pipeline orchestration (`calibration/pipeline.py`)
- [x] **6.3** Add CLI entry point

## Phase 7: Testing & Documentation

- [x] **7.1** Synthetic data tests (full pipeline with known ground truth)
- [ ] **7.2** Real data validation
- [ ] **7.3** Documentation and examples

---

## Post-MVP: Optimization and Feature Expansions

- [ ] **P.1** Simplify `Interface` class: remove `base_height` parameter, store only per-camera distances directly. Currently the calibration stages set `base_height=0` and put the full distance in `camera_offsets`, making `base_height` redundant. Consider whether shared base + offsets is ever needed, or if per-camera distances are always independent.

- [ ] **P.2** Consolidate synthetic data generation: Refactor existing unit tests (`test_interface_estimation.py`, `test_refinement.py`, `test_reprojection.py`) to use the centralized `tests/synthetic/ground_truth.py` module instead of duplicating `generate_synthetic_detections()` in each file.

- [x] **P.3** Performance optimization for refractive projection: Implement closed-form Newton-Raphson projection for flat interfaces (~10-20x speedup) and sparse Jacobian structure for `optimize_interface()` (~5-10x speedup). Required for practical calibration of large camera arrays (13+ cameras, 100+ frames).

- [x] **P.4** Expose `frame_step` in config and pipeline: Add `frame_step` to `CalibrationConfig`, parse from `detection.frame_step` in YAML, and pass through to `detect_all_frames()` and `calibrate_intrinsics_all()`. The parameter already exists in those functions but is never wired up from the pipeline. Essential for real-world videos (5 min × 30 FPS = 9,000 frames).

- [ ] **P.5** Standardize synthetic pipeline tests: Make test structure consistent across all scenarios in `test_full_pipeline.py`. Each scenario (ideal, realistic, minimal) should test calibration accuracy (rotation, translation, interface distance errors) and RMS reprojection error with scenario-appropriate thresholds. Remove misplaced ground-truth fixture tests (camera count, geometry checks) from calibration test classes — these already exist in `TestGenerateRealRigArray`.

- [x] **P.6** Support separate intrinsic board configuration: Allow users to specify a different ChArUco board for intrinsic (in-air) calibration vs extrinsic (underwater) calibration. Add optional `intrinsic_board` field to `CalibrationConfig`, parse it in `load_config()`, and pass it to Stage 1. Falls back to `board` when not set.

- [x] **P.7** Add `aquacal init` CLI command: Auto-generate config YAML by scanning intrinsic and extrinsic video directories. Extracts camera names from filenames via user-provided regex (default: full stem). Warns on camera name mismatches between directories. Also fix `example_config.yaml` to include `intrinsic_board` option.

- [x] **P.8** Add camera rig and reprojection quiver visualizations: 3D plot of camera positions, optical axes, and water surface plane; per-camera quiver plots of reprojection residuals at detected corner positions, colored by magnitude.

- [x] **P.9** Expose initial interface distances in config: Allow users to provide approximate camera-to-water-surface distances (per-camera dict or single scalar) via `interface.initial_distances` in YAML config, replacing the hardcoded 0.15m default for Stage 3 initialization.

- [x] **P.10** Save board reference images at pipeline start: Generate and save PNG images of the configured ChArUco board(s) to the output directory before Stage 1, so users can visually verify their config matches the physical board. Saves both extrinsic and intrinsic boards when a separate intrinsic board is configured.

- [x] **P.11** Add progress feedback to pipeline: Wire up existing `progress_callback` parameters in `calibrate_intrinsics_all()` and `detect_all_frames()`, and set `verbose=1` for Stage 3/4 optimizers. Provides per-camera and per-frame progress during the longest-running pipeline stages.

- [x] **P.12** Add legacy ChArUco board pattern support: Add `legacy_pattern: bool = False` to `BoardConfig`, call `setLegacyPattern(True)` in `get_opencv_board()` when set. Parse from both `board` and `intrinsic_board` config sections. Needed for older printed boards that have a marker in the top-left cell instead of a solid square.

- [x] **P.13** Fix validation metrics reporting zero

- [x] **P.14** Fix progress feedback gaps: Fix detection callback passing raw frame indices instead of processed count, add `verbose` parameter to `optimize_interface()` and `joint_refinement()`, wire CLI `-v` flag to enable scipy `verbose=2` for per-iteration optimizer progress.

- [x] **P.15** Refractive PnP initialization: Replace standard `cv2.solvePnP` with refractive-corrected PnP (6-param least_squares refinement) for Stage 2 extrinsic initialization and Stage 3 initial board poses. Standard PnP ignores refraction, causing ~2m Z spread in cameras that should be coplanar and preventing Stage 3 convergence.

- [ ] **P.16** Expose Stage 4 joint refinement in config: Add `refine_intrinsics: bool` to `CalibrationConfig`, parse from `optimization.refine_intrinsics` in YAML, and wire through to pipeline. Currently hardcoded to False via `getattr` fallback. Stage 4 jointly refines extrinsics, interface distances, board poses, and per-camera `fx, fy, cx, cy`. Should only be enabled after Stage 3 converges reliably.

- [ ] **P.17** Wire up `normal_fixed` config and add normal optimization: `interface_normal_fixed` is parsed from YAML and stored in config but never passed to the optimizers — the normal is always hardcoded to `[0, 0, -1]`. Add `refractive_project_general()` (Newton-Raphson for arbitrary normals), add 2 tilt-angle parameters to the Stage 3/4 optimization vector when `normal_fixed: false`, and wire through pipeline. Needed when camera rig is tilted relative to water surface.

- [x] **P.18** Water surface Z consistency diagnostic: After Stage 3, compute `water_z = camera_z + interface_distance` per camera and report the spread. Tests the physical constraint that the water surface is a single plane. Add to pipeline printout and `diagnostics.json`.

- [ ] **P.19** Fix low quality extrinsic initialization. Optimization is compensating, but the initial guess is bad despite P.15

- [x] **P.20** Improve 3D distance validation metric: Change `compute_3d_distance_errors()` from all N-choose-2 corner pairs to adjacent-only (single known ground truth = `square_size`). Add signed error (bias detection), RMSE, and percent error. Update pipeline printout and diagnostics recommendations.

- [x] **P.21** Three-panel camera rig visualization: Refactor `plot_camera_rig()` to produce a 1×3 figure with perspective, top-down, and side-on views. Replace inline scatter plot in pipeline.py with a call to the shared function.

- [ ] **P.22** Test and document sparse Jacobian OOM fix: Add unit tests for `dense_threshold` behavior in `make_sparse_jacobian_func()` and update changelog. The function was modified to return sparse matrices for large problems to avoid OOM, but was not tested or documented.

---

## Future: Advanced Optimization

- [ ] **F.1** Ceres Solver integration: Replace scipy `least_squares` with Ceres Solver for Stage 3/4 optimization. Ceres provides automatic differentiation (exact Jacobians), built-in Schur complement solver (marginalizes board poses, reducing effective problem to ~85 camera/interface params regardless of frame count), and robust loss functions. Requires writing a custom `RefractiveCostFunction` implementing the flat-interface projection with Snell's law. Integration via pyceres or pybind11 wrapper, with scipy as fallback for users without C++ dependency.

- [ ] **F.2** Intelligent frame selection: Add a frame selection step before Stage 3 optimization that caps the number of frames at a configurable budget (~50-80) while maximizing camera coverage, pose graph connectivity, and spatial/angular diversity of board poses. Ensures bounded optimization time regardless of input video length.
