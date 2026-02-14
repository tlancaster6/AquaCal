# Architecture

**Analysis Date:** 2026-02-14

## Pattern Overview

**Overall:** Four-stage refractive calibration pipeline with distributed responsibility across orthogonal modules.

**Key Characteristics:**
- Sequential stage-based architecture (intrinsics → extrinsics initialization → joint optimization → validation)
- Clear separation between geometry computation (`core/`), calibration logic (`calibration/`), and validation (`validation/`)
- Shared optimization backbone (`_optim_common.py`) used by both Stage 3 and Stage 4
- Reusable components for projection, detection, and triangulation
- End-to-end orchestration via `pipeline.py` with configurable parameters

## Layers

**Configuration/Schema Layer:**
- Purpose: Type definitions, dataclass structures, exceptions, validation setup
- Location: `src/aquacal/config/schema.py`
- Contains: `CalibrationConfig`, `CalibrationResult`, `CameraIntrinsics`, `CameraExtrinsics`, `BoardConfig`, `InterfaceParams`, `DetectionResult`, `BoardPose`, `DiagnosticsData`, exceptions
- Depends on: NumPy for type hints
- Used by: All other modules

**Core Geometry Layer:**
- Purpose: Pure geometric computations (no optimization), including Snell's law, projection, and ray tracing
- Location: `src/aquacal/core/`
- Contains:
  - `camera.py`: `Camera` (pinhole) and `FisheyeCamera` (equidistant) models with pixel-to-ray transforms
  - `refractive_geometry.py`: Snell's law application, Newton-Raphson/Brent projection through flat or tilted interface
  - `interface_model.py`: Water surface plane parameterization
  - `board.py`: ChArUco board corner geometry and transforms
- Depends on: `config/schema.py`, OpenCV (for projection), NumPy
- Used by: Calibration stages, validation, triangulation

**Input/Output Layer:**
- Purpose: Data ingestion and output serialization
- Location: `src/aquacal/io/`
- Contains:
  - `video.py`: `VideoSet` for synchronized multi-camera video loading with lazy frames
  - `detection.py`: ChArUco corner detection with filtering (min corners, collinearity checks)
  - `serialization.py`: JSON round-trip of `CalibrationResult`
- Depends on: `core/`, `config/schema.py`, OpenCV, NumPy
- Used by: Pipeline orchestration, validation

**Calibration/Optimization Layer:**
- Purpose: Four-stage calibration pipeline and shared optimization utilities
- Location: `src/aquacal/calibration/`
- Contains:
  - `_optim_common.py`: Parameter packing/unpacking, residual computation, Jacobian sparsity construction, bounds building (used by Stages 3 and 4)
  - `intrinsics.py` (Stage 1): Per-camera in-air OpenCV intrinsic calibration
  - `extrinsics.py` (Stage 2): BFS pose graph initialization, refractive PnP, multi-frame rotation averaging
  - `interface_estimation.py` (Stage 3): Joint refractive bundle adjustment (camera extrinsics + water_z + board poses ± tilt)
  - `refinement.py` (Stage 4): Optional joint refinement adding per-camera intrinsic parameters
  - `pipeline.py`: End-to-end orchestration, config loading, detection splitting, stage sequencing, diagnostics generation
- Depends on: `core/`, `config/schema.py`, `io/`, `utils/`, SciPy, NumPy
- Used by: CLI, public API

**Triangulation Layer:**
- Purpose: Refractive ray-based 3D reconstruction from detected 2D points
- Location: `src/aquacal/triangulation/triangulate.py`
- Contains: Ray tracing through water surface, linear intersection of refracted rays
- Depends on: `core/refractive_geometry.py`, `config/schema.py`, NumPy
- Used by: Validation (3D distance errors), downstream consumers

**Validation/Diagnostics Layer:**
- Purpose: Compute error metrics, generate reports, produce comparison analyses
- Location: `src/aquacal/validation/`
- Contains:
  - `reprojection.py`: Reprojection error computation on validation set
  - `reconstruction.py`: 3D distance errors via board corner triangulation, spatial per-measurement records (x, y, z, signed_error), depth binning
  - `diagnostics.py`: Report generation, statistical summaries, visualizations (camera rig plots, error heatmaps, quiver plots, recommendations)
  - `comparison.py`: Cross-run comparison of N calibration results with quality metric deltas and parameter differences (per-camera RMS charts, position overlays, Z dumbbell charts, depth-stratified error analysis, XY heatmap grids)
- Depends on: `core/`, `config/schema.py`, `triangulation/`, NumPy, Matplotlib
- Used by: Pipeline (final validation), CLI (compare command)

**Utilities Layer:**
- Purpose: Shared transformation and coordinate utilities
- Location: `src/aquacal/utils/transforms.py`
- Contains: Rotation matrix ↔ rotation vector conversions, pose composition, inversion
- Depends on: OpenCV, NumPy
- Used by: All calibration stages, core geometry

**CLI/Public API:**
- Purpose: Command-line interface and public library exports
- Location: `src/aquacal/cli.py`, `src/aquacal/__init__.py`, `src/aquacal/__main__.py`
- Exports: `run_calibration()`, `load_config()`, `load_calibration()`, `save_calibration()`, core types
- Commands: `calibrate`, `init`, `compare`
- Depends on: All internal modules

## Data Flow

**Stage 1: Intrinsic Calibration**

```
In-air videos (VideoSet)
  → Detect ChArUco in each frame per camera
  → OpenCV calibration_calibrateCameraCharuco()
  → Output: K, dist_coeffs per camera
```

Entry point: `calibrate_intrinsics_all()` in `intrinsics.py`

**Stage 2: Extrinsic Initialization**

```
Underwater videos (VideoSet)
  → Detect ChArUco in each frame (all cameras)
  → Filter frames (min corners, min cameras per frame)
  → Build pose graph (edges = shared board observations)
  → BFS traversal: camera ordering
    For each camera in order:
      → Collect all frames seeing both reference camera and target camera
      → Refractive PnP to get board poses
      → Multi-frame rotation averaging via SO(3)
      → Extrinsic chain: reference → target
  → Output: Initial R, t per camera (reference = identity)
```

Entry point: `estimate_extrinsics()` in `extrinsics.py`

**Stage 3: Joint Refractive Optimization**

```
Optimized detections + initial extrinsics
  → Residual: refractive reprojection error across all observations
  → Parameter vector layout:
      [tilt_rx, tilt_ry (if normal_fixed=False) |
       extrinsics_6*(N-1) | water_z_1 | board_poses_6*F | intrinsics_4*N (if refining)]
  → Optimization:
      - Initial: extrinsics, water_z, board poses
      - Bounds: water_z ∈ (0.01, 2.0), extrinsics unconstrained, board poses unconstrained
      - Cost function: robust loss (Huber/soft-L1) applied to stacked residuals
      - Jacobian: sparse finite differences via column grouping → dense solver (exact TR)
      - Solver: scipy.optimize.least_squares with trust region method
  → Output: Optimized extrinsics, water_z, board poses, optimized metrics
```

Entry point: `optimize_interface()` in `interface_estimation.py`

**Stage 3b: Auxiliary Camera Registration (Optional)**

```
For each auxiliary camera:
  → Fixed board poses + water_z from Stage 3
  → Objective: minimize reprojection error (6 extrinsic params or 10 with intrinsics)
  → Output: Auxiliary camera extrinsics ± refined intrinsics
```

Entry point: `register_auxiliary_camera()` in `interface_estimation.py`

**Stage 4: Joint Intrinsic Refinement (Optional)**

```
Stage 3 output + detection data
  → Joint optimization: same as Stage 3 + per-camera fx, fy, cx, cy (4*N params)
  → Distortion coefficients remain fixed
  → Output: Refined extrinsics, water_z, board poses, intrinsics
```

Entry point: `joint_refinement()` in `refinement.py`

**Validation**

```
Held-out test frames
  → Per-frame board pose refinement: minimize reprojection error over 6 pose params (camera params fixed)
  → Reprojection errors: detected_pixel - projected_pixel (computed via refractive_project)
  → 3D distance errors:
      - Triangulate board corners via refractive ray intersection
      - Compare adjacent corner distances to ground truth
      - Spatial per-measurement records: (x, y, z, signed_error)
  → Diagnostic generation:
      - Summary statistics (RMS, MAE, per-camera breakdown)
      - Visualization: camera rig positions, reprojection heatmaps, depth-stratified error plots
      - Recommendations based on error patterns
```

Entry points: `compute_reprojection_errors()`, `compute_3d_distance_errors()`, `generate_diagnostic_report()` in `validation/`

**Comparison (Multi-Run)**

```
N calibration results (from calibration.json files)
  → Load all results with assigned labels
  → Per-run quality metrics (reprojection RMS, 3D errors)
  → Camera-by-camera comparisons (position delta, orientation delta, RMS spread)
  → Cross-run spatial analysis (optional, from detection data):
      - Depth-stratified signed error: per-run Z range normalization, N equal bins, mean error per bin
      - XY error heatmaps: project per-measurement (x, y, signed_error) onto grid, per depth bin
  → Output: CSV tables + PNG plots
```

Entry point: `compare_calibrations()` in `comparison.py`

## State Management

**Global State:**
- `water_z`: Single global Z-coordinate of water surface (shared across all cameras, optimized once)
- `interface_normal`: Typically [0, 0, -1], optionally estimated (Stage 3 with `normal_fixed=False`)

**Per-Camera State:**
- Intrinsics: K, dist_coeffs (fixed after Stage 1, optionally refined in Stage 4)
- Extrinsics: R, t (optimized in Stages 2-4)
- Camera center: C = -R^T @ t (derived, not directly stored)

**Per-Frame State:**
- Board pose: rvec, tvec (optimized in Stages 3-4)

**Detection State:**
- Frame detections: camera → (corner_ids, corners_2d, num_corners) mapping
- Indexed by frame_idx; split into calibration/validation sets early

**Optimization State:**
- Packed into 1D parameter vector via `pack_params()` in `_optim_common.py`
- Unpacked after optimization via `unpack_params()`
- Bounds and sparsity pattern computed once before optimization

## Key Abstractions

**Camera Models:**
- `Camera`: Pinhole model with radial/tangential distortion (5 or 8 coefficients)
- `FisheyeCamera`: Equidistant fisheye model (4 coefficients, `cv2.fisheye` APIs)
- Factory: `create_camera()` dispatches based on `intrinsics.is_fisheye`
- Interface: Both inherit `Camera` base and implement `pixel_to_ray_world()`, `project_3d_world()`

**Board Geometry:**
- `BoardGeometry`: Wraps OpenCV ChArUco board; computes corner 3D positions given pose (rvec, tvec)
- Transforms: Applies board pose to unit-scaled reference corners
- Two instances: extrinsic board (main) and intrinsic board (optional, for Stage 1)

**Interface Model:**
- `Interface`: Stores normal, per-camera interface distances, refractive indices
- Methods: `get_interface_point()` (camera's intersection with plane), normal orientation helpers
- Flat plane approximation; no thickness or variation

**Refractive Projection:**
- `refractive_project()`: Single point projection through interface (Newton-Raphson or Brent)
- `refractive_project_batch()`: Multiple points, vectorized
- Auto-selects solver based on normal orientation (NR for flat, Brent for tilted)

**Detection Result:**
- `DetectionResult`: Container of per-frame detections across all cameras
- Methods: `get_frames_with_min_cameras()` for filtering by coverage
- Indexing: frame_idx → `FrameDetections` → camera_name → `Detection` (corners_2d, corner_ids, num_corners)

**Calibration Result:**
- `CalibrationResult`: Output of pipeline
- Contains: per-camera `CameraCalibration`, `InterfaceParams`, `BoardConfig`, `DiagnosticsData`, `CalibrationMetadata`
- Round-tripped via JSON serialization

**Parameter Packing:**
- `pack_params()` / `unpack_params()`: Bidirectional parameter vector ↔ structured objects
- Layout driven by parameter groups: reference tilt (0 or 2) → extrinsics (6*(N-1)) → water_z (1) → board_poses (6*F) → intrinsics (0 or 4*N)
- Enables efficient optimization via `scipy.optimize.least_squares`

**Sparse Jacobian:**
- `build_jac_sparsity()`: Computes sparsity pattern as connectivity matrix
- `build_jacobian_callable()`: Wraps sparse finite differences (column grouping) + dense solver
- Optimization benefit: ~13x fewer FD evaluations vs dense (for real rigs ~685 params)

## Entry Points

**CLI - calibrate:**
- Location: `cli.py:cmd_calibrate()`
- Invoked by: `python -m aquacal calibrate config.yaml`
- Flow: Load config → validate → `run_calibration()` → pipeline execution → exit with code

**CLI - init:**
- Location: `cli.py:cmd_init()`
- Invoked by: `python -m aquacal init --intrinsic-dir ... --extrinsic-dir ...`
- Flow: Scan directories → extract camera names → generate YAML template → write to file

**CLI - compare:**
- Location: `cli.py:cmd_compare()`
- Invoked by: `python -m aquacal compare dir1/ dir2/ ... -o output_dir/`
- Flow: Load calibration.json from each dir → `compare_calibrations()` → write CSV + PNG → exit with code

**Library - run_calibration():**
- Location: `pipeline.py:run_calibration()` (wrapper) → `run_calibration_from_config()`
- Accepts: YAML path or `CalibrationConfig` object
- Flow: Orchestrates all 4 stages + validation + diagnostics → returns `CalibrationResult`

**Library - load_calibration() / save_calibration():**
- Location: `io/serialization.py`
- Interface: Path ↔ `CalibrationResult` (JSON round-trip)

## Error Handling

**Strategy:** Explicit exceptions with context; early validation of config and data

**Patterns:**

1. **Configuration errors** (`ValueError`):
   - Missing required sections in YAML
   - Invalid parameter ranges (e.g., negative distances)
   - Camera mismatches (auxiliary not in main list, fisheye not in auxiliary)
   - Missing board specifications
   - Raised in: `pipeline.py:load_config()`

2. **Data insufficiency errors** (`InsufficientDataError`):
   - Not enough detections to form pose graph
   - Zero observations in a frame
   - Raised in: `extrinsics.py` (pose graph building), `_optim_common.py` (residual computation)

3. **Connectivity errors** (`ConnectivityError`):
   - Pose graph disconnected (some cameras unreachable from reference)
   - Raised in: `extrinsics.py:build_pose_graph()` (BFS traversal)

4. **Calibration errors** (`CalibrationError`, umbrella exception):
   - Any stage failure: intrinsics, extrinsics, optimization, validation
   - Caught in CLI, returned as exit code 3
   - Raised in: Stage entry points

5. **Projection failures** (handled gracefully):
   - `refractive_project()` returns `None` on TIR or no intersection
   - Downstream: Cost function assigns high residual (100.0 px)
   - Reprojection/triangulation: Skip observation, accumulate only valid ones

6. **Optimization convergence**:
   - Not treated as hard error; warnings issued if solver fails to converge
   - Result is still returned (may be suboptimal)

## Cross-Cutting Concerns

**Logging:** Console output via print() for progress (detection frames, stage completions, metrics). Verbose flag (`-v`) enables per-iteration optimizer output.

**Validation:**
- Intrinsic sanity checks post-Stage 1 (focal length range, principal point in image)
- Camera height spread monitoring (warns if cameras far from water surface)
- Reprojection RMS checks (warns if > 2.0 px on validation set)

**Authentication:** None (not required for a local calibration tool)

**Coordinate Conventions:**
- World: Z-down (into water), origin at reference camera, units meters
- Camera: OpenCV (Z-forward, X-right, Y-down)
- Interface normal: [0, 0, -1] points up (water → air)
- All calculations maintain these conventions (enforced via docstrings and type hints)

---

*Architecture analysis: 2026-02-14*
