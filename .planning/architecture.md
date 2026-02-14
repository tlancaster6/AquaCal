# AquaCal Design

## Overview

AquaCal is a Python library for calibrating multi-camera arrays (up to ~14 cameras)
that view an underwater volume through a flat air-water interface. It models Snell's
law refraction at the water surface to produce accurate camera calibrations suitable
for downstream 3D reconstruction and tracking. The pipeline takes in-air and underwater
calibration videos as input and produces per-camera intrinsics, extrinsics, and
water surface parameters.

## Architecture

```
src/aquacal/
├── config/
│   └── schema.py               # Dataclasses, type aliases, config, exceptions
├── core/
│   ├── board.py                 # ChArUco board geometry
│   ├── camera.py                # Camera models (pinhole + fisheye)
│   ├── interface_model.py       # Water surface plane model
│   └── refractive_geometry.py   # Snell's law, refractive projection, ray tracing
├── io/
│   ├── video.py                 # Synchronized multi-camera video loading
│   ├── detection.py             # ChArUco corner detection
│   └── serialization.py         # JSON save/load of CalibrationResult
├── calibration/
│   ├── _optim_common.py         # Shared optimization: packing, cost, sparse Jacobian
│   ├── intrinsics.py            # Stage 1: per-camera in-air calibration
│   ├── extrinsics.py            # Stage 2: BFS pose graph initialization
│   ├── interface_estimation.py  # Stage 3: joint refractive bundle adjustment
│   ├── refinement.py            # Stage 4: optional intrinsics refinement
│   └── pipeline.py              # End-to-end orchestration and config loading
├── triangulation/
│   └── triangulate.py           # Refractive ray triangulation
├── validation/
│   ├── reprojection.py          # Reprojection error computation
│   ├── reconstruction.py        # 3D distance errors (known board geometry)
│   ├── diagnostics.py           # Reports, plots, recommendations
│   └── comparison.py            # Cross-run comparison (metrics + parameters)
├── utils/
│   └── transforms.py            # Rotation/pose utilities
└── cli.py                       # CLI entry point (calibrate, init)
```

### Module Responsibilities

**config/schema.py**: All shared types. `CalibrationConfig` (input),
`CalibrationResult` (output), camera/board/interface dataclasses, detection
containers, custom exceptions.

**core/**: Pure geometry, no optimization. `camera.py` has `Camera` and
`FisheyeCamera` (pinhole and equidistant models). `refractive_geometry.py` is the
computational heart: Snell's law, Newton-Raphson refractive projection (flat
interface), Brent fallback (tilted interface), and batch projection.
`interface_model.py` stores the water surface parameters.

**io/**: Data in/out. `VideoSet` manages synchronized videos with lazy loading.
`detection.py` wraps OpenCV ChArUco detection with filtering (min corners,
collinearity). `serialization.py` round-trips `CalibrationResult` to JSON.

**calibration/**: The four pipeline stages plus shared optimization utilities.
`_optim_common.py` is the optimization backbone -- parameter vector packing/unpacking,
residual computation, Jacobian sparsity pattern, and the sparse-FD-with-dense-solver
approach. Stages 3 and 4 both delegate to this module. `pipeline.py` orchestrates
everything and handles config parsing.

**triangulation/**: Refractive triangulation via back-projection + linear ray
intersection. Used by validation and available for downstream consumers.

**validation/**: Reprojection error, 3D reconstruction accuracy (adjacent board
corner distances), spatial reconstruction error analysis, diagnostic reporting
(spatial error maps, depth-stratified analysis, camera rig plots, quiver plots,
recommendations), and cross-run comparison.

`reconstruction.py` computes 3D distance errors by triangulating board corners
and comparing adjacent-corner distances to known ground truth. It also provides
spatially-tagged reconstruction data: each measurement is a record of
`(x, y, z, signed_error)` where `(x, y, z)` is the midpoint of the two
triangulated corners and `signed_error = measured_distance - true_distance`.
This per-measurement spatial data feeds two analyses:

- **Depth-stratified signed error**: Bin measurements by Z coordinate, compute
  signed mean error per bin. Each run's Z range is computed from its own
  triangulated data (with outlier trimming), then divided into N equal slices.
  This per-run normalization is necessary because non-refractive calibration
  produces biased camera Z positions, shifting the absolute Z of triangulated
  points. The depth axis becomes relative (shallowest to deepest) rather than
  absolute. For a correctly modeled system (refractive), the signed mean should
  be flat across depth -- errors are random noise. For a refraction-naive model,
  the signed mean should show a slope: the non-refractive optimizer tunes biased
  parameters for the mean calibration depth, producing opposite-sign errors at
  shallow vs deep extremes that cancel in aggregate metrics but reveal systematic
  depth-dependent bias when stratified.

- **XY error heatmaps per depth slice**: Within each depth bin (same per-run
  normalized slices as above), project measurements onto an XY grid and compute
  mean signed error per cell. Displayed as a grid of heatmaps (rows = calibration
  runs, columns = depth slices from shallowest to deepest, viewed from above).
  Reveals lateral spatial patterns: radial bias from unmodeled refraction
  (stronger off-axis), per-camera systematic offsets, or edge effects. A
  well-calibrated refractive model should show uniform (near-zero) heatmaps; a
  non-refractive model should show structured patterns, especially radial
  gradients that grow with depth.

`comparison.py` loads N calibration results from output directories and produces
side-by-side quality metrics (reprojection RMS, 3D reconstruction errors) and
parameter differences (camera position and orientation deltas). Outputs CSV
tables and PNG comparison plots:

- **Per-camera RMS bar chart**: Grouped bars (one color per run) showing
  reprojection RMS for each camera. Cameras on X axis, RMS on Y.
- **Camera position overlay (top-down)**: XY scatter of camera positions from
  each run, with delta arrows (quiver) from run A to run B per camera.
- **Camera Z dumbbell chart**: One row per camera, horizontal axis is Z
  position. Each run is a dot; dots for the same camera are connected by a
  line segment. Segment length shows how much the Z estimate shifted.
- **Signed error vs depth**: Line plot with one line per run. X axis is
  normalized depth (each run's Z range mapped to N equal bins from its own
  data), Y axis is signed mean reconstruction error in that depth bin. Flat
  line near zero = good; sloped line = depth-dependent bias.
- **XY heatmap grid**: Grid of subplots (rows = runs, columns = depth slices
  from shallowest to deepest). Each subplot is a birds-eye heatmap of mean
  signed reconstruction error on an XY grid. Diverging colormap centered at
  zero. Reveals spatial structure in reconstruction errors that aggregate
  metrics hide.

The comparison workflow has two tiers. The basic tier (tasks 6.1-6.4) only reads
from saved `calibration.json` files (via `load_calibration()`) and uses the
pre-computed `DiagnosticsData` quality metrics. The spatial analysis tier
(tasks 6.5-6.7) requires re-running triangulation against detection data to
produce per-measurement spatial records, so it needs both the calibration result
and the original detections. The `aquacal compare` CLI supports both: basic
comparison from output directories alone, and spatial analysis when detection
data is also provided.

Camera matching across runs is by name; cameras present in some runs but not
others are included where available and marked absent elsewhere.

**utils/transforms.py**: Thin wrappers around `cv2.Rodrigues`, pose composition,
and inversion.

## Data Flow

The calibration pipeline runs four stages sequentially:

```
In-air videos -----> Stage 1: Intrinsic Calibration
                     Per-camera OpenCV calibration (pinhole or fisheye)
                     Output: K, dist_coeffs per camera
                         |
Underwater videos -> Detection
                     ChArUco detection across all cameras/frames
                     Split into calibration and validation sets
                         |
                     Stage 2: Extrinsic Initialization
                     BFS through pose graph (shared board observations link cameras)
                     Refractive PnP for board poses, multi-frame averaging
                     Output: initial R, t per camera (reference cam = identity)
                         |
                     Stage 3: Joint Refractive Optimization
                     Jointly optimizes: extrinsics 6*(N-1), water_z (1),
                       board poses 6*F, optional tilt (2)
                     Cost: refractive reprojection error across all observations
                     Output: refined extrinsics, water_z, board poses
                         |
                     Stage 3b: Auxiliary Camera Registration (optional)
                     Registers aux cameras against fixed board poses + water_z
                     6-param (extrinsics-only) or 10-param (+intrinsics) per aux cam
                         |
                     Stage 4: Intrinsic Refinement (optional)
                     Same as Stage 3 + per-camera fx, fy, cx, cy (adds 4*N params)
                     Distortion coefficients stay fixed
                         |
                     Validation
                     Reprojection error, 3D reconstruction metrics, diagnostics
                         |
                     Output: calibration.json, diagnostics.json, plots
```

## Key Interfaces

**Config -> Pipeline**: `CalibrationConfig` carries all user settings. `load_config()`
parses YAML into this dataclass. The pipeline reads it and passes relevant pieces to
each stage.

**Stages -> _optim_common**: Stages 3 and 4 share all optimization machinery. They
provide initial parameter values and config flags; `_optim_common` handles packing,
cost evaluation, sparsity, and bounds. The parameter vector layout is:
`[tilt(0 or 2) | extrinsics(6*(N-1)) | water_z(1) | board_poses(6*F) | intrinsics(4*N or 0)]`

**Optimization -> refractive_geometry**: The cost function calls
`refractive_project()` (or `refractive_project_batch()`) for every observation. This
is the inner loop -- Newton-Raphson projection performance is critical.

**Pipeline -> Validation**: After optimization, the pipeline builds a temporary
`CalibrationResult` and passes it to reprojection/reconstruction/diagnostics along
with detection data.

**Public API**: Top-level exports are `run_calibration()`, `load_config()`,
`load_calibration()`, `save_calibration()`, and core types (`CalibrationResult`,
`CameraCalibration`, etc.).

**Comparison**: `compare_calibrations()` takes N loaded `CalibrationResult` objects
(with user-assigned labels) and returns a structured comparison. The CLI
(`aquacal compare dir1/ dir2/ ...`) loads results, calls the API, and writes
CSV tables and PNG plots to an output directory.

## Design Decisions

**Single water_z instead of per-camera distances**: Camera Z position and interface
distance are mathematically degenerate -- the optimizer can trade one for the other.
A single global `water_z` eliminates this by construction. Each camera's physical gap
is derived as `h_c = water_z - C_z`. See `dev/GEOMETRY.md` Section 4.3-4.4.

**Sparse FD with exact TR solver**: `scipy.optimize.least_squares` with `jac_sparsity`
forces the LSMR solver, which diverges on our ill-conditioned problems. Instead, a
custom Jacobian callable uses sparse finite differences (column grouping) but returns
a dense matrix, enabling the stable exact (QR) trust-region solver. Falls back to
sparse (LSMR) for very large problems to avoid OOM.

**Auxiliary cameras**: Some cameras (e.g., wide-angle overview) can degrade the joint
optimization. These are excluded from Stages 2-4 and registered post-hoc against
fixed board poses and water_z, preventing them from poisoning the primary solution.

**Newton-Raphson projection**: The flat-interface Snell equation reduces to a 1D
monotonic root-finding problem. Newton-Raphson converges in 2-4 iterations, giving
~50x speedup over Brent bracket search. Auto-selected based on interface normal.
See `dev/GEOMETRY.md` Section 6.

**Fisheye model support**: `FisheyeCamera` subclass uses `cv2.fisheye` APIs for
wide-angle auxiliary cameras. Created via `create_camera()` factory based on
`intrinsics.is_fisheye`.

**Reference camera at origin**: Camera 0 is fixed at world origin (R=I, t=0)
throughout all stages. This anchors the coordinate system and removes 6 DOF from
the optimization.

## Synthetic Refractive Comparison

A suite of synthetic experiments demonstrating when and why the refractive model
matters. Uses the 13-camera realistic rig (`generate_real_rig_array`: center +
inner ring at 300mm + outer ring at 600mm, cameras pointing down, 56 deg FOV,
0.75m above water). Both models use identical infrastructure (joint BA, board
pose optimization, same initialization) -- only the physics model differs
(n_water = 1.333 vs n_water = 1.0). This isolates the refractive modeling
contribution from methodology improvements.

Experiment logic lives in reusable functions in `tests/synthetic/`. Two access
paths:

- **Pytest** (`tests/synthetic/test_refractive_comparison.py`): Marked
  `@pytest.mark.slow`. Runs experiments and asserts on key metrics (e.g.,
  "refractive focal length error < 0.5%, non-refractive > 2%"). Plots saved
  as test artifacts. Verifies correctness in CI.
- **Script** (`tests/synthetic/compare_refractive.py`): Run as
  `python tests/synthetic/compare_refractive.py --output-dir results/`.
  Calls the same experiment functions and writes the full plot suite and CSV
  summary to a user-specified directory. For interactive analysis and
  generating publication-quality figures.

Key helper: a dense XY grid board pose generator that places the board at a
regular grid of XY positions at a fixed Z depth with minimal tilt, ensuring
dense spatial coverage for heatmaps (~49 positions per depth x ~150 adjacent
pairs per position = ~7000 measurements per depth slice).

### Experiment 1: Parameter Fidelity

Calibrate both models on the same synthetic data (realistic rig, 0.5px noise,
board trajectory spanning Z = 0.9-1.5m). Compare recovered parameters against
known ground truth.

**Predictions**: The refractive model recovers true parameters. The non-refractive
model shows biased focal length (shortened to compensate for apparent depth),
biased camera Z positions (shifted to absorb the missing refraction), and
contaminated distortion coefficients (absorbing the radially-varying refractive
field). Extrinsic XY positions should be similar; Z positions should diverge.

**Plots**:
- Per-camera focal length error bar chart: grouped bars (refractive vs
  non-refractive), showing relative error (%) for each camera. Expect near-zero
  for refractive, systematic offset for non-refractive.
- Camera position comparison (top-down): XY scatter with true positions (black),
  refractive-recovered (blue), non-refractive-recovered (red), displacement
  arrows from true to recovered. XY should be similar; the visual story is that
  both models get XY right.
- Camera Z error bar chart: one horizontal bar per camera, two bars per camera
  (refractive, non-refractive), showing signed Z position error (mm). Expect
  near-zero for refractive, systematic shift for non-refractive.
- Distortion coefficient error: grouped bars showing absolute error in k1, k2
  per camera per model. Expect larger errors for non-refractive where the radial
  refractive field leaks into distortion.

### Experiment 2: Depth Generalization

Calibrate both models on a narrow depth band (board poses at Z = 0.95-1.05m),
then evaluate reconstruction accuracy on test data at multiple specific depths.
Test depths: Z = 0.80, 0.90, 1.00, 1.10, 1.20, 1.40, 1.70, 2.00m.

At each test depth, generate board poses on a dense 7x7 XY grid spanning the
rig footprint (+/-0.5m), with minimal tilt. Triangulate corners from both
calibrations and compute per-measurement (x, y, z, signed_error) records. This
reuses the spatial analysis infrastructure from tasks 6.5-6.7.

**Predictions**: At the calibration depth (~1.0m), both models perform similarly.
Away from calibration depth, the non-refractive model shows growing signed bias
(positive at deeper depths, negative at shallower -- or vice versa). The
refractive model stays flat. XY heatmaps at off-calibration depths should show
radial spatial structure for the non-refractive model (errors correlated with
distance from rig center / off-axis angle) and uniform noise for the refractive
model.

**Plots**:
- Signed mean error vs test depth: line plot with two lines (refractive,
  non-refractive). Calibration depth band shaded. The key plot -- should show
  flat near zero for refractive, sloped/curved for non-refractive.
- RMSE vs test depth: same layout, showing total error growth.
- Scale factor vs test depth: mean(measured_distance / true_distance) at each
  depth. Should be ~1.0 for refractive, drifting from 1.0 for non-refractive.
- XY heatmap grid: rows = [refractive, non-refractive], columns = 4-5
  representative test depths (e.g., 0.80, 1.00, 1.20, 1.70m). Diverging
  colormap centered at zero, shared color scale across all cells. The flagship
  visualization -- should show uniform blue/white for refractive row, emerging
  radial patterns for non-refractive row as depth departs from calibration.

### Experiment 3: Depth Scaling

Same rig geometry. Calibrate and evaluate at the SAME depth for each trial
(no generalization penalty). Sweep the depth: Z = 0.85, 1.0, 1.2, 1.5, 2.0,
2.5m. At each depth, generate a full calibration trajectory (30 frames spanning
that depth +/- 0.1m) and evaluate reconstruction on held-out test poses at the
same depth.

This isolates the question: even when the non-refractive model is optimally
tuned for the test depth, how does its accuracy scale with depth?

**Predictions**: At shallow depths (Z = 0.85m, only ~0.1m below the water
surface), refraction is minimal and both models perform similarly. As depth
increases, incidence angles grow and the refractive correction becomes more
significant. The non-refractive model's parameter bias grows, visible as
increasing focal length error and camera Z error. Reconstruction error should
grow roughly quadratically with depth for the non-refractive model.

**Plots**:
- Reconstruction RMSE vs calibration depth: two lines. Should diverge with
  depth.
- Focal length error vs calibration depth: two lines. Shows parameter bias
  growing with depth.
- Camera Z error vs calibration depth: two lines (mean across cameras). Shows
  extrinsic bias growing with depth.
- Combined summary: 2x3 grid showing XY heatmaps at depths 0.85, 1.5, 2.5m
  for both models. At shallow depth, both look clean; at deep, only the
  refractive model survives.

## Constraints & Assumptions

- Cameras are in air, pointing approximately downward at the water surface.
- The water surface is a flat plane (no waves, no curvature).
- Interface normal is approximately [0, 0, -1]; slight tilt estimated when
  `normal_fixed: false`.
- Cameras have non-overlapping fields of view -- extrinsic estimation chains through
  shared board observations via a connected pose graph.
- ChArUco calibration board with known dimensions.
- Hardware-synchronized cameras (same frame indices across cameras).
- All internal values in meters; millimeters only for display.
- Coordinate conventions and refractive geometry detailed in `dev/GEOMETRY.md`.
