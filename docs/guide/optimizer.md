# Optimizer Pipeline

AquaCal's calibration pipeline runs four sequential stages to produce a complete multi-camera calibration. This page explains how the pipeline works, what each stage optimizes, and the sparse Jacobian strategy used for efficient bundle adjustment.

## Pipeline Overview

The calibration proceeds through four stages:

```
Stage 1           Stage 2          Stage 3                 Stage 4
In-air        →   Extrinsic    →   Joint Refractive    →   Intrinsic
Intrinsics        Init (BFS)       Bundle Adjustment       Refinement
(OpenCV)          (PnP graph)      (nonlinear BA)          (optional)

Input:            Input:           Input:                  Input:
- in-air          - underwater     - Stage 2 extrinsics    - Stage 3 result
  videos            videos         - Stage 2 water_z       - same detections
- board           - intrinsics     - detections
  params            from Stage 1                           Output:
                                                            - refined K
Output:           Output:          Output:                   per camera
- K, dist         - R, t per       - refined R, t          - refined R, t
  per camera        camera         - global water_z        - refined water_z
                  - initial        - refined board poses   - refined boards
                    water_z
                  - board poses
```

Each stage builds on the previous one, progressively refining the calibration. Stages 3 and 4 use the same optimization infrastructure ({mod}`aquacal.calibration._optim_common`), just with different parameter sets.

## Stage 1: Intrinsic Calibration

**What it does:** Standard per-camera calibration using in-air checkerboard images.

**Method:** OpenCV's `cv2.calibrateCamera()` (pinhole model) or `cv2.fisheye.calibrate()` (equidistant fisheye model).

**Input:**
- In-air calibration videos (one per camera)
- ChArUco board parameters (squares_x, squares_y, square_size, marker_size)

**Output:**
- Camera intrinsic matrix **K** (3×3): `[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`
- Distortion coefficients (5-element or 8-element array)

**Why in-air?** Intrinsic calibration (lens distortion, focal length, principal point) requires many observations at varied angles and depths. Performing this underwater would be impractical. Since intrinsics are properties of the lens, not the medium, we can calibrate in air and use the same intrinsics for underwater observations.

See {func}`aquacal.calibration.intrinsics.calibrate_intrinsics_all` for implementation.

## Stage 2: Extrinsic Initialization

**What it does:** Initialize camera positions (extrinsics) and the water surface Z-coordinate by building a pose graph from shared underwater board observations.

**Method:**
1. Detect ChArUco corners in all cameras across all underwater frames
2. Build a **pose graph**: each camera-frame pair with detections is a node, edges connect nodes that observe the same board
3. **BFS traversal**: Starting from the reference camera (R=I, t=0), chain refractive PnP solutions through the graph to locate other cameras
4. Multi-frame averaging to reduce noise in extrinsic estimates
5. Compute initial water_z from the reference camera's average interface distance

**Input:**
- Underwater calibration videos (all cameras)
- Intrinsics from Stage 1
- ChArUco detections across all frames

**Output:**
- Initial extrinsics (R, t) for each camera
- Initial water_z (global water surface Z-coordinate)
- Initial board poses (rvec, tvec) for each frame

**Why BFS?** Cameras in an AquaCal rig typically have **non-overlapping fields of view**. Direct pairwise camera-to-camera pose estimation isn't possible. Instead, we use the calibration board as a "connector": if cam0 and cam1 both observe the same board pose, we can chain their board-to-camera transforms to get a cam0-to-cam1 transform.

See {func}`aquacal.calibration.extrinsics.estimate_extrinsics` for implementation.

## Stage 3: Joint Refractive Bundle Adjustment

**What it does:** Jointly optimize camera extrinsics, global water surface Z, and board poses to minimize refractive reprojection error.

**Method:** Nonlinear least-squares optimization using `scipy.optimize.least_squares`.

### Parameters Optimized

The parameter vector layout is:

```
[tilt(0 or 2) | extrinsics(6*(N-1)) | water_z(1) | board_poses(6*F) | intrinsics(4*N or 0)]
```

where:
- **tilt (0 or 2)**: Optional interface tilt parameters (rx, ry) if `normal_fixed=False`. Usually omitted (flat water assumption).
- **extrinsics (6 per camera)**: Rodrigues vector (3) + translation vector (3) for each **non-reference** camera. Reference camera (cam0) is fixed at R=I, t=0.
- **water_z (1)**: Global water surface Z-coordinate. Same value for all cameras.
- **board_poses (6 per frame)**: Rodrigues vector (3) + translation vector (3) for each frame's board pose.
- **intrinsics (0)**: Not optimized in Stage 3 (deferred to Stage 4 if requested).

**Example:** For a 3-camera rig with 50 frames:
- Extrinsics: 6 × (3 - 1) = 12 parameters
- water_z: 1 parameter
- Board poses: 6 × 50 = 300 parameters
- **Total: 313 parameters**

### Cost Function

The cost function computes **refractive reprojection error** for every observed corner across all cameras and frames:

$$
\text{residual}_{i,j,k} = \text{pixel}_{i,j,k}^{\text{observed}} - \text{refractive\_project}(\mathbf{Q}_{j,k}, \text{cam}_i, \text{water\_z})
$$

where:
- i = camera index
- j = frame index
- k = corner index within that frame
- **Q_{j,k}** is the 3D world position of corner k in frame j (computed from board pose)

The refractive projection uses the Newton-Raphson solver (see [Refractive Geometry](refractive_geometry.md)) to account for light bending at the water surface.

**Loss function:** Soft-L1 (Huber-like) loss for robustness to outliers:

$$
\rho(r) = 2 \left( \sqrt{1 + r^2} - 1 \right)
$$

This down-weights large residuals (e.g., detection errors, board motion blur) while preserving gradient information.

### Bounds

- **water_z**: `[0.01, 2.0]` meters (must be positive and below cameras; upper bound is generous)
- **Board tvec[2]** (Z-coordinate): Must be greater than water_z (boards are underwater)
- **Extrinsic rvec**: Unbounded (rotations can take any value)
- **Extrinsic tvec**: Unbounded (but typically stay near initial values from Stage 2)

See {func}`aquacal.calibration.interface_estimation.optimize_interface` for implementation.

:::{admonition} Gotcha: water_z is unobservable in non-refractive mode
:class: warning

When running calibration with `n_air = n_water = 1.0` (as a comparison baseline to standard calibration), water_z has **zero analytical gradient**. Light travels in straight lines regardless of the interface position.

Despite this, water_z may drift during optimization due to:
1. **Boundary penalties**: Soft constraint to keep board Z > water_z pushes water_z downward
2. **Numerical noise**: In a flat cost valley, small numerical errors accumulate

The final water_z value in non-refractive mode is arbitrary and meaningless. All other parameters (extrinsics, board poses) are unaffected.

Use refractive mode (default n_water = 1.333) for actual calibration. Non-refractive mode is useful only for controlled comparisons.
:::

## Stage 4: Intrinsic Refinement (Optional)

**What it does:** Re-optimize all parameters from Stage 3 **plus** per-camera focal length and principal point.

**Method:** Same as Stage 3, but the parameter vector includes:

```
intrinsics(4*N): [fx_0, fy_0, cx_0, cy_0, fx_1, fy_1, cx_1, cy_1, ...]
```

appended to the end (after board poses).

**When to use:**
- **Use Stage 4** if initial intrinsic calibration was noisy (few in-air frames, poor coverage)
- **Skip Stage 4** if intrinsics are well-determined from Stage 1 (typical case)

Refining intrinsics increases parameter count by 4N and can slow convergence. It's most useful when intrinsic estimates are suspect.

**Note:** Distortion coefficients are **always held fixed**. Only (fx, fy, cx, cy) are refined.

See {func}`aquacal.calibration.refinement.joint_refinement` for implementation.

## Sparse Jacobian Strategy

Bundle adjustment with hundreds of parameters is computationally expensive. The Jacobian matrix (derivatives of residuals w.r.t. parameters) is **sparse**: each reprojection residual depends on only a small subset of parameters.

### Block-Sparse Structure

Each residual (2D reprojection error of one corner) depends on:
- **6 extrinsic params** (for the camera observing it)
- **1 water_z param** (global)
- **6 board pose params** (for the frame containing it)
- **4 intrinsic params** (for the camera, if refining intrinsics)

Total: at most **14-17 columns** touched per residual row.

For a 13-camera, 100-frame rig:
- Parameters: ~630 (extrinsics + water_z + board poses)
- Each row touches ~13 columns → **98% sparse**

### Sparse Finite Differences with Dense Solver

`scipy.optimize.least_squares` supports Jacobian sparsity via the `jac_sparsity` parameter, but this **forces the LSMR trust-region solver**, which can diverge on ill-conditioned bundle adjustment problems.

AquaCal uses a **custom Jacobian callable** that:
1. Computes the Jacobian via sparse finite differences using `scipy.optimize._numdiff.approx_derivative` with column grouping
2. Returns a **dense** matrix (`.toarray()`)
3. Allows the **exact (QR) trust-region solver** to be used (stable, better convergence)

**Column grouping** reduces the number of function evaluations: independent columns can be finite-differenced simultaneously. For the 630-parameter rig:
- Without grouping: 630 evaluations
- With grouping: ~50 groups → **~12× fewer evaluations**

For very large problems (e.g., 1000+ parameters), a `dense_threshold` parameter automatically switches to sparse (LSMR) mode to avoid memory overflow.

See {func}`aquacal.calibration._optim_common.make_sparse_jacobian_func` for implementation.

:::{admonition} Note: Sparse Jacobian is an advanced optimization
:class: tip

The sparse Jacobian strategy is transparent to most users. It's enabled by default and auto-tunes based on problem size.

If you're modifying the calibration pipeline to add new parameter types, you'll need to update {func}`~aquacal.calibration._optim_common.build_jacobian_sparsity` to reflect which residuals depend on which parameters.
:::

## Auxiliary Camera Registration

Some cameras (e.g., wide-angle overview cameras) may degrade the joint optimization due to:
- Poor intrinsic calibration (fisheye lenses are harder to calibrate)
- Different viewing geometry (viewing at steep angles increases refractive effects)

AquaCal supports **auxiliary cameras** that are excluded from Stages 2-4 and registered post-hoc against fixed board poses and water_z.

**Method:**
1. Run Stages 1-4 with primary cameras only
2. For each auxiliary camera: 6-parameter (extrinsics-only) or 10-parameter (extrinsics + intrinsics) optimization against fixed board poses

This prevents auxiliary cameras from poisoning the primary solution while still providing calibrated extrinsics for all cameras.

See {func}`aquacal.calibration.interface_estimation.register_auxiliary_camera` for implementation.

## See Also

- [Refractive Geometry](refractive_geometry.md) — How refractive projection works and why it matters
- [Coordinate Conventions](coordinates.md) — Understanding extrinsics (R, t) and camera position C
- {mod}`aquacal.calibration` — Calibration pipeline API reference
- {mod}`aquacal.calibration._optim_common` — Shared optimization utilities (parameter packing, Jacobian sparsity)
