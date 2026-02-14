# AquaCal Knowledge Base

## Table of Contents
- Architecture (1 entry)
- Optimization & Performance (2 entries)
- Coordinate Frames & Geometry (2 entries)
- Calibration Lessons (1 entry)
- Known Issues & Workarounds (0 entries)
- Debugging Recipes (0 entries)

## Architecture

### Water-Z reparameterization to break height/distance degeneracy
**Context**: Camera Z position (in extrinsics) and interface distance are mathematically degenerate — the optimizer can trade one for the other, reaching valid but nonphysical solutions where cameras appear at very different heights above the water surface.
**Insight**: A single global `water_z` parameter replaces N independent per-camera interface distances. Each camera's distance is derived as `d_i = water_z - C_z_i`. This eliminates the degeneracy by construction: moving a camera's Z also changes its interface distance, so the optimizer can't play them against each other. The reference camera has `C_z = 0` (at origin), so `water_z = d_ref`. Auxiliary cameras use a 6-param (extrinsics-only) optimization since their distance is derived from the known `water_z`.
**References**: `src/aquacal/calibration/_optim_common.py:pack_params` (line 20), `src/aquacal/calibration/interface_estimation.py:optimize_interface` (line 140), CHANGELOG entry "P.18 Replace Per-Camera Interface Distances with Global Water Surface Z".
**Added**: 2026-02-12

## Optimization & Performance

### Sparse Jacobian without LSMR: custom callable approach
**Context**: Bundle adjustment in Stages 3/4 uses `scipy.optimize.least_squares`. The obvious way to exploit Jacobian sparsity — passing `jac_sparsity` — forces the LSMR trust-region solver, which diverges on our ill-conditioned problems while the dense exact (QR) solver converges fine.
**Insight**: Use a custom `jac` callable that computes the Jacobian via `scipy.optimize._numdiff.approx_derivative` with `sparsity=(pattern, groups)` and returns `.toarray()` (dense). This gives sparse finite-difference efficiency (only `len(groups)` evaluations instead of `n_params`) with the exact TR solver's stability. `group_columns()` from `scipy.optimize._numdiff` computes optimal column groupings — e.g. 13 groups instead of 33 columns for the 3-camera test case. For large problems, a `dense_threshold` parameter falls back to returning sparse (LSMR) to avoid OOM (e.g. 13-camera, 629-frame rig would need 13.5 GiB dense).
**References**: `src/aquacal/calibration/_optim_common.py:make_sparse_jacobian_func` (line 498), `group_columns` and `approx_derivative` imported from `scipy.optimize._numdiff` (line 10). `tr_solver='exact'` is incompatible with `jac_sparsity` parameter — see scipy docs.
**Added**: 2026-02-12

### Block-sparse Jacobian structure in bundle adjustment
**Context**: The Stage 3/4 cost function has natural sparsity: each reprojection residual depends only on one camera's extrinsics, the global water_z, and one board pose. Understanding this structure is essential for anyone modifying the sparsity pattern or adding new parameter types.
**Insight**: The Jacobian has a block-sparse structure where each row (residual) touches at most ~14 columns: 6 extrinsic params for one camera + 1 water_z + 6 board pose params (+ optionally 4 intrinsic params). Column grouping exploits this — independent columns can be finite-differenced simultaneously. This reduces function evaluations by 10-15x (e.g. ~50 groups instead of ~685 columns for a 13-camera rig). Adding a new parameter type requires updating `build_jacobian_sparsity()` to mark which residuals depend on it.
**References**: `src/aquacal/calibration/_optim_common.py:build_jacobian_sparsity` (line 200), `make_sparse_jacobian_func` (line 498). See also "Sparse Jacobian without LSMR" entry above for the solver-side details.
**Added**: 2026-02-12

## Coordinate Frames & Geometry

### `interface_distance` is a Z-coordinate, not a physical gap
**Context**: The name `interface_distance` suggests a camera-to-water distance, but all downstream code treats it as the Z-coordinate of the water surface.
**Insight**: Functions like Newton projection, Brent projection, and `get_interface_point` compute the camera-to-water gap internally as `h_c = interface_distance - C_z`. So `interface_distance` must be the absolute water surface Z, not the per-camera gap. When deriving from the global `water_z` parameter, the correct assignment is `interface_distance = water_z` for all cameras. Deriving it as `water_z - C_z` double-counts `C_z` because downstream code subtracts it again. This was the root cause of bug B.6.
**References**: `src/aquacal/calibration/_optim_common.py:unpack_params` (line 165), `src/aquacal/core/refractive_geometry.py` (line ~346), `src/aquacal/core/interface_model.py` (line ~81), `tasks/archive/b6_debug_report.md`.
**Added**: 2026-02-12

### Top-down camera rig plot CW/CCW flip from Y-axis convention
**Context**: The world frame is defined by the reference camera, where Y = camera-Y-down. Plotting with standard Y-up convention mirrors CW/CCW camera ordering in the top-down view.
**Insight**: Call `ax.invert_yaxis()` on the top-down subplot to match the camera Y-down convention. Z-negation and `invert_zaxis()` do NOT affect CW/CCW — they only change the vertical display direction. The CW/CCW flip is purely a Y-axis mismatch. This was the root cause of bug B.7.
**References**: `src/aquacal/validation/diagnostics.py:plot_camera_rig` (line 547), `tasks/archive/b7_report.md`.
**Added**: 2026-02-12

## Calibration Lessons

### water_z is unobservable in non-refractive mode (n_air == n_water)
**Context**: When running calibration with n_air=n_water=1.0 as a comparison baseline, water_z moves significantly (1.0 -> 0.35 -> 0.47) despite having zero analytical gradient.
**Insight**: With equal refractive indices, the projected pixel is exactly independent of water_z (Newton-Raphson converges in 0 iterations to the pinhole solution; the interface point lies on the C-to-Q ray, so perspective division cancels). Stage 3 movement is caused by the `h_q <= 0` boundary penalty driving water_z below all board corners. Stage 4 drift is accumulated numerical noise in a flat cost valley. The final water_z value is arbitrary and meaningless; all other parameters (extrinsics, intrinsics, board poses) are unaffected.
**References**: `_refractive_project_newton` line 357 (h_q guard), `compute_residuals` line 486 (100px penalty), `dev/tasks/water_z_nonrefractive_report.md`.
**Added**: 2026-02-13

## Known Issues & Workarounds

## Debugging Recipes
