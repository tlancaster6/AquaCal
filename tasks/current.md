# Task: P.23 Water Surface Consistency Regularization

## Objective

Add soft regularization residuals to the Stage 3/4 cost function that penalize inconsistency in the inferred water surface Z-coordinate across cameras, breaking the depth-interface distance degeneracy.

## Background

For cameras looking straight down through a flat horizontal interface, the optimizer can trade camera Z position against interface distance without affecting reprojection error — moving a camera down by X and reducing interface distance by X produces nearly identical projections. This causes a real-world 12-camera calibration to show 451mm of water surface Z spread despite excellent reprojection (0.88 px). Adding soft residuals that penalize water_z deviation across cameras breaks this degeneracy.

## Context Files

Read these files before starting (in order):

1. `src/aquacal/calibration/_optim_common.py` — Core shared optimization code:
   - `pack_params()` / `unpack_params()` (lines 26-180): Parameter layout. Camera extrinsics are packed as `[rvec, tvec]` per non-reference camera, followed by interface distances per camera, followed by board poses. The camera Z position is `t[5]` within each camera's 6 params (tvec[2]). Interface distances start at offset `6*(n_cams-1)`.
   - `compute_residuals()` (lines 342-441): Returns 1D array of `[rx, ry, ...]` pixel residuals. **New regularization residuals should be appended here.**
   - `build_jacobian_sparsity()` (lines 183-281): Builds sparsity pattern. **Must be extended with rows for the new residuals.**
   - `make_sparse_jacobian_func()` (lines 444-485): Creates sparse FD Jacobian callable. No changes needed — it operates on whatever sparsity pattern is provided.
2. `src/aquacal/calibration/interface_estimation.py` (lines 117-285) — `optimize_interface()`: Stage 3 entry point. Constructs `cost_args` tuple and passes to `compute_residuals`. Must pass the new weight parameter through.
3. `src/aquacal/calibration/refinement.py` (lines 30-207) — `joint_refinement()`: Stage 4 entry point. Same structure as Stage 3. Must also pass the weight.
4. `src/aquacal/config/schema.py` (lines 173-219) — `CalibrationConfig` dataclass. Add the new config field here.
5. `src/aquacal/calibration/pipeline.py` (lines 472-530) — Stage 3/4 call sites. Wire the config value to the optimizer calls.

## Modify

- `src/aquacal/calibration/_optim_common.py`
- `src/aquacal/calibration/interface_estimation.py`
- `src/aquacal/calibration/refinement.py`
- `src/aquacal/config/schema.py`
- `src/aquacal/calibration/pipeline.py`

## Do Not Modify

Everything not listed above. In particular:
- `src/aquacal/core/refractive_geometry.py` — no changes to projection
- `TASKS.md` — orchestrator maintains this
- Test files — regularization is tested via pipeline behavior, not unit tests

## Design

### Part 1: Regularization Residuals in `compute_residuals()`

Add a `water_z_weight: float = 0.0` parameter to `compute_residuals()`. When `weight > 0`, append regularization residuals after the reprojection residuals:

For each camera (including reference), compute:
```
water_z_i = C_i[2] + d_i
```
where `C_i` is the camera center in world coords and `d_i` is the interface distance.

Then compute the mean across all cameras:
```
water_z_mean = mean(water_z_i for all cameras)
```

Append one residual per camera:
```
r_reg_i = water_z_weight * (water_z_i - water_z_mean)
```

This gives N_cameras additional residuals (e.g., 12 for a 12-camera rig). These residuals are in meters scaled by the weight, so `water_z_weight=10.0` means 1mm of inconsistency costs 0.01 in residual space — comparable to a small fraction of a pixel.

**Implementation detail**: Compute `water_z_i` from the unpacked extrinsics and distances (already available in the function after `unpack_params`). The mean is computed fresh each evaluation, so it tracks the current parameter state.

**Important**: The regularization residuals should be appended to the residual array **after** the main reprojection residuals, in camera_order. When `water_z_weight == 0.0`, skip appending (preserve exact backward compatibility — same residual count, same behavior).

### Part 2: Sparsity Pattern in `build_jacobian_sparsity()`

Add `water_z_weight: float = 0.0` parameter. When `weight > 0`, append rows to the sparsity pattern for the regularization residuals.

Each regularization residual for camera `i` depends on:
- **Camera i's extrinsics** (6 params) — because C_i = -R_i^T @ t_i, so camera center Z depends on rvec and tvec
- **Camera i's interface distance** (1 param)
- **All other cameras' extrinsics and interface distances** — because the mean water_z depends on all cameras

Since the mean couples all cameras, each regularization residual depends on ALL camera extrinsic params and ALL interface distance params, but NOT on any board pose params or intrinsic params.

For the sparsity row for camera i's regularization residual:
```python
row = np.zeros(n_params, dtype=np.int8)
# All extrinsic params (all non-reference cameras)
row[:n_extrinsic_params] = 1
# All interface distance params
row[n_extrinsic_params:n_extrinsic_params + n_distance_params] = 1
# Board pose params: 0 (not dependent)
# Intrinsic params: 0 (not dependent)
```

Append one such row per camera (N_cameras rows total). This is a single dense block, but it's only N_cameras rows out of potentially tens of thousands, so the impact on FD efficiency is negligible.

### Part 3: Wire Through `optimize_interface()` and `joint_refinement()`

Add `water_z_weight: float = 0.0` parameter to both functions. Pass it through to:
1. The `cost_args` tuple (append as last element)
2. The `build_jacobian_sparsity()` call

### Part 4: Config Schema

Add to `CalibrationConfig`:
```python
water_z_weight: float = 0.0  # 0.0 = disabled
```

### Part 5: YAML Parsing in `load_config()`

Parse from the `optimization` section:
```yaml
optimization:
  water_z_weight: 10.0  # Regularization weight for water surface consistency
```

In `load_config()`, add:
```python
water_z_weight = opt.get("water_z_weight", 0.0)
```

Pass to `CalibrationConfig` constructor.

### Part 6: Pipeline Wiring

Pass `config.water_z_weight` to both `optimize_interface()` and `joint_refinement()` calls in `pipeline.py`.

## Acceptance Criteria

- [ ] `compute_residuals()` appends N_cameras regularization residuals when `water_z_weight > 0`
- [ ] `compute_residuals()` returns identical results when `water_z_weight == 0` (backward compatible)
- [ ] `build_jacobian_sparsity()` appends matching sparsity rows when `water_z_weight > 0`
- [ ] `optimize_interface()` accepts and forwards `water_z_weight`
- [ ] `joint_refinement()` accepts and forwards `water_z_weight`
- [ ] `CalibrationConfig` has `water_z_weight: float = 0.0`
- [ ] `load_config()` parses `optimization.water_z_weight`
- [ ] Pipeline passes `water_z_weight` to Stage 3/4
- [ ] No test failures: `pytest tests/unit/ -v`
- [ ] No modifications to files outside "Modify" list

## Notes

1. **Why penalize deviation from mean (not from reference)**: The reference camera has fixed extrinsics (C_ref_z = 0), so its water_z = d_ref. Penalizing vs reference would anchor everything to d_ref's initial value. Using the mean allows the whole rig to shift to the correct water surface while keeping cameras consistent with each other.

2. **Weight tuning**: With ~60K reprojection residuals at ~1px each, the total reprojection cost is ~60K. The 12 regularization residuals at weight=10 and 0.1m deviation give 12 * (10*0.1)^2 = 12. So weight=10 is a gentle nudge. Weight=100 would make it a strong constraint. Suggest starting with `water_z_weight: 10.0` for testing and adjusting from there.

3. **Why not a hard constraint**: A shared water_z parameter (reducing N distances to 1) would be cleaner but assumes perfectly flat water and identical camera mounting. Soft regularization allows small per-camera deviations while still breaking the degeneracy.

4. **Mean dependency and Jacobian**: The mean couples all cameras, making regularization rows dense in the camera/distance block. This is correct — each camera's water_z residual truly depends on all other cameras through the mean. The board pose columns remain zero, which is the dominant cost savings.

5. **Interaction with robust loss**: The `f_scale` parameter in `least_squares` applies to ALL residuals including regularization. If using Huber loss, regularization residuals larger than `f_scale` will be downweighted. This is actually desirable — it prevents one severely outlier camera from dominating the regularization. But the weight should be chosen so that "reasonable" deviations (< ~50mm) stay within the linear region of the loss.

## Model Recommendation

**Opus** — Modifying shared optimization code with coupled residuals and sparsity patterns requires careful reasoning about parameter layout and Jacobian structure.