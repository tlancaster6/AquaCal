# Task: P.24 Auxiliary Camera Registration

## Objective

Add support for auxiliary cameras (e.g., wide-angle overview camera) that are calibrated for intrinsics and registered to the rig coordinate system, but excluded from the joint Stage 3/4 optimization. This prevents poorly-modeled cameras from degrading the primary calibration while still placing them in the same world frame for downstream use (e.g., tracking association).

## Background

A central wide-angle camera sees the entire underwater volume (the union of all narrow cameras' FOVs) and is valuable for object tracking continuity. However, including it in Stage 3 joint optimization degrades the primary calibration — its ~4px intrinsic RMS (vs <1px for narrow cameras) introduces large residuals that bias other cameras' parameters. The solution: calibrate it separately against the fixed board poses from Stage 3, solving only its 7 free parameters (6 DOF extrinsics + 1 interface distance).

## Context Files

Read these files before starting (in order):

1. `src/aquacal/config/schema.py` (lines 173-219) — `CalibrationConfig` dataclass. Add `auxiliary_cameras` field.
2. `src/aquacal/config/schema.py` (lines 84-98) — `CameraCalibration` dataclass. May need an `is_auxiliary: bool` flag.
3. `src/aquacal/calibration/pipeline.py` (lines 47-200) — `load_config()`. Parse auxiliary cameras from YAML.
4. `src/aquacal/calibration/pipeline.py` (lines 330-720) — `run_calibration_from_config()`. Full pipeline flow. Stage 1 processes all cameras. Stages 2-3 use only `config.camera_names`. New Stage 3b goes after Stage 3 (or Stage 4), before validation.
5. `src/aquacal/calibration/interface_estimation.py` (lines 117-285) — `optimize_interface()` for reference on how the residual function works. Stage 3b is a simplified version.
6. `src/aquacal/calibration/extrinsics.py` (lines 106-192) — `refractive_solve_pnp()` for reference on single-camera refractive pose estimation. Stage 3b initial guess can come from this.
7. `src/aquacal/core/refractive_geometry.py` — `refractive_project_fast()` used in residual computation.
8. `src/aquacal/config/example_config.yaml` — Example config file. Add commented-out `auxiliary_cameras` section.
9. `src/aquacal/cli.py` (lines 274-381) — `_generate_config_yaml()`. The `aquacal init` command generates config YAML from scanned video directories. Must include `auxiliary_cameras` in generated output.

## Modify

- `src/aquacal/config/schema.py`
- `src/aquacal/calibration/pipeline.py`
- `src/aquacal/calibration/interface_estimation.py` (new function)
- `src/aquacal/config/example_config.yaml`
- `src/aquacal/cli.py`

## Do Not Modify

Everything not listed above. In particular:
- `src/aquacal/calibration/_optim_common.py` — shared optimization code stays as-is
- `src/aquacal/calibration/refinement.py` — Stage 4 unchanged
- `src/aquacal/calibration/intrinsics.py` — already handles rational model via `rational_model_cameras` parameter
- `TASKS.md` — orchestrator maintains this

## Design

### Part 1: Config Schema

Add to `CalibrationConfig`:
```python
auxiliary_cameras: list[str] = field(default_factory=list)
```

Auxiliary camera names must NOT appear in `camera_names` (the primary list). They must have entries in both `intrinsic_video_paths` and `extrinsic_video_paths`.

Add to `CameraCalibration`:
```python
is_auxiliary: bool = False
```

This flag is informational — downstream code can use it to exclude auxiliary cameras from triangulation/reconstruction.

### Part 2: YAML Config Format

```yaml
cameras: [cam0, cam1, ..., cam11]       # primary cameras (joint optimization)
auxiliary_cameras: [center_cam]          # registered post-hoc (optional)

paths:
  intrinsic_videos:
    cam0: path/to/cam0_inair.mp4
    # ... all primary cameras ...
    center_cam: path/to/center_inair.mp4    # auxiliary too
  extrinsic_videos:
    cam0: path/to/cam0_underwater.mp4
    # ... all primary cameras ...
    center_cam: path/to/center_underwater.mp4  # auxiliary too
```

### Part 3: YAML Parsing in `load_config()`

```python
auxiliary_cameras = data.get("auxiliary_cameras", [])
```

Validate:
- No overlap between `cameras` and `auxiliary_cameras`
- Each auxiliary camera has entries in both `intrinsic_video_paths` and `extrinsic_video_paths`
- If auxiliary cameras are specified, `rational_model_cameras` may include them

Pass to `CalibrationConfig` constructor.

### Part 4: Pipeline Changes

The pipeline needs these modifications to `run_calibration_from_config()`:

**Stage 1 — include auxiliary cameras**: `calibrate_intrinsics_all()` already operates on whatever paths are provided. Just ensure `config.intrinsic_video_paths` includes auxiliary cameras (it will, since they're in the `paths` section).

However, there's a subtlety: currently `intrinsic_video_paths` is built from `paths.intrinsic_videos` which may include auxiliary cameras. But only `config.camera_names` cameras are used in Stages 2-3. We need to make sure the intrinsics dict includes auxiliary cameras after Stage 1.

**Detection — include auxiliary cameras**: `detect_all_frames()` also operates on whatever paths are provided. The `extrinsic_video_paths` includes auxiliary cameras, so they'll be detected automatically. But the detection result will include the auxiliary cameras mixed in with primary cameras.

**Stages 2-3 — exclude auxiliary cameras**: Build a filtered `DetectionResult` that only includes primary cameras for Stage 2/3. Or more simply: `build_pose_graph()`, `estimate_extrinsics()`, and `optimize_interface()` only process cameras they find in their inputs. Since we only pass primary camera intrinsics/extrinsics to these functions, auxiliary cameras are automatically excluded.

The cleanest approach: after detection, split the detection data:
```python
# Filter detections to primary cameras only for Stages 2-3
primary_camera_set = set(config.camera_names)
# The detection result includes all cameras; Stage 2-3 functions
# only process cameras in the intrinsics/extrinsics dicts they receive,
# so auxiliary cameras are naturally excluded. But the frame filtering
# (min_cameras_per_frame) should only count primary cameras.
```

Actually, the simplest approach: run detection separately for primary and auxiliary cameras, OR run detection on all cameras together but pass only primary-camera intrinsics/extrinsics to Stages 2-3.

**Recommended approach**: Run detection on ALL cameras (primary + auxiliary). The existing pipeline functions only process cameras that exist in the `intrinsics` and `initial_extrinsics` dicts they receive, so auxiliary cameras in the detection results are silently ignored by Stages 2-3. For Stage 3b, we just need the auxiliary camera's detections from the same `DetectionResult`.

The key change: only pass primary camera intrinsics to `estimate_extrinsics()` and `optimize_interface()`. Currently the pipeline uses `intrinsics` (all cameras) — it needs to be filtered to primary-only for Stages 2-3.

**Stage 3b — register auxiliary cameras**: After Stage 3 (or Stage 4 if enabled), for each auxiliary camera:

```python
.# Compute mean water_z from primary cameras for regularization
primary_water_zs = [
    final_extrinsics[cam].C[2] + final_distances[cam]
    for cam in config.camera_names
]
target_water_z = float(np.mean(primary_water_zs))

for aux_cam in config.auxiliary_cameras:
    aux_ext, aux_dist, aux_rms = register_auxiliary_camera(
        camera_name=aux_cam,
        intrinsics=intrinsics[aux_cam],
        detections=all_detections,  # full detection set, not subsampled
        board_poses=board_poses_dict,  # fixed from Stage 3
        board=board,
        initial_interface_distance=config.initial_interface_distances.get(aux_cam, 0.15),
        interface_normal=interface_normal,
        n_air=config.n_air,
        n_water=config.n_water,
        target_water_z=target_water_z,
        water_z_weight=config.water_z_weight,
    )
```

**Output — merge into CalibrationResult**: Auxiliary cameras are added to `result.cameras` with `is_auxiliary=True`. They appear in `calibration.json` alongside primary cameras.

### Part 5: `register_auxiliary_camera()` Function

Add to `interface_estimation.py`:

```python
def register_auxiliary_camera(
    camera_name: str,
    intrinsics: CameraIntrinsics,
    detections: DetectionResult,
    board_poses: dict[int, BoardPose],
    board: BoardGeometry,
    initial_interface_distance: float = 0.15,
    interface_normal: Vec3 | None = None,
    n_air: float = 1.0,
    n_water: float = 1.333,
    min_corners: int = 4,
    target_water_z: float | None = None,
    water_z_weight: float = 10.0,
    verbose: int = 0,
) -> tuple[CameraExtrinsics, float, float]:
    """
    Register a single auxiliary camera against fixed board poses.

    Estimates the camera's extrinsics and interface distance by minimizing
    refractive reprojection error against known board poses from Stage 3.
    The board poses are treated as fixed ground truth.

    Args:
        camera_name: Name of the auxiliary camera
        intrinsics: Camera intrinsic parameters
        detections: Full detection results (must contain this camera's detections)
        board_poses: Fixed board poses from Stage 3 (frame_idx -> BoardPose)
        board: Board geometry
        initial_interface_distance: Starting interface distance estimate
        interface_normal: Interface normal (default [0, 0, -1])
        n_air: Refractive index of air
        n_water: Refractive index of water
        min_corners: Minimum corners per detection
        target_water_z: Target water surface Z from primary calibration.
            When provided, adds a soft regularization residual anchoring
            this camera's water_z to the primary cameras' mean.
        water_z_weight: Weight for the water_z regularization residual.
            Only used when target_water_z is not None.
        verbose: Verbosity level

    Returns:
        Tuple of (extrinsics, interface_distance, rms_error)
    """
```

**Implementation**:

1. **Collect observations**: Find all frames where this camera detected the board AND a board pose exists from Stage 3.

2. **Initial guess**: Use `refractive_solve_pnp()` on the frame with the most corners to get an initial extrinsics estimate. Initial interface distance from config.

3. **Parameter vector**: 7 parameters — `[rvec(3), tvec(3), interface_distance(1)]`

4. **Residual function**: For each frame, for each detected corner:
   - Look up the 3D board point from the fixed board pose
   - Project through the refractive model using the candidate camera extrinsics and interface distance
   - Residual = projected pixel - detected pixel

5. **Water surface regularization**: When `target_water_z` is provided, append one extra residual:
   ```
   r_reg = water_z_weight * (cam_z + interface_distance - target_water_z)
   ```
   This softly constrains the auxiliary camera to be consistent with the primary cameras' water surface. The pipeline passes in the mean water_z from Stage 3 as the target. Since there's only 1 camera and 7 params, this is just one extra residual — no sparsity concerns.

6. **Optimization**: `scipy.optimize.least_squares` with method="trf", bounds on interface distance [0.01, 2.0], Huber loss.

7. **Return**: Optimized extrinsics, interface distance, and RMS reprojection error.

### Part 6: Pipeline Printout

```
[Stage 3b] Registering auxiliary cameras...
  center_cam: 527 frames, 25529 corners
  center_cam: RMS 4.21 px, interface_d=0.892m
```

### Part 7: Serialization

The `is_auxiliary` flag needs to be saved/loaded in `calibration.json`. Check `serialization.py` to see if `CameraCalibration` fields are serialized dynamically or explicitly. If explicitly, add `is_auxiliary` to the serialization.

### Part 8: Example Config

Update `src/aquacal/config/example_config.yaml` to include a commented-out `auxiliary_cameras` section after `rational_model_cameras`:

```yaml
# Optional: auxiliary cameras registered post-hoc against the primary calibration
# These cameras go through intrinsic calibration and detection, but are excluded
# from joint optimization (Stages 2-3). Useful for wide-angle overview cameras.
# auxiliary_cameras:
#   - center_cam
```

### Part 9: CLI Init Command

Update `_generate_config_yaml()` in `cli.py` to include a commented-out `auxiliary_cameras` section in generated configs. Place it after the `rational_model_cameras` block:

```python
lines.extend([
    "",
    "# Optional: auxiliary cameras (registered post-hoc, excluded from joint optimization)",
    "# Move camera names from 'cameras' to here if they should not participate in Stage 3",
    "# auxiliary_cameras:",
])

for cam in camera_names:
    lines.append(f"  # - {cam}")
```

This lets users easily move a camera from `cameras` to `auxiliary_cameras` by commenting/uncommenting lines.

### Part 10: Rational Distortion Model Support

Auxiliary cameras must be supported by `rational_model_cameras`. This already works if:
- The auxiliary camera name appears in `rational_model_cameras` in the YAML
- `rational_model_cameras` is passed to `calibrate_intrinsics_all()` (it already is)
- `register_auxiliary_camera()` uses the intrinsics returned by Stage 1, which will already have 8 distortion coefficients if the rational model was used

No code changes needed for this — just ensure the `load_config()` validation doesn't reject rational model cameras that are auxiliary (it shouldn't, since `rational_model_cameras` is parsed independently from `cameras`). Verify this is the case.

## Acceptance Criteria

- [ ] `CalibrationConfig` has `auxiliary_cameras: list[str]` field (default empty)
- [ ] `CameraCalibration` has `is_auxiliary: bool = False` field
- [ ] `load_config()` parses `auxiliary_cameras` from YAML and validates no overlap with primary cameras
- [ ] Stage 1 calibrates intrinsics for both primary and auxiliary cameras
- [ ] Auxiliary cameras in `rational_model_cameras` use the 8-coefficient distortion model (verify no validation blocks this)
- [ ] Detection runs on all cameras (primary + auxiliary)
- [ ] Stages 2-3 operate on primary cameras only (auxiliary excluded)
- [ ] `register_auxiliary_camera()` optimizes 7 params (6 DOF extrinsics + 1 interface distance) against fixed board poses
- [ ] Auxiliary cameras appear in final `CalibrationResult` with `is_auxiliary=True`
- [ ] `is_auxiliary` flag is saved/loaded in `calibration.json`
- [ ] Pipeline prints auxiliary camera registration results
- [ ] `example_config.yaml` includes commented-out `auxiliary_cameras` section
- [ ] `aquacal init` generates config with commented-out `auxiliary_cameras` section
- [ ] No test failures: `pytest tests/unit/ -v`
- [ ] No modifications to files outside "Modify" list

## Notes

1. **Why not just use refractive_solve_pnp per-frame and average**: PnP estimates a camera-to-board transform per frame. Since the camera is fixed, we'd need to transform each estimate to world frame and average. This loses information — joint optimization over all frames simultaneously is more accurate and naturally handles outlier frames via robust loss.

2. **Detection filtering for auxiliary cameras**: The `min_cameras_per_frame` filter counts how many cameras see the board. When run on all cameras, an auxiliary camera's detections inflate this count. This is fine — it's conservative (keeps more frames). But the Stage 2 pose graph only uses primary cameras, so connectivity is unaffected.

3. **Validation**: Auxiliary cameras should probably be excluded from the validation metrics (reprojection RMS, 3D reconstruction) since their accuracy is expected to be lower. But their own reprojection RMS should be reported separately as a sanity check.

4. **Initial extrinsics for auxiliary camera**: The initial guess matters. `refractive_solve_pnp` on a single frame gives a rough pose. A better approach might be to run PnP on multiple frames and take the median translation. But for a 7-param optimization with hundreds of frames, convergence from a rough initial guess should be fine.

5. **Using all detections, not subsampled**: Stage 3b uses all frames (not the `max_calibration_frames`-subsampled set) since it's only 7 parameters — no memory/runtime concern. More data = better registration.

6. **Serialization of `is_auxiliary`**: Check whether `save_calibration()` in `serialization.py` uses `dataclasses.asdict()` or manual field listing. If manual, `is_auxiliary` needs to be added explicitly.

## Model Recommendation

**Opus** — This task involves pipeline architecture changes (filtering cameras across stages), a new optimization function, config/serialization updates, and careful reasoning about which data flows where. The `register_auxiliary_camera()` function needs to correctly assemble residuals from fixed board poses, which requires understanding the coordinate conventions.