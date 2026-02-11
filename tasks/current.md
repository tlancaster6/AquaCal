# Task: F.2a Uniform Frame Subsampling for Optimization

## Objective

Add a `max_calibration_frames` config parameter that caps the number of frames entering Stage 3/4 optimization via uniform temporal subsampling, while letting Stages 1-2 use the full detection set.

## Context Files

Read these files before starting (in order):

1. `src/aquacal/config/schema.py` (lines 173-219) — `CalibrationConfig` dataclass. Add new field here.
2. `src/aquacal/calibration/pipeline.py` (lines 47-195) — `load_config()`. Parse the new field from YAML `optimization` section.
3. `src/aquacal/calibration/pipeline.py` (lines 350-445) — Detection, split, Stage 2, and Stage 3 call site. The subsampling goes between the split (line 378) and Stage 3 (line 429). Stage 2 (line 382) must still use the full `cal_detections`.
4. `src/aquacal/calibration/pipeline.py` (lines 461-511) — Stage 4 call site. Must use the same subsampled detections as Stage 3.
5. `src/aquacal/config/schema.py` (lines 288-310) — `DetectionResult` dataclass. The subsampling function creates a new `DetectionResult` with a subset of frames.

## Modify

Files to create or edit:

- `src/aquacal/config/schema.py`
- `src/aquacal/calibration/pipeline.py`

## Do Not Modify

Everything not listed above. In particular:
- `src/aquacal/calibration/interface_estimation.py` — receives detections, no changes needed
- `src/aquacal/calibration/refinement.py` — receives detections, no changes needed
- `src/aquacal/calibration/_optim_common.py` — sparse Jacobian logic stays as-is
- `TASKS.md` — orchestrator maintains this

## Design

### Part 1: Config Schema

Add to `CalibrationConfig`:

```python
max_calibration_frames: int | None = None  # None = no limit, use all frames
```

This goes after `holdout_fraction`. When set, uniform temporal subsampling reduces calibration frames to at most this many before Stage 3. The full set is still used for Stage 2 (extrinsic init) and validation.

### Part 2: YAML Parsing in `load_config()`

Parse from the `optimization` section (alongside `robust_loss` and `loss_scale`):

```yaml
optimization:
  robust_loss: huber
  loss_scale: 1.0
  max_calibration_frames: 100  # optional, omit or null for no limit
```

In `load_config()`, add after line 160:

```python
max_cal_frames_raw = opt.get("max_calibration_frames", None)
max_cal_frames = int(max_cal_frames_raw) if max_cal_frames_raw is not None else None
```

Pass it to the `CalibrationConfig` constructor.

### Part 3: Subsampling in Pipeline

Add a helper function in `pipeline.py`:

```python
def _subsample_detections(
    detections: DetectionResult,
    max_frames: int,
) -> DetectionResult:
    """Uniformly subsample detection frames to at most max_frames.

    Selects frames at uniform temporal intervals from the sorted frame indices,
    preserving the first and last frames.

    Args:
        detections: Full detection result
        max_frames: Maximum number of frames to keep

    Returns:
        New DetectionResult with at most max_frames frames
    """
    frame_indices = sorted(detections.frames.keys())
    if len(frame_indices) <= max_frames:
        return detections

    # Uniform selection: np.linspace to pick evenly spaced indices
    selected_positions = np.round(
        np.linspace(0, len(frame_indices) - 1, max_frames)
    ).astype(int)
    selected_frames = {frame_indices[i] for i in selected_positions}

    return DetectionResult(
        frames={k: v for k, v in detections.frames.items() if k in selected_frames},
        camera_names=detections.camera_names,
        total_frames=detections.total_frames,
    )
```

In the pipeline, insert between the split and Stage 3:

```python
# --- Subsample for optimization if configured ---
optim_detections = cal_detections
if config.max_calibration_frames is not None and len(cal_detections.frames) > config.max_calibration_frames:
    optim_detections = _subsample_detections(cal_detections, config.max_calibration_frames)
    print(f"\n[Frame Selection] Subsampled {len(cal_detections.frames)} → {len(optim_detections.frames)} frames for optimization")
```

Key points:
- **Stage 2** (`build_pose_graph`, `estimate_extrinsics`): Keep using `cal_detections` (full set)
- **Stage 3** (`optimize_interface`): Use `optim_detections`
- **Stage 4** (`joint_refinement`): Use `optim_detections`
- **Validation pose estimation**: Uses `val_detections` (unchanged)
- **Metadata** (`num_frames_used`): Report `len(optim_detections.frames)` so output reflects what actually went into optimization

### Part 4: Pipeline Printout

After the subsampling step, print the frame budget info so the user can see what happened:

```
[Frame Selection] Subsampled 503 → 100 frames for optimization
```

If no subsampling is needed (already under budget), print nothing extra.

## Acceptance Criteria

- [ ] `CalibrationConfig` has `max_calibration_frames: int | None = None` field
- [ ] `load_config()` parses `optimization.max_calibration_frames` from YAML
- [ ] When `max_calibration_frames` is set and calibration frames exceed it, Stage 3/4 receive a subsampled `DetectionResult`
- [ ] When `max_calibration_frames` is `None` or frames are already under budget, behavior is identical to current (no subsampling)
- [ ] Stage 2 (extrinsic init) still uses the full calibration frame set
- [ ] Subsampling is uniform temporal (evenly spaced across sorted frame indices)
- [ ] Pipeline prints frame selection info when subsampling occurs
- [ ] No test failures: `pytest tests/unit/ -v`
- [ ] No modifications to files outside "Modify" list

## Notes

1. **Why uniform temporal**: It's the simplest scheme that still provides decent diversity — evenly spaced frames from a moving board are naturally spread in pose space. This is the baseline for F.2; intelligent selection (coverage-based, pose-diversity) will be layered on later.

2. **Why subsample after split, not before**: The validation holdout set should remain untouched — it measures how well the calibration generalizes. Subsampling only the optimization set is the right tradeoff.

3. **Why Stage 2 keeps all frames**: `build_pose_graph` and `estimate_extrinsics` are cheap O(frames × cameras) operations that benefit from more data. More frames = better pose graph connectivity = better initial extrinsics = easier convergence in Stage 3.

4. **`np.linspace` for uniform selection**: Using `np.linspace(0, N-1, budget)` rounded to int gives evenly spaced indices that always include the first and last frame. This is better than `sorted_frames[::step]` which might not hit exactly `budget` frames.

5. **Typical usage**: With `frame_step: 10` on a 5-min 30fps video, you get ~900 raw frames, ~629 after filtering, ~503 calibration frames after 20% holdout. Setting `max_calibration_frames: 100` reduces Stage 3 from 503 to 100 frames — params drop from ~3100 to ~685, Jacobian from ~8 GiB to ~350 MiB (dense-safe).

6. **`total_frames` in subsampled result**: Keep the original `total_frames` value — it represents how many frames were processed from video, not how many survived filtering. This is consistent with how `split_detections` handles it.

## Model Recommendation

**Sonnet** — Simple config addition, YAML parsing, and a helper function. No algorithmic complexity.
