# Task: P.11 Add Progress Feedback to Pipeline

## Objective

Wire up progress reporting for the longest-running pipeline stages so users see meaningful feedback instead of silence during real-data runs. Both `calibrate_intrinsics_all()` and `detect_all_frames()` already accept `progress_callback` parameters that the pipeline never passes. The Stage 3/4 optimizers have `verbose=0` that can be raised.

## Context Files

Read these files before starting (in order):

1. `src/aquacal/calibration/pipeline.py` (lines 290-310) — Stage 1 and extrinsic detection calls where callbacks need to be passed
2. `src/aquacal/calibration/intrinsics.py` (lines 168-215) — `calibrate_intrinsics_all()` has `progress_callback: Callable[[str, int, int], None]` (camera_name, current, total)
3. `src/aquacal/io/detection.py` (lines 82-160) — `detect_all_frames()` has `progress_callback: Callable[[int, int], None]` (current_frame, total_frames)
4. `src/aquacal/calibration/interface_estimation.py` (line 274) — `verbose=0` in `least_squares` call
5. `src/aquacal/calibration/refinement.py` (line 189) — `verbose=0` in `least_squares` call
6. `tests/unit/test_pipeline.py` — Existing tests use mocks for all stage functions

## Modify

- `src/aquacal/calibration/pipeline.py`

## Do Not Modify

Everything not listed above. In particular:
- `src/aquacal/calibration/intrinsics.py` — Already accepts `progress_callback`
- `src/aquacal/io/detection.py` — Already accepts `progress_callback`
- `src/aquacal/calibration/interface_estimation.py` — Do not change
- `src/aquacal/calibration/refinement.py` — Do not change
- `tests/unit/test_pipeline.py` — Existing tests use mocks; callbacks are optional kwargs so existing tests won't break

## Design

### Stage 1: Intrinsic Calibration

Pass a simple print callback to `calibrate_intrinsics_all()`:

```python
intrinsics_results = calibrate_intrinsics_all(
    video_paths={k: str(v) for k, v in config.intrinsic_video_paths.items()},
    board=intrinsic_board,
    min_corners=config.min_corners_per_frame,
    frame_step=config.frame_step,
    progress_callback=lambda name, cur, total: print(f"  Calibrating {name} ({cur}/{total})..."),
)
```

### Extrinsic Detection

Pass a print callback to `detect_all_frames()`. Since this processes many frames, don't print every frame — print at intervals:

```python
def _detection_progress(current: int, total: int) -> None:
    """Print detection progress at ~10% intervals."""
    if total > 0 and (current % max(1, total // 10) == 0 or current == total):
        print(f"  Frame {current}/{total} ({100 * current // total}%)")

all_detections = detect_all_frames(
    video_paths={k: str(v) for k, v in config.extrinsic_video_paths.items()},
    board=board,
    intrinsics={k: (v.K, v.dist_coeffs) for k, v in intrinsics.items()},
    min_corners=config.min_corners_per_frame,
    frame_step=config.frame_step,
    progress_callback=_detection_progress,
)
```

### Stage 3/4: Optimization

Do **not** modify `interface_estimation.py` or `refinement.py` directly. Instead, scipy's `verbose=1` prints iteration count and cost — useful but goes to stdout which is fine. Pass `verbose` through from the pipeline by adding it as a kwarg...

Actually, `optimize_interface()` and `joint_refinement()` don't expose `verbose` as a parameter. To avoid modifying those files, just print timing information around the calls instead:

```python
import time

print("\n[Stage 3] Interface and pose optimization...")
t0 = time.perf_counter()
# ... existing optimize_interface call ...
elapsed = time.perf_counter() - t0
print(f"  Stage 3 RMS: {stage3_rms:.3f} pixels ({elapsed:.1f}s)")
```

Do the same for Stage 4 (when it runs).

### Detection Progress as Local Function

Define `_detection_progress` as a local function inside `run_calibration_from_config()` or as a module-level helper. A local closure is cleanest since it won't be reused elsewhere.

## Acceptance Criteria

- [ ] `calibrate_intrinsics_all()` receives a progress callback that prints per-camera progress
- [ ] `detect_all_frames()` receives a progress callback that prints at ~10% intervals
- [ ] Stage 3 prints elapsed time after completion
- [ ] Stage 4 prints elapsed time after completion (when it runs)
- [ ] Existing tests pass: `pytest tests/unit/test_pipeline.py -v`
- [ ] Do NOT run the synthetic test suite

## Notes

1. **No new config options**: Progress is always printed. The pipeline already prints to stdout liberally — this just fills in the quiet gaps.

2. **Detection callback frequency**: With `frame_step=5` on a 5-min/30FPS video, there are ~1,800 frames. Printing every frame would flood stdout. Printing at ~10% intervals gives ~10 lines which is reasonable.

3. **Don't modify downstream modules**: The callbacks already exist. The optimizer `verbose` parameter is internal to those modules. Just add timing around the pipeline calls instead.

4. **Import `time`**: Add `import time` to pipeline.py imports.

5. **Tests won't break**: The mocked functions in `test_pipeline.py` will accept the new `progress_callback` kwargs silently since they're `MagicMock` objects.

## Model Recommendation

**Haiku** — Pure plumbing: pass two callbacks, add timing around two calls. No logic changes.