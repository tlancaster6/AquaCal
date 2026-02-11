# Task: P.14 Fix Progress Feedback Gaps

## Objective

Fix three progress feedback issues: (1) detection callback passes raw video frame indices instead of processed frame count, so the pipeline's 10%-interval logic never triggers; (2) Stage 3/4 optimizers run with no intermediate output; (3) the CLI `-v` flag is parsed but never used. Wire `-v` through to enable scipy `verbose=2` for per-iteration optimizer progress.

## Context Files

Read these files before starting (in order):

1. `src/aquacal/io/detection.py` (lines 127-161) — `detect_all_frames()` loop and callback invocation
2. `src/aquacal/calibration/interface_estimation.py` (lines 126-141, 264-275) — `optimize_interface()` signature and `least_squares` call
3. `src/aquacal/calibration/refinement.py` (lines 30-50, 178-190) — `joint_refinement()` signature and `least_squares` call
4. `src/aquacal/calibration/pipeline.py` (lines 342-354, 382-398, 401-425) — Pipeline detection callback and Stage 3/4 calls
5. `src/aquacal/cli.py` (lines 46-51, 102-150) — `-v` flag definition and `cmd_calibrate()`
6. `tests/unit/test_detection.py` — Existing detection tests

## Modify

- `src/aquacal/io/detection.py`
- `src/aquacal/calibration/interface_estimation.py`
- `src/aquacal/calibration/refinement.py`
- `src/aquacal/calibration/pipeline.py`
- `src/aquacal/cli.py`
- `tests/unit/test_detection.py`

## Do Not Modify

Everything not listed above.

## Design

### Fix 1: Detection callback (`detection.py`, lines 127-161)

**Bug**: `progress_callback(frame_idx + 1, total_frames)` passes the raw video frame index (e.g., 0, 60, 120 for `frame_step=60`) and the total raw frame count (~8000). The pipeline's callback prints at ~10% intervals using `current % max(1, total // 10) == 0`, which requires `current` divisible by ~800. With stepped frame indices, this never triggers.

**Fix**: Track a processed frame counter and compute the total frames to process:

```python
total_to_process = max(1, total_frames // frame_step)
processed_count = 0

for frame_idx, frame_dict in video_set.iterate_frames(step=frame_step):
    processed_count += 1
    # ... existing detection logic ...

    # Progress callback
    if progress_callback is not None:
        progress_callback(processed_count, total_to_process)
```

Add `total_to_process` and `processed_count = 0` before the loop (around line 129). Increment `processed_count` at the top of the loop body. Replace line 161 with `progress_callback(processed_count, total_to_process)`.

### Fix 2: Stage 3 verbose (`interface_estimation.py`)

Add `verbose: int = 0` parameter to `optimize_interface()`:

```python
def optimize_interface(
    ...
    use_fast_projection: bool = True,
    use_sparse_jacobian: bool = True,
    verbose: int = 0,  # <-- add this
) -> tuple[...]:
```

Pass it through to the `least_squares` call at line 274:

```python
    result = least_squares(
        ...
        verbose=verbose,  # was verbose=0
    )
```

Update the docstring to mention the new parameter.

### Fix 3: Stage 4 verbose (`refinement.py`)

Same pattern — add `verbose: int = 0` to `joint_refinement()` and pass through to `least_squares` at line 189. Update the docstring.

### Fix 4: Pipeline verbose parameter (`pipeline.py`)

Add `verbose: bool = False` parameter to both `run_calibration()` and `run_calibration_from_config()`:

```python
def run_calibration(config_path: str | Path, verbose: bool = False) -> CalibrationResult:
    config = load_config(config_path)
    return run_calibration_from_config(config, verbose=verbose)

def run_calibration_from_config(config: CalibrationConfig, verbose: bool = False) -> CalibrationResult:
    ...
```

Pass `verbose=2 if verbose else 0` to the optimizer calls:

**Stage 3** (~line 383):
```python
stage3_extrinsics, stage3_distances, stage3_poses, stage3_rms = optimize_interface(
    ...
    min_corners=config.min_corners_per_frame,
    verbose=2 if verbose else 0,
)
```

**Stage 4** (~line 411):
```python
) = joint_refinement(
    ...
    loss_scale=config.loss_scale,
    verbose=2 if verbose else 0,
)
```

scipy's `verbose=2` prints per-iteration progress: iteration number, cost, cost reduction, step norm, and optimality. `verbose=0` is silent (default when `-v` not passed).

**Note**: `verbose` is a runtime UI preference, not a calibration parameter — it does NOT go in `CalibrationConfig`.

### Fix 5: CLI wiring (`cli.py`)

In `cmd_calibrate()`, pass `args.verbose` to `run_calibration()`:

```python
# Currently (~line 143):
result = run_calibration(config_path)

# Change to:
result = run_calibration(config_path, verbose=args.verbose)
```

The `-v`/`--verbose` flag already exists in the CLI parser (line 47-51) and is stored in `args.verbose`, but was never used.

## Acceptance Criteria

- [ ] `detect_all_frames()` callback receives `(processed_count, total_to_process)` with sequential 1-based counts
- [ ] `optimize_interface()` accepts `verbose: int = 0` and passes it to `least_squares`
- [ ] `joint_refinement()` accepts `verbose: int = 0` and passes it to `least_squares`
- [ ] `run_calibration()` and `run_calibration_from_config()` accept `verbose: bool = False`
- [ ] Pipeline passes `verbose=2 if verbose else 0` to both optimizer calls
- [ ] `cmd_calibrate()` passes `args.verbose` to `run_calibration()`
- [ ] Default behavior (no `-v` flag) is unchanged: optimizers are silent
- [ ] With `-v` flag: optimizers print per-iteration progress
- [ ] New test in `test_detection.py`: callback receives sequential `(1, N), (2, N), ...` values when `frame_step > 1`
- [ ] Existing tests pass: `pytest tests/unit/test_detection.py tests/unit/test_pipeline.py tests/unit/test_interface_estimation.py tests/unit/test_refinement.py tests/unit/test_cli.py -v`
- [ ] Do NOT run the synthetic test suite

## Notes

1. **Default `verbose=0`**: Backward compatible. Only the CLI's `-v` flag triggers `verbose=2`. Direct callers (tests, scripts) get silent behavior by default.

2. **scipy verbose levels**: `0` = silent, `1` = termination report only, `2` = per-iteration progress. Level 2 is what we want for `-v` — it shows iteration count, cost, cost reduction, step norm, and optimality each iteration.

3. **`verbose=2` is NOT supported by `method='lm'`**: Stage 3/4 use `method='trf'` so this is fine. The validation pose estimation in P.13 uses `method='lm'` but that's not wired to verbose (and it's fast enough not to need it).

4. **Detection callback contract change**: The callback signature `(current: int, total: int)` is unchanged, but the *semantics* change from "raw frame index" to "processed frame count". This is a bug fix — the original intent was always sequential progress reporting.

5. **`total_to_process` is approximate**: `total_frames // frame_step` may be off by 1 depending on whether the last frame is included. Fine for progress display. `max(1, ...)` avoids division by zero.

## Model Recommendation

**Haiku** — Mechanical edits: add parameters, pass them through, wire CLI flag. One new test.
