# Troubleshooting

This page provides practical solutions to common calibration issues.

## Reference Camera Choice

**Problem:** Stage 3/4 optimization converges but produces poor extrinsic estimates or high validation errors.

**Why it happens:** The reference camera (first camera in your config's `cameras` list) defines the world coordinate origin. If the reference camera has poor intrinsic calibration, all other cameras inherit these errors during extrinsic initialization.

**Solution:**
1. Check Stage 1 RMS reprojection errors (printed during `aquacal calibrate`)
2. Set the camera with the **lowest Stage 1 RMS** as your reference camera
3. Edit your config: move that camera to the first position in the `cameras` list
4. Re-run calibration

**Example:**
```
Stage 1 complete.
  cam0: RMS = 0.42 pixels  ← good
  cam1: RMS = 1.89 pixels  ← poor intrinsics
  cam2: RMS = 0.38 pixels  ← best, use as reference
```
In this case, set `cam2` as the reference camera (first in config).

---

## High RMS in Stage 1 (Intrinsic Calibration)

**Problem:** Stage 1 reports RMS reprojection error > 1.0 pixels for one or more cameras.

**Possible causes:**
- Too few calibration frames
- Poor spatial coverage (board always in center of frame)
- Board measurements in config don't match physical board
- Low-quality video (motion blur, compression artifacts)

**Solutions:**

### 1. Lower `frame_step` for more data
Edit your config:
```yaml
detection:
  frame_step: 5  # Change to 2 or 1 for more frames
```
This uses more frames from your in-air videos, improving intrinsic estimates.

### 2. Verify board measurements
Measure your physical ChArUco board with calipers:
- `square_size`: Distance between inner corners of adjacent squares (in meters)
- `marker_size`: Black marker side length (in meters)

Even 1mm error can degrade calibration.

### 3. Check video quality
- Ensure board is fully visible and in focus
- Move board to different positions and orientations (not just center)
- Avoid motion blur (hold board steady or use faster shutter speed)

### 4. Use `max_calibration_frames` to speed up iterations
If Stage 1 is good but Stage 3/4 are slow, limit optimization frames:
```yaml
optimization:
  max_calibration_frames: 150  # Use subset for faster Stage 3/4
```
This speeds up iteration time while you debug other issues.

---

## Bad Round-Trip Errors or Diverging Optimization

**Problem:** Stage 3 or 4 fails to converge, or validation errors are much higher than training errors.

**Possible causes:**
- Disconnected pose graph (no overlapping camera views)
- Too few shared observations between cameras
- Initial water_z estimates are far from true value
- Overfitted intrinsic model (rare, but see next section)

**Solutions:**

### 1. Check for pose graph connectivity
All cameras must observe the calibration board in at least some **shared frames**. If cameras have completely non-overlapping views, extrinsic initialization will fail.

**Fix:** Ensure underwater videos include frames where the board is visible to multiple cameras simultaneously.

### 2. Improve initial water_z estimates
Add `initial_water_z` (approximate camera-to-water Z-distances) to your config:
```yaml
interface:
  initial_water_z:
    cam0: 0.20  # Measure with a ruler (meters)
    cam1: 0.22
    cam2: 0.18
```
Better initialization helps Stage 3 converge faster and more reliably.

### 3. Check water_z bounds
Ensure the final water_z value is within the optimization bounds `[0.01, 2.0]` meters. If your cameras are farther than 2m from the water surface, you'll need to modify the bounds in the source code.

---

(camera-models-and-overfitting)=
## Camera Models and Overfitting

**Problem:** Stage 1 RMS is very low (< 0.2 pixels), but Stage 3/4 validation errors are high or undistortion produces artifacts.

**Possible cause:** Overfitted distortion model. With few calibration frames, higher-order distortion coefficients can fit noise rather than true lens distortion, causing the model to "blow up" outside the calibrated image region.

**Camera models in AquaCal:**

| Model | Distortion Coeffs | When to Use | Auto-Simplification |
|-------|-------------------|-------------|---------------------|
| **Standard (pinhole)** | 5 params: k1, k2, p1, p2, k3 | Most cameras, moderate distortion | ✅ Yes (automatic) |
| **Rational** | 8 params: k1-k6, p1, p2 | Wide-angle lenses, extreme barrel/pincushion distortion | ❌ No |
| **Fisheye** | 4 params: k1-k4 (equidistant) | Ultra-wide or fisheye lenses (> 120° FOV) | ❌ No |

**Auto-simplification (standard model only):**
If the full 5-parameter model produces a bad undistortion roundtrip (monotonicity check fails), AquaCal automatically retries with simpler models:
1. Fix k3 to 0 (3-parameter model: k1, k2, p1, p2)
2. Fix k3 and k2 to 0 (1-parameter model: k1, p1, p2)

This prevents overfitting when calibration data is sparse.

**Solutions:**

### If using rational or fisheye models:
These models do NOT auto-simplify. If you suspect overfitting:
1. Remove the camera from `rational_model_cameras` or `fisheye_cameras` in your config
2. Let it use the standard 5-parameter model (with auto-simplification)
3. Re-run calibration

### Signs of overfitting:
- Stage 1 RMS < 0.2 pixels but high round-trip error warnings in console
- Distortion correction produces visible artifacts at image edges
- Validation RMS >> training RMS in Stage 3/4

### When to use each model:
- **Standard**: Default for most lenses (up to moderate wide-angle)
- **Rational**: Use only if standard model gives RMS > 1.0 pixels and you have 50+ calibration frames
- **Fisheye**: Use only for true fisheye lenses (GoPro, ultra-wide security cameras)

For detailed theory, see the {ref}`Camera Models <camera-models>` section in the Optimizer Pipeline guide.

---

## Allowed Camera Combinations

**Problem:** Config validation fails with `ValueError` about camera lists.

**Rules:**
- `fisheye_cameras` must be a **subset** of `auxiliary_cameras`
- `fisheye_cameras` must **not overlap** with `rational_model_cameras`
- `auxiliary_cameras` must **not overlap** with `cameras` (primary cameras)

**Example (valid):**
```yaml
cameras:
  - cam0  # Reference camera
  - cam1  # Primary camera

auxiliary_cameras:
  - cam2  # Auxiliary (registered post-hoc)
  - cam3  # Auxiliary (fisheye)

rational_model_cameras:
  - cam1  # Wide-angle primary camera

fisheye_cameras:
  - cam3  # Subset of auxiliary_cameras
```

**Example (invalid):**
```yaml
cameras:
  - cam0

auxiliary_cameras:
  - cam1

fisheye_cameras:
  - cam0  # ERROR: cam0 is in cameras, not auxiliary_cameras

rational_model_cameras:
  - cam1  # ERROR: cam1 is in both rational and fisheye

fisheye_cameras:
  - cam1
```

---

## Memory and Performance Issues

**Problem:** Calibration consumes too much RAM or takes too long.

**Solution:** Limit the number of frames used in Stage 3/4 optimization:
```yaml
optimization:
  max_calibration_frames: 100  # Reduce from default (null = no limit)
```

- Stage 1 (intrinsics) always uses all detected frames (controlled by `frame_step`)
- Stage 2 (extrinsic init) uses all frames for pose graph construction
- **Stage 3/4** can be limited to a random subset for faster optimization

Reducing to 100-150 frames typically has minimal impact on calibration quality while significantly reducing memory usage and runtime.

---

## No Detections Found

**Problem:** `aquacal calibrate` fails with "No ChArUco board detected in any frame".

**Possible causes:**
- Wrong board configuration (dictionary, dimensions)
- Board out of focus or too small in frame
- Poor lighting or motion blur

**Solutions:**
1. **Verify board config matches your physical board:**
   - Check `dictionary` (e.g., `DICT_4X4_50` vs. `DICT_6X6_250`)
   - Verify `squares_x` and `squares_y` match board layout
   - Confirm `legacy_pattern` setting (check if top-left cell has a marker)

2. **Improve video quality:**
   - Ensure board fills at least 30% of frame
   - Check focus (board should be sharp)
   - Increase lighting to avoid motion blur

3. **Lower `min_corners` threshold:**
   ```yaml
   detection:
     min_corners: 8  # Lower to 4 or 6 if board is partially visible
   ```

---

## See Also

- [CLI Reference](cli.md) — Command-line usage and options
- [Optimizer Pipeline](optimizer.md) — Understanding the calibration stages
- [Glossary](glossary.md) — Definitions of key terms
