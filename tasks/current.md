# Task: Bugfix — Filter Collinear ChArUco Detections

## Objective

Fix a crash in `calibrate_intrinsics_single()` caused by frames where all detected ChArUco corners are collinear (same row on the board). OpenCV's `calibrateCamera` requires a homography from object points, which fails on collinear configurations. Add a rank check to skip degenerate frames in both intrinsic calibration and extrinsic detection paths. Generate a visualization of the problematic frame for reference.

## Bug Details

**Error**: `cv2.error: (-215:Assertion failed) matH0.size() == Size(3, 3) in function 'cv::initIntrinsicParams2D'`

**Root cause**: Some frames in camera `e3v8250`'s intrinsic video have exactly 8 detected corners all lying on a single board row (all share the same y-coordinate in object space). The `min_corners >= 8` check passes, but collinear 3D points cannot define a homography.

**Failing scenario**: Real data, 13-camera rig, first camera (`e3v8250`), intrinsic calibration.

## Context Files

Read these files before starting (in order):

1. `tasks/handoff.md` — Full bug analysis with reproduction code
2. `src/aquacal/calibration/intrinsics.py` (lines 55-84) — `calibrate_intrinsics_single()` detection loop, the primary fix location
3. `src/aquacal/io/detection.py` (lines 140-148) — `detect_all_frames()` detection loop, secondary fix location
4. `src/aquacal/core/board.py` (lines 129-148) — `get_corner_array()` returns 3D positions for corner IDs
5. `tests/unit/test_intrinsics.py` — Existing intrinsic tests

## Modify

- `src/aquacal/calibration/intrinsics.py`
- `src/aquacal/io/detection.py`
- `tests/unit/test_intrinsics.py`

## Do Not Modify

Everything not listed above. In particular:
- `src/aquacal/core/board.py`
- `src/aquacal/calibration/pipeline.py`
- `src/aquacal/config/schema.py`

## Design

### Fix 1: Intrinsic calibration (`intrinsics.py`, lines 68-73)

After the existing `detect_charuco` call and `min_corners` filter, add a collinearity check:

```python
detection = detect_charuco(frame, board)
if detection is None or detection.num_corners < min_corners:
    continue

# Skip collinear detections (degenerate for homography estimation)
obj_pts = board.get_corner_array(detection.corner_ids)
if np.linalg.matrix_rank(obj_pts[:, :2] - obj_pts[0, :2]) < 2:
    continue

all_detections.append((detection.corner_ids, detection.corners_2d))
```

The check computes the rank of the 2D object point positions (x, y only, since z=0 for all board points). Rank < 2 means the points are collinear or coincident. This adds negligible cost — it's a small SVD on an (N, 2) matrix where N is typically 8-70.

`numpy` is already imported in `intrinsics.py`.

### Fix 2: Extrinsic detection (`detection.py`, lines 144-148)

Apply the same check. This path feeds into Stage 2/3/4 and while less likely to hit collinear detections (extrinsic board is larger, typically more corners), it's the same vulnerability:

```python
detection = detect_charuco(image, board, cam_matrix, dist_coeffs)

# Filter by min_corners and collinearity
if detection is not None and detection.num_corners >= min_corners:
    obj_pts = board.get_corner_array(detection.corner_ids)
    if np.linalg.matrix_rank(obj_pts[:, :2] - obj_pts[0, :2]) >= 2:
        frame_detections[cam_name] = detection
```

`numpy` is already imported in `detection.py`. `board` is already in scope (function parameter).

### Visualization: Save problematic frame with detections overlaid

Before or after applying the fix, generate a one-off PNG showing the problematic frame from camera `e3v8250` with the collinear detections drawn on it. Save it to the output directory for reference.

Create a standalone script `scripts/visualize_collinear_frame.py`:

```python
"""One-off visualization of the collinear detection that caused the calibrateCamera crash."""

import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
from aquacal.calibration.pipeline import load_config
from aquacal.core.board import BoardGeometry
from aquacal.io.detection import detect_charuco
from aquacal.io.video import VideoSet

CONFIG_PATH = r"C:\Users\tucke\Desktop\021026\021026_calibration\config.yaml"
OUTPUT_PATH = r"C:\Users\tucke\Desktop\021026\021026_calibration\output\debug_collinear_frame.png"

c = load_config(CONFIG_PATH)
board = BoardGeometry(c.intrinsic_board)

cam = "e3v8250"
vpath = str(c.intrinsic_video_paths[cam])

with VideoSet({cam: vpath}) as vs:
    for frame_idx, frames in vs.iterate_frames(step=c.frame_step):
        frame = frames[cam]
        if frame is None:
            continue
        det = detect_charuco(frame, board)
        if det is None or det.num_corners < 8:
            continue

        obj_pts = board.get_corner_array(det.corner_ids)
        rank = np.linalg.matrix_rank(obj_pts[:, :2] - obj_pts[0, :2])

        if rank < 2:
            # Draw detected corners on the frame
            vis = frame.copy()
            for i, (corner_id, pt) in enumerate(zip(det.corner_ids, det.corners_2d)):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(vis, (x, y), 8, (0, 0, 255), 2)
                cv2.putText(vis, str(int(corner_id)), (x + 10, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Add annotation
            cv2.putText(vis,
                        f"COLLINEAR: {det.num_corners} corners, rank={rank}, frame={frame_idx}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(vis,
                        f"Camera: {cam} | Unique y-values: {np.unique(obj_pts[:, 1])}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imwrite(OUTPUT_PATH, vis)
            print(f"Saved visualization to {OUTPUT_PATH}")
            print(f"  Frame index: {frame_idx}")
            print(f"  Corners: {det.num_corners}, IDs: {det.corner_ids.tolist()}")
            print(f"  Object y-values: {np.unique(obj_pts[:, 1])}")
            break
    else:
        print("No collinear frame found (bug may already be filtered)")
```

Run this script BEFORE applying the fix so the collinear frame is still reachable. If the fix is already applied, the script will report that no collinear frame was found, which is also fine.

## Acceptance Criteria

- [ ] `calibrate_intrinsics_single()` skips frames where detected object points have rank < 2
- [ ] `detect_all_frames()` skips detections where detected object points have rank < 2
- [ ] New test in `test_intrinsics.py`: frame with collinear corners (all on one board row) is filtered out
- [ ] Visualization script `scripts/visualize_collinear_frame.py` created and run
- [ ] `debug_collinear_frame.png` saved (or script reports no collinear frame found if fix was applied first)
- [ ] Existing tests pass: `pytest tests/unit/test_intrinsics.py -v`
- [ ] Pipeline test still passes: `pytest tests/unit/test_pipeline.py -v`
- [ ] Do NOT run the synthetic test suite
- [ ] Do NOT re-run the full calibration pipeline (separate task)

## Notes

1. **Run visualization FIRST**: The script finds the collinear frame by iterating the video. After the fix is applied, the frame will still exist in the video but the script handles both cases gracefully.

2. **The rank check is cheap**: `np.linalg.matrix_rank` on an (N, 2) matrix is a tiny SVD. N is typically 8-70 corners. No performance concern.

3. **Why both files**: The intrinsic path hits this bug now. The extrinsic path uses the same `min_corners` pattern and could hit it on a different dataset. Defensive fix in both locations.

4. **Test approach**: Create a mock detection with corners that are collinear in object space (e.g., corner IDs 0-7 on a board where those all fall on row 0). Verify the frame is excluded from calibration.

## Model Recommendation

**Sonnet** — Straightforward bug fix with clear reproduction steps. Two insertion points, one test, one script. No architectural decisions needed.