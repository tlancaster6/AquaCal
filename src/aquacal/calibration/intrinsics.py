"""Stage 1: Per-camera intrinsic calibration."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from aquacal.config.schema import CameraIntrinsics, BoardConfig
from aquacal.core.board import BoardGeometry
from aquacal.io.video import VideoSet
from aquacal.io.detection import detect_charuco


def validate_intrinsics(
    intrinsics: CameraIntrinsics,
    camera_name: str = "",
    max_roundtrip_error_px: float = 0.5,
    expected_fx: float | None = None,
    fx_tolerance_fraction: float = 0.3,
) -> list[str]:
    """
    Validate intrinsic calibration quality with post-calibration sanity checks.

    Detects bad intrinsic calibrations (broken undistortion, unstable distortion
    model, implausible focal length) before they cascade into wrong extrinsics.

    Args:
        intrinsics: The calibration result to validate
        camera_name: For warning messages (default "")
        max_roundtrip_error_px: Maximum acceptable undistortion roundtrip error
            (default 0.5 px)
        expected_fx: If provided, check that calibrated fx is within tolerance
            of this value (default None = skip check)
        fx_tolerance_fraction: Fractional tolerance for fx check (default 0.3 = 30%)

    Returns:
        List of warning strings (empty = all checks passed)

    Example:
        >>> warnings = validate_intrinsics(intrinsics, camera_name="cam0")
        >>> for w in warnings:
        ...     print(f"WARNING: {w}")
    """
    warnings = []
    K = intrinsics.K
    dist_coeffs = intrinsics.dist_coeffs
    w, h = intrinsics.image_size
    is_fisheye = intrinsics.is_fisheye

    # --- Check 1: Undistortion roundtrip ---
    # Generate grid of pixel coordinates (exclude 5% border)
    border = 0.05
    x_min, x_max = int(w * border), int(w * (1 - border))
    y_min, y_max = int(h * border), int(h * (1 - border))
    x_grid = np.linspace(x_min, x_max, 20)
    y_grid = np.linspace(y_min, y_max, 20)
    xx, yy = np.meshgrid(x_grid, y_grid)
    pixels = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)

    # Undistort pixels to get normalized coordinates
    pixels_reshaped = pixels.reshape(-1, 1, 2)
    if is_fisheye:
        D = dist_coeffs.reshape(4, 1)
        normalized = cv2.fisheye.undistortPoints(pixels_reshaped, K=K, D=D)
    else:
        normalized = cv2.undistortPoints(pixels_reshaped, K, dist_coeffs)

    normalized = normalized.reshape(-1, 2)

    # Re-project normalized coordinates back to pixels
    # Use identity R and t (points are already in camera frame)
    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)

    # Convert normalized coords to 3D points (z=1)
    points_3d = np.hstack([normalized, np.ones((len(normalized), 1))])
    points_3d = points_3d.reshape(-1, 1, 3).astype(np.float64)

    if is_fisheye:
        reprojected, _ = cv2.fisheye.projectPoints(points_3d, rvec, tvec, K, D)
    else:
        reprojected, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)

    reprojected = reprojected.reshape(-1, 2)

    # Compute max absolute error
    errors = np.abs(pixels - reprojected)
    max_err = np.max(errors)

    if max_err > max_roundtrip_error_px:
        warnings.append(
            f"[{camera_name}] Undistortion roundtrip error {max_err:.1f} px "
            f"(threshold: {max_roundtrip_error_px} px). "
            f"Intrinsic calibration may be unreliable."
        )

    # --- Check 2: Distortion monotonicity (pinhole only) ---
    if not is_fisheye:
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        # Compute r_max = distance from principal point to farthest corner
        corners = np.array([
            [0, 0], [w, 0], [0, h], [w, h]
        ], dtype=np.float64)
        # Convert to normalized coordinates
        r_values = []
        for corner in corners:
            dx = (corner[0] - cx) / fx
            dy = (corner[1] - cy) / fy
            r = np.sqrt(dx**2 + dy**2)
            r_values.append(r)
        r_max = max(r_values)

        # Sample radii from 0 to r_max
        r_samples = np.linspace(0, r_max, 200)

        # Compute distortion factor at each radius
        k1, k2, p1, p2, k3 = dist_coeffs[:5] if len(dist_coeffs) >= 5 else (*dist_coeffs, *[0]*(5-len(dist_coeffs)))

        # For 8-coeff rational model
        if len(dist_coeffs) == 8:
            k4, k5, k6 = dist_coeffs[5:8]
            numerator = 1 + k1 * r_samples**2 + k2 * r_samples**4 + k3 * r_samples**6
            denominator = 1 + k4 * r_samples**2 + k5 * r_samples**4 + k6 * r_samples**6
            distortion_factor = numerator / denominator
        else:
            # 5-coeff model
            distortion_factor = 1 + k1 * r_samples**2 + k2 * r_samples**4 + k3 * r_samples**6

        # Check for negative values
        if np.any(distortion_factor < 0):
            idx_negative = np.where(distortion_factor < 0)[0][0]
            r_val = r_samples[idx_negative]
            warnings.append(
                f"[{camera_name}] Distortion model goes negative at r={r_val:.3f} "
                f"(image covers r_max={r_max:.3f}). Model is ill-conditioned."
            )
        else:
            # Check for non-monotonicity (decreasing by more than 5%)
            diffs = np.diff(distortion_factor)
            decreases = diffs < 0
            if np.any(decreases):
                # Check if any decrease is > 5% of previous value
                for i in range(len(diffs)):
                    if diffs[i] < 0:
                        pct_decrease = abs(diffs[i]) / abs(distortion_factor[i]) if distortion_factor[i] != 0 else 0
                        if pct_decrease > 0.05:
                            r_val = r_samples[i+1]
                            warnings.append(
                                f"[{camera_name}] Distortion model is non-monotonic at r={r_val:.3f} "
                                f"(image covers r_max={r_max:.3f}). Model is ill-conditioned."
                            )
                            break

    # --- Check 3: Focal length plausibility (optional) ---
    if expected_fx is not None:
        fx_actual = K[0, 0]
        fy_actual = K[1, 1]

        # Check fx
        diff_fx = abs(fx_actual - expected_fx)
        pct_fx = diff_fx / expected_fx
        if pct_fx > fx_tolerance_fraction:
            warnings.append(
                f"[{camera_name}] Calibrated fx={fx_actual:.1f} differs from expected "
                f"{expected_fx:.1f} by {pct_fx*100:.0f}% (tolerance: {fx_tolerance_fraction*100:.0f}%)."
            )

        # Check fy similarly
        diff_fy = abs(fy_actual - expected_fx)
        pct_fy = diff_fy / expected_fx
        if pct_fy > fx_tolerance_fraction:
            warnings.append(
                f"[{camera_name}] Calibrated fy={fy_actual:.1f} differs from expected "
                f"{expected_fx:.1f} by {pct_fy*100:.0f}% (tolerance: {fx_tolerance_fraction*100:.0f}%)."
            )

    return warnings


def calibrate_intrinsics_single(
    video_path: str | Path,
    board: BoardGeometry,
    max_frames: int = 100,
    min_corners: int = 8,
    frame_step: int = 1,
    rational_model: bool = False,
    fisheye: bool = False,
) -> tuple[CameraIntrinsics, float]:
    """
    Calibrate intrinsics for a single camera from in-air video.

    Detects ChArUco corners in video frames, selects a subset with good
    spatial coverage, and runs OpenCV camera calibration.

    Args:
        video_path: Path to calibration video (in-air, no refraction)
        board: Board geometry
        max_frames: Maximum number of frames to use for calibration (default 100)
        min_corners: Minimum corners required per frame (default 8)
        frame_step: Process every Nth frame from video (default 1)
        rational_model: If True, use 8-coefficient rational distortion model
            instead of the standard 5-coefficient model. Use for wide-angle lenses.
        fisheye: If True, use equidistant fisheye calibration (cv2.fisheye.calibrate).
            Returns 4 distortion coefficients (k1-k4). Incompatible with rational_model.

    Returns:
        Tuple of (CameraIntrinsics, reprojection_error_rms):
        - CameraIntrinsics with K and dist_coeffs (5, 8, or 4 coefficients), and image_size
        - RMS reprojection error in pixels

    Raises:
        ValueError: If no valid frames found or calibration fails

    Example:
        >>> board = BoardGeometry(config)
        >>> intrinsics, error = calibrate_intrinsics_single("cam0_inair.mp4", board)
        >>> print(f"Reprojection error: {error:.3f} pixels")
    """
    video_path = Path(video_path)

    # Collect detections from video
    all_detections: list[tuple[NDArray[np.int32], NDArray[np.float64]]] = []
    image_size: tuple[int, int] | None = None

    with VideoSet({video_path.stem: str(video_path)}) as vs:
        for frame_idx, frames in vs.iterate_frames(step=frame_step):
            frame = frames[video_path.stem]
            if frame is None:
                continue

            # Capture image size from first frame
            if image_size is None:
                h, w = frame.shape[:2]
                image_size = (w, h)  # OpenCV convention: (width, height)

            # Detect ChArUco corners
            detection = detect_charuco(frame, board)
            if detection is None or detection.num_corners < min_corners:
                continue

            # Skip collinear detections (degenerate for homography estimation)
            obj_pts = board.get_corner_array(detection.corner_ids)
            if np.linalg.matrix_rank(obj_pts[:, :2] - obj_pts[0, :2]) < 2:
                continue

            all_detections.append((detection.corner_ids, detection.corners_2d))

    if not all_detections:
        raise ValueError(f"No valid frames found in {video_path}")

    if image_size is None:
        raise ValueError(f"Could not determine image size from {video_path}")

    # Select frames for calibration (with spatial coverage)
    selected = _select_calibration_frames(all_detections, max_frames, image_size)

    if len(selected) < 4:
        raise ValueError(
            f"Insufficient frames for calibration: {len(selected)} "
            f"(need at least 4)"
        )

    if fisheye and rational_model:
        raise ValueError("fisheye and rational_model are mutually exclusive")

    # Prepare data for OpenCV calibration
    object_points = []  # 3D points in board frame
    image_points = []   # 2D points in image

    for corner_ids, corners_2d in selected:
        obj_pts = board.get_corner_array(corner_ids)
        object_points.append(obj_pts.astype(np.float32))
        image_points.append(corners_2d.astype(np.float32))

    if fisheye:
        # cv2.fisheye.calibrate requires (1, N, 3) and (1, N, 2) per image
        fisheye_obj_pts = [pts.reshape(1, -1, 3) for pts in object_points]
        fisheye_img_pts = [pts.reshape(1, -1, 2) for pts in image_points]

        K_init = np.eye(3, dtype=np.float64)
        D_init = np.zeros((4, 1), dtype=np.float64)

        fisheye_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            + cv2.fisheye.CALIB_CHECK_COND
        )

        try:
            ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                fisheye_obj_pts,
                fisheye_img_pts,
                image_size,
                K_init,
                D_init,
                flags=fisheye_flags,
            )
        except cv2.error as e:
            raise ValueError(
                f"Fisheye calibration failed (ill-conditioned problem). "
                f"Try providing more calibration frames with varied board poses. "
                f"OpenCV error: {e}"
            ) from e

        intrinsics = CameraIntrinsics(
            K=K.astype(np.float64),
            dist_coeffs=D.flatten().astype(np.float64),
            image_size=image_size,
            is_fisheye=True,
        )

        return intrinsics, float(ret)

    # Run OpenCV calibration (pinhole model)
    flags = cv2.CALIB_RATIONAL_MODEL if rational_model else 0
    ret, K, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(  # type: ignore[call-overload]
        object_points,
        image_points,
        image_size,
        None,
        None,
        flags=flags,
    )

    if not ret:
        raise ValueError("OpenCV calibration failed")

    # Create CameraIntrinsics
    intrinsics = CameraIntrinsics(
        K=K.astype(np.float64),
        dist_coeffs=dist_coeffs.flatten().astype(np.float64),
        image_size=image_size,
    )

    return intrinsics, float(ret)  # ret is RMS reprojection error


def _select_calibration_frames(
    detections: list[tuple[NDArray[np.int32], NDArray[np.float64]]],
    max_frames: int,
    image_size: tuple[int, int],
) -> list[tuple[NDArray[np.int32], NDArray[np.float64]]]:
    """
    Select frames with good spatial coverage for calibration.

    Prioritizes frames where detected corners are spread across the image
    rather than clustered in one region.

    Args:
        detections: List of (corner_ids, corners_2d) tuples
        max_frames: Maximum number of frames to select
        image_size: Image dimensions (width, height)

    Returns:
        Selected subset of detections
    """
    if len(detections) <= max_frames:
        return detections

    # Score each detection by spatial coverage
    # Use standard deviation of corner positions as coverage metric
    scores = []
    for corner_ids, corners_2d in detections:
        # Normalize to [0, 1] range
        w, h = image_size
        normalized = corners_2d / np.array([w, h])

        # Coverage score: higher std = better spread
        std_x = np.std(normalized[:, 0])
        std_y = np.std(normalized[:, 1])
        coverage_score = std_x + std_y

        # Also consider number of corners (more is better)
        corner_score = len(corner_ids) / 50.0  # Normalize roughly

        scores.append(coverage_score + corner_score)

    # Select top-scoring frames
    indices = np.argsort(scores)[::-1][:max_frames]
    return [detections[i] for i in indices]


def calibrate_intrinsics_all(
    video_paths: dict[str, str | Path],
    board: BoardGeometry,
    max_frames: int = 100,
    min_corners: int = 8,
    frame_step: int = 1,
    rational_model_cameras: list[str] | None = None,
    fisheye_cameras: list[str] | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> dict[str, tuple[CameraIntrinsics, float]]:
    """
    Calibrate intrinsics for multiple cameras.

    Args:
        video_paths: Dict mapping camera_name to video file path
        board: Board geometry (same for all cameras)
        max_frames: Maximum frames per camera (default 100)
        min_corners: Minimum corners per frame (default 8)
        frame_step: Process every Nth frame (default 1)
        rational_model_cameras: List of camera names that should use the
            8-coefficient rational distortion model (default None = all use 5-coeff)
        fisheye_cameras: List of camera names that should use the equidistant
            fisheye model (default None = none use fisheye)
        progress_callback: Optional callback(camera_name, current_cam, total_cams)

    Returns:
        Dict mapping camera_name to (CameraIntrinsics, reprojection_error_rms)

    Raises:
        ValueError: If calibration fails for any camera

    Example:
        >>> paths = {'cam0': 'cam0.mp4', 'cam1': 'cam1.mp4'}
        >>> board = BoardGeometry(config)
        >>> results = calibrate_intrinsics_all(paths, board)
        >>> for name, (intrinsics, error) in results.items():
        ...     print(f"{name}: error={error:.3f}px")
    """
    results = {}
    camera_names = sorted(video_paths.keys())
    total = len(camera_names)

    for idx, name in enumerate(camera_names):
        if progress_callback is not None:
            progress_callback(name, idx + 1, total)

        rational = rational_model_cameras is not None and name in rational_model_cameras
        use_fisheye = fisheye_cameras is not None and name in fisheye_cameras
        intrinsics, error = calibrate_intrinsics_single(
            video_paths[name],
            board,
            max_frames=max_frames,
            min_corners=min_corners,
            frame_step=frame_step,
            rational_model=rational,
            fisheye=use_fisheye,
        )
        results[name] = (intrinsics, error)

    return results
