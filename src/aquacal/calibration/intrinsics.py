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


def calibrate_intrinsics_single(
    video_path: str | Path,
    board: BoardGeometry,
    max_frames: int = 100,
    min_corners: int = 8,
    frame_step: int = 1,
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

    Returns:
        Tuple of (CameraIntrinsics, reprojection_error_rms):
        - CameraIntrinsics with K, dist_coeffs (5 coefficients), and image_size
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

    # Prepare data for OpenCV calibration
    object_points = []  # 3D points in board frame
    image_points = []   # 2D points in image

    for corner_ids, corners_2d in selected:
        obj_pts = board.get_corner_array(corner_ids)
        object_points.append(obj_pts.astype(np.float32))
        image_points.append(corners_2d.astype(np.float32))

    # Run OpenCV calibration
    flags = 0  # Use defaults (5 distortion coefficients)
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

        intrinsics, error = calibrate_intrinsics_single(
            video_paths[name],
            board,
            max_frames=max_frames,
            min_corners=min_corners,
            frame_step=frame_step,
        )
        results[name] = (intrinsics, error)

    return results
