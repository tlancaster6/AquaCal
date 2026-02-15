"""ChArUco detection wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from numpy.typing import NDArray

from aquacal.config.schema import Detection, DetectionResult, FrameDetections
from aquacal.core.board import BoardGeometry
from aquacal.io.frameset import FrameSet
from aquacal.io.images import ImageSet
from aquacal.io.video import VideoSet


def detect_charuco(
    image: NDArray[np.uint8],
    board: BoardGeometry,
    camera_matrix: NDArray[np.float64] | None = None,
    dist_coeffs: NDArray[np.float64] | None = None,
) -> Detection | None:
    """
    Detect ChArUco corners in a single image.

    Uses OpenCV 4.6+ ArUco API. Converts BGR to grayscale internally if needed.

    Args:
        image: Grayscale (H, W) or BGR (H, W, 3) image as uint8
        board: Board geometry (provides OpenCV CharucoBoard)
        camera_matrix: Optional 3x3 intrinsic matrix for corner refinement
        dist_coeffs: Optional distortion coefficients for corner refinement

    Returns:
        Detection object containing corner_ids and corners_2d,
        or None if no corners detected.

    Example:
        >>> image = cv2.imread('calibration_frame.png')
        >>> board = BoardGeometry(config)
        >>> detection = detect_charuco(image, board)
        >>> if detection is not None:
        ...     print(f"Found {detection.num_corners} corners")
    """
    # Convert BGR to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Get OpenCV board
    cv_board = board.get_opencv_board()

    # Configure CharucoParameters with camera intrinsics if provided
    charuco_params = cv2.aruco.CharucoParameters()
    if camera_matrix is not None:
        charuco_params.cameraMatrix = camera_matrix
    if dist_coeffs is not None:
        charuco_params.distCoeffs = dist_coeffs

    # Create CharucoDetector with board and parameters
    detector = cv2.aruco.CharucoDetector(cv_board)
    detector.setCharucoParameters(charuco_params)

    # Detect ChArUco corners
    # Returns: (charuco_corners, charuco_ids, marker_corners, marker_ids)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
        gray
    )

    # Check if any ChArUco corners found
    if charuco_ids is None or len(charuco_ids) == 0:
        return None

    # Flatten arrays (OpenCV returns (N, 1, 2) for corners, (N, 1) for ids)
    corners_2d = charuco_corners.reshape(-1, 2)
    corner_ids = charuco_ids.flatten()

    return Detection(
        corner_ids=corner_ids.astype(np.int32), corners_2d=corners_2d.astype(np.float64)
    )


def _create_frame_source(paths: dict[str, str]) -> FrameSet:
    """
    Auto-detect frame source type and create appropriate object.

    Detects whether paths point to directories (images) or files (videos)
    and creates ImageSet or VideoSet accordingly.

    Args:
        paths: Dict mapping camera_name to path (directory or file)

    Returns:
        FrameSet implementation (ImageSet or VideoSet)

    Raises:
        ValueError: If paths contain a mix of directories and files
        FileNotFoundError: If any path does not exist

    Example:
        >>> paths = {'cam0': 'data/cam0/', 'cam1': 'data/cam1/'}
        >>> source = _create_frame_source(paths)  # Returns ImageSet
        >>> # or
        >>> paths = {'cam0': 'video0.mp4', 'cam1': 'video1.mp4'}
        >>> source = _create_frame_source(paths)  # Returns VideoSet
    """
    if not paths:
        raise ValueError("paths cannot be empty")

    # Check first path to determine type
    first_path = Path(next(iter(paths.values())))
    if not first_path.exists():
        raise FileNotFoundError(f"Path not found: {first_path}")

    is_directory = first_path.is_dir()

    # Validate all paths are the same type
    for camera_name, path_str in paths.items():
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if path.is_dir() != is_directory:
            raise ValueError(
                "Cannot mix directories and files in paths. "
                "All paths must be either directories (for images) or files (for videos)."
            )

    # Create appropriate frame source
    if is_directory:
        return ImageSet(paths)
    else:
        return VideoSet(paths)


def detect_all_frames(
    video_paths: dict[str, str] | FrameSet,
    board: BoardGeometry,
    intrinsics: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]
    | None = None,
    min_corners: int = 4,
    frame_step: int = 1,
    progress_callback: Callable[[int, int], None] | None = None,
) -> DetectionResult:
    """
    Detect ChArUco corners in all frames of synchronized frame source.

    Iterates through all cameras at each frame index, detects ChArUco corners,
    and organizes results into a DetectionResult.

    Supports both video files (VideoSet) and image directories (ImageSet)
    via automatic detection when dict of paths is passed.

    Args:
        video_paths: Dict mapping camera_name to path (video file or image dir),
                     or a FrameSet implementation (VideoSet/ImageSet).
                     If dict is passed, frame source is auto-detected.
        board: Board geometry
        intrinsics: Optional dict mapping camera_name to (K, dist_coeffs) tuple.
                    Used for corner refinement. Cameras not in dict use None.
        min_corners: Minimum corners required to keep a detection (default 4)
        frame_step: Process every Nth frame (default 1 = all frames)
        progress_callback: Optional callback(current_frame, total_frames) called
                          after processing each frame

    Returns:
        DetectionResult containing all valid detections organized by frame and camera.

    Example:
        >>> # With video files
        >>> paths = {'cam0': 'video0.mp4', 'cam1': 'video1.mp4'}
        >>> result = detect_all_frames(paths, board, min_corners=8, frame_step=5)
        >>>
        >>> # With image directories
        >>> paths = {'cam0': 'data/cam0/', 'cam1': 'data/cam1/'}
        >>> result = detect_all_frames(paths, board, min_corners=8)
        >>>
        >>> usable = result.get_frames_with_min_cameras(2)
        >>> print(f"Found {len(usable)} frames with 2+ cameras")
    """
    # Create appropriate FrameSet if dict passed
    if isinstance(video_paths, dict):
        frame_source = _create_frame_source(video_paths)
        owns_frame_source = True
    else:
        frame_source = video_paths
        owns_frame_source = False

    try:
        camera_names = frame_source.camera_names
        total_frames = frame_source.frame_count
        total_to_process = max(1, total_frames // frame_step)
        processed_count = 0
        frames: dict[int, FrameDetections] = {}

        for frame_idx, frame_dict in frame_source.iterate_frames(step=frame_step):
            processed_count += 1
            frame_detections: dict[str, Detection] = {}

            for cam_name, image in frame_dict.items():
                if image is None:
                    continue

                # Get intrinsics for this camera if available
                cam_matrix = None
                dist_coeffs = None
                if intrinsics is not None and cam_name in intrinsics:
                    cam_matrix, dist_coeffs = intrinsics[cam_name]

                # Detect corners
                detection = detect_charuco(image, board, cam_matrix, dist_coeffs)

                # Filter by min_corners and collinearity
                if detection is not None and detection.num_corners >= min_corners:
                    obj_pts = board.get_corner_array(detection.corner_ids)
                    if np.linalg.matrix_rank(obj_pts[:, :2] - obj_pts[0, :2]) >= 2:
                        frame_detections[cam_name] = detection

            # Only store frame if at least one camera detected the board
            if frame_detections:
                frames[frame_idx] = FrameDetections(
                    frame_idx=frame_idx, detections=frame_detections
                )

            # Progress callback
            if progress_callback is not None:
                progress_callback(processed_count, total_to_process)

        return DetectionResult(
            frames=frames, camera_names=camera_names, total_frames=total_frames
        )

    finally:
        # Clean up FrameSet if we created it
        if owns_frame_source:
            frame_source.close()
