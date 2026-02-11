"""3D reconstruction metrics using known ChArUco geometry."""

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import NDArray

from aquacal.config.schema import (
    CalibrationResult,
    DetectionResult,
    Vec3,
    Vec2,
)
from aquacal.core.board import BoardGeometry
from aquacal.triangulation.triangulate import triangulate_point


def get_adjacent_corner_pairs(board: BoardGeometry) -> list[tuple[int, int]]:
    """
    Get pairs of adjacent corners on the board (separated by one square_size).

    Adjacent means horizontally or vertically neighboring on the checker grid
    (not diagonal). Each pair appears once with lower ID first.

    Args:
        board: Board geometry

    Returns:
        List of (corner_id_1, corner_id_2) tuples
    """
    cols = board.config.squares_x - 1
    rows = board.config.squares_y - 1
    pairs = []
    for corner_id in range(cols * rows):
        col = corner_id % cols
        row = corner_id // cols
        # Right neighbor
        if col + 1 < cols:
            pairs.append((corner_id, corner_id + 1))
        # Down neighbor
        if row + 1 < rows:
            pairs.append((corner_id, corner_id + cols))
    return pairs


@dataclass
class DistanceErrors:
    """Container for 3D distance error statistics.

    Attributes:
        mean: Mean absolute distance error in meters
        std: Standard deviation of distance errors in meters
        max_error: Maximum distance error observed in meters
        num_comparisons: Total number of corner pairs compared
        per_corner_pair: Optional dict mapping (id1, id2) to signed error in meters
        signed_mean: Mean signed error in meters (+ = overestimate, - = underestimate)
        rmse: Root mean squared error in meters
        percent_error: (MAE / ground_truth_distance) * 100
        num_frames: Number of frames with valid measurements
    """

    mean: float
    std: float
    max_error: float
    num_comparisons: int
    per_corner_pair: dict[tuple[int, int], float] | None = None
    signed_mean: float = 0.0
    rmse: float = 0.0
    percent_error: float = 0.0
    num_frames: int = 0


def triangulate_charuco_corners(
    calibration: CalibrationResult,
    detections: DetectionResult,
    frame_idx: int,
) -> dict[int, Vec3]:
    """
    Triangulate all ChArUco corners visible in 2+ cameras for a single frame.

    Args:
        calibration: Complete calibration result
        detections: Detection result with pixel observations
        frame_idx: Frame index to process

    Returns:
        Dict mapping corner_id to triangulated 3D position in world frame.
        Only includes corners visible in at least 2 cameras.
        Empty dict if frame not in detections or insufficient observations.
    """
    if frame_idx not in detections.frames:
        return {}

    frame_det = detections.frames[frame_idx]

    # Build mapping: corner_id -> {cam_name: pixel}
    corner_observations: dict[int, dict[str, Vec2]] = {}

    for cam_name, detection in frame_det.detections.items():
        if cam_name not in calibration.cameras:
            continue
        for i, corner_id in enumerate(detection.corner_ids):
            cid = int(corner_id)
            if cid not in corner_observations:
                corner_observations[cid] = {}
            corner_observations[cid][cam_name] = detection.corners_2d[i]

    # Triangulate corners with 2+ observations
    result = {}
    for corner_id, obs in corner_observations.items():
        if len(obs) >= 2:
            point_3d = triangulate_point(calibration, obs)
            if point_3d is not None:
                result[corner_id] = point_3d

    return result


def compute_3d_distance_errors(
    calibration: CalibrationResult,
    detections: DetectionResult,
    board: BoardGeometry,
    include_per_pair: bool = False,
) -> DistanceErrors:
    """
    Compute 3D distance errors by comparing triangulated corner distances to known geometry.

    For each frame, triangulates corners visible in 2+ cameras, then compares
    distances between adjacent corners to the expected square_size. Aggregates
    statistics across all frames.

    Args:
        calibration: Complete calibration result
        detections: Detection result with pixel observations
        board: Board geometry with known corner positions
        include_per_pair: If True, populate per_corner_pair in result

    Returns:
        DistanceErrors with aggregated statistics across all frames.

    Notes:
        - Only compares adjacent corners (horizontally or vertically neighboring)
        - Only compares corners that were both successfully triangulated
        - Expected distance is always board.config.square_size for adjacent pairs
        - Returns DistanceErrors with mean=0, std=0, num_comparisons=0 if no valid pairs
    """
    # Precompute adjacent pairs
    adjacent_pairs = get_adjacent_corner_pairs(board)
    ground_truth_distance = board.config.square_size

    all_signed_errors = []
    per_pair = {} if include_per_pair else None
    num_frames_with_measurements = 0

    for frame_idx in detections.frames:
        corners_3d = triangulate_charuco_corners(calibration, detections, frame_idx)

        if len(corners_3d) < 2:
            continue

        frame_had_measurements = False

        # Compare adjacent pairs only
        for id1, id2 in adjacent_pairs:
            if id1 in corners_3d and id2 in corners_3d:
                # Triangulated distance
                actual_dist = np.linalg.norm(corners_3d[id1] - corners_3d[id2])

                # Signed error: positive if overestimate, negative if underestimate
                signed_error = actual_dist - ground_truth_distance
                all_signed_errors.append(signed_error)
                frame_had_measurements = True

                if per_pair is not None:
                    pair_key = (id1, id2)  # Already in order from get_adjacent_corner_pairs
                    # Keep worst absolute error for each pair across frames
                    abs_error = abs(signed_error)
                    if pair_key not in per_pair or abs_error > abs(per_pair[pair_key]):
                        per_pair[pair_key] = signed_error

        if frame_had_measurements:
            num_frames_with_measurements += 1

    if not all_signed_errors:
        warnings.warn("No valid 3D distance comparisons — returning NaN")
        return DistanceErrors(
            mean=float("nan"),
            std=float("nan"),
            max_error=float("nan"),
            num_comparisons=0,
            per_corner_pair=per_pair,
            signed_mean=float("nan"),
            rmse=float("nan"),
            percent_error=float("nan"),
            num_frames=0,
        )

    signed_arr = np.array(all_signed_errors)
    abs_arr = np.abs(signed_arr)

    return DistanceErrors(
        mean=float(np.mean(abs_arr)),
        std=float(np.std(abs_arr)),
        max_error=float(np.max(abs_arr)),
        num_comparisons=len(all_signed_errors),
        per_corner_pair=per_pair,
        signed_mean=float(np.mean(signed_arr)),
        rmse=float(np.sqrt(np.mean(signed_arr**2))),
        percent_error=float(np.mean(abs_arr) / ground_truth_distance * 100),
        num_frames=num_frames_with_measurements,
    )


def compute_board_planarity_error(
    triangulated_corners: dict[int, Vec3],
) -> float | None:
    """
    Compute RMS distance of triangulated corners from best-fit plane.

    Fits a plane to the triangulated corners using SVD, then computes
    the RMS of perpendicular distances from each corner to the plane.

    Args:
        triangulated_corners: Dict mapping corner_id to 3D position

    Returns:
        RMS planarity error in meters, or None if fewer than 3 corners.

    Notes:
        - Requires at least 3 corners to fit a plane
        - Lower values indicate better triangulation consistency
    """
    if len(triangulated_corners) < 3:
        return None

    points = np.array(list(triangulated_corners.values()))  # (N, 3)

    # Center the points
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # SVD to find best-fit plane
    # The plane normal is the singular vector with smallest singular value
    _, _, Vt = np.linalg.svd(centered)
    normal = Vt[-1]  # Last row = smallest singular value direction

    # Compute distances to plane (plane passes through centroid)
    distances = np.abs(centered @ normal)

    return float(np.sqrt(np.mean(distances**2)))
