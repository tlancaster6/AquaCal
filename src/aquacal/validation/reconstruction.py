"""3D reconstruction metrics using known ChArUco geometry."""

from dataclasses import dataclass

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


@dataclass
class DistanceErrors:
    """Container for 3D distance error statistics.

    Attributes:
        mean: Mean absolute distance error in meters
        std: Standard deviation of distance errors in meters
        max_error: Maximum distance error observed in meters
        num_comparisons: Total number of corner pairs compared
        per_corner_pair: Optional dict mapping (id1, id2) to error in meters
    """

    mean: float
    std: float
    max_error: float
    num_comparisons: int
    per_corner_pair: dict[tuple[int, int], float] | None = None


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
    pairwise distances to expected distances from board geometry. Aggregates
    statistics across all frames.

    Args:
        calibration: Complete calibration result
        detections: Detection result with pixel observations
        board: Board geometry with known corner positions
        include_per_pair: If True, populate per_corner_pair in result

    Returns:
        DistanceErrors with aggregated statistics across all frames.

    Notes:
        - Only compares corners that were both successfully triangulated
        - Expected distance computed from board.corner_positions
        - Returns DistanceErrors with mean=0, std=0, num_comparisons=0 if no valid pairs
    """
    all_errors = []
    per_pair = {} if include_per_pair else None

    for frame_idx in detections.frames:
        corners_3d = triangulate_charuco_corners(calibration, detections, frame_idx)

        if len(corners_3d) < 2:
            continue

        corner_ids = sorted(corners_3d.keys())

        # Compare all pairs of triangulated corners
        for i, id1 in enumerate(corner_ids):
            for id2 in corner_ids[i + 1 :]:
                # Triangulated distance
                actual_dist = np.linalg.norm(corners_3d[id1] - corners_3d[id2])

                # Expected distance from board geometry
                pos1 = board.corner_positions[id1]
                pos2 = board.corner_positions[id2]
                expected_dist = np.linalg.norm(pos1 - pos2)

                error = abs(actual_dist - expected_dist)
                all_errors.append(error)

                if per_pair is not None:
                    pair_key = (min(id1, id2), max(id1, id2))
                    # Keep worst error for each pair across frames
                    if pair_key not in per_pair or error > per_pair[pair_key]:
                        per_pair[pair_key] = error

    if not all_errors:
        return DistanceErrors(
            mean=0.0,
            std=0.0,
            max_error=0.0,
            num_comparisons=0,
            per_corner_pair=per_pair,
        )

    errors_arr = np.array(all_errors)
    return DistanceErrors(
        mean=float(np.mean(errors_arr)),
        std=float(np.std(errors_arr)),
        max_error=float(np.max(errors_arr)),
        num_comparisons=len(all_errors),
        per_corner_pair=per_pair,
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
