"""3D reconstruction metrics using known ChArUco geometry."""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aquacal.config.schema import (
    CalibrationResult,
    DetectionResult,
    Vec2,
    Vec3,
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
class SpatialMeasurements:
    """Per-measurement spatial data from 3D reconstruction validation.

    Each measurement is a distance comparison between two adjacent
    triangulated board corners. The position is the midpoint of the
    two corners in world frame.

    Attributes:
        positions: (N, 3) array of midpoint positions in world frame (meters)
        signed_errors: (N,) array of signed distance errors (meters).
            Positive = overestimate, negative = underestimate.
        frame_indices: (N,) array of frame index for each measurement
    """

    positions: NDArray[np.float64]  # (N, 3)
    signed_errors: NDArray[np.float64]  # (N,)
    frame_indices: NDArray[np.int32]  # (N,)


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
        spatial: Per-measurement spatial data (optional)
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
    spatial: SpatialMeasurements | None = None


@dataclass
class DepthBinnedErrors:
    """Depth-stratified signed error statistics.

    Attributes:
        bin_edges: (B+1,) array of bin edges in meters (Z coordinates)
        bin_centers: (B,) array of bin center Z values in meters
        signed_means: (B,) array of mean signed error per bin (meters)
        signed_stds: (B,) array of std of signed error per bin (meters)
        counts: (B,) array of number of measurements per bin
    """

    bin_edges: NDArray[np.float64]  # (B+1,)
    bin_centers: NDArray[np.float64]  # (B,)
    signed_means: NDArray[np.float64]  # (B,)
    signed_stds: NDArray[np.float64]  # (B,)
    counts: NDArray[np.int32]  # (B,)


@dataclass
class SpatialErrorGrid:
    """XY error grids within depth slices.

    Each depth slice has a 2D grid of mean signed errors and counts.

    Attributes:
        depth_bin_edges: (B+1,) array of depth bin edges in meters
        x_edges: (Gx+1,) array of X grid edges in meters
        y_edges: (Gy+1,) array of Y grid edges in meters
        grids: (B, Gy, Gx) array of mean signed error per cell (meters).
            NaN where no measurements fall.
        counts: (B, Gy, Gx) array of measurement counts per cell
    """

    depth_bin_edges: NDArray[np.float64]  # (B+1,)
    x_edges: NDArray[np.float64]  # (Gx+1,)
    y_edges: NDArray[np.float64]  # (Gy+1,)
    grids: NDArray[np.float64]  # (B, Gy, Gx)
    counts: NDArray[np.int32]  # (B, Gy, Gx)


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
    include_spatial: bool = False,
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
        include_spatial: If True, populate spatial field with per-measurement data

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

    # Spatial measurement accumulators
    spatial_positions = [] if include_spatial else None
    spatial_errors = [] if include_spatial else None
    spatial_frames = [] if include_spatial else None

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
                    pair_key = (
                        id1,
                        id2,
                    )  # Already in order from get_adjacent_corner_pairs
                    # Keep worst absolute error for each pair across frames
                    abs_error = abs(signed_error)
                    if pair_key not in per_pair or abs_error > abs(per_pair[pair_key]):
                        per_pair[pair_key] = signed_error

                if include_spatial:
                    # Record spatial measurement
                    midpoint = (corners_3d[id1] + corners_3d[id2]) / 2.0
                    spatial_positions.append(midpoint)
                    spatial_errors.append(signed_error)
                    spatial_frames.append(frame_idx)

        if frame_had_measurements:
            num_frames_with_measurements += 1

    if not all_signed_errors:
        warnings.warn("No valid 3D distance comparisons - returning NaN")
        # Create empty spatial measurements if requested
        spatial = None
        if include_spatial:
            spatial = SpatialMeasurements(
                positions=np.empty((0, 3), dtype=np.float64),
                signed_errors=np.empty((0,), dtype=np.float64),
                frame_indices=np.empty((0,), dtype=np.int32),
            )
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
            spatial=spatial,
        )

    signed_arr = np.array(all_signed_errors)
    abs_arr = np.abs(signed_arr)

    # Construct spatial measurements if requested
    spatial = None
    if include_spatial:
        spatial = SpatialMeasurements(
            positions=np.array(spatial_positions, dtype=np.float64),
            signed_errors=np.array(spatial_errors, dtype=np.float64),
            frame_indices=np.array(spatial_frames, dtype=np.int32),
        )

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
        spatial=spatial,
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


def bin_by_depth(spatial: SpatialMeasurements, n_bins: int = 10) -> DepthBinnedErrors:
    """
    Bin spatial measurements by Z coordinate and compute statistics per bin.

    Args:
        spatial: SpatialMeasurements containing positions and signed errors
        n_bins: Number of depth bins (default 10)

    Returns:
        DepthBinnedErrors with statistics per bin

    Raises:
        ValueError: If spatial has zero measurements

    Notes:
        - Bins with zero measurements have NaN for signed_means and signed_stds
        - Bin edges are computed using np.linspace from min to max Z
    """
    if len(spatial.positions) == 0:
        raise ValueError("Cannot bin empty spatial measurements")

    # Extract Z coordinates
    z_values = spatial.positions[:, 2]

    # Create bin edges
    z_min = np.min(z_values)
    z_max = np.max(z_values)
    bin_edges = np.linspace(z_min, z_max, n_bins + 1)

    # Compute bin centers
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Bin the measurements
    bin_indices = np.digitize(z_values, bin_edges) - 1
    # Handle edge case: values exactly at z_max get assigned to bin n_bins
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Compute statistics per bin
    signed_means = np.full(n_bins, np.nan, dtype=np.float64)
    signed_stds = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.int32)

    for i in range(n_bins):
        mask = bin_indices == i
        counts[i] = np.sum(mask)
        if counts[i] > 0:
            bin_errors = spatial.signed_errors[mask]
            signed_means[i] = np.mean(bin_errors)
            signed_stds[i] = np.std(bin_errors)

    return DepthBinnedErrors(
        bin_edges=bin_edges,
        bin_centers=bin_centers,
        signed_means=signed_means,
        signed_stds=signed_stds,
        counts=counts,
    )


def compute_xy_error_grids(
    spatial: SpatialMeasurements,
    depth_bin_edges: NDArray[np.float64],
    xy_grid_size: tuple[int, int] = (8, 8),
    xy_range: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> SpatialErrorGrid:
    """
    Bin spatial measurements into XY grids within depth slices.

    Args:
        spatial: SpatialMeasurements containing positions and signed errors
        depth_bin_edges: (B+1,) array of depth bin edges in meters (Z coordinates).
            Use the bin_edges from DepthBinnedErrors to ensure consistency.
        xy_grid_size: Tuple (n_x_bins, n_y_bins) for the XY grid within each depth slice
        xy_range: Optional ((x_min, x_max), (y_min, y_max)). If None, derived from data extent.

    Returns:
        SpatialErrorGrid with XY grids for each depth bin

    Raises:
        ValueError: If spatial has zero measurements

    Notes:
        - For each depth bin, selects measurements whose Z falls within [edge_i, edge_{i+1})
        - Last bin uses <= for right edge to include boundary values
        - Within each depth slice, bins by X and Y using np.linspace for edges
        - Cells with zero measurements get NaN in grids array
    """
    if len(spatial.positions) == 0:
        raise ValueError("Cannot compute XY grids for empty spatial measurements")

    # Extract coordinates
    x_values = spatial.positions[:, 0]
    y_values = spatial.positions[:, 1]
    z_values = spatial.positions[:, 2]

    # Determine XY range
    if xy_range is None:
        x_min, x_max = np.min(x_values), np.max(x_values)
        y_min, y_max = np.min(y_values), np.max(y_values)
    else:
        (x_min, x_max), (y_min, y_max) = xy_range

    # Create XY bin edges
    n_x_bins, n_y_bins = xy_grid_size
    x_edges = np.linspace(x_min, x_max, n_x_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_y_bins + 1)

    # Number of depth bins
    n_depth_bins = len(depth_bin_edges) - 1

    # Initialize output arrays
    grids = np.full((n_depth_bins, n_y_bins, n_x_bins), np.nan, dtype=np.float64)
    counts = np.zeros((n_depth_bins, n_y_bins, n_x_bins), dtype=np.int32)

    # Process each depth bin
    for d in range(n_depth_bins):
        # Select measurements in this depth bin
        if d == n_depth_bins - 1:
            # Last bin: include right edge
            depth_mask = (z_values >= depth_bin_edges[d]) & (
                z_values <= depth_bin_edges[d + 1]
            )
        else:
            depth_mask = (z_values >= depth_bin_edges[d]) & (
                z_values < depth_bin_edges[d + 1]
            )

        if not np.any(depth_mask):
            continue  # No measurements in this depth bin

        # Extract measurements in this depth slice
        x_slice = x_values[depth_mask]
        y_slice = y_values[depth_mask]
        errors_slice = spatial.signed_errors[depth_mask]

        # Digitize into XY bins
        # digitize returns 1-indexed bins, so we subtract 1
        x_bins = np.digitize(x_slice, x_edges) - 1
        y_bins = np.digitize(y_slice, y_edges) - 1

        # Clip to valid range (handles edge cases)
        x_bins = np.clip(x_bins, 0, n_x_bins - 1)
        y_bins = np.clip(y_bins, 0, n_y_bins - 1)

        # Accumulate signed errors and counts per XY cell
        for i in range(len(x_slice)):
            xi = x_bins[i]
            yi = y_bins[i]
            counts[d, yi, xi] += 1

        # Compute mean signed error per cell
        for yi in range(n_y_bins):
            for xi in range(n_x_bins):
                if counts[d, yi, xi] > 0:
                    # Find all measurements in this cell
                    cell_mask = (x_bins == xi) & (y_bins == yi)
                    cell_errors = errors_slice[cell_mask]
                    grids[d, yi, xi] = np.mean(cell_errors)

    return SpatialErrorGrid(
        depth_bin_edges=depth_bin_edges,
        x_edges=x_edges,
        y_edges=y_edges,
        grids=grids,
        counts=counts,
    )


def save_spatial_measurements(spatial: SpatialMeasurements, path: Path) -> None:
    """
    Save spatial measurements to CSV file.

    Args:
        spatial: SpatialMeasurements to save
        path: Output file path

    Notes:
        - CSV columns: x, y, z, signed_error, frame_idx
        - One row per measurement
    """
    df = pd.DataFrame(
        {
            "x": spatial.positions[:, 0],
            "y": spatial.positions[:, 1],
            "z": spatial.positions[:, 2],
            "signed_error": spatial.signed_errors,
            "frame_idx": spatial.frame_indices,
        }
    )
    df.to_csv(path, index=False)


def load_spatial_measurements(path: Path) -> SpatialMeasurements:
    """
    Load spatial measurements from CSV file.

    Args:
        path: Path to CSV file created by save_spatial_measurements()

    Returns:
        SpatialMeasurements with data from CSV

    Raises:
        FileNotFoundError: If path doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"Spatial measurements file not found: {path}")

    df = pd.read_csv(path)

    positions = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    signed_errors = df["signed_error"].to_numpy(dtype=np.float64)
    frame_indices = df["frame_idx"].to_numpy(dtype=np.int32)

    return SpatialMeasurements(
        positions=positions,
        signed_errors=signed_errors,
        frame_indices=frame_indices,
    )
