"""Detailed error analysis and diagnostic reporting."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aquacal.config.schema import (
    CalibrationResult,
    DetectionResult,
    BoardPose,
)
from aquacal.core.board import BoardGeometry
from aquacal.validation.reprojection import ReprojectionErrors
from aquacal.validation.reconstruction import DistanceErrors


@dataclass
class DiagnosticReport:
    """Complete diagnostic report for calibration quality.

    Attributes:
        reprojection: Reprojection error statistics
        reconstruction: 3D reconstruction error statistics
        spatial_error_maps: Dict mapping camera name to error heatmap array
        depth_errors: DataFrame with depth-stratified error analysis
        recommendations: List of human-readable recommendations
        summary: Dict of key summary statistics
    """

    reprojection: ReprojectionErrors
    reconstruction: DistanceErrors
    spatial_error_maps: dict[str, NDArray[np.float64]]
    depth_errors: pd.DataFrame
    recommendations: list[str]
    summary: dict[str, float]


def compute_spatial_error_map(
    reprojection_errors: ReprojectionErrors,
    detections: DetectionResult,
    camera_name: str,
    image_size: tuple[int, int],
    grid_size: tuple[int, int] = (10, 10),
) -> NDArray[np.float64]:
    """
    Compute spatial distribution of reprojection errors for one camera.

    Bins the image into a grid and computes mean error magnitude in each cell.

    Args:
        reprojection_errors: Pre-computed reprojection errors (needs residuals)
        detections: Detection result with pixel positions
        camera_name: Camera to analyze
        image_size: (width, height) in pixels
        grid_size: (cols, rows) for binning

    Returns:
        2D array of shape (rows, cols) with mean error magnitude per cell.
        Cells with no observations contain NaN.

    Notes:
        - Error magnitude is sqrt(residual_x^2 + residual_y^2)
        - Useful for identifying spatially-varying calibration issues
    """
    width, height = image_size
    cols, rows = grid_size
    cell_w, cell_h = width / cols, height / rows

    # Accumulate errors per cell
    error_sums = np.zeros((rows, cols), dtype=np.float64)
    error_counts = np.zeros((rows, cols), dtype=np.int32)

    # Need to match residuals to pixel positions
    # This requires iterating through detections in same order as residuals were computed
    residual_idx = 0
    for frame_idx in sorted(detections.frames.keys()):
        frame_det = detections.frames[frame_idx]
        for cam in sorted(frame_det.detections.keys()):
            detection = frame_det.detections[cam]
            for i in range(len(detection.corner_ids)):
                if cam == camera_name:
                    if residual_idx < len(reprojection_errors.residuals):
                        px = detection.corners_2d[i]
                        residual = reprojection_errors.residuals[residual_idx]
                        error_mag = np.sqrt(residual[0] ** 2 + residual[1] ** 2)

                        col = min(int(px[0] / cell_w), cols - 1)
                        row = min(int(px[1] / cell_h), rows - 1)
                        error_sums[row, col] += error_mag
                        error_counts[row, col] += 1
                residual_idx += 1

    # Compute mean, NaN where no observations
    result = np.full((rows, cols), np.nan)
    mask = error_counts > 0
    result[mask] = error_sums[mask] / error_counts[mask]
    return result


def compute_depth_stratified_errors(
    calibration: CalibrationResult,
    detections: DetectionResult,
    board_poses: dict[int, BoardPose],
    reprojection_errors: ReprojectionErrors,
    board: BoardGeometry,
    num_bins: int = 5,
) -> pd.DataFrame:
    """
    Analyze reprojection error as a function of depth below water surface.

    Args:
        calibration: Complete calibration result
        detections: Detection result with pixel observations
        board_poses: Dict mapping frame_idx to BoardPose
        reprojection_errors: Pre-computed reprojection errors
        board: Board geometry
        num_bins: Number of depth bins

    Returns:
        DataFrame with columns:
        - depth_min: Lower bound of depth bin (meters)
        - depth_max: Upper bound of depth bin (meters)
        - mean_error: Mean reprojection error in bin (pixels)
        - std_error: Std deviation of error in bin (pixels)
        - num_observations: Count of observations in bin

    Notes:
        - Depth is measured from water surface (Z - interface_z)
        - Helps identify depth-dependent calibration issues
    """
    # Collect (depth, error) pairs
    depth_error_pairs = []

    residual_idx = 0
    for frame_idx in sorted(detections.frames.keys()):
        if frame_idx not in board_poses:
            continue

        bp = board_poses[frame_idx]
        corners_3d = board.transform_corners(bp.rvec, bp.tvec)
        frame_det = detections.frames[frame_idx]

        for cam_name in sorted(frame_det.detections.keys()):
            if cam_name not in calibration.cameras:
                continue

            cam_calib = calibration.cameras[cam_name]
            interface_z = cam_calib.interface_distance
            detection = frame_det.detections[cam_name]

            for i, corner_id in enumerate(detection.corner_ids):
                if residual_idx < len(reprojection_errors.residuals):
                    corner_3d = corners_3d[int(corner_id)]
                    depth = corner_3d[2] - interface_z  # Depth below surface

                    residual = reprojection_errors.residuals[residual_idx]
                    error = np.sqrt(residual[0] ** 2 + residual[1] ** 2)

                    depth_error_pairs.append((depth, error))
                residual_idx += 1

    if not depth_error_pairs:
        return pd.DataFrame(
            columns=[
                "depth_min",
                "depth_max",
                "mean_error",
                "std_error",
                "num_observations",
            ]
        )

    depths, errors = zip(*depth_error_pairs)
    depths = np.array(depths)
    errors = np.array(errors)

    # Create bins
    depth_min, depth_max = depths.min(), depths.max()
    bin_edges = np.linspace(depth_min, depth_max, num_bins + 1)

    rows = []
    for i in range(num_bins):
        mask = (depths >= bin_edges[i]) & (depths < bin_edges[i + 1])
        if i == num_bins - 1:  # Include right edge in last bin
            mask = (depths >= bin_edges[i]) & (depths <= bin_edges[i + 1])

        bin_errors = errors[mask]
        rows.append(
            {
                "depth_min": bin_edges[i],
                "depth_max": bin_edges[i + 1],
                "mean_error": np.mean(bin_errors) if len(bin_errors) > 0 else np.nan,
                "std_error": np.std(bin_errors) if len(bin_errors) > 0 else np.nan,
                "num_observations": len(bin_errors),
            }
        )

    return pd.DataFrame(rows)


def generate_recommendations(
    reprojection: ReprojectionErrors,
    reconstruction: DistanceErrors,
    depth_errors: pd.DataFrame,
) -> list[str]:
    """
    Generate human-readable recommendations based on diagnostic results.

    Args:
        reprojection: Reprojection error statistics
        reconstruction: 3D reconstruction error statistics
        depth_errors: Depth-stratified error table

    Returns:
        List of recommendation strings.

    Example recommendations:
        - "Reprojection RMS (0.8 px) is within acceptable range (<1.0 px)"
        - "Camera 'cam2' has elevated error (1.5 px) - check lens/mounting"
        - "Error increases with depth - consider re-estimating interface"
    """
    recs = []

    # Reprojection RMS thresholds
    if reprojection.rms < 0.5:
        recs.append(
            f"Reprojection RMS ({reprojection.rms:.2f} px) is excellent (<0.5 px)"
        )
    elif reprojection.rms < 1.0:
        recs.append(f"Reprojection RMS ({reprojection.rms:.2f} px) is good (<1.0 px)")
    else:
        recs.append(
            f"Reprojection RMS ({reprojection.rms:.2f} px) is elevated - review calibration"
        )

    # Per-camera outliers
    if reprojection.per_camera:
        mean_cam_error = np.mean(list(reprojection.per_camera.values()))
        for cam, err in reprojection.per_camera.items():
            if err > mean_cam_error * 1.5:
                recs.append(
                    f"Camera '{cam}' has elevated error ({err:.2f} px) - check lens/mounting"
                )

    # 3D reconstruction
    if reconstruction.num_comparisons > 0:
        if reconstruction.mean < 0.001:
            recs.append(
                f"3D reconstruction error ({reconstruction.mean*1000:.2f} mm) is excellent"
            )
        elif reconstruction.mean < 0.002:
            recs.append(
                f"3D reconstruction error ({reconstruction.mean*1000:.2f} mm) is good"
            )
        else:
            recs.append(
                f"3D reconstruction error ({reconstruction.mean*1000:.2f} mm) is elevated"
            )

    # Depth trend
    if len(depth_errors) >= 2:
        errors = depth_errors["mean_error"].dropna()
        if len(errors) >= 2:
            first_half = errors.iloc[: len(errors) // 2].mean()
            second_half = errors.iloc[len(errors) // 2 :].mean()
            if second_half > first_half * 1.3:
                recs.append(
                    "Error increases with depth - consider re-estimating interface parameters"
                )

    return recs


def generate_diagnostic_report(
    calibration: CalibrationResult,
    detections: DetectionResult,
    board_poses: dict[int, BoardPose],
    reprojection_errors: ReprojectionErrors,
    reconstruction_errors: DistanceErrors,
    board: BoardGeometry,
) -> DiagnosticReport:
    """
    Generate complete diagnostic report.

    Args:
        calibration: Complete calibration result
        detections: Detection result
        board_poses: Dict mapping frame_idx to BoardPose
        reprojection_errors: Pre-computed from reprojection.py
        reconstruction_errors: Pre-computed from reconstruction.py
        board: Board geometry

    Returns:
        DiagnosticReport with all analysis results.
    """
    # Compute spatial error maps for each camera
    spatial_error_maps = {}
    for cam_name, cam_calib in calibration.cameras.items():
        image_size = cam_calib.intrinsics.image_size
        error_map = compute_spatial_error_map(
            reprojection_errors, detections, cam_name, image_size
        )
        spatial_error_maps[cam_name] = error_map

    # Compute depth-stratified errors
    depth_errors = compute_depth_stratified_errors(
        calibration, detections, board_poses, reprojection_errors, board
    )

    # Generate recommendations
    recommendations = generate_recommendations(
        reprojection_errors, reconstruction_errors, depth_errors
    )

    # Build summary statistics
    summary = {
        "reprojection_rms": reprojection_errors.rms,
        "reprojection_num_obs": float(reprojection_errors.num_observations),
        "reconstruction_mean": reconstruction_errors.mean,
        "reconstruction_std": reconstruction_errors.std,
        "reconstruction_max": reconstruction_errors.max_error,
        "reconstruction_num_comparisons": float(reconstruction_errors.num_comparisons),
    }

    return DiagnosticReport(
        reprojection=reprojection_errors,
        reconstruction=reconstruction_errors,
        spatial_error_maps=spatial_error_maps,
        depth_errors=depth_errors,
        recommendations=recommendations,
        summary=summary,
    )


def save_diagnostic_report(
    report: DiagnosticReport,
    output_dir: Path,
    save_images: bool = True,
) -> dict[str, Path]:
    """
    Save diagnostic report to disk.

    Creates:
    - diagnostics.json: Summary statistics and recommendations
    - spatial_error_{cam}.png: Per-camera error heatmaps (if save_images=True)
    - depth_errors.csv: Depth-stratified error table

    Args:
        report: DiagnosticReport to save
        output_dir: Directory for output files (created if doesn't exist)
        save_images: Whether to render and save heatmap images

    Returns:
        Dict mapping output type to file path:
        - "json": Path to diagnostics.json
        - "csv": Path to depth_errors.csv
        - "images": Dict of camera_name -> image path (if save_images=True)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {}

    # Save JSON summary
    json_path = output_dir / "diagnostics.json"
    json_data = {
        "summary": report.summary,
        "recommendations": report.recommendations,
        "reprojection": {
            "rms": report.reprojection.rms,
            "per_camera": report.reprojection.per_camera,
            "num_observations": report.reprojection.num_observations,
        },
        "reconstruction": {
            "mean": report.reconstruction.mean,
            "std": report.reconstruction.std,
            "max_error": report.reconstruction.max_error,
            "num_comparisons": report.reconstruction.num_comparisons,
        },
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    result["json"] = json_path

    # Save CSV
    csv_path = output_dir / "depth_errors.csv"
    report.depth_errors.to_csv(csv_path, index=False)
    result["csv"] = csv_path

    # Save images
    if save_images:
        import matplotlib.pyplot as plt

        result["images"] = {}
        for cam_name, error_map in report.spatial_error_maps.items():
            img_path = output_dir / f"spatial_error_{cam_name}.png"

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(error_map, cmap="hot", interpolation="nearest")
            ax.set_title(f"Spatial Error Map: {cam_name}")
            ax.set_xlabel("Image X (binned)")
            ax.set_ylabel("Image Y (binned)")
            plt.colorbar(im, ax=ax, label="Mean Error (pixels)")

            fig.savefig(img_path, dpi=100, bbox_inches="tight")
            plt.close(fig)

            result["images"][cam_name] = img_path

    return result
