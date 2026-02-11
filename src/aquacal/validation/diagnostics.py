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


def compute_water_surface_consistency(
    calibration: CalibrationResult,
) -> dict[str, float]:
    """
    Compute water surface Z consistency across cameras.

    For each camera, computes water_z = camera_center_z + interface_distance.
    If calibration is consistent, all cameras should agree on the same Z.

    Args:
        calibration: Complete calibration result

    Returns:
        Dict with keys:
        - "mean": Mean water surface Z across cameras
        - "std": Standard deviation of water surface Z
        - "spread": Max - min water surface Z
        - "per_camera": Dict mapping camera name to water surface Z
    """
    water_zs = {}
    for cam_name, cam_calib in calibration.cameras.items():
        C = cam_calib.extrinsics.C
        water_zs[cam_name] = C[2] + cam_calib.interface_distance

    zs = np.array(list(water_zs.values()))
    return {
        "mean": float(np.mean(zs)),
        "std": float(np.std(zs)),
        "spread": float(np.ptp(zs)),
        "per_camera": water_zs,
    }


def generate_recommendations(
    reprojection: ReprojectionErrors,
    reconstruction: DistanceErrors,
    depth_errors: pd.DataFrame,
    water_surface: dict | None = None,
) -> list[str]:
    """
    Generate human-readable recommendations based on diagnostic results.

    Args:
        reprojection: Reprojection error statistics
        reconstruction: 3D reconstruction error statistics
        depth_errors: Depth-stratified error table
        water_surface: Optional water surface consistency data from compute_water_surface_consistency()

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

        # Scale bias detection
        if abs(reconstruction.signed_mean) > 0.001:
            bias_mm = reconstruction.signed_mean * 1000
            sign = "+" if bias_mm > 0 else ""
            if bias_mm > 0:
                recs.append(
                    f"Systematic scale bias detected ({sign}{bias_mm:.1f} mm) — "
                    f"distances overestimated (check n_water or interface distance)"
                )
            else:
                recs.append(
                    f"Systematic scale bias detected ({sign}{bias_mm:.1f} mm) — "
                    f"distances underestimated"
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

    # Water surface consistency
    if water_surface is not None:
        spread = water_surface["spread"]
        if spread < 0.02:
            recs.append(
                f"Water surface Z spread ({spread*1000:.1f} mm) is excellent — flat water assumption holds"
            )
        elif spread < 0.05:
            recs.append(f"Water surface Z spread ({spread*1000:.1f} mm) is acceptable")
        elif spread < 0.10:
            recs.append(
                f"Water surface Z spread ({spread*1000:.1f} mm) is elevated — consider enabling normal_fixed: false"
            )
        else:
            recs.append(
                f"Water surface Z spread ({spread*1000:.1f} mm) is large — check extrinsics and interface distances"
            )

        # Flag outlier cameras (>2 std from mean)
        mean_z = water_surface["mean"]
        std_z = water_surface["std"]
        if std_z > 0:
            for cam, z in water_surface["per_camera"].items():
                if abs(z - mean_z) > 2 * std_z:
                    recs.append(
                        f"Camera '{cam}' water surface Z is an outlier ({z:.4f} vs mean {mean_z:.4f})"
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

    # Compute water surface consistency
    water_surface = compute_water_surface_consistency(calibration)

    # Generate recommendations
    recommendations = generate_recommendations(
        reprojection_errors, reconstruction_errors, depth_errors, water_surface
    )

    # Build summary statistics
    summary = {
        "reprojection_rms": reprojection_errors.rms,
        "reprojection_num_obs": float(reprojection_errors.num_observations),
        "reconstruction_mean": reconstruction_errors.mean,
        "reconstruction_std": reconstruction_errors.std,
        "reconstruction_max": reconstruction_errors.max_error,
        "reconstruction_num_comparisons": float(reconstruction_errors.num_comparisons),
        "reconstruction_signed_mean": reconstruction_errors.signed_mean,
        "reconstruction_rmse": reconstruction_errors.rmse,
        "reconstruction_percent_error": reconstruction_errors.percent_error,
        "reconstruction_num_frames": float(reconstruction_errors.num_frames),
        "water_surface_z_mean": water_surface["mean"],
        "water_surface_z_std": water_surface["std"],
        "water_surface_z_spread": water_surface["spread"],
    }

    return DiagnosticReport(
        reprojection=reprojection_errors,
        reconstruction=reconstruction_errors,
        spatial_error_maps=spatial_error_maps,
        depth_errors=depth_errors,
        recommendations=recommendations,
        summary=summary,
    )


def plot_camera_rig(
    calibration: CalibrationResult,
    arrow_length: float = 0.05,
    title: str = "Camera Rig Layout",
) -> "plt.Figure":
    """
    Plot 3D camera rig with positions, orientations, and water surface in three viewing angles.

    Args:
        calibration: CalibrationResult with camera extrinsics and interface distances
        arrow_length: Length of optical axis arrows in world units (meters)
        title: Overall figure title

    Returns:
        matplotlib Figure with three subplots (perspective, top-down, side view)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Collect camera positions and interface z-coordinates
    camera_positions = []
    camera_names = []
    interface_zs = []

    for cam_name, cam_calib in calibration.cameras.items():
        C = cam_calib.extrinsics.C
        camera_positions.append(C)
        camera_names.append(cam_name)
        # Interface z = camera z + interface_distance (Z-down frame)
        interface_z = C[2] + cam_calib.interface_distance
        interface_zs.append(interface_z)

    camera_positions = np.array(camera_positions)

    # Generate colors for cameras
    colors = plt.cm.tab10(np.linspace(0, 1, len(calibration.cameras)))

    # Helper function to plot on a single axes
    def plot_on_axes(ax: "plt.Axes") -> None:
        """Plot camera rig on given 3D axes."""
        # Plot camera positions
        ax.scatter(
            camera_positions[:, 0],
            camera_positions[:, 1],
            camera_positions[:, 2],
            c=colors,
            s=100,
            marker="o",
            label="Cameras",
        )

        # Plot camera labels and optical axis arrows
        for i, (cam_name, cam_calib) in enumerate(calibration.cameras.items()):
            C = cam_calib.extrinsics.C
            R = cam_calib.extrinsics.R

            # Add label
            ax.text(C[0], C[1], C[2], f"  {cam_name}", fontsize=9)

            # Optical axis direction: Z-forward in camera frame = third column of R.T
            # (or equivalently, third row of R)
            optical_axis = R.T @ np.array([0.0, 0.0, 1.0])

            # Draw arrow
            ax.quiver(
                C[0],
                C[1],
                C[2],
                optical_axis[0] * arrow_length,
                optical_axis[1] * arrow_length,
                optical_axis[2] * arrow_length,
                color=colors[i],
                arrow_length_ratio=0.3,
                linewidth=2,
            )

        # Plot water surface plane
        mean_interface_z = np.mean(interface_zs)

        # Create grid for plane
        x_range = camera_positions[:, 0]
        y_range = camera_positions[:, 1]
        x_span = x_range.max() - x_range.min()
        y_span = y_range.max() - y_range.min()

        # Extend plane beyond camera positions
        margin = max(x_span, y_span) * 0.3 if max(x_span, y_span) > 0 else 0.1
        x_min, x_max = x_range.min() - margin, x_range.max() + margin
        y_min, y_max = y_range.min() - margin, y_range.max() + margin

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10)
        )
        zz = np.full_like(xx, mean_interface_z)

        ax.plot_surface(xx, yy, zz, alpha=0.3, color="lightblue", label="Water Surface")

        # Set labels
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m, down)")

        # Invert Z axis so "up" appears visually upward
        ax.invert_zaxis()

        # Equal aspect ratio
        ax.set_box_aspect([1, 1, 1])

    # Create 1x3 figure with three 3D subplots
    fig = plt.figure(figsize=(20, 7))
    axes = [fig.add_subplot(1, 3, i + 1, projection="3d") for i in range(3)]

    # Plot on each axes
    for ax in axes:
        plot_on_axes(ax)

    # Set viewing angles for each panel
    axes[0].view_init(elev=30, azim=-60)
    axes[0].set_title("Perspective")

    axes[1].view_init(elev=90, azim=-90)
    axes[1].set_title("Top-Down (XY)")

    axes[2].view_init(elev=0, azim=0)
    axes[2].set_title("Side (XZ)")

    # Set overall figure title
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

    return fig


def plot_reprojection_quiver(
    calibration: CalibrationResult,
    detections: DetectionResult,
    reprojection_errors: ReprojectionErrors,
    camera_name: str,
    ax: "plt.Axes | None" = None,
    scale: float = 1.0,
) -> "plt.Figure":
    """
    Plot reprojection error quiver for one camera.

    Args:
        calibration: CalibrationResult (for image_size)
        detections: DetectionResult with pixel positions
        reprojection_errors: Pre-computed errors with residuals array
        camera_name: Which camera to plot
        ax: Optional existing axes. If None, creates new figure.
        scale: Arrow scale factor (1.0 = true pixel scale, >1 exaggerates for visibility)

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.get_figure()

    # Get image size
    if camera_name not in calibration.cameras:
        raise ValueError(f"Camera '{camera_name}' not in calibration")

    image_size = calibration.cameras[camera_name].intrinsics.image_size
    width, height = image_size

    # Collect pixel positions and residuals for this camera
    # Must match iteration order from compute_reprojection_errors()
    pixel_positions = []
    residuals_list = []

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
                        pixel_positions.append(px)
                        residuals_list.append(residual)
                residual_idx += 1

    if not pixel_positions:
        # No detections for this camera, create empty plot
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_xlabel("u (pixels)")
        ax.set_ylabel("v (pixels)")
        ax.set_title(f"{camera_name} — No detections")
        return fig

    pixel_positions = np.array(pixel_positions)
    residuals_array = np.array(residuals_list)

    # Compute residual magnitudes for coloring
    magnitudes = np.sqrt(residuals_array[:, 0] ** 2 + residuals_array[:, 1] ** 2)

    # Plot quiver
    quiver = ax.quiver(
        pixel_positions[:, 0],
        pixel_positions[:, 1],
        residuals_array[:, 0] * scale,
        residuals_array[:, 1] * scale,
        magnitudes,
        cmap="viridis",
        scale=1.0,
        scale_units="xy",
        angles="xy",
        width=0.003,
    )

    # Add colorbar
    cbar = plt.colorbar(quiver, ax=ax, label="Residual Magnitude (pixels)")

    # Set axes
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Invert Y axis
    ax.set_xlabel("u (pixels)")
    ax.set_ylabel("v (pixels)")

    # Compute RMS for this camera
    rms = calibration.cameras[camera_name].intrinsics.image_size
    if camera_name in reprojection_errors.per_camera:
        rms_val = reprojection_errors.per_camera[camera_name]
        ax.set_title(f"{camera_name} — RMS: {rms_val:.3f} px")
    else:
        ax.set_title(f"{camera_name}")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    return fig


def save_diagnostic_report(
    report: DiagnosticReport,
    calibration: CalibrationResult,
    detections: DetectionResult,
    output_dir: Path,
    save_images: bool = True,
) -> dict[str, Path]:
    """
    Save diagnostic report to disk.

    Creates:
    - diagnostics.json: Summary statistics and recommendations
    - spatial_error_{cam}.png: Per-camera error heatmaps (if save_images=True)
    - camera_rig.png: 3D camera rig visualization (if save_images=True)
    - quiver_{cam}.png: Per-camera reprojection error quiver plots (if save_images=True)
    - depth_errors.csv: Depth-stratified error table

    Args:
        report: DiagnosticReport to save
        calibration: CalibrationResult for camera rig and quiver plots
        detections: DetectionResult for quiver plots
        output_dir: Directory for output files (created if doesn't exist)
        save_images: Whether to render and save images

    Returns:
        Dict mapping output type to file path:
        - "json": Path to diagnostics.json
        - "csv": Path to depth_errors.csv
        - "images": Dict of camera_name -> spatial error image path (if save_images=True)
        - "rig": Path to camera_rig.png (if save_images=True)
        - "quiver": Dict of camera_name -> quiver image path (if save_images=True)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {}

    # Save JSON summary
    json_path = output_dir / "diagnostics.json"

    # Compute water surface consistency
    water_surface = compute_water_surface_consistency(calibration)

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
            "signed_mean": report.reconstruction.signed_mean,
            "rmse": report.reconstruction.rmse,
            "percent_error": report.reconstruction.percent_error,
            "num_frames": report.reconstruction.num_frames,
        },
        "water_surface": {
            "mean": water_surface["mean"],
            "std": water_surface["std"],
            "spread": water_surface["spread"],
            "per_camera": water_surface["per_camera"],
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

        # Spatial error maps
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

        # Camera rig plot
        rig_path = output_dir / "camera_rig.png"
        fig = plot_camera_rig(calibration)
        fig.savefig(rig_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        result["rig"] = rig_path

        # Reprojection quiver plots
        result["quiver"] = {}
        for cam_name in calibration.cameras.keys():
            quiver_path = output_dir / f"quiver_{cam_name}.png"
            fig = plot_reprojection_quiver(
                calibration, detections, report.reprojection, cam_name
            )
            fig.savefig(quiver_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            result["quiver"][cam_name] = quiver_path

    return result
