"""Cross-run calibration comparison."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aquacal.config.schema import CalibrationResult
from aquacal.utils.transforms import matrix_to_rvec


@dataclass
class ComparisonResult:
    """Structured comparison of N calibration runs.

    Attributes:
        labels: User-assigned labels for each run
        metric_table: DataFrame with rows = runs (indexed by label), columns = quality metrics
        per_camera_metrics: Dict mapping camera_name -> DataFrame with rows = runs, columns = per-camera metrics
        parameter_diffs: DataFrame with pairwise parameter differences between runs
    """

    labels: list[str]
    metric_table: pd.DataFrame
    per_camera_metrics: dict[str, pd.DataFrame]
    parameter_diffs: pd.DataFrame


def compare_calibrations(
    results: list[CalibrationResult],
    labels: list[str],
) -> ComparisonResult:
    """Compare N calibration runs and return structured differences.

    Args:
        results: List of calibration results to compare (must have len >= 2)
        labels: User-assigned labels for each run (must be unique)

    Returns:
        ComparisonResult containing metric tables and parameter differences

    Raises:
        ValueError: If len(results) < 2, len(results) != len(labels), or labels not unique
    """
    # Validation
    if len(results) < 2:
        raise ValueError(f"Need at least 2 results to compare, got {len(results)}")
    if len(results) != len(labels):
        raise ValueError(f"Mismatch: {len(results)} results but {len(labels)} labels")
    if len(set(labels)) != len(labels):
        raise ValueError(f"Labels must be unique, got duplicates in {labels}")

    # Build metric table
    metric_table = _build_metric_table(results, labels)

    # Build per-camera metrics
    per_camera_metrics = _build_per_camera_metrics(results, labels)

    # Build parameter diffs
    parameter_diffs = _build_parameter_diffs(results, labels)

    return ComparisonResult(
        labels=labels,
        metric_table=metric_table,
        per_camera_metrics=per_camera_metrics,
        parameter_diffs=parameter_diffs,
    )


def _build_metric_table(
    results: list[CalibrationResult], labels: list[str]
) -> pd.DataFrame:
    """Build overall quality metrics table.

    Columns:
        - reprojection_rms: Overall RMS reprojection error (pixels)
        - reproj_3d_mean: Mean 3D reconstruction error (meters)
        - reproj_3d_std: Std of 3D reconstruction error (meters)
        - reproj_3d_rmse: RMSE of 3D reconstruction (meters)
        - reproj_3d_pct: Percent error of 3D reconstruction
        - water_z: Interface distance value (meters)
        - num_cameras: Number of cameras in the result
        - num_frames_used: From metadata
    """
    rows = []
    for result, label in zip(results, labels):
        diag = result.diagnostics
        meta = result.metadata

        # Compute RMSE and percent error from mean and std
        # RMSE = sqrt(mean^2 + std^2) for zero-mean errors
        # But validation errors are absolute distances, so RMSE = sqrt(mean(x^2))
        # We have mean and std, so we can compute: RMSE^2 = mean^2 + std^2 (if errors are zero-mean)
        # Actually, std^2 = mean((x - mu)^2), so mean(x^2) = std^2 + mean^2
        rmse_3d = np.sqrt(
            diag.validation_3d_error_mean**2 + diag.validation_3d_error_std**2
        )

        # Percent error: not well-defined without a reference scale
        # Use mean error / mean depth as a rough proxy
        # But we don't have depth info here... use mean error as percentage of water_z
        # Actually, the task says "percent error of 3D reconstruction" but doesn't specify the denominator
        # Let's use the mean error itself as a percentage (mean / typical_scale * 100)
        # Since we don't have a clear scale, let's use mean_error / water_z * 100 as a rough metric
        # Get water_z from first camera's interface_distance
        first_camera = next(iter(result.cameras.values()))
        water_z = first_camera.interface_distance

        pct_error = (
            (diag.validation_3d_error_mean / water_z * 100) if water_z > 0 else 0.0
        )

        rows.append(
            {
                "reprojection_rms": diag.reprojection_error_rms,
                "reproj_3d_mean": diag.validation_3d_error_mean,
                "reproj_3d_std": diag.validation_3d_error_std,
                "reproj_3d_rmse": rmse_3d,
                "reproj_3d_pct": pct_error,
                "water_z": water_z,
                "num_cameras": len(result.cameras),
                "num_frames_used": meta.num_frames_used,
            }
        )

    df = pd.DataFrame(rows, index=labels)
    df.index.name = "run"
    return df


def _build_per_camera_metrics(
    results: list[CalibrationResult], labels: list[str]
) -> dict[str, pd.DataFrame]:
    """Build per-camera metrics dict.

    For each camera found across ALL results, create a DataFrame with:
        - rows = runs (indexed by label)
        - columns = per-camera metrics

    Columns:
        - reprojection_rms: Per-camera RMS (pixels)
        - position_x, position_y, position_z: Camera center in world frame (meters)
        - interface_distance: Per-camera interface distance (meters)
        - fx, fy, cx, cy: Intrinsic parameters (pixels)
    """
    # Collect all camera names across all results
    all_camera_names = set()
    for result in results:
        all_camera_names.update(result.cameras.keys())

    per_camera_metrics = {}

    for cam_name in sorted(all_camera_names):
        rows = []
        for result, label in zip(results, labels):
            if cam_name in result.cameras:
                cam = result.cameras[cam_name]
                C = cam.extrinsics.C
                K = cam.intrinsics.K
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]

                # Get per-camera reprojection RMS
                reproj_rms = result.diagnostics.reprojection_error_per_camera.get(
                    cam_name, np.nan
                )

                rows.append(
                    {
                        "reprojection_rms": reproj_rms,
                        "position_x": C[0],
                        "position_y": C[1],
                        "position_z": C[2],
                        "interface_distance": cam.interface_distance,
                        "fx": fx,
                        "fy": fy,
                        "cx": cx,
                        "cy": cy,
                    }
                )
            else:
                # Camera not present in this result - fill with NaN
                rows.append(
                    {
                        "reprojection_rms": np.nan,
                        "position_x": np.nan,
                        "position_y": np.nan,
                        "position_z": np.nan,
                        "interface_distance": np.nan,
                        "fx": np.nan,
                        "fy": np.nan,
                        "cx": np.nan,
                        "cy": np.nan,
                    }
                )

        df = pd.DataFrame(rows, index=labels)
        df.index.name = "run"
        per_camera_metrics[cam_name] = df

    return per_camera_metrics


def _build_parameter_diffs(
    results: list[CalibrationResult], labels: list[str]
) -> pd.DataFrame:
    """Build pairwise parameter differences table.

    For each pair of runs (a, b) and each camera in their intersection, compute:
        - position_delta_mm: Euclidean distance between camera centers (millimeters)
        - orientation_delta_deg: Angular distance between rotations (degrees)
        - fx_delta, fy_delta, cx_delta, cy_delta: Absolute intrinsic differences (pixels)

    Returns DataFrame with columns:
        - label_a, label_b: The two run labels
        - camera: Camera name
        - position_delta_mm, orientation_delta_deg, fx_delta, fy_delta, cx_delta, cy_delta
    """
    rows = []

    n = len(results)
    for i in range(n):
        for j in range(i + 1, n):
            result_a, label_a = results[i], labels[i]
            result_b, label_b = results[j], labels[j]

            # Find intersection of camera names
            cameras_a = set(result_a.cameras.keys())
            cameras_b = set(result_b.cameras.keys())
            common_cameras = cameras_a & cameras_b

            for cam_name in sorted(common_cameras):
                cam_a = result_a.cameras[cam_name]
                cam_b = result_b.cameras[cam_name]

                # Position delta (in mm)
                C_a = cam_a.extrinsics.C
                C_b = cam_b.extrinsics.C
                position_delta_m = np.linalg.norm(C_a - C_b)
                position_delta_mm = position_delta_m * 1000.0

                # Orientation delta (in degrees)
                R_a = cam_a.extrinsics.R
                R_b = cam_b.extrinsics.R
                R_diff = R_a @ R_b.T
                rvec_diff = matrix_to_rvec(R_diff)
                orientation_delta_rad = np.linalg.norm(rvec_diff)
                orientation_delta_deg = np.degrees(orientation_delta_rad)

                # Intrinsic deltas
                K_a = cam_a.intrinsics.K
                K_b = cam_b.intrinsics.K
                fx_delta = abs(K_a[0, 0] - K_b[0, 0])
                fy_delta = abs(K_a[1, 1] - K_b[1, 1])
                cx_delta = abs(K_a[0, 2] - K_b[0, 2])
                cy_delta = abs(K_a[1, 2] - K_b[1, 2])

                rows.append(
                    {
                        "label_a": label_a,
                        "label_b": label_b,
                        "camera": cam_name,
                        "position_delta_mm": position_delta_mm,
                        "orientation_delta_deg": orientation_delta_deg,
                        "fx_delta": fx_delta,
                        "fy_delta": fy_delta,
                        "cx_delta": cx_delta,
                        "cy_delta": cy_delta,
                    }
                )

    # Create DataFrame with explicit columns to ensure correct shape even when empty
    columns = [
        "label_a",
        "label_b",
        "camera",
        "position_delta_mm",
        "orientation_delta_deg",
        "fx_delta",
        "fy_delta",
        "cx_delta",
        "cy_delta",
    ]
    df = pd.DataFrame(rows, columns=columns)
    return df


def write_comparison_report(
    comparison: ComparisonResult,
    results: list[CalibrationResult],
    output_dir: str | Path,
    save_plots: bool = True,
    depth_data: dict[str, "DepthBinnedErrors"] | None = None,
    spatial_data: dict[str, "SpatialMeasurements"] | None = None,
) -> dict[str, Path]:
    """Write comparison report to disk as CSV tables and PNG plots.

    Args:
        comparison: ComparisonResult from compare_calibrations()
        results: List of CalibrationResult objects used to generate comparison
        output_dir: Directory for output files (created if doesn't exist)
        save_plots: Whether to generate and save PNG plots
        depth_data: Optional dict mapping run label -> DepthBinnedErrors for depth plot
        spatial_data: Optional dict mapping run label -> SpatialMeasurements for XY heatmaps

    Returns:
        Dict mapping output type to file path:
        - "metrics_csv": metrics_summary.csv
        - "per_camera_csv": per_camera_metrics.csv
        - "parameter_diffs_csv": parameter_diffs.csv
        - "rms_bar_chart": rms_bar_chart.png (if save_plots=True)
        - "position_overlay": position_overlay.png (if save_plots=True)
        - "z_position_dumbbell": z_position_dumbbell.png (if save_plots=True)
        - "depth_error_plot": depth_error_comparison.png (if depth_data and save_plots=True)
        - "depth_binned_csv": depth_binned_errors.csv (if depth_data)
        - "xy_error_heatmaps": xy_error_heatmaps.png (if spatial_data and save_plots=True)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {}

    # Write CSV: metrics_summary.csv
    metrics_csv = output_dir / "metrics_summary.csv"
    # Reset index to make label a column
    metrics_df = comparison.metric_table.reset_index()
    metrics_df.to_csv(metrics_csv, index=False)
    result["metrics_csv"] = metrics_csv

    # Write CSV: per_camera_metrics.csv (stacked)
    per_camera_csv = output_dir / "per_camera_metrics.csv"
    stacked_rows = []
    for cam_name, cam_df in comparison.per_camera_metrics.items():
        # Reset index to get run label as column
        cam_df_reset = cam_df.reset_index()
        cam_df_reset.insert(0, "camera", cam_name)
        stacked_rows.append(cam_df_reset)

    if stacked_rows:
        stacked_df = pd.concat(stacked_rows, ignore_index=True)
        stacked_df.to_csv(per_camera_csv, index=False)
    else:
        # Empty DataFrame with expected columns
        pd.DataFrame(columns=["camera", "run"]).to_csv(per_camera_csv, index=False)
    result["per_camera_csv"] = per_camera_csv

    # Write CSV: parameter_diffs.csv
    parameter_diffs_csv = output_dir / "parameter_diffs.csv"
    comparison.parameter_diffs.to_csv(parameter_diffs_csv, index=False)
    result["parameter_diffs_csv"] = parameter_diffs_csv

    # Generate plots if requested
    if save_plots:
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt

        # RMS bar chart
        rms_chart_path = output_dir / "rms_bar_chart.png"
        fig = plot_rms_bar_chart(comparison)
        fig.savefig(rms_chart_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        result["rms_bar_chart"] = rms_chart_path

        # Position overlay
        position_overlay_path = output_dir / "position_overlay.png"
        fig = plot_position_overlay(comparison, results)
        fig.savefig(position_overlay_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        result["position_overlay"] = position_overlay_path

        # Z position dumbbell
        z_position_dumbbell_path = output_dir / "z_position_dumbbell.png"
        fig = plot_z_position_dumbbell(comparison)
        fig.savefig(z_position_dumbbell_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        result["z_position_dumbbell"] = z_position_dumbbell_path

        # Depth error comparison plot (if depth_data provided)
        if depth_data is not None:
            depth_error_plot_path = output_dir / "depth_error_comparison.png"
            fig = plot_depth_error_comparison(depth_data)
            fig.savefig(depth_error_plot_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            result["depth_error_plot"] = depth_error_plot_path

        # XY error heatmaps (if spatial_data provided)
        if spatial_data is not None:
            from aquacal.validation.reconstruction import (
                SpatialMeasurements,
                bin_by_depth,
                compute_xy_error_grids,
            )

            # Compute depth bin edges (use from depth_data if available, else compute from first spatial)
            if depth_data is not None:
                # Use bin edges from first depth data entry
                first_depth_label = sorted(depth_data.keys())[0]
                depth_bin_edges = depth_data[first_depth_label].bin_edges
            else:
                # Compute bin edges from first spatial entry
                first_spatial_label = sorted(spatial_data.keys())[0]
                binned = bin_by_depth(spatial_data[first_spatial_label])
                depth_bin_edges = binned.bin_edges

            # Compute global XY range across all runs
            all_x = np.concatenate([s.positions[:, 0] for s in spatial_data.values()])
            all_y = np.concatenate([s.positions[:, 1] for s in spatial_data.values()])
            xy_range = ((all_x.min(), all_x.max()), (all_y.min(), all_y.max()))

            # Compute XY error grids for each run
            error_grids = {}
            for label, spatial in spatial_data.items():
                grid = compute_xy_error_grids(
                    spatial, depth_bin_edges=depth_bin_edges, xy_range=xy_range
                )
                error_grids[label] = grid

            # Plot and save heatmaps
            xy_error_heatmaps_path = output_dir / "xy_error_heatmaps.png"
            fig = plot_xy_error_heatmaps(error_grids)
            fig.savefig(xy_error_heatmaps_path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            result["xy_error_heatmaps"] = xy_error_heatmaps_path

    # Depth binned errors CSV (if depth_data provided, regardless of save_plots)
    if depth_data is not None:
        from aquacal.validation.reconstruction import DepthBinnedErrors

        depth_binned_csv = output_dir / "depth_binned_errors.csv"
        # Stack all runs into one CSV
        rows = []
        for label, binned in depth_data.items():
            for i in range(len(binned.bin_centers)):
                rows.append(
                    {
                        "label": label,
                        "bin_center_z": binned.bin_centers[i],
                        "signed_mean_mm": binned.signed_means[i]
                        * 1000.0,  # Convert to mm
                        "signed_std_mm": binned.signed_stds[i]
                        * 1000.0,  # Convert to mm
                        "count": binned.counts[i],
                    }
                )
        depth_df = pd.DataFrame(rows)
        depth_df.to_csv(depth_binned_csv, index=False)
        result["depth_binned_csv"] = depth_binned_csv

    return result


def plot_rms_bar_chart(comparison: ComparisonResult):
    """Create grouped bar chart of per-camera reprojection RMS.

    Args:
        comparison: ComparisonResult with per_camera_metrics

    Returns:
        matplotlib Figure object
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    # Get all camera names (sorted alphabetically)
    camera_names = sorted(comparison.per_camera_metrics.keys())
    n_cameras = len(camera_names)
    n_runs = len(comparison.labels)

    if n_cameras == 0:
        # No cameras - return empty plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No cameras found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Per-Camera Reprojection RMS Comparison")
        return fig

    # Prepare data: matrix of [n_cameras x n_runs]
    rms_matrix = np.full((n_cameras, n_runs), np.nan)
    for i, cam_name in enumerate(camera_names):
        cam_df = comparison.per_camera_metrics[cam_name]
        for j, label in enumerate(comparison.labels):
            if label in cam_df.index:
                rms_value = cam_df.loc[label, "reprojection_rms"]
                if not pd.isna(rms_value):
                    rms_matrix[i, j] = rms_value

    # Create bar chart
    fig, ax = plt.subplots(figsize=(max(8, n_cameras * 0.8), 6))

    bar_width = 0.8 / n_runs
    x = np.arange(n_cameras)

    # Plot bars for each run
    for j, label in enumerate(comparison.labels):
        offset = (j - n_runs / 2 + 0.5) * bar_width
        rms_values = rms_matrix[:, j]
        ax.bar(x + offset, rms_values, bar_width, label=label)

    ax.set_xlabel("Camera")
    ax.set_ylabel("RMS Reprojection Error (pixels)")
    ax.set_title("Per-Camera Reprojection RMS Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(camera_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    return fig


def plot_position_overlay(
    comparison: ComparisonResult, results: list[CalibrationResult]
):
    """Create 2D top-down scatter plot of camera positions.

    Args:
        comparison: ComparisonResult with labels
        results: List of CalibrationResult objects

    Returns:
        matplotlib Figure object
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot camera positions for each run
    for run_idx, (result, label) in enumerate(zip(results, comparison.labels)):
        positions_x = []
        positions_y = []
        camera_names = []

        for cam_name, cam in sorted(result.cameras.items()):
            C = cam.extrinsics.C
            positions_x.append(C[0])
            positions_y.append(C[1])
            camera_names.append(cam_name)

        # Plot markers
        ax.scatter(positions_x, positions_y, label=label, s=100, alpha=0.7)

        # Annotate camera names (only for first run to avoid clutter)
        if run_idx == 0:
            for i, cam_name in enumerate(camera_names):
                ax.annotate(
                    cam_name,
                    (positions_x[i], positions_y[i]),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.8,
                )

    ax.set_xlabel("World X (meters)")
    ax.set_ylabel("World Y (meters)")
    ax.set_title("Camera Positions (Top-Down)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    # Invert Y-axis to match camera Y-down convention
    # See KB: "Top-down camera rig plot CW/CCW flip from Y-axis convention"
    ax.invert_yaxis()

    return fig


def plot_z_position_dumbbell(comparison: ComparisonResult):
    """Create dumbbell chart of camera Z positions across runs.

    For each camera (sorted alphabetically), plots one dot per run at its Z position,
    connected by a horizontal line. Cameras missing from a run are skipped.

    Args:
        comparison: ComparisonResult with per_camera_metrics

    Returns:
        matplotlib Figure object
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    # Get all camera names (sorted alphabetically)
    camera_names = sorted(comparison.per_camera_metrics.keys())
    n_cameras = len(camera_names)
    n_runs = len(comparison.labels)

    if n_cameras == 0:
        # No cameras - return empty plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No cameras found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Camera Z Positions Across Runs")
        return fig

    # Create figure
    fig, ax = plt.subplots(figsize=(10, max(6, n_cameras * 0.5)))

    # Set up color cycle for runs
    colors = plt.cm.tab10(np.linspace(0, 1, n_runs))

    # Y-positions for cameras
    y_positions = np.arange(n_cameras)

    # Plot dumbbell for each camera
    for i, cam_name in enumerate(camera_names):
        cam_df = comparison.per_camera_metrics[cam_name]

        # Get Z positions for each run
        z_values = []
        valid_run_indices = []
        for j, label in enumerate(comparison.labels):
            if label in cam_df.index:
                z_val = cam_df.loc[label, "position_z"]
                if not pd.isna(z_val):
                    z_values.append(z_val)
                    valid_run_indices.append(j)

        # Skip if no valid data
        if len(z_values) == 0:
            continue

        # Plot dots for each run
        for z_val, run_idx in zip(z_values, valid_run_indices):
            ax.scatter(
                z_val,
                i,
                s=80,
                color=colors[run_idx],
                alpha=0.8,
                zorder=3,
            )

        # Draw connecting line if there are multiple valid points
        if len(z_values) > 1:
            z_min = min(z_values)
            z_max = max(z_values)
            ax.hlines(
                i,
                z_min,
                z_max,
                colors="gray",
                linestyles="solid",
                linewidth=1,
                alpha=0.5,
                zorder=1,
            )

    # Set labels and title
    ax.set_ylabel("Camera")
    ax.set_xlabel("Z position (meters)")
    ax.set_title("Camera Z Positions Across Runs")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(camera_names)
    ax.grid(axis="x", alpha=0.3)

    # Create legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors[j], alpha=0.8, label=label)
        for j, label in enumerate(comparison.labels)
    ]
    ax.legend(handles=legend_elements, loc="best")

    return fig


def plot_xy_error_heatmaps(grids: dict[str, "SpatialErrorGrid"]):
    """Create heatmap grid of XY error distributions across depth slices.

    Args:
        grids: Dict mapping run label -> SpatialErrorGrid

    Returns:
        matplotlib Figure object

    Notes:
        - Subplot grid: rows = runs (sorted by label), columns = depth bins
        - Each subplot shows mean signed error in mm using diverging colormap
        - Symmetric color scale centered at zero across all runs
        - Depth bins with zero measurements across ALL runs are skipped
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    # Import locally to avoid circular dependency
    from aquacal.validation.reconstruction import SpatialErrorGrid

    if not grids:
        raise ValueError("No grids provided for heatmap")

    # Sort run labels for consistent ordering
    sorted_labels = sorted(grids.keys())
    n_runs = len(sorted_labels)

    # Get grid dimensions from first grid (all should be the same)
    first_grid = grids[sorted_labels[0]]
    n_depth_bins = first_grid.grids.shape[0]
    n_y = first_grid.grids.shape[1]
    n_x = first_grid.grids.shape[2]

    # Identify depth bins with at least one run having measurements
    valid_depth_bins = []
    for d in range(n_depth_bins):
        has_data = any(np.any(grid.counts[d] > 0) for grid in grids.values())
        if has_data:
            valid_depth_bins.append(d)

    if not valid_depth_bins:
        # All bins are empty - return empty figure
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            "No spatial data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Signed Error by XY Position and Depth")
        return fig

    n_cols = len(valid_depth_bins)

    # Determine symmetric color scale across all runs and bins
    all_values = []
    for grid in grids.values():
        for d in valid_depth_bins:
            slice_data = grid.grids[d]
            valid_values = slice_data[~np.isnan(slice_data)]
            if len(valid_values) > 0:
                all_values.extend(valid_values)

    if len(all_values) == 0:
        vmax = 1.0  # Fallback if all grids are NaN
    else:
        vmax = np.max(np.abs(all_values))

    # Convert to millimeters
    vmax_mm = vmax * 1000.0
    vmin_mm = -vmax_mm

    # Create figure
    figsize = (3 * n_cols + 1.5, 3 * n_runs + 1)
    fig, axes = plt.subplots(
        n_runs, n_cols, figsize=figsize, squeeze=False, constrained_layout=True
    )

    # Plot each run x depth bin combination
    for row_idx, label in enumerate(sorted_labels):
        grid = grids[label]

        for col_idx, depth_idx in enumerate(valid_depth_bins):
            ax = axes[row_idx, col_idx]

            # Get the signed error grid for this depth slice (in mm)
            error_grid_mm = grid.grids[depth_idx] * 1000.0  # Convert to mm

            # Get physical extent for this grid
            x_min, x_max = grid.x_edges[0], grid.x_edges[-1]
            y_min, y_max = grid.y_edges[0], grid.y_edges[-1]
            extent = [x_min, x_max, y_min, y_max]

            # Create norm for symmetric colormap
            norm = TwoSlopeNorm(vmin=vmin_mm, vcenter=0, vmax=vmax_mm)

            # Plot heatmap (origin="upper" to match Y-down convention)
            im = ax.imshow(
                error_grid_mm,
                cmap="RdBu_r",
                norm=norm,
                extent=extent,
                origin="upper",
                aspect="auto",
                interpolation="nearest",
            )

            # Column titles (depth ranges) on top row only
            if row_idx == 0:
                z_min = grid.depth_bin_edges[depth_idx]
                z_max = grid.depth_bin_edges[depth_idx + 1]
                ax.set_title(f"Z: {z_min:.2f}-{z_max:.2f} m", fontsize=10)

            # Row labels (run names) on leftmost column only
            if col_idx == 0:
                ax.set_ylabel(f"{label}\nY (meters)", fontsize=9)
            else:
                ax.set_ylabel("")

            # X-axis label on bottom row only
            if row_idx == n_runs - 1:
                ax.set_xlabel("X (meters)", fontsize=9)
            else:
                ax.set_xlabel("")

            # Tick formatting
            ax.tick_params(labelsize=8)

    # Add shared colorbar
    cbar = fig.colorbar(
        axes[0, 0].images[0],
        ax=axes,
        orientation="vertical",
        fraction=0.02,
        pad=0.02,
        label="Signed error (mm)",
    )
    cbar.ax.tick_params(labelsize=8)

    # Overall title
    fig.suptitle("Signed Error by XY Position and Depth", fontsize=14)

    return fig


def plot_depth_error_comparison(depth_data: dict[str, "DepthBinnedErrors"]):
    """Create line plot of depth-stratified signed error across runs.

    Args:
        depth_data: Dict mapping run label -> DepthBinnedErrors

    Returns:
        matplotlib Figure object

    Notes:
        - X-axis: depth (Z position in meters)
        - Y-axis: signed mean error in millimeters
        - One line per run with legend
        - Bins with zero counts are skipped (NaN values create gaps)
        - Optional shaded region showing +/- 1 std per run
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    # Import DepthBinnedErrors locally to avoid circular import
    from aquacal.validation.reconstruction import DepthBinnedErrors

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each run
    colors = plt.cm.tab10(np.linspace(0, 1, len(depth_data)))

    for i, (label, binned) in enumerate(depth_data.items()):
        # Convert to millimeters
        signed_means_mm = binned.signed_means * 1000.0
        signed_stds_mm = binned.signed_stds * 1000.0

        # Plot line (NaN values create gaps)
        ax.plot(
            binned.bin_centers,
            signed_means_mm,
            marker="o",
            label=label,
            color=colors[i],
            linewidth=2,
            markersize=6,
        )

        # Optional: shaded region for +/- 1 std
        # Only plot where we have valid data (non-NaN)
        valid_mask = ~np.isnan(signed_means_mm)
        if np.any(valid_mask):
            ax.fill_between(
                binned.bin_centers[valid_mask],
                (signed_means_mm - signed_stds_mm)[valid_mask],
                (signed_means_mm + signed_stds_mm)[valid_mask],
                color=colors[i],
                alpha=0.2,
            )

    # Add horizontal reference line at Y=0
    ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    # Labels and formatting
    ax.set_xlabel("Depth Z (meters)")
    ax.set_ylabel("Signed mean error (mm)")
    ax.set_title("Depth-Stratified Signed Error")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig
