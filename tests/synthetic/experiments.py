"""Phase 7 synthetic refractive comparison experiments.

Three experiments comparing refractive (n_water=1.333) vs non-refractive
(n_water=1.0) calibration on synthetic data with known ground truth.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path for imports when running as script
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tests.synthetic.experiment_helpers import (  # noqa: E402
    calibrate_synthetic,
    compute_per_camera_errors,
    evaluate_reconstruction,
)
from tests.synthetic.ground_truth import (  # noqa: E402
    SyntheticScenario,
    create_scenario,
)

# Consistent color palette
COLOR_REFRACTIVE = "#2196F3"  # Blue
COLOR_NON_REFRACTIVE = "#F44336"  # Red
LABEL_REFRACTIVE = "Refractive"
LABEL_NON_REFRACTIVE = "Non-refractive"


def run_experiment_1(output_dir: str | Path, seed: int = 42) -> dict:
    """
    Experiment 1: Parameter Fidelity.

    Calibrate both refractive and non-refractive models on the same synthetic data,
    compare recovered parameters against ground truth.

    Args:
        output_dir: Directory to save plots and CSV
        seed: Random seed for reproducibility

    Returns:
        Dict with results:
        - 'errors_refractive': per-camera errors for refractive model
        - 'errors_nonrefractive': per-camera errors for non-refractive model
        - 'scenario': the SyntheticScenario used
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Experiment 1: Parameter Fidelity")

    # Generate realistic scenario
    scenario = create_scenario("realistic", seed=seed)
    print(
        f"  Scenario: {len(scenario.intrinsics)} cameras, "
        f"{len(scenario.board_poses)} frames, {scenario.noise_std}px noise"
    )

    # Calibrate with refractive model
    print("  Calibrating refractive model...")
    result_refractive, _ = calibrate_synthetic(
        scenario, n_water=1.333, refine_intrinsics=True
    )
    errors_refractive = compute_per_camera_errors(result_refractive, scenario)

    # Calibrate with non-refractive model
    print("  Calibrating non-refractive model...")
    result_nonrefractive, _ = calibrate_synthetic(
        scenario, n_water=1.0, refine_intrinsics=True
    )
    errors_nonrefractive = compute_per_camera_errors(result_nonrefractive, scenario)

    # Generate plots
    print("  Generating plots...")
    _plot_focal_length_error(errors_refractive, errors_nonrefractive, output_dir)
    _plot_camera_xy_positions(
        scenario, result_refractive, result_nonrefractive, output_dir
    )
    _plot_camera_z_error(errors_refractive, errors_nonrefractive, output_dir)
    _plot_distortion_error(errors_refractive, errors_nonrefractive, output_dir)

    # Save CSV
    _save_parameter_errors_csv(errors_refractive, errors_nonrefractive, output_dir)

    print(f"  Results saved to {output_dir}")

    return {
        "errors_refractive": errors_refractive,
        "errors_nonrefractive": errors_nonrefractive,
        "scenario": scenario,
    }


def _plot_focal_length_error(
    errors_refr: dict, errors_nonrefr: dict, output_dir: Path
) -> None:
    """Plot 1: Focal length recovery error (grouped bar chart)."""
    camera_names = sorted(errors_refr.keys())
    x = np.arange(len(camera_names))
    width = 0.35

    focal_errors_refr = [
        errors_refr[cam]["focal_length_error_pct"] for cam in camera_names
    ]
    focal_errors_nonrefr = [
        errors_nonrefr[cam]["focal_length_error_pct"] for cam in camera_names
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        focal_errors_refr,
        width,
        label=LABEL_REFRACTIVE,
        color=COLOR_REFRACTIVE,
    )
    ax.bar(
        x + width / 2,
        focal_errors_nonrefr,
        width,
        label=LABEL_NON_REFRACTIVE,
        color=COLOR_NON_REFRACTIVE,
    )

    ax.set_xlabel("Camera")
    ax.set_ylabel("Focal Length Error (%)")
    ax.set_title("Focal Length Recovery Error")
    ax.set_xticks(x)
    ax.set_xticklabels(camera_names, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp1_focal_length_error.png", dpi=150)
    plt.close()


def _plot_camera_xy_positions(
    scenario: SyntheticScenario,
    result_refr,
    result_nonrefr,
    output_dir: Path,
) -> None:
    """Plot 2: Camera XY position recovery (top-down scatter with arrows)."""
    fig, ax = plt.subplots(figsize=(8, 8))

    camera_names = sorted(scenario.intrinsics.keys())

    for cam_name in camera_names:
        # Ground truth position
        C_gt = scenario.extrinsics[cam_name].C
        x_gt, y_gt = C_gt[0], C_gt[1]

        # Refractive position
        C_refr = result_refr.cameras[cam_name].extrinsics.C
        x_refr, y_refr = C_refr[0], C_refr[1]

        # Non-refractive position
        C_nonrefr = result_nonrefr.cameras[cam_name].extrinsics.C
        x_nonrefr, y_nonrefr = C_nonrefr[0], C_nonrefr[1]

        # Plot ground truth (black dot)
        if cam_name == camera_names[0]:
            ax.scatter(
                x_gt,
                y_gt,
                color="black",
                s=100,
                zorder=3,
                label="Ground Truth",
                marker="o",
            )
        else:
            ax.scatter(x_gt, y_gt, color="black", s=100, zorder=3, marker="o")

        # Plot refractive (blue dot + arrow)
        if cam_name == camera_names[0]:
            ax.scatter(
                x_refr,
                y_refr,
                color=COLOR_REFRACTIVE,
                s=50,
                zorder=4,
                label=LABEL_REFRACTIVE,
                alpha=0.7,
            )
        else:
            ax.scatter(
                x_refr, y_refr, color=COLOR_REFRACTIVE, s=50, zorder=4, alpha=0.7
            )
        ax.arrow(
            x_gt,
            y_gt,
            x_refr - x_gt,
            y_refr - y_gt,
            color=COLOR_REFRACTIVE,
            alpha=0.5,
            head_width=0.02,
            head_length=0.01,
            length_includes_head=True,
        )

        # Plot non-refractive (red dot + arrow)
        if cam_name == camera_names[0]:
            ax.scatter(
                x_nonrefr,
                y_nonrefr,
                color=COLOR_NON_REFRACTIVE,
                s=50,
                zorder=4,
                label=LABEL_NON_REFRACTIVE,
                alpha=0.7,
            )
        else:
            ax.scatter(
                x_nonrefr,
                y_nonrefr,
                color=COLOR_NON_REFRACTIVE,
                s=50,
                zorder=4,
                alpha=0.7,
            )
        ax.arrow(
            x_gt,
            y_gt,
            x_nonrefr - x_gt,
            y_nonrefr - y_gt,
            color=COLOR_NON_REFRACTIVE,
            alpha=0.5,
            head_width=0.02,
            head_length=0.01,
            length_includes_head=True,
        )

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Camera XY Position Recovery")
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp1_camera_xy_positions.png", dpi=150)
    plt.close()


def _plot_camera_z_error(
    errors_refr: dict, errors_nonrefr: dict, output_dir: Path
) -> None:
    """Plot 3: Camera Z position error (horizontal grouped bar chart)."""
    camera_names = sorted(errors_refr.keys())
    y = np.arange(len(camera_names))
    height = 0.35

    z_errors_refr = [errors_refr[cam]["z_position_error_mm"] for cam in camera_names]
    z_errors_nonrefr = [
        errors_nonrefr[cam]["z_position_error_mm"] for cam in camera_names
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        y - height / 2,
        z_errors_refr,
        height,
        label=LABEL_REFRACTIVE,
        color=COLOR_REFRACTIVE,
    )
    ax.barh(
        y + height / 2,
        z_errors_nonrefr,
        height,
        label=LABEL_NON_REFRACTIVE,
        color=COLOR_NON_REFRACTIVE,
    )

    ax.set_ylabel("Camera")
    ax.set_xlabel("Z Position Error (mm)")
    ax.set_title("Camera Z Position Error")
    ax.set_yticks(y)
    ax.set_yticklabels(camera_names)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp1_camera_z_error.png", dpi=150)
    plt.close()


def _plot_distortion_error(
    errors_refr: dict, errors_nonrefr: dict, output_dir: Path
) -> None:
    """Plot 4: Distortion coefficient error (grouped bar chart)."""
    camera_names = sorted(errors_refr.keys())
    x = np.arange(len(camera_names))
    width = 0.35

    # Compute max distortion error per camera (max of |k1_err|, |k2_err|)
    dist_errors_refr = [
        max(abs(errors_refr[cam]["k1_error"]), abs(errors_refr[cam]["k2_error"]))
        for cam in camera_names
    ]
    dist_errors_nonrefr = [
        max(abs(errors_nonrefr[cam]["k1_error"]), abs(errors_nonrefr[cam]["k2_error"]))
        for cam in camera_names
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        dist_errors_refr,
        width,
        label=LABEL_REFRACTIVE,
        color=COLOR_REFRACTIVE,
    )
    ax.bar(
        x + width / 2,
        dist_errors_nonrefr,
        width,
        label=LABEL_NON_REFRACTIVE,
        color=COLOR_NON_REFRACTIVE,
    )

    ax.set_xlabel("Camera")
    ax.set_ylabel("Max Distortion Error (|k1|, |k2|)")
    ax.set_title("Distortion Coefficient Error")
    ax.set_xticks(x)
    ax.set_xticklabels(camera_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Check if all errors are zero
    if max(max(dist_errors_refr), max(dist_errors_nonrefr)) < 1e-9:
        ax.text(
            0.5,
            0.95,
            "Note: Distortion not refined in Stage 4 (all errors zero)",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
            style="italic",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()
    plt.savefig(output_dir / "exp1_distortion_error.png", dpi=150)
    plt.close()


def _save_parameter_errors_csv(
    errors_refr: dict, errors_nonrefr: dict, output_dir: Path
) -> None:
    """Save per-camera parameter errors to CSV."""
    csv_path = output_dir / "exp1_parameter_errors.csv"

    camera_names = sorted(errors_refr.keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "camera",
                "focal_length_error_pct_refr",
                "focal_length_error_pct_nonrefr",
                "z_error_mm_refr",
                "z_error_mm_nonrefr",
                "k1_error_refr",
                "k1_error_nonrefr",
                "k2_error_refr",
                "k2_error_nonrefr",
            ]
        )

        for cam in camera_names:
            writer.writerow(
                [
                    cam,
                    f"{errors_refr[cam]['focal_length_error_pct']:.4f}",
                    f"{errors_nonrefr[cam]['focal_length_error_pct']:.4f}",
                    f"{errors_refr[cam]['z_position_error_mm']:.4f}",
                    f"{errors_nonrefr[cam]['z_position_error_mm']:.4f}",
                    f"{errors_refr[cam]['k1_error']:.6f}",
                    f"{errors_nonrefr[cam]['k1_error']:.6f}",
                    f"{errors_refr[cam]['k2_error']:.6f}",
                    f"{errors_nonrefr[cam]['k2_error']:.6f}",
                ]
            )


def run_experiment_2(output_dir: str | Path, seed: int = 42) -> dict:
    """
    Experiment 2: Depth Generalization.

    Calibrate both models on a narrow depth band, evaluate reconstruction
    accuracy at multiple test depths using dense XY grid poses.

    Args:
        output_dir: Directory to save plots and CSV
        seed: Random seed for reproducibility

    Returns:
        Dict with results per depth and model
    """
    from aquacal.config.schema import BoardConfig
    from aquacal.core.board import BoardGeometry
    from tests.synthetic.ground_truth import (
        SyntheticScenario,
        generate_dense_xy_grid,
        generate_real_rig_array,
        generate_real_rig_trajectory,
        generate_synthetic_detections,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Experiment 2: Depth Generalization")

    # Generate 13-camera rig
    intrinsics, extrinsics, water_zs = generate_real_rig_array(
        height_above_water=0.75, height_variation=0.002, seed=seed
    )

    # Board config (matches realistic scenario)
    board_config = BoardConfig(
        squares_x=12,
        squares_y=9,
        square_size=0.060,
        marker_size=0.045,
        dictionary="DICT_5X5_100",
    )
    board = BoardGeometry(board_config)

    # Calibration trajectory: narrow depth band (0.95-1.05m, 50 frames)
    calib_poses = generate_real_rig_trajectory(
        n_frames=50, depth_range=(0.95, 1.05), seed=seed
    )

    # Build calibration scenario
    calib_scenario = SyntheticScenario(
        name="narrow_depth",
        board_config=board_config,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        water_zs=water_zs,
        board_poses=calib_poses,
        noise_std=0.5,
        description="Narrow depth band calibration (0.95-1.05m)",
    )

    # Calibrate both models
    print("  Calibrating refractive model...")
    result_refractive, _ = calibrate_synthetic(
        calib_scenario, n_water=1.333, refine_intrinsics=True
    )

    print("  Calibrating non-refractive model...")
    result_nonrefractive, _ = calibrate_synthetic(
        calib_scenario, n_water=1.0, refine_intrinsics=True
    )

    # Test depths
    test_depths = [0.80, 0.90, 1.00, 1.10, 1.20, 1.40, 1.70, 2.00]

    # Results accumulators
    results_refr = []
    results_nonrefr = []
    spatial_data_refr = {}  # depth -> SpatialMeasurements
    spatial_data_nonrefr = {}

    for depth in test_depths:
        print(f"  Evaluating depth Z={depth:.2f}m...")

        # Generate dense 7x7 XY grid test poses
        test_poses = generate_dense_xy_grid(
            depth=depth,
            n_grid=7,
            xy_extent=0.5,
            tilt_deg=3.0,
            frame_offset=1000,
            seed=seed,
        )

        # Generate test detections (using ground truth parameters)
        test_detections = generate_synthetic_detections(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            water_zs=water_zs,
            board=board,
            board_poses=test_poses,
            noise_std=0.5,
            seed=seed + int(depth * 100),  # Vary seed per depth
        )

        # Evaluate reconstruction for both calibrations
        errors_refr = evaluate_reconstruction(result_refractive, board, test_detections)
        errors_nonrefr = evaluate_reconstruction(
            result_nonrefractive, board, test_detections
        )

        # Compute metrics
        signed_mean_refr = errors_refr.signed_mean * 1000  # to mm
        signed_mean_nonrefr = errors_nonrefr.signed_mean * 1000
        rmse_refr = errors_refr.rmse * 1000
        rmse_nonrefr = errors_nonrefr.rmse * 1000

        # Scale factor: mean(measured/true)
        # For each measurement, measured = true + signed_error
        # So measured/true = 1 + signed_error/true
        true_dist = board.config.square_size
        scale_refr = 1.0 + (errors_refr.signed_mean / true_dist)
        scale_nonrefr = 1.0 + (errors_nonrefr.signed_mean / true_dist)

        n_measurements_refr = errors_refr.num_comparisons
        n_measurements_nonrefr = errors_nonrefr.num_comparisons

        # Warn if too few measurements
        if n_measurements_refr < 100:
            print(
                f"    Warning: Only {n_measurements_refr} measurements for "
                f"refractive at depth {depth}m"
            )
        if n_measurements_nonrefr < 100:
            print(
                f"    Warning: Only {n_measurements_nonrefr} measurements for "
                f"non-refractive at depth {depth}m"
            )

        results_refr.append(
            {
                "depth": depth,
                "signed_mean_mm": signed_mean_refr,
                "rmse_mm": rmse_refr,
                "scale": scale_refr,
                "n_measurements": n_measurements_refr,
            }
        )

        results_nonrefr.append(
            {
                "depth": depth,
                "signed_mean_mm": signed_mean_nonrefr,
                "rmse_mm": rmse_nonrefr,
                "scale": scale_nonrefr,
                "n_measurements": n_measurements_nonrefr,
            }
        )

        # Store spatial measurements for heatmaps
        if errors_refr.spatial is not None:
            spatial_data_refr[depth] = errors_refr.spatial
        if errors_nonrefr.spatial is not None:
            spatial_data_nonrefr[depth] = errors_nonrefr.spatial

    # Generate plots
    print("  Generating plots...")
    _plot_signed_error_vs_depth(results_refr, results_nonrefr, output_dir)
    _plot_rmse_vs_depth(results_refr, results_nonrefr, output_dir)
    _plot_scale_factor_vs_depth(results_refr, results_nonrefr, output_dir)
    _plot_xy_heatmaps(spatial_data_refr, spatial_data_nonrefr, output_dir)

    # Save CSV
    _save_depth_metrics_csv(results_refr, results_nonrefr, output_dir)

    print(f"  Results saved to {output_dir}")

    return {
        "results_refractive": results_refr,
        "results_nonrefractive": results_nonrefr,
        "spatial_refractive": spatial_data_refr,
        "spatial_nonrefractive": spatial_data_nonrefr,
    }


def _plot_signed_error_vs_depth(
    results_refr: list, results_nonrefr: list, output_dir: Path
) -> None:
    """Plot signed mean error vs test depth."""
    depths_refr = [r["depth"] for r in results_refr]
    signed_means_refr = [r["signed_mean_mm"] for r in results_refr]

    depths_nonrefr = [r["depth"] for r in results_nonrefr]
    signed_means_nonrefr = [r["signed_mean_mm"] for r in results_nonrefr]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Shaded calibration band
    ax.axvspan(0.95, 1.05, alpha=0.2, color="gray", label="Calibration Range")

    # Lines
    ax.plot(
        depths_refr,
        signed_means_refr,
        marker="o",
        color=COLOR_REFRACTIVE,
        label=LABEL_REFRACTIVE,
        linewidth=2,
    )
    ax.plot(
        depths_nonrefr,
        signed_means_nonrefr,
        marker="s",
        color=COLOR_NON_REFRACTIVE,
        label=LABEL_NON_REFRACTIVE,
        linewidth=2,
    )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Test Depth (m)")
    ax.set_ylabel("Signed Mean Error (mm)")
    ax.set_title("Signed Mean Error vs Test Depth")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp2_signed_error_vs_depth.png", dpi=150)
    plt.close()


def _plot_rmse_vs_depth(
    results_refr: list, results_nonrefr: list, output_dir: Path
) -> None:
    """Plot RMSE vs test depth."""
    depths_refr = [r["depth"] for r in results_refr]
    rmse_refr = [r["rmse_mm"] for r in results_refr]

    depths_nonrefr = [r["depth"] for r in results_nonrefr]
    rmse_nonrefr = [r["rmse_mm"] for r in results_nonrefr]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Shaded calibration band
    ax.axvspan(0.95, 1.05, alpha=0.2, color="gray", label="Calibration Range")

    # Lines
    ax.plot(
        depths_refr,
        rmse_refr,
        marker="o",
        color=COLOR_REFRACTIVE,
        label=LABEL_REFRACTIVE,
        linewidth=2,
    )
    ax.plot(
        depths_nonrefr,
        rmse_nonrefr,
        marker="s",
        color=COLOR_NON_REFRACTIVE,
        label=LABEL_NON_REFRACTIVE,
        linewidth=2,
    )

    ax.set_xlabel("Test Depth (m)")
    ax.set_ylabel("Reconstruction RMSE (mm)")
    ax.set_title("Reconstruction RMSE vs Test Depth")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp2_rmse_vs_depth.png", dpi=150)
    plt.close()


def _plot_scale_factor_vs_depth(
    results_refr: list, results_nonrefr: list, output_dir: Path
) -> None:
    """Plot scale factor vs test depth."""
    depths_refr = [r["depth"] for r in results_refr]
    scale_refr = [r["scale"] for r in results_refr]

    depths_nonrefr = [r["depth"] for r in results_nonrefr]
    scale_nonrefr = [r["scale"] for r in results_nonrefr]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Shaded calibration band
    ax.axvspan(0.95, 1.05, alpha=0.2, color="gray", label="Calibration Range")

    # Lines
    ax.plot(
        depths_refr,
        scale_refr,
        marker="o",
        color=COLOR_REFRACTIVE,
        label=LABEL_REFRACTIVE,
        linewidth=2,
    )
    ax.plot(
        depths_nonrefr,
        scale_nonrefr,
        marker="s",
        color=COLOR_NON_REFRACTIVE,
        label=LABEL_NON_REFRACTIVE,
        linewidth=2,
    )

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Test Depth (m)")
    ax.set_ylabel("Scale Factor (measured/true)")
    ax.set_title("Scale Factor vs Test Depth")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp2_scale_factor_vs_depth.png", dpi=150)
    plt.close()


def _plot_xy_heatmaps(
    spatial_refr: dict, spatial_nonrefr: dict, output_dir: Path
) -> None:
    """Plot XY heatmap grid across depth slices."""
    # Select representative depths (4-5 columns)
    all_depths = sorted(spatial_refr.keys())
    if len(all_depths) >= 5:
        # Take first, 25%, 50%, 75%, last
        indices = [
            0,
            len(all_depths) // 4,
            len(all_depths) // 2,
            3 * len(all_depths) // 4,
            -1,
        ]
        selected_depths = [all_depths[i] for i in indices]
    else:
        selected_depths = all_depths

    n_cols = len(selected_depths)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    # Compute global vmin/vmax for shared color scale
    all_errors = []
    for depth in selected_depths:
        if depth in spatial_refr and spatial_refr[depth] is not None:
            all_errors.extend(spatial_refr[depth].signed_errors * 1000)  # to mm
        if depth in spatial_nonrefr and spatial_nonrefr[depth] is not None:
            all_errors.extend(spatial_nonrefr[depth].signed_errors * 1000)

    vmax = np.percentile(np.abs(all_errors), 95) if len(all_errors) > 0 else 1.0
    vmin = -vmax

    # Create heatmaps
    for col_idx, depth in enumerate(selected_depths):
        # Refractive (top row)
        ax_refr = axes[0, col_idx] if n_cols > 1 else axes[0]
        if depth in spatial_refr and spatial_refr[depth] is not None:
            _plot_single_xy_heatmap(
                ax_refr,
                spatial_refr[depth],
                vmin,
                vmax,
                f"{depth:.2f}m",
                show_ylabel=(col_idx == 0),
                ylabel_text=LABEL_REFRACTIVE,
            )
        else:
            ax_refr.text(0.5, 0.5, "No data", ha="center", va="center")
            ax_refr.set_title(f"{depth:.2f}m")

        # Non-refractive (bottom row)
        ax_nonrefr = axes[1, col_idx] if n_cols > 1 else axes[1]
        if depth in spatial_nonrefr and spatial_nonrefr[depth] is not None:
            _plot_single_xy_heatmap(
                ax_nonrefr,
                spatial_nonrefr[depth],
                vmin,
                vmax,
                "",
                show_ylabel=(col_idx == 0),
                ylabel_text=LABEL_NON_REFRACTIVE,
            )
        else:
            ax_nonrefr.text(0.5, 0.5, "No data", ha="center", va="center")

    # Shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Signed Error (mm)")

    fig.suptitle("Spatial Reconstruction Error by Depth", fontsize=14, y=0.98)

    plt.savefig(output_dir / "exp2_xy_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()


def _plot_single_xy_heatmap(
    ax, spatial, vmin, vmax, title, show_ylabel=False, ylabel_text=""
) -> None:
    """Helper to plot a single XY heatmap from SpatialMeasurements."""
    positions = spatial.positions  # (N, 3)
    signed_errors = spatial.signed_errors * 1000  # to mm

    # Extract XY coordinates
    x = positions[:, 0]
    y = positions[:, 1]

    # Bin into grid
    n_bins = 10
    x_edges = np.linspace(x.min() - 0.05, x.max() + 0.05, n_bins + 1)
    y_edges = np.linspace(y.min() - 0.05, y.max() + 0.05, n_bins + 1)

    # Compute mean signed error per cell
    grid = np.full((n_bins, n_bins), np.nan)
    for i in range(n_bins):
        for j in range(n_bins):
            mask = (
                (x >= x_edges[i])
                & (x < x_edges[i + 1])
                & (y >= y_edges[j])
                & (y < y_edges[j + 1])
            )
            if np.any(mask):
                grid[j, i] = np.mean(signed_errors[mask])

    # Plot heatmap
    _im = ax.imshow(
        grid,
        origin="lower",
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (m)", fontsize=8)
    if show_ylabel:
        ax.set_ylabel(ylabel_text + "\nY (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")


def _save_depth_metrics_csv(
    results_refr: list, results_nonrefr: list, output_dir: Path
) -> None:
    """Save depth-stratified metrics to CSV."""
    csv_path = output_dir / "exp2_depth_metrics.csv"

    # Merge results by depth
    depths = sorted(
        set([r["depth"] for r in results_refr] + [r["depth"] for r in results_nonrefr])
    )

    refr_by_depth = {r["depth"]: r for r in results_refr}
    nonrefr_by_depth = {r["depth"]: r for r in results_nonrefr}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "depth_m",
                "signed_mean_mm_refr",
                "signed_mean_mm_nonrefr",
                "rmse_mm_refr",
                "rmse_mm_nonrefr",
                "scale_refr",
                "scale_nonrefr",
                "n_measurements_refr",
                "n_measurements_nonrefr",
            ]
        )

        for depth in depths:
            r_refr = refr_by_depth.get(depth, {})
            r_nonrefr = nonrefr_by_depth.get(depth, {})

            writer.writerow(
                [
                    f"{depth:.2f}",
                    f"{r_refr.get('signed_mean_mm', 0.0):.4f}",
                    f"{r_nonrefr.get('signed_mean_mm', 0.0):.4f}",
                    f"{r_refr.get('rmse_mm', 0.0):.4f}",
                    f"{r_nonrefr.get('rmse_mm', 0.0):.4f}",
                    f"{r_refr.get('scale', 1.0):.6f}",
                    f"{r_nonrefr.get('scale', 1.0):.6f}",
                    r_refr.get("n_measurements", 0),
                    r_nonrefr.get("n_measurements", 0),
                ]
            )


def run_experiment_3(output_dir: str | Path, seed: int = 42) -> dict:
    """
    Experiment 3: Depth Scaling.

    For each of several depths, calibrate both models on data at that depth
    and evaluate reconstruction at the same depth using dense XY grid poses.

    Args:
        output_dir: Directory to save plots and CSV
        seed: Random seed for reproducibility

    Returns:
        Dict with results per depth and model
    """
    from aquacal.config.schema import BoardConfig
    from aquacal.core.board import BoardGeometry
    from tests.synthetic.ground_truth import (
        SyntheticScenario,
        generate_dense_xy_grid,
        generate_real_rig_array,
        generate_real_rig_trajectory,
        generate_synthetic_detections,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Experiment 3: Depth Scaling")

    # Generate 13-camera rig
    intrinsics, extrinsics, water_zs = generate_real_rig_array(
        height_above_water=0.75, height_variation=0.002, seed=seed
    )

    # Board config (matches realistic scenario)
    board_config = BoardConfig(
        squares_x=12,
        squares_y=9,
        square_size=0.060,
        marker_size=0.045,
        dictionary="DICT_5X5_100",
    )
    board = BoardGeometry(board_config)

    # Sweep depths
    sweep_depths = [0.85, 1.0, 1.2, 1.5, 2.0, 2.5]

    # Results accumulators
    results_refr = []
    results_nonrefr = []
    spatial_data_refr = {}  # depth -> SpatialMeasurements
    spatial_data_nonrefr = {}

    for depth_idx, depth in enumerate(sweep_depths):
        depth_seed = seed + depth_idx * 100

        print(f"  Depth {depth:.2f}m:")

        # Calibration trajectory: depth +/- 0.1m, 30 frames
        calib_poses = generate_real_rig_trajectory(
            n_frames=30,
            depth_range=(depth - 0.1, depth + 0.1),
            seed=depth_seed,
        )

        # Build calibration scenario
        calib_scenario = SyntheticScenario(
            name=f"depth_{depth:.2f}m",
            board_config=board_config,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            water_zs=water_zs,
            board_poses=calib_poses,
            noise_std=0.5,
            description=f"Calibration at depth {depth:.2f}m",
        )

        # Calibrate refractive model
        print("    Calibrating refractive model...")
        result_refractive, _ = calibrate_synthetic(
            calib_scenario, n_water=1.333, refine_intrinsics=True
        )
        errors_refr = compute_per_camera_errors(result_refractive, calib_scenario)

        # Calibrate non-refractive model
        print("    Calibrating non-refractive model...")
        result_nonrefractive, _ = calibrate_synthetic(
            calib_scenario, n_water=1.0, refine_intrinsics=True
        )
        errors_nonrefr = compute_per_camera_errors(result_nonrefractive, calib_scenario)

        # Generate test poses at same depth (dense 7x7 grid)
        test_poses = generate_dense_xy_grid(
            depth=depth,
            n_grid=7,
            xy_extent=0.5,
            tilt_deg=3.0,
            frame_offset=1000,
            seed=depth_seed + 1,
        )

        # Generate test detections (using ground truth parameters)
        test_detections = generate_synthetic_detections(
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            water_zs=water_zs,
            board=board,
            board_poses=test_poses,
            noise_std=0.5,
            seed=depth_seed + 2,
        )

        # Count cameras with detections
        cam_names_with_detections = set()
        for frame_det in test_detections.frames.values():
            cam_names_with_detections.update(frame_det.detections.keys())
        n_cameras_with_detections = len(cam_names_with_detections)

        if n_cameras_with_detections < 5:
            print(
                f"    Warning: Only {n_cameras_with_detections} cameras have detections "
                f"at depth {depth}m"
            )

        # Evaluate reconstruction
        dist_errors_refr = evaluate_reconstruction(
            result_refractive, board, test_detections
        )
        dist_errors_nonrefr = evaluate_reconstruction(
            result_nonrefractive, board, test_detections
        )

        # Compute aggregate metrics
        rmse_refr = dist_errors_refr.rmse * 1000  # to mm
        rmse_nonrefr = dist_errors_nonrefr.rmse * 1000

        # Mean focal length error across cameras (%)
        focal_err_refr = np.mean(
            [abs(e["focal_length_error_pct"]) for e in errors_refr.values()]
        )
        focal_err_nonrefr = np.mean(
            [abs(e["focal_length_error_pct"]) for e in errors_nonrefr.values()]
        )

        # Mean absolute Z error across cameras (mm)
        z_err_refr = np.mean(
            [abs(e["z_position_error_mm"]) for e in errors_refr.values()]
        )
        z_err_nonrefr = np.mean(
            [abs(e["z_position_error_mm"]) for e in errors_nonrefr.values()]
        )

        results_refr.append(
            {
                "depth": depth,
                "rmse_mm": rmse_refr,
                "focal_err_pct": focal_err_refr,
                "z_err_mm": z_err_refr,
                "n_cameras": n_cameras_with_detections,
            }
        )

        results_nonrefr.append(
            {
                "depth": depth,
                "rmse_mm": rmse_nonrefr,
                "focal_err_pct": focal_err_nonrefr,
                "z_err_mm": z_err_nonrefr,
                "n_cameras": n_cameras_with_detections,
            }
        )

        # Store spatial measurements for heatmaps (at select depths)
        if depth in [0.85, 1.5, 2.5]:
            if dist_errors_refr.spatial is not None:
                spatial_data_refr[depth] = dist_errors_refr.spatial
            if dist_errors_nonrefr.spatial is not None:
                spatial_data_nonrefr[depth] = dist_errors_nonrefr.spatial

    # Generate plots
    print("  Generating plots...")
    _plot_exp3_rmse_vs_depth(results_refr, results_nonrefr, output_dir)
    _plot_exp3_focal_error_vs_depth(results_refr, results_nonrefr, output_dir)
    _plot_exp3_z_error_vs_depth(results_refr, results_nonrefr, output_dir)
    _plot_exp3_xy_heatmaps(spatial_data_refr, spatial_data_nonrefr, output_dir)

    # Save CSV
    _save_exp3_depth_scaling_csv(results_refr, results_nonrefr, output_dir)

    print(f"  Results saved to {output_dir}")

    return {
        "results_refractive": results_refr,
        "results_nonrefractive": results_nonrefr,
        "spatial_refractive": spatial_data_refr,
        "spatial_nonrefractive": spatial_data_nonrefr,
    }


def _plot_exp3_rmse_vs_depth(
    results_refr: list, results_nonrefr: list, output_dir: Path
) -> None:
    """Plot RMSE vs calibration/test depth."""
    depths_refr = [r["depth"] for r in results_refr]
    rmse_refr = [r["rmse_mm"] for r in results_refr]

    depths_nonrefr = [r["depth"] for r in results_nonrefr]
    rmse_nonrefr = [r["rmse_mm"] for r in results_nonrefr]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        depths_refr,
        rmse_refr,
        marker="o",
        color=COLOR_REFRACTIVE,
        label=LABEL_REFRACTIVE,
        linewidth=2,
    )
    ax.plot(
        depths_nonrefr,
        rmse_nonrefr,
        marker="s",
        color=COLOR_NON_REFRACTIVE,
        label=LABEL_NON_REFRACTIVE,
        linewidth=2,
    )

    ax.set_xlabel("Calibration/Test Depth (m)")
    ax.set_ylabel("Reconstruction RMSE (mm)")
    ax.set_title("Reconstruction RMSE vs Depth (Same-Depth Calibration)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp3_rmse_vs_depth.png", dpi=150)
    plt.close()


def _plot_exp3_focal_error_vs_depth(
    results_refr: list, results_nonrefr: list, output_dir: Path
) -> None:
    """Plot mean focal length error vs calibration depth."""
    depths_refr = [r["depth"] for r in results_refr]
    focal_err_refr = [r["focal_err_pct"] for r in results_refr]

    depths_nonrefr = [r["depth"] for r in results_nonrefr]
    focal_err_nonrefr = [r["focal_err_pct"] for r in results_nonrefr]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        depths_refr,
        focal_err_refr,
        marker="o",
        color=COLOR_REFRACTIVE,
        label=LABEL_REFRACTIVE,
        linewidth=2,
    )
    ax.plot(
        depths_nonrefr,
        focal_err_nonrefr,
        marker="s",
        color=COLOR_NON_REFRACTIVE,
        label=LABEL_NON_REFRACTIVE,
        linewidth=2,
    )

    ax.set_xlabel("Calibration Depth (m)")
    ax.set_ylabel("Mean Focal Length Error (%)")
    ax.set_title("Focal Length Error vs Calibration Depth")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp3_focal_error_vs_depth.png", dpi=150)
    plt.close()


def _plot_exp3_z_error_vs_depth(
    results_refr: list, results_nonrefr: list, output_dir: Path
) -> None:
    """Plot mean camera Z error vs calibration depth."""
    depths_refr = [r["depth"] for r in results_refr]
    z_err_refr = [r["z_err_mm"] for r in results_refr]

    depths_nonrefr = [r["depth"] for r in results_nonrefr]
    z_err_nonrefr = [r["z_err_mm"] for r in results_nonrefr]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        depths_refr,
        z_err_refr,
        marker="o",
        color=COLOR_REFRACTIVE,
        label=LABEL_REFRACTIVE,
        linewidth=2,
    )
    ax.plot(
        depths_nonrefr,
        z_err_nonrefr,
        marker="s",
        color=COLOR_NON_REFRACTIVE,
        label=LABEL_NON_REFRACTIVE,
        linewidth=2,
    )

    ax.set_xlabel("Calibration Depth (m)")
    ax.set_ylabel("Mean Absolute Camera Z Error (mm)")
    ax.set_title("Camera Z Position Error vs Calibration Depth")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "exp3_z_error_vs_depth.png", dpi=150)
    plt.close()


def _plot_exp3_xy_heatmaps(
    spatial_refr: dict, spatial_nonrefr: dict, output_dir: Path
) -> None:
    """Plot XY heatmap grid at select depths (0.85, 1.5, 2.5m)."""
    selected_depths = [0.85, 1.5, 2.5]
    n_cols = len(selected_depths)

    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))

    # Compute global vmin/vmax for shared color scale
    all_errors = []
    for depth in selected_depths:
        if depth in spatial_refr and spatial_refr[depth] is not None:
            all_errors.extend(spatial_refr[depth].signed_errors * 1000)  # to mm
        if depth in spatial_nonrefr and spatial_nonrefr[depth] is not None:
            all_errors.extend(spatial_nonrefr[depth].signed_errors * 1000)

    vmax = np.percentile(np.abs(all_errors), 95) if len(all_errors) > 0 else 1.0
    vmin = -vmax

    # Create heatmaps
    for col_idx, depth in enumerate(selected_depths):
        # Refractive (top row)
        ax_refr = axes[0, col_idx]
        if depth in spatial_refr and spatial_refr[depth] is not None:
            _plot_single_xy_heatmap(
                ax_refr,
                spatial_refr[depth],
                vmin,
                vmax,
                f"{depth:.2f}m",
                show_ylabel=(col_idx == 0),
                ylabel_text=LABEL_REFRACTIVE,
            )
        else:
            ax_refr.text(0.5, 0.5, "No data", ha="center", va="center")
            ax_refr.set_title(f"{depth:.2f}m")

        # Non-refractive (bottom row)
        ax_nonrefr = axes[1, col_idx]
        if depth in spatial_nonrefr and spatial_nonrefr[depth] is not None:
            _plot_single_xy_heatmap(
                ax_nonrefr,
                spatial_nonrefr[depth],
                vmin,
                vmax,
                "",
                show_ylabel=(col_idx == 0),
                ylabel_text=LABEL_NON_REFRACTIVE,
            )
        else:
            ax_nonrefr.text(0.5, 0.5, "No data", ha="center", va="center")

    # Shared colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Signed Error (mm)")

    fig.suptitle("Spatial Error at Shallow/Medium/Deep Depths", fontsize=14, y=0.98)

    plt.savefig(output_dir / "exp3_xy_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()


def _save_exp3_depth_scaling_csv(
    results_refr: list, results_nonrefr: list, output_dir: Path
) -> None:
    """Save depth scaling metrics to CSV."""
    csv_path = output_dir / "exp3_depth_scaling.csv"

    # Merge results by depth
    depths = sorted(
        set([r["depth"] for r in results_refr] + [r["depth"] for r in results_nonrefr])
    )

    refr_by_depth = {r["depth"]: r for r in results_refr}
    nonrefr_by_depth = {r["depth"]: r for r in results_nonrefr}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "depth_m",
                "rmse_mm_refr",
                "rmse_mm_nonrefr",
                "focal_err_pct_refr",
                "focal_err_pct_nonrefr",
                "z_err_mm_refr",
                "z_err_mm_nonrefr",
                "n_cameras_refr",
                "n_cameras_nonrefr",
            ]
        )

        for depth in depths:
            r_refr = refr_by_depth.get(depth, {})
            r_nonrefr = nonrefr_by_depth.get(depth, {})

            writer.writerow(
                [
                    f"{depth:.2f}",
                    f"{r_refr.get('rmse_mm', 0.0):.4f}",
                    f"{r_nonrefr.get('rmse_mm', 0.0):.4f}",
                    f"{r_refr.get('focal_err_pct', 0.0):.4f}",
                    f"{r_nonrefr.get('focal_err_pct', 0.0):.4f}",
                    f"{r_refr.get('z_err_mm', 0.0):.4f}",
                    f"{r_nonrefr.get('z_err_mm', 0.0):.4f}",
                    r_refr.get("n_cameras", 0),
                    r_nonrefr.get("n_cameras", 0),
                ]
            )


def assemble_summary(output_dir: str | Path) -> None:
    """
    Assemble summary CSV from individual experiment outputs.

    Reads per-experiment CSVs and produces summary.csv with key findings.
    Skips experiments whose CSV doesn't exist.

    Args:
        output_dir: Directory containing experiment outputs
    """
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.csv"

    rows = []

    # Experiment 1: Parameter fidelity
    exp1_csv = output_dir / "exp1_parameter_errors.csv"
    if exp1_csv.exists():
        with open(exp1_csv, "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)

        # Compute mean focal length error (%) for both models
        focal_refr = np.mean(
            [float(row["focal_length_error_pct_refr"]) for row in data]
        )
        focal_nonrefr = np.mean(
            [float(row["focal_length_error_pct_nonrefr"]) for row in data]
        )

        # Compute mean absolute Z error (mm) for both models
        z_refr = np.mean([abs(float(row["z_error_mm_refr"])) for row in data])
        z_nonrefr = np.mean([abs(float(row["z_error_mm_nonrefr"])) for row in data])

        rows.append(
            {
                "experiment": "1_parameter_fidelity",
                "metric": "mean_focal_error_pct_refr",
                "value": f"{focal_refr:.4f}",
            }
        )
        rows.append(
            {
                "experiment": "1_parameter_fidelity",
                "metric": "mean_focal_error_pct_nonrefr",
                "value": f"{focal_nonrefr:.4f}",
            }
        )
        rows.append(
            {
                "experiment": "1_parameter_fidelity",
                "metric": "mean_z_error_mm_refr",
                "value": f"{z_refr:.4f}",
            }
        )
        rows.append(
            {
                "experiment": "1_parameter_fidelity",
                "metric": "mean_z_error_mm_nonrefr",
                "value": f"{z_nonrefr:.4f}",
            }
        )

    # Experiment 2: Depth generalization
    exp2_csv = output_dir / "exp2_depth_metrics.csv"
    if exp2_csv.exists():
        with open(exp2_csv, "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)

        # Signed error range (max - min across depths)
        signed_refr = [float(row["signed_mean_mm_refr"]) for row in data]
        signed_nonrefr = [float(row["signed_mean_mm_nonrefr"]) for row in data]

        range_refr = max(signed_refr) - min(signed_refr)
        range_nonrefr = max(signed_nonrefr) - min(signed_nonrefr)

        # RMSE at calibration depth (1.0m) vs extreme depth (2.0m)
        calib_row = next((r for r in data if float(r["depth_m"]) == 1.0), None)
        extreme_row = next((r for r in data if float(r["depth_m"]) == 2.0), None)

        if calib_row and extreme_row:
            rmse_calib_refr = float(calib_row["rmse_mm_refr"])
            rmse_extreme_refr = float(extreme_row["rmse_mm_refr"])
            rmse_calib_nonrefr = float(calib_row["rmse_mm_nonrefr"])
            rmse_extreme_nonrefr = float(extreme_row["rmse_mm_nonrefr"])

            rows.extend(
                [
                    {
                        "experiment": "2_depth_generalization",
                        "metric": "signed_error_range_mm_refr",
                        "value": f"{range_refr:.4f}",
                    },
                    {
                        "experiment": "2_depth_generalization",
                        "metric": "signed_error_range_mm_nonrefr",
                        "value": f"{range_nonrefr:.4f}",
                    },
                    {
                        "experiment": "2_depth_generalization",
                        "metric": "rmse_at_calib_depth_refr",
                        "value": f"{rmse_calib_refr:.4f}",
                    },
                    {
                        "experiment": "2_depth_generalization",
                        "metric": "rmse_at_extreme_depth_refr",
                        "value": f"{rmse_extreme_refr:.4f}",
                    },
                    {
                        "experiment": "2_depth_generalization",
                        "metric": "rmse_at_calib_depth_nonrefr",
                        "value": f"{rmse_calib_nonrefr:.4f}",
                    },
                    {
                        "experiment": "2_depth_generalization",
                        "metric": "rmse_at_extreme_depth_nonrefr",
                        "value": f"{rmse_extreme_nonrefr:.4f}",
                    },
                ]
            )

    # Experiment 3: Depth scaling
    exp3_csv = output_dir / "exp3_depth_scaling.csv"
    if exp3_csv.exists():
        with open(exp3_csv, "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)

        # RMSE at shallowest vs deepest
        depths = [float(row["depth_m"]) for row in data]
        shallow_row = data[depths.index(min(depths))]
        deep_row = data[depths.index(max(depths))]

        rmse_shallow_refr = float(shallow_row["rmse_mm_refr"])
        rmse_deep_refr = float(deep_row["rmse_mm_refr"])
        rmse_shallow_nonrefr = float(shallow_row["rmse_mm_nonrefr"])
        rmse_deep_nonrefr = float(deep_row["rmse_mm_nonrefr"])

        rows.extend(
            [
                {
                    "experiment": "3_depth_scaling",
                    "metric": "rmse_at_shallowest_refr",
                    "value": f"{rmse_shallow_refr:.4f}",
                },
                {
                    "experiment": "3_depth_scaling",
                    "metric": "rmse_at_deepest_refr",
                    "value": f"{rmse_deep_refr:.4f}",
                },
                {
                    "experiment": "3_depth_scaling",
                    "metric": "rmse_at_shallowest_nonrefr",
                    "value": f"{rmse_shallow_nonrefr:.4f}",
                },
                {
                    "experiment": "3_depth_scaling",
                    "metric": "rmse_at_deepest_nonrefr",
                    "value": f"{rmse_deep_nonrefr:.4f}",
                },
            ]
        )

    # Write summary CSV
    with open(summary_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=["experiment", "metric", "value"])
            writer.writeheader()
            writer.writerows(rows)
        else:
            # Empty summary if no experiments ran
            f.write("experiment,metric,value\n")

    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    # Smoke test: run experiment 1
    import tempfile

    print("=== Smoke Test: Experiment 1 ===\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        results = run_experiment_1(tmpdir, seed=42)

        print("\n--- Results Summary ---")
        print("Refractive model:")
        for cam, errs in sorted(results["errors_refractive"].items()):
            print(
                f"  {cam}: focal={errs['focal_length_error_pct']:6.2f}%, "
                f"z={errs['z_position_error_mm']:6.2f}mm"
            )

        print("\nNon-refractive model:")
        for cam, errs in sorted(results["errors_nonrefractive"].items()):
            print(
                f"  {cam}: focal={errs['focal_length_error_pct']:6.2f}%, "
                f"z={errs['z_position_error_mm']:6.2f}mm"
            )

        print(f"\nPlots saved to: {tmpdir}")
        print("=== Smoke test complete ===")
