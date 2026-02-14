"""Shared experiment helpers for Phase 7 synthetic refractive comparison.

Provides calibration and evaluation utilities used by all three experiments.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

from aquacal.config.schema import (
    BoardPose,
    CalibrationResult,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    DetectionResult,
    InterfaceParams,
    DiagnosticsData,
    CalibrationMetadata,
)
from aquacal.core.board import BoardGeometry
from aquacal.calibration.extrinsics import build_pose_graph, estimate_extrinsics
from aquacal.calibration.interface_estimation import optimize_interface
from aquacal.calibration.refinement import joint_refinement
from aquacal.validation.reconstruction import (
    compute_3d_distance_errors,
    DistanceErrors,
)

# Add project root to path for imports when running as script
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from tests.synthetic.ground_truth import (
    SyntheticScenario,
    generate_synthetic_detections,
)


def calibrate_synthetic(
    scenario: SyntheticScenario,
    n_water: float,
    refine_intrinsics: bool = True,
) -> tuple[CalibrationResult, DetectionResult]:
    """
    Run full calibration pipeline (Stages 2-4) on synthetic data.

    Args:
        scenario: Synthetic scenario with ground truth
        n_water: Target refractive index for water (1.0 for non-refractive, 1.333 for refractive)
        refine_intrinsics: If True, run Stage 4 with intrinsic refinement

    Returns:
        Tuple of (CalibrationResult, DetectionResult). The detections are needed
        downstream for reconstruction evaluation.
    """
    # Create board geometry
    board = BoardGeometry(scenario.board_config)

    # Generate synthetic detections
    detections = generate_synthetic_detections(
        intrinsics=scenario.intrinsics,
        extrinsics=scenario.extrinsics,
        interface_distances=scenario.interface_distances,
        board=board,
        board_poses=scenario.board_poses,
        noise_std=scenario.noise_std,
        seed=42,
    )

    # Stage 2: Extrinsic initialization
    print("Stage 2: Extrinsic initialization...")
    reference_camera = "cam0"
    pose_graph = build_pose_graph(detections, min_cameras=2)
    initial_extrinsics = estimate_extrinsics(
        pose_graph, scenario.intrinsics, board, reference_camera
    )

    # Stage 3: Joint refractive optimization
    print("Stage 3: Joint refractive optimization...")
    interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    opt_extrinsics, opt_distances, opt_poses, rms = optimize_interface(
        detections=detections,
        intrinsics=scenario.intrinsics,
        initial_extrinsics=initial_extrinsics,
        board=board,
        reference_camera=reference_camera,
        interface_normal=interface_normal,
        n_air=1.0,
        n_water=n_water,
        loss="huber",
        loss_scale=1.0,
        min_corners=4,
    )

    # Stage 4: Intrinsic refinement (if requested)
    if refine_intrinsics:
        print("Stage 4: Intrinsic refinement...")
        stage3_result = (opt_extrinsics, opt_distances, opt_poses, rms)
        opt_extrinsics, opt_distances, opt_poses, opt_intrinsics, rms = joint_refinement(
            stage3_result=stage3_result,
            detections=detections,
            intrinsics=scenario.intrinsics,
            board=board,
            reference_camera=reference_camera,
            refine_intrinsics=True,
            interface_normal=interface_normal,
            n_air=1.0,
            n_water=n_water,
            loss="huber",
            loss_scale=1.0,
            min_corners=4,
        )
    else:
        # Use ground truth intrinsics
        opt_intrinsics = scenario.intrinsics

    # Build CalibrationResult
    cameras = {}
    for cam_name in scenario.intrinsics:
        cameras[cam_name] = CameraCalibration(
            name=cam_name,
            intrinsics=opt_intrinsics[cam_name],
            extrinsics=opt_extrinsics[cam_name],
            interface_distance=opt_distances[cam_name],
        )

    interface_params = InterfaceParams(
        normal=interface_normal,
        n_air=1.0,
        n_water=n_water,
    )

    diagnostics = DiagnosticsData(
        reprojection_error_rms=rms,
        reprojection_error_per_camera={},
        validation_3d_error_mean=0.0,
        validation_3d_error_std=0.0,
    )

    metadata = CalibrationMetadata(
        calibration_date="synthetic",
        software_version="test",
        config_hash="synthetic",
        num_frames_used=len(opt_poses),
        num_frames_holdout=0,
    )

    result = CalibrationResult(
        cameras=cameras,
        interface=interface_params,
        board=scenario.board_config,
        diagnostics=diagnostics,
        metadata=metadata,
    )

    return result, detections


def evaluate_reconstruction(
    calibration: CalibrationResult,
    board: BoardGeometry,
    test_detections: DetectionResult,
) -> DistanceErrors:
    """
    Evaluate reconstruction quality on test data.

    Args:
        calibration: Calibration result from calibrate_synthetic
        board: Board geometry
        test_detections: Detection result for test poses

    Returns:
        DistanceErrors with reconstruction error statistics and spatial measurements
    """
    return compute_3d_distance_errors(
        calibration=calibration,
        detections=test_detections,
        board=board,
        include_per_pair=False,
        include_spatial=True,
    )


def compute_per_camera_errors(
    result: CalibrationResult,
    ground_truth: SyntheticScenario,
) -> dict[str, dict[str, float]]:
    """
    Compute per-camera parameter errors vs ground truth.

    Args:
        result: Calibration result
        ground_truth: Synthetic scenario with known ground truth

    Returns:
        Dict keyed by camera name, each value containing:
        - focal_length_error_pct: Relative error in fx (%)
        - z_position_error_mm: Signed Z position error (mm)
        - xy_position_error_mm: XY position error magnitude (mm)
        - k1_error: Absolute error in k1 distortion coefficient
        - k2_error: Absolute error in k2 distortion coefficient
    """
    errors = {}

    for cam_name in ground_truth.intrinsics:
        if cam_name not in result.cameras:
            continue

        gt_intr = ground_truth.intrinsics[cam_name]
        gt_extr = ground_truth.extrinsics[cam_name]

        cal = result.cameras[cam_name]
        cal_intr = cal.intrinsics
        cal_extr = cal.extrinsics

        # Focal length error (relative, %)
        fx_gt = gt_intr.K[0, 0]
        fx_cal = cal_intr.K[0, 0]
        focal_length_error_pct = (fx_cal - fx_gt) / fx_gt * 100

        # Camera position errors
        C_gt = gt_extr.C
        C_cal = cal_extr.C

        # Z position error (signed, mm)
        z_position_error_mm = (C_cal[2] - C_gt[2]) * 1000

        # XY position error (magnitude, mm)
        xy_diff = C_cal[:2] - C_gt[:2]
        xy_position_error_mm = np.linalg.norm(xy_diff) * 1000

        # Distortion coefficient errors
        k1_gt = gt_intr.dist_coeffs[0]
        k2_gt = gt_intr.dist_coeffs[1]
        k1_cal = cal_intr.dist_coeffs[0]
        k2_cal = cal_intr.dist_coeffs[1]

        k1_error = k1_cal - k1_gt
        k2_error = k2_cal - k2_gt

        errors[cam_name] = {
            "focal_length_error_pct": focal_length_error_pct,
            "z_position_error_mm": z_position_error_mm,
            "xy_position_error_mm": xy_position_error_mm,
            "k1_error": k1_error,
            "k2_error": k2_error,
        }

    return errors


if __name__ == "__main__":
    # Smoke test: run calibration with both refractive models
    from tests.synthetic.ground_truth import create_scenario

    print("=== Smoke Test: Calibrate synthetic scenario with both models ===\n")

    # Create realistic scenario
    scenario = create_scenario("realistic", seed=42)
    print(f"Scenario: {scenario.description}")
    print(f"Cameras: {len(scenario.intrinsics)}")
    print(f"Frames: {len(scenario.board_poses)}")
    print(f"Noise: {scenario.noise_std}px\n")

    # Calibrate with refractive model
    print("--- Refractive Model (n_water=1.333) ---")
    result_refractive, detections = calibrate_synthetic(
        scenario, n_water=1.333, refine_intrinsics=True
    )
    errors_refractive = compute_per_camera_errors(result_refractive, scenario)
    print(f"Reprojection RMS: {result_refractive.diagnostics.reprojection_error_rms:.4f} px")

    # Print per-camera errors
    print("\nPer-camera errors (refractive):")
    print(f"{'Camera':<8} {'Focal(%)':>10} {'Z(mm)':>10} {'XY(mm)':>10}")
    for cam_name, errs in errors_refractive.items():
        print(
            f"{cam_name:<8} "
            f"{errs['focal_length_error_pct']:>10.3f} "
            f"{errs['z_position_error_mm']:>10.3f} "
            f"{errs['xy_position_error_mm']:>10.3f}"
        )

    # Calibrate with non-refractive model
    print("\n--- Non-Refractive Model (n_water=1.0) ---")
    result_nonrefractive, _ = calibrate_synthetic(
        scenario, n_water=1.0, refine_intrinsics=True
    )
    errors_nonrefractive = compute_per_camera_errors(result_nonrefractive, scenario)
    print(f"Reprojection RMS: {result_nonrefractive.diagnostics.reprojection_error_rms:.4f} px")

    # Print per-camera errors
    print("\nPer-camera errors (non-refractive):")
    print(f"{'Camera':<8} {'Focal(%)':>10} {'Z(mm)':>10} {'XY(mm)':>10}")
    for cam_name, errs in errors_nonrefractive.items():
        print(
            f"{cam_name:<8} "
            f"{errs['focal_length_error_pct']:>10.3f} "
            f"{errs['z_position_error_mm']:>10.3f} "
            f"{errs['xy_position_error_mm']:>10.3f}"
        )

    print("\n=== Smoke test complete ===")
