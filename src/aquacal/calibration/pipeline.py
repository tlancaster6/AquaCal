"""End-to-end calibration pipeline orchestration."""

from __future__ import annotations

import hashlib
import importlib.metadata
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from aquacal.config.schema import (
    BoardConfig,
    BoardPose,
    CalibrationConfig,
    CalibrationMetadata,
    CalibrationResult,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    DetectionResult,
    DiagnosticsData,
    FrameDetections,
    InterfaceParams,
)
from aquacal.core.board import BoardGeometry
from aquacal.calibration.intrinsics import calibrate_intrinsics_all
from aquacal.calibration.extrinsics import build_pose_graph, estimate_extrinsics
from aquacal.calibration.interface_estimation import (
    optimize_interface,
    register_auxiliary_camera,
    _compute_initial_board_poses,
)
from aquacal.calibration.refinement import joint_refinement
from aquacal.io.detection import detect_all_frames
from aquacal.io.serialization import save_calibration
from aquacal.validation.reprojection import compute_reprojection_errors
from aquacal.validation.reconstruction import compute_3d_distance_errors
from aquacal.validation.diagnostics import (
    generate_diagnostic_report,
    save_diagnostic_report,
)
from aquacal.utils.transforms import matrix_to_rvec


def load_config(config_path: str | Path) -> CalibrationConfig:
    """
    Load calibration configuration from YAML file.

    Args:
        config_path: Path to config.yaml file

    Returns:
        CalibrationConfig populated from file

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid or missing required fields
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    # Validate required sections
    required = ["board", "cameras", "paths"]
    for key in required:
        if key not in data:
            raise ValueError(f"Missing required config section: {key}")

    # Build BoardConfig (extrinsic/underwater board)
    board_data = data["board"]
    board = BoardConfig(
        squares_x=board_data["squares_x"],
        squares_y=board_data["squares_y"],
        square_size=board_data["square_size"],
        marker_size=board_data["marker_size"],
        dictionary=board_data.get("dictionary", "DICT_4X4_50"),
        legacy_pattern=board_data.get("legacy_pattern", False),
    )

    # Build optional intrinsic BoardConfig (if provided)
    intrinsic_board = None
    if "intrinsic_board" in data:
        intrinsic_data = data["intrinsic_board"]
        intrinsic_board = BoardConfig(
            squares_x=intrinsic_data["squares_x"],
            squares_y=intrinsic_data["squares_y"],
            square_size=intrinsic_data["square_size"],
            marker_size=intrinsic_data["marker_size"],
            dictionary=intrinsic_data.get("dictionary", "DICT_4X4_50"),
            legacy_pattern=intrinsic_data.get("legacy_pattern", False),
        )

    # Build paths
    paths = data["paths"]
    intrinsic_paths = {k: Path(v) for k, v in paths["intrinsic_videos"].items()}
    extrinsic_paths = {k: Path(v) for k, v in paths["extrinsic_videos"].items()}
    output_dir = Path(paths["output_dir"])

    # Auxiliary cameras (parsed early since initial_distances references it)
    auxiliary_cameras = data.get("auxiliary_cameras", [])
    if auxiliary_cameras:
        overlap = set(data["cameras"]) & set(auxiliary_cameras)
        if overlap:
            raise ValueError(
                f"auxiliary_cameras must not overlap with cameras. "
                f"Overlap: {sorted(overlap)}"
            )
        for aux_cam in auxiliary_cameras:
            if aux_cam not in intrinsic_paths:
                raise ValueError(
                    f"Auxiliary camera '{aux_cam}' missing from "
                    f"paths.intrinsic_videos"
                )
            if aux_cam not in extrinsic_paths:
                raise ValueError(
                    f"Auxiliary camera '{aux_cam}' missing from "
                    f"paths.extrinsic_videos"
                )

    # Interface settings
    interface = data.get("interface", {})
    n_air = interface.get("n_air", 1.0)
    n_water = interface.get("n_water", 1.333)
    normal_fixed = interface.get("normal_fixed", True)

    # Parse initial_interface_distances (optional)
    initial_interface_distances = None
    if "initial_distances" in interface:
        raw_distances = interface["initial_distances"]

        # Handle scalar format (apply to all cameras including auxiliary)
        if isinstance(raw_distances, (int, float)):
            if raw_distances <= 0:
                raise ValueError(
                    f"initial_distances must be positive, got {raw_distances}"
                )
            initial_interface_distances = {
                cam: float(raw_distances)
                for cam in data["cameras"] + auxiliary_cameras
            }
        # Handle dict format (per-camera)
        elif isinstance(raw_distances, dict):
            # Validate all cameras are covered
            missing_cameras = set(data["cameras"]) - set(raw_distances.keys())
            if missing_cameras:
                raise ValueError(
                    f"initial_distances dict must cover all cameras. "
                    f"Missing: {sorted(missing_cameras)}"
                )

            # Validate all distances are positive
            for cam, dist in raw_distances.items():
                if dist <= 0:
                    raise ValueError(
                        f"initial_distances['{cam}'] must be positive, got {dist}"
                    )

            # Warn about extra cameras (not in cameras or auxiliary list)
            extra_cameras = (
                set(raw_distances.keys())
                - set(data["cameras"])
                - set(auxiliary_cameras)
            )
            if extra_cameras:
                import sys
                print(
                    f"Warning: initial_distances contains cameras not in cameras list: "
                    f"{sorted(extra_cameras)}",
                    file=sys.stderr,
                )

            initial_interface_distances = {k: float(v) for k, v in raw_distances.items()}
        else:
            raise ValueError(
                f"initial_distances must be a number or dict, got {type(raw_distances).__name__}"
            )

    # Optimization settings
    opt = data.get("optimization", {})
    robust_loss = opt.get("robust_loss", "huber")
    loss_scale = opt.get("loss_scale", 1.0)
    max_cal_frames_raw = opt.get("max_calibration_frames", None)
    max_cal_frames = int(max_cal_frames_raw) if max_cal_frames_raw is not None else None
    refine_intrinsics = opt.get("refine_intrinsics", False)
    refine_auxiliary_intrinsics = opt.get("refine_auxiliary_intrinsics", False)
    # Detection settings
    det = data.get("detection", {})
    min_corners = det.get("min_corners", 8)
    min_cameras = det.get("min_cameras", 2)
    frame_step = det.get("frame_step", 1)

    # Camera model settings
    rational_model_cameras = data.get("rational_model_cameras", [])
    fisheye_cameras = data.get("fisheye_cameras", [])

    # Validate fisheye_cameras: must be subset of auxiliary_cameras
    if fisheye_cameras:
        non_aux = set(fisheye_cameras) - set(auxiliary_cameras)
        if non_aux:
            raise ValueError(
                f"fisheye_cameras must be a subset of auxiliary_cameras. "
                f"Not in auxiliary_cameras: {sorted(non_aux)}"
            )
        # Validate no overlap with rational_model_cameras
        overlap = set(fisheye_cameras) & set(rational_model_cameras)
        if overlap:
            raise ValueError(
                f"fisheye_cameras and rational_model_cameras must be disjoint. "
                f"Overlap: {sorted(overlap)}"
            )

    # Validation settings
    val = data.get("validation", {})
    holdout_fraction = val.get("holdout_fraction", 0.2)
    save_detailed = val.get("save_detailed_residuals", True)

    return CalibrationConfig(
        board=board,
        camera_names=data["cameras"],
        intrinsic_video_paths=intrinsic_paths,
        extrinsic_video_paths=extrinsic_paths,
        output_dir=output_dir,
        intrinsic_board=intrinsic_board,
        n_air=n_air,
        n_water=n_water,
        interface_normal_fixed=normal_fixed,
        robust_loss=robust_loss,
        loss_scale=loss_scale,
        min_corners_per_frame=min_corners,
        min_cameras_per_frame=min_cameras,
        frame_step=frame_step,
        holdout_fraction=holdout_fraction,
        max_calibration_frames=max_cal_frames,
        refine_intrinsics=refine_intrinsics,
        refine_auxiliary_intrinsics=refine_auxiliary_intrinsics,
        save_detailed_residuals=save_detailed,
        initial_interface_distances=initial_interface_distances,
        rational_model_cameras=rational_model_cameras,
        auxiliary_cameras=auxiliary_cameras,
        fisheye_cameras=fisheye_cameras,
    )


def split_detections(
    detections: DetectionResult,
    holdout_fraction: float,
    seed: int = 42,
) -> tuple[DetectionResult, DetectionResult]:
    """
    Split detections into calibration and validation sets.

    Randomly assigns entire frames to either set (not individual detections).

    Args:
        detections: Full detection result
        holdout_fraction: Fraction of frames for validation (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (calibration_detections, validation_detections)
    """
    frame_indices = list(detections.frames.keys())

    rng = random.Random(seed)
    rng.shuffle(frame_indices)

    n_holdout = int(len(frame_indices) * holdout_fraction)
    holdout_indices = set(frame_indices[:n_holdout])
    calibration_indices = set(frame_indices[n_holdout:])

    cal_frames = {idx: detections.frames[idx] for idx in calibration_indices}
    val_frames = {idx: detections.frames[idx] for idx in holdout_indices}

    cal_detections = DetectionResult(
        frames=cal_frames,
        camera_names=detections.camera_names,
        total_frames=len(cal_frames),
    )
    val_detections = DetectionResult(
        frames=val_frames,
        camera_names=detections.camera_names,
        total_frames=len(val_frames),
    )

    return cal_detections, val_detections


def _subsample_detections(
    detections: DetectionResult,
    max_frames: int,
) -> DetectionResult:
    """Uniformly subsample detection frames to at most max_frames.

    Selects frames at uniform temporal intervals from the sorted frame indices,
    preserving the first and last frames.

    Args:
        detections: Full detection result
        max_frames: Maximum number of frames to keep

    Returns:
        New DetectionResult with at most max_frames frames
    """
    frame_indices = sorted(detections.frames.keys())
    if len(frame_indices) <= max_frames:
        return detections

    # Uniform selection: np.linspace to pick evenly spaced indices
    selected_positions = np.round(
        np.linspace(0, len(frame_indices) - 1, max_frames)
    ).astype(int)
    selected_frames = {frame_indices[i] for i in selected_positions}

    return DetectionResult(
        frames={k: v for k, v in detections.frames.items() if k in selected_frames},
        camera_names=detections.camera_names,
        total_frames=detections.total_frames,
    )


def _save_board_reference_images(
    board: BoardGeometry,
    intrinsic_board: BoardGeometry | None,
    output_dir: Path,
) -> None:
    """
    Save reference PNG images of configured board(s) for visual verification.

    Generates grayscale ChArUco board images at 800x600 resolution with 50px
    margin. Saves extrinsic board always; saves intrinsic board only if it
    differs from extrinsic board.

    Args:
        board: Extrinsic board geometry
        intrinsic_board: Intrinsic board geometry (may be same as board)
        output_dir: Directory to save images
    """
    # Generate and save extrinsic board image
    cv_board = board.get_opencv_board()
    board_img = cv_board.generateImage((800, 600), marginSize=50)
    cv2.imwrite(str(output_dir / "board_extrinsic.png"), board_img)

    # Save intrinsic board image only if it differs from extrinsic board
    if intrinsic_board is not board:
        cv_intr_board = intrinsic_board.get_opencv_board()
        intr_img = cv_intr_board.generateImage((800, 600), marginSize=50)
        cv2.imwrite(str(output_dir / "board_intrinsic.png"), intr_img)


def run_calibration(config_path: str | Path, verbose: bool = False) -> CalibrationResult:
    """
    Run complete calibration pipeline from config file.

    Loads configuration from YAML and delegates to run_calibration_from_config().

    Args:
        config_path: Path to config.yaml file
        verbose: If True, enable per-iteration progress output from optimizers

    Returns:
        Complete CalibrationResult

    Raises:
        FileNotFoundError: If config or video files not found
        CalibrationError: If any calibration stage fails
    """
    config = load_config(config_path)
    return run_calibration_from_config(config, verbose=verbose)


def run_calibration_from_config(config: CalibrationConfig, verbose: bool = False) -> CalibrationResult:
    """
    Run complete calibration pipeline from configuration object.

    Pipeline stages:
    1. Detect ChArUco in intrinsic (in-air) videos
    2. Run Stage 1: Intrinsic calibration
    3. Detect ChArUco in extrinsic (underwater) videos
    4. Split underwater detections into calibration/validation sets
    5. Run Stage 2: Extrinsic initialization
    6. Run Stage 3: Interface and pose optimization
    7. Optionally run Stage 4: Joint refinement
    8. Run validation on held-out data
    9. Generate and save diagnostics
    10. Save final calibration result

    Args:
        config: Complete calibration configuration
        verbose: If True, enable per-iteration progress output from optimizers

    Returns:
        CalibrationResult with all calibrations and diagnostics

    Raises:
        CalibrationError: If any stage fails
        InsufficientDataError: If not enough detections
        ConnectivityError: If pose graph is disconnected
    """
    board = BoardGeometry(config.board)

    # Intrinsic board: use separate board if provided, else fall back to extrinsic board
    intrinsic_board = BoardGeometry(config.intrinsic_board) if config.intrinsic_board else board

    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Save board reference images for visual verification
    _save_board_reference_images(board, intrinsic_board, config.output_dir)

    print("=" * 60)
    print("AquaCal Calibration Pipeline")
    print("=" * 60)

    # --- Stage 1: Intrinsic Calibration ---
    print("\n[Stage 1] Intrinsic calibration (in-air)...")
    intrinsics_results = calibrate_intrinsics_all(
        video_paths={k: str(v) for k, v in config.intrinsic_video_paths.items()},
        board=intrinsic_board,
        min_corners=config.min_corners_per_frame,
        frame_step=config.frame_step,
        rational_model_cameras=config.rational_model_cameras or None,
        fisheye_cameras=config.fisheye_cameras or None,
        progress_callback=lambda name, cur, total: print(f"  Calibrating {name} ({cur}/{total})..."),
    )
    # Extract just intrinsics from (intrinsics, error) tuples
    intrinsics = {name: result[0] for name, result in intrinsics_results.items()}
    for name, (_, rms) in intrinsics_results.items():
        print(f"  {name}: RMS {rms:.3f} px")
    print(f"  Calibrated {len(intrinsics)} cameras")

    # Validate intrinsics
    from aquacal.calibration.intrinsics import validate_intrinsics
    for name, (intr, _) in intrinsics_results.items():
        warnings = validate_intrinsics(intr, camera_name=name)
        for w in warnings:
            print(f"  WARNING: {w}")

    # --- Detect in underwater videos ---
    print("\n[Detection] Detecting ChArUco in underwater videos...")

    def _detection_progress(current: int, total: int) -> None:
        """Print detection progress at ~10% intervals."""
        if total > 0 and (current % max(1, total // 10) == 0 or current == total):
            print(f"  Frame {current}/{total} ({100 * current // total}%)")

    all_detections = detect_all_frames(
        video_paths={k: str(v) for k, v in config.extrinsic_video_paths.items()},
        board=board,
        intrinsics={k: (v.K, v.dist_coeffs) for k, v in intrinsics.items()},
        min_corners=config.min_corners_per_frame,
        frame_step=config.frame_step,
        progress_callback=_detection_progress,
    )
    usable_frames = all_detections.get_frames_with_min_cameras(
        config.min_cameras_per_frame
    )
    print(f"  Found {len(usable_frames)} usable frames")

    # --- Split calibration/validation ---
    print(f"\n[Split] Holdout fraction: {config.holdout_fraction}")
    cal_detections, val_detections = split_detections(
        all_detections, config.holdout_fraction
    )
    print(f"  Calibration frames: {len(cal_detections.frames)}")
    print(f"  Validation frames: {len(val_detections.frames)}")

    # --- Filter to primary cameras for Stages 2-3 ---
    primary_camera_set = set(config.camera_names)
    primary_intrinsics = {
        k: v for k, v in intrinsics.items() if k in primary_camera_set
    }

    # Filter detection frames to primary cameras only
    primary_cal_frames = {}
    for frame_idx, frame_det in cal_detections.frames.items():
        primary_dets = {
            k: v for k, v in frame_det.detections.items()
            if k in primary_camera_set
        }
        if primary_dets:
            primary_cal_frames[frame_idx] = FrameDetections(
                frame_idx=frame_idx, detections=primary_dets
            )
    primary_cal_detections = DetectionResult(
        frames=primary_cal_frames,
        camera_names=config.camera_names,
        total_frames=cal_detections.total_frames,
    )

    # --- Stage 2: Extrinsic Initialization ---
    print("\n[Stage 2] Extrinsic initialization...")
    pose_graph = build_pose_graph(primary_cal_detections, config.min_cameras_per_frame)
    reference_camera = config.camera_names[0]
    interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    extrinsics = estimate_extrinsics(
        pose_graph, primary_intrinsics, board, reference_camera,
        interface_distances=config.initial_interface_distances,
        interface_normal=interface_normal,
        n_air=config.n_air,
        n_water=config.n_water,
        progress_callback=lambda cam, cur, total: print(
            f"  Averaging poses..." if cam == "_averaging" else f"  Located {cam} ({cur}/{total})"
        ),
    )
    print(f"  Initialized {len(extrinsics)} camera poses")

    # Build initial CalibrationResult for saving and visualization
    initial_interface_dists = {}
    for cam_name in extrinsics:
        if config.initial_interface_distances is not None:
            initial_interface_dists[cam_name] = config.initial_interface_distances.get(cam_name, 0.15)
        else:
            initial_interface_dists[cam_name] = 0.15

    initial_result = _build_calibration_result(
        intrinsics=primary_intrinsics,
        extrinsics=extrinsics,
        interface_distances=initial_interface_dists,
        board_config=config.board,
        interface_params=InterfaceParams(
            normal=interface_normal,
            n_air=config.n_air,
            n_water=config.n_water,
        ),
        diagnostics=DiagnosticsData(
            reprojection_error_rms=0.0,
            reprojection_error_per_camera={},
            validation_3d_error_mean=0.0,
            validation_3d_error_std=0.0,
        ),
        metadata=CalibrationMetadata(
            calibration_date=datetime.now().isoformat(),
            software_version=importlib.metadata.version("aquacal"),
            config_hash=_compute_config_hash(config),
            num_frames_used=0,
            num_frames_holdout=0,
        ),
    )

    # Save pre-optimization calibration
    save_calibration(initial_result, config.output_dir / "calibration_initial.json")
    print(f"  Saved calibration_initial.json")

    # Save initial camera rig visualization (pre-optimization)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from aquacal.validation.diagnostics import plot_camera_rig

        fig = plot_camera_rig(
            initial_result,
            title="Stage 2: Initial Camera Positions (pre-optimization)",
        )
        fig.savefig(
            str(config.output_dir / "camera_rig_initial.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        print(f"  Saved camera_rig_initial.png")
    except Exception as e:
        print(f"  Warning: Could not save camera_rig_initial.png: {e}")

    # --- Subsample for optimization if configured ---
    optim_detections = primary_cal_detections
    if config.max_calibration_frames is not None and len(primary_cal_detections.frames) > config.max_calibration_frames:
        optim_detections = _subsample_detections(primary_cal_detections, config.max_calibration_frames)
        print(f"\n[Frame Selection] Subsampled {len(primary_cal_detections.frames)} -> {len(optim_detections.frames)} frames for optimization")

    # --- Stage 3: Interface Optimization ---
    print("\n[Stage 3] Interface and pose optimization...")

    t0 = time.perf_counter()
    stage3_extrinsics, stage3_distances, stage3_poses, stage3_rms = optimize_interface(
        detections=optim_detections,
        intrinsics=primary_intrinsics,
        initial_extrinsics=extrinsics,
        board=board,
        reference_camera=reference_camera,
        initial_interface_distances=config.initial_interface_distances,
        interface_normal=interface_normal,
        n_air=config.n_air,
        n_water=config.n_water,
        loss=config.robust_loss,
        loss_scale=config.loss_scale,
        min_corners=config.min_corners_per_frame,
        verbose=2 if verbose else 0,
        normal_fixed=config.interface_normal_fixed,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Stage 3 RMS: {stage3_rms:.3f} pixels ({elapsed:.1f}s)")

    if not config.interface_normal_fixed:
        ref_R = stage3_extrinsics[reference_camera].R
        ref_rvec = matrix_to_rvec(ref_R)
        tilt_deg = np.degrees(np.linalg.norm(ref_rvec[:2]))
        print(f"  Estimated reference camera tilt: {tilt_deg:.2f} degrees")

    # Water surface and camera heights
    # Get water_z from any camera (all have same value after optimization)
    water_z = list(stage3_distances.values())[0]
    print(f"  Water surface Z: {water_z:.4f} m")
    print("  Camera heights above water (h_c):")

    heights = []
    for cam_name in sorted(stage3_extrinsics.keys()):
        C = stage3_extrinsics[cam_name].C
        cam_z = C[2]
        h_c = water_z - cam_z  # camera-to-water vertical distance
        heights.append(h_c)
        print(f"    {cam_name}: cam_z={cam_z:.4f}  h_c={h_c:.4f}")

    heights = np.array(heights)
    print(f"  Camera height spread: {np.ptp(heights):.4f} m")

    # --- Stage 4: Optional Joint Refinement ---
    refine_intrinsics = config.refine_intrinsics

    if refine_intrinsics:
        print("\n[Stage 4] Joint refinement with intrinsics...")
        stage3_result = (stage3_extrinsics, stage3_distances, stage3_poses, stage3_rms)
        t0 = time.perf_counter()
        (
            final_extrinsics,
            final_distances,
            final_poses,
            final_intrinsics,
            final_rms,
        ) = joint_refinement(
            stage3_result=stage3_result,
            detections=optim_detections,
            intrinsics=primary_intrinsics,
            board=board,
            reference_camera=reference_camera,
            refine_intrinsics=True,
            interface_normal=interface_normal,
            n_air=config.n_air,
            n_water=config.n_water,
            loss=config.robust_loss,
            loss_scale=config.loss_scale,
            verbose=2 if verbose else 0,
            normal_fixed=config.interface_normal_fixed,
        )
        elapsed = time.perf_counter() - t0
        print(f"  Stage 4 RMS: {final_rms:.3f} pixels ({elapsed:.1f}s)")

        # Water surface and camera heights after refinement
        water_z_final = list(final_distances.values())[0]
        print(f"  Water surface Z (after refinement): {water_z_final:.4f} m")
        print("  Camera heights above water (h_c):")

        heights_final = []
        for cam_name in sorted(final_extrinsics.keys()):
            C = final_extrinsics[cam_name].C
            cam_z = C[2]
            h_c = water_z_final - cam_z
            heights_final.append(h_c)
            print(f"    {cam_name}: cam_z={cam_z:.4f}  h_c={h_c:.4f}")

        heights_final = np.array(heights_final)
        print(f"  Camera height spread: {np.ptp(heights_final):.4f} m")
    else:
        print("\n[Stage 4] Skipped (refine_intrinsics=False)")
        final_extrinsics = stage3_extrinsics
        final_distances = stage3_distances
        final_poses = stage3_poses
        final_intrinsics = primary_intrinsics
        final_rms = stage3_rms

    # Convert poses list to dict
    board_poses_dict = {bp.frame_idx: bp for bp in final_poses}

    # --- Stage 3b/4b: Register Auxiliary Cameras ---
    aux_extrinsics = {}
    aux_distances = {}
    if config.auxiliary_cameras:
        stage_label = "Stage 4b" if config.refine_auxiliary_intrinsics else "Stage 3b"
        print(f"\n[{stage_label}] Registering {len(config.auxiliary_cameras)} auxiliary camera(s)...")

        # Derive water_z from Stage 3 output (reference camera has C_z = 0)
        water_z = float(final_distances[reference_camera])

        for aux_cam in config.auxiliary_cameras:
            # Count observations
            n_frames = 0
            n_corners = 0
            for frame_idx, frame_det in all_detections.frames.items():
                if aux_cam in frame_det.detections and frame_idx in board_poses_dict:
                    n_frames += 1
                    n_corners += frame_det.detections[aux_cam].num_corners

            print(f"  {aux_cam}: {n_frames} frames, {n_corners} corners")

            try:
                result = register_auxiliary_camera(
                    camera_name=aux_cam,
                    intrinsics=intrinsics[aux_cam],
                    detections=all_detections,
                    board_poses=board_poses_dict,
                    board=board,
                    water_z=water_z,
                    interface_normal=interface_normal,
                    n_air=config.n_air,
                    n_water=config.n_water,
                    refine_intrinsics=config.refine_auxiliary_intrinsics,
                    verbose=2 if verbose else 0,
                )

                # Handle variable-length return
                if config.refine_auxiliary_intrinsics:
                    aux_ext, aux_dist, aux_rms, aux_intr = result
                    intrinsics[aux_cam] = aux_intr
                    print(f"  {aux_cam}: RMS {aux_rms:.2f} px, interface_d={aux_dist:.4f}m (intrinsics refined)")
                else:
                    aux_ext, aux_dist, aux_rms = result
                    print(f"  {aux_cam}: RMS {aux_rms:.2f} px, interface_d={aux_dist:.4f}m")

                aux_extrinsics[aux_cam] = aux_ext
                aux_distances[aux_cam] = aux_dist
            except Exception as e:
                print(f"  {aux_cam}: FAILED - {e}")

        # Merge auxiliary cameras into working dicts so validation includes them
        if aux_extrinsics:
            final_extrinsics.update(aux_extrinsics)
            final_distances.update(aux_distances)
            final_intrinsics.update({cam: intrinsics[cam] for cam in aux_extrinsics})

    # Estimate board poses for validation frames
    print("\n[Validation] Estimating board poses for held-out frames...")
    val_initial_poses = _compute_initial_board_poses(
        val_detections,
        final_intrinsics,
        final_extrinsics,
        board,
        min_corners=config.min_corners_per_frame,
        n_water=config.n_water,
    )

    val_refined_poses = _estimate_validation_poses(
        val_detections,
        val_initial_poses,
        final_intrinsics,
        final_extrinsics,
        final_distances,
        board,
        interface_normal,
        config.n_air,
        config.n_water,
    )
    board_poses_dict.update(val_refined_poses)
    print(f"  Estimated {len(val_refined_poses)} validation frame poses")

    # --- Validation ---
    print("\n[Validation] Computing errors on held-out data...")

    # Build temporary CalibrationResult for validation functions
    interface_params = InterfaceParams(
        normal=interface_normal,
        n_air=config.n_air,
        n_water=config.n_water,
    )

    # Determine primary and auxiliary cameras
    aux_cam_names = set(config.auxiliary_cameras) if config.auxiliary_cameras else set()
    primary_cam_names = set(final_intrinsics.keys()) - aux_cam_names

    # Build full result with all cameras (for board pose estimation and plots)
    temp_result = _build_calibration_result(
        intrinsics=final_intrinsics,
        extrinsics=final_extrinsics,
        interface_distances=final_distances,
        board_config=config.board,
        interface_params=interface_params,
        diagnostics=DiagnosticsData(
            reprojection_error_rms=0.0,
            reprojection_error_per_camera={},
            validation_3d_error_mean=0.0,
            validation_3d_error_std=0.0,
        ),
        metadata=CalibrationMetadata(
            calibration_date="",
            software_version="",
            config_hash="",
            num_frames_used=0,
            num_frames_holdout=0,
        ),
        auxiliary_cameras=aux_cam_names,
    )

    # --- Primary camera validation ---
    primary_result = _filter_cameras(temp_result, primary_cam_names)

    # Reprojection errors on validation set (primary cameras only)
    primary_reproj = compute_reprojection_errors(
        primary_result, val_detections, board_poses_dict
    )

    # 3D reconstruction errors (primary cameras only)
    primary_3d = compute_3d_distance_errors(
        primary_result, val_detections, board, include_spatial=True
    )

    # Save spatial measurements if available
    if primary_3d.spatial is not None and len(primary_3d.spatial.positions) > 0:
        from aquacal.validation.reconstruction import save_spatial_measurements

        spatial_csv_path = config.output_dir / "spatial_measurements.csv"
        save_spatial_measurements(primary_3d.spatial, spatial_csv_path)

    # Print primary camera metrics
    if np.isnan(primary_reproj.rms):
        print("  Primary cameras:")
        print("    Reprojection RMS: N/A (no valid observations)")
    else:
        print("  Primary cameras:")
        print(f"    Reprojection RMS: {primary_reproj.rms:.3f} pixels")

    if np.isnan(primary_3d.mean):
        print("    3D distance error: N/A (no valid comparisons)")
    else:
        print(
            f"    3D distance error: MAE {primary_3d.mean*1000:.2f} mm, "
            f"RMSE {primary_3d.rmse*1000:.2f} mm "
            f"({primary_3d.percent_error:.1f}% of square size)"
        )
        if abs(primary_3d.signed_mean) > 0.0005:  # > 0.5mm bias
            sign = "+" if primary_3d.signed_mean > 0 else ""
            bias_type = "overestimate" if primary_3d.signed_mean > 0 else "underestimate"
            print(
                f"    Scale bias: {sign}{primary_3d.signed_mean*1000:.2f} mm ({bias_type})"
            )

    # --- Auxiliary camera validation (if any) ---
    aux_reproj = None
    if aux_cam_names:
        aux_result = _filter_cameras(temp_result, aux_cam_names)
        aux_reproj = compute_reprojection_errors(
            aux_result, val_detections, board_poses_dict
        )

        print("  Auxiliary cameras:")
        for cam_name in sorted(aux_cam_names):
            if cam_name in aux_reproj.per_camera:
                rms = aux_reproj.per_camera[cam_name]
                print(f"    {cam_name}: RMS {rms:.3f} pixels")
            else:
                print(f"    {cam_name}: RMS N/A (no valid observations)")

    # Store primary metrics for later use
    reproj_errors = primary_reproj
    reconstruction_errors = primary_3d

    # --- Generate Diagnostics ---
    print("\n[Diagnostics] Generating report...")
    diagnostic_report = generate_diagnostic_report(
        calibration=primary_result,  # Use primary-only for summary stats
        detections=val_detections,
        board_poses=board_poses_dict,
        reprojection_errors=reproj_errors,
        reconstruction_errors=reconstruction_errors,
        board=board,
        auxiliary_reprojection=aux_reproj,
    )

    # Save diagnostics (uses full temp_result for plots, but report has primary-only stats)
    save_diagnostic_report(
        diagnostic_report,
        temp_result,  # Full result for plots
        val_detections,
        config.output_dir,
        save_images=True,
        auxiliary_reprojection=aux_reproj,
    )
    print(f"  Saved diagnostics to {config.output_dir}")

    # --- Build Final Result ---
    diagnostics = DiagnosticsData(
        reprojection_error_rms=reproj_errors.rms,
        reprojection_error_per_camera=reproj_errors.per_camera,
        validation_3d_error_mean=reconstruction_errors.mean,
        validation_3d_error_std=reconstruction_errors.std,
        per_corner_residuals=(
            reproj_errors.residuals if config.save_detailed_residuals else None
        ),
        per_frame_errors=(
            reproj_errors.per_frame if config.save_detailed_residuals else None
        ),
    )

    metadata = CalibrationMetadata(
        calibration_date=datetime.now().isoformat(),
        software_version=importlib.metadata.version("aquacal"),
        config_hash=_compute_config_hash(config),
        num_frames_used=len(optim_detections.frames),
        num_frames_holdout=len(val_detections.frames),
    )

    result = _build_calibration_result(
        intrinsics=final_intrinsics,
        extrinsics=final_extrinsics,
        interface_distances=final_distances,
        board_config=config.board,
        interface_params=interface_params,
        diagnostics=diagnostics,
        metadata=metadata,
        auxiliary_cameras=set(config.auxiliary_cameras),
    )

    # --- Save Calibration ---
    print("\n[Save] Saving calibration result...")
    output_path = config.output_dir / "calibration.json"
    save_calibration(result, output_path)
    print(f"  Saved to {output_path}")

    print("\n" + "=" * 60)
    print("Calibration complete!")
    print("  Primary cameras:")
    if np.isnan(reproj_errors.rms):
        print("    Reprojection RMS: N/A")
    else:
        print(f"    Reprojection RMS: {reproj_errors.rms:.3f} pixels")
    if np.isnan(reconstruction_errors.mean):
        print("    3D error: N/A")
    else:
        print(
            f"    3D error: MAE {reconstruction_errors.mean*1000:.2f} mm, "
            f"RMSE {reconstruction_errors.rmse*1000:.2f} mm "
            f"({reconstruction_errors.percent_error:.1f}%)"
        )
    if aux_cam_names:
        print("  Auxiliary cameras:")
        for cam_name in sorted(aux_cam_names):
            if aux_reproj and cam_name in aux_reproj.per_camera:
                rms = aux_reproj.per_camera[cam_name]
                print(f"    {cam_name}: RMS {rms:.3f} pixels")
    print("=" * 60)

    return result


def _estimate_validation_poses(
    detections: DetectionResult,
    initial_poses: dict[int, BoardPose],
    intrinsics: dict[str, CameraIntrinsics],
    extrinsics: dict[str, CameraExtrinsics],
    interface_distances: dict[str, float],
    board: BoardGeometry,
    interface_normal: np.ndarray,
    n_air: float,
    n_water: float,
) -> dict[int, BoardPose]:
    """Refine board poses for validation frames via per-frame optimization.

    For each frame, minimizes refractive reprojection error over the 6 pose
    parameters (rvec, tvec) while holding all camera parameters fixed.

    Args:
        detections: Detection results for validation frames
        initial_poses: PnP-initialized board poses
        intrinsics: Per-camera intrinsics
        extrinsics: Per-camera extrinsics
        interface_distances: Per-camera interface distances
        board: Board geometry
        interface_normal: Interface normal vector
        n_air: Refractive index of air
        n_water: Refractive index of water

    Returns:
        Dict mapping frame_idx to refined BoardPose
    """
    from scipy.optimize import least_squares

    from aquacal.core.camera import create_camera
    from aquacal.core.interface_model import Interface
    from aquacal.core.refractive_geometry import refractive_project

    refined_poses = {}

    for frame_idx, initial_pose in initial_poses.items():
        if frame_idx not in detections.frames:
            continue
        frame_det = detections.frames[frame_idx]

        # Build cameras and interface objects
        cameras = {}
        for cam_name in frame_det.detections:
            if cam_name not in intrinsics:
                continue
            cameras[cam_name] = create_camera(
                cam_name, intrinsics[cam_name], extrinsics[cam_name]
            )

        interface = Interface(
            normal=interface_normal,
            camera_distances=interface_distances,
            n_air=n_air,
            n_water=n_water,
        )

        # Cost function: refractive reprojection residuals for this frame
        def frame_residuals(params):
            rvec = params[:3]
            tvec = params[3:]
            corners_3d = board.transform_corners(rvec, tvec)

            residuals = []
            for cam_name, det in frame_det.detections.items():
                if cam_name not in cameras:
                    continue
                camera = cameras[cam_name]
                for i, corner_id in enumerate(det.corner_ids):
                    pt_3d = corners_3d[int(corner_id)]
                    projected = refractive_project(
                        camera, interface, pt_3d
                    )
                    if projected is not None:
                        residuals.append(det.corners_2d[i, 0] - projected[0])
                        residuals.append(det.corners_2d[i, 1] - projected[1])
                    else:
                        residuals.append(100.0)
                        residuals.append(100.0)

            return residuals if residuals else [0.0, 0.0]

        x0 = np.concatenate([initial_pose.rvec, initial_pose.tvec])

        result = least_squares(
            frame_residuals, x0, method="lm", max_nfev=100
        )

        refined_poses[frame_idx] = BoardPose(
            frame_idx=frame_idx,
            rvec=result.x[:3],
            tvec=result.x[3:],
        )

    return refined_poses


def _filter_cameras(
    result: CalibrationResult,
    camera_names: set[str],
) -> CalibrationResult:
    """
    Create a new CalibrationResult containing only the specified cameras.

    Args:
        result: Original CalibrationResult
        camera_names: Set of camera names to include

    Returns:
        New CalibrationResult with filtered cameras
    """
    filtered_cameras = {
        name: calib
        for name, calib in result.cameras.items()
        if name in camera_names
    }

    return CalibrationResult(
        cameras=filtered_cameras,
        interface=result.interface,
        board=result.board,
        diagnostics=result.diagnostics,
        metadata=result.metadata,
    )


def _build_calibration_result(
    intrinsics: dict[str, CameraIntrinsics],
    extrinsics: dict[str, CameraExtrinsics],
    interface_distances: dict[str, float],
    board_config: BoardConfig,
    interface_params: InterfaceParams,
    diagnostics: DiagnosticsData,
    metadata: CalibrationMetadata,
    auxiliary_cameras: set[str] | None = None,
) -> CalibrationResult:
    """
    Assemble final CalibrationResult from components.

    Args:
        intrinsics: Per-camera intrinsic parameters
        extrinsics: Per-camera extrinsic parameters
        interface_distances: Per-camera interface distances
        board_config: Board configuration used
        interface_params: Interface parameters (normal, refractive indices)
        diagnostics: Validation diagnostics
        metadata: Calibration metadata
        auxiliary_cameras: Set of auxiliary camera names

    Returns:
        Complete CalibrationResult
    """
    cameras = {}
    for cam_name in intrinsics:
        cameras[cam_name] = CameraCalibration(
            name=cam_name,
            intrinsics=intrinsics[cam_name],
            extrinsics=extrinsics[cam_name],
            interface_distance=interface_distances[cam_name],
            is_auxiliary=cam_name in (auxiliary_cameras or set()),
        )

    return CalibrationResult(
        cameras=cameras,
        interface=interface_params,
        board=board_config,
        diagnostics=diagnostics,
        metadata=metadata,
    )


def _compute_config_hash(config: CalibrationConfig) -> str:
    """
    Compute hash of configuration for reproducibility tracking.

    Args:
        config: Calibration configuration

    Returns:
        Hex string hash of configuration
    """
    # Create deterministic string representation
    hash_input = (
        f"{config.board.squares_x},{config.board.squares_y},"
        f"{config.board.square_size},{config.board.marker_size},"
        f"{config.n_air},{config.n_water},"
        f"{config.robust_loss},{config.loss_scale},"
        f"{config.holdout_fraction}"
    )

    # Include intrinsic_board if provided
    if config.intrinsic_board is not None:
        hash_input += (
            f",intrinsic:{config.intrinsic_board.squares_x},"
            f"{config.intrinsic_board.squares_y},"
            f"{config.intrinsic_board.square_size},"
            f"{config.intrinsic_board.marker_size}"
        )

    # Include initial_interface_distances if provided
    if config.initial_interface_distances is not None:
        # Sort by camera name for deterministic hash
        sorted_distances = sorted(config.initial_interface_distances.items())
        distance_str = ",".join(f"{cam}:{dist}" for cam, dist in sorted_distances)
        hash_input += f",init_dist:{distance_str}"

    return hashlib.md5(hash_input.encode()).hexdigest()[:12]
