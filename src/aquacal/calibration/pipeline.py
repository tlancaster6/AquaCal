"""End-to-end calibration pipeline orchestration."""

from __future__ import annotations

import hashlib
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

    # Interface settings
    interface = data.get("interface", {})
    n_air = interface.get("n_air", 1.0)
    n_water = interface.get("n_water", 1.333)
    normal_fixed = interface.get("normal_fixed", True)

    # Parse initial_interface_distances (optional)
    initial_interface_distances = None
    if "initial_distances" in interface:
        raw_distances = interface["initial_distances"]

        # Handle scalar format (apply to all cameras)
        if isinstance(raw_distances, (int, float)):
            if raw_distances <= 0:
                raise ValueError(
                    f"initial_distances must be positive, got {raw_distances}"
                )
            initial_interface_distances = {
                cam: float(raw_distances) for cam in data["cameras"]
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

            # Warn about extra cameras (not in cameras list)
            extra_cameras = set(raw_distances.keys()) - set(data["cameras"])
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

    # Detection settings
    det = data.get("detection", {})
    min_corners = det.get("min_corners", 8)
    min_cameras = det.get("min_cameras", 2)
    frame_step = det.get("frame_step", 1)

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
        save_detailed_residuals=save_detailed,
        initial_interface_distances=initial_interface_distances,
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
        progress_callback=lambda name, cur, total: print(f"  Calibrating {name} ({cur}/{total})..."),
    )
    # Extract just intrinsics from (intrinsics, error) tuples
    intrinsics = {name: result[0] for name, result in intrinsics_results.items()}
    print(f"  Calibrated {len(intrinsics)} cameras")

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

    # --- Stage 2: Extrinsic Initialization ---
    print("\n[Stage 2] Extrinsic initialization...")
    pose_graph = build_pose_graph(cal_detections, config.min_cameras_per_frame)
    reference_camera = config.camera_names[0]
    extrinsics = estimate_extrinsics(pose_graph, intrinsics, board, reference_camera)
    print(f"  Initialized {len(extrinsics)} camera poses")

    # --- Stage 3: Interface Optimization ---
    print("\n[Stage 3] Interface and pose optimization...")
    interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    t0 = time.perf_counter()
    stage3_extrinsics, stage3_distances, stage3_poses, stage3_rms = optimize_interface(
        detections=cal_detections,
        intrinsics=intrinsics,
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
    )
    elapsed = time.perf_counter() - t0
    print(f"  Stage 3 RMS: {stage3_rms:.3f} pixels ({elapsed:.1f}s)")

    # --- Stage 4: Optional Joint Refinement ---
    # Check if config has refine_intrinsics attribute (may not exist in base schema)
    refine_intrinsics = getattr(config, "refine_intrinsics", False)

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
            detections=cal_detections,
            intrinsics=intrinsics,
            board=board,
            reference_camera=reference_camera,
            refine_intrinsics=True,
            interface_normal=interface_normal,
            n_air=config.n_air,
            n_water=config.n_water,
            loss=config.robust_loss,
            loss_scale=config.loss_scale,
            verbose=2 if verbose else 0,
        )
        elapsed = time.perf_counter() - t0
        print(f"  Stage 4 RMS: {final_rms:.3f} pixels ({elapsed:.1f}s)")
    else:
        print("\n[Stage 4] Skipped (refine_intrinsics=False)")
        final_extrinsics = stage3_extrinsics
        final_distances = stage3_distances
        final_poses = stage3_poses
        final_intrinsics = intrinsics
        final_rms = stage3_rms

    # Convert poses list to dict
    board_poses_dict = {bp.frame_idx: bp for bp in final_poses}

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
    )

    # Reprojection errors on validation set
    reproj_errors = compute_reprojection_errors(
        temp_result, val_detections, board_poses_dict
    )
    if np.isnan(reproj_errors.rms):
        print("  WARNING: Validation reprojection RMS: N/A (no valid observations)")
    else:
        print(f"  Validation reprojection RMS: {reproj_errors.rms:.3f} pixels")

    # 3D reconstruction errors
    reconstruction_errors = compute_3d_distance_errors(
        temp_result, val_detections, board
    )
    if np.isnan(reconstruction_errors.mean):
        print("  WARNING: 3D reconstruction mean error: N/A (no valid comparisons)")
    else:
        print(f"  3D reconstruction mean error: {reconstruction_errors.mean*1000:.2f} mm")

    # --- Generate Diagnostics ---
    print("\n[Diagnostics] Generating report...")
    diagnostic_report = generate_diagnostic_report(
        calibration=temp_result,
        detections=val_detections,
        board_poses=board_poses_dict,
        reprojection_errors=reproj_errors,
        reconstruction_errors=reconstruction_errors,
        board=board,
    )

    # Save diagnostics
    save_diagnostic_report(
        diagnostic_report,
        temp_result,
        val_detections,
        config.output_dir,
        save_images=True,
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
        software_version="0.1.0",  # TODO: get from package
        config_hash=_compute_config_hash(config),
        num_frames_used=len(cal_detections.frames),
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
    )

    # --- Save Calibration ---
    print("\n[Save] Saving calibration result...")
    output_path = config.output_dir / "calibration.json"
    save_calibration(result, output_path)
    print(f"  Saved to {output_path}")

    print("\n" + "=" * 60)
    print("Calibration complete!")
    if np.isnan(reproj_errors.rms):
        print("  Reprojection RMS: N/A")
    else:
        print(f"  Reprojection RMS: {reproj_errors.rms:.3f} pixels")
    if np.isnan(reconstruction_errors.mean):
        print("  3D error: N/A")
    else:
        print(
            f"  3D error: {reconstruction_errors.mean*1000:.2f} "
            f"± {reconstruction_errors.std*1000:.2f} mm"
        )
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

    from aquacal.core.camera import Camera
    from aquacal.core.interface_model import Interface
    from aquacal.core.refractive_geometry import refractive_project_fast

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
            cameras[cam_name] = Camera(
                cam_name, intrinsics[cam_name], extrinsics[cam_name]
            )

        interface = Interface(
            normal=interface_normal,
            base_height=0.0,
            camera_offsets=interface_distances,
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
                    projected = refractive_project_fast(
                        camera, interface, pt_3d
                    )
                    if projected is not None:
                        residuals.append(det.corners_2d[i, 0] - projected[0])
                        residuals.append(det.corners_2d[i, 1] - projected[1])

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


def _build_calibration_result(
    intrinsics: dict[str, CameraIntrinsics],
    extrinsics: dict[str, CameraExtrinsics],
    interface_distances: dict[str, float],
    board_config: BoardConfig,
    interface_params: InterfaceParams,
    diagnostics: DiagnosticsData,
    metadata: CalibrationMetadata,
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
