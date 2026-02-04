"""Stage 3 refractive optimization for interface estimation.

This module implements joint optimization of camera extrinsics, per-camera
interface distances, and board poses using underwater ChArUco detections.
"""

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from aquacal.config.schema import (
    CameraIntrinsics,
    CameraExtrinsics,
    BoardPose,
    DetectionResult,
    InsufficientDataError,
    ConvergenceError,
    Vec3,
)
from aquacal.core.board import BoardGeometry
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_project
from aquacal.utils.transforms import (
    rvec_to_matrix,
    matrix_to_rvec,
    compose_poses,
    invert_pose,
)


def _compute_initial_board_poses(
    detections: DetectionResult,
    intrinsics: dict[str, CameraIntrinsics],
    extrinsics: dict[str, CameraExtrinsics],
    board: BoardGeometry,
    min_corners: int = 4,
    n_water: float = 1.333,
) -> dict[int, BoardPose]:
    """
    Compute initial board poses via solvePnP for each frame.

    For each frame, uses the camera with the most detected corners to estimate
    the board pose via cv2.solvePnP. The pose is transformed to world frame
    using that camera's extrinsics.

    Note: solvePnP assumes pinhole projection, but the detected corners were
    projected through a refractive interface. This causes an "apparent depth"
    effect where objects appear closer. We apply an approximate correction
    by scaling the Z component of the translation by n_water.

    Args:
        detections: Detection results
        intrinsics: Per-camera intrinsics
        extrinsics: Per-camera extrinsics (for transforming to world frame)
        board: Board geometry
        min_corners: Minimum corners required for PnP
        n_water: Refractive index of water (for depth correction)

    Returns:
        Dict mapping frame_idx to BoardPose (in world frame)
    """
    board_poses = {}

    for frame_idx, frame_det in detections.frames.items():
        # Find camera with most corners
        best_cam = None
        best_count = 0
        for cam_name, det in frame_det.detections.items():
            if det.num_corners >= min_corners and det.num_corners > best_count:
                best_cam = cam_name
                best_count = det.num_corners

        if best_cam is None:
            continue

        det = frame_det.detections[best_cam]

        # Get 3D object points
        object_points = board.get_corner_array(det.corner_ids).astype(np.float32)
        image_points = det.corners_2d.astype(np.float32)

        intr = intrinsics[best_cam]
        success, rvec_bc, tvec_bc = cv2.solvePnP(
            object_points,
            image_points,
            intr.K.astype(np.float64),
            intr.dist_coeffs.astype(np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            continue

        rvec_bc = rvec_bc.flatten()
        tvec_bc = tvec_bc.flatten()

        # Apply approximate depth correction for refraction
        # solvePnP gives "apparent" depth; real depth is approximately n_water times larger
        # This is a rough approximation but provides better initialization
        tvec_bc[2] *= n_water

        # Transform board pose from camera frame to world frame
        # board_in_world = camera_in_world @ board_in_camera
        ext = extrinsics[best_cam]
        R_cw, t_cw = invert_pose(ext.R, ext.t)  # camera in world
        R_bc = rvec_to_matrix(rvec_bc)
        R_bw, t_bw = compose_poses(R_cw, t_cw, R_bc, tvec_bc)
        rvec_bw = matrix_to_rvec(R_bw)

        board_poses[frame_idx] = BoardPose(
            frame_idx=frame_idx,
            rvec=rvec_bw,
            tvec=t_bw,
        )

    return board_poses


def _pack_params(
    extrinsics: dict[str, CameraExtrinsics],
    interface_distances: dict[str, float],
    board_poses: dict[int, BoardPose],
    reference_camera: str,
    camera_order: list[str],
    frame_order: list[int],
) -> NDArray[np.float64]:
    """
    Pack optimization parameters into a 1D array.

    Parameter layout:
    - For each non-reference camera (in camera_order, skipping reference):
        cam_rvec (3), cam_tvec (3)
    - For each camera (in camera_order, including reference):
        interface_distance (1)
    - For each frame (in frame_order):
        board_rvec (3), board_tvec (3)

    Total length: 6*(N_cams-1) + N_cams + 6*N_frames

    Args:
        extrinsics: Camera extrinsics dict
        interface_distances: Per-camera interface distances
        board_poses: Board poses dict (frame_idx -> BoardPose)
        reference_camera: Name of reference camera (skipped in extrinsics packing)
        camera_order: Ordered list of camera names
        frame_order: Ordered list of frame indices

    Returns:
        1D parameter vector
    """
    params = []

    # Pack camera extrinsics (skip reference camera)
    for cam_name in camera_order:
        if cam_name == reference_camera:
            continue
        ext = extrinsics[cam_name]
        rvec = matrix_to_rvec(ext.R)
        params.extend(rvec.tolist())
        params.extend(ext.t.tolist())

    # Pack interface distances (all cameras, including reference)
    for cam_name in camera_order:
        params.append(interface_distances[cam_name])

    # Pack board poses
    for frame_idx in frame_order:
        bp = board_poses[frame_idx]
        params.extend(bp.rvec.tolist())
        params.extend(bp.tvec.tolist())

    return np.array(params, dtype=np.float64)


def _unpack_params(
    params: NDArray[np.float64],
    reference_camera: str,
    reference_extrinsics: CameraExtrinsics,
    camera_order: list[str],
    frame_order: list[int],
) -> tuple[dict[str, CameraExtrinsics], dict[str, float], dict[int, BoardPose]]:
    """
    Unpack 1D parameter array into structured objects.

    Args:
        params: 1D parameter vector from optimizer
        reference_camera: Name of reference camera
        reference_extrinsics: Extrinsics for reference camera (fixed)
        camera_order: Ordered list of camera names
        frame_order: Ordered list of frame indices

    Returns:
        Tuple of (extrinsics_dict, distances_dict, board_poses_dict)
    """
    idx = 0
    n_cams = len(camera_order)

    # Unpack camera extrinsics (skip reference camera)
    extrinsics = {}
    for cam_name in camera_order:
        if cam_name == reference_camera:
            extrinsics[cam_name] = reference_extrinsics
        else:
            rvec = params[idx : idx + 3]
            tvec = params[idx + 3 : idx + 6]
            idx += 6
            R = rvec_to_matrix(rvec)
            extrinsics[cam_name] = CameraExtrinsics(R=R, t=tvec.copy())

    # Unpack interface distances (all cameras)
    interface_distances = {}
    for cam_name in camera_order:
        interface_distances[cam_name] = params[idx]
        idx += 1

    # Unpack board poses
    board_poses = {}
    for frame_idx in frame_order:
        rvec = params[idx : idx + 3]
        tvec = params[idx + 3 : idx + 6]
        idx += 6
        board_poses[frame_idx] = BoardPose(
            frame_idx=frame_idx,
            rvec=rvec.copy(),
            tvec=tvec.copy(),
        )

    return extrinsics, interface_distances, board_poses


def _cost_function(
    params: NDArray[np.float64],
    detections: DetectionResult,
    intrinsics: dict[str, CameraIntrinsics],
    board: BoardGeometry,
    reference_camera: str,
    reference_extrinsics: CameraExtrinsics,
    interface_normal: Vec3,
    n_air: float,
    n_water: float,
    camera_order: list[str],
    frame_order: list[int],
    min_corners: int,
) -> NDArray[np.float64]:
    """
    Compute reprojection residuals for all observations.

    For each frame, camera, and detected corner:
    1. Get corner 3D position from board pose
    2. Refractive project through interface to pixel
    3. Compute residual (predicted - detected) in x and y

    Args:
        params: Current parameter vector
        detections: Detection results
        intrinsics: Per-camera intrinsics (fixed)
        board: Board geometry
        reference_camera: Name of reference camera
        reference_extrinsics: Extrinsics for reference camera (fixed)
        interface_normal: Interface normal vector
        n_air: Refractive index of air
        n_water: Refractive index of water
        camera_order: Ordered list of camera names
        frame_order: Ordered list of frame indices
        min_corners: Minimum corners per detection

    Returns:
        1D array of residuals [r0_x, r0_y, r1_x, r1_y, ...] in pixels
    """
    # Unpack current parameters
    extrinsics, interface_distances, board_poses = _unpack_params(
        params, reference_camera, reference_extrinsics, camera_order, frame_order
    )

    residuals = []

    for frame_idx in frame_order:
        if frame_idx not in detections.frames:
            continue

        board_pose = board_poses[frame_idx]
        # Transform board corners to world frame
        corners_3d = board.transform_corners(board_pose.rvec, board_pose.tvec)

        for cam_name in camera_order:
            frame_det = detections.frames[frame_idx]
            if cam_name not in frame_det.detections:
                continue

            detection = frame_det.detections[cam_name]
            if detection.num_corners < min_corners:
                continue

            # Build Camera object
            camera = Camera(cam_name, intrinsics[cam_name], extrinsics[cam_name])

            # Build Interface object for this camera
            # Use base_height=0 and put actual distance in camera_offsets
            interface = Interface(
                normal=interface_normal,
                base_height=0.0,
                camera_offsets={cam_name: interface_distances[cam_name]},
                n_air=n_air,
                n_water=n_water,
            )

            for i, corner_id in enumerate(detection.corner_ids):
                point_3d = corners_3d[corner_id]
                detected_px = detection.corners_2d[i]

                projected = refractive_project(camera, interface, point_3d)
                if projected is None:
                    # Use large residual for failed projections
                    residuals.extend([100.0, 100.0])
                else:
                    residuals.extend(
                        [
                            projected[0] - detected_px[0],
                            projected[1] - detected_px[1],
                        ]
                    )

    return np.array(residuals, dtype=np.float64)


def optimize_interface(
    detections: DetectionResult,
    intrinsics: dict[str, CameraIntrinsics],
    initial_extrinsics: dict[str, CameraExtrinsics],
    board: BoardGeometry,
    reference_camera: str,
    initial_interface_distances: dict[str, float] | None = None,
    interface_normal: Vec3 | None = None,
    n_air: float = 1.0,
    n_water: float = 1.333,
    loss: str = "huber",
    loss_scale: float = 1.0,
    min_corners: int = 4,
) -> tuple[dict[str, CameraExtrinsics], dict[str, float], list[BoardPose], float]:
    """
    Jointly optimize camera extrinsics, interface distances, and board poses.

    This is Stage 3 of the calibration pipeline. It refines the initial estimates
    from Stage 2 by accounting for refraction at the air-water interface.

    Args:
        detections: Underwater ChArUco detections from detect_all_frames
        intrinsics: Per-camera intrinsic parameters (fixed during optimization)
        initial_extrinsics: Initial camera extrinsics from Stage 2
        board: ChArUco board geometry
        reference_camera: Camera name to fix at origin (extrinsics not optimized)
        initial_interface_distances: Optional initial distances per camera.
            If None, defaults to 0.15m for all cameras.
        interface_normal: Interface normal vector. If None, uses [0, 0, -1].
            Normal is fixed during optimization.
        n_air: Refractive index of air (default 1.0)
        n_water: Refractive index of water (default 1.333)
        loss: Robust loss function ("linear", "huber", "soft_l1", "cauchy")
        loss_scale: Scale parameter for robust loss in pixels
        min_corners: Minimum corners per detection to include in optimization

    Returns:
        Tuple of:
        - dict[str, CameraExtrinsics]: Optimized extrinsics for all cameras
        - dict[str, float]: Optimized interface distances per camera
        - list[BoardPose]: Optimized board poses for each frame used
        - float: Final RMS reprojection error in pixels

    Raises:
        InsufficientDataError: If no valid frames for optimization
        ConvergenceError: If optimization fails to converge
        ValueError: If reference_camera not in initial_extrinsics
    """
    # Validate reference camera
    if reference_camera not in initial_extrinsics:
        raise ValueError(
            f"reference_camera '{reference_camera}' not in initial_extrinsics. "
            f"Available cameras: {list(initial_extrinsics.keys())}"
        )

    # Set default interface normal
    if interface_normal is None:
        interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        interface_normal = np.asarray(interface_normal, dtype=np.float64)

    # Set default interface distances
    if initial_interface_distances is None:
        initial_interface_distances = {
            cam: 0.15 for cam in initial_extrinsics.keys()
        }

    # Create ordered lists for consistent parameter packing
    camera_order = sorted(initial_extrinsics.keys())

    # Compute initial board poses
    initial_board_poses = _compute_initial_board_poses(
        detections, intrinsics, initial_extrinsics, board, min_corners, n_water
    )

    # Filter to frames with valid board poses
    frame_order = sorted(initial_board_poses.keys())

    # Check for sufficient data
    if len(frame_order) == 0:
        raise InsufficientDataError(
            "No valid frames for optimization. "
            f"Need at least {min_corners} corners per detection."
        )

    # Pack initial parameters
    initial_params = _pack_params(
        initial_extrinsics,
        initial_interface_distances,
        initial_board_poses,
        reference_camera,
        camera_order,
        frame_order,
    )

    # Build bounds arrays
    n_cams = len(camera_order)
    n_frames = len(frame_order)
    n_extrinsic_params = 6 * (n_cams - 1)  # 6 params per non-reference camera
    n_distance_params = n_cams
    n_pose_params = 6 * n_frames
    total_params = n_extrinsic_params + n_distance_params + n_pose_params

    lower = np.full(total_params, -np.inf)
    upper = np.full(total_params, np.inf)

    # Interface distances: [0.01, 2.0] meters
    dist_start = n_extrinsic_params
    dist_end = dist_start + n_distance_params
    lower[dist_start:dist_end] = 0.01
    upper[dist_start:dist_end] = 2.0

    # Reference extrinsics (fixed during optimization)
    reference_extrinsics = initial_extrinsics[reference_camera]

    # Run optimization
    result = least_squares(
        _cost_function,
        x0=initial_params,
        args=(
            detections,
            intrinsics,
            board,
            reference_camera,
            reference_extrinsics,
            interface_normal,
            n_air,
            n_water,
            camera_order,
            frame_order,
            min_corners,
        ),
        method="trf",  # Trust Region Reflective (supports bounds)
        loss=loss,
        f_scale=loss_scale,
        bounds=(lower, upper),
        verbose=0,
    )

    if result.status <= 0:
        raise ConvergenceError(f"Optimization failed: {result.message}")

    # Unpack optimized parameters
    opt_extrinsics, opt_distances, opt_board_poses = _unpack_params(
        result.x,
        reference_camera,
        reference_extrinsics,
        camera_order,
        frame_order,
    )

    # Compute final RMS error
    rms_error = np.sqrt(np.mean(result.fun**2))

    # Convert board_poses dict to list
    board_poses_list = [opt_board_poses[frame_idx] for frame_idx in frame_order]

    return opt_extrinsics, opt_distances, board_poses_list, rms_error
