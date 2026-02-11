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
from aquacal.calibration._optim_common import (
    pack_params,
    unpack_params,
    build_jacobian_sparsity,
    build_bounds,
    compute_residuals,
    make_sparse_jacobian_func,
)
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
    use_fast_projection: bool = True,
    use_sparse_jacobian: bool = True,
    verbose: int = 0,
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
        use_fast_projection: Use fast Newton-based projection (default True).
            Only works with horizontal interface (normal = [0, 0, -1]).
        use_sparse_jacobian: Use sparse Jacobian structure (default True).
            Dramatically improves performance for large camera arrays.
        verbose: Verbosity level for scipy.optimize.least_squares (default 0).
            0 = silent, 1 = one-line per iteration, 2 = full per-iteration report.

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
    initial_params = pack_params(
        initial_extrinsics,
        initial_interface_distances,
        initial_board_poses,
        reference_camera,
        camera_order,
        frame_order,
    )

    # Build bounds
    lower, upper = build_bounds(camera_order, frame_order, reference_camera)

    # Reference extrinsics (fixed during optimization)
    reference_extrinsics = initial_extrinsics[reference_camera]

    # Build cost function args
    cost_args = (
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
        False,  # refine_intrinsics
        use_fast_projection,
    )

    # Build sparse Jacobian if enabled
    jac = "2-point"
    if use_sparse_jacobian:
        jac_sparsity = build_jacobian_sparsity(
            detections,
            reference_camera,
            camera_order,
            frame_order,
            min_corners,
        )
        jac = make_sparse_jacobian_func(
            compute_residuals, cost_args, jac_sparsity, (lower, upper),
        )

    # Run optimization
    result = least_squares(
        compute_residuals,
        x0=initial_params,
        args=cost_args,
        method="trf",
        loss=loss,
        f_scale=loss_scale,
        bounds=(lower, upper),
        jac=jac,
        verbose=verbose,
    )

    if result.status <= 0:
        raise ConvergenceError(f"Optimization failed: {result.message}")

    # Unpack optimized parameters
    opt_extrinsics, opt_distances, opt_board_poses, _ = unpack_params(
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
