"""Stage 3 refractive optimization for interface estimation.

This module implements joint optimization of camera extrinsics, per-camera
interface distances, and board poses using underwater ChArUco detections.
"""

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
from aquacal.calibration.extrinsics import refractive_solve_pnp
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
    interface_distances: dict[str, float] | None = None,
    interface_normal: NDArray[np.float64] | None = None,
    n_air: float = 1.0,
    n_water: float = 1.333,
) -> dict[int, BoardPose]:
    """
    Compute initial board poses via refractive PnP for each frame.

    For each frame, uses the camera with the most detected corners to estimate
    the board pose using refractive_solve_pnp, which accounts for refraction
    at the air-water interface.

    Args:
        detections: Detection results
        intrinsics: Per-camera intrinsics
        extrinsics: Per-camera extrinsics (for transforming to world frame)
        board: Board geometry
        min_corners: Minimum corners required for PnP
        interface_distances: Per-camera interface distances. If None,
            defaults to 0.15m for all cameras.
        interface_normal: Interface normal vector. If None, uses [0, 0, -1].
        n_air: Refractive index of air (default 1.0)
        n_water: Refractive index of water (default 1.333)

    Returns:
        Dict mapping frame_idx to BoardPose (in world frame)
    """
    if interface_distances is None:
        interface_distances = {cam: 0.15 for cam in intrinsics.keys()}

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

        result = refractive_solve_pnp(
            intrinsics[best_cam], det.corners_2d, det.corner_ids, board,
            interface_distances.get(best_cam, 0.15),
            interface_normal, n_air, n_water,
        )
        if result is None:
            continue

        rvec_bc, tvec_bc = result

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
    water_z_weight: float = 0.0,
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
        water_z_weight: Weight for water surface consistency regularization.
            0.0 disables regularization (default).

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
        detections, intrinsics, initial_extrinsics, board, min_corners,
        interface_distances=initial_interface_distances,
        interface_normal=interface_normal,
        n_air=n_air, n_water=n_water,
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
        water_z_weight,
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
            water_z_weight=water_z_weight,
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


def register_auxiliary_camera(
    camera_name: str,
    intrinsics: CameraIntrinsics,
    detections: DetectionResult,
    board_poses: dict[int, BoardPose],
    board: BoardGeometry,
    initial_interface_distance: float = 0.15,
    interface_normal: Vec3 | None = None,
    n_air: float = 1.0,
    n_water: float = 1.333,
    min_corners: int = 4,
    target_water_z: float | None = None,
    water_z_weight: float = 10.0,
    verbose: int = 0,
) -> tuple[CameraExtrinsics, float, float]:
    """Register a single auxiliary camera against fixed board poses.

    Estimates the camera's extrinsics and interface distance by minimizing
    refractive reprojection error against known board poses from Stage 3.
    The board poses are treated as fixed ground truth.

    Args:
        camera_name: Name of the auxiliary camera
        intrinsics: Camera intrinsic parameters
        detections: Full detection results (must contain this camera's detections)
        board_poses: Fixed board poses from Stage 3 (frame_idx -> BoardPose)
        board: Board geometry
        initial_interface_distance: Starting interface distance estimate
        interface_normal: Interface normal (default [0, 0, -1])
        n_air: Refractive index of air
        n_water: Refractive index of water
        min_corners: Minimum corners per detection
        target_water_z: Target water surface Z from primary calibration.
            When provided, adds a soft regularization residual anchoring
            this camera's water_z to the primary cameras' mean.
        water_z_weight: Weight for the water_z regularization residual.
            Only used when target_water_z is not None.
        verbose: Verbosity level

    Returns:
        Tuple of (extrinsics, interface_distance, rms_error)

    Raises:
        InsufficientDataError: If no usable frames found
    """
    from aquacal.core.camera import Camera
    from aquacal.core.interface_model import Interface
    from aquacal.core.refractive_geometry import refractive_project_fast

    if interface_normal is None:
        interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        interface_normal = np.asarray(interface_normal, dtype=np.float64)

    # --- Step 1: Collect observations ---
    # Find frames where this camera has detections AND a board pose exists
    obs_frames = []  # list of (frame_idx, corner_ids, corners_2d)
    total_corners = 0

    for frame_idx, frame_det in detections.frames.items():
        if camera_name not in frame_det.detections:
            continue
        if frame_idx not in board_poses:
            continue
        det = frame_det.detections[camera_name]
        if det.num_corners < min_corners:
            continue
        obs_frames.append((frame_idx, det.corner_ids, det.corners_2d))
        total_corners += det.num_corners

    if len(obs_frames) == 0:
        raise InsufficientDataError(
            f"No usable frames for auxiliary camera '{camera_name}'. "
            f"Need frames with both detections (>={min_corners} corners) "
            f"and existing board poses."
        )

    # --- Step 2: Initial guess via refractive PnP ---
    # Use the frame with the most corners
    best_frame = max(obs_frames, key=lambda x: len(x[1]))
    best_frame_idx, best_ids, best_corners = best_frame

    pnp_result = refractive_solve_pnp(
        intrinsics, best_corners, best_ids, board,
        initial_interface_distance, interface_normal, n_air, n_water,
    )

    if pnp_result is None:
        raise InsufficientDataError(
            f"Refractive PnP failed for auxiliary camera '{camera_name}' "
            f"on frame {best_frame_idx} ({len(best_ids)} corners)."
        )

    rvec_bc, tvec_bc = pnp_result  # board-in-camera pose

    # Transform to world frame: camera extrinsics from board pose
    bp = board_poses[best_frame_idx]
    R_bw = rvec_to_matrix(bp.rvec)
    R_bc = rvec_to_matrix(rvec_bc)
    # world_to_cam = board_to_cam @ world_to_board
    R_wb, t_wb = invert_pose(R_bw, bp.tvec)
    R_wc, t_wc = compose_poses(R_bc, tvec_bc, R_wb, t_wb)

    initial_rvec = matrix_to_rvec(R_wc)
    initial_tvec = t_wc

    # --- Step 3: Build parameter vector ---
    # [rvec(3), tvec(3), interface_distance(1)] = 7 params
    x0 = np.concatenate([initial_rvec, initial_tvec, [initial_interface_distance]])

    # --- Step 4: Residual function ---
    def residuals(x):
        rvec = x[:3]
        tvec = x[3:6]
        iface_dist = x[6]

        R = rvec_to_matrix(rvec)
        ext = CameraExtrinsics(R=R, t=tvec)
        camera = Camera(camera_name, intrinsics, ext)
        interface = Interface(
            normal=interface_normal,
            base_height=0.0,
            camera_offsets={camera_name: iface_dist},
            n_air=n_air,
            n_water=n_water,
        )

        resid = []
        for frame_idx, corner_ids, corners_2d in obs_frames:
            bp = board_poses[frame_idx]
            corners_3d = board.transform_corners(bp.rvec, bp.tvec)

            for i, cid in enumerate(corner_ids):
                pt_3d = corners_3d[int(cid)]
                projected = refractive_project_fast(camera, interface, pt_3d)
                if projected is not None:
                    resid.append(projected[0] - corners_2d[i, 0])
                    resid.append(projected[1] - corners_2d[i, 1])
                else:
                    resid.append(100.0)
                    resid.append(100.0)

        # Water surface Z regularization
        if target_water_z is not None and water_z_weight > 0:
            cam_center = -R.T @ tvec
            water_z = cam_center[2] + iface_dist
            resid.append(water_z_weight * (water_z - target_water_z))

        return np.array(resid, dtype=np.float64)

    # --- Step 5: Bounds ---
    lower = np.full(7, -np.inf)
    upper = np.full(7, np.inf)
    lower[6] = 0.01  # interface_distance lower bound
    upper[6] = 2.0   # interface_distance upper bound

    # --- Step 6: Optimize ---
    result = least_squares(
        residuals,
        x0=x0,
        method="trf",
        loss="huber",
        f_scale=1.0,
        bounds=(lower, upper),
        verbose=verbose,
    )

    # --- Step 7: Extract results ---
    opt_rvec = result.x[:3]
    opt_tvec = result.x[3:6]
    opt_iface_dist = float(result.x[6])

    opt_R = rvec_to_matrix(opt_rvec)
    opt_extrinsics = CameraExtrinsics(R=opt_R, t=opt_tvec)

    # RMS over reprojection residuals only (exclude regularization term)
    n_reproj = total_corners * 2
    reproj_residuals = result.fun[:n_reproj]
    rms_error = float(np.sqrt(np.mean(reproj_residuals**2)))

    return opt_extrinsics, opt_iface_dist, rms_error
