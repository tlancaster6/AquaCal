"""Stage 4 joint refinement with optional intrinsics optimization.

This module implements the final optional refinement stage that re-optimizes
all parameters from Stage 3, with the option to also refine camera intrinsics
(focal length and principal point).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from aquacal.config.schema import (
    CameraIntrinsics,
    CameraExtrinsics,
    BoardPose,
    DetectionResult,
    ConvergenceError,
    Vec3,
)
from aquacal.core.board import BoardGeometry
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_project
from aquacal.utils.transforms import rvec_to_matrix, matrix_to_rvec


def _pack_params_with_intrinsics(
    extrinsics: dict[str, CameraExtrinsics],
    interface_distances: dict[str, float],
    board_poses: dict[int, BoardPose],
    intrinsics: dict[str, CameraIntrinsics],
    reference_camera: str,
    camera_order: list[str],
    frame_order: list[int],
    refine_intrinsics: bool,
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
    - If refine_intrinsics, for each camera (in camera_order):
        fx (1), fy (1), cx (1), cy (1)

    Total length:
    - Without intrinsics: 6*(N_cams-1) + N_cams + 6*N_frames
    - With intrinsics: above + 4*N_cams

    Args:
        extrinsics: Camera extrinsics dict
        interface_distances: Per-camera interface distances
        board_poses: Board poses dict (frame_idx -> BoardPose)
        intrinsics: Per-camera intrinsics
        reference_camera: Name of reference camera (skipped in extrinsics packing)
        camera_order: Ordered list of camera names
        frame_order: Ordered list of frame indices
        refine_intrinsics: Whether to include intrinsics in parameter vector

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

    # Pack intrinsics if refining
    if refine_intrinsics:
        for cam_name in camera_order:
            K = intrinsics[cam_name].K
            params.extend([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])  # fx, fy, cx, cy

    return np.array(params, dtype=np.float64)


def _unpack_params_with_intrinsics(
    params: NDArray[np.float64],
    reference_camera: str,
    reference_extrinsics: CameraExtrinsics,
    base_intrinsics: dict[str, CameraIntrinsics],
    camera_order: list[str],
    frame_order: list[int],
    refine_intrinsics: bool,
) -> tuple[
    dict[str, CameraExtrinsics],
    dict[str, float],
    dict[int, BoardPose],
    dict[str, CameraIntrinsics],
]:
    """
    Unpack 1D parameter array into structured objects.

    Args:
        params: 1D parameter vector
        reference_camera: Name of reference camera
        reference_extrinsics: Fixed extrinsics for reference camera
        base_intrinsics: Base intrinsics (for distortion coeffs and image_size)
        camera_order: Ordered list of camera names
        frame_order: Ordered list of frame indices
        refine_intrinsics: Whether intrinsics are included in params

    Returns:
        Tuple of (extrinsics_dict, distances_dict, board_poses_dict, intrinsics_dict)
    """
    idx = 0
    n_cams = len(camera_order)
    n_frames = len(frame_order)

    # Unpack camera extrinsics (skip reference camera)
    extrinsics_out = {}
    for cam_name in camera_order:
        if cam_name == reference_camera:
            extrinsics_out[cam_name] = reference_extrinsics
        else:
            rvec = params[idx : idx + 3]
            tvec = params[idx + 3 : idx + 6]
            idx += 6
            R = rvec_to_matrix(rvec)
            extrinsics_out[cam_name] = CameraExtrinsics(R=R, t=tvec.copy())

    # Unpack interface distances (all cameras)
    distances_out = {}
    for cam_name in camera_order:
        distances_out[cam_name] = params[idx]
        idx += 1

    # Unpack board poses
    board_poses_out = {}
    for frame_idx in frame_order:
        rvec = params[idx : idx + 3]
        tvec = params[idx + 3 : idx + 6]
        idx += 6
        board_poses_out[frame_idx] = BoardPose(
            frame_idx=frame_idx,
            rvec=rvec.copy(),
            tvec=tvec.copy(),
        )

    # Unpack intrinsics
    intrinsics_out = {}
    if refine_intrinsics:
        for cam_name in camera_order:
            fx, fy, cx, cy = params[idx : idx + 4]
            idx += 4
            base = base_intrinsics[cam_name]
            K_new = np.array(
                [
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
            intrinsics_out[cam_name] = CameraIntrinsics(
                K=K_new,
                dist_coeffs=base.dist_coeffs.copy(),  # Keep original distortion
                image_size=base.image_size,
            )
    else:
        # Return copies of base intrinsics
        for cam_name in camera_order:
            base = base_intrinsics[cam_name]
            intrinsics_out[cam_name] = CameraIntrinsics(
                K=base.K.copy(),
                dist_coeffs=base.dist_coeffs.copy(),
                image_size=base.image_size,
            )

    return extrinsics_out, distances_out, board_poses_out, intrinsics_out


def _cost_function_with_intrinsics(
    params: NDArray[np.float64],
    detections: DetectionResult,
    base_intrinsics: dict[str, CameraIntrinsics],
    board: BoardGeometry,
    reference_camera: str,
    reference_extrinsics: CameraExtrinsics,
    interface_normal: Vec3,
    n_air: float,
    n_water: float,
    camera_order: list[str],
    frame_order: list[int],
    min_corners: int,
    refine_intrinsics: bool,
) -> NDArray[np.float64]:
    """
    Compute reprojection residuals for all observations.

    Similar to Stage 3 cost function, but optionally uses refined intrinsics
    from the parameter vector instead of fixed intrinsics.

    Args:
        params: Current parameter vector
        detections: Detection results
        base_intrinsics: Base intrinsics (used for distortion and as fallback)
        board: Board geometry
        reference_camera: Name of reference camera
        reference_extrinsics: Extrinsics for reference camera (fixed)
        interface_normal: Interface normal vector
        n_air: Refractive index of air
        n_water: Refractive index of water
        camera_order: Ordered list of camera names
        frame_order: Ordered list of frame indices
        min_corners: Minimum corners per detection
        refine_intrinsics: Whether intrinsics are being refined

    Returns:
        1D array of residuals [r0_x, r0_y, r1_x, r1_y, ...] in pixels
    """
    # Unpack current parameters
    extrinsics, interface_distances, board_poses, intrinsics = _unpack_params_with_intrinsics(
        params,
        reference_camera,
        reference_extrinsics,
        base_intrinsics,
        camera_order,
        frame_order,
        refine_intrinsics,
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

            # Build Camera object (uses intrinsics from unpacked params)
            camera = Camera(cam_name, intrinsics[cam_name], extrinsics[cam_name])

            # Build Interface object for this camera
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


def _build_bounds(
    camera_order: list[str],
    frame_order: list[int],
    reference_camera: str,
    base_intrinsics: dict[str, CameraIntrinsics],
    refine_intrinsics: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Build lower and upper bounds for optimization.

    Args:
        camera_order: Ordered list of camera names
        frame_order: Ordered list of frame indices
        reference_camera: Name of reference camera
        base_intrinsics: Base intrinsics (for setting focal length bounds)
        refine_intrinsics: Whether intrinsics are being refined

    Returns:
        Tuple of (lower_bounds, upper_bounds) arrays
    """
    n_cams = len(camera_order)
    n_frames = len(frame_order)
    n_extrinsic_params = 6 * (n_cams - 1)
    n_distance_params = n_cams
    n_pose_params = 6 * n_frames
    n_intrinsic_params = 4 * n_cams if refine_intrinsics else 0
    total = n_extrinsic_params + n_distance_params + n_pose_params + n_intrinsic_params

    lower = np.full(total, -np.inf)
    upper = np.full(total, np.inf)

    # Interface distances: [0.01, 2.0]
    dist_start = n_extrinsic_params
    dist_end = dist_start + n_distance_params
    lower[dist_start:dist_end] = 0.01
    upper[dist_start:dist_end] = 2.0

    # Intrinsic bounds
    if refine_intrinsics:
        intr_start = n_extrinsic_params + n_distance_params + n_pose_params
        for i, cam_name in enumerate(camera_order):
            base = base_intrinsics[cam_name]
            fx, fy = base.K[0, 0], base.K[1, 1]
            w, h = base.image_size
            offset = intr_start + i * 4

            # fx, fy: [0.5*initial, 2.0*initial]
            lower[offset] = 0.5 * fx
            upper[offset] = 2.0 * fx
            lower[offset + 1] = 0.5 * fy
            upper[offset + 1] = 2.0 * fy

            # cx: [0, width], cy: [0, height]
            lower[offset + 2] = 0
            upper[offset + 2] = w
            lower[offset + 3] = 0
            upper[offset + 3] = h

    return lower, upper


def joint_refinement(
    stage3_result: tuple[
        dict[str, CameraExtrinsics],
        dict[str, float],
        list[BoardPose],
        float,
    ],
    detections: DetectionResult,
    intrinsics: dict[str, CameraIntrinsics],
    board: BoardGeometry,
    reference_camera: str,
    refine_intrinsics: bool = False,
    interface_normal: Vec3 | None = None,
    n_air: float = 1.0,
    n_water: float = 1.333,
    loss: str = "huber",
    loss_scale: float = 1.0,
    min_corners: int = 4,
) -> tuple[
    dict[str, CameraExtrinsics],
    dict[str, float],
    list[BoardPose],
    dict[str, CameraIntrinsics],
    float,
]:
    """
    Jointly refine all calibration parameters, optionally including intrinsics.

    This is Stage 4 of the calibration pipeline. It takes the output of Stage 3
    and performs additional optimization. When refine_intrinsics=True, it also
    optimizes focal lengths and principal points.

    Args:
        stage3_result: Output tuple from optimize_interface:
            (extrinsics, interface_distances, board_poses, rms_error)
        detections: Underwater ChArUco detections
        intrinsics: Per-camera intrinsic parameters (used as initial values)
        board: ChArUco board geometry
        reference_camera: Camera name fixed at origin
        refine_intrinsics: If True, also optimize fx, fy, cx, cy per camera
        interface_normal: Interface normal vector. If None, uses [0, 0, -1].
        n_air: Refractive index of air
        n_water: Refractive index of water
        loss: Robust loss function ("linear", "huber", "soft_l1", "cauchy")
        loss_scale: Scale parameter for robust loss in pixels
        min_corners: Minimum corners per detection to include

    Returns:
        Tuple of:
        - dict[str, CameraExtrinsics]: Refined extrinsics for all cameras
        - dict[str, float]: Refined interface distances per camera
        - list[BoardPose]: Refined board poses
        - dict[str, CameraIntrinsics]: Refined intrinsics (modified if refine_intrinsics=True,
          otherwise copies of input)
        - float: Final RMS reprojection error in pixels

    Raises:
        ConvergenceError: If optimization fails to converge
        ValueError: If reference_camera not in stage3_result extrinsics

    Notes:
        - When refine_intrinsics=False, this is essentially re-running Stage 3
          optimization from the Stage 3 solution (useful for verifying convergence)
        - Distortion coefficients are NOT refined (kept fixed)
        - Intrinsic bounds: fx, fy in [0.5*initial, 2.0*initial],
          cx, cy in [0, image_width] and [0, image_height]
    """
    # Validate inputs
    extrinsics_in, distances_in, poses_in, _ = stage3_result
    if reference_camera not in extrinsics_in:
        raise ValueError(
            f"reference_camera '{reference_camera}' not in stage3_result extrinsics. "
            f"Available cameras: {list(extrinsics_in.keys())}"
        )

    # Setup
    if interface_normal is None:
        interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        interface_normal = np.asarray(interface_normal, dtype=np.float64)

    camera_order = sorted(extrinsics_in.keys())
    board_poses_dict = {bp.frame_idx: bp for bp in poses_in}
    frame_order = sorted(board_poses_dict.keys())

    if not frame_order:
        raise ConvergenceError("No board poses from Stage 3")

    reference_extrinsics = extrinsics_in[reference_camera]

    # Pack initial parameters
    initial_params = _pack_params_with_intrinsics(
        extrinsics_in,
        distances_in,
        board_poses_dict,
        intrinsics,
        reference_camera,
        camera_order,
        frame_order,
        refine_intrinsics,
    )

    # Build bounds
    lower, upper = _build_bounds(
        camera_order,
        frame_order,
        reference_camera,
        intrinsics,
        refine_intrinsics,
    )

    # Run optimization
    result = least_squares(
        _cost_function_with_intrinsics,
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
            refine_intrinsics,
        ),
        method="trf",
        loss=loss,
        f_scale=loss_scale,
        bounds=(lower, upper),
        verbose=0,
    )

    if result.status <= 0:
        raise ConvergenceError(f"Optimization failed: {result.message}")

    # Unpack results
    ext_out, dist_out, poses_out, intr_out = _unpack_params_with_intrinsics(
        result.x,
        reference_camera,
        reference_extrinsics,
        intrinsics,
        camera_order,
        frame_order,
        refine_intrinsics,
    )

    # Convert board poses dict to sorted list
    poses_list = [poses_out[idx] for idx in sorted(poses_out.keys())]

    rms_error = np.sqrt(np.mean(result.fun**2))

    return ext_out, dist_out, poses_list, intr_out, rms_error
