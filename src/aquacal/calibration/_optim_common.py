"""Shared optimization utilities for calibration stages 3 and 4.

Provides parameter packing/unpacking, Jacobian sparsity construction,
bounds building, residual computation, and sparse Jacobian helpers
used by both interface_estimation (Stage 3) and refinement (Stage 4).
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize._numdiff import approx_derivative, group_columns

from aquacal.config.schema import (
    CameraIntrinsics,
    CameraExtrinsics,
    BoardPose,
    DetectionResult,
    Vec3,
)
from aquacal.core.board import BoardGeometry
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_project
from aquacal.utils.transforms import rvec_to_matrix, matrix_to_rvec


def pack_params(
    extrinsics: dict[str, CameraExtrinsics],
    water_z: float,
    board_poses: dict[int, BoardPose],
    reference_camera: str,
    camera_order: list[str],
    frame_order: list[int],
    intrinsics: dict[str, CameraIntrinsics] | None = None,
    refine_intrinsics: bool = False,
    normal_fixed: bool = True,
) -> NDArray[np.float64]:
    """
    Pack optimization parameters into a 1D array.

    Parameter layout (when normal_fixed=False, tilt params are prepended):
    - If not normal_fixed: reference camera tilt rx, ry (2)
    - For each non-reference camera (in camera_order, skipping reference):
        cam_rvec (3), cam_tvec (3)
    - water_z (1): global water surface Z coordinate
    - For each frame (in frame_order):
        board_rvec (3), board_tvec (3)
    - If refine_intrinsics, for each camera (in camera_order):
        fx (1), fy (1), cx (1), cy (1)

    Args:
        extrinsics: Camera extrinsics dict
        water_z: Global water surface Z coordinate. This is stored as the
            interface_distance for all cameras (a Z-coordinate, not a distance).
        board_poses: Board poses dict (frame_idx -> BoardPose)
        reference_camera: Name of reference camera (skipped in extrinsics packing)
        camera_order: Ordered list of camera names
        frame_order: Ordered list of frame indices
        intrinsics: Per-camera intrinsics (required if refine_intrinsics=True)
        refine_intrinsics: Whether to include intrinsics in parameter vector
        normal_fixed: If False, prepend 2 tilt params (rx, ry) for reference camera

    Returns:
        1D parameter vector
    """
    params = []

    # Pack reference camera tilt (if estimating)
    if not normal_fixed:
        rvec = matrix_to_rvec(extrinsics[reference_camera].R)
        params.extend(rvec[:2].tolist())

    # Pack camera extrinsics (skip reference camera)
    for cam_name in camera_order:
        if cam_name == reference_camera:
            continue
        ext = extrinsics[cam_name]
        rvec = matrix_to_rvec(ext.R)
        params.extend(rvec.tolist())
        params.extend(ext.t.tolist())

    # Pack water surface Z (single parameter replacing N distances)
    params.append(water_z)

    # Pack board poses
    for frame_idx in frame_order:
        bp = board_poses[frame_idx]
        params.extend(bp.rvec.tolist())
        params.extend(bp.tvec.tolist())

    # Pack intrinsics if refining
    if refine_intrinsics:
        for cam_name in camera_order:
            K = intrinsics[cam_name].K
            params.extend([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])

    return np.array(params, dtype=np.float64)


def unpack_params(
    params: NDArray[np.float64],
    reference_camera: str,
    reference_extrinsics: CameraExtrinsics,
    camera_order: list[str],
    frame_order: list[int],
    base_intrinsics: dict[str, CameraIntrinsics] | None = None,
    refine_intrinsics: bool = False,
    normal_fixed: bool = True,
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
        reference_extrinsics: Fixed extrinsics for reference camera (used when
            normal_fixed=True; ignored when normal_fixed=False since reference
            extrinsics come from the parameter vector)
        camera_order: Ordered list of camera names
        frame_order: Ordered list of frame indices
        base_intrinsics: Base intrinsics (for distortion coeffs and image_size).
            Required if refine_intrinsics=True. When False, copies are returned
            if provided, otherwise an empty dict.
        refine_intrinsics: Whether intrinsics are included in params
        normal_fixed: If False, first 2 params are tilt (rx, ry) for reference camera

    Returns:
        Tuple of (extrinsics_dict, distances_dict, board_poses_dict, intrinsics_dict)
    """
    idx = 0

    # Unpack reference camera tilt (if estimating)
    if not normal_fixed:
        rx, ry = params[0], params[1]
        idx = 2
        R_ref = rvec_to_matrix(np.array([rx, ry, 0.0]))
        ref_ext = CameraExtrinsics(R=R_ref, t=np.zeros(3, dtype=np.float64))
    else:
        ref_ext = reference_extrinsics

    # Unpack camera extrinsics (skip reference camera)
    extrinsics_out = {}
    for cam_name in camera_order:
        if cam_name == reference_camera:
            extrinsics_out[cam_name] = ref_ext
        else:
            rvec = params[idx : idx + 3]
            tvec = params[idx + 3 : idx + 6]
            idx += 6
            R = rvec_to_matrix(rvec)
            extrinsics_out[cam_name] = CameraExtrinsics(R=R, t=tvec.copy())

    # Unpack water surface Z (single parameter)
    water_z = float(params[idx])
    idx += 1

    # Derive per-camera interface distances from water_z
    # interface_distance is the Z-coordinate of the water surface for all cameras
    distances_out = {}
    for cam_name in camera_order:
        distances_out[cam_name] = water_z

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
                [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                dtype=np.float64,
            )
            intrinsics_out[cam_name] = CameraIntrinsics(
                K=K_new,
                dist_coeffs=base.dist_coeffs.copy(),
                image_size=base.image_size,
            )
    elif base_intrinsics is not None:
        for cam_name in camera_order:
            base = base_intrinsics[cam_name]
            intrinsics_out[cam_name] = CameraIntrinsics(
                K=base.K.copy(),
                dist_coeffs=base.dist_coeffs.copy(),
                image_size=base.image_size,
            )

    return extrinsics_out, distances_out, board_poses_out, intrinsics_out


def build_jacobian_sparsity(
    detections: DetectionResult,
    reference_camera: str,
    camera_order: list[str],
    frame_order: list[int],
    min_corners: int,
    refine_intrinsics: bool = False,
    normal_fixed: bool = True,
) -> NDArray[np.int8]:
    """
    Build sparse Jacobian structure matrix.

    Each residual (x and y error for one corner observation) depends only on:
    - Tilt params: 2 params (only if normal_fixed=False AND camera is reference)
    - Camera extrinsics: 6 params (or 0 if reference camera)
    - water_z: 1 param (dense — ALL cameras depend on it)
    - Board pose for that frame: 6 params
    - Camera intrinsics for that camera: 4 params (if refine_intrinsics)

    Args:
        detections: Detection results
        reference_camera: Name of reference camera (no extrinsic params)
        camera_order: Ordered list of camera names
        frame_order: Ordered list of frame indices
        min_corners: Minimum corners per detection
        refine_intrinsics: Whether intrinsics are in the parameter vector
        normal_fixed: If False, 2 tilt params are prepended to the parameter vector

    Returns:
        Sparse matrix of shape (n_residuals, n_params) with 1s where
        Jacobian may be non-zero.
    """
    n_cams = len(camera_order)
    n_frames = len(frame_order)

    n_tilt_params = 0 if normal_fixed else 2
    n_extrinsic_params = 6 * (n_cams - 1)
    n_water_z_params = 1
    n_pose_params = 6 * n_frames
    n_intrinsic_params = 4 * n_cams if refine_intrinsics else 0
    n_params = (
        n_tilt_params
        + n_extrinsic_params
        + n_water_z_params
        + n_pose_params
        + n_intrinsic_params
    )

    # Build camera index mappings
    cam_to_ext_idx = {}
    ext_idx = 0
    for cam_name in camera_order:
        if cam_name != reference_camera:
            cam_to_ext_idx[cam_name] = ext_idx
            ext_idx += 1

    cam_to_cam_idx = {cam: i for i, cam in enumerate(camera_order)}
    frame_to_pose_idx = {frame: i for i, frame in enumerate(frame_order)}

    water_z_col = n_tilt_params + n_extrinsic_params

    residual_rows = []

    for frame_idx in frame_order:
        if frame_idx not in detections.frames:
            continue

        frame_det = detections.frames[frame_idx]
        pose_idx = frame_to_pose_idx[frame_idx]

        for cam_name in camera_order:
            if cam_name not in frame_det.detections:
                continue

            detection = frame_det.detections[cam_name]
            if detection.num_corners < min_corners:
                continue

            for _ in range(detection.num_corners):
                row = np.zeros(n_params, dtype=np.int8)

                # 0. Tilt params (reference camera residuals only)
                if not normal_fixed and cam_name == reference_camera:
                    row[0:2] = 1

                # 1. Camera extrinsics (if not reference)
                if cam_name in cam_to_ext_idx:
                    ext_start = n_tilt_params + cam_to_ext_idx[cam_name] * 6
                    row[ext_start : ext_start + 6] = 1

                # 2. water_z affects ALL cameras (dense column)
                row[water_z_col] = 1

                # 3. Board pose for this frame
                pose_start = (
                    n_tilt_params + n_extrinsic_params + n_water_z_params + pose_idx * 6
                )
                row[pose_start : pose_start + 6] = 1

                # 4. Camera intrinsics (if refining)
                if refine_intrinsics:
                    cam_idx = cam_to_cam_idx[cam_name]
                    intr_start = (
                        n_tilt_params
                        + n_extrinsic_params
                        + n_water_z_params
                        + n_pose_params
                        + cam_idx * 4
                    )
                    row[intr_start : intr_start + 4] = 1

                # Two residuals (x and y) with same sparsity pattern
                residual_rows.append(row)
                residual_rows.append(row.copy())

    return np.array(residual_rows, dtype=np.int8)


def build_bounds(
    camera_order: list[str],
    frame_order: list[int],
    reference_camera: str,
    base_intrinsics: dict[str, CameraIntrinsics] | None = None,
    refine_intrinsics: bool = False,
    normal_fixed: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Build lower and upper bounds for optimization.

    Args:
        camera_order: Ordered list of camera names
        frame_order: Ordered list of frame indices
        reference_camera: Name of reference camera
        base_intrinsics: Base intrinsics (required if refine_intrinsics=True)
        refine_intrinsics: Whether intrinsics are being refined
        normal_fixed: If False, 2 tilt bounds are prepended

    Returns:
        Tuple of (lower_bounds, upper_bounds) arrays
    """
    n_cams = len(camera_order)
    n_frames = len(frame_order)
    n_tilt_params = 0 if normal_fixed else 2
    n_extrinsic_params = 6 * (n_cams - 1)
    n_water_z_params = 1
    n_pose_params = 6 * n_frames
    n_intrinsic_params = 4 * n_cams if refine_intrinsics else 0
    total = (
        n_tilt_params
        + n_extrinsic_params
        + n_water_z_params
        + n_pose_params
        + n_intrinsic_params
    )

    lower = np.full(total, -np.inf)
    upper = np.full(total, np.inf)

    # Tilt bounds: [-0.2, 0.2] radians (~11 degrees)
    if not normal_fixed:
        lower[0:2] = -0.2
        upper[0:2] = 0.2

    # Water surface Z bound: [0.01, 2.0] meters
    water_z_idx = n_tilt_params + n_extrinsic_params
    lower[water_z_idx] = 0.01
    upper[water_z_idx] = 2.0

    # Intrinsic bounds
    if refine_intrinsics:
        intr_start = (
            n_tilt_params + n_extrinsic_params + n_water_z_params + n_pose_params
        )
        for i, cam_name in enumerate(camera_order):
            base = base_intrinsics[cam_name]
            fx, fy = base.K[0, 0], base.K[1, 1]
            w, h = base.image_size
            offset = intr_start + i * 4

            lower[offset] = 0.5 * fx
            upper[offset] = 2.0 * fx
            lower[offset + 1] = 0.5 * fy
            upper[offset + 1] = 2.0 * fy
            lower[offset + 2] = 0
            upper[offset + 2] = w
            lower[offset + 3] = 0
            upper[offset + 3] = h

    return lower, upper


def compute_residuals(
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
    refine_intrinsics: bool = False,
    normal_fixed: bool = True,
) -> NDArray[np.float64]:
    """
    Compute reprojection residuals for all observations.

    When refine_intrinsics=False, intrinsics are taken from base_intrinsics.
    When refine_intrinsics=True, intrinsics are unpacked from the parameter vector.

    Args:
        params: Current parameter vector
        detections: Detection results
        base_intrinsics: Base intrinsics (fixed when not refining, used for
            distortion coeffs and image_size when refining)
        board: Board geometry
        reference_camera: Name of reference camera
        reference_extrinsics: Extrinsics for reference camera (fixed when
            normal_fixed=True; ignored when normal_fixed=False)
        interface_normal: Interface normal vector
        n_air: Refractive index of air
        n_water: Refractive index of water
        camera_order: Ordered list of camera names
        frame_order: Ordered list of frame indices
        min_corners: Minimum corners per detection
        refine_intrinsics: Whether intrinsics are being refined
        normal_fixed: If False, first 2 params are tilt for reference camera

    Returns:
        1D array of residuals [r0_x, r0_y, r1_x, r1_y, ...] in pixels.
    """
    extrinsics, interface_distances, board_poses, intrinsics = unpack_params(
        params,
        reference_camera,
        reference_extrinsics,
        camera_order,
        frame_order,
        base_intrinsics=base_intrinsics,
        refine_intrinsics=refine_intrinsics,
        normal_fixed=normal_fixed,
    )

    residuals = []

    for frame_idx in frame_order:
        if frame_idx not in detections.frames:
            continue

        board_pose = board_poses[frame_idx]
        corners_3d = board.transform_corners(board_pose.rvec, board_pose.tvec)

        for cam_name in camera_order:
            frame_det = detections.frames[frame_idx]
            if cam_name not in frame_det.detections:
                continue

            detection = frame_det.detections[cam_name]
            if detection.num_corners < min_corners:
                continue

            camera = Camera(cam_name, intrinsics[cam_name], extrinsics[cam_name])

            interface = Interface(
                normal=interface_normal,
                camera_distances={cam_name: interface_distances[cam_name]},
                n_air=n_air,
                n_water=n_water,
            )

            for i, corner_id in enumerate(detection.corner_ids):
                point_3d = corners_3d[corner_id]
                detected_px = detection.corners_2d[i]

                projected = refractive_project(camera, interface, point_3d)

                if projected is None:
                    residuals.extend([100.0, 100.0])
                else:
                    residuals.extend(
                        [
                            projected[0] - detected_px[0],
                            projected[1] - detected_px[1],
                        ]
                    )

    return np.array(residuals, dtype=np.float64)


def make_sparse_jacobian_func(
    cost_func,
    cost_args: tuple,
    jac_sparsity: NDArray[np.int8],
    bounds: tuple[NDArray[np.float64], NDArray[np.float64]],
    dense_threshold: int = 500_000_000,
):
    """
    Create a Jacobian callable that uses sparse finite differences.

    Uses sparse column grouping for efficient FD computation. Returns a
    dense matrix for small problems (enabling the exact trust-region solver)
    or a sparse matrix for large problems (using LSMR solver to avoid OOM).

    Args:
        cost_func: The cost function
        cost_args: Arguments to pass to cost_func after params
        jac_sparsity: Sparsity pattern matrix (n_residuals, n_params)
        bounds: Tuple of (lower, upper) bound arrays
        dense_threshold: Maximum number of elements (rows*cols) before
            returning sparse instead of dense. Default 500M (~4 GiB).

    Returns:
        Callable that takes (params, *args) and returns Jacobian matrix
    """
    groups = group_columns(jac_sparsity)
    n_elements = jac_sparsity.shape[0] * jac_sparsity.shape[1]
    use_dense = n_elements <= dense_threshold

    def jac_func(params, *args):
        J_sparse = approx_derivative(
            lambda x: cost_func(x, *args),
            params,
            method="2-point",
            sparsity=(jac_sparsity, groups),
            bounds=bounds,
        )
        if use_dense:
            return J_sparse.toarray() if hasattr(J_sparse, "toarray") else np.asarray(J_sparse)
        return J_sparse

    return jac_func
