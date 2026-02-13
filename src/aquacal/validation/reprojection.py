"""Reprojection error computation for calibration validation."""

from dataclasses import dataclass
import warnings

import numpy as np
from numpy.typing import NDArray

from aquacal.config.schema import (
    CalibrationResult,
    DetectionResult,
    BoardPose,
    Detection,
)
from aquacal.core.board import BoardGeometry
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_project


@dataclass
class ReprojectionErrors:
    """Container for reprojection error statistics.

    Attributes:
        rms: Overall RMS reprojection error in pixels
        per_camera: Dict mapping camera name to RMS error for that camera
        per_frame: Dict mapping frame index to RMS error for that frame
        residuals: (N, 2) array of per-corner residuals (detected - projected)
        num_observations: Total number of corner observations used
    """

    rms: float
    per_camera: dict[str, float]
    per_frame: dict[int, float]
    residuals: NDArray[np.float64]  # (N, 2)
    num_observations: int


def compute_reprojection_errors(
    calibration: CalibrationResult,
    detections: DetectionResult,
    board_poses: dict[int, BoardPose],
) -> ReprojectionErrors:
    """
    Compute reprojection errors for all observations.

    For each frame, camera, and detected corner:
    1. Transform corner from board frame to world frame using board pose
    2. Project through refractive interface using refractive_project()
    3. Compute residual: detected_pixel - projected_pixel

    Args:
        calibration: Complete calibration result containing camera calibrations
                     and interface parameters
        detections: Detection result with 2D corner observations
        board_poses: Dict mapping frame_idx to BoardPose (optimized poses)

    Returns:
        ReprojectionErrors with all statistics computed

    Notes:
        - Skips observations where refractive_project() returns None
        - RMS is sqrt(mean(residual_x^2 + residual_y^2))
    """
    board = BoardGeometry(calibration.board)

    all_residuals = []
    per_camera_residuals = {cam: [] for cam in calibration.cameras}
    per_frame_residuals = {idx: [] for idx in board_poses}

    for frame_idx, frame_det in detections.frames.items():
        if frame_idx not in board_poses:
            continue

        board_pose = board_poses[frame_idx]
        corners_3d = board.transform_corners(board_pose.rvec, board_pose.tvec)

        for cam_name, detection in frame_det.detections.items():
            if cam_name not in calibration.cameras:
                continue

            cam_calib = calibration.cameras[cam_name]
            camera = Camera(cam_name, cam_calib.intrinsics, cam_calib.extrinsics)
            interface = Interface(
                normal=calibration.interface.normal,
                camera_distances={cam_name: cam_calib.interface_distance},
                n_air=calibration.interface.n_air,
                n_water=calibration.interface.n_water,
            )

            for i, corner_id in enumerate(detection.corner_ids):
                point_3d = corners_3d[int(corner_id)]
                projected = refractive_project(camera, interface, point_3d)

                if projected is not None:
                    detected = detection.corners_2d[i]
                    residual = detected - projected
                    all_residuals.append(residual)
                    per_camera_residuals[cam_name].append(residual)
                    per_frame_residuals[frame_idx].append(residual)

    # Helper function to compute RMS from list of residuals
    def compute_rms(residuals: list[NDArray]) -> float:
        """Compute RMS from list of (2,) residual arrays."""
        if not residuals:
            warnings.warn("No residuals to compute RMS - returning NaN")
            return float("nan")
        arr = np.array(residuals)  # (N, 2)
        return np.sqrt(np.mean(arr[:, 0] ** 2 + arr[:, 1] ** 2))

    # Build final result
    residuals_array = np.array(all_residuals) if all_residuals else np.empty((0, 2))

    per_camera_rms = {
        cam: compute_rms(resids)
        for cam, resids in per_camera_residuals.items()
        if resids  # Only include cameras with observations
    }

    per_frame_rms = {
        idx: compute_rms(resids)
        for idx, resids in per_frame_residuals.items()
        if resids  # Only include frames with observations
    }

    return ReprojectionErrors(
        rms=compute_rms(all_residuals),
        per_camera=per_camera_rms,
        per_frame=per_frame_rms,
        residuals=residuals_array,
        num_observations=len(all_residuals),
    )


def compute_reprojection_error_single(
    camera: Camera,
    interface: Interface,
    board: BoardGeometry,
    board_pose: BoardPose,
    detection: Detection,
) -> tuple[NDArray[np.float64], NDArray[np.int32]] | tuple[None, None]:
    """
    Compute reprojection errors for a single camera/frame pair.

    Args:
        camera: Camera object with intrinsics and extrinsics
        interface: Interface object configured for this camera
        board: Board geometry for corner positions
        board_pose: Pose of board for this frame
        detection: Detected corners in this camera/frame

    Returns:
        Tuple of (residuals, valid_ids):
        - residuals: (M, 2) array of pixel residuals for valid corners
        - valid_ids: (M,) array of corner IDs that were successfully projected
        Returns (None, None) if no corners could be projected.
    """
    corners_3d = board.transform_corners(board_pose.rvec, board_pose.tvec)

    residuals_list = []
    valid_ids_list = []

    for i, corner_id in enumerate(detection.corner_ids):
        point_3d = corners_3d[int(corner_id)]
        projected = refractive_project(camera, interface, point_3d)

        if projected is not None:
            detected = detection.corners_2d[i]
            residual = detected - projected
            residuals_list.append(residual)
            valid_ids_list.append(corner_id)

    if not residuals_list:
        return None, None

    residuals = np.array(residuals_list, dtype=np.float64)
    valid_ids = np.array(valid_ids_list, dtype=np.int32)

    return residuals, valid_ids
