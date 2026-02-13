"""Type definitions and schema for AquaCal calibration system.

This module defines all dataclasses, type aliases, and custom exceptions used
throughout the library. No validation is performed at runtime; shapes are
documented in docstrings and type hints.

Coordinate conventions:
- World frame: Z-down (into water), origin at reference camera
- Camera frame: Z-forward, X-right, Y-down (OpenCV convention)
- Interface normal: [0, 0, -1] points up (water toward air)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path
import numpy as np
from numpy.typing import NDArray

# Type aliases
Vec3 = NDArray[np.float64]  # shape (3,)
Mat3 = NDArray[np.float64]  # shape (3, 3)
Vec2 = NDArray[np.float64]  # shape (2,)


@dataclass
class BoardConfig:
    """ChArUco board specification.

    Attributes:
        squares_x: Number of chessboard squares in X direction
        squares_y: Number of chessboard squares in Y direction
        square_size: Size of each square in meters
        marker_size: Size of ArUco markers in meters
        dictionary: ArUco dictionary name (e.g., "DICT_4X4_50")
        legacy_pattern: If True, board uses legacy ChArUco pattern with marker in top-left cell.
            Default False (new pattern with solid square in top-left cell)
    """
    squares_x: int
    squares_y: int
    square_size: float  # meters
    marker_size: float  # meters
    dictionary: str  # e.g., "DICT_4X4_50"
    legacy_pattern: bool = False


@dataclass
class CameraIntrinsics:
    """Intrinsic parameters for a single camera.

    Attributes:
        K: 3x3 intrinsic matrix
        dist_coeffs: Distortion coefficients. For pinhole: length 5 or 8
            [k1, k2, p1, p2, k3, ...]. For fisheye: length 4 [k1, k2, k3, k4].
        image_size: Image dimensions as (width, height) in pixels
        is_fisheye: If True, uses equidistant fisheye projection model
    """
    K: Mat3  # 3x3 intrinsic matrix
    dist_coeffs: NDArray[np.float64]  # pinhole: length 5 or 8; fisheye: length 4
    image_size: tuple[int, int]  # (width, height)
    is_fisheye: bool = False


@dataclass
class CameraExtrinsics:
    """Extrinsic parameters for a single camera.

    Defines the transformation from world coordinates to camera coordinates.

    Attributes:
        R: 3x3 rotation matrix, world -> camera
        t: 3x1 translation vector, world -> camera
    """
    R: Mat3  # 3x3 rotation matrix, world -> camera
    t: Vec3  # 3x1 translation vector

    @property
    def C(self) -> Vec3:
        """Camera center in world coordinates.

        Returns:
            3-element array representing camera center position in world frame
        """
        return -self.R.T @ self.t


@dataclass
class CameraCalibration:
    """Complete calibration for a single camera.

    Attributes:
        name: Camera identifier (e.g., "cam0", "cam1")
        intrinsics: Intrinsic camera parameters
        extrinsics: Extrinsic camera parameters (pose in world frame)
        interface_distance: Z-coordinate of the water surface in world frame (meters).
            Same for all cameras after optimization. Despite the name, this is a
            coordinate, not a per-camera distance.
        is_auxiliary: If True, this camera was registered post-hoc against
            fixed board poses (excluded from joint Stage 3/4 optimization).
    """
    name: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    interface_distance: float
    is_auxiliary: bool = False


@dataclass
class InterfaceParams:
    """Refractive interface (water surface) parameters.

    Note: Per-camera interface distances are stored in each CameraCalibration object,
    not here. This dataclass holds only the shared interface properties.

    Attributes:
        normal: Unit vector pointing from water toward air (typically [0, 0, -1])
        n_air: Refractive index of air (default 1.0)
        n_water: Refractive index of water (default 1.333 for fresh water at 20°C)
    """
    normal: Vec3  # unit vector, points from water toward air (typically [0, 0, -1])
    n_air: float = 1.0
    n_water: float = 1.333


@dataclass
class CalibrationResult:
    """Complete calibration output.

    Attributes:
        cameras: Dictionary mapping camera names to their calibrations
        interface: Refractive interface parameters
        board: ChArUco board configuration used
        diagnostics: Calibration quality metrics
        metadata: Metadata for reproducibility
    """
    cameras: dict[str, CameraCalibration]
    interface: InterfaceParams
    board: BoardConfig
    diagnostics: DiagnosticsData
    metadata: CalibrationMetadata


@dataclass
class DiagnosticsData:
    """Calibration quality metrics.

    Attributes:
        reprojection_error_rms: Overall RMS reprojection error in pixels (primary cameras only)
        reprojection_error_per_camera: Per-camera RMS reprojection errors (primary cameras only)
        validation_3d_error_mean: Mean 3D reconstruction error in meters (holdout set, primary cameras only)
        validation_3d_error_std: Standard deviation of 3D errors in meters (primary cameras only)
        per_corner_residuals: Optional (N, 2) array of pixel errors for each corner
        per_frame_errors: Optional dict mapping frame index to error value
    """
    reprojection_error_rms: float  # pixels (primary cameras only)
    reprojection_error_per_camera: dict[str, float]  # primary cameras only
    validation_3d_error_mean: float  # meters (primary cameras only)
    validation_3d_error_std: float  # meters (primary cameras only)
    per_corner_residuals: Optional[NDArray[np.float64]] = None  # (N, 2) pixel errors
    per_frame_errors: Optional[dict[int, float]] = None


@dataclass
class CalibrationMetadata:
    """Metadata for reproducibility.

    Attributes:
        calibration_date: ISO format date string
        software_version: Version of aquacal library used
        config_hash: Hash of configuration for reproducibility
        num_frames_used: Number of frames used in calibration
        num_frames_holdout: Number of frames held out for validation
    """
    calibration_date: str
    software_version: str
    config_hash: str
    num_frames_used: int
    num_frames_holdout: int


@dataclass
class CalibrationConfig:
    """Input configuration for calibration pipeline.

    Attributes:
        board: ChArUco board specification (extrinsic/underwater board)
        camera_names: List of camera identifiers
        intrinsic_video_paths: Dict mapping camera names to intrinsic calibration videos
        extrinsic_video_paths: Dict mapping camera names to extrinsic calibration videos
        output_dir: Directory for output files
        intrinsic_board: Optional separate board for in-air intrinsic calibration (defaults to None, uses board)
        n_air: Refractive index of air (default 1.0)
        n_water: Refractive index of water (default 1.333)
        interface_normal_fixed: Whether to fix interface normal to [0, 0, -1]
        robust_loss: Loss function for optimization ("huber", "soft_l1", "linear")
        loss_scale: Scale parameter for robust loss in pixels
        min_corners_per_frame: Minimum corners required to use a detection
        min_cameras_per_frame: Minimum cameras required to use a frame
        frame_step: Process every Nth frame (1 = all frames, default 1)
        holdout_fraction: Fraction of frames to hold out for validation
        max_calibration_frames: Maximum number of frames for Stages 3-4 optimization.
            None (default) = use all calibration frames. When set, calibration frames
            are uniformly subsampled to this limit before optimization.
        refine_intrinsics: If True, Stage 4 jointly refines per-camera focal lengths
            (fx, fy) and principal points (cx, cy) alongside extrinsics and interface
            distances. Distortion coefficients are NOT refined. Only enable after
            Stage 3 converges reliably. Default False (Stage 4 skipped).
        save_detailed_residuals: Whether to save per-corner residuals
        initial_interface_distances: Optional dict mapping camera names to approximate
            camera-to-water-surface distances in meters. When None, all cameras default
            to 0.15m. Doesn't need to be exact — within 2-3x of the true value is
            sufficient for good initialization in Stage 3.
        rational_model_cameras: List of camera names that should use the 8-coefficient
            rational distortion model instead of the standard 5-coefficient model.
            Use for wide-angle lenses where 5 coefficients are insufficient.
        auxiliary_cameras: List of auxiliary camera names registered post-hoc against
            fixed board poses. These cameras are calibrated for intrinsics and
            detected, but excluded from joint Stage 3/4 optimization. Must not
            overlap with camera_names.
        fisheye_cameras: List of camera names that should use the equidistant
            fisheye projection model. Must be a subset of auxiliary_cameras
            and must not overlap with rational_model_cameras.
        refine_auxiliary_intrinsics: If True, Stage 4b refines auxiliary camera
            intrinsics (fx, fy, cx, cy) alongside extrinsics. Requires
            auxiliary_cameras to be set. Independent of refine_intrinsics (which
            controls primary camera refinement in Stage 4). Distortion coefficients
            are NOT refined.
    """
    board: BoardConfig
    camera_names: list[str]
    intrinsic_video_paths: dict[str, Path]
    extrinsic_video_paths: dict[str, Path]
    output_dir: Path
    intrinsic_board: BoardConfig | None = None
    n_air: float = 1.0
    n_water: float = 1.333
    interface_normal_fixed: bool = False
    robust_loss: str = "huber"  # "huber", "soft_l1", "linear"
    loss_scale: float = 1.0  # pixels
    min_corners_per_frame: int = 8
    min_cameras_per_frame: int = 2
    frame_step: int = 1  # Process every Nth frame (1 = all frames)
    holdout_fraction: float = 0.2  # Random selection; frames are held out entirely (not per-detection)
    max_calibration_frames: int | None = None  # None = no limit, use all frames
    refine_intrinsics: bool = False
    refine_auxiliary_intrinsics: bool = False  # If True, Stage 4b refines auxiliary camera intrinsics (fx, fy, cx, cy) alongside extrinsics. Requires auxiliary_cameras to be set. Independent of refine_intrinsics (which controls primary camera refinement in Stage 4). Distortion coefficients are NOT refined.
    save_detailed_residuals: bool = True
    initial_interface_distances: dict[str, float] | None = None
    rational_model_cameras: list[str] = field(default_factory=list)
    auxiliary_cameras: list[str] = field(default_factory=list)
    fisheye_cameras: list[str] = field(default_factory=list)


@dataclass
class BoardPose:
    """Pose of ChArUco board in a single frame.

    Attributes:
        frame_idx: Frame index in the video sequence
        rvec: Rodrigues rotation vector (3,)
        tvec: Translation vector (3,)
    """
    frame_idx: int
    rvec: Vec3  # Rodrigues rotation vector
    tvec: Vec3  # translation vector


@dataclass
class Detection:
    """ChArUco detection in a single image.

    Attributes:
        corner_ids: Array of corner IDs detected, shape (N,)
        corners_2d: Array of 2D corner positions in pixels, shape (N, 2)
    """
    corner_ids: NDArray[np.int32]  # shape (N,)
    corners_2d: NDArray[np.float64]  # shape (N, 2)

    @property
    def num_corners(self) -> int:
        """Number of detected corners.

        Returns:
            Count of detected corners
        """
        return len(self.corner_ids)


@dataclass
class FrameDetections:
    """Detections across all cameras for a single frame.

    Attributes:
        frame_idx: Frame index in the video sequence
        detections: Dict mapping camera names to Detection objects
    """
    frame_idx: int
    detections: dict[str, Detection]  # camera_name -> Detection

    @property
    def cameras_with_detections(self) -> list[str]:
        """List of camera names that detected the board in this frame.

        Returns:
            List of camera names
        """
        return list(self.detections.keys())

    @property
    def num_cameras(self) -> int:
        """Number of cameras that detected the board in this frame.

        Returns:
            Count of cameras with detections
        """
        return len(self.detections)


@dataclass
class DetectionResult:
    """All detections from a video set.

    Attributes:
        frames: Dict mapping frame indices to FrameDetections
        camera_names: List of all camera names in the dataset
        total_frames: Total number of frames processed
    """
    frames: dict[int, FrameDetections]  # frame_idx -> FrameDetections
    camera_names: list[str]
    total_frames: int

    def get_frames_with_min_cameras(self, min_cameras: int) -> list[int]:
        """Return frame indices where at least min_cameras see the board.

        Args:
            min_cameras: Minimum number of cameras required

        Returns:
            List of frame indices meeting the criterion
        """
        return [idx for idx, fd in self.frames.items() if fd.num_cameras >= min_cameras]


# --- Custom Exceptions ---

class CalibrationError(Exception):
    """Base class for calibration-related errors."""
    pass


class InsufficientDataError(CalibrationError):
    """Raised when there isn't enough data for calibration."""
    pass


class ConvergenceError(CalibrationError):
    """Raised when optimization fails to converge."""
    pass


class ConnectivityError(CalibrationError):
    """Raised when pose graph is not connected (cameras cannot be linked)."""
    pass
