# Agent Implementation Specification: Refractive Multi-Camera Calibration

This document provides explicit specifications for AI agent (Claude Code) implementation. It includes exact type definitions, function signatures, test cases, and implementation order.

---

## Implementation Order (Dependency Graph)

```
Phase 1 (no dependencies):
  ├── config/schema.py
  ├── utils/transforms.py
  └── core/board.py

Phase 2 (depends on Phase 1):
  ├── core/camera.py        (depends on: transforms)
  └── core/interface.py     (depends on: nothing)

Phase 3 (depends on Phase 2):
  └── core/refractive_geometry.py  (depends on: camera, interface, transforms)

Phase 4 (depends on Phase 1):
  ├── io/video.py           (depends on: nothing)
  └── io/detection.py       (depends on: board)

Phase 5 (depends on Phases 3, 4):
  └── io/serialization.py   (depends on: schema)

Phase 6 (depends on Phases 3, 4):
  ├── calibration/intrinsics.py    (depends on: camera, board, detection)
  └── triangulation/triangulate.py (depends on: refractive_geometry)

Phase 7 (depends on Phase 6):
  └── calibration/extrinsics.py    (depends on: camera, board, detection, transforms)

Phase 8 (depends on Phase 7):
  └── calibration/interface.py     (depends on: camera, interface, refractive_geometry, board)

Phase 9 (depends on Phase 8):
  ├── calibration/refinement.py    (depends on: interface stage)
  ├── validation/reprojection.py   (depends on: refractive_geometry)
  ├── validation/reconstruction.py (depends on: triangulate, board)
  └── validation/diagnostics.py    (depends on: reprojection, reconstruction)

Phase 10 (depends on Phase 9):
  └── calibration/pipeline.py      (depends on: all calibration and validation modules)
```

---

## Type Definitions

All types use dataclasses or numpy arrays. Use `from __future__ import annotations` for forward references.

### `config/schema.py`

```python
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
    """ChArUco board specification."""
    squares_x: int
    squares_y: int
    square_size: float  # meters
    marker_size: float  # meters
    dictionary: str  # e.g., "DICT_4X4_50"


@dataclass
class CameraIntrinsics:
    """Intrinsic parameters for a single camera."""
    K: Mat3  # 3x3 intrinsic matrix
    dist_coeffs: NDArray[np.float64]  # length 5 or 8: [k1, k2, p1, p2, k3, ...]
    image_size: tuple[int, int]  # (width, height)


@dataclass
class CameraExtrinsics:
    """Extrinsic parameters for a single camera."""
    R: Mat3  # 3x3 rotation matrix, world -> camera
    t: Vec3  # 3x1 translation vector
    
    @property
    def C(self) -> Vec3:
        """Camera center in world coordinates."""
        return -self.R.T @ self.t


@dataclass
class CameraCalibration:
    """Complete calibration for a single camera."""
    name: str
    intrinsics: CameraIntrinsics
    extrinsics: CameraExtrinsics
    interface_distance: float  # meters from camera center to water surface


@dataclass
class InterfaceParams:
    """Refractive interface (water surface) parameters."""
    normal: Vec3  # unit vector, points from water toward air (typically [0, 0, 1])
    n_air: float = 1.0
    n_water: float = 1.333


@dataclass
class CalibrationResult:
    """Complete calibration output."""
    cameras: dict[str, CameraCalibration]
    interface: InterfaceParams
    board: BoardConfig
    diagnostics: DiagnosticsData
    metadata: CalibrationMetadata


@dataclass
class DiagnosticsData:
    """Calibration quality metrics."""
    reprojection_error_rms: float  # pixels
    reprojection_error_per_camera: dict[str, float]
    validation_3d_error_mean: float  # meters
    validation_3d_error_std: float  # meters
    per_corner_residuals: Optional[NDArray[np.float64]] = None  # (N, 2) pixel errors
    per_frame_errors: Optional[dict[int, float]] = None


@dataclass
class CalibrationMetadata:
    """Metadata for reproducibility."""
    calibration_date: str
    software_version: str
    config_hash: str
    num_frames_used: int
    num_frames_holdout: int


@dataclass 
class CalibrationConfig:
    """Input configuration for calibration pipeline."""
    board: BoardConfig
    camera_names: list[str]
    intrinsic_video_paths: dict[str, Path]
    extrinsic_video_paths: dict[str, Path]
    output_dir: Path
    n_air: float = 1.0
    n_water: float = 1.333
    interface_normal_fixed: bool = True
    robust_loss: str = "huber"  # "huber", "soft_l1", "linear"
    loss_scale: float = 1.0  # pixels
    min_corners_per_frame: int = 8
    min_cameras_per_frame: int = 2
    holdout_fraction: float = 0.2
    save_detailed_residuals: bool = True


@dataclass
class BoardPose:
    """Pose of ChArUco board in a single frame."""
    frame_idx: int
    rvec: Vec3  # Rodrigues rotation vector
    tvec: Vec3  # translation vector


@dataclass
class Detection:
    """ChArUco detection in a single image."""
    corner_ids: NDArray[np.int32]  # shape (N,)
    corners_2d: NDArray[np.float64]  # shape (N, 2)
    
    @property
    def num_corners(self) -> int:
        return len(self.corner_ids)


@dataclass
class FrameDetections:
    """Detections across all cameras for a single frame."""
    frame_idx: int
    detections: dict[str, Detection]  # camera_name -> Detection
    
    @property
    def cameras_with_detections(self) -> list[str]:
        return list(self.detections.keys())
    
    @property
    def num_cameras(self) -> int:
        return len(self.detections)


@dataclass
class DetectionResult:
    """All detections from a video set."""
    frames: dict[int, FrameDetections]  # frame_idx -> FrameDetections
    camera_names: list[str]
    total_frames: int
    
    def get_frames_with_min_cameras(self, min_cameras: int) -> list[int]:
        """Return frame indices where at least min_cameras see the board."""
        return [idx for idx, fd in self.frames.items() if fd.num_cameras >= min_cameras]
```

---

## Module Specifications

### `utils/transforms.py`

```python
"""Rotation and coordinate transform utilities."""

import numpy as np
from numpy.typing import NDArray
import cv2

Vec3 = NDArray[np.float64]
Mat3 = NDArray[np.float64]


def rvec_to_matrix(rvec: Vec3) -> Mat3:
    """
    Convert Rodrigues vector to rotation matrix.
    
    Args:
        rvec: Rotation vector, shape (3,)
        
    Returns:
        R: Rotation matrix, shape (3, 3)
        
    Example:
        >>> rvec = np.array([0.0, 0.0, np.pi/2])
        >>> R = rvec_to_matrix(rvec)
        >>> np.allclose(R @ np.array([1, 0, 0]), np.array([0, 1, 0]))
        True
    """
    pass  # Implementation


def matrix_to_rvec(R: Mat3) -> Vec3:
    """
    Convert rotation matrix to Rodrigues vector.
    
    Args:
        R: Rotation matrix, shape (3, 3)
        
    Returns:
        rvec: Rotation vector, shape (3,)
        
    Example:
        >>> R = np.eye(3)
        >>> rvec = matrix_to_rvec(R)
        >>> np.allclose(rvec, np.zeros(3))
        True
    """
    pass  # Implementation


def compose_poses(R1: Mat3, t1: Vec3, R2: Mat3, t2: Vec3) -> tuple[Mat3, Vec3]:
    """
    Compose two poses: T_combined = T1 @ T2.
    
    If T1 transforms from frame A to frame B, and T2 transforms from frame B to frame C,
    then T_combined transforms from frame A to frame C.
    
    Args:
        R1, t1: First pose (rotation matrix, translation)
        R2, t2: Second pose
        
    Returns:
        R_combined, t_combined: Composed pose
        
    Example:
        >>> R1, t1 = np.eye(3), np.array([1, 0, 0])
        >>> R2, t2 = np.eye(3), np.array([0, 1, 0])
        >>> R, t = compose_poses(R1, t1, R2, t2)
        >>> np.allclose(t, np.array([1, 1, 0]))
        True
    """
    pass  # Implementation


def invert_pose(R: Mat3, t: Vec3) -> tuple[Mat3, Vec3]:
    """
    Invert a pose transformation.
    
    Args:
        R, t: Pose to invert
        
    Returns:
        R_inv, t_inv: Inverted pose such that compose_poses(R, t, R_inv, t_inv) = identity
        
    Example:
        >>> R = np.eye(3)
        >>> t = np.array([1, 2, 3])
        >>> R_inv, t_inv = invert_pose(R, t)
        >>> np.allclose(t_inv, np.array([-1, -2, -3]))
        True
    """
    pass  # Implementation


def camera_center(R: Mat3, t: Vec3) -> Vec3:
    """
    Compute camera center in world coordinates.
    
    The camera center C satisfies: t = -R @ C
    
    Args:
        R: Rotation matrix (world to camera)
        t: Translation vector
        
    Returns:
        C: Camera center in world frame, shape (3,)
        
    Example:
        >>> R = np.eye(3)
        >>> t = np.array([0, 0, 5])
        >>> C = camera_center(R, t)
        >>> np.allclose(C, np.array([0, 0, -5]))
        True
    """
    pass  # Implementation
```

---

### `core/board.py`

```python
"""ChArUco board geometry and utilities."""

import numpy as np
from numpy.typing import NDArray
import cv2

# Import from schema
from config.schema import BoardConfig, Vec3


class BoardGeometry:
    """
    ChArUco board 3D geometry.
    
    The board frame has origin at the top-left corner (when viewed from front),
    with X pointing right, Y pointing down, and Z pointing into the board.
    
    Attributes:
        config: Board configuration
        corner_positions: Dict mapping corner_id to 3D position in board frame
        num_corners: Total number of interior corners
    """
    
    def __init__(self, config: BoardConfig):
        """
        Initialize board geometry from config.
        
        Args:
            config: Board configuration
            
        Example:
            >>> config = BoardConfig(squares_x=8, squares_y=6, square_size=0.03, 
            ...                       marker_size=0.022, dictionary="DICT_4X4_50")
            >>> board = BoardGeometry(config)
            >>> board.num_corners
            35
        """
        pass  # Implementation
    
    @property
    def corner_positions(self) -> dict[int, Vec3]:
        """
        Get 3D positions of all corners in board frame.
        
        Returns:
            Dict mapping corner_id (int) to position (3,) in meters
            
        Example:
            >>> board = BoardGeometry(config)
            >>> pos = board.corner_positions[0]
            >>> pos.shape
            (3,)
        """
        pass  # Implementation
    
    def get_opencv_board(self) -> cv2.aruco.CharucoBoard:
        """
        Get OpenCV CharucoBoard object for detection.
        
        Returns:
            OpenCV CharucoBoard instance
        """
        pass  # Implementation
    
    def transform_corners(self, rvec: Vec3, tvec: Vec3) -> dict[int, Vec3]:
        """
        Transform all corners from board frame to world frame.
        
        Args:
            rvec: Rotation vector (board to world)
            tvec: Translation vector (board to world)
            
        Returns:
            Dict mapping corner_id to 3D position in world frame
            
        Example:
            >>> board = BoardGeometry(config)
            >>> # Identity transform
            >>> world_pts = board.transform_corners(np.zeros(3), np.zeros(3))
            >>> np.allclose(world_pts[0], board.corner_positions[0])
            True
        """
        pass  # Implementation
    
    def get_corner_array(self, corner_ids: NDArray[np.int32]) -> NDArray[np.float64]:
        """
        Get 3D positions for specific corners as array.
        
        Args:
            corner_ids: Array of corner IDs to retrieve
            
        Returns:
            Array of shape (N, 3) with 3D positions in board frame
            
        Example:
            >>> board = BoardGeometry(config)
            >>> pts = board.get_corner_array(np.array([0, 1, 2]))
            >>> pts.shape
            (3, 3)
        """
        pass  # Implementation
```

---

### `core/camera.py`

```python
"""Camera model and projection operations (without refraction)."""

import numpy as np
from numpy.typing import NDArray

from config.schema import CameraIntrinsics, CameraExtrinsics, Vec3, Vec2, Mat3


class Camera:
    """
    Camera model combining intrinsics and extrinsics.
    
    Handles standard pinhole projection with distortion, but NOT refraction.
    For refractive projection, use refractive_geometry module.
    """
    
    def __init__(self, name: str, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics):
        """
        Initialize camera.
        
        Args:
            name: Camera identifier
            intrinsics: Intrinsic parameters
            extrinsics: Extrinsic parameters
        """
        pass  # Implementation
    
    @property
    def K(self) -> Mat3:
        """Intrinsic matrix."""
        pass
    
    @property
    def R(self) -> Mat3:
        """Rotation matrix (world to camera)."""
        pass
    
    @property
    def t(self) -> Vec3:
        """Translation vector."""
        pass
    
    @property
    def C(self) -> Vec3:
        """Camera center in world coordinates."""
        pass
    
    @property
    def P(self) -> NDArray[np.float64]:
        """3x4 projection matrix (without distortion)."""
        pass
    
    def world_to_camera(self, point_world: Vec3) -> Vec3:
        """
        Transform point from world to camera coordinates.
        
        Args:
            point_world: 3D point in world frame
            
        Returns:
            3D point in camera frame
            
        Example:
            >>> cam = Camera(...)  # camera at origin, looking down -Z
            >>> p_world = np.array([0, 0, -5])
            >>> p_cam = cam.world_to_camera(p_world)
            >>> p_cam[2] > 0  # point is in front of camera
            True
        """
        pass  # Implementation
    
    def project(self, point_world: Vec3, apply_distortion: bool = True) -> Vec2:
        """
        Project 3D point to 2D pixel coordinates (standard projection, no refraction).
        
        Args:
            point_world: 3D point in world frame
            apply_distortion: Whether to apply lens distortion
            
        Returns:
            2D pixel coordinates
            
        Example:
            >>> cam = Camera(...)
            >>> pixel = cam.project(np.array([0, 0, 1]))
            >>> pixel.shape
            (2,)
        """
        pass  # Implementation
    
    def pixel_to_ray(self, pixel: Vec2, undistort: bool = True) -> Vec3:
        """
        Back-project pixel to unit ray in camera frame.
        
        Args:
            pixel: 2D pixel coordinates
            undistort: Whether to undistort the pixel first
            
        Returns:
            Unit direction vector in camera frame (Z forward)
            
        Example:
            >>> cam = Camera(...)  # with principal point at (320, 240)
            >>> ray = cam.pixel_to_ray(np.array([320, 240]))
            >>> np.allclose(ray, np.array([0, 0, 1]))
            True
        """
        pass  # Implementation
    
    def pixel_to_ray_world(self, pixel: Vec2, undistort: bool = True) -> tuple[Vec3, Vec3]:
        """
        Back-project pixel to ray in world frame.
        
        Args:
            pixel: 2D pixel coordinates
            undistort: Whether to undistort the pixel first
            
        Returns:
            Tuple of (ray_origin, ray_direction) in world frame.
            ray_origin is the camera center.
            ray_direction is a unit vector.
            
        Example:
            >>> cam = Camera(...)
            >>> origin, direction = cam.pixel_to_ray_world(np.array([320, 240]))
            >>> np.allclose(origin, cam.C)
            True
            >>> np.isclose(np.linalg.norm(direction), 1.0)
            True
        """
        pass  # Implementation


def undistort_points(
    points: NDArray[np.float64],
    K: Mat3,
    dist_coeffs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Undistort pixel coordinates.
    
    Args:
        points: Pixel coordinates, shape (N, 2)
        K: Intrinsic matrix
        dist_coeffs: Distortion coefficients
        
    Returns:
        Undistorted pixel coordinates, shape (N, 2)
        
    Example:
        >>> K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float)
        >>> dist = np.zeros(5)
        >>> pts = np.array([[320, 240], [100, 100]])
        >>> undist = undistort_points(pts, K, dist)
        >>> np.allclose(undist, pts)  # no distortion, should be unchanged
        True
    """
    pass  # Implementation
```

---

### `core/interface.py`

```python
"""Refractive interface (water surface) model."""

import numpy as np
from numpy.typing import NDArray

from config.schema import InterfaceParams, Vec3


class Interface:
    """
    Planar refractive interface (air-water boundary).
    
    The interface is defined by a plane (normal + point) and refractive indices.
    Supports per-camera distance offsets for cameras at different heights.
    """
    
    def __init__(
        self,
        normal: Vec3,
        base_height: float,
        camera_offsets: dict[str, float],
        n_air: float = 1.0,
        n_water: float = 1.333
    ):
        """
        Initialize interface.
        
        Args:
            normal: Unit normal vector pointing from water to air (typically [0,0,1])
            base_height: Base height of interface in world coordinates
            camera_offsets: Per-camera offset from base_height (reference camera should be 0)
            n_air: Refractive index of air
            n_water: Refractive index of water
            
        Example:
            >>> interface = Interface(
            ...     normal=np.array([0, 0, 1]),
            ...     base_height=0.15,
            ...     camera_offsets={'cam0': 0.0, 'cam1': 0.01, 'cam2': -0.005},
            ...     n_air=1.0,
            ...     n_water=1.333
            ... )
        """
        pass  # Implementation
    
    def get_interface_distance(self, camera_name: str) -> float:
        """
        Get distance from camera to interface for a specific camera.
        
        Args:
            camera_name: Name of camera
            
        Returns:
            Distance in meters (base_height + offset for this camera)
        """
        pass  # Implementation
    
    def get_interface_point(self, camera_center: Vec3, camera_name: str) -> Vec3:
        """
        Get the point where the camera's optical axis intersects the interface.
        
        For cameras looking straight down, this is directly below the camera center.
        
        Args:
            camera_center: Camera center in world coordinates
            camera_name: Name of camera (for offset lookup)
            
        Returns:
            3D point on interface plane
        """
        pass  # Implementation
    
    @property
    def n_ratio_air_to_water(self) -> float:
        """Ratio n_air / n_water for Snell's law (air to water)."""
        return self.n_air / self.n_water
    
    @property
    def n_ratio_water_to_air(self) -> float:
        """Ratio n_water / n_air for Snell's law (water to air)."""
        return self.n_water / self.n_air


def ray_plane_intersection(
    ray_origin: Vec3,
    ray_direction: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3
) -> tuple[Vec3, float] | tuple[None, None]:
    """
    Compute intersection of ray with plane.
    
    Args:
        ray_origin: Origin of ray
        ray_direction: Direction of ray (need not be unit)
        plane_point: Any point on the plane
        plane_normal: Normal vector of plane (need not be unit)
        
    Returns:
        Tuple of (intersection_point, t) where intersection = origin + t * direction.
        Returns (None, None) if ray is parallel to plane.
        
    Example:
        >>> origin = np.array([0, 0, 1])
        >>> direction = np.array([0, 0, -1])
        >>> plane_pt = np.array([0, 0, 0])
        >>> plane_n = np.array([0, 0, 1])
        >>> pt, t = ray_plane_intersection(origin, direction, plane_pt, plane_n)
        >>> np.allclose(pt, np.array([0, 0, 0]))
        True
        >>> t
        1.0
    """
    pass  # Implementation
```

---

### `core/refractive_geometry.py`

```python
"""
Core refractive geometry operations.

This module provides ray tracing through the air-water interface using Snell's law.
It is used by both calibration and downstream triangulation.
"""

import numpy as np
from numpy.typing import NDArray

from config.schema import Vec3, Vec2
from core.camera import Camera
from core.interface import Interface, ray_plane_intersection


def snells_law_3d(
    incident_direction: Vec3,
    surface_normal: Vec3,
    n_ratio: float
) -> Vec3 | None:
    """
    Apply Snell's law in 3D to compute refracted ray direction.
    
    Args:
        incident_direction: Unit vector of incoming ray direction
        surface_normal: Unit normal of interface, pointing from medium 1 to medium 2
        n_ratio: Ratio n1/n2 of refractive indices
        
    Returns:
        Unit vector of refracted ray direction, or None if total internal reflection.
        
    Notes:
        - incident_direction should point TOWARD the interface
        - surface_normal should point from the incident medium to the transmitted medium
        - For air-to-water: normal points "up" (from water to air), so negate it internally
        
    Example:
        >>> # Normal incidence: ray passes straight through
        >>> incident = np.array([0, 0, -1])  # pointing down
        >>> normal = np.array([0, 0, 1])     # pointing up
        >>> n_ratio = 1.0 / 1.333
        >>> refracted = snells_law_3d(incident, normal, n_ratio)
        >>> np.allclose(refracted, np.array([0, 0, -1]))
        True
        
        >>> # Oblique incidence
        >>> incident = np.array([0.5, 0, -np.sqrt(0.75)])  # 30° from vertical
        >>> incident = incident / np.linalg.norm(incident)
        >>> refracted = snells_law_3d(incident, normal, n_ratio)
        >>> # Ray should bend toward normal (steeper in water)
        >>> abs(refracted[0]) < abs(incident[0])
        True
    """
    pass  # Implementation


def trace_ray_air_to_water(
    camera: Camera,
    interface: Interface,
    pixel: Vec2
) -> tuple[Vec3, Vec3] | tuple[None, None]:
    """
    Trace ray from camera through air-water interface.
    
    Args:
        camera: Camera object
        interface: Interface object (must have camera in its offsets)
        pixel: 2D pixel coordinates
        
    Returns:
        Tuple of (intersection_point, refracted_direction) where:
        - intersection_point: where ray hits interface (in world coords)
        - refracted_direction: unit direction of ray in water
        Returns (None, None) if ray doesn't hit interface or TIR occurs.
        
    Example:
        >>> cam = Camera(...)
        >>> interface = Interface(...)
        >>> pt, direction = trace_ray_air_to_water(cam, interface, np.array([320, 240]))
        >>> pt[2]  # Z coordinate should be at interface height
        0.15  # approximately
    """
    pass  # Implementation


def refractive_project(
    camera: Camera,
    interface: Interface,
    point_3d: Vec3
) -> Vec2 | None:
    """
    Project 3D point in water to 2D pixel through refractive interface.
    
    This is the FORWARD projection used for computing reprojection error.
    
    Args:
        camera: Camera object
        interface: Interface object
        point_3d: 3D point in water (world coordinates)
        
    Returns:
        2D pixel coordinates, or None if projection fails (point behind camera,
        ray doesn't hit interface, TIR, etc.)
        
    Notes:
        This requires finding where the refracted ray from the 3D point would
        intersect the interface and then refract into air toward the camera.
        It's the inverse of trace_ray_air_to_water, which is NOT analytically
        invertible, so this uses iterative refinement.
        
    Example:
        >>> cam = Camera(...)
        >>> interface = Interface(...)
        >>> point = np.array([0.1, 0.2, -0.3])  # 30cm below interface
        >>> pixel = refractive_project(cam, interface, point)
        >>> pixel.shape
        (2,)
    """
    pass  # Implementation


def refractive_back_project(
    camera: Camera,
    interface: Interface,
    pixel: Vec2
) -> tuple[Vec3, Vec3] | tuple[None, None]:
    """
    Back-project pixel to ray in water.
    
    This is used for triangulation. Returns a ray (origin + direction) in the
    water volume that corresponds to the given pixel.
    
    Args:
        camera: Camera object
        interface: Interface object
        pixel: 2D pixel coordinates
        
    Returns:
        Tuple of (ray_origin, ray_direction) where:
        - ray_origin: point on interface where ray enters water
        - ray_direction: unit direction of ray in water
        Returns (None, None) if back-projection fails.
        
    Example:
        >>> cam = Camera(...)
        >>> interface = Interface(...)
        >>> origin, direction = refractive_back_project(cam, interface, np.array([320, 240]))
        >>> origin[2]  # should be at interface
        0.15
        >>> direction[2] < 0  # should point into water (down)
        True
    """
    pass  # Implementation
```

---

### `io/detection.py`

```python
"""ChArUco detection wrapper."""

import numpy as np
from numpy.typing import NDArray
import cv2

from config.schema import Detection, FrameDetections, DetectionResult
from core.board import BoardGeometry


def detect_charuco(
    image: NDArray[np.uint8],
    board: BoardGeometry,
    camera_matrix: NDArray[np.float64] | None = None,
    dist_coeffs: NDArray[np.float64] | None = None
) -> Detection | None:
    """
    Detect ChArUco corners in a single image.
    
    Args:
        image: Grayscale or BGR image
        board: Board geometry
        camera_matrix: Optional intrinsic matrix for corner refinement
        dist_coeffs: Optional distortion coefficients for corner refinement
        
    Returns:
        Detection object, or None if no corners detected.
        
    Example:
        >>> image = cv2.imread('test.png')
        >>> board = BoardGeometry(config)
        >>> detection = detect_charuco(image, board)
        >>> if detection is not None:
        ...     print(f"Found {detection.num_corners} corners")
    """
    pass  # Implementation


def detect_all_frames(
    video_paths: dict[str, str],
    board: BoardGeometry,
    intrinsics: dict[str, tuple[NDArray, NDArray]] | None = None,
    min_corners: int = 4,
    frame_step: int = 1,
    progress_callback: callable | None = None
) -> DetectionResult:
    """
    Detect ChArUco corners in all frames of synchronized videos.
    
    Args:
        video_paths: Dict mapping camera_name to video file path
        board: Board geometry
        intrinsics: Optional dict mapping camera_name to (K, dist_coeffs)
        min_corners: Minimum corners required to keep a detection
        frame_step: Process every Nth frame (1 = all frames)
        progress_callback: Optional callback(current_frame, total_frames)
        
    Returns:
        DetectionResult containing all valid detections organized by frame and camera.
        
    Example:
        >>> paths = {'cam0': 'video0.mp4', 'cam1': 'video1.mp4'}
        >>> board = BoardGeometry(config)
        >>> detections = detect_all_frames(paths, board, min_corners=8)
        >>> usable_frames = detections.get_frames_with_min_cameras(2)
        >>> print(f"Found {len(usable_frames)} frames with 2+ cameras")
    """
    pass  # Implementation
```

---

### `triangulation/triangulate.py`

```python
"""Refractive triangulation for 3D reconstruction."""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import least_squares

from config.schema import CalibrationResult, Vec3, Vec2
from core.camera import Camera
from core.interface import Interface
from core.refractive_geometry import refractive_back_project


def triangulate_point(
    calibration: CalibrationResult,
    observations: dict[str, Vec2]
) -> Vec3 | None:
    """
    Triangulate a single 3D point from multi-camera observations.
    
    Args:
        calibration: Complete calibration result
        observations: Dict mapping camera_name to 2D pixel coordinates.
                     Must have at least 2 cameras.
                     
    Returns:
        3D point in world coordinates, or None if triangulation fails.
        
    Example:
        >>> calib = load_calibration('calibration.json')
        >>> obs = {'cam0': np.array([320, 240]), 'cam3': np.array([315, 238])}
        >>> point_3d = triangulate_point(calib, obs)
        >>> point_3d[2] < 0  # should be below water surface
        True
    """
    pass  # Implementation


def triangulate_rays(
    rays: list[tuple[Vec3, Vec3]]
) -> Vec3:
    """
    Find 3D point minimizing distance to all rays.
    
    Args:
        rays: List of (origin, direction) tuples. Directions must be unit vectors.
        
    Returns:
        3D point that minimizes sum of squared distances to all rays.
        
    Example:
        >>> # Two rays that should intersect at (0, 0, -1)
        >>> ray1 = (np.array([1, 0, 0]), np.array([-1, 0, -1]) / np.sqrt(2))
        >>> ray2 = (np.array([-1, 0, 0]), np.array([1, 0, -1]) / np.sqrt(2))
        >>> point = triangulate_rays([ray1, ray2])
        >>> np.allclose(point, np.array([0, 0, -1]), atol=1e-6)
        True
    """
    pass  # Implementation


def point_to_ray_distance(point: Vec3, ray_origin: Vec3, ray_direction: Vec3) -> float:
    """
    Compute perpendicular distance from point to ray.
    
    Args:
        point: 3D point
        ray_origin: Origin of ray
        ray_direction: Unit direction of ray
        
    Returns:
        Perpendicular distance from point to ray.
        
    Example:
        >>> point = np.array([1, 0, 0])
        >>> origin = np.array([0, 0, 0])
        >>> direction = np.array([0, 0, 1])
        >>> dist = point_to_ray_distance(point, origin, direction)
        >>> np.isclose(dist, 1.0)
        True
    """
    pass  # Implementation
```

---

## Error Handling Conventions

1. **Invalid input**: Raise `ValueError` with descriptive message
2. **File not found**: Raise `FileNotFoundError`
3. **Calibration failure** (e.g., insufficient data): Raise `CalibrationError` (custom exception)
4. **Geometric impossibility** (e.g., TIR, ray misses plane): Return `None`, don't raise
5. **Convergence failure**: Return result with `success=False` flag, don't raise

```python
# In config/schema.py, add:

class CalibrationError(Exception):
    """Raised when calibration cannot be completed."""
    pass

class InsufficientDataError(CalibrationError):
    """Raised when there isn't enough data for calibration."""
    pass

class ConvergenceError(CalibrationError):
    """Raised when optimization fails to converge."""
    pass

class ConnectivityError(CalibrationError):
    """Raised when pose graph is not connected."""
    pass
```

---

## Testing Requirements

Each module must have corresponding tests in `tests/` directory.

### Required Test Categories

1. **Unit tests**: Test individual functions with known inputs/outputs
2. **Round-trip tests**: Test that inverse operations recover original values
3. **Synthetic tests**: Test full pipelines on synthetic data with known ground truth
4. **Edge cases**: Test boundary conditions (zero rotation, normal incidence, etc.)

### Example Test Structure

```python
# tests/test_refractive_geometry.py

import numpy as np
import pytest
from core.refractive_geometry import snells_law_3d, trace_ray_air_to_water


class TestSnellsLaw:
    """Tests for Snell's law implementation."""
    
    def test_normal_incidence(self):
        """Ray perpendicular to surface should pass straight through."""
        incident = np.array([0, 0, -1])
        normal = np.array([0, 0, 1])
        n_ratio = 1.0 / 1.333
        
        refracted = snells_law_3d(incident, normal, n_ratio)
        
        assert refracted is not None
        np.testing.assert_allclose(refracted, incident, atol=1e-10)
    
    def test_bends_toward_normal_entering_denser_medium(self):
        """Ray entering water (denser) should bend toward normal."""
        incident = np.array([0.5, 0, -np.sqrt(0.75)])
        incident = incident / np.linalg.norm(incident)
        normal = np.array([0, 0, 1])
        n_ratio = 1.0 / 1.333  # air to water
        
        refracted = snells_law_3d(incident, normal, n_ratio)
        
        assert refracted is not None
        # Angle from vertical should be smaller after refraction
        cos_incident = abs(np.dot(incident, normal))
        cos_refracted = abs(np.dot(refracted, normal))
        assert cos_refracted > cos_incident
    
    def test_total_internal_reflection(self):
        """Grazing angle from water to air should cause TIR."""
        # Critical angle for water->air is about 48.6°
        # Use 60° from normal (should cause TIR)
        incident = np.array([np.sin(np.radians(60)), 0, np.cos(np.radians(60))])
        normal = np.array([0, 0, 1])  # pointing up
        n_ratio = 1.333 / 1.0  # water to air
        
        refracted = snells_law_3d(incident, normal, n_ratio)
        
        assert refracted is None
```

---

## File Format Specifications

### Configuration File (`config.yaml`)

```yaml
board:
  squares_x: 8
  squares_y: 6
  square_size: 0.030  # meters
  marker_size: 0.022  # meters
  dictionary: "DICT_4X4_50"

cameras:
  - cam0
  - cam1
  - cam2
  - cam3
  # ... up to cam13

paths:
  intrinsic_videos:
    cam0: "/data/intrinsic/cam0.mp4"
    cam1: "/data/intrinsic/cam1.mp4"
    # ...
  extrinsic_videos:
    cam0: "/data/extrinsic/cam0.mp4"
    cam1: "/data/extrinsic/cam1.mp4"
    # ...
  output_dir: "/data/calibration_output"

interface:
  n_air: 1.0
  n_water: 1.333
  normal_fixed: true  # if false, estimate tilt

optimization:
  robust_loss: "huber"
  loss_scale: 1.0

detection:
  min_corners: 8
  min_cameras: 2
  frame_step: 5  # process every 5th frame

validation:
  holdout_fraction: 0.2
  save_detailed_residuals: true
```

### Calibration Output (`calibration.json` + `calibration_arrays.npz`)

JSON contains metadata and scalar values; NPZ contains arrays.

```json
{
  "metadata": {
    "calibration_date": "2025-02-02T14:30:00",
    "software_version": "0.1.0",
    "config_hash": "abc123...",
    "num_frames_used": 150,
    "num_frames_holdout": 38
  },
  "interface": {
    "normal": [0.0, 0.0, 1.0],
    "n_air": 1.0,
    "n_water": 1.333
  },
  "board": {
    "squares_x": 8,
    "squares_y": 6,
    "square_size": 0.030,
    "marker_size": 0.022,
    "dictionary": "DICT_4X4_50"
  },
  "cameras": {
    "cam0": {
      "image_size": [1920, 1080],
      "interface_distance": 0.152,
      "intrinsics_key": "cam0_intrinsics",
      "extrinsics_key": "cam0_extrinsics"
    },
    "cam1": {
      "image_size": [1920, 1080],
      "interface_distance": 0.148,
      "intrinsics_key": "cam1_intrinsics",
      "extrinsics_key": "cam1_extrinsics"
    }
  },
  "diagnostics": {
    "reprojection_error_rms": 0.42,
    "reprojection_error_per_camera": {
      "cam0": 0.38,
      "cam1": 0.45
    },
    "validation_3d_error_mean": 0.0012,
    "validation_3d_error_std": 0.0008
  }
}
```

```python
# calibration_arrays.npz contains:
# - cam0_intrinsics_K: (3, 3) float64
# - cam0_intrinsics_dist: (5,) float64
# - cam0_extrinsics_R: (3, 3) float64
# - cam0_extrinsics_t: (3,) float64
# - cam1_intrinsics_K: (3, 3) float64
# ... etc
# - per_corner_residuals: (N, 2) float64 (optional)
```

---

## Implementation Notes for Agent

1. **Start each module with imports and type definitions** before implementing functions.

2. **Use NumPy broadcasting** where possible for efficiency, but prioritize correctness first.

3. **OpenCV conventions**: 
   - Image coordinates are (x, y) = (column, row)
   - `cv2.Rodrigues` converts between rotation vectors and matrices
   - Distortion coefficients order: `[k1, k2, p1, p2, k3]`

4. **Coordinate frame conventions**:
   - World frame: Z points up (out of water), X/Y horizontal
   - Camera frame: Z points forward (into scene), X right, Y down
   - Interface normal: [0, 0, 1] means pointing up (from water to air)

5. **Testing as you go**: After implementing each function, write a simple test before moving on.

6. **Avoid premature optimization**: Get it working correctly first, then optimize if needed.
