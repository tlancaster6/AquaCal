"""Camera model and projection operations (without refraction)."""

import cv2
import numpy as np
from numpy.typing import NDArray

from aquacal.config.schema import CameraExtrinsics, CameraIntrinsics, Mat3, Vec2, Vec3


class Camera:
    """
    Camera model combining intrinsics and extrinsics.

    Handles standard pinhole projection with distortion, but NOT refraction.
    For refractive projection, use refractive_geometry module.

    Attributes:
        name: Camera identifier string
        intrinsics: CameraIntrinsics dataclass
        extrinsics: CameraExtrinsics dataclass

    Example:
        >>> from aquacal.core.camera import Camera
        >>> from aquacal.config.schema import CameraIntrinsics, CameraExtrinsics
        >>> import numpy as np
        >>> # Create camera with intrinsics and extrinsics
        >>> camera = Camera("cam1", intrinsics, extrinsics)
        >>> point_3d = np.array([1.0, 0.5, 2.0])
        >>> pixel = camera.project(point_3d)

    Note:
        For coordinate system conventions, see the
        :doc:`Coordinate Conventions </guide/coordinates>` guide.
    """

    def __init__(
        self, name: str, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics
    ):
        """
        Initialize camera.

        Args:
            name: Camera identifier
            intrinsics: Intrinsic parameters
            extrinsics: Extrinsic parameters
        """
        self.name = name
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics

    @property
    def K(self) -> Mat3:
        """3x3 intrinsic matrix."""
        return self.intrinsics.K

    @property
    def dist_coeffs(self) -> NDArray[np.float64]:
        """Distortion coefficients."""
        return self.intrinsics.dist_coeffs

    @property
    def R(self) -> Mat3:
        """3x3 rotation matrix (world to camera)."""
        return self.extrinsics.R

    @property
    def t(self) -> Vec3:
        """Translation vector."""
        return self.extrinsics.t

    @property
    def C(self) -> Vec3:
        """Camera center in world coordinates. Delegates to extrinsics.C."""
        return self.extrinsics.C

    @property
    def image_size(self) -> tuple[int, int]:
        """Image size as (width, height)."""
        return self.intrinsics.image_size

    @property
    def P(self) -> NDArray[np.float64]:
        """
        3x4 projection matrix (without distortion).

        P = K @ [R | t]

        Note: This is the ideal pinhole projection. For projection with
        distortion, use the project() method.
        """
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return self.K @ Rt

    def world_to_camera(self, point_world: Vec3) -> Vec3:
        """
        Transform point from world to camera coordinates.

        Args:
            point_world: 3D point in world frame, shape (3,)

        Returns:
            3D point in camera frame, shape (3,)

        Formula: p_cam = R @ p_world + t
        """
        return self.R @ point_world + self.t

    def project(self, point_world: Vec3, apply_distortion: bool = True) -> Vec2 | None:
        """
        Project 3D world point to 2D pixel coordinates.

        Args:
            point_world: 3D point in world frame, shape (3,)
            apply_distortion: If True, apply lens distortion. Default True.

        Returns:
            2D pixel coordinates shape (2,), or None if point is behind camera.

        Notes:
            - Returns None if point's Z coordinate in camera frame is ≤ 0
            - Uses cv2.projectPoints when apply_distortion=True
            - Uses direct K @ (p_cam / p_cam[2]) when apply_distortion=False
        """
        p_cam = self.world_to_camera(point_world)

        # Point behind camera
        if p_cam[2] <= 0:
            return None

        if apply_distortion:
            # cv2.projectPoints expects object points and rvec/tvec
            # Use identity transform since point is already in camera frame
            pts, _ = cv2.projectPoints(
                p_cam.reshape(1, 1, 3),
                np.zeros(3),  # rvec = identity
                np.zeros(3),  # tvec = zero
                self.K,
                self.dist_coeffs,
            )
            return pts.reshape(2).astype(np.float64)
        else:
            # Ideal pinhole projection (no distortion)
            p_normalized = p_cam[:2] / p_cam[2]
            pixel = self.K[:2, :2] @ p_normalized + self.K[:2, 2]
            return pixel.astype(np.float64)

    def pixel_to_ray(self, pixel: Vec2, undistort: bool = True) -> Vec3:
        """
        Back-project pixel to unit ray in camera frame.

        Args:
            pixel: 2D pixel coordinates, shape (2,)
            undistort: If True, undistort pixel first. Default True.

        Returns:
            Unit direction vector in camera frame (Z forward), shape (3,)

        Notes:
            - Principal point maps to ray [0, 0, 1]
            - Uses cv2.undistortPoints when undistort=True
        """
        if undistort:
            # cv2.undistortPoints returns normalized coordinates
            # Ensure input is float64 and properly shaped
            pixel_input = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
            pts_undist = cv2.undistortPoints(pixel_input, self.K, self.dist_coeffs)
            # pts_undist is in normalized camera coordinates (x/z, y/z)
            x_norm, y_norm = pts_undist.reshape(2)
        else:
            # Convert pixel to normalized coordinates manually
            # [x_norm, y_norm, 1]^T = K^{-1} @ [u, v, 1]^T
            pixel_h = np.array([pixel[0], pixel[1], 1.0])
            K_inv = np.linalg.inv(self.K)
            p_norm = K_inv @ pixel_h
            x_norm, y_norm = p_norm[0], p_norm[1]

        # Create direction vector and normalize
        direction = np.array([x_norm, y_norm, 1.0])
        return direction / np.linalg.norm(direction)

    def pixel_to_ray_world(
        self, pixel: Vec2, undistort: bool = True
    ) -> tuple[Vec3, Vec3]:
        """
        Back-project pixel to ray in world frame.

        Args:
            pixel: 2D pixel coordinates, shape (2,)
            undistort: If True, undistort pixel first. Default True.

        Returns:
            Tuple of (ray_origin, ray_direction):
            - ray_origin: Camera center in world frame, shape (3,)
            - ray_direction: Unit direction vector in world frame, shape (3,)
        """
        ray_cam = self.pixel_to_ray(pixel, undistort)
        # Transform direction from camera to world frame
        # R transforms world->camera, so R.T transforms camera->world
        ray_world = self.R.T @ ray_cam
        return self.C, ray_world


class FisheyeCamera(Camera):
    """
    Fisheye (equidistant) camera model.

    Overrides projection and back-projection to use OpenCV's fisheye module.
    The equidistant model is appropriate for wide-angle lenses where the
    standard pinhole + polynomial distortion model fails.

    Attributes:
        name: Camera identifier string
        intrinsics: CameraIntrinsics dataclass (must have is_fisheye=True, 4 dist coeffs)
        extrinsics: CameraExtrinsics dataclass
    """

    def project(self, point_world: Vec3, apply_distortion: bool = True) -> Vec2 | None:
        """
        Project 3D world point to 2D pixel using fisheye model.

        Args:
            point_world: 3D point in world frame, shape (3,)
            apply_distortion: If True, apply fisheye distortion. Default True.

        Returns:
            2D pixel coordinates shape (2,), or None if point is behind camera.
        """
        p_cam = self.world_to_camera(point_world)

        if p_cam[2] <= 0:
            return None

        if apply_distortion:
            # cv2.fisheye.projectPoints expects D as (4, 1)
            D = self.dist_coeffs.reshape(4, 1)
            pts, _ = cv2.fisheye.projectPoints(
                p_cam.reshape(1, 1, 3),
                np.zeros(3),  # rvec = identity
                np.zeros(3),  # tvec = zero
                self.K,
                D,
            )
            return pts.reshape(2).astype(np.float64)
        else:
            # Ideal pinhole projection (no distortion) - same as base class
            p_normalized = p_cam[:2] / p_cam[2]
            pixel = self.K[:2, :2] @ p_normalized + self.K[:2, 2]
            return pixel.astype(np.float64)

    def pixel_to_ray(self, pixel: Vec2, undistort: bool = True) -> Vec3:
        """
        Back-project pixel to unit ray in camera frame using fisheye model.

        Args:
            pixel: 2D pixel coordinates, shape (2,)
            undistort: If True, undistort pixel first. Default True.

        Returns:
            Unit direction vector in camera frame (Z forward), shape (3,)
        """
        if undistort:
            pixel_input = np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2)
            D = self.dist_coeffs.reshape(4, 1)
            pts_undist = cv2.fisheye.undistortPoints(pixel_input, K=self.K, D=D)
            x_norm, y_norm = pts_undist.reshape(2)
        else:
            pixel_h = np.array([pixel[0], pixel[1], 1.0])
            K_inv = np.linalg.inv(self.K)
            p_norm = K_inv @ pixel_h
            x_norm, y_norm = p_norm[0], p_norm[1]

        direction = np.array([x_norm, y_norm, 1.0])
        return direction / np.linalg.norm(direction)


def create_camera(
    name: str, intrinsics: CameraIntrinsics, extrinsics: CameraExtrinsics
) -> Camera:
    """Create Camera or FisheyeCamera based on intrinsics.is_fisheye.

    Args:
        name: Camera identifier
        intrinsics: Intrinsic parameters
        extrinsics: Extrinsic parameters

    Returns:
        FisheyeCamera if intrinsics.is_fisheye is True, Camera otherwise.
    """
    if intrinsics.is_fisheye:
        return FisheyeCamera(name, intrinsics, extrinsics)
    return Camera(name, intrinsics, extrinsics)


def undistort_points(
    points: NDArray[np.float64], K: Mat3, dist_coeffs: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Undistort pixel coordinates.

    Args:
        points: Pixel coordinates, shape (N, 2)
        K: 3x3 intrinsic matrix
        dist_coeffs: Distortion coefficients (any valid OpenCV length)

    Returns:
        Undistorted pixel coordinates, shape (N, 2)

    Notes:
        - Thin wrapper around cv2.undistortPoints
        - Output is in pixel coordinates (not normalized), same as input
    """
    # cv2.undistortPoints returns normalized coords, need to re-project to pixels
    pts_normalized = cv2.undistortPoints(
        points.reshape(-1, 1, 2).astype(np.float64), K, dist_coeffs
    )
    # Re-project to pixel coordinates using K
    # [u, v]^T = K[:2,:2] @ [x_norm, y_norm]^T + K[:2, 2]
    pts_normalized = pts_normalized.reshape(-1, 2)
    undistorted = (K[:2, :2] @ pts_normalized.T).T + K[:2, 2]
    return undistorted
