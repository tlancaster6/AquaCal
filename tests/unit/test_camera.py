"""Unit tests for camera model (core/camera.py)."""

import numpy as np
import pytest

from aquacal.config.schema import CameraIntrinsics, CameraExtrinsics
from aquacal.core.camera import Camera, undistort_points


@pytest.fixture
def simple_camera():
    """Camera at origin, looking down +Z, 640x480, f=500."""
    intrinsics = CameraIntrinsics(
        K=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64),
        dist_coeffs=np.zeros(5),
        image_size=(640, 480),
    )
    extrinsics = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))
    return Camera("test_cam", intrinsics, extrinsics)


class TestCameraProperties:
    def test_camera_center_at_origin(self, simple_camera):
        np.testing.assert_allclose(simple_camera.C, np.zeros(3))

    def test_projection_matrix_shape(self, simple_camera):
        assert simple_camera.P.shape == (3, 4)

    def test_projection_matrix_values(self, simple_camera):
        # P = K @ [R | t] = K @ [I | 0] = [K | 0]
        expected = np.hstack([simple_camera.K, np.zeros((3, 1))])
        np.testing.assert_allclose(simple_camera.P, expected)


class TestWorldToCamera:
    def test_origin_stays_at_origin(self, simple_camera):
        p_cam = simple_camera.world_to_camera(np.zeros(3))
        np.testing.assert_allclose(p_cam, np.zeros(3))

    def test_point_in_front(self, simple_camera):
        p_world = np.array([0, 0, 5])
        p_cam = simple_camera.world_to_camera(p_world)
        np.testing.assert_allclose(p_cam, np.array([0, 0, 5]))


class TestProject:
    def test_principal_point(self, simple_camera):
        """Point on optical axis projects to principal point."""
        pixel = simple_camera.project(np.array([0, 0, 1]))
        np.testing.assert_allclose(pixel, np.array([320, 240]), atol=1e-10)

    def test_point_behind_camera_returns_none(self, simple_camera):
        """Point with Z <= 0 in camera frame returns None."""
        assert simple_camera.project(np.array([0, 0, -1])) is None
        assert simple_camera.project(np.array([0, 0, 0])) is None

    def test_project_offset_point(self, simple_camera):
        """Point offset from axis projects away from principal point."""
        # Point at (1, 0, 1) should project to (500 + 320, 240) = (820, 240)
        # But that's outside image - still valid projection
        pixel = simple_camera.project(np.array([1, 0, 1]))
        np.testing.assert_allclose(pixel, np.array([820, 240]), atol=1e-10)

    def test_project_no_distortion(self, simple_camera):
        """Project without distortion flag."""
        pixel = simple_camera.project(np.array([0, 0, 1]), apply_distortion=False)
        np.testing.assert_allclose(pixel, np.array([320, 240]), atol=1e-10)


class TestPixelToRay:
    def test_principal_point_ray(self, simple_camera):
        """Principal point should give ray along +Z."""
        ray = simple_camera.pixel_to_ray(np.array([320, 240]))
        np.testing.assert_allclose(ray, np.array([0, 0, 1]), atol=1e-10)

    def test_ray_is_unit(self, simple_camera):
        """All rays should be unit vectors."""
        ray = simple_camera.pixel_to_ray(np.array([100, 200]))
        np.testing.assert_allclose(np.linalg.norm(ray), 1.0)


class TestPixelToRayWorld:
    def test_ray_origin_is_camera_center(self, simple_camera):
        origin, direction = simple_camera.pixel_to_ray_world(np.array([320, 240]))
        np.testing.assert_allclose(origin, simple_camera.C)

    def test_ray_direction_is_unit(self, simple_camera):
        origin, direction = simple_camera.pixel_to_ray_world(np.array([100, 200]))
        np.testing.assert_allclose(np.linalg.norm(direction), 1.0)


class TestProjectBackprojectRoundTrip:
    def test_round_trip_no_distortion(self, simple_camera):
        """project -> pixel_to_ray_world should recover original ray direction."""
        p_world = np.array([0.5, -0.3, 2.0])
        pixel = simple_camera.project(p_world, apply_distortion=False)
        origin, direction = simple_camera.pixel_to_ray_world(pixel, undistort=False)

        # Ray from camera through p_world
        expected_dir = p_world - origin
        expected_dir = expected_dir / np.linalg.norm(expected_dir)
        np.testing.assert_allclose(direction, expected_dir, atol=1e-10)


class TestUndistortPoints:
    def test_no_distortion_unchanged(self):
        """With zero distortion, points should be unchanged."""
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5)
        pts = np.array([[320, 240], [100, 100], [500, 400]], dtype=np.float64)

        undist = undistort_points(pts, K, dist)
        np.testing.assert_allclose(undist, pts, atol=1e-10)

    def test_output_shape(self):
        """Output shape should match input."""
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5)
        pts = np.array([[100, 200], [300, 400]], dtype=np.float64)

        undist = undistort_points(pts, K, dist)
        assert undist.shape == (2, 2)
