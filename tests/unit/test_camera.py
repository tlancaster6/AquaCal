"""Unit tests for camera model (core/camera.py)."""

import numpy as np
import pytest

from aquacal.config.schema import CameraIntrinsics, CameraExtrinsics
from aquacal.core.camera import Camera, FisheyeCamera, create_camera, undistort_points


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


# --- FisheyeCamera Tests ---

@pytest.fixture
def fisheye_intrinsics():
    """Fisheye intrinsics with zero distortion for predictable testing."""
    return CameraIntrinsics(
        K=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64),
        dist_coeffs=np.zeros(4, dtype=np.float64),
        image_size=(640, 480),
        is_fisheye=True,
    )


@pytest.fixture
def fisheye_camera(fisheye_intrinsics):
    """FisheyeCamera at origin, looking down +Z."""
    extrinsics = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))
    return FisheyeCamera("fisheye_cam", fisheye_intrinsics, extrinsics)


@pytest.fixture
def fisheye_intrinsics_with_distortion():
    """Fisheye intrinsics with non-zero distortion."""
    return CameraIntrinsics(
        K=np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64),
        dist_coeffs=np.array([0.05, -0.01, 0.002, -0.001], dtype=np.float64),
        image_size=(640, 480),
        is_fisheye=True,
    )


@pytest.fixture
def fisheye_camera_distorted(fisheye_intrinsics_with_distortion):
    """FisheyeCamera with non-zero distortion."""
    extrinsics = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))
    return FisheyeCamera("fisheye_dist", fisheye_intrinsics_with_distortion, extrinsics)


class TestFisheyeCameraProject:
    def test_principal_point(self, fisheye_camera):
        """Point on optical axis projects to principal point."""
        pixel = fisheye_camera.project(np.array([0, 0, 1.0]))
        np.testing.assert_allclose(pixel, np.array([320, 240]), atol=1e-6)

    def test_point_behind_camera_returns_none(self, fisheye_camera):
        """Point with Z <= 0 returns None."""
        assert fisheye_camera.project(np.array([0, 0, -1.0])) is None
        assert fisheye_camera.project(np.array([0, 0, 0.0])) is None

    def test_project_no_distortion(self, fisheye_camera):
        """No-distortion path uses ideal pinhole."""
        pixel = fisheye_camera.project(np.array([0, 0, 1.0]), apply_distortion=False)
        np.testing.assert_allclose(pixel, np.array([320, 240]), atol=1e-10)

    def test_project_offset_no_distortion(self, fisheye_camera):
        """Offset point without distortion matches pinhole."""
        pixel = fisheye_camera.project(np.array([1, 0, 1.0]), apply_distortion=False)
        np.testing.assert_allclose(pixel, np.array([820, 240]), atol=1e-10)


class TestFisheyeCameraPixelToRay:
    def test_principal_point_ray(self, fisheye_camera):
        """Principal point gives ray along +Z."""
        ray = fisheye_camera.pixel_to_ray(np.array([320.0, 240.0]))
        np.testing.assert_allclose(ray, np.array([0, 0, 1]), atol=1e-6)

    def test_ray_is_unit(self, fisheye_camera):
        """All rays should be unit vectors."""
        ray = fisheye_camera.pixel_to_ray(np.array([100.0, 200.0]))
        np.testing.assert_allclose(np.linalg.norm(ray), 1.0, atol=1e-10)

    def test_no_undistort_matches_pinhole(self, fisheye_camera):
        """Without undistortion, behaves like pinhole."""
        ray = fisheye_camera.pixel_to_ray(np.array([320.0, 240.0]), undistort=False)
        np.testing.assert_allclose(ray, np.array([0, 0, 1]), atol=1e-10)


class TestFisheyeRoundTrip:
    def test_round_trip_zero_distortion(self, fisheye_camera):
        """project -> pixel_to_ray recovers original ray direction (zero distortion)."""
        p_world = np.array([0.3, -0.2, 2.0])
        pixel = fisheye_camera.project(p_world, apply_distortion=True)
        assert pixel is not None

        ray = fisheye_camera.pixel_to_ray(pixel, undistort=True)

        # Expected: unit ray from origin to p_world
        expected_dir = p_world / np.linalg.norm(p_world)
        np.testing.assert_allclose(ray, expected_dir, atol=1e-6)

    def test_round_trip_with_distortion(self, fisheye_camera_distorted):
        """project -> pixel_to_ray recovers ray direction with non-zero distortion."""
        p_world = np.array([0.3, -0.2, 2.0])
        pixel = fisheye_camera_distorted.project(p_world, apply_distortion=True)
        assert pixel is not None

        ray = fisheye_camera_distorted.pixel_to_ray(pixel, undistort=True)

        expected_dir = p_world / np.linalg.norm(p_world)
        np.testing.assert_allclose(ray, expected_dir, atol=1e-6)

    def test_round_trip_wide_angle(self, fisheye_camera_distorted):
        """Roundtrip works for wide-angle rays (large off-axis angles)."""
        # Point well off-axis
        p_world = np.array([1.5, 1.0, 2.0])
        pixel = fisheye_camera_distorted.project(p_world, apply_distortion=True)
        assert pixel is not None

        ray = fisheye_camera_distorted.pixel_to_ray(pixel, undistort=True)

        expected_dir = p_world / np.linalg.norm(p_world)
        np.testing.assert_allclose(ray, expected_dir, atol=1e-6)


class TestCreateCamera:
    def test_returns_camera_for_pinhole(self):
        """create_camera() returns Camera for non-fisheye intrinsics."""
        intrinsics = CameraIntrinsics(
            K=np.eye(3), dist_coeffs=np.zeros(5), image_size=(640, 480),
        )
        extrinsics = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))
        cam = create_camera("test", intrinsics, extrinsics)
        assert type(cam) is Camera

    def test_returns_fisheye_for_fisheye(self):
        """create_camera() returns FisheyeCamera for fisheye intrinsics."""
        intrinsics = CameraIntrinsics(
            K=np.eye(3), dist_coeffs=np.zeros(4), image_size=(640, 480),
            is_fisheye=True,
        )
        extrinsics = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))
        cam = create_camera("test", intrinsics, extrinsics)
        assert type(cam) is FisheyeCamera

    def test_fisheye_is_subclass_of_camera(self):
        """FisheyeCamera is a Camera (isinstance check)."""
        intrinsics = CameraIntrinsics(
            K=np.eye(3), dist_coeffs=np.zeros(4), image_size=(640, 480),
            is_fisheye=True,
        )
        extrinsics = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))
        cam = create_camera("test", intrinsics, extrinsics)
        assert isinstance(cam, Camera)
