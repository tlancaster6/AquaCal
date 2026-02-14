"""Tests for refractive triangulation."""

import numpy as np
import pytest

from aquacal.config.schema import (
    BoardConfig,
    CalibrationMetadata,
    CalibrationResult,
    CameraCalibration,
    CameraExtrinsics,
    CameraIntrinsics,
    DiagnosticsData,
    InterfaceParams,
)
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_project
from aquacal.triangulation.triangulate import (
    point_to_ray_distance,
    triangulate_point,
    triangulate_rays,
)


@pytest.fixture
def board_config() -> BoardConfig:
    return BoardConfig(
        squares_x=6,
        squares_y=5,
        square_size=0.04,
        marker_size=0.03,
        dictionary="DICT_4X4_50",
    )


@pytest.fixture
def intrinsics() -> dict[str, CameraIntrinsics]:
    """Intrinsics for 3 cameras."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return {
        "cam0": CameraIntrinsics(
            K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)
        ),
        "cam1": CameraIntrinsics(
            K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)
        ),
        "cam2": CameraIntrinsics(
            K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)
        ),
    }


@pytest.fixture
def extrinsics() -> dict[str, CameraExtrinsics]:
    """Camera extrinsics for 3 cameras in a stereo-like configuration."""
    return {
        "cam0": CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.zeros(3, dtype=np.float64),
        ),
        "cam1": CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.3, 0.0, 0.0], dtype=np.float64),  # 30cm to the right
        ),
        "cam2": CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.0, 0.3, 0.0], dtype=np.float64),  # 30cm down
        ),
    }


@pytest.fixture
def interface_distances() -> dict[str, float]:
    """Interface distances for 3 cameras (all same Z since cameras at same height)."""
    return {"cam0": 0.15, "cam1": 0.15, "cam2": 0.15}


@pytest.fixture
def calibration_result(
    board_config, intrinsics, extrinsics, interface_distances
) -> CalibrationResult:
    """Build a complete CalibrationResult for testing."""
    cameras = {}
    for cam_name in intrinsics:
        cameras[cam_name] = CameraCalibration(
            name=cam_name,
            intrinsics=intrinsics[cam_name],
            extrinsics=extrinsics[cam_name],
            interface_distance=interface_distances[cam_name],
        )

    return CalibrationResult(
        cameras=cameras,
        interface=InterfaceParams(
            normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            n_air=1.0,
            n_water=1.333,
        ),
        board=board_config,
        diagnostics=DiagnosticsData(
            reprojection_error_rms=0.0,
            reprojection_error_per_camera={},
            validation_3d_error_mean=0.0,
            validation_3d_error_std=0.0,
        ),
        metadata=CalibrationMetadata(
            calibration_date="2026-02-04",
            software_version="0.1.0",
            config_hash="test",
            num_frames_used=5,
            num_frames_holdout=0,
        ),
    )


class TestPointToRayDistance:
    """Tests for point_to_ray_distance function."""

    def test_perpendicular_distance(self):
        """Test perpendicular distance from point to ray."""
        # Point at (1, 0, 0), ray along Z-axis from origin
        point = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        ray_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ray_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        distance = point_to_ray_distance(point, ray_origin, ray_direction)
        assert np.isclose(distance, 1.0)

    def test_point_on_ray(self):
        """Test that point on ray has zero distance."""
        # Point on ray should have distance 0
        point = np.array([0.0, 0.0, 5.0], dtype=np.float64)
        ray_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ray_direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        distance = point_to_ray_distance(point, ray_origin, ray_direction)
        assert np.isclose(distance, 0.0)

    def test_diagonal_ray(self):
        """Test distance to diagonal ray."""
        # Ray along (1, 1, 1) direction
        point = np.array([2.0, 0.0, 0.0], dtype=np.float64)
        ray_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ray_direction = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        ray_direction = ray_direction / np.linalg.norm(ray_direction)

        distance = point_to_ray_distance(point, ray_origin, ray_direction)
        # Expected: perpendicular distance from (2,0,0) to line through origin along (1,1,1)
        # Projection of (2,0,0) onto (1,1,1) is (2/3, 2/3, 2/3)
        # Perpendicular component is (2,0,0) - (2/3, 2/3, 2/3) = (4/3, -2/3, -2/3)
        # Distance is sqrt((4/3)^2 + (-2/3)^2 + (-2/3)^2) = sqrt(24/9) = sqrt(8/3)
        expected = np.sqrt(8.0 / 3.0)
        assert np.isclose(distance, expected)

    def test_non_unit_direction(self):
        """Test that function works with non-unit direction vectors."""
        # Same as test_perpendicular_distance but with non-unit direction
        point = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        ray_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ray_direction = np.array([0.0, 0.0, 2.0], dtype=np.float64)  # Not unit

        distance = point_to_ray_distance(point, ray_origin, ray_direction)
        # Should still work correctly
        assert np.isclose(distance, 1.0)


class TestTriangulateRays:
    """Tests for triangulate_rays function."""

    def test_two_intersecting_rays(self):
        """Test triangulation with two rays that intersect exactly."""
        # Two rays that intersect at (1, 2, 3)
        ray1_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ray1_direction = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ray1_direction = ray1_direction / np.linalg.norm(ray1_direction)

        ray2_origin = np.array([2.0, 0.0, 0.0], dtype=np.float64)
        ray2_direction = np.array([-1.0, 2.0, 3.0], dtype=np.float64)
        ray2_direction = ray2_direction / np.linalg.norm(ray2_direction)

        rays = [(ray1_origin, ray1_direction), (ray2_origin, ray2_direction)]

        result = triangulate_rays(rays)

        # Result should be close to intersection point
        expected = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert np.allclose(result, expected, atol=1e-6)

    def test_three_rays_noisy(self):
        """Test triangulation with three rays that don't perfectly intersect."""
        # Three rays pointing towards roughly the same point but slightly offset
        target = np.array([1.0, 1.0, 2.0], dtype=np.float64)

        ray1_origin = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        ray1_direction = (target - ray1_origin) / np.linalg.norm(target - ray1_origin)

        ray2_origin = np.array([2.0, 0.0, 0.1], dtype=np.float64)
        ray2_direction = (target - ray2_origin) / np.linalg.norm(target - ray2_origin)

        ray3_origin = np.array([0.0, 2.0, -0.1], dtype=np.float64)
        ray3_direction = (target - ray3_origin) / np.linalg.norm(target - ray3_origin)

        rays = [
            (ray1_origin, ray1_direction),
            (ray2_origin, ray2_direction),
            (ray3_origin, ray3_direction),
        ]

        result = triangulate_rays(rays)

        # Result should be close to target point
        assert np.allclose(result, target, atol=0.2)

    def test_raises_for_single_ray(self):
        """Test that ValueError is raised for single ray."""
        ray = (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        with pytest.raises(ValueError, match="at least 2 rays"):
            triangulate_rays([ray])

    def test_raises_for_parallel_rays(self):
        """Test that ValueError is raised for parallel rays (degenerate)."""
        # Two parallel rays - degenerate configuration
        ray1 = (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))
        ray2 = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))

        with pytest.raises(ValueError, match="Degenerate"):
            triangulate_rays([ray1, ray2])


class TestTriangulatePoint:
    """Tests for triangulate_point function."""

    def test_synthetic_point_reconstruction(
        self, calibration_result, intrinsics, extrinsics, interface_distances
    ):
        """Test triangulation of a known 3D point from synthetic observations."""
        # Known 3D point underwater
        point_3d = np.array([0.1, 0.05, 0.35], dtype=np.float64)

        # Create shared interface with all cameras
        interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        interface = Interface(
            normal=interface_normal,
            camera_distances=interface_distances,  # All cameras
            n_air=1.0,
            n_water=1.333,
        )

        # Project to each camera to get pixel observations
        observations = {}
        for cam_name in intrinsics:
            camera = Camera(cam_name, intrinsics[cam_name], extrinsics[cam_name])
            projected = refractive_project(camera, interface, point_3d)
            if projected is not None:
                observations[cam_name] = projected

        # Triangulate from observations
        result = triangulate_point(calibration_result, observations)

        # Result should match original point
        assert result is not None
        error = np.linalg.norm(result - point_3d)
        assert error < 1e-6  # Sub-micrometer accuracy expected

    def test_returns_none_single_observation(self, calibration_result):
        """Test that None is returned with only one observation."""
        observations = {"cam0": np.array([320.0, 240.0], dtype=np.float64)}

        result = triangulate_point(calibration_result, observations)
        assert result is None

    def test_returns_none_invalid_camera(self, calibration_result):
        """Test graceful handling when camera not in calibration."""
        observations = {
            "invalid_cam": np.array([320.0, 240.0], dtype=np.float64),
            "another_invalid": np.array([400.0, 300.0], dtype=np.float64),
        }

        result = triangulate_point(calibration_result, observations)
        assert result is None

    def test_returns_none_mixed_valid_invalid_cameras(self, calibration_result):
        """Test handling when only one camera is valid."""
        observations = {
            "cam0": np.array([320.0, 240.0], dtype=np.float64),
            "invalid_cam": np.array([400.0, 300.0], dtype=np.float64),
        }

        # Only one valid camera, so should return None
        result = triangulate_point(calibration_result, observations)
        assert result is None

    def test_round_trip_multiple_points(
        self, calibration_result, intrinsics, extrinsics, interface_distances
    ):
        """Test round-trip projection and triangulation for multiple points."""
        # Multiple 3D points underwater at different depths and positions
        test_points = [
            np.array([0.0, 0.0, 0.35], dtype=np.float64),
            np.array([0.1, 0.05, 0.30], dtype=np.float64),
            np.array([-0.05, 0.08, 0.40], dtype=np.float64),
            np.array([0.08, -0.06, 0.32], dtype=np.float64),
        ]

        interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        # Create shared interface with all cameras
        interface = Interface(
            normal=interface_normal,
            camera_distances=interface_distances,  # All cameras
            n_air=1.0,
            n_water=1.333,
        )

        for point_3d in test_points:
            # Project to all cameras
            observations = {}
            for cam_name in intrinsics:
                camera = Camera(cam_name, intrinsics[cam_name], extrinsics[cam_name])
                projected = refractive_project(camera, interface, point_3d)
                if projected is not None:
                    observations[cam_name] = projected

            # Triangulate
            result = triangulate_point(calibration_result, observations)

            # Verify reconstruction accuracy
            assert result is not None
            error = np.linalg.norm(result - point_3d)
            assert error < 1e-6  # Sub-micrometer accuracy expected

    def test_handles_back_projection_failures(
        self, calibration_result, intrinsics, extrinsics, interface_distances
    ):
        """Test graceful handling when refractive_back_project fails."""
        # Use observations from point above interface (may cause back-projection failure)
        # Or use pixels far outside the image that would fail geometric constraints
        observations = {
            "cam0": np.array([320.0, 100.0], dtype=np.float64),
            "cam1": np.array([320.0, 100.0], dtype=np.float64),
        }

        # This might return None if back-projection fails, but shouldn't crash
        result = triangulate_point(calibration_result, observations)
        # We accept either a valid point or None, just ensure no crash
        # (Result depends on whether the rays can be back-projected)
        assert result is None or isinstance(result, np.ndarray)

    def test_with_two_cameras_only(
        self, calibration_result, intrinsics, extrinsics, interface_distances
    ):
        """Test triangulation with exactly two cameras."""
        point_3d = np.array([0.05, 0.03, 0.33], dtype=np.float64)
        interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        # Create shared interface with all cameras
        interface = Interface(
            normal=interface_normal,
            camera_distances=interface_distances,  # All cameras
            n_air=1.0,
            n_water=1.333,
        )

        # Use only cam0 and cam1
        observations = {}
        for cam_name in ["cam0", "cam1"]:
            camera = Camera(cam_name, intrinsics[cam_name], extrinsics[cam_name])
            projected = refractive_project(camera, interface, point_3d)
            if projected is not None:
                observations[cam_name] = projected

        result = triangulate_point(calibration_result, observations)

        # Should work with just two cameras
        assert result is not None
        error = np.linalg.norm(result - point_3d)
        assert error < 1e-6  # Sub-micrometer accuracy expected

    def test_empty_observations(self, calibration_result):
        """Test handling of empty observations dict."""
        observations = {}
        result = triangulate_point(calibration_result, observations)
        assert result is None
