import numpy as np
import pytest
from aquacal.core.interface_model import Interface, ray_plane_intersection


@pytest.fixture
def simple_interface():
    """Horizontal interface at Z=0.15, cameras cam0 and cam1."""
    return Interface(
        normal=np.array([0, 0, -1]),  # Points up (from water toward air)
        base_height=0.15,
        camera_offsets={'cam0': 0.0, 'cam1': 0.01, 'cam2': -0.005},
        n_air=1.0,
        n_water=1.333
    )


class TestInterfaceInit:
    def test_normal_is_normalized(self):
        """Normal should be normalized even if input isn't unit."""
        interface = Interface(
            normal=np.array([0, 0, -2]),  # Not unit
            base_height=0.1,
            camera_offsets={'cam0': 0.0}
        )
        np.testing.assert_allclose(np.linalg.norm(interface.normal), 1.0)
        np.testing.assert_allclose(interface.normal, np.array([0, 0, -1]))

    def test_stores_parameters(self, simple_interface):
        assert simple_interface.base_height == 0.15
        assert simple_interface.n_air == 1.0
        assert simple_interface.n_water == 1.333


class TestGetInterfaceDistance:
    def test_reference_camera(self, simple_interface):
        """Reference camera (offset=0) gets base_height."""
        assert simple_interface.get_interface_distance('cam0') == 0.15

    def test_positive_offset(self, simple_interface):
        """Positive offset increases distance."""
        assert simple_interface.get_interface_distance('cam1') == 0.16

    def test_negative_offset(self, simple_interface):
        """Negative offset decreases distance."""
        assert simple_interface.get_interface_distance('cam2') == 0.145

    def test_unknown_camera_raises(self, simple_interface):
        """Unknown camera should raise KeyError."""
        with pytest.raises(KeyError):
            simple_interface.get_interface_distance('unknown_cam')


class TestGetInterfacePoint:
    def test_point_xy_from_camera(self, simple_interface):
        """Interface point should have same XY as camera center."""
        camera_center = np.array([1.0, 2.0, 0.0])
        point = simple_interface.get_interface_point(camera_center, 'cam0')
        assert point[0] == 1.0
        assert point[1] == 2.0

    def test_point_z_from_interface(self, simple_interface):
        """Interface point Z should be base_height + offset."""
        camera_center = np.array([1.0, 2.0, 0.0])
        point = simple_interface.get_interface_point(camera_center, 'cam0')
        assert point[2] == 0.15

    def test_camera_z_is_ignored(self, simple_interface):
        """Camera center Z should not affect interface point Z."""
        camera_center_a = np.array([1.0, 2.0, 0.0])
        camera_center_b = np.array([1.0, 2.0, 0.05])  # Different Z
        point_a = simple_interface.get_interface_point(camera_center_a, 'cam0')
        point_b = simple_interface.get_interface_point(camera_center_b, 'cam0')
        np.testing.assert_allclose(point_a, point_b)

    def test_uses_camera_offset(self, simple_interface):
        """Different cameras get different interface Z."""
        camera_center = np.array([0.0, 0.0, 0.0])
        point_cam0 = simple_interface.get_interface_point(camera_center, 'cam0')
        point_cam1 = simple_interface.get_interface_point(camera_center, 'cam1')
        assert point_cam0[2] == 0.15
        assert point_cam1[2] == 0.16


class TestRefractiveIndexRatios:
    def test_air_to_water_ratio(self, simple_interface):
        expected = 1.0 / 1.333
        np.testing.assert_allclose(simple_interface.n_ratio_air_to_water, expected)

    def test_water_to_air_ratio(self, simple_interface):
        expected = 1.333 / 1.0
        np.testing.assert_allclose(simple_interface.n_ratio_water_to_air, expected)


class TestRayPlaneIntersection:
    def test_basic_intersection(self):
        """Ray pointing at plane should intersect."""
        origin = np.array([0.0, 0.0, 1.0])
        direction = np.array([0.0, 0.0, -1.0])
        plane_pt = np.array([0.0, 0.0, 0.0])
        plane_n = np.array([0.0, 0.0, 1.0])

        pt, t = ray_plane_intersection(origin, direction, plane_pt, plane_n)

        np.testing.assert_allclose(pt, np.array([0, 0, 0]))
        assert t == 1.0

    def test_angled_intersection(self):
        """Angled ray should intersect at correct point."""
        origin = np.array([0.0, 0.0, 1.0])
        direction = np.array([1.0, 0.0, -1.0])  # 45 degrees
        plane_pt = np.array([0.0, 0.0, 0.0])
        plane_n = np.array([0.0, 0.0, 1.0])

        pt, t = ray_plane_intersection(origin, direction, plane_pt, plane_n)

        np.testing.assert_allclose(pt, np.array([1, 0, 0]))
        assert t == 1.0

    def test_negative_t_intersection(self):
        """Ray pointing away from plane should still return intersection with negative t."""
        origin = np.array([0.0, 0.0, 1.0])
        direction = np.array([0.0, 0.0, 1.0])  # Pointing away from Z=0 plane
        plane_pt = np.array([0.0, 0.0, 0.0])
        plane_n = np.array([0.0, 0.0, 1.0])

        pt, t = ray_plane_intersection(origin, direction, plane_pt, plane_n)

        np.testing.assert_allclose(pt, np.array([0, 0, 0]))
        assert t == -1.0  # Negative t

    def test_parallel_ray_returns_none(self):
        """Ray parallel to plane should return (None, None)."""
        origin = np.array([0.0, 0.0, 1.0])
        direction = np.array([1.0, 0.0, 0.0])  # Parallel to XY plane
        plane_pt = np.array([0.0, 0.0, 0.0])
        plane_n = np.array([0.0, 0.0, 1.0])

        pt, t = ray_plane_intersection(origin, direction, plane_pt, plane_n)

        assert pt is None
        assert t is None

    def test_non_unit_direction(self):
        """Non-unit direction should work correctly."""
        origin = np.array([0.0, 0.0, 2.0])
        direction = np.array([0.0, 0.0, -2.0])  # Magnitude 2
        plane_pt = np.array([0.0, 0.0, 0.0])
        plane_n = np.array([0.0, 0.0, 1.0])

        pt, t = ray_plane_intersection(origin, direction, plane_pt, plane_n)

        np.testing.assert_allclose(pt, np.array([0, 0, 0]))
        assert t == 1.0  # t=1 means origin + 1*direction = [0,0,0]

    def test_non_unit_normal(self):
        """Non-unit normal should work correctly."""
        origin = np.array([0.0, 0.0, 1.0])
        direction = np.array([0.0, 0.0, -1.0])
        plane_pt = np.array([0.0, 0.0, 0.0])
        plane_n = np.array([0.0, 0.0, 5.0])  # Magnitude 5

        pt, t = ray_plane_intersection(origin, direction, plane_pt, plane_n)

        np.testing.assert_allclose(pt, np.array([0, 0, 0]))
        assert t == 1.0

    def test_offset_plane(self):
        """Plane not at origin should work correctly."""
        origin = np.array([0.0, 0.0, 0.0])
        direction = np.array([0.0, 0.0, 1.0])
        plane_pt = np.array([5.0, 5.0, 0.15])  # Plane at Z=0.15
        plane_n = np.array([0.0, 0.0, 1.0])

        pt, t = ray_plane_intersection(origin, direction, plane_pt, plane_n)

        np.testing.assert_allclose(pt, np.array([0, 0, 0.15]))
        np.testing.assert_allclose(t, 0.15)
