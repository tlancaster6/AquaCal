"""Tests for rotation and transform utilities."""

import numpy as np
import pytest

from aquacal.utils.transforms import (
    rvec_to_matrix,
    matrix_to_rvec,
    compose_poses,
    invert_pose,
    camera_center,
)


class TestRvecToMatrix:
    """Tests for rvec_to_matrix function."""

    def test_90_degree_rotation_around_z(self):
        """90° rotation around Z should map X to Y."""
        rvec = np.array([0.0, 0.0, np.pi / 2])
        R = rvec_to_matrix(rvec)
        result = R @ np.array([1, 0, 0])
        expected = np.array([0, 1, 0])
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_identity_rotation(self):
        """Zero rotation vector should give identity matrix."""
        rvec = np.array([0.0, 0.0, 0.0])
        R = rvec_to_matrix(rvec)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_small_angle_stability(self):
        """Small rotation angles should be numerically stable."""
        rvec = np.array([1e-8, 0.0, 0.0])
        R = rvec_to_matrix(rvec)
        # Should be close to identity for very small angles
        np.testing.assert_allclose(R, np.eye(3), atol=1e-7)


class TestMatrixToRvec:
    """Tests for matrix_to_rvec function."""

    def test_identity_matrix(self):
        """Identity matrix should give zero rotation vector."""
        R = np.eye(3)
        rvec = matrix_to_rvec(R)
        np.testing.assert_allclose(rvec, np.zeros(3), atol=1e-10)

    def test_output_shape(self):
        """Output should be shape (3,) not (3,1)."""
        R = np.eye(3)
        rvec = matrix_to_rvec(R)
        assert rvec.shape == (3,), f"Expected shape (3,), got {rvec.shape}"

    def test_90_degree_rotation(self):
        """Should correctly convert a 90° rotation matrix."""
        # Create a 90° rotation around Z
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        rvec = matrix_to_rvec(R)
        # Should be approximately [0, 0, pi/2]
        expected = np.array([0, 0, np.pi / 2])
        np.testing.assert_allclose(rvec, expected, atol=1e-10)


class TestRoundTrips:
    """Tests for round-trip conversions."""

    def test_rvec_round_trip(self):
        """Converting rvec->matrix->rvec should recover original."""
        rvec = np.array([0.1, 0.2, 0.3])
        R = rvec_to_matrix(rvec)
        rvec_recovered = matrix_to_rvec(R)
        np.testing.assert_allclose(rvec_recovered, rvec, atol=1e-10)

    def test_matrix_round_trip(self):
        """Converting matrix->rvec->matrix should recover original."""
        rvec_temp = np.array([0.5, -0.3, 0.7])
        R = rvec_to_matrix(rvec_temp)
        rvec = matrix_to_rvec(R)
        R_recovered = rvec_to_matrix(rvec)
        np.testing.assert_allclose(R_recovered, R, atol=1e-10)


class TestComposePoses:
    """Tests for compose_poses function."""

    def test_two_translations(self):
        """Composing two pure translations should add them."""
        R1, t1 = np.eye(3), np.array([1, 0, 0])
        R2, t2 = np.eye(3), np.array([0, 1, 0])
        R, t = compose_poses(R1, t1, R2, t2)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(t, np.array([1, 1, 0]), atol=1e-10)

    def test_rotation_then_translation(self):
        """Rotation then translation should transform the second translation."""
        # Rotate 90° around Z, then translate
        R1 = rvec_to_matrix(np.array([0, 0, np.pi / 2]))
        t1 = np.array([1, 0, 0])
        R2 = np.eye(3)
        t2 = np.array([1, 0, 0])
        R, t = compose_poses(R1, t1, R2, t2)
        # R1 @ t2 = [0, 1, 0], so t = [0, 1, 0] + [1, 0, 0] = [1, 1, 0]
        expected_t = np.array([1, 1, 0])
        np.testing.assert_allclose(t, expected_t, atol=1e-10)

    def test_compose_with_inverse_gives_identity(self):
        """Composing pose with its inverse gives identity."""
        R = rvec_to_matrix(np.array([0.5, -0.3, 0.7]))
        t = np.array([1.0, -2.0, 3.0])

        R_inv, t_inv = invert_pose(R, t)
        R_result, t_result = compose_poses(R, t, R_inv, t_inv)

        np.testing.assert_allclose(R_result, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(t_result, np.zeros(3), atol=1e-10)


class TestInvertPose:
    """Tests for invert_pose function."""

    def test_identity_pose(self):
        """Inverting identity should give identity."""
        R = np.eye(3)
        t = np.array([1, 2, 3])
        R_inv, t_inv = invert_pose(R, t)
        np.testing.assert_allclose(R_inv, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(t_inv, np.array([-1, -2, -3]), atol=1e-10)

    def test_double_inversion_gives_original(self):
        """Inverting twice should give original pose."""
        R = rvec_to_matrix(np.array([0.2, 0.3, 0.4]))
        t = np.array([1.0, 2.0, 3.0])

        R_inv, t_inv = invert_pose(R, t)
        R_back, t_back = invert_pose(R_inv, t_inv)

        np.testing.assert_allclose(R_back, R, atol=1e-10)
        np.testing.assert_allclose(t_back, t, atol=1e-10)

    def test_rotated_pose(self):
        """Inverting a rotated pose."""
        R = rvec_to_matrix(np.array([0, 0, np.pi / 2]))  # 90° around Z
        t = np.array([1, 0, 0])
        R_inv, t_inv = invert_pose(R, t)

        # Check that compose gives identity
        R_id, t_id = compose_poses(R, t, R_inv, t_inv)
        np.testing.assert_allclose(R_id, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(t_id, np.zeros(3), atol=1e-10)


class TestCameraCenter:
    """Tests for camera_center function."""

    def test_identity_rotation_positive_z(self):
        """Camera center with identity rotation and positive Z translation."""
        R = np.eye(3)
        t = np.array([0, 0, 5])
        C = camera_center(R, t)
        np.testing.assert_allclose(C, np.array([0, 0, -5]), atol=1e-10)

    def test_identity_rotation_xyz_translation(self):
        """Camera center with identity rotation and general translation."""
        R = np.eye(3)
        t = np.array([1, 2, 3])
        C = camera_center(R, t)
        # C = -R.T @ t = -I @ t = -t
        np.testing.assert_allclose(C, np.array([-1, -2, -3]), atol=1e-10)

    def test_rotated_camera(self):
        """Camera center with non-identity rotation."""
        # Camera rotated 90° around Z, then translated
        R = rvec_to_matrix(np.array([0, 0, np.pi / 2]))
        t = np.array([1, 0, 0])
        C = camera_center(R, t)
        # C = -R.T @ t
        # R.T rotates -90° around Z, so [1,0,0] -> [0,-1,0]
        # C = -[0,-1,0] = [0,1,0]
        np.testing.assert_allclose(C, np.array([0, 1, 0]), atol=1e-10)

    def test_camera_center_satisfies_equation(self):
        """Verify that t = -R @ C holds for computed camera center."""
        R = rvec_to_matrix(np.array([0.3, 0.4, 0.5]))
        t = np.array([1.5, -2.3, 4.7])
        C = camera_center(R, t)

        # Check that t = -R @ C
        t_check = -R @ C
        np.testing.assert_allclose(t_check, t, atol=1e-10)


class TestEdgeCases:
    """Tests for edge cases and numerical stability."""

    def test_180_degree_rotation(self):
        """180° rotation should be handled correctly."""
        rvec = np.array([np.pi, 0, 0])  # 180° around X
        R = rvec_to_matrix(rvec)
        rvec_back = matrix_to_rvec(R)
        # 180° rotations can have sign ambiguity, but should be close in magnitude
        assert np.abs(np.linalg.norm(rvec_back) - np.pi) < 1e-6

    def test_very_small_rotation(self):
        """Very small rotations should be stable."""
        rvec = np.array([1e-10, 1e-10, 1e-10])
        R = rvec_to_matrix(rvec)
        rvec_back = matrix_to_rvec(R)
        # Should be close to zero or the original small value
        assert np.linalg.norm(rvec_back) < 1e-8

    def test_zero_translation(self):
        """Poses with zero translation should work correctly."""
        R = rvec_to_matrix(np.array([0.1, 0.2, 0.3]))
        t = np.zeros(3)
        R_inv, t_inv = invert_pose(R, t)
        np.testing.assert_allclose(t_inv, np.zeros(3), atol=1e-10)

        C = camera_center(R, t)
        np.testing.assert_allclose(C, np.zeros(3), atol=1e-10)
