"""Rotation and coordinate transform utilities."""

import numpy as np
from numpy.typing import NDArray
import cv2

# Local type aliases (matches schema.py but no import needed)
Vec3 = NDArray[np.float64]
Mat3 = NDArray[np.float64]


def rvec_to_matrix(rvec: Vec3) -> Mat3:
    """
    Convert Rodrigues vector to rotation matrix.

    Args:
        rvec: Rotation vector, shape (3,). The axis is rvec/||rvec|| and
            the angle is ||rvec|| in radians.

    Returns:
        R: Rotation matrix, shape (3, 3)

    Example:
        >>> rvec = np.array([0.0, 0.0, np.pi/2])
        >>> R = rvec_to_matrix(rvec)
        >>> np.allclose(R @ np.array([1, 0, 0]), np.array([0, 1, 0]))
        True
    """
    R, _ = cv2.Rodrigues(rvec)
    return np.asarray(R, dtype=np.float64)


def matrix_to_rvec(R: Mat3) -> Vec3:
    """
    Convert rotation matrix to Rodrigues vector.

    Args:
        R: Rotation matrix, shape (3, 3)

    Returns:
        rvec: Rotation vector, shape (3,). The axis is rvec/||rvec|| and
            the angle is ||rvec|| in radians.

    Example:
        >>> R = np.eye(3)
        >>> rvec = matrix_to_rvec(R)
        >>> np.allclose(rvec, np.zeros(3))
        True
    """
    rvec, _ = cv2.Rodrigues(R)
    return np.asarray(rvec.flatten(), dtype=np.float64)


def compose_poses(R1: Mat3, t1: Vec3, R2: Mat3, t2: Vec3) -> tuple[Mat3, Vec3]:
    """
    Compose two poses: T_combined = T1 @ T2.

    If T1 transforms from frame A to frame B, and T2 transforms from frame B
    to frame C, then T_combined transforms from frame A to frame C.

    Args:
        R1: First rotation matrix, shape (3, 3)
        t1: First translation vector, shape (3,)
        R2: Second rotation matrix, shape (3, 3)
        t2: Second translation vector, shape (3,)

    Returns:
        R_combined: Combined rotation matrix, shape (3, 3)
        t_combined: Combined translation vector, shape (3,)

    Example:
        >>> R1, t1 = np.eye(3), np.array([1, 0, 0])
        >>> R2, t2 = np.eye(3), np.array([0, 1, 0])
        >>> R, t = compose_poses(R1, t1, R2, t2)
        >>> np.allclose(t, np.array([1, 1, 0]))
        True
    """
    R_combined = R1 @ R2
    t_combined = R1 @ t2 + t1
    return R_combined, t_combined


def invert_pose(R: Mat3, t: Vec3) -> tuple[Mat3, Vec3]:
    """
    Invert a pose transformation.

    Args:
        R: Rotation matrix, shape (3, 3)
        t: Translation vector, shape (3,)

    Returns:
        R_inv: Inverted rotation matrix, shape (3, 3)
        t_inv: Inverted translation vector, shape (3,)

    Example:
        >>> R = np.eye(3)
        >>> t = np.array([1, 2, 3])
        >>> R_inv, t_inv = invert_pose(R, t)
        >>> np.allclose(t_inv, np.array([-1, -2, -3]))
        True
    """
    R_inv = R.T
    t_inv = -R.T @ t
    return R_inv, t_inv


def camera_center(R: Mat3, t: Vec3) -> Vec3:
    """
    Compute camera center in world coordinates.

    The camera center C satisfies: t = -R @ C

    Args:
        R: Rotation matrix (world to camera), shape (3, 3)
        t: Translation vector, shape (3,)

    Returns:
        C: Camera center in world frame, shape (3,)

    Example:
        >>> R = np.eye(3)
        >>> t = np.array([0, 0, 5])
        >>> C = camera_center(R, t)
        >>> np.allclose(C, np.array([0, 0, -5]))
        True
    """
    C = -R.T @ t
    return C
