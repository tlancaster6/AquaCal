"""ChArUco board geometry and utilities."""

import cv2
import numpy as np
from numpy.typing import NDArray

from aquacal.config.schema import BoardConfig, Vec3


class BoardGeometry:
    """
    ChArUco board 3D geometry.

    The board frame has origin at the top-left corner (when viewed from front),
    with X pointing right, Y pointing down, and Z pointing into the board
    (away from viewer). This matches OpenCV 4.6+ CharucoBoard convention.

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
        self.config = config
        self._corner_positions = self._compute_corner_positions()

    def _compute_corner_positions(self) -> dict[int, Vec3]:
        """
        Compute 3D positions of all interior corners.

        Returns:
            Dict mapping corner_id to 3D position in board frame (meters)
        """
        positions = {}
        cols = self.config.squares_x - 1
        rows = self.config.squares_y - 1
        for corner_id in range(cols * rows):
            col = corner_id % cols
            row = corner_id // cols
            positions[corner_id] = np.array(
                [col * self.config.square_size, row * self.config.square_size, 0.0],
                dtype=np.float64,
            )
        return positions

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
        return self._corner_positions

    @property
    def num_corners(self) -> int:
        """
        Get total number of interior corners.

        Returns:
            Number of corners = (squares_x - 1) * (squares_y - 1)
        """
        return (self.config.squares_x - 1) * (self.config.squares_y - 1)

    def get_opencv_board(self) -> cv2.aruco.CharucoBoard:
        """
        Get OpenCV CharucoBoard object for detection.

        Returns:
            OpenCV CharucoBoard instance
        """
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, self.config.dictionary)
        )
        board = cv2.aruco.CharucoBoard(
            (self.config.squares_x, self.config.squares_y),
            self.config.square_size,
            self.config.marker_size,
            dictionary,
        )
        if self.config.legacy_pattern:
            board.setLegacyPattern(True)
        return board

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
        R, _ = cv2.Rodrigues(rvec)
        return {
            corner_id: R @ pos + tvec
            for corner_id, pos in self.corner_positions.items()
        }

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
        return np.array(
            [self.corner_positions[int(corner_id)] for corner_id in corner_ids],
            dtype=np.float64,
        )
