"""Tests for board geometry module."""

import numpy as np
import pytest

from aquacal.config.schema import BoardConfig
from aquacal.core.board import BoardGeometry


@pytest.fixture
def board_config():
    """Standard 8x6 board configuration."""
    return BoardConfig(
        squares_x=8,
        squares_y=6,
        square_size=0.03,  # 3cm
        marker_size=0.022,
        dictionary="DICT_4X4_50"
    )


class TestBoardGeometry:
    """Test suite for BoardGeometry class."""

    def test_num_corners(self, board_config):
        """8x6 board has 7x5=35 interior corners."""
        board = BoardGeometry(board_config)
        assert board.num_corners == 35

    def test_corner_0_at_origin(self, board_config):
        """Corner 0 should be at origin."""
        board = BoardGeometry(board_config)
        np.testing.assert_allclose(
            board.corner_positions[0],
            np.array([0, 0, 0])
        )

    def test_corner_1_position(self, board_config):
        """Corner 1 should be one square_size to the right."""
        board = BoardGeometry(board_config)
        np.testing.assert_allclose(
            board.corner_positions[1],
            np.array([0.03, 0, 0])
        )

    def test_corner_7_position(self, board_config):
        """Corner 7 (first of second row) should be one square_size down."""
        board = BoardGeometry(board_config)
        np.testing.assert_allclose(
            board.corner_positions[7],
            np.array([0, 0.03, 0])
        )

    def test_last_corner_position(self, board_config):
        """Last corner (ID 34) should be at bottom-right."""
        board = BoardGeometry(board_config)
        # 7x5 grid, so last corner is at (6, 4) in grid coordinates
        np.testing.assert_allclose(
            board.corner_positions[34],
            np.array([6 * 0.03, 4 * 0.03, 0])
        )

    def test_all_corners_have_z_zero(self, board_config):
        """All corners should have Z=0 (board is planar)."""
        board = BoardGeometry(board_config)
        for corner_id, pos in board.corner_positions.items():
            assert pos[2] == 0.0, f"Corner {corner_id} has non-zero Z: {pos[2]}"

    def test_corner_positions_dict_complete(self, board_config):
        """Corner positions dict should have all corner IDs."""
        board = BoardGeometry(board_config)
        assert len(board.corner_positions) == 35
        for i in range(35):
            assert i in board.corner_positions

    def test_transform_identity(self, board_config):
        """Identity transform should not change positions."""
        board = BoardGeometry(board_config)
        transformed = board.transform_corners(np.zeros(3), np.zeros(3))
        for corner_id, pos in board.corner_positions.items():
            np.testing.assert_allclose(
                transformed[corner_id],
                pos,
                rtol=1e-10,
                err_msg=f"Corner {corner_id} not preserved under identity transform"
            )

    def test_transform_translation(self, board_config):
        """Pure translation should shift all corners."""
        board = BoardGeometry(board_config)
        tvec = np.array([1.0, 2.0, 3.0])
        transformed = board.transform_corners(np.zeros(3), tvec)
        for corner_id, pos in board.corner_positions.items():
            np.testing.assert_allclose(
                transformed[corner_id],
                pos + tvec,
                rtol=1e-10,
                err_msg=f"Corner {corner_id} not correctly translated"
            )

    def test_transform_rotation_90deg(self, board_config):
        """90-degree rotation about Z should swap and negate appropriately."""
        board = BoardGeometry(board_config)
        # 90-degree rotation about Z axis
        rvec = np.array([0, 0, np.pi / 2])
        transformed = board.transform_corners(rvec, np.zeros(3))

        # After 90deg rotation about Z: [x, y, z] -> [-y, x, z]
        # Corner 1 at [0.03, 0, 0] should go to [0, 0.03, 0]
        np.testing.assert_allclose(
            transformed[1],
            np.array([0, 0.03, 0]),
            atol=1e-10
        )

    def test_get_corner_array_shape(self, board_config):
        """get_corner_array should return (N, 3) array."""
        board = BoardGeometry(board_config)
        ids = np.array([0, 1, 7, 8])
        pts = board.get_corner_array(ids)
        assert pts.shape == (4, 3)

    def test_get_corner_array_values(self, board_config):
        """get_corner_array values should match corner_positions."""
        board = BoardGeometry(board_config)
        ids = np.array([0, 1])
        pts = board.get_corner_array(ids)
        np.testing.assert_allclose(pts[0], board.corner_positions[0])
        np.testing.assert_allclose(pts[1], board.corner_positions[1])

    def test_get_corner_array_preserves_order(self, board_config):
        """get_corner_array should preserve the order of input IDs."""
        board = BoardGeometry(board_config)
        ids = np.array([7, 1, 0, 8])
        pts = board.get_corner_array(ids)

        np.testing.assert_allclose(pts[0], board.corner_positions[7])
        np.testing.assert_allclose(pts[1], board.corner_positions[1])
        np.testing.assert_allclose(pts[2], board.corner_positions[0])
        np.testing.assert_allclose(pts[3], board.corner_positions[8])

    def test_opencv_board_creation(self, board_config):
        """Should create valid OpenCV CharucoBoard."""
        board = BoardGeometry(board_config)
        cv_board = board.get_opencv_board()
        assert cv_board is not None

        # Verify it has the expected number of corners
        chessboard_corners = cv_board.getChessboardCorners()
        assert len(chessboard_corners) == 35

    def test_opencv_board_size_matches(self, board_config):
        """OpenCV board should have correct size."""
        board = BoardGeometry(board_config)
        cv_board = board.get_opencv_board()

        # Verify board size matches configuration
        board_size = cv_board.getChessboardSize()
        assert board_size[0] == board_config.squares_x
        assert board_size[1] == board_config.squares_y

    def test_different_board_sizes(self):
        """Test with different board configurations."""
        # 5x4 board
        config = BoardConfig(
            squares_x=5,
            squares_y=4,
            square_size=0.05,
            marker_size=0.04,
            dictionary="DICT_5X5_100"
        )
        board = BoardGeometry(config)

        assert board.num_corners == 4 * 3  # 12 corners
        # Last corner should be at (3*0.05, 2*0.05, 0)
        np.testing.assert_allclose(
            board.corner_positions[11],
            np.array([3 * 0.05, 2 * 0.05, 0])
        )

    def test_corner_positions_property_returns_same_dict(self, board_config):
        """Calling corner_positions multiple times should return the same dict."""
        board = BoardGeometry(board_config)
        pos1 = board.corner_positions
        pos2 = board.corner_positions
        assert pos1 is pos2  # Should be the same object

    def test_board_frame_convention(self, board_config):
        """Verify board frame convention: origin at corner 0, X right, Y down, Z into board."""
        board = BoardGeometry(board_config)

        # Origin at corner 0
        assert np.allclose(board.corner_positions[0], [0, 0, 0])

        # X increases to the right (columns)
        assert board.corner_positions[1][0] > board.corner_positions[0][0]
        assert board.corner_positions[1][1] == board.corner_positions[0][1]

        # Y increases downward (rows)
        assert board.corner_positions[7][1] > board.corner_positions[0][1]
        assert board.corner_positions[7][0] == board.corner_positions[0][0]

        # Z is zero for all corners (planar board)
        for pos in board.corner_positions.values():
            assert pos[2] == 0.0
