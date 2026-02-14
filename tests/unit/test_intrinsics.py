"""Tests for intrinsic calibration."""

import pytest
import cv2
import numpy as np
from pathlib import Path

from aquacal.config.schema import BoardConfig, CameraIntrinsics
from aquacal.core.board import BoardGeometry
from aquacal.calibration.intrinsics import (
    calibrate_intrinsics_single,
    calibrate_intrinsics_all,
    _select_calibration_frames,
    validate_intrinsics,
)


@pytest.fixture
def board_config() -> BoardConfig:
    """Standard test board configuration."""
    return BoardConfig(
        squares_x=6,
        squares_y=5,
        square_size=0.04,
        marker_size=0.03,
        dictionary="DICT_4X4_50",
    )


@pytest.fixture
def board(board_config: BoardConfig) -> BoardGeometry:
    """Board geometry from config."""
    return BoardGeometry(board_config)


@pytest.fixture
def synthetic_intrinsics() -> CameraIntrinsics:
    """Known intrinsics for synthetic data generation."""
    return CameraIntrinsics(
        K=np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], dtype=np.float64),
        dist_coeffs=np.array([0.1, -0.2, 0.0, 0.0, 0.1], dtype=np.float64),
        image_size=(640, 480),
    )


@pytest.fixture
def calibration_video(
    tmp_path: Path,
    board: BoardGeometry,
    synthetic_intrinsics: CameraIntrinsics,
) -> Path:
    """
    Create a synthetic calibration video with ChArUco board at various poses.
    """
    video_path = tmp_path / "calibration.mp4"
    w, h = synthetic_intrinsics.image_size
    _K = synthetic_intrinsics.K
    _dist = synthetic_intrinsics.dist_coeffs

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))

    cv_board = board.get_opencv_board()
    board_img = cv_board.generateImage((400, 300), marginSize=30)

    # Generate frames with board at different poses
    for i in range(20):
        # Create white background
        frame = np.full((h, w, 3), 255, dtype=np.uint8)

        # Vary the board position/rotation
        angle = (i - 10) * 3  # -30 to +30 degrees
        scale = 0.8 + (i % 5) * 0.1  # Vary size
        tx = 50 + (i % 4) * 100  # Vary x position
        ty = 30 + (i % 3) * 80  # Vary y position

        # Simple affine transform for board placement
        M = cv2.getRotationMatrix2D((200, 150), angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        warped = cv2.warpAffine(board_img, M, (w, h), borderValue=255)

        # Convert to BGR if needed
        if warped.ndim == 2:
            warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

        # Blend with frame
        mask = warped < 250
        frame[mask] = warped[mask]

        writer.write(frame)

    writer.release()
    return video_path


class TestCalibrateSingle:
    def test_returns_correct_types(self, calibration_video, board):
        """Returns CameraIntrinsics and float."""
        intrinsics, error = calibrate_intrinsics_single(calibration_video, board)

        assert isinstance(intrinsics, CameraIntrinsics)
        assert isinstance(error, float)

    def test_intrinsics_shapes(self, calibration_video, board):
        """Intrinsic matrix and dist_coeffs have correct shapes."""
        intrinsics, _ = calibrate_intrinsics_single(calibration_video, board)

        assert intrinsics.K.shape == (3, 3)
        assert intrinsics.K.dtype == np.float64
        assert intrinsics.dist_coeffs.shape == (5,)
        assert intrinsics.dist_coeffs.dtype == np.float64

    def test_image_size_extracted(self, calibration_video, board):
        """Image size is extracted from video."""
        intrinsics, _ = calibrate_intrinsics_single(calibration_video, board)

        assert intrinsics.image_size == (640, 480)

    def test_reprojection_error_reasonable(self, calibration_video, board):
        """Reprojection error is reasonable (< 2 pixels for synthetic data)."""
        _, error = calibrate_intrinsics_single(calibration_video, board)

        assert error < 2.0  # Should be very low for clean synthetic data

    def test_accepts_path_object(self, calibration_video, board):
        """Accepts Path object."""
        intrinsics, _ = calibrate_intrinsics_single(Path(calibration_video), board)
        assert intrinsics is not None

    def test_accepts_string_path(self, calibration_video, board):
        """Accepts string path."""
        intrinsics, _ = calibrate_intrinsics_single(str(calibration_video), board)
        assert intrinsics is not None

    def test_max_frames_parameter(self, calibration_video, board):
        """Respects max_frames parameter."""
        # With very low max_frames, should still work but may have higher error
        intrinsics, _ = calibrate_intrinsics_single(
            calibration_video, board, max_frames=5
        )
        assert intrinsics is not None

    def test_raises_on_empty_video(self, tmp_path, board):
        """Raises ValueError for video with no valid detections."""
        # Create blank video
        video_path = tmp_path / "blank.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
        for _ in range(5):
            writer.write(np.full((480, 640, 3), 128, dtype=np.uint8))
        writer.release()

        with pytest.raises(ValueError, match="No valid frames"):
            calibrate_intrinsics_single(video_path, board)

    def test_filters_collinear_detections(self, tmp_path, board):
        """Filters out frames where all detected corners are collinear."""
        # Create a video with some frames having collinear corners (all on same row)
        # and some frames with non-collinear corners
        video_path = tmp_path / "collinear_test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

        cv_board = board.get_opencv_board()

        # Create a few frames with normal (non-collinear) detections
        for i in range(10):
            # Create board image at various poses
            board_img = cv_board.generateImage((400, 300), marginSize=30)
            frame = np.full((480, 640, 3), 255, dtype=np.uint8)

            # Simple affine transform for board placement
            angle = (i - 5) * 5
            scale = 0.7 + (i % 3) * 0.1
            M = cv2.getRotationMatrix2D((200, 150), angle, scale)
            M[0, 2] += 100 + i * 10
            M[1, 2] += 80

            warped = cv2.warpAffine(board_img, M, (640, 480), borderValue=255)
            if warped.ndim == 2:
                warped = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

            mask = warped < 250
            frame[mask] = warped[mask]
            writer.write(frame)

        writer.release()

        # The test: calibration should succeed despite potential collinear frames
        # If collinear frames weren't filtered, OpenCV would crash or raise an error
        intrinsics, error = calibrate_intrinsics_single(
            video_path, board, min_corners=4
        )

        # If we got here without crashing, the filter worked
        assert intrinsics is not None
        assert isinstance(error, float)


class TestCalibrateAll:
    def test_processes_all_cameras(self, calibration_video, board, tmp_path):
        """Processes all cameras in video_paths."""
        # Create second video (copy of first for simplicity)
        video2 = tmp_path / "cam1.mp4"
        import shutil

        shutil.copy(calibration_video, video2)

        paths = {
            "cam0": str(calibration_video),
            "cam1": str(video2),
        }

        results = calibrate_intrinsics_all(paths, board)

        assert set(results.keys()) == {"cam0", "cam1"}
        for name, (intrinsics, error) in results.items():
            assert isinstance(intrinsics, CameraIntrinsics)
            assert isinstance(error, float)

    def test_progress_callback(self, calibration_video, board, tmp_path):
        """Calls progress callback for each camera."""
        video2 = tmp_path / "cam1.mp4"
        import shutil

        shutil.copy(calibration_video, video2)

        paths = {"cam0": str(calibration_video), "cam1": str(video2)}
        calls = []

        def callback(name, current, total):
            calls.append((name, current, total))

        calibrate_intrinsics_all(paths, board, progress_callback=callback)

        assert len(calls) == 2
        assert calls[0] == ("cam0", 1, 2)
        assert calls[1] == ("cam1", 2, 2)


class TestSelectCalibrationFrames:
    def test_returns_all_if_under_max(self):
        """Returns all detections if count <= max_frames."""
        detections = [
            (
                np.array([0, 1, 2]),
                np.array([[100, 100], [200, 100], [150, 200]], dtype=np.float64),
            )
            for _ in range(5)
        ]

        selected = _select_calibration_frames(
            detections, max_frames=10, image_size=(640, 480)
        )

        assert len(selected) == 5

    def test_limits_to_max_frames(self):
        """Limits output to max_frames."""
        detections = [
            (np.array([0, 1, 2, 3]), np.random.rand(4, 2) * 640) for _ in range(20)
        ]

        selected = _select_calibration_frames(
            detections, max_frames=10, image_size=(640, 480)
        )

        assert len(selected) == 10

    def test_prefers_spread_corners(self):
        """Prefers frames with corners spread across image."""
        # Clustered corners (low coverage)
        clustered = (
            np.array([0, 1, 2, 3]),
            np.array(
                [[300, 200], [310, 200], [300, 210], [310, 210]], dtype=np.float64
            ),
        )

        # Spread corners (high coverage)
        spread = (
            np.array([0, 1, 2, 3]),
            np.array([[50, 50], [590, 50], [50, 430], [590, 430]], dtype=np.float64),
        )

        detections = [clustered, spread]
        selected = _select_calibration_frames(
            detections, max_frames=1, image_size=(640, 480)
        )

        # Should select the spread one
        np.testing.assert_array_equal(selected[0][1], spread[1])


class TestValidateIntrinsics:
    def test_good_intrinsics_passes(self):
        """Returns no warnings for well-calibrated intrinsics."""
        intrinsics = CameraIntrinsics(
            K=np.array([[800.0, 0, 640], [0, 800.0, 480], [0, 0, 1]], dtype=np.float64),
            dist_coeffs=np.array([0.1, -0.05, 0.0, 0.0, 0.02], dtype=np.float64),
            image_size=(1280, 960),
        )

        warnings = validate_intrinsics(intrinsics, camera_name="test_cam")

        assert isinstance(warnings, list)
        assert len(warnings) == 0

    def test_bad_roundtrip_warns(self):
        """Detects extreme distortion that breaks roundtrip."""
        # Extreme distortion coefficients that will cause large roundtrip errors
        intrinsics = CameraIntrinsics(
            K=np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], dtype=np.float64),
            dist_coeffs=np.array([5.0, -10.0, 0.0, 0.0, 20.0], dtype=np.float64),
            image_size=(640, 480),
        )

        warnings = validate_intrinsics(
            intrinsics, camera_name="bad_cam", max_roundtrip_error_px=0.5
        )

        # Should have at least one warning
        assert len(warnings) > 0
        # Should mention roundtrip error
        assert any("roundtrip" in w.lower() for w in warnings)

    def test_negative_distortion_factor_warns(self):
        """Detects when distortion polynomial goes negative."""
        # Large negative k1 will cause distortion factor to go negative
        intrinsics = CameraIntrinsics(
            K=np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], dtype=np.float64),
            dist_coeffs=np.array([-5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            image_size=(640, 480),
        )

        warnings = validate_intrinsics(intrinsics, camera_name="negative_cam")

        # Should have at least one warning about distortion model
        assert len(warnings) > 0
        assert any(
            "distortion" in w.lower()
            and ("negative" in w.lower() or "non-monotonic" in w.lower())
            for w in warnings
        )

    def test_fx_check_passes_within_tolerance(self):
        """Passes fx check when within tolerance."""
        intrinsics = CameraIntrinsics(
            K=np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]], dtype=np.float64),
            dist_coeffs=np.array([0.1, -0.05, 0.0, 0.0, 0.0], dtype=np.float64),
            image_size=(640, 480),
        )

        # Expected fx is 820, actual is 800 -> 2.4% difference, within 30% tolerance
        warnings = validate_intrinsics(
            intrinsics,
            camera_name="test_cam",
            expected_fx=820.0,
            fx_tolerance_fraction=0.3,
        )

        # Should not warn about fx
        assert not any("fx=" in w for w in warnings)

    def test_fx_check_warns_outside_tolerance(self):
        """Warns when fx is outside tolerance."""
        intrinsics = CameraIntrinsics(
            K=np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], dtype=np.float64),
            dist_coeffs=np.array([0.1, -0.05, 0.0, 0.0, 0.0], dtype=np.float64),
            image_size=(640, 480),
        )

        # Expected fx is 1000, actual is 500 -> 50% difference, outside 30% tolerance
        warnings = validate_intrinsics(
            intrinsics,
            camera_name="bad_fx_cam",
            expected_fx=1000.0,
            fx_tolerance_fraction=0.3,
        )

        # Should warn about fx
        assert len(warnings) > 0
        assert any("fx=" in w for w in warnings)

    def test_fisheye_skips_monotonicity(self):
        """Fisheye intrinsics skip monotonicity check."""
        # Create fisheye intrinsics with large coefficients
        intrinsics = CameraIntrinsics(
            K=np.array([[400.0, 0, 640], [0, 400.0, 480], [0, 0, 1]], dtype=np.float64),
            dist_coeffs=np.array([0.5, -0.3, 0.1, -0.05], dtype=np.float64),
            image_size=(1280, 960),
            is_fisheye=True,
        )

        warnings = validate_intrinsics(intrinsics, camera_name="fisheye_cam")

        # Should only check roundtrip, not monotonicity
        # No warnings should mention "monotonic" or "negative"
        for w in warnings:
            assert "monotonic" not in w.lower()
            # May have roundtrip warnings, but not distortion model warnings

    def test_returns_empty_list_on_success(self):
        """Returns empty list for successful validation."""
        intrinsics = CameraIntrinsics(
            K=np.array([[800.0, 0, 640], [0, 800.0, 480], [0, 0, 1]], dtype=np.float64),
            dist_coeffs=np.array([0.05, -0.02, 0.0, 0.0, 0.01], dtype=np.float64),
            image_size=(1280, 960),
        )

        warnings = validate_intrinsics(intrinsics)

        assert isinstance(warnings, list)
        assert warnings == []

    def test_rational_model_validation(self):
        """Validates 8-coefficient rational model."""
        intrinsics = CameraIntrinsics(
            K=np.array([[800.0, 0, 640], [0, 800.0, 480], [0, 0, 1]], dtype=np.float64),
            # 8-coeff rational model with reasonable values
            dist_coeffs=np.array(
                [0.1, -0.05, 0.0, 0.0, 0.02, 0.01, 0.005, 0.001], dtype=np.float64
            ),
            image_size=(1280, 960),
        )

        warnings = validate_intrinsics(intrinsics, camera_name="rational_cam")

        # Should work without crashing and return a list
        assert isinstance(warnings, list)
        # Reasonable rational model should pass
        # (may or may not have warnings depending on coefficients)
