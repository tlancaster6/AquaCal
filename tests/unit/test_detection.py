"""Unit tests for ChArUco detection."""

import pytest
import cv2
import numpy as np
from pathlib import Path

from aquacal.config.schema import (
    BoardConfig,
    Detection,
    FrameDetections,
    DetectionResult,
)
from aquacal.core.board import BoardGeometry
from aquacal.io.detection import detect_charuco, detect_all_frames
from aquacal.io.video import VideoSet


@pytest.fixture
def board_config() -> BoardConfig:
    """Standard test board configuration."""
    return BoardConfig(
        squares_x=5,
        squares_y=4,
        square_size=0.04,
        marker_size=0.03,
        dictionary="DICT_4X4_50",
    )


@pytest.fixture
def board(board_config: BoardConfig) -> BoardGeometry:
    """Board geometry from config."""
    return BoardGeometry(board_config)


@pytest.fixture
def charuco_image(board: BoardGeometry) -> np.ndarray:
    """
    Generate a synthetic ChArUco board image.

    Uses OpenCV to draw the board, then applies a perspective transform
    to simulate a real camera view.
    """
    cv_board = board.get_opencv_board()

    # Draw board at high resolution
    board_img = cv_board.generateImage((800, 600), marginSize=50)

    # Convert to 3-channel for consistency
    if board_img.ndim == 2:
        board_img = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)

    return board_img


@pytest.fixture
def charuco_image_warped(charuco_image: np.ndarray) -> np.ndarray:
    """
    ChArUco image with perspective warp to simulate real camera view.
    """
    h, w = charuco_image.shape[:2]

    # Define perspective transform (slight rotation/tilt)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = np.float32([[50, 30], [w - 30, 50], [w - 50, h - 30], [30, h - 50]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(charuco_image, M, (w, h), borderValue=(255, 255, 255))

    return warped


@pytest.fixture
def test_videos_with_charuco(
    tmp_path: Path, charuco_image_warped: np.ndarray
) -> dict[str, str]:
    """
    Create test videos containing ChArUco board frames.
    """
    paths = {}
    h, w = charuco_image_warped.shape[:2]

    for cam_name in ["cam0", "cam1"]:
        path = tmp_path / f"{cam_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))

        for i in range(10):
            # Add slight variation per frame (brightness)
            frame = charuco_image_warped.copy()
            frame = cv2.convertScaleAbs(frame, alpha=1.0 + i * 0.01, beta=0)
            writer.write(frame)

        writer.release()
        paths[cam_name] = str(path)

    return paths


class TestDetectCharuco:
    def test_detects_corners_in_valid_image(self, board, charuco_image):
        """Detects corners in a clean ChArUco image."""
        detection = detect_charuco(charuco_image, board)

        assert detection is not None
        assert detection.num_corners > 0
        assert detection.corner_ids.shape == (detection.num_corners,)
        assert detection.corners_2d.shape == (detection.num_corners, 2)

    def test_detects_corners_in_warped_image(self, board, charuco_image_warped):
        """Detects corners in a perspective-warped image."""
        detection = detect_charuco(charuco_image_warped, board)

        assert detection is not None
        assert detection.num_corners > 0

    def test_returns_none_for_blank_image(self, board):
        """Returns None when no board is visible."""
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        detection = detect_charuco(blank, board)

        assert detection is None

    def test_handles_grayscale_input(self, board, charuco_image):
        """Accepts grayscale image input."""
        gray = cv2.cvtColor(charuco_image, cv2.COLOR_BGR2GRAY)
        detection = detect_charuco(gray, board)

        assert detection is not None

    def test_handles_bgr_input(self, board, charuco_image):
        """Accepts BGR image input."""
        assert charuco_image.ndim == 3
        detection = detect_charuco(charuco_image, board)

        assert detection is not None

    def test_with_intrinsics(self, board, charuco_image_warped):
        """Uses intrinsics for corner refinement."""
        # Simple intrinsic matrix
        h, w = charuco_image_warped.shape[:2]
        K = np.array([[500, 0, w / 2], [0, 500, h / 2], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)

        detection = detect_charuco(charuco_image_warped, board, K, dist)

        assert detection is not None

    def test_corner_ids_are_int32(self, board, charuco_image):
        """corner_ids has correct dtype."""
        detection = detect_charuco(charuco_image, board)

        assert detection is not None
        assert detection.corner_ids.dtype == np.int32

    def test_corners_2d_are_float64(self, board, charuco_image):
        """corners_2d has correct dtype."""
        detection = detect_charuco(charuco_image, board)

        assert detection is not None
        assert detection.corners_2d.dtype == np.float64


class TestDetectAllFrames:
    def test_with_dict_paths(self, board, test_videos_with_charuco):
        """Accepts dict of video paths."""
        result = detect_all_frames(test_videos_with_charuco, board)

        assert isinstance(result, DetectionResult)
        assert result.camera_names == ["cam0", "cam1"]
        assert result.total_frames == 10

    def test_with_video_set(self, board, test_videos_with_charuco):
        """Accepts VideoSet object."""
        with VideoSet(test_videos_with_charuco) as vs:
            result = detect_all_frames(vs, board)

        assert isinstance(result, DetectionResult)

    def test_frame_step(self, board, test_videos_with_charuco):
        """Respects frame_step parameter."""
        result = detect_all_frames(test_videos_with_charuco, board, frame_step=2)

        # Should process frames 0, 2, 4, 6, 8 (5 frames)
        processed_indices = set(result.frames.keys())
        expected = {0, 2, 4, 6, 8}
        assert processed_indices.issubset(expected)

    def test_min_corners_filtering(self, board, test_videos_with_charuco):
        """Filters detections by min_corners."""
        # With very high min_corners, might get fewer results
        result_low = detect_all_frames(test_videos_with_charuco, board, min_corners=4)
        result_high = detect_all_frames(
            test_videos_with_charuco, board, min_corners=100
        )

        # High threshold should have fewer or equal frames
        assert len(result_high.frames) <= len(result_low.frames)

    def test_progress_callback(self, board, test_videos_with_charuco):
        """Calls progress callback after each frame."""
        calls = []

        def callback(current, total):
            calls.append((current, total))

        detect_all_frames(test_videos_with_charuco, board, progress_callback=callback)

        assert len(calls) == 10  # 10 frames
        assert all(total == 10 for _, total in calls)

    def test_progress_callback_with_frame_step(self, board, test_videos_with_charuco):
        """Callback receives sequential processed counts with frame_step > 1."""
        calls = []

        def callback(current, total):
            calls.append((current, total))

        # With 10 total frames and frame_step=2, should process 5 frames
        detect_all_frames(
            test_videos_with_charuco, board, frame_step=2, progress_callback=callback
        )

        # Should have 5 calls with sequential counts
        assert len(calls) == 5
        assert calls == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]

    def test_returns_correct_structure(self, board, test_videos_with_charuco):
        """Returns properly structured DetectionResult."""
        result = detect_all_frames(test_videos_with_charuco, board)

        for frame_idx, frame_det in result.frames.items():
            assert isinstance(frame_det, FrameDetections)
            assert frame_det.frame_idx == frame_idx
            for cam_name, det in frame_det.detections.items():
                assert isinstance(det, Detection)
                assert cam_name in result.camera_names

    def test_with_intrinsics(self, board, test_videos_with_charuco):
        """Uses intrinsics when provided."""
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        intrinsics = {"cam0": (K, dist), "cam1": (K, dist)}

        result = detect_all_frames(
            test_videos_with_charuco, board, intrinsics=intrinsics
        )

        assert isinstance(result, DetectionResult)

    def test_partial_intrinsics(self, board, test_videos_with_charuco):
        """Works with intrinsics for only some cameras."""
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
        intrinsics = {"cam0": (K, dist)}  # Only cam0

        result = detect_all_frames(
            test_videos_with_charuco, board, intrinsics=intrinsics
        )

        assert isinstance(result, DetectionResult)

    def test_get_frames_with_min_cameras(self, board, test_videos_with_charuco):
        """DetectionResult.get_frames_with_min_cameras works correctly."""
        result = detect_all_frames(test_videos_with_charuco, board)

        frames_2cam = result.get_frames_with_min_cameras(2)
        frames_1cam = result.get_frames_with_min_cameras(1)

        # 2-camera frames should be subset of 1-camera frames
        assert set(frames_2cam).issubset(set(frames_1cam))


class TestDetectAllFramesEdgeCases:
    def test_no_detections(self, board, tmp_path):
        """Handles videos with no detectable boards."""
        # Create blank video
        paths = {}
        for cam in ["cam0"]:
            path = tmp_path / f"{cam}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(path), fourcc, 30.0, (640, 480))
            for _ in range(5):
                writer.write(np.full((480, 640, 3), 128, dtype=np.uint8))
            writer.release()
            paths[cam] = str(path)

        result = detect_all_frames(paths, board)

        assert len(result.frames) == 0
        assert result.total_frames == 5

    def test_cleans_up_video_set(self, board, test_videos_with_charuco):
        """Cleans up internally created VideoSet."""
        # This is hard to test directly, but we can verify no errors occur
        result = detect_all_frames(test_videos_with_charuco, board)
        assert result is not None
