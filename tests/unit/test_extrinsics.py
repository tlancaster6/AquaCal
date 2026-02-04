"""Unit tests for extrinsic calibration module."""

import pytest
import numpy as np

from aquacal.config.schema import (
    BoardConfig,
    CameraIntrinsics,
    Detection,
    FrameDetections,
    DetectionResult,
    ConnectivityError,
)
from aquacal.core.board import BoardGeometry
from aquacal.calibration.extrinsics import (
    Observation,
    PoseGraph,
    estimate_board_pose,
    build_pose_graph,
    estimate_extrinsics,
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
def board(board_config) -> BoardGeometry:
    return BoardGeometry(board_config)


@pytest.fixture
def intrinsics() -> dict[str, CameraIntrinsics]:
    """Intrinsics for 3 cameras."""
    K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)
    return {
        "cam0": CameraIntrinsics(K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)),
        "cam1": CameraIntrinsics(K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)),
        "cam2": CameraIntrinsics(K=K.copy(), dist_coeffs=dist.copy(), image_size=(640, 480)),
    }


@pytest.fixture
def synthetic_detection(board) -> Detection:
    """Detection with all corners visible."""
    num_corners = board.num_corners
    corner_ids = np.arange(num_corners, dtype=np.int32)
    # Simulate corners in a grid pattern in image
    corners_2d = np.array(
        [[100 + (i % 5) * 80, 100 + (i // 5) * 80] for i in range(num_corners)],
        dtype=np.float64,
    )
    return Detection(corner_ids=corner_ids, corners_2d=corners_2d)


def make_connected_detections(cameras: list[str]) -> DetectionResult:
    """
    Create DetectionResult where all cameras are connected.

    Frame 0: cam0, cam1 see board
    Frame 1: cam1, cam2 see board
    (connects cam0-cam1-cam2 through cam1)
    """
    frames = {}

    # Frame 0: cam0 and cam1
    det = Detection(
        corner_ids=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
        corners_2d=np.array(
            [[100, 100], [180, 100], [260, 100], [100, 180], [180, 180], [260, 180]],
            dtype=np.float64,
        ),
    )
    frames[0] = FrameDetections(
        frame_idx=0,
        detections={
            "cam0": det,
            "cam1": Detection(
                corner_ids=det.corner_ids.copy(), corners_2d=det.corners_2d + 10
            ),
        },
    )

    # Frame 1: cam1 and cam2
    frames[1] = FrameDetections(
        frame_idx=1,
        detections={
            "cam1": Detection(
                corner_ids=det.corner_ids.copy(), corners_2d=det.corners_2d + 20
            ),
            "cam2": Detection(
                corner_ids=det.corner_ids.copy(), corners_2d=det.corners_2d + 30
            ),
        },
    )

    return DetectionResult(
        frames=frames,
        camera_names=["cam0", "cam1", "cam2"],
        total_frames=2,
    )


def make_disconnected_detections() -> DetectionResult:
    """Create DetectionResult with disconnected cameras (cam0, cam1 not linked to cam2, cam3)."""
    frames = {}

    det = Detection(
        corner_ids=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
        corners_2d=np.array(
            [[100, 100], [180, 100], [260, 100], [100, 180], [180, 180], [260, 180]],
            dtype=np.float64,
        ),
    )

    # Frame 0: only cam0 and cam1
    frames[0] = FrameDetections(
        frame_idx=0,
        detections={
            "cam0": det,
            "cam1": Detection(
                corner_ids=det.corner_ids.copy(), corners_2d=det.corners_2d + 10
            ),
        },
    )

    # Frame 1: only cam2 and cam3 (no connection to cam0/cam1)
    frames[1] = FrameDetections(
        frame_idx=1,
        detections={
            "cam2": Detection(
                corner_ids=det.corner_ids.copy(), corners_2d=det.corners_2d + 20
            ),
            "cam3": Detection(
                corner_ids=det.corner_ids.copy(), corners_2d=det.corners_2d + 30
            ),
        },
    )

    return DetectionResult(
        frames=frames,
        camera_names=["cam0", "cam1", "cam2", "cam3"],
        total_frames=2,
    )


class TestObservationDataclass:
    def test_observation_attributes(self):
        """Observation stores camera, frame_idx, corner_ids, and corners_2d."""
        obs = Observation(
            camera="cam0",
            frame_idx=5,
            corner_ids=np.array([0, 1, 2], dtype=np.int32),
            corners_2d=np.array([[100, 100], [200, 100], [150, 150]], dtype=np.float64),
        )
        assert obs.camera == "cam0"
        assert obs.frame_idx == 5
        assert obs.corner_ids.shape == (3,)
        assert obs.corners_2d.shape == (3, 2)


class TestPoseGraphDataclass:
    def test_pose_graph_attributes(self):
        """PoseGraph has camera_names, frame_indices, observations, adjacency."""
        graph = PoseGraph(
            camera_names=["cam0", "cam1"],
            frame_indices=[0, 1],
            observations=[],
            adjacency={"cam0": {"f0"}, "cam1": {"f0"}, "f0": {"cam0", "cam1"}},
        )
        assert graph.camera_names == ["cam0", "cam1"]
        assert graph.frame_indices == [0, 1]
        assert "cam0" in graph.adjacency
        assert "f0" in graph.adjacency["cam0"]


class TestEstimateBoardPose:
    def test_returns_rvec_tvec(self, board, intrinsics, synthetic_detection):
        """Returns (rvec, tvec) tuple."""
        result = estimate_board_pose(
            intrinsics["cam0"],
            synthetic_detection.corners_2d,
            synthetic_detection.corner_ids,
            board,
        )
        assert result is not None
        rvec, tvec = result
        assert rvec.shape == (3,)
        assert tvec.shape == (3,)

    def test_returns_none_for_few_points(self, board, intrinsics):
        """Returns None if fewer than 4 corners."""
        result = estimate_board_pose(
            intrinsics["cam0"],
            np.array([[100, 100], [200, 100], [150, 200]], dtype=np.float64),
            np.array([0, 1, 2], dtype=np.int32),
            board,
        )
        assert result is None

    def test_returns_float64_arrays(self, board, intrinsics, synthetic_detection):
        """Returns arrays with float64 dtype."""
        result = estimate_board_pose(
            intrinsics["cam0"],
            synthetic_detection.corners_2d,
            synthetic_detection.corner_ids,
            board,
        )
        assert result is not None
        rvec, tvec = result
        assert rvec.dtype == np.float64
        assert tvec.dtype == np.float64


class TestBuildPoseGraph:
    def test_builds_graph_from_detections(self):
        """Builds PoseGraph from connected detections."""
        detections = make_connected_detections(["cam0", "cam1", "cam2"])
        graph = build_pose_graph(detections)

        assert set(graph.camera_names) == {"cam0", "cam1", "cam2"}
        assert len(graph.frame_indices) == 2
        assert len(graph.observations) == 4  # 2 + 2 observations

    def test_raises_connectivity_error(self):
        """Raises ConnectivityError for disconnected graph."""
        detections = make_disconnected_detections()

        with pytest.raises(ConnectivityError) as exc_info:
            build_pose_graph(detections)

        # Error message should mention components
        assert "component" in str(exc_info.value).lower()

    def test_respects_min_cameras(self):
        """Filters frames by min_cameras."""
        # Create detection with frame having only 1 camera
        frames = {
            0: FrameDetections(
                frame_idx=0,
                detections={
                    "cam0": Detection(
                        corner_ids=np.array([0, 1, 2, 3], dtype=np.int32),
                        corners_2d=np.array(
                            [[100, 100], [200, 100], [100, 200], [200, 200]],
                            dtype=np.float64,
                        ),
                    ),
                },
            ),
        }
        detections = DetectionResult(
            frames=frames, camera_names=["cam0"], total_frames=1
        )

        with pytest.raises(ConnectivityError):
            build_pose_graph(detections, min_cameras=2)

    def test_adjacency_is_bipartite(self):
        """Adjacency connects cameras to frames and vice versa."""
        detections = make_connected_detections(["cam0", "cam1", "cam2"])
        graph = build_pose_graph(detections)

        # cam1 should be connected to both f0 and f1
        assert "f0" in graph.adjacency["cam1"]
        assert "f1" in graph.adjacency["cam1"]

        # f0 should be connected to cam0 and cam1
        assert "cam0" in graph.adjacency["f0"]
        assert "cam1" in graph.adjacency["f0"]

    def test_error_includes_component_details(self):
        """ConnectivityError message includes details about components."""
        detections = make_disconnected_detections()

        with pytest.raises(ConnectivityError) as exc_info:
            build_pose_graph(detections)

        error_msg = str(exc_info.value)
        # Should mention Component and list cameras
        assert "Component" in error_msg
        assert "cam0" in error_msg or "cam1" in error_msg


class TestEstimateExtrinsics:
    def test_reference_camera_at_origin(self, board, intrinsics):
        """Reference camera has identity pose."""
        detections = make_connected_detections(["cam0", "cam1", "cam2"])
        graph = build_pose_graph(detections)

        extrinsics_result = estimate_extrinsics(
            graph, intrinsics, board, reference_camera="cam0"
        )

        cam0 = extrinsics_result["cam0"]
        np.testing.assert_allclose(cam0.R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(cam0.t, np.zeros(3), atol=1e-10)

    def test_all_cameras_get_pose(self, board, intrinsics):
        """All cameras in graph receive extrinsics."""
        detections = make_connected_detections(["cam0", "cam1", "cam2"])
        graph = build_pose_graph(detections)

        extrinsics_result = estimate_extrinsics(graph, intrinsics, board)

        assert set(extrinsics_result.keys()) == {"cam0", "cam1", "cam2"}
        for cam, ext in extrinsics_result.items():
            assert ext.R.shape == (3, 3)
            assert ext.t.shape == (3,)

    def test_default_reference_camera(self, board, intrinsics):
        """Uses first sorted camera as default reference."""
        detections = make_connected_detections(["cam0", "cam1", "cam2"])
        graph = build_pose_graph(detections)

        extrinsics_result = estimate_extrinsics(graph, intrinsics, board)

        # cam0 should be at origin (first sorted)
        np.testing.assert_allclose(extrinsics_result["cam0"].R, np.eye(3), atol=1e-10)

    def test_raises_for_invalid_reference(self, board, intrinsics):
        """Raises ValueError for invalid reference camera."""
        detections = make_connected_detections(["cam0", "cam1", "cam2"])
        graph = build_pose_graph(detections)

        with pytest.raises(ValueError, match="not in pose graph"):
            estimate_extrinsics(graph, intrinsics, board, reference_camera="camX")

    def test_raises_for_missing_intrinsics(self, board):
        """Raises ValueError if intrinsics missing for a camera."""
        detections = make_connected_detections(["cam0", "cam1", "cam2"])
        graph = build_pose_graph(detections)

        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        partial_intrinsics = {
            "cam0": CameraIntrinsics(
                K=K, dist_coeffs=np.zeros(5), image_size=(640, 480)
            ),
            # cam1 and cam2 missing
        }

        with pytest.raises(ValueError, match="Missing intrinsics"):
            estimate_extrinsics(graph, partial_intrinsics, board)

    def test_different_reference_camera(self, board, intrinsics):
        """Can use different camera as reference."""
        detections = make_connected_detections(["cam0", "cam1", "cam2"])
        graph = build_pose_graph(detections)

        extrinsics_result = estimate_extrinsics(
            graph, intrinsics, board, reference_camera="cam1"
        )

        # cam1 should be at origin
        np.testing.assert_allclose(extrinsics_result["cam1"].R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(extrinsics_result["cam1"].t, np.zeros(3), atol=1e-10)

    def test_extrinsics_are_valid_rotations(self, board, intrinsics):
        """All rotation matrices are valid (orthonormal, det=1)."""
        detections = make_connected_detections(["cam0", "cam1", "cam2"])
        graph = build_pose_graph(detections)

        extrinsics_result = estimate_extrinsics(graph, intrinsics, board)

        for cam, ext in extrinsics_result.items():
            # Check orthonormality: R @ R.T should be identity
            np.testing.assert_allclose(ext.R @ ext.R.T, np.eye(3), atol=1e-6)
            # Check determinant is +1 (proper rotation)
            np.testing.assert_allclose(np.linalg.det(ext.R), 1.0, atol=1e-6)
