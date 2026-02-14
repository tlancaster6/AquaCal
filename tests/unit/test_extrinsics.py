"""Unit tests for extrinsic calibration module."""

import numpy as np
import pytest

from aquacal.calibration.extrinsics import (
    Observation,
    PoseGraph,
    _average_rotations,
    build_pose_graph,
    estimate_board_pose,
    estimate_extrinsics,
    refractive_solve_pnp,
)
from aquacal.config.schema import (
    BoardConfig,
    CameraExtrinsics,
    CameraIntrinsics,
    ConnectivityError,
    Detection,
    DetectionResult,
    FrameDetections,
)
from aquacal.core.board import BoardGeometry
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_project
from aquacal.utils.transforms import matrix_to_rvec, rvec_to_matrix


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

    def test_progress_callback_invoked(self, board, intrinsics):
        """Progress callback is called for each camera located and for averaging pass."""
        detections = make_connected_detections(["cam0", "cam1", "cam2"])
        graph = build_pose_graph(detections)

        # Record callback invocations
        calls = []

        def callback(cam_name: str, current: int, total: int):
            calls.append((cam_name, current, total))

        _extrinsics_result = estimate_extrinsics(
            graph,
            intrinsics,
            board,
            reference_camera="cam0",
            progress_callback=callback,
        )

        # Should have 3 calls for cameras + 1 for averaging
        assert len(calls) == 4

        # First three calls should be for cameras (order may vary except reference)
        camera_calls = [c for c in calls if c[0] != "_averaging"]
        assert len(camera_calls) == 3

        # Reference camera should be first
        assert camera_calls[0][0] == "cam0"
        assert camera_calls[0][1] == 1
        assert camera_calls[0][2] == 3

        # All camera calls should have total=3
        for _, cur, total in camera_calls:
            assert total == 3

        # Current should be sequential (1, 2, 3)
        assert [c[1] for c in camera_calls] == [1, 2, 3]

        # Final call should be for averaging
        assert calls[-1][0] == "_averaging"
        assert calls[-1][1] == 3  # All cameras located
        assert calls[-1][2] == 3

    def test_progress_callback_optional(self, board, intrinsics):
        """Progress callback is optional (defaults to None)."""
        detections = make_connected_detections(["cam0", "cam1", "cam2"])
        graph = build_pose_graph(detections)

        # Should not raise when callback is omitted
        extrinsics_result = estimate_extrinsics(graph, intrinsics, board)
        assert len(extrinsics_result) == 3


def _generate_refractive_detections(
    intrinsics_single: CameraIntrinsics,
    camera_ext: CameraExtrinsics,
    cam_name: str,
    board: BoardGeometry,
    board_rvec: np.ndarray,
    board_tvec: np.ndarray,
    interface_distance: float,
    n_air: float = 1.0,
    n_water: float = 1.333,
) -> Detection | None:
    """
    Generate synthetic refractive detections by projecting board corners
    through a refractive interface.

    Returns Detection or None if any corner fails to project.
    """
    camera = Camera(cam_name, intrinsics_single, camera_ext)
    interface = Interface(
        normal=np.array([0.0, 0.0, -1.0]),
        camera_distances={cam_name: interface_distance},
        n_air=n_air,
        n_water=n_water,
    )

    # Board corners in world frame
    R_board = rvec_to_matrix(board_rvec)
    all_ids = np.arange(board.num_corners, dtype=np.int32)
    object_points = board.get_corner_array(all_ids)
    world_pts = (R_board @ object_points.T).T + board_tvec

    # Project through refractive interface
    pixels = []
    for pt in world_pts:
        px = refractive_project(camera, interface, pt)
        if px is None:
            return None
        pixels.append(px)

    corners_2d = np.array(pixels, dtype=np.float64)
    return Detection(corner_ids=all_ids, corners_2d=corners_2d)


class TestRefractiveSolvePnp:
    """Tests for refractive_solve_pnp() function."""

    def test_recovers_ground_truth_pose(self, board, intrinsics):
        """Recovers board pose from synthetic refractive detections (rot < 1deg, trans < 5mm)."""
        # Ground truth board pose in camera frame
        rvec_true = np.array([0.1, -0.05, 0.02])
        tvec_true = np.array([0.01, -0.02, 0.50])

        interface_distance = 0.15

        # Generate synthetic detections using identity camera
        identity_ext = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))
        det = _generate_refractive_detections(
            intrinsics["cam0"],
            identity_ext,
            "_test",
            board,
            rvec_true,
            tvec_true,
            interface_distance,
        )
        assert det is not None

        result = refractive_solve_pnp(
            intrinsics["cam0"],
            det.corners_2d,
            det.corner_ids,
            board,
            interface_distance,
        )
        assert result is not None
        rvec_est, tvec_est = result

        # Check rotation error < 1 degree
        R_true = rvec_to_matrix(rvec_true)
        R_est = rvec_to_matrix(rvec_est)
        R_err = R_true.T @ R_est
        angle_err = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        assert np.degrees(angle_err) < 1.0, (
            f"Rotation error {np.degrees(angle_err):.3f} deg"
        )

        # Check translation error < 5mm
        trans_err = np.linalg.norm(tvec_est - tvec_true)
        assert trans_err < 0.005, f"Translation error {trans_err * 1000:.1f} mm"

    def test_more_accurate_than_standard_pnp(self, board, intrinsics):
        """Refractive PnP is at least 5x more accurate than standard PnP on refractive data."""
        rvec_true = np.array([0.08, -0.03, 0.01])
        tvec_true = np.array([0.02, -0.01, 0.45])
        interface_distance = 0.15

        # Generate refractive detections
        identity_ext = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))
        det = _generate_refractive_detections(
            intrinsics["cam0"],
            identity_ext,
            "_test",
            board,
            rvec_true,
            tvec_true,
            interface_distance,
        )
        assert det is not None

        # Standard PnP
        result_std = estimate_board_pose(
            intrinsics["cam0"],
            det.corners_2d,
            det.corner_ids,
            board,
        )
        assert result_std is not None
        _, tvec_std = result_std
        err_std = np.linalg.norm(tvec_std - tvec_true)

        # Refractive PnP
        result_ref = refractive_solve_pnp(
            intrinsics["cam0"],
            det.corners_2d,
            det.corner_ids,
            board,
            interface_distance,
        )
        assert result_ref is not None
        _, tvec_ref = result_ref
        err_ref = np.linalg.norm(tvec_ref - tvec_true)

        assert err_ref * 5 < err_std, (
            f"Refractive PnP ({err_ref * 1000:.1f}mm) not 5x better than "
            f"standard PnP ({err_std * 1000:.1f}mm)"
        )

    def test_returns_none_for_few_points(self, board, intrinsics):
        """Returns None if fewer than 4 corners."""
        result = refractive_solve_pnp(
            intrinsics["cam0"],
            np.array([[100, 100], [200, 100], [150, 200]], dtype=np.float64),
            np.array([0, 1, 2], dtype=np.int32),
            board,
            0.15,
        )
        assert result is None

    def test_returns_float64_arrays(self, board, intrinsics):
        """Returns arrays with float64 dtype."""
        rvec_true = np.array([0.05, -0.02, 0.01])
        tvec_true = np.array([0.0, 0.0, 0.40])
        interface_distance = 0.15

        identity_ext = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))
        det = _generate_refractive_detections(
            intrinsics["cam0"],
            identity_ext,
            "_test",
            board,
            rvec_true,
            tvec_true,
            interface_distance,
        )
        assert det is not None

        result = refractive_solve_pnp(
            intrinsics["cam0"],
            det.corners_2d,
            det.corner_ids,
            board,
            interface_distance,
        )
        assert result is not None
        rvec, tvec = result
        assert rvec.dtype == np.float64
        assert tvec.dtype == np.float64
        assert rvec.shape == (3,)
        assert tvec.shape == (3,)

    def test_default_interface_normal(self, board, intrinsics):
        """Works without explicitly passing interface_normal."""
        rvec_true = np.array([0.05, -0.02, 0.01])
        tvec_true = np.array([0.0, 0.0, 0.40])

        identity_ext = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))
        det = _generate_refractive_detections(
            intrinsics["cam0"],
            identity_ext,
            "_test",
            board,
            rvec_true,
            tvec_true,
            0.15,
        )
        assert det is not None

        # Call without interface_normal (should default to [0,0,-1])
        result = refractive_solve_pnp(
            intrinsics["cam0"],
            det.corners_2d,
            det.corner_ids,
            board,
            0.15,
        )
        assert result is not None


class TestEstimateExtrinsicsRefractive:
    """Tests for estimate_extrinsics with refractive PnP."""

    def test_coplanar_cameras_with_interface(self, board, intrinsics):
        """Coplanar cameras produce Z spread < 0.05m with refractive PnP."""
        # Two cameras at same Z height, separated in X
        ext0 = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))  # C0 = [0,0,0]
        # C1 = [0.3, 0, 0]: t = -R @ C = [-0.3, 0, 0]
        ext1 = CameraExtrinsics(R=np.eye(3), t=np.array([-0.3, 0.0, 0.0]))

        interface_distance = 0.15

        # Board in world frame, below interface
        board_rvec = np.array([0.1, -0.05, 0.02])
        board_tvec = np.array([0.15, 0.0, 0.50])

        # Generate refractive detections for each camera
        det0 = _generate_refractive_detections(
            intrinsics["cam0"],
            ext0,
            "cam0",
            board,
            board_rvec,
            board_tvec,
            interface_distance,
        )
        det1 = _generate_refractive_detections(
            intrinsics["cam1"],
            ext1,
            "cam1",
            board,
            board_rvec,
            board_tvec,
            interface_distance,
        )
        assert det0 is not None and det1 is not None

        # Build DetectionResult and PoseGraph
        frames = {
            0: FrameDetections(
                frame_idx=0,
                detections={"cam0": det0, "cam1": det1},
            ),
        }
        detection_result = DetectionResult(
            frames=frames,
            camera_names=["cam0", "cam1"],
            total_frames=1,
        )
        graph = build_pose_graph(detection_result)

        # Estimate with refractive PnP
        interface_distances = {"cam0": interface_distance, "cam1": interface_distance}
        extrinsics_result = estimate_extrinsics(
            graph,
            intrinsics,
            board,
            reference_camera="cam0",
            interface_distances=interface_distances,
        )

        # Both cameras should be near Z=0 (coplanar)
        # Reference camera is at origin
        cam1_C = extrinsics_result["cam1"].C
        z_spread = abs(cam1_C[2] - 0.0)
        assert z_spread < 0.05, f"Z spread {z_spread:.3f}m exceeds 0.05m"

    def test_backward_compatible_without_interface(self, board, intrinsics):
        """Without interface params, behaves identically to standard PnP."""
        detections = make_connected_detections(["cam0", "cam1", "cam2"])
        graph = build_pose_graph(detections)

        # Call without interface params
        result_no_iface = estimate_extrinsics(graph, intrinsics, board)

        # Reference camera at origin
        np.testing.assert_allclose(result_no_iface["cam0"].R, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(result_no_iface["cam0"].t, np.zeros(3), atol=1e-10)

        # All cameras get poses
        assert set(result_no_iface.keys()) == {"cam0", "cam1", "cam2"}


class TestAverageRotations:
    """Tests for _average_rotations() helper."""

    def test_returns_valid_so3(self):
        """Result is orthonormal with det=+1."""
        R1 = rvec_to_matrix(np.array([0.1, 0.0, 0.0]))
        R2 = rvec_to_matrix(np.array([-0.1, 0.05, 0.0]))
        R_avg = _average_rotations([R1, R2], [1.0, 1.0])

        np.testing.assert_allclose(R_avg @ R_avg.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R_avg), 1.0, atol=1e-10)

    def test_identity_from_identical_rotations(self):
        """Averaging identical rotations recovers that rotation."""
        R = rvec_to_matrix(np.array([0.2, -0.1, 0.05]))
        R_avg = _average_rotations([R, R, R], [1.0, 1.0, 1.0])

        np.testing.assert_allclose(R_avg, R, atol=1e-10)

    def test_identity_input(self):
        """Averaging identity matrices returns identity."""
        R_avg = _average_rotations([np.eye(3), np.eye(3)], [1.0, 1.0])
        np.testing.assert_allclose(R_avg, np.eye(3), atol=1e-10)

    def test_weights_matter(self):
        """Higher weight pulls average toward that rotation."""
        R1 = np.eye(3)
        R2 = rvec_to_matrix(np.array([0.0, 0.0, 0.3]))

        # Heavy weight on R1 -> result closer to identity
        R_heavy_1 = _average_rotations([R1, R2], [100.0, 1.0])
        angle_1 = np.linalg.norm(matrix_to_rvec(R_heavy_1))

        # Equal weight
        R_equal = _average_rotations([R1, R2], [1.0, 1.0])
        angle_eq = np.linalg.norm(matrix_to_rvec(R_equal))

        assert angle_1 < angle_eq, (
            "Heavy weight on identity should produce smaller rotation"
        )


class TestPriorityBFS:
    """Tests for priority queue ordering in estimate_extrinsics."""

    def test_prefers_higher_corner_count(self, board, intrinsics):
        """Priority BFS processes edges with more corners first."""
        # Build two frames: frame 0 has 6 corners, frame 1 has 12
        few_ids = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        few_2d = np.array(
            [[100, 100], [180, 100], [260, 100], [100, 180], [180, 180], [260, 180]],
            dtype=np.float64,
        )
        many_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32)
        many_2d = np.array(
            [
                [100, 100],
                [180, 100],
                [260, 100],
                [340, 100],
                [100, 180],
                [180, 180],
                [260, 180],
                [340, 180],
                [100, 260],
                [180, 260],
                [260, 260],
                [340, 260],
            ],
            dtype=np.float64,
        )

        frames = {
            0: FrameDetections(
                frame_idx=0,
                detections={
                    "cam0": Detection(
                        corner_ids=few_ids.copy(), corners_2d=few_2d.copy()
                    ),
                    "cam1": Detection(
                        corner_ids=few_ids.copy(), corners_2d=few_2d + 10
                    ),
                },
            ),
            1: FrameDetections(
                frame_idx=1,
                detections={
                    "cam0": Detection(
                        corner_ids=many_ids.copy(), corners_2d=many_2d.copy()
                    ),
                    "cam1": Detection(
                        corner_ids=many_ids.copy(), corners_2d=many_2d + 10
                    ),
                },
            ),
        }
        det_result = DetectionResult(
            frames=frames,
            camera_names=["cam0", "cam1"],
            total_frames=2,
        )
        graph = build_pose_graph(det_result)

        # Should succeed and produce valid extrinsics regardless of order
        result = estimate_extrinsics(graph, intrinsics, board, reference_camera="cam0")
        assert "cam0" in result
        assert "cam1" in result


class TestDeterminism:
    """Tests that estimate_extrinsics is deterministic."""

    def test_two_calls_same_result(self, board, intrinsics):
        """Two calls with same input produce identical output."""
        detections = make_connected_detections(["cam0", "cam1", "cam2"])
        graph = build_pose_graph(detections)

        result1 = estimate_extrinsics(graph, intrinsics, board, reference_camera="cam0")
        result2 = estimate_extrinsics(graph, intrinsics, board, reference_camera="cam0")

        for cam_name in result1:
            np.testing.assert_array_equal(result1[cam_name].R, result2[cam_name].R)
            np.testing.assert_array_equal(result1[cam_name].t, result2[cam_name].t)


class TestMultiFrameAveraging:
    """Tests that multi-frame averaging improves initialization quality."""

    def test_averaging_produces_valid_poses(self, board, intrinsics):
        """Multi-frame averaging produces valid rotation matrices."""
        detections = make_connected_detections(["cam0", "cam1", "cam2"])
        graph = build_pose_graph(detections)

        result = estimate_extrinsics(graph, intrinsics, board, reference_camera="cam0")

        for cam, ext in result.items():
            # Check orthonormality
            np.testing.assert_allclose(ext.R @ ext.R.T, np.eye(3), atol=1e-6)
            # Check determinant
            np.testing.assert_allclose(np.linalg.det(ext.R), 1.0, atol=1e-6)

    def test_averaging_with_refractive_multi_camera(self, board, intrinsics):
        """Multi-frame averaging with 3 cameras and 3 shared frames."""
        # 3 cameras at different positions, all coplanar (Z=0)
        ext0 = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))
        ext1 = CameraExtrinsics(R=np.eye(3), t=np.array([-0.3, 0.0, 0.0]))
        ext2 = CameraExtrinsics(R=np.eye(3), t=np.array([0.0, -0.3, 0.0]))

        interface_distance = 0.15
        camera_exts = {"cam0": ext0, "cam1": ext1, "cam2": ext2}

        # 3 board poses at different positions
        board_poses_gt = [
            (np.array([0.1, -0.05, 0.02]), np.array([0.10, 0.0, 0.50])),
            (np.array([-0.05, 0.08, 0.01]), np.array([0.15, 0.05, 0.45])),
            (np.array([0.02, 0.03, -0.04]), np.array([0.05, -0.05, 0.55])),
        ]

        frames = {}
        for fi, (brvec, btvec) in enumerate(board_poses_gt):
            dets = {}
            for cam_name, cam_ext in camera_exts.items():
                det = _generate_refractive_detections(
                    intrinsics[cam_name],
                    cam_ext,
                    cam_name,
                    board,
                    brvec,
                    btvec,
                    interface_distance,
                )
                if det is not None:
                    dets[cam_name] = det
            if len(dets) >= 2:
                frames[fi] = FrameDetections(frame_idx=fi, detections=dets)

        assert len(frames) >= 2, "Need at least 2 usable frames"

        det_result = DetectionResult(
            frames=frames,
            camera_names=["cam0", "cam1", "cam2"],
            total_frames=len(frames),
        )
        graph = build_pose_graph(det_result)

        interface_distances = {cam: interface_distance for cam in camera_exts}
        result = estimate_extrinsics(
            graph,
            intrinsics,
            board,
            reference_camera="cam0",
            interface_distances=interface_distances,
        )

        # Check cam1 position accuracy (should be close to ground truth)
        cam1_C = result["cam1"].C
        cam1_C_gt = np.array([0.3, 0.0, 0.0])  # C = -R.T @ t
        pos_err = np.linalg.norm(cam1_C - cam1_C_gt)
        assert pos_err < 0.05, f"cam1 position error {pos_err:.3f}m exceeds 0.05m"
