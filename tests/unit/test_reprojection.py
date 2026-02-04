"""Tests for reprojection error computation."""

import pytest
import numpy as np

from aquacal.config.schema import (
    BoardConfig,
    BoardPose,
    CameraIntrinsics,
    CameraExtrinsics,
    CameraCalibration,
    CalibrationResult,
    InterfaceParams,
    DiagnosticsData,
    CalibrationMetadata,
    Detection,
    FrameDetections,
    DetectionResult,
)
from aquacal.core.board import BoardGeometry
from aquacal.core.camera import Camera
from aquacal.core.interface_model import Interface
from aquacal.core.refractive_geometry import refractive_project
from aquacal.validation.reprojection import (
    compute_reprojection_errors,
    compute_reprojection_error_single,
    ReprojectionErrors,
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
    """Camera extrinsics for 3 cameras.

    Note: Camera offsets kept small (0.08m) so that the board at Z=0.35m
    is visible in all cameras' 640x480 field of view.
    """
    return {
        "cam0": CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.zeros(3, dtype=np.float64),
        ),
        "cam1": CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.08, 0.0, 0.0], dtype=np.float64),
        ),
        "cam2": CameraExtrinsics(
            R=np.eye(3, dtype=np.float64),
            t=np.array([0.0, 0.08, 0.0], dtype=np.float64),
        ),
    }


@pytest.fixture
def interface_distances() -> dict[str, float]:
    return {"cam0": 0.15, "cam1": 0.16, "cam2": 0.14}


@pytest.fixture
def board_poses() -> dict[int, BoardPose]:
    """Board poses for 5 frames underwater."""
    poses = {}
    for i in range(5):
        x_offset = 0.03 * (i % 3 - 1)
        y_offset = 0.03 * (i // 3 - 0.5)
        poses[i] = BoardPose(
            frame_idx=i,
            rvec=np.array([0.05 * i, 0.0, 0.0], dtype=np.float64),
            tvec=np.array([x_offset, y_offset, 0.35], dtype=np.float64),
        )
    return poses


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


def generate_synthetic_detections(
    intrinsics: dict[str, CameraIntrinsics],
    extrinsics: dict[str, CameraExtrinsics],
    interface_distances: dict[str, float],
    board: BoardGeometry,
    board_poses: dict[int, BoardPose],
    noise_std: float = 0.0,
) -> DetectionResult:
    """Generate synthetic detections using refractive_project."""
    interface_normal = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    frames = {}

    for frame_idx, bp in board_poses.items():
        corners_3d = board.transform_corners(bp.rvec, bp.tvec)
        detections_dict = {}

        for cam_name in intrinsics:
            camera = Camera(cam_name, intrinsics[cam_name], extrinsics[cam_name])
            interface = Interface(
                normal=interface_normal,
                base_height=0.0,
                camera_offsets={cam_name: interface_distances[cam_name]},
            )

            corner_ids = []
            corners_2d = []

            for corner_id in range(board.num_corners):
                point_3d = corners_3d[corner_id]
                projected = refractive_project(camera, interface, point_3d)

                if projected is not None:
                    w, h = intrinsics[cam_name].image_size
                    if 0 <= projected[0] < w and 0 <= projected[1] < h:
                        corner_ids.append(corner_id)
                        px = projected.copy()
                        if noise_std > 0:
                            px += np.random.normal(0, noise_std, 2)
                        corners_2d.append(px)

            if len(corner_ids) >= 4:
                detections_dict[cam_name] = Detection(
                    corner_ids=np.array(corner_ids, dtype=np.int32),
                    corners_2d=np.array(corners_2d, dtype=np.float64),
                )

        if detections_dict:
            frames[frame_idx] = FrameDetections(
                frame_idx=frame_idx,
                detections=detections_dict,
            )

    return DetectionResult(
        frames=frames,
        camera_names=list(intrinsics.keys()),
        total_frames=len(board_poses),
    )


class TestReprojectionErrors:
    """Tests for reprojection error computation."""

    def test_perfect_reprojection_zero_error(
        self,
        calibration_result,
        intrinsics,
        extrinsics,
        interface_distances,
        board,
        board_poses,
    ):
        """Test that perfect synthetic data gives near-zero reprojection error."""
        # Generate synthetic detections with no noise
        detections = generate_synthetic_detections(
            intrinsics, extrinsics, interface_distances, board, board_poses, noise_std=0.0
        )

        # Compute reprojection errors
        errors = compute_reprojection_errors(calibration_result, detections, board_poses)

        # Should be very close to zero
        assert errors.rms < 1e-6
        assert errors.num_observations > 0
        assert errors.residuals.shape == (errors.num_observations, 2)

    def test_noisy_reprojection_matches_noise(
        self,
        calibration_result,
        intrinsics,
        extrinsics,
        interface_distances,
        board,
        board_poses,
    ):
        """Test that noisy synthetic data gives expected RMS error."""
        noise_std = 0.5
        np.random.seed(42)

        # Generate synthetic detections with noise
        detections = generate_synthetic_detections(
            intrinsics,
            extrinsics,
            interface_distances,
            board,
            board_poses,
            noise_std=noise_std,
        )

        # Compute reprojection errors
        errors = compute_reprojection_errors(calibration_result, detections, board_poses)

        # RMS should be approximately equal to noise_std
        # Allow for some statistical variation (within 30%)
        assert 0.3 < errors.rms < 0.8
        assert errors.num_observations > 0

    def test_per_camera_breakdown(
        self,
        calibration_result,
        intrinsics,
        extrinsics,
        interface_distances,
        board,
        board_poses,
    ):
        """Test that per-camera errors are computed correctly."""
        detections = generate_synthetic_detections(
            intrinsics, extrinsics, interface_distances, board, board_poses, noise_std=0.0
        )

        errors = compute_reprojection_errors(calibration_result, detections, board_poses)

        # All cameras should have entries
        assert len(errors.per_camera) == 3
        assert "cam0" in errors.per_camera
        assert "cam1" in errors.per_camera
        assert "cam2" in errors.per_camera

        # Each camera should have near-zero error for perfect data
        for cam_name, rms in errors.per_camera.items():
            assert rms < 1e-6

    def test_per_frame_breakdown(
        self,
        calibration_result,
        intrinsics,
        extrinsics,
        interface_distances,
        board,
        board_poses,
    ):
        """Test that per-frame errors are computed correctly."""
        detections = generate_synthetic_detections(
            intrinsics, extrinsics, interface_distances, board, board_poses, noise_std=0.0
        )

        errors = compute_reprojection_errors(calibration_result, detections, board_poses)

        # All frames should have entries
        assert len(errors.per_frame) == 5
        for i in range(5):
            assert i in errors.per_frame

        # Each frame should have near-zero error for perfect data
        for frame_idx, rms in errors.per_frame.items():
            assert rms < 1e-6

    def test_residuals_shape(
        self,
        calibration_result,
        intrinsics,
        extrinsics,
        interface_distances,
        board,
        board_poses,
    ):
        """Test that residuals array has correct shape."""
        detections = generate_synthetic_detections(
            intrinsics, extrinsics, interface_distances, board, board_poses, noise_std=0.0
        )

        errors = compute_reprojection_errors(calibration_result, detections, board_poses)

        # Residuals should be (N, 2)
        assert errors.residuals.ndim == 2
        assert errors.residuals.shape[1] == 2
        assert errors.residuals.shape[0] == errors.num_observations

    def test_single_camera_single_frame(
        self, intrinsics, extrinsics, interface_distances, board, board_poses
    ):
        """Test compute_reprojection_error_single() in isolation."""
        # Use first frame and first camera
        board_pose = board_poses[0]
        cam_name = "cam0"

        camera = Camera(cam_name, intrinsics[cam_name], extrinsics[cam_name])
        interface = Interface(
            normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            base_height=0.0,
            camera_offsets={cam_name: interface_distances[cam_name]},
        )

        # Generate synthetic detection for this camera/frame
        corners_3d = board.transform_corners(board_pose.rvec, board_pose.tvec)
        corner_ids = []
        corners_2d = []

        for corner_id in range(board.num_corners):
            point_3d = corners_3d[corner_id]
            projected = refractive_project(camera, interface, point_3d)

            if projected is not None:
                w, h = intrinsics[cam_name].image_size
                if 0 <= projected[0] < w and 0 <= projected[1] < h:
                    corner_ids.append(corner_id)
                    corners_2d.append(projected.copy())

        detection = Detection(
            corner_ids=np.array(corner_ids, dtype=np.int32),
            corners_2d=np.array(corners_2d, dtype=np.float64),
        )

        # Compute reprojection error
        residuals, valid_ids = compute_reprojection_error_single(
            camera, interface, board, board_pose, detection
        )

        # Should not be None
        assert residuals is not None
        assert valid_ids is not None

        # Should have same number of valid corners as detected
        assert len(valid_ids) == len(corner_ids)
        assert residuals.shape == (len(valid_ids), 2)

        # Error should be near zero for perfect data
        rms = np.sqrt(np.mean(residuals[:, 0] ** 2 + residuals[:, 1] ** 2))
        assert rms < 1e-6

    def test_handles_projection_failures(
        self, calibration_result, intrinsics, extrinsics, interface_distances, board
    ):
        """Test graceful handling when some projections fail."""
        # Create a board pose where board is very close to interface (may cause TIR)
        board_pose = BoardPose(
            frame_idx=0,
            rvec=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            tvec=np.array([0.0, 0.0, 0.16], dtype=np.float64),  # Very close to interface
        )

        # Generate detections - some corners may fail to project
        detections = generate_synthetic_detections(
            intrinsics,
            extrinsics,
            interface_distances,
            board,
            {0: board_pose},
            noise_std=0.0,
        )

        # Should not crash even if some projections fail
        errors = compute_reprojection_errors(
            calibration_result, detections, {0: board_pose}
        )

        # Should have some valid observations (or possibly none)
        assert errors.num_observations >= 0
        assert errors.residuals.shape[0] == errors.num_observations

    def test_empty_detections(self, calibration_result):
        """Test graceful handling when no detections present."""
        # Create empty detection result
        detections = DetectionResult(
            frames={},
            camera_names=["cam0", "cam1", "cam2"],
            total_frames=0,
        )

        board_poses = {}

        # Should not crash
        errors = compute_reprojection_errors(calibration_result, detections, board_poses)

        # Should have zero observations
        assert errors.num_observations == 0
        assert errors.rms == 0.0
        assert errors.residuals.shape == (0, 2)
        assert len(errors.per_camera) == 0
        assert len(errors.per_frame) == 0

    def test_single_returns_none_for_no_valid_projections(
        self, intrinsics, extrinsics, interface_distances, board
    ):
        """Test that compute_reprojection_error_single returns None when no corners project."""
        # Create a board pose above the interface (in air)
        board_pose = BoardPose(
            frame_idx=0,
            rvec=np.array([0.0, 0.0, 0.0], dtype=np.float64),
            tvec=np.array([0.0, 0.0, -0.1], dtype=np.float64),  # Above interface
        )

        cam_name = "cam0"
        camera = Camera(cam_name, intrinsics[cam_name], extrinsics[cam_name])
        interface = Interface(
            normal=np.array([0.0, 0.0, -1.0], dtype=np.float64),
            base_height=0.0,
            camera_offsets={cam_name: interface_distances[cam_name]},
        )

        # Create a detection with some corners
        detection = Detection(
            corner_ids=np.array([0, 1, 2], dtype=np.int32),
            corners_2d=np.array([[100, 100], [200, 200], [300, 300]], dtype=np.float64),
        )

        # Should return None, None since board is above interface
        residuals, valid_ids = compute_reprojection_error_single(
            camera, interface, board, board_pose, detection
        )

        assert residuals is None
        assert valid_ids is None
