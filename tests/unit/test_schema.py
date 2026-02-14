"""Unit tests for configuration schema and dataclasses."""

import numpy as np
import pytest
from pathlib import Path

from aquacal.config.schema import (
    Vec2,
    Vec3,
    Mat3,
    BoardConfig,
    CameraIntrinsics,
    CameraExtrinsics,
    CameraCalibration,
    InterfaceParams,
    CalibrationResult,
    DiagnosticsData,
    CalibrationMetadata,
    CalibrationConfig,
    BoardPose,
    Detection,
    FrameDetections,
    DetectionResult,
    CalibrationError,
    InsufficientDataError,
    ConvergenceError,
    ConnectivityError,
)


class TestTypeAliases:
    """Test that type aliases are defined and usable."""

    def test_vec3_alias(self):
        """Vec3 should accept 3-element float arrays."""
        vec: Vec3 = np.array([1.0, 2.0, 3.0])
        assert vec.shape == (3,)
        assert vec.dtype == np.float64

    def test_mat3_alias(self):
        """Mat3 should accept 3x3 float arrays."""
        mat: Mat3 = np.eye(3)
        assert mat.shape == (3, 3)
        assert mat.dtype == np.float64

    def test_vec2_alias(self):
        """Vec2 should accept 2-element float arrays."""
        vec: Vec2 = np.array([1.0, 2.0])
        assert vec.shape == (2,)
        assert vec.dtype == np.float64


class TestBoardConfig:
    """Test BoardConfig dataclass."""

    def test_creation(self):
        """Should create valid board config."""
        board = BoardConfig(
            squares_x=5,
            squares_y=7,
            square_size=0.04,
            marker_size=0.03,
            dictionary="DICT_4X4_50",
        )
        assert board.squares_x == 5
        assert board.squares_y == 7
        assert board.square_size == 0.04
        assert board.marker_size == 0.03
        assert board.dictionary == "DICT_4X4_50"


class TestCameraIntrinsics:
    """Test CameraIntrinsics dataclass."""

    def test_creation(self):
        """Should create valid intrinsics."""
        K = np.array([[1000, 0, 640], [0, 1000, 480], [0, 0, 1]], dtype=np.float64)
        dist = np.array([0.1, -0.2, 0.001, 0.002, 0.05], dtype=np.float64)
        intrinsics = CameraIntrinsics(K=K, dist_coeffs=dist, image_size=(1280, 960))

        assert intrinsics.K.shape == (3, 3)
        assert intrinsics.dist_coeffs.shape == (5,)
        assert intrinsics.image_size == (1280, 960)


class TestCameraExtrinsics:
    """Test CameraExtrinsics dataclass and computed properties."""

    def test_camera_center_at_origin(self):
        """Camera at origin should have C = [0,0,0]."""
        ext = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))
        np.testing.assert_allclose(ext.C, np.zeros(3))

    def test_camera_center_translated(self):
        """Camera with t=[0,0,5] and R=I should have C=[0,0,-5]."""
        ext = CameraExtrinsics(R=np.eye(3), t=np.array([0.0, 0.0, 5.0]))
        np.testing.assert_allclose(ext.C, np.array([0.0, 0.0, -5.0]))

    def test_camera_center_rotated_and_translated(self):
        """Test camera center with arbitrary rotation and translation."""
        # 90 degree rotation about Z axis
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        t = np.array([1.0, 2.0, 3.0])
        ext = CameraExtrinsics(R=R, t=t)

        # C = -R.T @ t
        expected_C = -R.T @ t
        np.testing.assert_allclose(ext.C, expected_C)


class TestCameraCalibration:
    """Test CameraCalibration dataclass."""

    def test_creation(self):
        """Should create valid camera calibration."""
        K = np.eye(3)
        dist = np.zeros(5)
        intrinsics = CameraIntrinsics(K=K, dist_coeffs=dist, image_size=(640, 480))
        extrinsics = CameraExtrinsics(R=np.eye(3), t=np.zeros(3))

        calib = CameraCalibration(
            name="cam0",
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            interface_distance=0.12,
        )

        assert calib.name == "cam0"
        assert calib.interface_distance == 0.12
        assert calib.intrinsics == intrinsics
        assert calib.extrinsics == extrinsics


class TestInterfaceParams:
    """Test InterfaceParams dataclass."""

    def test_default_values(self):
        """Should use default refractive indices."""
        interface = InterfaceParams(normal=np.array([0.0, 0.0, -1.0]))
        assert interface.n_air == 1.0
        assert interface.n_water == 1.333
        np.testing.assert_array_equal(interface.normal, [0, 0, -1])

    def test_custom_values(self):
        """Should allow custom refractive indices."""
        interface = InterfaceParams(
            normal=np.array([0.0, 0.0, -1.0]), n_air=1.0, n_water=1.34
        )
        assert interface.n_water == 1.34


class TestDiagnosticsData:
    """Test DiagnosticsData dataclass."""

    def test_required_fields(self):
        """Should create with required fields only."""
        diag = DiagnosticsData(
            reprojection_error_rms=0.5,
            reprojection_error_per_camera={"cam0": 0.4, "cam1": 0.6},
            validation_3d_error_mean=0.002,
            validation_3d_error_std=0.001,
        )
        assert diag.reprojection_error_rms == 0.5
        assert diag.per_corner_residuals is None
        assert diag.per_frame_errors is None

    def test_optional_fields(self):
        """Should accept optional residual arrays."""
        residuals = np.random.randn(100, 2)
        frame_errors = {0: 0.3, 1: 0.4, 2: 0.5}

        diag = DiagnosticsData(
            reprojection_error_rms=0.5,
            reprojection_error_per_camera={"cam0": 0.4},
            validation_3d_error_mean=0.002,
            validation_3d_error_std=0.001,
            per_corner_residuals=residuals,
            per_frame_errors=frame_errors,
        )
        assert diag.per_corner_residuals.shape == (100, 2)
        assert diag.per_frame_errors[1] == 0.4


class TestCalibrationMetadata:
    """Test CalibrationMetadata dataclass."""

    def test_creation(self):
        """Should create valid metadata."""
        meta = CalibrationMetadata(
            calibration_date="2026-02-03",
            software_version="0.1.0",
            config_hash="abc123",
            num_frames_used=50,
            num_frames_holdout=10,
        )
        assert meta.calibration_date == "2026-02-03"
        assert meta.software_version == "0.1.0"
        assert meta.num_frames_used == 50
        assert meta.num_frames_holdout == 10


class TestCalibrationConfig:
    """Test CalibrationConfig dataclass."""

    def test_creation_with_defaults(self):
        """Should create config with default values."""
        board = BoardConfig(5, 7, 0.04, 0.03, "DICT_4X4_50")
        config = CalibrationConfig(
            board=board,
            camera_names=["cam0", "cam1"],
            intrinsic_video_paths={"cam0": Path("cam0_int.mp4")},
            extrinsic_video_paths={"cam0": Path("cam0_ext.mp4")},
            output_dir=Path("output"),
        )

        assert config.n_air == 1.0
        assert config.n_water == 1.333
        assert config.interface_normal_fixed is False
        assert config.robust_loss == "huber"
        assert config.min_cameras_per_frame == 2
        assert config.refine_intrinsics is False

    def test_creation_with_custom_values(self):
        """Should accept custom parameter values."""
        board = BoardConfig(5, 7, 0.04, 0.03, "DICT_4X4_50")
        config = CalibrationConfig(
            board=board,
            camera_names=["cam0"],
            intrinsic_video_paths={},
            extrinsic_video_paths={},
            output_dir=Path("."),
            n_water=1.34,
            robust_loss="soft_l1",
            min_corners_per_frame=10,
        )

        assert config.n_water == 1.34
        assert config.robust_loss == "soft_l1"
        assert config.min_corners_per_frame == 10


class TestCalibrationResult:
    """Test CalibrationResult dataclass."""

    def test_creation(self):
        """Should create complete calibration result."""
        board = BoardConfig(5, 7, 0.04, 0.03, "DICT_4X4_50")
        interface = InterfaceParams(normal=np.array([0.0, 0.0, -1.0]))
        diag = DiagnosticsData(
            reprojection_error_rms=0.5,
            reprojection_error_per_camera={},
            validation_3d_error_mean=0.002,
            validation_3d_error_std=0.001,
        )
        meta = CalibrationMetadata("2026-02-03", "0.1.0", "abc", 50, 10)

        result = CalibrationResult(
            cameras={},
            interface=interface,
            board=board,
            diagnostics=diag,
            metadata=meta,
        )

        assert result.board == board
        assert result.interface == interface
        assert result.diagnostics == diag
        assert result.metadata == meta


class TestBoardPose:
    """Test BoardPose dataclass."""

    def test_creation(self):
        """Should create valid board pose."""
        pose = BoardPose(
            frame_idx=5, rvec=np.array([0.1, 0.2, 0.3]), tvec=np.array([1.0, 2.0, 3.0])
        )
        assert pose.frame_idx == 5
        assert pose.rvec.shape == (3,)
        assert pose.tvec.shape == (3,)


class TestDetection:
    """Test Detection dataclass and computed properties."""

    def test_num_corners_property(self):
        """Should compute number of corners correctly."""
        det = Detection(
            corner_ids=np.array([0, 1, 2, 3]),
            corners_2d=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
        )
        assert det.num_corners == 4

    def test_empty_detection(self):
        """Should handle empty detection."""
        det = Detection(
            corner_ids=np.array([], dtype=np.int32),
            corners_2d=np.array([], dtype=np.float64).reshape(0, 2),
        )
        assert det.num_corners == 0


class TestFrameDetections:
    """Test FrameDetections dataclass and computed properties."""

    def test_cameras_with_detections(self):
        """Should return list of camera names."""
        det0 = Detection(
            corner_ids=np.array([0, 1]),
            corners_2d=np.array([[0, 0], [1, 0]], dtype=np.float64),
        )
        det1 = Detection(
            corner_ids=np.array([0, 1]),
            corners_2d=np.array([[0, 0], [1, 0]], dtype=np.float64),
        )

        frame_det = FrameDetections(
            frame_idx=0, detections={"cam0": det0, "cam1": det1}
        )

        cameras = frame_det.cameras_with_detections
        assert len(cameras) == 2
        assert "cam0" in cameras
        assert "cam1" in cameras

    def test_num_cameras(self):
        """Should count cameras with detections."""
        det0 = Detection(
            corner_ids=np.array([0]), corners_2d=np.array([[0, 0]], dtype=np.float64)
        )

        frame_det = FrameDetections(
            frame_idx=0, detections={"cam0": det0, "cam1": det0, "cam2": det0}
        )

        assert frame_det.num_cameras == 3

    def test_empty_frame(self):
        """Should handle frame with no detections."""
        frame_det = FrameDetections(frame_idx=0, detections={})
        assert frame_det.num_cameras == 0
        assert len(frame_det.cameras_with_detections) == 0


class TestDetectionResult:
    """Test DetectionResult dataclass and methods."""

    def test_get_frames_with_min_cameras(self):
        """Should filter frames by minimum camera count."""
        det = Detection(
            corner_ids=np.array([0]), corners_2d=np.array([[0, 0]], dtype=np.float64)
        )

        # Frame 0: 1 camera
        # Frame 1: 2 cameras
        # Frame 2: 3 cameras
        frames = {
            0: FrameDetections(0, {"cam0": det}),
            1: FrameDetections(1, {"cam0": det, "cam1": det}),
            2: FrameDetections(2, {"cam0": det, "cam1": det, "cam2": det}),
        }

        result = DetectionResult(
            frames=frames, camera_names=["cam0", "cam1", "cam2"], total_frames=10
        )

        # Test different thresholds
        assert result.get_frames_with_min_cameras(1) == [0, 1, 2]
        assert result.get_frames_with_min_cameras(2) == [1, 2]
        assert result.get_frames_with_min_cameras(3) == [2]
        assert result.get_frames_with_min_cameras(4) == []

    def test_empty_result(self):
        """Should handle detection result with no frames."""
        result = DetectionResult(frames={}, camera_names=["cam0"], total_frames=0)
        assert result.get_frames_with_min_cameras(1) == []


class TestExceptionHierarchy:
    """Test custom exception hierarchy."""

    def test_calibration_error_base(self):
        """CalibrationError should be base exception."""
        err = CalibrationError("test")
        assert isinstance(err, Exception)
        assert str(err) == "test"

    def test_insufficient_data_error(self):
        """InsufficientDataError should inherit from CalibrationError."""
        err = InsufficientDataError("not enough data")
        assert isinstance(err, CalibrationError)
        assert isinstance(err, Exception)

    def test_convergence_error(self):
        """ConvergenceError should inherit from CalibrationError."""
        err = ConvergenceError("did not converge")
        assert isinstance(err, CalibrationError)
        assert isinstance(err, Exception)

    def test_connectivity_error(self):
        """ConnectivityError should inherit from CalibrationError."""
        err = ConnectivityError("graph not connected")
        assert isinstance(err, CalibrationError)
        assert isinstance(err, Exception)

    def test_catch_specific_error(self):
        """Should be able to catch specific error types."""
        with pytest.raises(InsufficientDataError):
            raise InsufficientDataError("test")

    def test_catch_base_error(self):
        """Should be able to catch all calibration errors via base class."""
        with pytest.raises(CalibrationError):
            raise InsufficientDataError("test")

        with pytest.raises(CalibrationError):
            raise ConvergenceError("test")

        with pytest.raises(CalibrationError):
            raise ConnectivityError("test")


class TestPublicAPI:
    """Verify top-level imports work."""

    def test_top_level_imports(self):
        """All tier-1 exports are importable from aquacal."""
        from aquacal import (
            load_calibration,
            save_calibration,
            CalibrationResult,
            CameraCalibration,
            CameraIntrinsics,
            CameraExtrinsics,
            run_calibration,
            load_config,
        )

        # Verify they're the real objects, not None
        assert callable(load_calibration)
        assert callable(save_calibration)
        assert callable(run_calibration)
        assert callable(load_config)

    def test_version_string(self):
        """__version__ is a non-empty string."""
        import aquacal

        assert isinstance(aquacal.__version__, str)
        assert len(aquacal.__version__) > 0

    def test_subpackage_imports(self):
        """Tier-2 subpackage imports still work."""
        from aquacal.core import Camera, Interface, refractive_project
        from aquacal.calibration import optimize_interface
        from aquacal.triangulation import triangulate_point

        assert callable(refractive_project)
