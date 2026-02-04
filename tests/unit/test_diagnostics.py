"""Unit tests for diagnostics module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aquacal.config.schema import (
    BoardConfig,
    CalibrationResult,
    CameraCalibration,
    CameraIntrinsics,
    CameraExtrinsics,
    InterfaceParams,
    DiagnosticsData,
    CalibrationMetadata,
    DetectionResult,
    FrameDetections,
    Detection,
    BoardPose,
)
from aquacal.core.board import BoardGeometry
from aquacal.validation.reprojection import ReprojectionErrors
from aquacal.validation.reconstruction import DistanceErrors
from aquacal.validation.diagnostics import (
    DiagnosticReport,
    compute_spatial_error_map,
    compute_depth_stratified_errors,
    generate_recommendations,
    generate_diagnostic_report,
    save_diagnostic_report,
)


@pytest.fixture
def board_config():
    """Standard board configuration."""
    return BoardConfig(
        squares_x=5, squares_y=4, square_size=0.03, marker_size=0.022, dictionary="DICT_4X4_50"
    )


@pytest.fixture
def board_geometry(board_config):
    """Board geometry instance."""
    return BoardGeometry(board_config)


@pytest.fixture
def camera_calibration():
    """Single camera calibration."""
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
    intrinsics = CameraIntrinsics(K=K, dist_coeffs=np.zeros(5), image_size=(640, 480))
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.5])
    extrinsics = CameraExtrinsics(R=R, t=t)
    return CameraCalibration(
        name="cam0", intrinsics=intrinsics, extrinsics=extrinsics, interface_distance=0.5
    )


@pytest.fixture
def calibration_result(board_config, camera_calibration):
    """Full calibration result with one camera."""
    cameras = {"cam0": camera_calibration}
    interface = InterfaceParams(normal=np.array([0.0, 0.0, -1.0]))
    diagnostics = DiagnosticsData(
        reprojection_error_rms=0.5,
        reprojection_error_per_camera={"cam0": 0.5},
        validation_3d_error_mean=0.001,
        validation_3d_error_std=0.0005,
    )
    metadata = CalibrationMetadata(
        calibration_date="2026-01-01",
        software_version="0.1.0",
        config_hash="abc123",
        num_frames_used=10,
        num_frames_holdout=2,
    )
    return CalibrationResult(
        cameras=cameras,
        interface=interface,
        board=board_config,
        diagnostics=diagnostics,
        metadata=metadata,
    )


@pytest.fixture
def simple_detections():
    """Simple detection result with known pixel positions."""
    # Frame 0: 4 corners in cam0
    det0 = Detection(
        corner_ids=np.array([0, 1, 2, 3], dtype=np.int32),
        corners_2d=np.array([[100, 100], [200, 100], [100, 200], [200, 200]], dtype=np.float64),
    )
    frame0 = FrameDetections(frame_idx=0, detections={"cam0": det0})

    return DetectionResult(frames={0: frame0}, camera_names=["cam0"], total_frames=1)


@pytest.fixture
def simple_reprojection_errors():
    """Simple reprojection errors matching simple_detections."""
    residuals = np.array(
        [[0.5, 0.0], [1.0, 0.0], [0.0, 0.5], [0.0, 1.0]], dtype=np.float64
    )
    return ReprojectionErrors(
        rms=0.6,
        per_camera={"cam0": 0.6},
        per_frame={0: 0.6},
        residuals=residuals,
        num_observations=4,
    )


@pytest.fixture
def board_poses():
    """Simple board poses for testing."""
    return {
        0: BoardPose(frame_idx=0, rvec=np.zeros(3), tvec=np.array([0.0, 0.0, 1.0])),
    }


class TestComputeSpatialErrorMap:
    """Tests for compute_spatial_error_map()."""

    def test_basic_binning(self, simple_reprojection_errors, simple_detections):
        """Test that errors are binned to correct grid cells."""
        error_map = compute_spatial_error_map(
            reprojection_errors=simple_reprojection_errors,
            detections=simple_detections,
            camera_name="cam0",
            image_size=(640, 480),
            grid_size=(4, 4),
        )

        assert error_map.shape == (4, 4)

        # Corner at (100, 100) with error [0.5, 0.0] -> magnitude 0.5
        # Grid cell: col=0, row=0
        assert np.isclose(error_map[0, 0], 0.5)

        # Corner at (200, 100) with error [1.0, 0.0] -> magnitude 1.0
        # Grid cell: col=1, row=0
        assert np.isclose(error_map[0, 1], 1.0)

        # Corner at (100, 200) with error [0.0, 0.5] -> magnitude 0.5
        # Grid cell: col=0, row=1
        assert np.isclose(error_map[1, 0], 0.5)

        # Corner at (200, 200) with error [0.0, 1.0] -> magnitude 1.0
        # Grid cell: col=1, row=1
        assert np.isclose(error_map[1, 1], 1.0)

    def test_empty_cells_contain_nan(self, simple_reprojection_errors, simple_detections):
        """Test that cells with no observations contain NaN."""
        error_map = compute_spatial_error_map(
            reprojection_errors=simple_reprojection_errors,
            detections=simple_detections,
            camera_name="cam0",
            image_size=(640, 480),
            grid_size=(4, 4),
        )

        # Most cells should be empty (only 4 corners in top-left region)
        num_nan = np.sum(np.isnan(error_map))
        assert num_nan > 8  # Most of the 16 cells should be NaN

    def test_different_camera(self, simple_reprojection_errors, simple_detections):
        """Test that querying a different camera returns all NaN."""
        error_map = compute_spatial_error_map(
            reprojection_errors=simple_reprojection_errors,
            detections=simple_detections,
            camera_name="cam1",  # Not in detections
            image_size=(640, 480),
            grid_size=(4, 4),
        )

        # All cells should be NaN since cam1 has no detections
        assert np.all(np.isnan(error_map))


class TestComputeDepthStratifiedErrors:
    """Tests for compute_depth_stratified_errors()."""

    def test_basic_binning(
        self,
        calibration_result,
        simple_detections,
        board_poses,
        simple_reprojection_errors,
        board_geometry,
    ):
        """Test correct binning and statistics computation."""
        df = compute_depth_stratified_errors(
            calibration=calibration_result,
            detections=simple_detections,
            board_poses=board_poses,
            reprojection_errors=simple_reprojection_errors,
            board=board_geometry,
            num_bins=2,
        )

        assert len(df) == 2
        assert list(df.columns) == [
            "depth_min",
            "depth_max",
            "mean_error",
            "std_error",
            "num_observations",
        ]

        # All observations should be binned
        assert df["num_observations"].sum() == 4

    def test_empty_detections(
        self, calibration_result, board_poses, simple_reprojection_errors, board_geometry
    ):
        """Test that empty detections return empty DataFrame with correct columns."""
        empty_detections = DetectionResult(frames={}, camera_names=["cam0"], total_frames=0)

        df = compute_depth_stratified_errors(
            calibration=calibration_result,
            detections=empty_detections,
            board_poses=board_poses,
            reprojection_errors=simple_reprojection_errors,
            board=board_geometry,
            num_bins=5,
        )

        assert len(df) == 0
        assert list(df.columns) == [
            "depth_min",
            "depth_max",
            "mean_error",
            "std_error",
            "num_observations",
        ]

    def test_depth_calculation(
        self,
        calibration_result,
        simple_detections,
        board_poses,
        simple_reprojection_errors,
        board_geometry,
    ):
        """Test that depth is calculated correctly (Z - interface_z)."""
        df = compute_depth_stratified_errors(
            calibration=calibration_result,
            detections=simple_detections,
            board_poses=board_poses,
            reprojection_errors=simple_reprojection_errors,
            board=board_geometry,
            num_bins=1,
        )

        # Board is at Z=1.0, interface_distance=0.5
        # Depth should be around 1.0 - 0.5 = 0.5
        assert df.iloc[0]["depth_min"] < 0.6
        assert df.iloc[0]["depth_max"] > 0.4


class TestGenerateRecommendations:
    """Tests for generate_recommendations()."""

    def test_good_calibration(self):
        """Test recommendations for good calibration."""
        reproj = ReprojectionErrors(
            rms=0.4,
            per_camera={"cam0": 0.4, "cam1": 0.4},
            per_frame={0: 0.4},
            residuals=np.zeros((10, 2)),
            num_observations=10,
        )
        recon = DistanceErrors(mean=0.0005, std=0.0002, max_error=0.001, num_comparisons=20)
        depth_errors = pd.DataFrame(
            {
                "depth_min": [0.5, 1.0],
                "depth_max": [1.0, 1.5],
                "mean_error": [0.3, 0.35],
                "std_error": [0.1, 0.1],
                "num_observations": [5, 5],
            }
        )

        recs = generate_recommendations(reproj, recon, depth_errors)

        # Should mention excellent reprojection and reconstruction
        assert any("excellent" in r.lower() for r in recs)
        assert len(recs) >= 2

    def test_elevated_camera_error(self):
        """Test recommendation for camera with elevated error."""
        reproj = ReprojectionErrors(
            rms=0.8,
            per_camera={"cam0": 0.5, "cam1": 1.6},  # cam1 is significantly elevated
            per_frame={0: 0.8},
            residuals=np.zeros((10, 2)),
            num_observations=10,
        )
        recon = DistanceErrors(mean=0.001, std=0.0005, max_error=0.002, num_comparisons=20)
        depth_errors = pd.DataFrame(
            {
                "depth_min": [0.5],
                "depth_max": [1.0],
                "mean_error": [0.8],
                "std_error": [0.1],
                "num_observations": [10],
            }
        )

        recs = generate_recommendations(reproj, recon, depth_errors)

        # Should flag cam1
        assert any("cam1" in r and "elevated" in r.lower() for r in recs)

    def test_depth_trend(self):
        """Test recommendation for depth-dependent errors."""
        reproj = ReprojectionErrors(
            rms=0.6,
            per_camera={"cam0": 0.6},
            per_frame={0: 0.6},
            residuals=np.zeros((10, 2)),
            num_observations=10,
        )
        recon = DistanceErrors(mean=0.001, std=0.0005, max_error=0.002, num_comparisons=20)
        depth_errors = pd.DataFrame(
            {
                "depth_min": [0.5, 1.0, 1.5, 2.0],
                "depth_max": [1.0, 1.5, 2.0, 2.5],
                "mean_error": [0.3, 0.35, 0.5, 0.7],  # Increasing with depth
                "std_error": [0.1, 0.1, 0.1, 0.1],
                "num_observations": [5, 5, 5, 5],
            }
        )

        recs = generate_recommendations(reproj, recon, depth_errors)

        # Should mention depth trend
        assert any("depth" in r.lower() and "interface" in r.lower() for r in recs)

    def test_elevated_reprojection_error(self):
        """Test recommendation for elevated overall reprojection error."""
        reproj = ReprojectionErrors(
            rms=1.5,  # Elevated
            per_camera={"cam0": 1.5},
            per_frame={0: 1.5},
            residuals=np.zeros((10, 2)),
            num_observations=10,
        )
        recon = DistanceErrors(mean=0.001, std=0.0005, max_error=0.002, num_comparisons=20)
        depth_errors = pd.DataFrame(
            {
                "depth_min": [0.5],
                "depth_max": [1.0],
                "mean_error": [1.5],
                "std_error": [0.1],
                "num_observations": [10],
            }
        )

        recs = generate_recommendations(reproj, recon, depth_errors)

        # Should mention elevated error
        assert any("elevated" in r.lower() and "review" in r.lower() for r in recs)


class TestGenerateDiagnosticReport:
    """Tests for generate_diagnostic_report()."""

    def test_integration(
        self,
        calibration_result,
        simple_detections,
        board_poses,
        simple_reprojection_errors,
        board_geometry,
    ):
        """Test full report generation with synthetic data."""
        recon_errors = DistanceErrors(
            mean=0.001, std=0.0005, max_error=0.002, num_comparisons=20
        )

        report = generate_diagnostic_report(
            calibration=calibration_result,
            detections=simple_detections,
            board_poses=board_poses,
            reprojection_errors=simple_reprojection_errors,
            reconstruction_errors=recon_errors,
            board=board_geometry,
        )

        # Check all fields are populated
        assert isinstance(report, DiagnosticReport)
        assert report.reprojection == simple_reprojection_errors
        assert report.reconstruction == recon_errors
        assert "cam0" in report.spatial_error_maps
        assert isinstance(report.depth_errors, pd.DataFrame)
        assert len(report.recommendations) > 0
        assert "reprojection_rms" in report.summary
        assert "reconstruction_mean" in report.summary

    def test_spatial_error_maps_all_cameras(
        self,
        board_config,
        simple_detections,
        board_poses,
        simple_reprojection_errors,
        board_geometry,
    ):
        """Test that spatial error maps are generated for all cameras."""
        # Add second camera
        K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
        intrinsics = CameraIntrinsics(K=K, dist_coeffs=np.zeros(5), image_size=(640, 480))
        R = np.eye(3)
        t = np.array([0.1, 0.0, 0.5])
        extrinsics = CameraExtrinsics(R=R, t=t)
        cam1 = CameraCalibration(
            name="cam1", intrinsics=intrinsics, extrinsics=extrinsics, interface_distance=0.5
        )

        # Build calibration with two cameras
        cam0 = CameraCalibration(
            name="cam0",
            intrinsics=intrinsics,
            extrinsics=CameraExtrinsics(R=np.eye(3), t=np.array([0.0, 0.0, 0.5])),
            interface_distance=0.5,
        )

        cameras = {"cam0": cam0, "cam1": cam1}
        interface = InterfaceParams(normal=np.array([0.0, 0.0, -1.0]))
        diagnostics = DiagnosticsData(
            reprojection_error_rms=0.5,
            reprojection_error_per_camera={"cam0": 0.5, "cam1": 0.5},
            validation_3d_error_mean=0.001,
            validation_3d_error_std=0.0005,
        )
        metadata = CalibrationMetadata(
            calibration_date="2026-01-01",
            software_version="0.1.0",
            config_hash="abc123",
            num_frames_used=10,
            num_frames_holdout=2,
        )
        calibration = CalibrationResult(
            cameras=cameras,
            interface=interface,
            board=board_config,
            diagnostics=diagnostics,
            metadata=metadata,
        )

        recon_errors = DistanceErrors(
            mean=0.001, std=0.0005, max_error=0.002, num_comparisons=20
        )

        report = generate_diagnostic_report(
            calibration=calibration,
            detections=simple_detections,
            board_poses=board_poses,
            reprojection_errors=simple_reprojection_errors,
            reconstruction_errors=recon_errors,
            board=board_geometry,
        )

        # Both cameras should have error maps
        assert "cam0" in report.spatial_error_maps
        assert "cam1" in report.spatial_error_maps


class TestSaveDiagnosticReport:
    """Tests for save_diagnostic_report()."""

    def test_creates_all_files(
        self,
        calibration_result,
        simple_detections,
        board_poses,
        simple_reprojection_errors,
        board_geometry,
    ):
        """Test that JSON, CSV, and PNG files are created."""
        recon_errors = DistanceErrors(
            mean=0.001, std=0.0005, max_error=0.002, num_comparisons=20
        )

        report = generate_diagnostic_report(
            calibration=calibration_result,
            detections=simple_detections,
            board_poses=board_poses,
            reprojection_errors=simple_reprojection_errors,
            reconstruction_errors=recon_errors,
            board=board_geometry,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_diagnostic_report(report, Path(tmpdir), save_images=True)

            # Check JSON file
            assert "json" in result
            assert result["json"].exists()
            with open(result["json"]) as f:
                data = json.load(f)
                assert "summary" in data
                assert "recommendations" in data

            # Check CSV file
            assert "csv" in result
            assert result["csv"].exists()
            df = pd.read_csv(result["csv"])
            assert "depth_min" in df.columns

            # Check image files
            assert "images" in result
            assert "cam0" in result["images"]
            assert result["images"]["cam0"].exists()

    def test_no_images_when_disabled(
        self,
        calibration_result,
        simple_detections,
        board_poses,
        simple_reprojection_errors,
        board_geometry,
    ):
        """Test that no PNG files are created when save_images=False."""
        recon_errors = DistanceErrors(
            mean=0.001, std=0.0005, max_error=0.002, num_comparisons=20
        )

        report = generate_diagnostic_report(
            calibration=calibration_result,
            detections=simple_detections,
            board_poses=board_poses,
            reprojection_errors=simple_reprojection_errors,
            reconstruction_errors=recon_errors,
            board=board_geometry,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_diagnostic_report(report, Path(tmpdir), save_images=False)

            # Should have JSON and CSV
            assert "json" in result
            assert "csv" in result
            assert result["json"].exists()
            assert result["csv"].exists()

            # Should NOT have images
            assert "images" not in result

            # Verify no PNG files in directory
            png_files = list(Path(tmpdir).glob("*.png"))
            assert len(png_files) == 0

    def test_creates_output_directory(
        self,
        calibration_result,
        simple_detections,
        board_poses,
        simple_reprojection_errors,
        board_geometry,
    ):
        """Test that output directory is created if it doesn't exist."""
        recon_errors = DistanceErrors(
            mean=0.001, std=0.0005, max_error=0.002, num_comparisons=20
        )

        report = generate_diagnostic_report(
            calibration=calibration_result,
            detections=simple_detections,
            board_poses=board_poses,
            reprojection_errors=simple_reprojection_errors,
            reconstruction_errors=recon_errors,
            board=board_geometry,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_dir = Path(tmpdir) / "nested" / "output" / "dir"
            assert not nested_dir.exists()

            result = save_diagnostic_report(report, nested_dir, save_images=False)

            # Directory should be created
            assert nested_dir.exists()
            assert result["json"].exists()
            assert result["csv"].exists()
